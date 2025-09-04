"""
Agent Chat API - Socket.IO-based chat with SSE proxy to AI agents
"""

import asyncio
import json

# Import logging
import logging
import os
import uuid
from datetime import datetime

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Import Socket.IO instance
from ..socketio_app import get_socketio_instance

sio = get_socketio_instance()

# Create router
router = APIRouter(prefix="/api/agent-chat", tags=["agent-chat"])

# Simple in-memory session storage
sessions: dict[str, dict] = {}


# Request/Response models
class CreateSessionRequest(BaseModel):
    project_id: str | None = None
    agent_type: str = "rag"


class ChatMessage(BaseModel):
    id: str
    content: str
    sender: str
    timestamp: datetime
    agent_type: str | None = None


# REST Endpoints (minimal for frontend compatibility)
@router.post("/sessions")
async def create_session(request: CreateSessionRequest):
    """Create a new chat session."""
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "id": session_id,
        "session_id": session_id,  # Frontend expects this
        "project_id": request.project_id,
        "agent_type": request.agent_type,
        "messages": [],
        "created_at": datetime.now().isoformat(),
    }
    logger.info(f"Created chat session {session_id} with agent_type: {request.agent_type}")
    return {"session_id": session_id}


@router.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get session information."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return sessions[session_id]


@router.post("/sessions/{session_id}/messages")
async def send_message(session_id: str, request: dict):
    """REST endpoint for sending messages (triggers Socket.IO event internally)."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    # Store user message
    user_msg = {
        "id": str(uuid.uuid4()),
        "content": request.get("message", ""),
        "sender": "user",
        "timestamp": datetime.now().isoformat(),
    }
    sessions[session_id]["messages"].append(user_msg)

    # Emit to Socket.IO room
    await sio.emit("message", {"type": "message", "data": user_msg}, room=f"chat_{session_id}")

    # Trigger agent response via Socket.IO
    asyncio.create_task(
        process_agent_response(session_id, request.get("message", ""), request.get("context", {}))
    )

    return {"status": "sent"}


# Socket.IO Event Handlers
@sio.event
async def join_chat(sid, data):
    """Join a chat room."""
    session_id = data.get("session_id")
    if session_id:
        await sio.enter_room(sid, f"chat_{session_id}")
        logger.info(f"Client {sid} joined chat room {session_id}")
        # Send connection confirmation
        await sio.emit(
            "connection_confirmed",
            {"type": "connection_confirmed", "session_id": session_id},
            to=sid,
        )


@sio.event
async def leave_chat(sid, data):
    """Leave a chat room."""
    session_id = data.get("session_id")
    if session_id:
        await sio.leave_room(sid, f"chat_{session_id}")
        logger.info(f"Client {sid} left chat room {session_id}")


@sio.event
async def chat_message(sid, data):
    """Handle chat message via Socket.IO."""
    session_id = data.get("session_id")
    message = data.get("message")
    context = data.get("context", {})

    if not session_id or not message:
        await sio.emit("error", {"type": "error", "error": "Missing session_id or message"}, to=sid)
        return

    # Store user message
    if session_id in sessions:
        user_msg = {
            "id": str(uuid.uuid4()),
            "content": message,
            "sender": "user",
            "timestamp": datetime.now().isoformat(),
        }
        sessions[session_id]["messages"].append(user_msg)

        # Echo user message to room
        await sio.emit("message", {"type": "message", "data": user_msg}, room=f"chat_{session_id}")

    # Process agent response
    await process_agent_response(session_id, message, context)


# Helper function to process agent responses
async def process_agent_response(session_id: str, message: str, context: dict):
    """Stream agent response via SSE and emit to Socket.IO."""
    if session_id not in sessions:
        return

    agent_type = sessions[session_id].get("agent_type", "rag")
    room = f"chat_{session_id}"

    # Emit typing indicator
    await sio.emit("typing", {"type": "typing", "is_typing": True}, room=room)

    try:
        # Call agents service with SSE streaming
        agents_port = os.getenv("ARCHON_AGENTS_PORT")
        if not agents_port:
            raise ValueError(
                "ARCHON_AGENTS_PORT environment variable is required. "
                "Please set it in your .env file or environment."
            )
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
            async with client.stream(
                "POST",
                f"http://archon-agents:{agents_port}/agents/{agent_type}/stream",
                json={"agent_type": agent_type, "prompt": message, "context": context},
            ) as response:
                if response.status_code != 200:
                    await sio.emit(
                        "error",
                        {"type": "error", "error": f"Agent service error: {response.status_code}"},
                        room=room,
                    )
                    return

                # Collect chunks for complete message
                full_content = ""

                # Stream SSE chunks to Socket.IO
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        try:
                            chunk_data = json.loads(line[6:])
                            chunk_content = chunk_data.get("content", "")

                            # Accumulate content
                            full_content += chunk_content

                            # Emit streaming chunk
                            await sio.emit(
                                "stream_chunk",
                                {"type": "stream_chunk", "content": chunk_content},
                                room=room,
                            )

                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse SSE chunk: {line}")

                # Create complete agent message
                agent_msg = {
                    "id": str(uuid.uuid4()),
                    "content": full_content,
                    "sender": "agent",
                    "agent_type": agent_type,
                    "timestamp": datetime.now().isoformat(),
                }

                # Store in session
                sessions[session_id]["messages"].append(agent_msg)

                # Emit complete message
                await sio.emit("message", {"type": "message", "data": agent_msg}, room=room)

                # Emit stream complete
                await sio.emit("stream_complete", {"type": "stream_complete"}, room=room)

    except Exception as e:
        logger.error(f"Error processing agent response: {e}")
        await sio.emit("error", {"type": "error", "error": str(e)}, room=room)
    finally:
        # Stop typing indicator
        await sio.emit("typing", {"type": "typing", "is_typing": False}, room=room)


# MEMORY LEAK FIX: Session cleanup function
def cleanup_chat_sessions(max_age_hours: int = 2) -> int:
    """
    Clean up chat sessions older than max_age_hours.
    Returns the number of sessions cleaned up.
    
    Args:
        max_age_hours: Maximum age of sessions to keep (default: 2 hours)
    
    Returns:
        Number of sessions cleaned up
    """
    from datetime import datetime, timedelta
    
    current_time = datetime.now()
    cutoff_time = current_time - timedelta(hours=max_age_hours)
    sessions_to_remove = []
    
    for session_id, session_data in sessions.items():
        try:
            # Parse the created_at timestamp
            created_at_str = session_data.get("created_at", "")
            if created_at_str:
                # Handle both ISO format and string format
                if isinstance(created_at_str, str):
                    # Try parsing ISO format first
                    try:
                        created_at = datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
                        # Remove timezone info for comparison
                        created_at = created_at.replace(tzinfo=None)
                    except ValueError:
                        # Fallback to assuming it's already a datetime string
                        logger.warning(f"Could not parse created_at for session {session_id}: {created_at_str}")
                        continue
                else:
                    created_at = created_at_str
                
                # Check if session is too old
                if created_at < cutoff_time:
                    sessions_to_remove.append(session_id)
                    logger.debug(f"📂 [SESSION CLEANUP] Marking session {session_id} for cleanup (created: {created_at}, cutoff: {cutoff_time})")
        except Exception as e:
            logger.warning(f"📂 [SESSION CLEANUP] Error checking session {session_id} age: {e}")
            # If we can't determine age, err on the side of caution and don't remove
            continue
    
    # Remove the old sessions
    cleaned_count = 0
    for session_id in sessions_to_remove:
        try:
            del sessions[session_id]
            cleaned_count += 1
            logger.debug(f"📂 [SESSION CLEANUP] Removed old chat session: {session_id}")
        except KeyError:
            # Session already removed somehow
            continue
    
    if cleaned_count > 0:
        logger.info(f"📂 [SESSION CLEANUP] Cleaned up {cleaned_count} old chat sessions (older than {max_age_hours}h)")
    
    return cleaned_count

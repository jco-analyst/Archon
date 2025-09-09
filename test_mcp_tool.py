#!/usr/bin/env python3
"""
Test script for the new add_markdown_document MCP tool
"""
import asyncio
import httpx
import json

async def test_mcp_add_markdown():
    """Test the add_markdown_document MCP tool"""
    
    # Test markdown content
    test_content = """# Test Documentation

This is a test markdown document to verify the new MCP tool functionality.

## Features

- Direct service layer integration
- Synchronous processing
- Proper error handling
- Structured JSON responses

## Code Example

```python
def hello_world():
    print("Hello from Archon MCP!")
    return True
```

## Implementation Details

This document tests the following pipeline:
1. Content processing via DocumentStorageService
2. Text chunking with smart_chunk_text_async
3. Embedding generation via create_embeddings_batch  
4. Storage in archon_crawled_pages table
5. Source management in archon_sources table

The tool bypasses HTTP APIs and uses direct service layer calls for reliable processing.
"""

    # MCP tool call payload with proper JSON-RPC structure
    payload = {
        "jsonrpc": "2.0",
        "id": "test-1",
        "method": "tools/call",
        "params": {
            "name": "add_markdown_document",
            "arguments": {
                "content": test_content,
                "filename": "test_mcp_doc.md",
                "tags": ["test", "mcp", "documentation"],
                "knowledge_type": "documentation"
            }
        }
    }

    try:
        # Initialize MCP session first
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        
        async with httpx.AsyncClient() as client:
            # First, initialize the session
            init_payload = {
                "jsonrpc": "2.0",
                "id": "init-1",
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {}
                    },
                    "clientInfo": {
                        "name": "test-client",
                        "version": "1.0.0"
                    }
                }
            }
            
            init_response = await client.post(
                "http://localhost:8051/mcp",
                json=init_payload,
                headers=headers,
                timeout=30.0
            )
            
            print(f"Init Status: {init_response.status_code}")
            print(f"Init Headers: {dict(init_response.headers)}")
            
            if init_response.status_code != 200:
                print(f"Failed to initialize session: {init_response.text}")
                return
                
            # Get session ID from headers
            session_id = init_response.headers.get('mcp-session-id')
            if session_id:
                headers['mcp-session-id'] = session_id
                print(f"Session ID: {session_id}")
            
            # Send initialized notification to complete handshake
            init_complete = {
                "jsonrpc": "2.0",
                "method": "notifications/initialized"
            }
            
            await client.post(
                "http://localhost:8051/mcp",
                json=init_complete,
                headers=headers,
                timeout=30.0
            )
            
            print("Initialization handshake completed")
            
            # Now call the tool
            response = await client.post(
                "http://localhost:8051/mcp",
                json=payload,
                headers=headers,
                timeout=60.0  # Increased timeout for processing
            )
            
            print(f"Status Code: {response.status_code}")
            print(f"Response Headers: {dict(response.headers)}")
            
            if response.status_code == 200:
                # Handle text/event-stream response
                response_text = response.text
                print("Raw Response:")
                print(response_text)
                
                # Try to parse JSON from the response
                try:
                    # Handle server-sent events format
                    if "data:" in response_text:
                        # Extract JSON from SSE format
                        lines = response_text.strip().split('\n')
                        for line in lines:
                            if line.startswith('data:'):
                                json_data = line[5:].strip()  # Remove "data:" prefix
                                if json_data and json_data != '[DONE]':
                                    result = json.loads(json_data)
                                    print("✅ MCP Tool Call Successful!")
                                    print(json.dumps(result, indent=2))
                                    return
                    else:
                        result = json.loads(response_text)
                        print("✅ MCP Tool Call Successful!")
                        print(json.dumps(result, indent=2))
                except json.JSONDecodeError:
                    print("✅ MCP Tool response received but couldn't parse JSON:")
                    print(f"Response: {response_text}")
            else:
                print("❌ MCP Tool Call Failed!")
                print(f"Response: {response.text}")
                
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_mcp_add_markdown())
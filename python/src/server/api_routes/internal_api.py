"""
Internal API endpoints for inter-service communication.

These endpoints are meant to be called only by other services in the Archon system,
not by external clients. They provide internal functionality like credential sharing.
"""

import logging
import os
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from ..services.credential_service import credential_service

logger = logging.getLogger(__name__)

# Create router with internal prefix
router = APIRouter(prefix="/internal", tags=["internal"])

# Simple IP-based access control for internal endpoints
ALLOWED_INTERNAL_IPS = [
    "127.0.0.1",  # Localhost
    "172.18.0.0/16",  # Docker network range
    "archon-agents",  # Docker service name
    "archon-mcp",  # Docker service name
]


def is_internal_request(request: Request) -> bool:
    """Check if request is from an internal source."""
    client_host = request.client.host if request.client else None

    if not client_host:
        return False

    # Check if it's a Docker network IP (172.16.0.0/12 range)
    if client_host.startswith("172."):
        parts = client_host.split(".")
        if len(parts) == 4:
            second_octet = int(parts[1])
            # Docker uses 172.16.0.0 - 172.31.255.255
            if 16 <= second_octet <= 31:
                logger.info(f"Allowing Docker network request from {client_host}")
                return True

    # Check if it's localhost
    if client_host in ["127.0.0.1", "::1", "localhost"]:
        return True

    return False


@router.get("/health")
async def internal_health():
    """Internal health check endpoint."""
    return {"status": "healthy", "service": "internal-api"}


@router.get("/credentials/agents")
async def get_agent_credentials(request: Request) -> dict[str, Any]:
    """
    Get credentials needed by the agents service.

    This endpoint is only accessible from internal services and provides
    the necessary credentials for AI agents to function.
    """
    # Check if request is from internal source
    if not is_internal_request(request):
        logger.warning(f"Unauthorized access to internal credentials from {request.client.host}")
        raise HTTPException(status_code=403, detail="Access forbidden")

    try:
        # Get credentials needed by agents
                # Get credentials needed by agents
        credentials = {
            # OpenAI credentials
            "OPENAI_API_KEY": await credential_service.get_credential(
                "OPENAI_API_KEY", decrypt=True
            ),
            "OPENAI_MODEL": await credential_service.get_credential(
                "OPENAI_MODEL", default="gpt-4o-mini"
            ),
            # Provider configuration - CRITICAL for OpenAI Free wrapper detection
            "LLM_PROVIDER": await credential_service.get_credential(
                "LLM_PROVIDER", default="openai"
            ),
            # Model configurations
            "DOCUMENT_AGENT_MODEL": await credential_service.get_credential(
                "DOCUMENT_AGENT_MODEL", default="openai:gpt-4o"
            ),
            "RAG_AGENT_MODEL": await credential_service.get_credential(
                "RAG_AGENT_MODEL", default="openai:gpt-4o-mini"
            ),
            "TASK_AGENT_MODEL": await credential_service.get_credential(
                "TASK_AGENT_MODEL", default="openai:gpt-4o"
            ),
            # Rate limiting settings
            "AGENT_RATE_LIMIT_ENABLED": await credential_service.get_credential(
                "AGENT_RATE_LIMIT_ENABLED", default="true"
            ),
            "AGENT_MAX_RETRIES": await credential_service.get_credential(
                "AGENT_MAX_RETRIES", default="3"
            ),
            # MCP endpoint
            "MCP_SERVICE_URL": f"http://archon-mcp:{os.getenv('ARCHON_MCP_PORT')}",
            # Additional settings
            "LOG_LEVEL": await credential_service.get_credential("LOG_LEVEL", default="INFO"),
        }

        # Filter out None values
        credentials = {k: v for k, v in credentials.items() if v is not None}

        logger.info(f"Provided credentials to agents service from {request.client.host}")
        return credentials

    except Exception as e:
        logger.error(f"Error retrieving agent credentials: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve credentials")


@router.get("/credentials/mcp")
async def get_mcp_credentials(request: Request) -> dict[str, Any]:
    """
    Get credentials needed by the MCP service.

    This endpoint provides credentials for the MCP service if needed in the future.
    """
    # Check if request is from internal source
    if not is_internal_request(request):
        logger.warning(f"Unauthorized access to internal credentials from {request.client.host}")
        raise HTTPException(status_code=403, detail="Access forbidden")

    try:
        credentials = {
            # MCP might need some credentials in the future
            "LOG_LEVEL": await credential_service.get_credential("LOG_LEVEL", default="INFO"),
        }

        logger.info(f"Provided credentials to MCP service from {request.client.host}")
        return credentials

    except Exception as e:
        logger.error(f"Error retrieving MCP credentials: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve credentials")


# Memory monitoring endpoints (Phase 4 memory leak prevention)
@router.get("/memory/status")
async def get_memory_status(request: Request) -> dict[str, Any]:
    """
    Get current memory usage status and metrics.
    
    This endpoint provides comprehensive memory monitoring data including
    process memory usage, system metrics, and Archon-specific object counts.
    """
    # Check if request is from internal source
    if not is_internal_request(request):
        logger.warning(f"Unauthorized access to memory status from {request.client.host}")
        raise HTTPException(status_code=403, detail="Access forbidden")

    try:
        from ..services.memory_monitor import get_memory_monitor
        
        monitor = get_memory_monitor()
        report = monitor.get_memory_report()
        
        logger.info(f"Provided memory status to {request.client.host}")
        return report

    except Exception as e:
        logger.error(f"Error retrieving memory status: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve memory status")


@router.get("/memory/metrics")
async def get_current_memory_metrics(request: Request) -> dict[str, Any]:
    """
    Get current memory metrics snapshot.
    
    Provides real-time memory usage data for monitoring and alerting.
    """
    # Check if request is from internal source  
    if not is_internal_request(request):
        logger.warning(f"Unauthorized access to memory metrics from {request.client.host}")
        raise HTTPException(status_code=403, detail="Access forbidden")

    try:
        from ..services.memory_monitor import get_memory_monitor
        
        monitor = get_memory_monitor()
        current_metrics = monitor.get_current_metrics()
        
        # Convert to dict for JSON serialization
        metrics_dict = {
            "timestamp": current_metrics.timestamp.isoformat(),
            "process_memory_mb": current_metrics.process_memory_mb,
            "system_memory_percent": current_metrics.system_memory_percent,
            "virtual_memory_mb": current_metrics.virtual_memory_mb,
            "rss_memory_mb": current_metrics.rss_memory_mb,
            "heap_objects": current_metrics.heap_objects,
            "gc_collections": current_metrics.gc_collections,
            "custom_metrics": current_metrics.custom_metrics
        }
        
        logger.debug(f"Provided memory metrics to {request.client.host}")
        return metrics_dict

    except Exception as e:
        logger.error(f"Error retrieving memory metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve memory metrics")


@router.post("/memory/cleanup")
async def trigger_memory_cleanup(request: Request) -> dict[str, Any]:
    """
    Trigger manual memory cleanup operations.
    
    Forces cleanup of chat sessions, garbage collection, and other
    memory leak prevention operations.
    """
    # Check if request is from internal source
    if not is_internal_request(request):
        logger.warning(f"Unauthorized access to memory cleanup from {request.client.host}")
        raise HTTPException(status_code=403, detail="Access forbidden")

    try:
        from ..services.memory_monitor import get_memory_monitor
        
        monitor = get_memory_monitor()
        
        # Trigger the periodic cleanup manually
        await monitor._run_periodic_cleanup()
        
        # Get updated metrics after cleanup
        updated_metrics = monitor.get_current_metrics()
        
        result = {
            "status": "cleanup_completed",
            "timestamp": updated_metrics.timestamp.isoformat(),
            "memory_after_cleanup_mb": updated_metrics.process_memory_mb,
            "objects_after_cleanup": updated_metrics.heap_objects,
            "custom_metrics": updated_metrics.custom_metrics
        }
        
        logger.info(f"Manual memory cleanup completed for request from {request.client.host}")
        return result

    except Exception as e:
        logger.error(f"Error during memory cleanup: {e}")
        raise HTTPException(status_code=500, detail="Failed to perform memory cleanup")

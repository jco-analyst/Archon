"""
Settings API endpoints for Archon

Handles:
- OpenAI API key management
- Other credentials and configuration
- Settings storage and retrieval
"""

from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# Import logging
from ..config.logfire_config import logfire
from ..services.credential_service import credential_service, initialize_credentials
from ..utils import get_supabase_client

router = APIRouter(prefix="/api", tags=["settings"])


class CredentialRequest(BaseModel):
    key: str
    value: str
    is_encrypted: bool = False
    category: str | None = None
    description: str | None = None


class CredentialUpdateRequest(BaseModel):
    value: str
    is_encrypted: bool | None = None
    category: str | None = None
    description: str | None = None

# OpenAI Free Provider Configuration Models
class OpenAIFreeConfigRequest(BaseModel):
    fallback_provider: str | None = None
    enable_token_tracking: bool = True


class OpenAIFreeUsageResponse(BaseModel):
    provider_name: str
    usage_date: str
    models: dict[str, Any]
    total_tokens_used: int
    total_token_limit: int
    total_remaining: int



class CredentialResponse(BaseModel):
    success: bool
    message: str


# Credential Management Endpoints
@router.get("/credentials")
async def list_credentials(category: str | None = None):
    """List all credentials and their categories."""
    try:
        logfire.info(f"Listing credentials | category={category}")
        credentials = await credential_service.list_all_credentials()

        if category:
            # Filter by category
            credentials = [cred for cred in credentials if cred.category == category]

        result_count = len(credentials)
        logfire.info(
            f"Credentials listed successfully | count={result_count} | category={category}"
        )

        return [
            {
                "key": cred.key,
                "value": cred.value,
                "encrypted_value": cred.encrypted_value,
                "is_encrypted": cred.is_encrypted,
                "category": cred.category,
                "description": cred.description,
            }
            for cred in credentials
        ]
    except Exception as e:
        logfire.error(f"Error listing credentials | category={category} | error={str(e)}")
        raise HTTPException(status_code=500, detail={"error": str(e)})


@router.get("/credentials/categories/{category}")
async def get_credentials_by_category(category: str):
    """Get all credentials for a specific category."""
    try:
        logfire.info(f"Getting credentials by category | category={category}")
        credentials = await credential_service.get_credentials_by_category(category)

        logfire.info(
            f"Credentials retrieved by category | category={category} | count={len(credentials)}"
        )

        return {"credentials": credentials}
    except Exception as e:
        logfire.error(
            f"Error getting credentials by category | category={category} | error={str(e)}"
        )
        raise HTTPException(status_code=500, detail={"error": str(e)})


@router.post("/credentials")
async def create_credential(request: CredentialRequest):
    """Create or update a credential."""
    try:
        logfire.info(
            f"Creating/updating credential | key={request.key} | is_encrypted={request.is_encrypted} | category={request.category}"
        )

        success = await credential_service.set_credential(
            key=request.key,
            value=request.value,
            is_encrypted=request.is_encrypted,
            category=request.category,
            description=request.description,
        )

        if success:
            logfire.info(
                f"Credential saved successfully | key={request.key} | is_encrypted={request.is_encrypted}"
            )

            return {
                "success": True,
                "message": f"Credential {request.key} {'encrypted and ' if request.is_encrypted else ''}saved successfully",
            }
        else:
            logfire.error(f"Failed to save credential | key={request.key}")
            raise HTTPException(status_code=500, detail={"error": "Failed to save credential"})

    except Exception as e:
        logfire.error(f"Error creating credential | key={request.key} | error={str(e)}")
        raise HTTPException(status_code=500, detail={"error": str(e)})


# Define optional settings with their default values
# These are user preferences that should return defaults instead of 404
# This prevents console errors in the frontend when settings haven't been explicitly set
# The frontend can check the 'is_default' flag to know if it's a default or user-set value
OPTIONAL_SETTINGS_WITH_DEFAULTS = {
    "DISCONNECT_SCREEN_ENABLED": "true",  # Show disconnect screen when server is unavailable
    "PROJECTS_ENABLED": "false",  # Enable project management features
    "LOGFIRE_ENABLED": "false",  # Enable Pydantic Logfire integration
}


@router.get("/credentials/{key}")
async def get_credential(key: str, decrypt: bool = True):
    """Get a specific credential by key."""
    try:
        logfire.info(f"Getting credential | key={key} | decrypt={decrypt}")
        value = await credential_service.get_credential(key, decrypt=decrypt)

        if value is None:
            # Check if this is an optional setting with a default value
            if key in OPTIONAL_SETTINGS_WITH_DEFAULTS:
                logfire.info(f"Returning default value for optional setting | key={key}")
                return {
                    "key": key,
                    "value": OPTIONAL_SETTINGS_WITH_DEFAULTS[key],
                    "is_default": True,
                    "category": "features",
                    "description": f"Default value for {key}",
                }

            logfire.warning(f"Credential not found | key={key}")
            raise HTTPException(status_code=404, detail={"error": f"Credential {key} not found"})

        logfire.info(f"Credential retrieved successfully | key={key}")

        # For encrypted credentials, return metadata instead of the actual value for security
        if isinstance(value, dict) and value.get("is_encrypted") and not decrypt:
            return {
                "key": key,
                "is_encrypted": True,
                "category": value.get("category"),
                "description": value.get("description"),
                "has_value": bool(value.get("encrypted_value")),
            }

        return {"key": key, "value": value, "is_encrypted": False}

    except HTTPException:
        raise
    except Exception as e:
        logfire.error(f"Error getting credential | key={key} | error={str(e)}")
        raise HTTPException(status_code=500, detail={"error": str(e)})


@router.put("/credentials/{key}")
async def update_credential(key: str, request: dict[str, Any]):
    """Update an existing credential."""
    try:
        logfire.info(f"Updating credential | key={key}")

        # Handle both CredentialUpdateRequest and full Credential object formats
        if isinstance(request, dict):
            # If the request contains a 'value' field directly, use it
            value = request.get("value", "")
            is_encrypted = request.get("is_encrypted")
            category = request.get("category")
            description = request.get("description")
        else:
            value = request.value
            is_encrypted = request.is_encrypted
            category = request.category
            description = request.description

        # Get existing credential to preserve metadata if not provided
        existing_creds = await credential_service.list_all_credentials()
        existing = next((c for c in existing_creds if c.key == key), None)

        if existing is None:
            # If credential doesn't exist, create it
            is_encrypted = is_encrypted if is_encrypted is not None else False
            logfire.info(f"Creating new credential via PUT | key={key}")
        else:
            # Preserve existing values if not provided
            if is_encrypted is None:
                is_encrypted = existing.is_encrypted
            if category is None:
                category = existing.category
            if description is None:
                description = existing.description
            logfire.info(f"Updating existing credential | key={key} | category={category}")

        success = await credential_service.set_credential(
            key=key,
            value=value,
            is_encrypted=is_encrypted,
            category=category,
            description=description,
        )

        if success:
            logfire.info(
                f"Credential updated successfully | key={key} | is_encrypted={is_encrypted}"
            )

            return {"success": True, "message": f"Credential {key} updated successfully"}
        else:
            logfire.error(f"Failed to update credential | key={key}")
            raise HTTPException(status_code=500, detail={"error": "Failed to update credential"})

    except Exception as e:
        logfire.error(f"Error updating credential | key={key} | error={str(e)}")
        raise HTTPException(status_code=500, detail={"error": str(e)})


@router.delete("/credentials/{key}")
async def delete_credential(key: str):
    """Delete a credential."""
    try:
        logfire.info(f"Deleting credential | key={key}")
        success = await credential_service.delete_credential(key)

        if success:
            logfire.info(f"Credential deleted successfully | key={key}")

            return {"success": True, "message": f"Credential {key} deleted successfully"}
        else:
            logfire.error(f"Failed to delete credential | key={key}")
            raise HTTPException(status_code=500, detail={"error": "Failed to delete credential"})

    except Exception as e:
        logfire.error(f"Error deleting credential | key={key} | error={str(e)}")
        raise HTTPException(status_code=500, detail={"error": str(e)})


@router.post("/credentials/initialize")
async def initialize_credentials_endpoint():
    """Reload credentials from database."""
    try:
        logfire.info("Reloading credentials from database")
        await initialize_credentials()

        logfire.info("Credentials reloaded successfully")

        return {"success": True, "message": "Credentials reloaded from database"}
    except Exception as e:
        logfire.error(f"Error reloading credentials | error={str(e)}")
        raise HTTPException(status_code=500, detail={"error": str(e)})


@router.get("/database/metrics")
async def database_metrics():
    """Get database metrics and statistics."""
    try:
        logfire.info("Getting database metrics")
        supabase_client = get_supabase_client()

        # Get various table counts
        tables_info = {}

        # Get projects count
        projects_response = (
            supabase_client.table("archon_projects").select("id", count="exact").execute()
        )
        tables_info["projects"] = (
            projects_response.count if projects_response.count is not None else 0
        )

        # Get tasks count
        tasks_response = supabase_client.table("archon_tasks").select("id", count="exact").execute()
        tables_info["tasks"] = tasks_response.count if tasks_response.count is not None else 0

        # Get crawled pages count
        pages_response = (
            supabase_client.table("archon_crawled_pages").select("id", count="exact").execute()
        )
        tables_info["crawled_pages"] = (
            pages_response.count if pages_response.count is not None else 0
        )

        # Get settings count
        settings_response = (
            supabase_client.table("archon_settings").select("id", count="exact").execute()
        )
        tables_info["settings"] = (
            settings_response.count if settings_response.count is not None else 0
        )

        total_records
# OpenAI Free Provider Endpoints
@router.post("/openai-free/config")
async def configure_openai_free(request: OpenAIFreeConfigRequest):
    """Configure OpenAI Free provider settings including fallback provider."""
    try:
        logfire.info(f"Configuring OpenAI Free provider | fallback={request.fallback_provider}")
        
        # Store fallback provider setting
        if request.fallback_provider:
            success = await credential_service.set_credential(
                key="OPENAI_FREE_FALLBACK_PROVIDER",
                value=request.fallback_provider,
                is_encrypted=False,
                category="rag_strategy",
                description="Fallback provider when OpenAI Free token limits are exceeded"
            )
            
            if not success:
                raise HTTPException(status_code=500, detail={"error": "Failed to save fallback provider setting"})
        
        # Store token tracking setting
        success = await credential_service.set_credential(
            key="OPENAI_FREE_TOKEN_TRACKING_ENABLED",
            value=str(request.enable_token_tracking).lower(),
            is_encrypted=False,
            category="rag_strategy",
            description="Enable token tracking for OpenAI Free provider"
        )
        
        if not success:
            raise HTTPException(status_code=500, detail={"error": "Failed to save token tracking setting"})
        
        logfire.info(f"OpenAI Free provider configured successfully | fallback={request.fallback_provider}")
        
        return {
            "success": True,
            "message": "OpenAI Free provider configured successfully",
            "fallback_provider": request.fallback_provider,
            "token_tracking_enabled": request.enable_token_tracking
        }
        
    except Exception as e:
        logfire.error(f"Error configuring OpenAI Free provider | error={str(e)}")
        raise HTTPException(status_code=500, detail={"error": str(e)})


@router.get("/openai-free/usage", response_model=OpenAIFreeUsageResponse)
async def get_openai_free_usage():
    """Get current token usage statistics for OpenAI Free provider."""
    try:
        logfire.info("Getting OpenAI Free usage statistics")
        
        # Import here to avoid circular imports
        from ..services.token_tracking_service import token_tracking_service
        
        usage_summary = await token_tracking_service.get_provider_usage_summary("openai_free")
        
        if usage_summary.get("error"):
            logfire.error(f"Error getting usage summary | error={usage_summary['error']}")
            raise HTTPException(status_code=500, detail={"error": usage_summary["error"]})
        
        logfire.info(f"Usage statistics retrieved | total_used={usage_summary.get('total_tokens_used', 0)}")
        
        return OpenAIFreeUsageResponse(
            provider_name=usage_summary["provider_name"],
            usage_date=usage_summary["usage_date"],
            models=usage_summary["models"],
            total_tokens_used=usage_summary["total_tokens_used"],
            total_token_limit=usage_summary["total_token_limit"],
            total_remaining=usage_summary["total_remaining"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logfire.error(f"Error getting OpenAI Free usage | error={str(e)}")
        raise HTTPException(status_code=500, detail={"error": str(e)})


@router.get("/openai-free/config")
async def get_openai_free_config():
    """Get current OpenAI Free provider configuration."""
    try:
        logfire.info("Getting OpenAI Free provider configuration")
        
        # Get fallback provider
        fallback_provider = await credential_service.get_credential("OPENAI_FREE_FALLBACK_PROVIDER")
        
        # Get token tracking setting
        token_tracking_enabled = await credential_service.get_credential("OPENAI_FREE_TOKEN_TRACKING_ENABLED", "true")
        
        config = {
            "fallback_provider": fallback_provider,
            "enable_token_tracking": str(token_tracking_enabled).lower() == "true",
            "available_fallback_providers": ["openai", "google", "ollama", "localcloudcode"]
        }
        
        logfire.info(f"Configuration retrieved | fallback={fallback_provider}")
        
        return config
        
    except Exception as e:
        logfire.error(f"Error getting OpenAI Free configuration | error={str(e)}")
        raise HTTPException(status_code=500, detail={"error": str(e)})


@router.delete("/openai-free/usage/cleanup")
async def cleanup_old_usage_records(days_to_keep: int = 30):
    """Clean up old token usage records."""
    try:
        logfire.info(f"Cleaning up token usage records older than {days_to_keep} days")
        
        # Import here to avoid circular imports  
        from ..services.token_tracking_service import token_tracking_service
        
        cleanup_result = await token_tracking_service.cleanup_old_usage_records(days_to_keep)
        
        if not cleanup_result.get("success"):
            raise HTTPException(status_code=500, detail={"error": cleanup_result.get("error", "Cleanup failed")})
        
        logfire.info(f"Cleanup completed | deleted_count={cleanup_result['deleted_count']}")
        
        return {
            "success": True,
            "message": f"Cleaned up {cleanup_result['deleted_count']} old usage records",
            "deleted_count": cleanup_result["deleted_count"],
            "cutoff_date": cleanup_result["cutoff_date"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logfire.error(f"Error cleaning up usage records | error={str(e)}")
        raise HTTPException(status_code=500, detail={"error": str(e)})
 = sum(tables_info.values())
        logfire.info(
            f"Database metrics retrieved | total_records={total_records} | tables={tables_info}"
        )

        return {
            "status": "healthy",
            "database": "supabase",
            "tables": tables_info,
            "total_records": total_records,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logfire.error(f"Error getting database metrics | error={str(e)}")
        raise HTTPException(status_code=500, detail={"error": str(e)})


@router.get("/settings/health")
async def settings_health():
    """Health check for settings API."""
    logfire.info("Settings health check requested")
    result = {"status": "healthy", "service": "settings"}

    return result

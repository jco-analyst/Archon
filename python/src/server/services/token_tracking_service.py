"""
Token tracking service for managing daily token usage limits for OpenAI Free provider.

This service tracks token consumption per model per day, enforces limits, and provides
usage statistics. Token limits are reset daily at midnight UTC.
"""

import os
from datetime import date, datetime, timezone
from typing import Dict, Any, Optional

from supabase import Client, create_client

from ..config.logfire_config import get_logger

logger = get_logger(__name__)


class TokenTrackingService:
    """Service for tracking and managing daily token usage for providers with limits."""
    
    def __init__(self):
        self._supabase: Client | None = None
        
        # OpenAI Free tier daily limits (tokens per day)
        self.OPENAI_FREE_LIMITS = {
            # Premium models - 250k tokens/day
            "gpt-5": 250000,
            "gpt-5-chat-latest": 250000,
            "gpt-4.1": 250000,
            "gpt-4o": 250000,
            "o1": 250000,
            "o3": 250000,
            
            # Mini models - 2.5M tokens/day
            "gpt-5-mini": 2500000,
            "gpt-5-nano": 2500000,
            "gpt-4.1-mini": 2500000,
            "gpt-4.1-nano": 2500000,
            "gpt-4o-mini": 2500000,
            "o1-mini": 2500000,
            "o3-mini": 2500000,
            "o4-mini": 2500000,
            "codex-mini-latest": 2500000,
        }
    
    def _get_supabase_client(self) -> Client:
        """Get or create Supabase client."""
        if self._supabase is None:
            url = os.getenv("SUPABASE_URL")
            key = os.getenv("SUPABASE_SERVICE_KEY")
            
            if not url or not key:
                raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set")
            
            self._supabase = create_client(url, key)
            logger.info("Token tracking service initialized with Supabase client")
        
        return self._supabase
    
    async def track_token_usage(
        self,
        provider_name: str,
        model_name: str,
        tokens_used: int
    ) -> Dict[str, Any]:
        """
        Track token usage for a specific provider and model.
        
        Args:
            provider_name: Name of the provider (e.g., "openai_free")
            model_name: Name of the model (e.g., "gpt-4o-mini")
            tokens_used: Number of tokens consumed
            
        Returns:
            Dict containing usage info and whether limit was exceeded
        """
        try:
            supabase = self._get_supabase_client()
            today = date.today()
            
            # Get current usage for today
            current_usage = await self.get_daily_usage(provider_name, model_name, today)
            new_total = current_usage["tokens_used"] + tokens_used
            
            # Get token limit for this model
            token_limit = self._get_token_limit(provider_name, model_name)
            
            # Update or insert usage record
            usage_data = {
                "provider_name": provider_name,
                "model_name": model_name,
                "usage_date": today.isoformat(),
                "tokens_used": new_total,
                "token_limit": token_limit,
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
            
            # Upsert the record
            result = (
                supabase.table("archon_token_usage")
                .upsert(usage_data, on_conflict="provider_name,model_name,usage_date")
                .execute()
            )
            
            limit_exceeded = new_total > token_limit
            remaining_tokens = max(0, token_limit - new_total)
            
            logger.info(
                f"Token usage tracked: {provider_name}/{model_name} - "
                f"{tokens_used} tokens used, {new_total}/{token_limit} daily total, "
                f"{'LIMIT EXCEEDED' if limit_exceeded else f'{remaining_tokens} remaining'}"
            )
            
            return {
                "success": True,
                "tokens_used": new_total,
                "token_limit": token_limit,
                "remaining_tokens": remaining_tokens,
                "limit_exceeded": limit_exceeded,
                "usage_date": today.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error tracking token usage: {e}")
            return {
                "success": False,
                "error": str(e),
                "tokens_used": 0,
                "token_limit": 0,
                "remaining_tokens": 0,
                "limit_exceeded": False
            }
    
    async def get_daily_usage(
        self,
        provider_name: str,
        model_name: str,
        usage_date: Optional[date] = None
    ) -> Dict[str, Any]:
        """
        Get current daily usage for a provider and model.
        
        Args:
            provider_name: Name of the provider
            model_name: Name of the model
            usage_date: Date to check (defaults to today)
            
        Returns:
            Dict containing current usage information
        """
        try:
            supabase = self._get_supabase_client()
            check_date = usage_date or date.today()
            
            result = (
                supabase.table("archon_token_usage")
                .select("*")
                .eq("provider_name", provider_name)
                .eq("model_name", model_name)
                .eq("usage_date", check_date.isoformat())
                .execute()
            )
            
            if result.data:
                usage_record = result.data[0]
                return {
                    "tokens_used": usage_record["tokens_used"],
                    "token_limit": usage_record["token_limit"],
                    "remaining_tokens": max(0, usage_record["token_limit"] - usage_record["tokens_used"]),
                    "usage_date": usage_record["usage_date"],
                    "limit_exceeded": usage_record["tokens_used"] > usage_record["token_limit"]
                }
            else:
                # No usage record exists yet for today
                token_limit = self._get_token_limit(provider_name, model_name)
                return {
                    "tokens_used": 0,
                    "token_limit": token_limit,
                    "remaining_tokens": token_limit,
                    "usage_date": check_date.isoformat(),
                    "limit_exceeded": False
                }
                
        except Exception as e:
            logger.error(f"Error getting daily usage: {e}")
            return {
                "tokens_used": 0,
                "token_limit": 0,
                "remaining_tokens": 0,
                "usage_date": check_date.isoformat(),
                "limit_exceeded": False,
                "error": str(e)
            }
    
    async def check_token_limit(
        self,
        provider_name: str,
        model_name: str,
        tokens_needed: int
    ) -> Dict[str, Any]:
        """
        Check if a token request would exceed daily limits.
        
        Args:
            provider_name: Name of the provider
            model_name: Name of the model
            tokens_needed: Number of tokens needed for the request
            
        Returns:
            Dict indicating if request is allowed and usage info
        """
        try:
            current_usage = await self.get_daily_usage(provider_name, model_name)
            
            if current_usage.get("error"):
                # If there's an error getting usage, allow the request but log it
                logger.warning(f"Could not check token limit, allowing request: {current_usage.get('error')}")
                return {"allowed": True, "reason": "usage_check_failed"}
            
            would_exceed = (current_usage["tokens_used"] + tokens_needed) > current_usage["token_limit"]
            
            return {
                "allowed": not would_exceed,
                "current_usage": current_usage["tokens_used"],
                "token_limit": current_usage["token_limit"],
                "remaining_tokens": current_usage["remaining_tokens"],
                "tokens_needed": tokens_needed,
                "would_exceed_limit": would_exceed,
                "reason": "limit_exceeded" if would_exceed else "within_limit"
            }
            
        except Exception as e:
            logger.error(f"Error checking token limit: {e}")
            # On error, allow the request to avoid breaking functionality
            return {"allowed": True, "reason": "limit_check_failed", "error": str(e)}
    
    async def get_provider_usage_summary(
        self,
        provider_name: str,
        usage_date: Optional[date] = None
    ) -> Dict[str, Any]:
        """
        Get usage summary for all models of a provider.
        
        Args:
            provider_name: Name of the provider
            usage_date: Date to check (defaults to today)
            
        Returns:
            Dict containing usage summary for all models
        """
        try:
            supabase = self._get_supabase_client()
            check_date = usage_date or date.today()
            
            result = (
                supabase.table("archon_token_usage")
                .select("*")
                .eq("provider_name", provider_name)
                .eq("usage_date", check_date.isoformat())
                .execute()
            )
            
            models_usage = {}
            total_used = 0
            total_limit = 0
            
            # Process existing usage records
            for record in result.data:
                model_name = record["model_name"]
                tokens_used = record["tokens_used"]
                token_limit = record["token_limit"]
                
                models_usage[model_name] = {
                    "tokens_used": tokens_used,
                    "token_limit": token_limit,
                    "remaining_tokens": max(0, token_limit - tokens_used),
                    "limit_exceeded": tokens_used > token_limit
                }
                
                total_used += tokens_used
                total_limit += token_limit
            
            # Add models with no usage records (if we know their limits)
            if provider_name == "openai_free":
                for model_name, limit in self.OPENAI_FREE_LIMITS.items():
                    if model_name not in models_usage:
                        models_usage[model_name] = {
                            "tokens_used": 0,
                            "token_limit": limit,
                            "remaining_tokens": limit,
                            "limit_exceeded": False
                        }
                        total_limit += limit
            
            return {
                "provider_name": provider_name,
                "usage_date": check_date.isoformat(),
                "models": models_usage,
                "total_tokens_used": total_used,
                "total_token_limit": total_limit,
                "total_remaining": max(0, total_limit - total_used)
            }
            
        except Exception as e:
            logger.error(f"Error getting provider usage summary: {e}")
            return {
                "provider_name": provider_name,
                "usage_date": check_date.isoformat(),
                "models": {},
                "total_tokens_used": 0,
                "total_token_limit": 0,
                "total_remaining": 0,
                "error": str(e)
            }
    
    def _get_token_limit(self, provider_name: str, model_name: str) -> int:
        """Get token limit for a specific provider and model."""
        if provider_name == "openai_free":
            return self.OPENAI_FREE_LIMITS.get(model_name, 0)
        
        # Add other providers with limits here in the future
        return 0
    
    async def cleanup_old_usage_records(self, days_to_keep: int = 30) -> Dict[str, Any]:
        """
        Clean up old token usage records.
        
        Args:
            days_to_keep: Number of days of records to keep
            
        Returns:
            Dict with cleanup results
        """
        try:
            supabase = self._get_supabase_client()
            cutoff_date = date.today().replace(day=date.today().day - days_to_keep)
            
            result = (
                supabase.table("archon_token_usage")
                .delete()
                .lt("usage_date", cutoff_date.isoformat())
                .execute()
            )
            
            deleted_count = len(result.data) if result.data else 0
            logger.info(f"Cleaned up {deleted_count} old token usage records (older than {cutoff_date})")
            
            return {
                "success": True,
                "deleted_count": deleted_count,
                "cutoff_date": cutoff_date.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error cleaning up old usage records: {e}")
            return {"success": False, "error": str(e)}


# Global instance
token_tracking_service = TokenTrackingService()
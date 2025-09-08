"""
OpenAI Free Provider Wrapper with Token Tracking and Fallback.

This module provides a wrapper around the OpenAI client that tracks token usage
and automatically falls back to an alternate provider when limits are exceeded.
"""

import asyncio
from typing import Any, Dict, Optional, AsyncIterator
from contextlib import asynccontextmanager

import openai
from openai.types.chat import ChatCompletion

from ..config.logfire_config import get_logger
from .token_tracking_service import token_tracking_service
from .credential_service import credential_service

logger = get_logger(__name__)


class OpenAIFreeClientWrapper:
    """
    Wrapper for OpenAI client that tracks token usage and handles fallback.
    
    This wrapper maintains the same API interface as openai.AsyncOpenAI but adds:
    - Token usage tracking for daily limits
    - Automatic fallback to alternate provider when limits exceeded
    - Transparent error handling and notifications
    """

    def __init__(self, openai_client: openai.AsyncOpenAI, fallback_provider: Optional[str] = None):
        self.openai_client = openai_client
        self.fallback_provider = fallback_provider
        self._fallback_client = None
        self._is_using_fallback = False
        
    async def _get_fallback_client(self):
        """Get or create fallback client when needed."""
        if self._fallback_client is None and self.fallback_provider:
            try:
                # Import here to avoid circular import
                from .llm_provider_service import get_llm_client
                
                logger.info(f"Initializing fallback client for provider: {self.fallback_provider}")
                # This returns an async context manager, so we need to enter it
                self._fallback_context = get_llm_client(provider=self.fallback_provider)
                self._fallback_client = await self._fallback_context.__aenter__()
                
                logger.info(f"Fallback client initialized successfully: {self.fallback_provider}")
            except Exception as e:
                logger.error(f"Failed to initialize fallback client: {e}")
                self._fallback_client = None
                
        return self._fallback_client
    
    async def _check_token_limits(self, model: str, estimated_tokens: int = 1000) -> Dict[str, Any]:
        """
        Check if the request would exceed token limits.
        
        Args:
            model: Model name being used
            estimated_tokens: Estimated tokens for the request
            
        Returns:
            Dict with check results and recommendations
        """
        try:
            limit_check = await token_tracking_service.check_token_limit(
                "openai_free", model, estimated_tokens
            )
            return limit_check
        except Exception as e:
            logger.error(f"Error checking token limits: {e}")
            # On error, allow the request to avoid breaking functionality
            return {"allowed": True, "reason": "limit_check_failed", "error": str(e)}
    
    async def _track_token_usage(self, model: str, completion: ChatCompletion) -> Dict[str, Any]:
        """
        Track actual token usage from a completed request.
        
        Args:
            model: Model name that was used
            completion: The chat completion response
            
        Returns:
            Dict with tracking results
        """
        try:
            if hasattr(completion, 'usage') and completion.usage:
                total_tokens = completion.usage.total_tokens
                
                tracking_result = await token_tracking_service.track_token_usage(
                    "openai_free", model, total_tokens
                )
                
                if tracking_result.get("limit_exceeded"):
                    logger.warning(
                        f"Token limit exceeded for {model}: {tracking_result['tokens_used']}/{tracking_result['token_limit']}"
                    )
                
                return tracking_result
            else:
                logger.warning("No usage information available in completion response")
                return {"success": False, "error": "no_usage_info"}
                
        except Exception as e:
            logger.error(f"Error tracking token usage: {e}")
            return {"success": False, "error": str(e)}
    
    async def _should_use_fallback(self, model: str, estimated_tokens: int = 1000) -> bool:
        """
        Determine if we should use the fallback provider instead of OpenAI Free.
        
        Args:
            model: Model name being requested
            estimated_tokens: Estimated tokens for the request
            
        Returns:
            True if should use fallback, False otherwise
        """
        if not self.fallback_provider:
            return False
            
        limit_check = await self._check_token_limits(model, estimated_tokens)
        
        if not limit_check.get("allowed", True):
            logger.info(
                f"OpenAI Free token limit would be exceeded for {model}, switching to fallback: {self.fallback_provider}"
            )
            return True
            
        return False

    async def _estimate_tokens(self, messages: list, model: str) -> int:
        """
        Rough estimation of tokens needed for a chat completion request.
        
        This is a simple estimation - in practice you might want to use
        tiktoken or a similar library for more accurate token counting.
        
        Args:
            messages: Chat messages list
            model: Model name
            
        Returns:
            Estimated token count
        """
        try:
            # Rough estimation: ~4 characters per token for most models
            total_chars = 0
            for message in messages:
                if isinstance(message, dict) and "content" in message:
                    content = message["content"]
                    if isinstance(content, str):
                        total_chars += len(content)
                    elif isinstance(content, list):
                        # Handle multimodal content
                        for part in content:
                            if isinstance(part, dict) and part.get("type") == "text":
                                total_chars += len(part.get("text", ""))
            
            # Add overhead for system tokens, formatting, etc.
            estimated_tokens = max(100, (total_chars // 4) + 200)
            
            return min(estimated_tokens, 4000)  # Cap at reasonable maximum
            
        except Exception as e:
            logger.error(f"Error estimating tokens: {e}")
            return 1000  # Default estimate

    async def create_chat_completion(self, **kwargs) -> ChatCompletion:
        """
        Create a chat completion with token tracking and fallback support.
        
        This method maintains the same interface as openai.AsyncOpenAI.chat.completions.create()
        but adds token limit checking and fallback functionality.
        """
        model = kwargs.get("model", "gpt-4o-mini")
        messages = kwargs.get("messages", [])
        
        # Estimate tokens needed for this request
        estimated_tokens = await self._estimate_tokens(messages, model)
        
        # Check if we should use fallback due to token limits
        should_fallback = await self._should_use_fallback(model, estimated_tokens)
        
        if should_fallback:
            self._is_using_fallback = True
            fallback_client = await self._get_fallback_client()
            
            if fallback_client:
                logger.info(f"Using fallback provider {self.fallback_provider} for request")
                try:
                    # Use fallback client
                    completion = await fallback_client.chat.completions.create(**kwargs)
                    logger.info(f"Request completed successfully using fallback provider")
                    return completion
                except Exception as e:
                    logger.error(f"Fallback provider failed, trying OpenAI Free anyway: {e}")
                    # Continue to try OpenAI Free as last resort
            else:
                logger.warning("Fallback provider not available, proceeding with OpenAI Free")
        
        # Use OpenAI Free provider
        self._is_using_fallback = False
        try:
            logger.debug(f"Making request to OpenAI Free with model: {model}")
            completion = await self.openai_client.chat.completions.create(**kwargs)
            
            # Track actual token usage
            tracking_result = await self._track_token_usage(model, completion)
            
            if tracking_result.get("limit_exceeded"):
                logger.warning(
                    f"OpenAI Free daily limit exceeded after this request. "
                    f"Future requests will use fallback provider: {self.fallback_provider}"
                )
            
            logger.debug(f"Request completed successfully using OpenAI Free")
            return completion
            
        except Exception as e:
            logger.error(f"OpenAI Free request failed: {e}")
            
            # If OpenAI Free fails and we haven't tried fallback yet, try it now
            if not should_fallback and self.fallback_provider:
                logger.info("Trying fallback provider due to OpenAI Free failure")
                fallback_client = await self._get_fallback_client()
                if fallback_client:
                    try:
                        completion = await fallback_client.chat.completions.create(**kwargs)
                        logger.info("Request completed using fallback after OpenAI Free failure")
                        return completion
                    except Exception as fallback_error:
                        logger.error(f"Fallback also failed: {fallback_error}")
            
            # Re-raise original exception if no fallback worked
            raise e

    async def get_usage_summary(self) -> Dict[str, Any]:
        """Get current token usage summary for OpenAI Free provider."""
        try:
            return await token_tracking_service.get_provider_usage_summary("openai_free")
        except Exception as e:
            logger.error(f"Error getting usage summary: {e}")
            return {"error": str(e)}

    @property
    def chat(self):
        """Provide access to chat completions with our wrapper."""
        return ChatCompletionsWrapper(self)

    @property
    def is_using_fallback(self) -> bool:
        """Check if the wrapper is currently using fallback provider."""
        return self._is_using_fallback

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup."""
        if hasattr(self, '_fallback_context'):
            try:
                await self._fallback_context.__aexit__(exc_type, exc_val, exc_tb)
            except Exception as e:
                logger.error(f"Error cleaning up fallback context: {e}")


class ChatCompletionsWrapper:
    """Wrapper for chat.completions to provide the expected API structure."""
    
    def __init__(self, client_wrapper: OpenAIFreeClientWrapper):
        self.client_wrapper = client_wrapper
        
    @property
    def completions(self):
        """Provide access to completions create method."""
        return CompletionsWrapper(self.client_wrapper)


class CompletionsWrapper:
    """Wrapper for completions create method."""
    
    def __init__(self, client_wrapper: OpenAIFreeClientWrapper):
        self.client_wrapper = client_wrapper
        
    async def create(self, **kwargs) -> ChatCompletion:
        """Create chat completion using the wrapper."""
        return await self.client_wrapper.create_chat_completion(**kwargs)


@asynccontextmanager
async def get_openai_free_client(fallback_provider: Optional[str] = None) -> AsyncIterator[OpenAIFreeClientWrapper]:
    """
    Create OpenAI Free client with token tracking and fallback.
    
    Args:
        fallback_provider: Provider to use when token limits are exceeded
        
    Yields:
        OpenAIFreeClientWrapper: Configured client wrapper
    """
    # Get OpenAI API key
    api_key = await credential_service.get_credential("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key required for OpenAI Free provider")
    
    # Create base OpenAI client
    openai_client = openai.AsyncOpenAI(api_key=api_key, base_url="https://api.openai.com/v1")
    
    # Create wrapper with fallback support
    wrapper = OpenAIFreeClientWrapper(openai_client, fallback_provider)
    
    try:
        async with wrapper:
            yield wrapper
    finally:
        # Cleanup handled in wrapper's __aexit__
        pass
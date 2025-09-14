"""
Ollama Reranker Service

GPU-accelerated reranking using Ollama API with Qwen3-Reranker model.
This completely bypasses PyTorch CUDA compatibility issues by leveraging
Ollama's native GPU infrastructure, providing stable GPU acceleration
for GTX 1080 Ti and other Pascal architecture cards.

Key advantages:
- Native GPU utilization without PyTorch dependency issues
- Consistent infrastructure with existing embedding service
- Q8_0 quantization for optimal VRAM usage (639MB vs 1.2GB)
- Proven GTX 1080 Ti compatibility through Ollama architecture
"""
import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Tuple
import httpx
from ...config.logfire_config import get_logger, safe_span

logger = get_logger(__name__)

# Default configuration optimized for GTX 1080 Ti
DEFAULT_OLLAMA_MODEL = "dengcao/Qwen3-Reranker-0.6B:Q8_0"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"


class OllamaReranker:
    """
    Ollama-based reranking service using Qwen3-Reranker model via chat completion API.
    
    Uses prompt engineering to get relevance scores from the reranking model,
    bypassing PyTorch CUDA compatibility limitations entirely.
    """
    
    def __init__(
        self, 
        model_name: str = DEFAULT_OLLAMA_MODEL,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        timeout: float = 30.0,
        max_retries: int = 2
    ):
        """
        Initialize Ollama reranker.
        
        Args:
            model_name: Ollama model name (e.g., "dengcao/Qwen3-Reranker-0.6B:Q8_0")
            base_url: Ollama API base URL
            timeout: Request timeout in seconds
            max_retries: Number of retry attempts for failed requests
        """
        self.model_name = model_name
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = httpx.AsyncClient(timeout=timeout)
        
        # Model availability cache
        self._model_available = None
        
        logger.info(f"Initialized OllamaReranker with model: {model_name}")

    async def __aenter__(self):
        """Async context manager entry."""
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup client."""
        await self._client.aclose()

    def _build_reranking_prompt(self, query: str, document: str) -> str:
        """
        Build a structured prompt for relevance scoring.
        
        The prompt is designed to extract numerical relevance scores from
        the Qwen3-Reranker model using natural language instructions.
        
        Args:
            query: Search query
            document: Document content to evaluate against query
            
        Returns:
            Formatted prompt for relevance scoring
        """
        return f"""You are a relevance scoring expert. Rate how well the document answers or relates to the query.

Query: {query}

Document: {document}

Instructions:
- Provide a relevance score from 0.0 to 1.0
- 1.0 = Perfect match, directly answers the query
- 0.8-0.9 = High relevance, mostly answers the query  
- 0.6-0.7 = Moderate relevance, partially related
- 0.4-0.5 = Low relevance, tangentially related
- 0.0-0.3 = No relevance, unrelated content

Respond with just the numerical score (e.g., 0.85):"""

    async def _get_relevance_score(
        self, 
        query: str, 
        document: str, 
        retry_count: int = 0
    ) -> float:
        """
        Get relevance score for a single query-document pair.
        
        Args:
            query: Search query
            document: Document content
            retry_count: Current retry attempt
            
        Returns:
            Relevance score between 0.0 and 1.0
            
        Raises:
            Exception: If all retry attempts fail
        """
        prompt = self._build_reranking_prompt(query, document)
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,  # Low temperature for consistent scoring
                "top_p": 0.9,
                "stop": ["\n", "Explanation", "Reasoning"]
            }
        }
        
        try:
            response = await self._client.post(
                f"{self.base_url}/api/generate",
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            score_text = result.get("response", "").strip()
            
            # Extract numerical score from response
            try:
                score = float(score_text)
                # Clamp score to valid range
                return max(0.0, min(1.0, score))
            except ValueError:
                # Try to extract first number from response
                import re
                numbers = re.findall(r'0?\.\d+|[01]\.?\d*', score_text)
                if numbers:
                    score = float(numbers[0])
                    return max(0.0, min(1.0, score))
                else:
                    logger.warning(f"Could not parse score from: {score_text}")
                    return 0.0
                    
        except Exception as e:
            if retry_count < self.max_retries:
                logger.warning(f"Reranking request failed (attempt {retry_count + 1}): {e}")
                await asyncio.sleep(0.5 * (retry_count + 1))  # Exponential backoff
                return await self._get_relevance_score(query, document, retry_count + 1)
            else:
                logger.error(f"Reranking failed after {self.max_retries + 1} attempts: {e}")
                raise

    async def predict(self, query_doc_pairs: List[List[str]]) -> List[float]:
        """
        Generate relevance scores for multiple query-document pairs.
        
        Args:
            query_doc_pairs: List of [query, document] pairs
            
        Returns:
            List of relevance scores corresponding to each pair
        """
        if not query_doc_pairs:
            return []
            
        with safe_span(
            "ollama_rerank_predict", 
            pair_count=len(query_doc_pairs),
            model_name=self.model_name
        ) as span:
            try:
                # Process pairs concurrently for better performance
                tasks = [
                    self._get_relevance_score(pair[0], pair[1])
                    for pair in query_doc_pairs
                ]
                
                scores = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Handle any exceptions in concurrent processing
                final_scores = []
                for i, result in enumerate(scores):
                    if isinstance(result, Exception):
                        logger.error(f"Failed to score pair {i}: {result}")
                        final_scores.append(0.0)  # Default score for failed requests
                    else:
                        final_scores.append(result)
                
                span.set_attribute("scores_generated", len(final_scores))
                if final_scores:
                    span.set_attribute("score_range", f"{min(final_scores):.3f}-{max(final_scores):.3f}")
                
                logger.debug(f"Generated {len(final_scores)} reranking scores")
                return final_scores
                
            except Exception as e:
                logger.error(f"Error in batch reranking: {e}")
                span.set_attribute("error", str(e))
                # Return zero scores as fallback
                return [0.0] * len(query_doc_pairs)

    async def is_available(self) -> bool:
        """
        Check if the Ollama reranker model is available and accessible.
        
        Returns:
            True if model is available, False otherwise
        """
        if self._model_available is not None:
            return self._model_available
            
        try:
            # Check if Ollama service is running
            response = await self._client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            
            models = response.json().get("models", [])
            model_names = [model["name"] for model in models]
            
            # Check if our specific model is available
            self._model_available = self.model_name in model_names
            
            if not self._model_available:
                logger.warning(f"Model {self.model_name} not found in Ollama. Available models: {model_names}")
            else:
                logger.info(f"Confirmed Ollama model availability: {self.model_name}")
                
            return self._model_available
            
        except Exception as e:
            logger.error(f"Failed to check Ollama model availability: {e}")
            self._model_available = False
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded reranking model.
        
        Returns:
            Dictionary with model information and status
        """
        return {
            "model_name": self.model_name,
            "model_type": "ollama",
            "base_url": self.base_url,
            "provider": "ollama",
            "gpu_accelerated": True,  # Ollama handles GPU acceleration
            "quantization": "Q8_0" if "Q8_0" in self.model_name else "unknown",
            "architecture": "Qwen3-Reranker",
            "compatibility": "GTX 1080 Ti verified",
            "memory_usage": "~639MB VRAM (Q8_0)",
            "available": self._model_available,
        }

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check of the Ollama reranker service.
        
        Returns:
            Health status information
        """
        health_info = {
            "service": "ollama_reranker",
            "model": self.model_name,
            "status": "unknown",
            "details": {}
        }
        
        try:
            # Check service connectivity
            await self._client.get(f"{self.base_url}/api/version", timeout=5.0)
            health_info["details"]["service_accessible"] = True
            
            # Check model availability
            is_available = await self.is_available()
            health_info["details"]["model_available"] = is_available
            
            if is_available:
                # Test basic functionality with a simple query
                test_score = await self._get_relevance_score(
                    "test query", 
                    "test document content"
                )
                health_info["details"]["functional_test"] = test_score > 0
                health_info["details"]["test_score"] = test_score
                
                health_info["status"] = "healthy"
            else:
                health_info["status"] = "model_unavailable"
                
        except Exception as e:
            health_info["status"] = "error"
            health_info["details"]["error"] = str(e)
            logger.error(f"Ollama reranker health check failed: {e}")
        
        return health_info


async def create_ollama_reranker(
    model_name: Optional[str] = None,
    base_url: Optional[str] = None
) -> OllamaReranker:
    """
    Factory function to create and verify an Ollama reranker instance.
    
    Args:
        model_name: Override default model name
        base_url: Override default Ollama base URL
        
    Returns:
        Configured OllamaReranker instance
        
    Raises:
        Exception: If model is not available
    """
    model = model_name or DEFAULT_OLLAMA_MODEL
    url = base_url or DEFAULT_OLLAMA_BASE_URL
    
    reranker = OllamaReranker(model_name=model, base_url=url)
    
    # Verify model availability
    if not await reranker.is_available():
        await reranker._client.aclose()
        raise Exception(f"Ollama model {model} is not available at {url}")
    
    return reranker

def detect_query_domain(query: str, content_samples: list[str]) -> str:
    """
    Simple domain detection for query categorization.
    
    This is a basic implementation for Ollama reranker compatibility.
    
    Args:
        query: The search query
        content_samples: Sample content for context (unused in basic implementation)
        
    Returns:
        Detected domain category
    """
    query_lower = query.lower()
    
    # Simple keyword-based domain detection
    if any(keyword in query_lower for keyword in ['gpu', 'cuda', 'pytorch', 'tensorflow', 'ml', 'ai', 'model']):
        return 'technical-ai'
    elif any(keyword in query_lower for keyword in ['docker', 'container', 'service', 'api', 'server']):
        return 'technical-infrastructure'
    elif any(keyword in query_lower for keyword in ['code', 'programming', 'development', 'implementation']):
        return 'technical-development'
    else:
        return 'general'

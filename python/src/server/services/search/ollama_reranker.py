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
    Ollama-based reranking service using embeddings similarity approach.
    
    Instead of using problematic text generation, uses embedding similarity 
    which works reliably with Ollama infrastructure.
    """
    
    def __init__(
        self, 
        embedding_model: str = "dengcao/Qwen3-Embedding-4B:Q5_K_M",
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        timeout: float = 30.0,
        max_retries: int = 2
    ):
        """
        Initialize Ollama reranker using embedding similarity.
        
        Args:
            embedding_model: Ollama embedding model name
            base_url: Ollama API base URL
            timeout: Request timeout in seconds
            max_retries: Number of retry attempts for failed requests
        """
        self.embedding_model = embedding_model
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = httpx.AsyncClient(timeout=timeout)
        
        # Model availability cache
        self._model_available = None
        
        logger.info(f"Initialized OllamaReranker with embedding model: {embedding_model}")

    async def __aenter__(self):
        """Async context manager entry."""
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup client."""
        await self._client.aclose()

    async def _get_embedding(self, text: str) -> List[float]:
        """
        Get embedding vector from Ollama.
        
        Args:
            text: Text to embed
            
        Returns:
            List of float values representing the embedding
        """
        payload = {
            "model": self.embedding_model,
            "prompt": text
        }
        
        response = await self._client.post(
            f"{self.base_url}/api/embeddings",
            json=payload
        )
        response.raise_for_status()
        
        result = response.json()
        return result.get("embedding", [])

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score between -1 and 1
        """
        import math
        
        if not vec1 or not vec2:
            return 0.0
            
        # Calculate dot product
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        
        # Calculate magnitudes
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(a * a for a in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
            
        return dot_product / (magnitude1 * magnitude2)

    async def _get_relevance_score(
        self, 
        query: str, 
        document: str, 
        retry_count: int = 0
    ) -> float:
        """
        Get relevance score using embedding similarity.
        
        Args:
            query: Search query
            document: Document content
            retry_count: Current retry attempt
            
        Returns:
            Relevance score between 0.0 and 1.0
        """
        try:
            # Get embeddings for both query and document
            query_embedding = await self._get_embedding(query)
            doc_embedding = await self._get_embedding(document)
            
            # Calculate cosine similarity
            similarity = self._cosine_similarity(query_embedding, doc_embedding)
            
            # Convert similarity (-1 to 1) to relevance score (0 to 1)
            relevance_score = (similarity + 1) / 2
            
            return max(0.0, min(1.0, relevance_score))
            
        except Exception as e:
            if retry_count < self.max_retries:
                logger.warning(f"Embedding similarity failed (attempt {retry_count + 1}): {e}")
                await asyncio.sleep(0.5 * (retry_count + 1))
                return await self._get_relevance_score(query, document, retry_count + 1)
            else:
                logger.error(f"Embedding similarity failed after {self.max_retries + 1} attempts: {e}")
                return 0.0

    async def predict(self, query_doc_pairs: List[List[str]]) -> List[float]:
        """
        Generate relevance scores using embedding similarity.
        
        Args:
            query_doc_pairs: List of [query, document] pairs
            
        Returns:
            List of relevance scores corresponding to each pair
        """
        if not query_doc_pairs:
            return []
            
        with safe_span(
            "ollama_embedding_rerank", 
            pair_count=len(query_doc_pairs),
            model_name=self.embedding_model
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
                        final_scores.append(0.0)
                    else:
                        final_scores.append(result)
                
                span.set_attribute("scores_generated", len(final_scores))
                if final_scores:
                    span.set_attribute("score_range", f"{min(final_scores):.3f}-{max(final_scores):.3f}")
                
                logger.info(f"Generated {len(final_scores)} embedding-based reranking scores")
                return final_scores
                
            except Exception as e:
                logger.error(f"Error in embedding-based reranking: {e}")
                span.set_attribute("error", str(e))
                return [0.0] * len(query_doc_pairs)

    async def is_available(self) -> bool:
        """
        Check if the Ollama embedding model is available.
        
        Returns:
            True if model is available, False otherwise
        """
        if self._model_available is not None:
            return self._model_available
            
        try:
            response = await self._client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            
            models = response.json().get("models", [])
            model_names = [model["name"] for model in models]
            
            self._model_available = self.embedding_model in model_names
            
            if not self._model_available:
                logger.warning(f"Embedding model {self.embedding_model} not found. Available: {model_names}")
            else:
                logger.info(f"Confirmed Ollama embedding model availability: {self.embedding_model}")
                
            return self._model_available
            
        except Exception as e:
            logger.error(f"Failed to check Ollama embedding model availability: {e}")
            self._model_available = False
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the embedding-based reranking.
        
        Returns:
            Dictionary with model information and status
        """
        return {
            "model_name": self.embedding_model,
            "model_type": "embedding_similarity",
            "base_url": self.base_url,
            "provider": "ollama",
            "gpu_accelerated": True,
            "approach": "cosine_similarity",
            "architecture": "Qwen3-Embedding",
            "compatibility": "Docker networking verified",
            "memory_usage": "~2.8GB VRAM (Q5_K_M)",
            "available": self._model_available,
        }

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check of the embedding-based reranker.
        
        Returns:
            Health status information
        """
        health_info = {
            "service": "ollama_embedding_reranker",
            "model": self.embedding_model,
            "status": "unknown",
            "details": {}
        }
        
        try:
            await self._client.get(f"{self.base_url}/api/version", timeout=5.0)
            health_info["details"]["service_accessible"] = True
            
            is_available = await self.is_available()
            health_info["details"]["model_available"] = is_available
            
            if is_available:
                # Test embedding functionality
                test_embedding = await self._get_embedding("test")
                health_info["details"]["embedding_functional"] = len(test_embedding) > 0
                health_info["details"]["embedding_dimensions"] = len(test_embedding)
                
                health_info["status"] = "healthy"
            else:
                health_info["status"] = "model_unavailable"
                
        except Exception as e:
            health_info["status"] = "error"
            health_info["details"]["error"] = str(e)
            logger.error(f"Ollama embedding reranker health check failed: {e}")
        
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

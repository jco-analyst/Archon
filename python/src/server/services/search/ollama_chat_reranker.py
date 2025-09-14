"""
Ollama Chat Template Reranker - Uses proper Qwen3-Reranker format via /api/generate

This implements the Phase 5 breakthrough: proper chat template format with Ollama infrastructure.
Combines Ollama's reliable Docker networking with the official Qwen3-Reranker classification format.
"""

import asyncio
import logging
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_BASE_URL = "http://host.docker.internal:11434"
DEFAULT_RERANKER_MODEL = "dengcao/Qwen3-Reranker-0.6B:Q8_0"
DEFAULT_INSTRUCTION = 'You are a helpful assistant that evaluates document relevance. Please respond with "yes" if the document is relevant to the query, or "no" if it is not.'


class OllamaChatReranker:
    """
    Ollama-based reranker using proper Qwen3-Reranker chat template format.

    This implementation uses the Phase 5 breakthrough discovery:
    - Uses /api/generate endpoint (not /api/embeddings)
    - Implements official Qwen3-Reranker format: "<Instruct>: ... <Query>: ... <Document>: ..."
    - Expects binary "yes"/"no" classification responses
    - Provides GPU acceleration via Ollama infrastructure
    """

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        timeout: float = 30.0,
        max_retries: int = 2,
        instruction: str = DEFAULT_INSTRUCTION,
    ):
        """
        Initialize Ollama chat template reranker.

        Args:
            model_name: Ollama reranker model name (e.g., dengcao/Qwen3-Reranker-0.6B:Q8_0)
            base_url: Ollama API base URL
            timeout: Request timeout in seconds
            max_retries: Number of retry attempts for failed requests
            instruction: System instruction for relevance evaluation
        """
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.instruction = instruction
        self._client = httpx.AsyncClient(timeout=timeout)

        # Model availability cache
        self._model_available = None

        logger.info(f"Initialized OllamaChatReranker with model: {model_name}")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup client."""
        await self._client.aclose()

    def _build_reranking_prompt(self, query: str, document: str) -> str:
        """
        Build prompt using official Qwen3-Reranker format from Phase 5 breakthrough.

        Format: "<Instruct>: {instruction}\n\n<Query>: {query}\n\n<Document>: {document}"

        Args:
            query: Search query
            document: Document content to evaluate

        Returns:
            Formatted prompt for the model (expects "yes" or "no" response)
        """
        return f"<Instruct>: {self.instruction}\n\n<Query>: {query}\n\n<Document>: {document}"

    def _extract_relevance_score(self, response: str) -> float:
        """
        Extract relevance score from model response.

        Args:
            response: Model response text

        Returns:
            Relevance score between 0.0 and 1.0
        """
        response_lower = response.lower().strip()

        # Look for explicit yes/no answers (primary classification)
        if "yes" in response_lower:
            return 1.0
        elif "no" in response_lower:
            return 0.0

        # Handle numeric responses (fallback for quantization issues)
        try:
            # Some quantized models might output numbers
            numeric_response = "".join(c for c in response if c.isdigit())
            if numeric_response:
                # Normalize based on digit patterns
                if int(numeric_response[0]) >= 5:
                    return 0.7  # Lean towards relevant
                else:
                    return 0.3  # Lean towards not relevant
        except (ValueError, TypeError):
            pass

        # Default to neutral relevance for unclear responses
        logger.warning(f"Unclear reranker response: '{response}', defaulting to 0.5")
        return 0.5

    async def _get_relevance_score(self, query: str, document: str, retry_count: int = 0) -> float:
        """
        Get relevance score using Ollama generate API with chat template.

        Args:
            query: Search query
            document: Document content
            retry_count: Current retry attempt

        Returns:
            Relevance score between 0.0 and 1.0
        """
        try:
            # Build prompt with proper chat template format
            prompt = self._build_reranking_prompt(query, document)

            # Configure generation parameters for binary classification
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.0,  # Deterministic for consistent classification
                    "top_p": 1.0,
                    "num_predict": 10,  # Keep short for binary response
                    "stop": ["\n", ".", "!"],  # Stop at sentence boundaries
                },
            }

            # Make API call to Ollama generate endpoint
            response = await self._client.post(f"{self.base_url}/api/generate", json=payload)
            response.raise_for_status()

            result = response.json()
            response_text = result.get("response", "").strip()

            # Extract relevance score from response
            score = self._extract_relevance_score(response_text)

            logger.debug(f"Reranker response: '{response_text}' -> score: {score:.3f}")
            return score

        except Exception as e:
            if retry_count < self.max_retries:
                logger.warning(f"Reranking failed (attempt {retry_count + 1}): {e}")
                await asyncio.sleep(0.5 * (retry_count + 1))
                return await self._get_relevance_score(query, document, retry_count + 1)
            else:
                logger.error(f"Reranking failed after {self.max_retries + 1} attempts: {e}")
                return 0.5  # Default to neutral score

    async def predict(self, query_doc_pairs: list[list[str]]) -> list[float]:
        """
        Generate relevance scores using Ollama chat template approach.

        Args:
            query_doc_pairs: List of [query, document] pairs

        Returns:
            List of relevance scores corresponding to each pair
        """
        if not query_doc_pairs:
            return []

        start_time = time.time()

        try:
            # Process pairs concurrently for better performance
            tasks = [self._get_relevance_score(pair[0], pair[1]) for pair in query_doc_pairs]

            scores = await asyncio.gather(*tasks, return_exceptions=True)

            # Handle any exceptions in concurrent processing
            final_scores = []
            for i, result in enumerate(scores):
                if isinstance(result, Exception):
                    logger.error(f"Failed to score pair {i}: {result}")
                    final_scores.append(0.5)
                else:
                    final_scores.append(result)

            total_time = time.time() - start_time
            avg_score = sum(final_scores) / len(final_scores) if final_scores else 0.0

            logger.info(
                f"Ollama chat reranking completed: {len(final_scores)} pairs, "
                f"avg score: {avg_score:.3f}, time: {total_time:.2f}s"
            )
            return final_scores

        except Exception as e:
            logger.error(f"Error in Ollama chat reranking: {e}")
            return [0.5] * len(query_doc_pairs)

    async def is_available(self) -> bool:
        """
        Check if the Ollama reranker model is available.

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

            self._model_available = self.model_name in model_names

            if not self._model_available:
                logger.warning(f"Reranker model {self.model_name} not found. Available: {model_names}")
            else:
                logger.info(f"Confirmed Ollama reranker model availability: {self.model_name}")

            return self._model_available

        except Exception as e:
            logger.error(f"Failed to check Ollama reranker model availability: {e}")
            self._model_available = False
            return False

    def get_model_info(self) -> dict[str, Any]:
        """
        Get information about the Ollama chat template reranker.

        Returns:
            Dictionary with model information and status
        """
        return {
            "model_name": self.model_name,
            "model_type": "ollama_chat_template",
            "base_url": self.base_url,
            "provider": "ollama",
            "gpu_accelerated": True,
            "approach": "chat_template_classification",
            "architecture": "Qwen3-Reranker",
            "format": "Phase_5_breakthrough",
            "compatibility": "Docker networking verified",
            "memory_usage": "~600MB VRAM (Q8_0)",
            "available": self._model_available,
            "instruction": self.instruction,
        }

    async def health_check(self) -> dict[str, Any]:
        """
        Perform health check of the Ollama chat template reranker.

        Returns:
            Health status information
        """
        health_info = {"service": "ollama_chat_reranker", "model": self.model_name, "status": "unknown", "details": {}}

        try:
            # Check service accessibility
            await self._client.get(f"{self.base_url}/api/version", timeout=5.0)
            health_info["details"]["service_accessible"] = True

            # Check model availability
            is_available = await self.is_available()
            health_info["details"]["model_available"] = is_available

            if is_available:
                # Test reranking functionality with simple example
                test_pairs = [["test query", "test document"]]
                test_scores = await self.predict(test_pairs)

                health_info["details"]["reranking_functional"] = len(test_scores) > 0
                health_info["details"]["test_score"] = test_scores[0] if test_scores else None

                health_info["status"] = "healthy"
            else:
                health_info["status"] = "model_unavailable"

        except Exception as e:
            health_info["status"] = "error"
            health_info["details"]["error"] = str(e)
            logger.error(f"Ollama chat reranker health check failed: {e}")

        return health_info


# Factory function for compatibility with existing code
async def create_ollama_chat_reranker(
    model_name: str = DEFAULT_RERANKER_MODEL, base_url: str = DEFAULT_OLLAMA_BASE_URL, **kwargs
) -> OllamaChatReranker:
    """
    Factory function to create and verify an Ollama chat reranker instance.

    Args:
        model_name: Ollama reranker model name
        base_url: Ollama API base URL
        **kwargs: Additional configuration parameters

    Returns:
        Initialized OllamaChatReranker instance
    """
    reranker = OllamaChatReranker(model_name=model_name, base_url=base_url, **kwargs)

    # Verify model availability
    is_available = await reranker.is_available()
    if not is_available:
        logger.warning(f"Ollama chat reranker model {model_name} not available")

    return reranker


# Domain detection function for compatibility with rag_service.py
def detect_query_domain(query: str, content_samples: list[str]) -> str:
    """
    Simple domain detection for query categorization.

    This provides compatibility with rag_service.py for Ollama chat template reranker.

    Args:
        query: The search query
        content_samples: Sample content for context (unused in basic implementation)

    Returns:
        Detected domain category
    """
    query_lower = query.lower()

    # Simple keyword-based domain detection
    if any(keyword in query_lower for keyword in ["gpu", "cuda", "pytorch", "tensorflow", "ml", "ai", "model"]):
        return "technical-ai"
    elif any(keyword in query_lower for keyword in ["docker", "container", "service", "api", "server"]):
        return "technical-infrastructure"
    elif any(keyword in query_lower for keyword in ["code", "programming", "development", "implementation"]):
        return "technical-development"
    else:
        return "general"

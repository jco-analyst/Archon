"""
Reranking Strategy

Implements result reranking using multiple providers:
- HuggingFace: Qwen3 reranker models (PyTorch-based)
- Ollama: GPU-accelerated reranking via Ollama API (bypasses PyTorch CUDA issues)

The Ollama provider is recommended for GTX 1080 Ti and Pascal architecture GPUs
as it completely bypasses PyTorch CUDA compatibility limitations while providing
stable GPU acceleration.
"""
import os
from typing import Any, Optional, Union

try:
    from .qwen3_reranker import Qwen3Reranker
    QWEN3_AVAILABLE = True
except ImportError:
    Qwen3Reranker = None
    QWEN3_AVAILABLE = False

try:
    from .qwen3_gguf_reranker import Qwen3GGUFReranker
    GGUF_AVAILABLE = True
except ImportError:
    Qwen3GGUFReranker = None
    GGUF_AVAILABLE = False

try:
    from .ollama_reranker import OllamaReranker
    OLLAMA_AVAILABLE = True
except ImportError:
    OllamaReranker = None
    OLLAMA_AVAILABLE = False

from ...config.logfire_config import get_logger, safe_span

logger = get_logger(__name__)

# Default reranking models by provider
DEFAULT_RERANKING_MODEL = "Qwen/Qwen3-Reranker-0.6B"
DEFAULT_OLLAMA_MODEL = "dengcao/Qwen3-Reranker-0.6B:Q8_0"


class RerankingStrategy:
    """Strategy class implementing result reranking using multiple providers"""

    def __init__(self, model_name: str = DEFAULT_RERANKING_MODEL, credential_service=None, provider: str = "huggingface"):
        """
        Initialize reranking strategy.

        Args:
            model_name: Name/path of the reranker model to use
            credential_service: Credential service for configuration (required)
            provider: Reranking provider ("huggingface", "ollama")
        """
        self.model_name = model_name
        self.credential_service = credential_service
        self.provider = provider
        self.model = self._load_model()

    @classmethod
    async def from_credential_service(cls, credential_service) -> "RerankingStrategy":
        """
        Create a RerankingStrategy from credential service configuration.

        Args:
            credential_service: Service for retrieving reranking configuration

        Returns:
            RerankingStrategy instance configured from credentials
        """
        if not credential_service:
            logger.error("Credential service required for reranking strategy")
            return None
            
        try:
            # Get reranking configuration from credential service
            use_reranking = credential_service.get_bool_setting("USE_RERANKING", False)
            if not use_reranking:
                logger.info("Reranking disabled via USE_RERANKING setting")
                return None
                
            provider = credential_service.get_setting_sync("RERANKING_PROVIDER", "ollama")
            
            # Set model name based on provider
            if provider == "ollama":
                model_name = credential_service.get_setting_sync("RERANKING_MODEL", DEFAULT_OLLAMA_MODEL)
            else:
                model_name = credential_service.get_setting_sync("RERANKING_MODEL", DEFAULT_RERANKING_MODEL)
            
            logger.info(f"Creating reranking strategy with provider: {provider}, model: {model_name}")
            
            instance = cls(model_name=model_name, credential_service=credential_service, provider=provider)
            
            # For Ollama provider, we need async initialization
            if provider == "ollama" and instance.model:
                # Verify model availability asynchronously
                if hasattr(instance.model, 'is_available'):
                    is_available = await instance.model.is_available()
                    if not is_available:
                        logger.error(f"Ollama model {model_name} is not available")
                        return None
            
            return instance
            
        except Exception as e:
            logger.error(f"Failed to create reranking strategy from credentials: {e}")
            return None

    def _load_model(self) -> Optional[Union[Qwen3Reranker, Qwen3GGUFReranker, OllamaReranker]]:
        """Load the reranker model based on provider configuration."""
        if not self.credential_service:
            logger.error("Credential service required for model loading")
            return None
        
        try:
            if self.provider == "ollama":
                return self._load_ollama_model()
            elif self.provider == "huggingface":
                return self._load_huggingface_model()
            else:
                logger.error(f"Unsupported reranking provider: {self.provider}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to load {self.provider} reranking model: {e}")
            return None

    def _load_ollama_model(self) -> Optional[OllamaReranker]:
        """Load Ollama reranker model."""
        if not OLLAMA_AVAILABLE:
            logger.error("Ollama reranker not available - missing ollama_reranker module")
            return None
        
        logger.info(f"Loading Ollama reranking model: {self.model_name}")
        
        # Get Ollama configuration
        base_url = self.credential_service.get_setting_sync("OLLAMA_BASE_URL", "http://localhost:11434")
        
        return OllamaReranker(
            model_name=self.model_name,
            base_url=base_url,
            timeout=30.0,
            max_retries=2
        )

    def _load_huggingface_model(self) -> Optional[Union[Qwen3Reranker, Qwen3GGUFReranker]]:
        """Load HuggingFace reranker model (original implementation)."""
        # Check if GGUF model should be used
        use_gguf = self.credential_service.get_bool_setting("USE_GGUF_RERANKING", True)
        
        if use_gguf and GGUF_AVAILABLE:
            logger.info("Loading GGUF quantized reranker model")
            return Qwen3GGUFReranker(
                n_threads=4,  # Optimize for CPU performance
                temperature=0.0,  # Deterministic for reranking
                verbose=False,
            )
        
        # Fall back to standard transformers model
        if not QWEN3_AVAILABLE:
            logger.error("Neither GGUF nor transformers reranker available")
            return None
        
        logger.info(f"Loading standard reranking model: {self.model_name}")
        
        # Force settings optimized for GTX 1080 Ti
        return Qwen3Reranker(
            model_name=self.model_name,
            max_length=8192,
            use_flash_attention=False,  # Not supported on Pascal
            torch_dtype="float32",     # Optimal for GTX 1080 Ti
            device="cuda" if self._is_cuda_available() else "cpu",
        )
    
    def _is_cuda_available(self) -> bool:
        """Check if CUDA is available for GPU acceleration."""
        # Skip torch import for Ollama provider (uses its own GPU handling)
        if self.provider == "ollama":
            return True  # Ollama handles GPU detection internally
            
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False


    def is_available(self) -> bool:
        """Check if reranking is available (model loaded successfully)."""
        return self.model is not None

    def build_query_document_pairs(
        self, query: str, results: list[dict[str, Any]], content_key: str = "content"
    ) -> tuple[list[list[str]], list[int]]:
        """
        Build query-document pairs for the reranking model.

        Args:
            query: The search query
            results: List of search results
            content_key: The key in each result dict containing text content for reranking

        Returns:
            Tuple of (query-document pairs, valid indices)
        """
        texts = []
        valid_indices = []

        for i, result in enumerate(results):
            content = result.get(content_key, "")
            if content and isinstance(content, str):
                texts.append(content)
                valid_indices.append(i)
            else:
                logger.warning(f"Result {i} has no valid content for reranking")

        query_doc_pairs = [[query, text] for text in texts]
        return query_doc_pairs, valid_indices

    def apply_rerank_scores(
        self,
        results: list[dict[str, Any]],
        scores: list[float],
        valid_indices: list[int],
        top_k: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """
        Apply reranking scores to results and sort them.

        Args:
            results: Original search results
            scores: Reranking scores from the model
            valid_indices: Indices of results that were scored
            top_k: Optional limit on number of results to return

        Returns:
            Reranked and sorted list of results
        """
        # Add rerank scores to valid results
        for i, valid_idx in enumerate(valid_indices):
            results[valid_idx]["rerank_score"] = float(scores[i])

        # Sort results by rerank score (descending - highest relevance first)
        reranked_results = sorted(results, key=lambda x: x.get("rerank_score", -1.0), reverse=True)

        # Apply top_k limit if specified
        if top_k is not None and top_k > 0:
            reranked_results = reranked_results[:top_k]

        return reranked_results

    async def rerank_results(
        self,
        query: str,
        results: list[dict[str, Any]],
        content_key: str = "content",
        top_k: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """
        Rerank search results using the configured reranker model.

        Args:
            query: The search query used to retrieve results
            results: List of search results to rerank
            content_key: The key in each result dict containing text content for reranking
            top_k: Optional limit on number of results to return after reranking

        Returns:
            Reranked list of results ordered by rerank_score (highest first)
        """
        if not self.model or not results:
            logger.debug("Reranking skipped - no model or no results")
            return results

        with safe_span(
            "rerank_results", 
            result_count=len(results), 
            model_name=self.model_name,
            provider=self.provider
        ) as span:
            try:
                # Build query-document pairs
                query_doc_pairs, valid_indices = self.build_query_document_pairs(
                    query, results, content_key
                )

                if not query_doc_pairs:
                    logger.warning("No valid texts found for reranking")
                    return results

                # Get reranking scores from the model
                with safe_span("model_predict"):
                    scores = await self.model.predict(query_doc_pairs)

                # Apply scores and sort results
                reranked_results = self.apply_rerank_scores(results, scores, valid_indices, top_k)

                span.set_attribute("reranked_count", len(reranked_results))
                if len(scores) > 0:
                    span.set_attribute("score_range", f"{min(scores):.3f}-{max(scores):.3f}")
                    logger.debug(
                        f"Reranked {len(query_doc_pairs)} results using {self.provider}, score range: {min(scores):.3f}-{max(scores):.3f}"
                    )

                return reranked_results

            except Exception as e:
                logger.error(f"Error during {self.provider} reranking: {e}")
                span.set_attribute("error", str(e))
                return results

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the loaded reranking model."""
        base_info = {
            "model_name": self.model_name,
            "provider": self.provider,
            "available": self.is_available(),
            "qwen3_available": QWEN3_AVAILABLE,
            "gguf_available": GGUF_AVAILABLE,
            "ollama_available": OLLAMA_AVAILABLE,
            "model_loaded": self.model is not None,
        }
        
        # Add provider-specific info
        if self.model and hasattr(self.model, 'get_model_info'):
            base_info.update({"model_info": self.model.get_model_info()})
        else:
            base_info["model_type"] = "none"
            
        return base_info


async def create_reranking_strategy(credential_service) -> Optional[RerankingStrategy]:
    """
    Factory function to create a reranking strategy from credential service.
    
    Args:
        credential_service: Service for retrieving configuration
        
    Returns:
        RerankingStrategy instance or None if disabled/failed
    """
    return await RerankingStrategy.from_credential_service(credential_service)
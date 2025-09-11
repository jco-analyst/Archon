"""
Reranking Strategy

Implements result reranking using Qwen3 reranker models to improve search result ordering.
The reranking process re-scores search results based on query-document relevance using
a trained neural model, typically improving precision over initial retrieval scores.

Uses Qwen/Qwen3-Reranker-0.6B model by default for optimal GTX 1080 Ti compatibility.
"""

import os
from typing import Any, Optional, Union

try:
    from .qwen3_reranker import Qwen3Reranker
    QWEN3_AVAILABLE = True
except ImportError:
    Qwen3Reranker = None
    QWEN3_AVAILABLE = False

from ...config.logfire_config import get_logger, safe_span

logger = get_logger(__name__)

# Default reranking models optimized for GTX 1080 Ti
DEFAULT_RERANKING_MODEL = "Qwen/Qwen3-Reranker-0.6B"


class RerankingStrategy:
    """Strategy class implementing result reranking using Qwen3 reranker models"""

    def __init__(self, model_name: str = DEFAULT_RERANKING_MODEL, credential_service=None):
        """
        Initialize reranking strategy.

        Args:
            model_name: Name/path of the Qwen3 reranker model to use
            credential_service: Credential service for configuration (required)
        """
        self.model_name = model_name
        self.credential_service = credential_service
        self.model = self._load_model()

    @classmethod
    def from_credential_service(cls, credential_service) -> "RerankingStrategy":
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
                
            model_name = credential_service.get_setting("RERANKING_MODEL", DEFAULT_RERANKING_MODEL)
            provider = credential_service.get_setting("RERANKING_PROVIDER", "huggingface")
            
            if provider != "huggingface":
                logger.warning(f"Only HuggingFace provider supported, got: {provider}")
                return None
            
            logger.info(f"Creating reranking strategy with model: {model_name}")
            return cls(model_name=model_name, credential_service=credential_service)
            
        except Exception as e:
            logger.error(f"Failed to create reranking strategy from credentials: {e}")
            return None

    def _load_model(self) -> Optional[Qwen3Reranker]:
        """Load the Qwen3 reranker model."""
        if not QWEN3_AVAILABLE:
            logger.error("Qwen3Reranker not available - install transformers library")
            return None
        
        if not self.credential_service:
            logger.error("Credential service required for model loading")
            return None
        
        try:
            logger.info(f"Loading Qwen3 reranking model: {self.model_name}")
            
            # Force settings optimized for GTX 1080 Ti
            # Use float32 for Pascal architecture compatibility
            # Use 0.6B model as default for memory constraints
            return Qwen3Reranker(
                model_name=self.model_name,
                max_length=8192,
                use_flash_attention=False,  # Not supported on Pascal
                torch_dtype="float32",     # Optimal for GTX 1080 Ti
                device="cuda" if self._is_cuda_available() else "cpu",
            )
        except Exception as e:
            logger.error(f"Failed to load Qwen3 reranking model {self.model_name}: {e}")
            return None
    
    def _is_cuda_available(self) -> bool:
        """Check if CUDA is available for GPU acceleration."""
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
        Rerank search results using the Qwen3 reranker model.

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
            "rerank_results", result_count=len(results), model_name=self.model_name
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
                        f"Reranked {len(query_doc_pairs)} results, score range: {min(scores):.3f}-{max(scores):.3f}"
                    )

                return reranked_results

            except Exception as e:
                logger.error(f"Error during reranking: {e}")
                span.set_attribute("error", str(e))
                return results

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the loaded reranking model."""
        base_info = {
            "model_name": self.model_name,
            "available": self.is_available(),
            "qwen3_available": QWEN3_AVAILABLE,
            "model_loaded": self.model is not None,
            "model_type": "qwen3" if isinstance(self.model, Qwen3Reranker) else "none",
        }
        
        # Add Qwen3-specific information
        if isinstance(self.model, Qwen3Reranker):
            base_info.update({
                "model_info": self.model.get_model_info(),
            })
            
        return base_info


def create_reranking_strategy(credential_service) -> Optional[RerankingStrategy]:
    """
    Factory function to create a reranking strategy from credential service.
    
    Args:
        credential_service: Service for retrieving configuration
        
    Returns:
        RerankingStrategy instance or None if disabled/failed
    """
    return RerankingStrategy.from_credential_service(credential_service)
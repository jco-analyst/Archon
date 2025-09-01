"""
Reranking Strategy

Implements result reranking using CrossEncoder models to improve search result ordering.
The reranking process re-scores search results based on query-document relevance using
a trained neural model, typically improving precision over initial retrieval scores.

Uses the cross-encoder/ms-marco-MiniLM-L-6-v2 model for reranking by default.
"""

import os
from typing import Any, Optional, Union

try:
    from sentence_transformers import CrossEncoder
    CROSSENCODER_AVAILABLE = True
except ImportError:
    CrossEncoder = None
    CROSSENCODER_AVAILABLE = False

try:
    from .qwen3_reranker import Qwen3Reranker, create_qwen3_reranker
    QWEN3_AVAILABLE = True
except ImportError:
    Qwen3Reranker = None
    create_qwen3_reranker = None
    QWEN3_AVAILABLE = False

from ...config.logfire_config import get_logger, safe_span

logger = get_logger(__name__)

# Default reranking models
DEFAULT_RERANKING_MODEL = "Qwen/Qwen3-Reranker-4B"
QWEN3_RERANKING_MODEL = "Qwen/Qwen3-Reranker-4B"


class RerankingStrategy:
    """Strategy class implementing result reranking using CrossEncoder models"""

    def __init__(
        self, model_name: str = DEFAULT_RERANKING_MODEL, model_instance: Optional[Any] = None
    ):
        """
        Initialize reranking strategy.

        Args:
            model_name: Name/path of the CrossEncoder model to use
            model_instance: Pre-loaded CrossEncoder instance or any object with a predict method (optional)
        """
        self.model_name = model_name
        self.model = model_instance or self._load_model()

    @classmethod
    def from_model(cls, model: Any, model_name: str = "custom_model") -> "RerankingStrategy":
        """
        Create a RerankingStrategy from any model with a predict method.

        This factory method is useful for tests or when using non-CrossEncoder models.

        Args:
            model: Any object with a predict(pairs) method
            model_name: Optional name for the model

        Returns:
            RerankingStrategy instance using the provided model
        """
        return cls(model_name=model_name, model_instance=model)

    def _load_model(self) -> Union[CrossEncoder, Qwen3Reranker, None]:
        """Load the appropriate reranker model (CrossEncoder or Qwen3Reranker)."""
        
        # Check if this is a Qwen3 model
        if self.model_name.startswith("Qwen/") or "qwen" in self.model_name.lower():
            return self._load_qwen3_model()
        else:
            return self._load_crossencoder_model()
    
    def _load_qwen3_model(self) -> Optional[Qwen3Reranker]:
        """Load the Qwen3 reranker model."""
        if not QWEN3_AVAILABLE:
            logger.warning("Qwen3Reranker not available - falling back to CrossEncoder")
            return self._load_crossencoder_model()
        
        try:
            logger.info(f"Loading Qwen3 reranking model: {self.model_name}")
            
            # Get configuration options from environment or defaults
            use_flash_attention = os.getenv("QWEN3_USE_FLASH_ATTENTION", "false").lower() == "true"
            max_length = int(os.getenv("QWEN3_MAX_LENGTH", "8192"))
            torch_dtype = os.getenv("QWEN3_TORCH_DTYPE", None)
            device = os.getenv("QWEN3_DEVICE", None)
            
            # Parse torch dtype if provided
            dtype = None
            if torch_dtype:
                import torch
                dtype = getattr(torch, torch_dtype, None)
            
            return Qwen3Reranker(
                model_name=self.model_name,
                max_length=max_length,
                use_flash_attention=use_flash_attention,
                torch_dtype=dtype,
                device=device,
            )
        except Exception as e:
            logger.error(f"Failed to load Qwen3 reranking model {self.model_name}: {e}")
            logger.info("Falling back to CrossEncoder model")
            return self._load_crossencoder_model()
    
    def _load_crossencoder_model(self) -> Optional[CrossEncoder]:
        """Load the CrossEncoder model for reranking."""
        if not CROSSENCODER_AVAILABLE:
            logger.warning("sentence-transformers not available - reranking disabled")
            return None

        try:
            logger.info(f"Loading CrossEncoder reranking model: {self.model_name}")
            return CrossEncoder(self.model_name)
        except Exception as e:
            logger.error(f"Failed to load CrossEncoder reranking model {self.model_name}: {e}")
            return None


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
            content_key: The key in each result dict containing text content

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
        domain: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Rerank search results using the model (CrossEncoder or Qwen3Reranker).

        Args:
            query: The search query used to retrieve results
            results: List of search results to rerank
            content_key: The key in each result dict containing text content for reranking
            top_k: Optional limit on number of results to return after reranking
            domain: Optional domain context for dynamic instruction generation (Qwen3 only)

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
                    # Check if this is a Qwen3 model that supports async predict with domain
                    if hasattr(self.model, 'predict') and hasattr(self.model, 'enable_dynamic_instructions'):
                        # Qwen3Reranker with dynamic instruction support
                        scores = await self.model.predict(query_doc_pairs, domain=domain)
                    else:
                        # Standard CrossEncoder or non-async model
                        if hasattr(self.model, 'predict'):
                            scores = self.model.predict(query_doc_pairs)
                        else:
                            logger.error(f"Model {type(self.model)} does not have predict method")
                            return results

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
            "crossencoder_available": CROSSENCODER_AVAILABLE,
            "qwen3_available": QWEN3_AVAILABLE,
            "model_loaded": self.model is not None,
        }
        
        # Add model-specific information
        if isinstance(self.model, Qwen3Reranker):
            base_info.update({
                "model_type": "qwen3",
                "model_info": self.model.get_model_info(),
            })
        elif hasattr(self.model, 'max_length'):  # CrossEncoder
            base_info.update({
                "model_type": "crossencoder",
                "max_length": getattr(self.model, 'max_length', None),
            })
        else:
            base_info["model_type"] = "unknown"
            
        return base_info



class RerankingConfig:
    """Configuration helper for reranking settings with Qwen3 support"""

    @staticmethod
    def from_credential_service(credential_service) -> dict[str, Any]:
        """Load reranking configuration from credential service."""
        try:
            use_reranking = credential_service.get_bool_setting("USE_RERANKING", False)
            model_name = credential_service.get_setting("RERANKING_MODEL", DEFAULT_RERANKING_MODEL)
            top_k = int(credential_service.get_setting("RERANKING_TOP_K", "0"))
            
            # Qwen3-specific settings
            qwen3_config = {}
            if model_name.startswith("Qwen/") or "qwen" in model_name.lower():
                qwen3_config = {
                    "use_flash_attention": credential_service.get_bool_setting("QWEN3_USE_FLASH_ATTENTION", False),
                    "max_length": int(credential_service.get_setting("QWEN3_MAX_LENGTH", "8192")),
                    "torch_dtype": credential_service.get_setting("QWEN3_TORCH_DTYPE", None),
                    "device": credential_service.get_setting("QWEN3_DEVICE", None),
                    "instruction": credential_service.get_setting("QWEN3_INSTRUCTION", None),
                }

            config = {
                "enabled": use_reranking,
                "model_name": model_name,
                "top_k": top_k if top_k > 0 else None,
            }
            
            if qwen3_config:
                config["qwen3"] = qwen3_config
                
            return config
            
        except Exception as e:
            logger.error(f"Error loading reranking config: {e}")
            return {"enabled": False, "model_name": DEFAULT_RERANKING_MODEL, "top_k": None}

    @staticmethod
    def from_env() -> dict[str, Any]:
        """Load reranking configuration from environment variables."""
        model_name = os.getenv("RERANKING_MODEL", DEFAULT_RERANKING_MODEL)
        
        config = {
            "enabled": os.getenv("USE_RERANKING", "false").lower() in ("true", "1", "yes", "on"),
            "model_name": model_name,
            "top_k": int(os.getenv("RERANKING_TOP_K", "0")) or None,
        }
        
        # Add Qwen3-specific configuration if using Qwen3 model
        if model_name.startswith("Qwen/") or "qwen" in model_name.lower():
            config["qwen3"] = {
                "use_flash_attention": os.getenv("QWEN3_USE_FLASH_ATTENTION", "false").lower() == "true",
                "max_length": int(os.getenv("QWEN3_MAX_LENGTH", "8192")),
                "torch_dtype": os.getenv("QWEN3_TORCH_DTYPE", None),
                "device": os.getenv("QWEN3_DEVICE", None),
                "instruction": os.getenv("QWEN3_INSTRUCTION", None),
            }
            
        return config
    
    @staticmethod
    def create_reranking_strategy(config: dict[str, Any]) -> "RerankingStrategy":
        """Create a RerankingStrategy instance from configuration."""
        if not config.get("enabled", False):
            return None
            
        model_name = config.get("model_name", DEFAULT_RERANKING_MODEL)
        
        # Check if this is a Qwen3 model and we have Qwen3-specific config
        if ("Qwen/" in model_name or "qwen" in model_name.lower()) and config.get("qwen3"):
            qwen3_config = config["qwen3"]
            
            if QWEN3_AVAILABLE:
                try:
                    # Parse torch dtype if provided
                    torch_dtype = qwen3_config.get("torch_dtype")
                    dtype = None
                    if torch_dtype:
                        import torch
                        dtype = getattr(torch, torch_dtype, None)
                    
                    qwen3_model = Qwen3Reranker(
                        model_name=model_name,
                        instruction=qwen3_config.get("instruction"),
                        max_length=qwen3_config.get("max_length", 8192),
                        use_flash_attention=qwen3_config.get("use_flash_attention", False),
                        torch_dtype=dtype,
                        device=qwen3_config.get("device"),
                    )
                    
                    return RerankingStrategy.from_model(qwen3_model, model_name)
                except Exception as e:
                    logger.error(f"Failed to create Qwen3 reranker from config: {e}")
                    # Fall back to standard strategy
        
        # Use standard RerankingStrategy
        return RerankingStrategy(model_name=model_name)


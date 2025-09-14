"""
Qwen3 GGUF Reranker Implementation

Fast CPU-based reranker using quantized GGUF models via llama-cpp-python.
Designed for high-performance CPU inference with 5-bit quantization.
"""

from typing import Any, Optional, List
import os
import time
import asyncio

try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    Llama = None
    LLAMA_CPP_AVAILABLE = False

from ...config.logfire_config import get_logger

logger = get_logger(__name__)

DEFAULT_GGUF_MODEL_PATH = "/root/.cache/huggingface/hub/models--DevQuasar--Qwen.Qwen3-Reranker-0.6B-GGUF/Qwen.Qwen3-Reranker-0.6B.Q5_K_M.gguf"
DEFAULT_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"


class Qwen3GGUFReranker:
    """
    High-performance GGUF-based Qwen3 Reranker for CPU inference.
    
    Uses quantized models for fast CPU processing with minimal memory footprint.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_GGUF_MODEL_PATH,
        n_ctx: int = 2048,
        n_threads: int = 4,
        temperature: float = 0.0,
        top_p: float = 1.0,
        verbose: bool = False,
    ):
        """
        Initialize the GGUF Qwen3Reranker.

        Args:
            model_path: Path to GGUF model file
            n_ctx: Context window size
            n_threads: Number of CPU threads to use
            temperature: Sampling temperature (0.0 for deterministic)
            top_p: Top-p sampling parameter
            verbose: Enable verbose logging
        """
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.temperature = temperature
        self.top_p = top_p
        self.verbose = verbose
        self.instruction = DEFAULT_INSTRUCTION
        
        self.model = None
        self.device = "cpu"  # GGUF models run on CPU
        
        self._load_model()

    def _load_model(self) -> bool:
        """Load the GGUF model."""
        if not LLAMA_CPP_AVAILABLE:
            logger.error("llama-cpp-python not available - GGUF reranker disabled")
            return False

        if not os.path.exists(self.model_path):
            logger.error(f"GGUF model file not found: {self.model_path}")
            return False

        try:
            logger.info(f"Loading GGUF reranker model: {self.model_path}")
            
            self.model = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                verbose=self.verbose,
                # Optimize for reranking tasks
                n_batch=1,  # Process one at a time for better latency
            )
            
            logger.info(f"GGUF reranker loaded successfully with {self.n_threads} threads")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load GGUF reranker model: {e}")
            self.model = None
            return False

    def is_available(self) -> bool:
        """Check if the model is loaded and available."""
        return self.model is not None

    def _build_reranking_prompt(self, query: str, document: str) -> str:
        """
        Build a prompt for binary relevance classification using official Qwen3-Reranker format.
        
        Uses the official Qwen3-Reranker template from Hugging Face:
        "<Instruct>: {instruction}\n\n<Query>: {query}\n\n<Document>: {doc}"
        
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
        
        # Look for explicit yes/no answers
        if "yes" in response_lower:
            return 1.0
        elif "no" in response_lower:
            return 0.0
        
        # Look for confidence indicators
        high_confidence_words = ["relevant", "matches", "related", "pertinent", "appropriate"]
        low_confidence_words = ["irrelevant", "unrelated", "inappropriate", "off-topic"]
        
        if any(word in response_lower for word in high_confidence_words):
            return 0.8
        elif any(word in response_lower for word in low_confidence_words):
            return 0.2
        
        # Default to neutral relevance
        return 0.5

    async def predict(self, query_doc_pairs: List[List[str]]) -> List[float]:
        """
        Predict relevance scores for query-document pairs using GGUF model.
        
        Args:
            query_doc_pairs: List of [query, document] pairs
            
        Returns:
            List of relevance scores (0.0 to 1.0)
        """
        if not self.is_available():
            logger.error("GGUF model not available for prediction")
            return [0.0] * len(query_doc_pairs)

        scores = []
        
        try:
            for i, (query, document) in enumerate(query_doc_pairs):
                try:
                    # Build prompt
                    prompt = self._build_reranking_prompt(query, document)
                    
                    # Generate response
                    start_time = time.time()
                    response = self.model(
                        prompt,
                        max_tokens=50,  # Keep response short for binary classification
                        temperature=self.temperature,
                        top_p=self.top_p,
                        stop=["<|im_end|>", "\n\n"],  # Stop tokens
                    )
                    
                    inference_time = time.time() - start_time
                    
                    # Extract text and score
                    response_text = response["choices"][0]["text"]
                    score = self._extract_relevance_score(response_text)
                    scores.append(score)
                    
                    if self.verbose or i == 0:  # Log first prediction details
                        logger.debug(f"Pair {i+1}: {inference_time*1000:.1f}ms -> {score:.3f} ({response_text.strip()[:50]}...)")
                        
                except Exception as e:
                    logger.warning(f"Failed to score pair {i+1}: {e}")
                    scores.append(0.0)
            
            avg_score = sum(scores) / len(scores) if scores else 0.0
            logger.info(f"GGUF reranking completed: {len(scores)} pairs, avg score: {avg_score:.3f}")
            return scores
            
        except Exception as e:
            logger.error(f"GGUF prediction failed: {e}")
            return [0.0] * len(query_doc_pairs)

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the loaded GGUF model."""
        return {
            "model_path": self.model_path,
            "model_type": "GGUF",
            "quantization": "Q5_K_M",
            "available": self.is_available(),
            "device": self.device,
            "n_ctx": self.n_ctx,
            "n_threads": self.n_threads,
            "temperature": self.temperature,
            "instruction": self.instruction,
        }


def detect_query_domain(query: str) -> str:
    """
    Simple domain detection for query categorization.
    
    Args:
        query: Search query to analyze
        
    Returns:
        Detected domain category
    """
    query_lower = query.lower()
    
    # Technical/programming keywords
    if any(word in query_lower for word in ['gpu', 'cuda', 'reranking', 'embedding', 'model', 'architecture', 'docker', 'api']):
        return 'technical'
    
    # Business/product keywords  
    if any(word in query_lower for word in ['user', 'product', 'business', 'strategy', 'feature', 'requirement']):
        return 'business'
    
    # Default to general
    return 'general'
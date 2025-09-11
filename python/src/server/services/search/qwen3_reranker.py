"""
Qwen3 Reranker Implementation

Simplified reranker using Qwen/Qwen3-Reranker models for improved search result ranking.
Optimized for GTX 1080 Ti with float32 precision and simplified instruction handling.
"""

from typing import Any, Optional
import torch
import os

try:
    from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    AutoModel = None
    AutoTokenizer = None
    AutoModelForCausalLM = None
    TRANSFORMERS_AVAILABLE = False

from ...config.logfire_config import get_logger

logger = get_logger(__name__)

DEFAULT_QWEN3_MODEL = "Qwen/Qwen3-Reranker-0.6B"
DEFAULT_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"


class Qwen3Reranker:
    """
    Simplified Qwen3 Reranker implementation optimized for GTX 1080 Ti.
    
    Removes complex dynamic instruction generation in favor of simple, reliable operation.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_QWEN3_MODEL,
        max_length: int = 8192,
        use_flash_attention: bool = False,
        torch_dtype: str = "float32",
        device: Optional[str] = None,
    ):
        """
        Initialize the Qwen3Reranker.

        Args:
            model_name: HuggingFace model path for Qwen3 reranker
            max_length: Maximum token length for inputs
            use_flash_attention: Whether to use flash attention (disabled for Pascal)
            torch_dtype: PyTorch data type as string (e.g., 'float32', 'float16')
            device: Device to load model on (e.g., 'cuda', 'cpu')
        """
        self.model_name = model_name
        self.max_length = max_length
        self.use_flash_attention = use_flash_attention
        self.instruction = DEFAULT_INSTRUCTION
        
        # Parse torch dtype from string
        self.torch_dtype = getattr(torch, torch_dtype) if torch_dtype else torch.float32
        
        self.model = None
        self.tokenizer = None
        self.token_false_id = None
        self.token_true_id = None
        self.prefix_tokens = None
        self.suffix_tokens = None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self._load_model()

    def _load_model(self) -> bool:
        """Load the Qwen3 reranker model and tokenizer."""
        if not TRANSFORMERS_AVAILABLE:
            logger.error("transformers library not available - Qwen3 reranker disabled")
            return False

        try:
            logger.info(f"Loading Qwen3 reranker model: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                padding_side='left'
            )
            
            # Prepare model loading kwargs - optimized for GTX 1080 Ti
            model_kwargs = {
                "torch_dtype": self.torch_dtype,  # Force float32 for Pascal compatibility
            }
            
            # Flash attention disabled for Pascal architecture
            if self.use_flash_attention:
                logger.warning("Flash attention disabled for Pascal architecture compatibility")
                model_kwargs["attn_implementation"] = "eager"
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                **model_kwargs
            ).eval()
            
            # Move to device
            if self.device != "cpu":
                try:
                    self.model = self.model.to(self.device)
                    logger.info(f"Moved model to {self.device}")
                except Exception as e:
                    logger.warning(f"Failed to move model to {self.device}, using CPU: {e}")
                    self.device = "cpu"
            
            # Setup token IDs for yes/no responses
            self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
            self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
            
            # Setup simplified prefix and suffix tokens
            prefix = ("﻿<|im_start|>system\\nJudge whether the Document meets the requirements "
                     "based on the Query and the Instruct provided. Note that the answer can "
                     "only be \\\"yes\\\" or \\\"no\\\".<|im_end|>\\n<|im_start|>user\\n")
            suffix = "<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n"
            
            self.prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
            self.suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)
            
            logger.info(f"Qwen3 reranker loaded successfully on {self.device} with {self.torch_dtype}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Qwen3 reranker model: {e}")
            self.model = None
            self.tokenizer = None
            return False

    def is_available(self) -> bool:
        """Check if the model is loaded and available."""
        return self.model is not None and self.tokenizer is not None

    def _build_input_text(self, query: str, document: str) -> str:
        """
        Build input text for the reranker model with simplified instruction.
        
        Args:
            query: Search query
            document: Document content to rank
            
        Returns:
            Formatted input text for the model
        """
        return (
            f"Query: {query}\\n"
            f"Document: {document}\\n"  
            f"Instruct: {self.instruction}"
        )

    async def predict(self, query_doc_pairs: list[list[str]]) -> list[float]:
        """
        Predict relevance scores for query-document pairs.
        
        Args:
            query_doc_pairs: List of [query, document] pairs
            
        Returns:
            List of relevance scores (0.0 to 1.0)
        """
        if not self.is_available():
            logger.error("Model not available for prediction")
            return [0.0] * len(query_doc_pairs)

        try:
            scores = []
            
            # Process each query-document pair
            for query, document in query_doc_pairs:
                try:
                    # Build input text
                    input_text = self._build_input_text(query, document)
                    
                    # Tokenize with prefix and suffix
                    input_ids = (
                        self.prefix_tokens + 
                        self.tokenizer.encode(input_text, add_special_tokens=False) +
                        self.suffix_tokens
                    )
                    
                    # Truncate if too long
                    if len(input_ids) > self.max_length:
                        # Keep prefix, truncate middle, keep suffix
                        prefix_len = len(self.prefix_tokens)
                        suffix_len = len(self.suffix_tokens)
                        available_len = self.max_length - prefix_len - suffix_len
                        
                        middle_tokens = self.tokenizer.encode(input_text, add_special_tokens=False)
                        truncated_middle = middle_tokens[:available_len]
                        
                        input_ids = self.prefix_tokens + truncated_middle + self.suffix_tokens
                    
                    # Convert to tensor
                    input_tensor = torch.tensor([input_ids], device=self.device)
                    
                    # Get model prediction
                    with torch.no_grad():
                        outputs = self.model(input_tensor)
                        logits = outputs.logits[0, -1, :]  # Last token logits
                        
                        # Get yes/no probabilities
                        yes_prob = torch.softmax(logits[[self.token_false_id, self.token_true_id]], dim=0)[1]
                        score = float(yes_prob.cpu())
                        scores.append(score)
                        
                except Exception as e:
                    logger.warning(f"Failed to score query-document pair: {e}")
                    scores.append(0.0)
            
            logger.debug(f"Processed {len(query_doc_pairs)} pairs, avg score: {sum(scores)/len(scores):.3f}")
            return scores
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return [0.0] * len(query_doc_pairs)

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the loaded model."""
        info = {
            "model_name": self.model_name,
            "available": self.is_available(),
            "device": self.device,
            "torch_dtype": str(self.torch_dtype),
            "max_length": self.max_length,
            "use_flash_attention": self.use_flash_attention,
        }
        
        if self.model:
            try:
                # Get memory usage if on CUDA
                if self.device.startswith("cuda"):
                    memory_allocated = torch.cuda.memory_allocated(self.device) / 1024**3
                    memory_reserved = torch.cuda.memory_reserved(self.device) / 1024**3
                    info.update({
                        "gpu_memory_allocated_gb": round(memory_allocated, 2),
                        "gpu_memory_reserved_gb": round(memory_reserved, 2),
                    })
            except Exception as e:
                logger.debug(f"Could not get GPU memory info: {e}")
        
        return info
"""
Qwen3 Reranker Implementation

Custom reranker using Qwen/Qwen3-Reranker-4B model for improved search result ranking.
This implementation adapts the Qwen3 reranker to work with Archon's RerankingStrategy interface.
"""

from typing import Any, Optional
import torch

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

DEFAULT_QWEN3_MODEL = "Qwen/Qwen3-Reranker-4B"
DEFAULT_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"


class Qwen3Reranker:
    """
    Qwen3 Reranker implementation that adapts the Qwen3-Reranker-4B model
    to work with Archon's RerankingStrategy interface.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_QWEN3_MODEL,
        instruction: Optional[str] = None,
        max_length: int = 8192,
        use_flash_attention: bool = False,
        torch_dtype: Optional[torch.dtype] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize the Qwen3Reranker.

        Args:
            model_name: HuggingFace model path for Qwen3 reranker
            instruction: Task instruction for the reranker
            max_length: Maximum token length for inputs
            use_flash_attention: Whether to use flash attention for acceleration
            torch_dtype: PyTorch data type (e.g., torch.float16)
            device: Device to load model on (e.g., 'cuda', 'cpu')
        """
        self.model_name = model_name
        self.instruction = instruction or DEFAULT_INSTRUCTION
        self.max_length = max_length
        self.use_flash_attention = use_flash_attention
        self.torch_dtype = torch_dtype
        
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
            
            # Prepare model loading kwargs
            model_kwargs = {}
            if self.torch_dtype:
                model_kwargs["torch_dtype"] = self.torch_dtype
            if self.use_flash_attention:
                model_kwargs["attn_implementation"] = "flash_attention_2"
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                **model_kwargs
            ).eval()
            
            # Move to device if specified
            if self.device != "cpu":
                try:
                    self.model = self.model.to(self.device)
                except Exception as e:
                    logger.warning(f"Failed to move model to {self.device}, using CPU: {e}")
                    self.device = "cpu"
            
            # Setup token IDs
            self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
            self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
            
            # Setup prefix and suffix tokens
            prefix = ("<|im_start|>system\nJudge whether the Document meets the requirements "
                     "based on the Query and the Instruct provided. Note that the answer can "
                     "only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
            suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            
            self.prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
            self.suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)
            
            logger.info(f"Qwen3 reranker loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Qwen3 reranker model: {e}")
            self.model = None
            self.tokenizer = None
            return False

    def is_available(self) -> bool:
        """Check if the model is loaded and available."""
        return self.model is not None and self.tokenizer is not None

    def format_instruction(self, query: str, document: str) -> str:
        """
        Format the instruction, query, and document according to Qwen3 format.
        
        Args:
            query: Search query
            document: Document to evaluate
            
        Returns:
            Formatted instruction string
        """
        return f"<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: {document}"

    def process_inputs(self, pairs: list[str]) -> dict[str, torch.Tensor]:
        """
        Process input pairs into tokenized format expected by the model.
        
        Args:
            pairs: List of formatted instruction strings
            
        Returns:
            Tokenized inputs as tensors
        """
        # Tokenize without special tokens first
        inputs = self.tokenizer(
            pairs,
            padding=False,
            truncation='longest_first',
            return_attention_mask=False,
            max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        )
        
        # Add prefix and suffix tokens to each sequence
        for i, token_ids in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = self.prefix_tokens + token_ids + self.suffix_tokens
        
        # Pad and convert to tensors
        inputs = self.tokenizer.pad(
            inputs,
            padding=True,
            return_tensors="pt",
            max_length=self.max_length
        )
        
        # Move to model device
        for key in inputs:
            inputs[key] = inputs[key].to(self.model.device)
            
        return inputs

    @torch.no_grad()
    def compute_logits(self, inputs: dict[str, torch.Tensor]) -> list[float]:
        """
        Compute relevance scores from model logits.
        
        Args:
            inputs: Tokenized inputs
            
        Returns:
            List of relevance scores between 0 and 1
        """
        # Get model outputs
        outputs = self.model(**inputs)
        batch_scores = outputs.logits[:, -1, :]  # Last token logits
        
        # Extract yes/no token probabilities
        true_vector = batch_scores[:, self.token_true_id]
        false_vector = batch_scores[:, self.token_false_id]
        
        # Stack and apply log softmax
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        
        # Get probability of "yes" token
        scores = batch_scores[:, 1].exp().tolist()
        
        return scores

    def predict(self, query_doc_pairs: list[list[str]]) -> list[float]:
        """
        Predict relevance scores for query-document pairs.
        
        This method implements the interface expected by RerankingStrategy.
        
        Args:
            query_doc_pairs: List of [query, document] pairs
            
        Returns:
            List of relevance scores (higher = more relevant)
        """
        if not self.is_available():
            logger.warning("Qwen3 reranker not available, returning zero scores")
            return [0.0] * len(query_doc_pairs)
        
        if not query_doc_pairs:
            return []
        
        try:
            # Format all pairs according to Qwen3 instruction format
            formatted_pairs = []
            for query, document in query_doc_pairs:
                formatted_pair = self.format_instruction(query, document)
                formatted_pairs.append(formatted_pair)
            
            # Process inputs in batches to handle memory constraints
            batch_size = min(8, len(formatted_pairs))  # Conservative batch size
            all_scores = []
            
            for i in range(0, len(formatted_pairs), batch_size):
                batch_pairs = formatted_pairs[i:i + batch_size]
                
                # Tokenize and process batch
                inputs = self.process_inputs(batch_pairs)
                
                # Compute scores
                batch_scores = self.compute_logits(inputs)
                all_scores.extend(batch_scores)
                
                logger.debug(f"Processed batch {i//batch_size + 1}, scores: {batch_scores}")
            
            logger.debug(f"Qwen3 reranker processed {len(query_doc_pairs)} pairs, "
                        f"score range: {min(all_scores):.3f}-{max(all_scores):.3f}")
            
            return all_scores
            
        except Exception as e:
            logger.error(f"Error in Qwen3 reranker prediction: {e}")
            # Return neutral scores on error
            return [0.5] * len(query_doc_pairs)

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "available": self.is_available(),
            "device": self.device,
            "max_length": self.max_length,
            "use_flash_attention": self.use_flash_attention,
            "torch_dtype": str(self.torch_dtype) if self.torch_dtype else None,
            "transformers_available": TRANSFORMERS_AVAILABLE,
        }


def create_qwen3_reranker(
    model_name: str = DEFAULT_QWEN3_MODEL,
    instruction: Optional[str] = None,
    **kwargs
) -> Qwen3Reranker:
    """
    Factory function to create a Qwen3Reranker instance.
    
    Args:
        model_name: Model name/path
        instruction: Task instruction
        **kwargs: Additional arguments for Qwen3Reranker
        
    Returns:
        Configured Qwen3Reranker instance
    """
    return Qwen3Reranker(
        model_name=model_name,
        instruction=instruction,
        **kwargs
    )
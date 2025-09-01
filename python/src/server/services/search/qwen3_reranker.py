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
    Qwen3 Reranker implementation with dynamic instruction generation.
    
    Supports both static instructions and LLM-generated dynamic instructions
    based on query context and domain.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_QWEN3_MODEL,
        instruction: Optional[str] = None,
        max_length: int = 8192,
        use_flash_attention: bool = False,
        torch_dtype: Optional[torch.dtype] = None,
        device: Optional[str] = None,
        enable_dynamic_instructions: bool = True,
    ):
        """
        Initialize the Qwen3Reranker.

        Args:
            model_name: HuggingFace model path for Qwen3 reranker
            instruction: Static task instruction (overrides dynamic generation if set)
            max_length: Maximum token length for inputs
            use_flash_attention: Whether to use flash attention for acceleration
            torch_dtype: PyTorch data type (e.g., torch.float16)
            device: Device to load model on (e.g., 'cuda', 'cpu')
            enable_dynamic_instructions: Whether to enable LLM-generated instructions
        """
        self.model_name = model_name
        self.static_instruction = instruction
        self.max_length = max_length
        self.use_flash_attention = use_flash_attention
        self.torch_dtype = torch_dtype
        self.enable_dynamic_instructions = enable_dynamic_instructions
        
        # Initialize instruction generator if dynamic instructions enabled
        self.instruction_generator = InstructionGenerator() if enable_dynamic_instructions else None
        
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
                except Exception as e
class InstructionHistoryService:
    """Service to track and learn from instruction generation history for better domain detection."""
    
    def __init__(self):
        self.history_cache = {}
        self.domain_patterns = {}
        self._load_history()
    
    def _get_history_file_path(self) -> str:
        """Get the path to the instruction history file."""
        import os
        from pathlib import Path
        
        # Store in the project's cache directory
        cache_dir = Path(__file__).parent.parent.parent.parent.parent / "cache"
        cache_dir.mkdir(exist_ok=True)
        return str(cache_dir / "instruction_history.json")
    
    def _load_history(self):
        """Load instruction history from persistent storage."""
        try:
            import json
            history_file = self._get_history_file_path()
            
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    self.history_cache = data.get('history', {})
                    self.domain_patterns = data.get('patterns', {})
                    logger.debug(f"Loaded {len(self.history_cache)} instruction history entries")
        except Exception as e:
            logger.warning(f"Failed to load instruction history: {e}")
            self.history_cache = {}
            self.domain_patterns = {}
    
    def _save_history(self):
        """Save instruction history to persistent storage."""
        try:
            import json
            history_file = self._get_history_file_path()
            
            data = {
                'history': self.history_cache,
                'patterns': self.domain_patterns,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(history_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.debug(f"Saved {len(self.history_cache)} instruction history entries")
        except Exception as e:
            logger.error(f"Failed to save instruction history: {e}")
    
    def record_instruction(self, query: str, detected_domain: str, generated_instruction: str, 
                          document_context: Optional[list[str]] = None, success_score: Optional[float] = None):
        """
        Record a successful query-domain-instruction combination.
        
        Args:
            query: The original search query
            detected_domain: The domain that was detected
            generated_instruction: The instruction that was generated
            document_context: Optional document snippets that influenced detection
            success_score: Optional success score (e.g., average rerank improvement)
        """
        try:
            # Create a record
            record = {
                'query': query,
                'detected_domain': detected_domain,
                'generated_instruction': generated_instruction,
                'timestamp': datetime.now().isoformat(),
                'query_length': len(query),
                'query_words': len(query.split()),
                'success_score': success_score
            }
            
            # Add document context if provided
            if document_context:
                record['document_context'] = document_context[:3]  # Store first 3 for context
                record['context_keywords'] = self._extract_context_keywords(document_context)
            
            # Store in history
            query_hash = hash(query.lower().strip())
            self.history_cache[str(query_hash)] = record
            
            # Update domain patterns for better detection
            self._update_domain_patterns(query, detected_domain, document_context)
            
            # Save to disk periodically
            if len(self.history_cache) % 10 == 0:  # Save every 10 entries
                self._save_history()
                
            logger.debug(f"Recorded instruction for query: '{query[:50]}...' -> domain: {detected_domain}")
            
        except Exception as e:
            logger.error(f"Failed to record instruction history: {e}")
    
    def _extract_context_keywords(self, document_context: list[str]) -> list[str]:
        """Extract meaningful keywords from document context."""
        import re
        
        all_text = ' '.join(document_context).lower()
        # Extract technical terms, programming languages, frameworks
        keywords = re.findall(r'\b(?:javascript|python|react|vue|angular|node|express|django|flask|'
                             r'security|authentication|jwt|oauth|api|database|mongodb|postgresql|'
                             r'docker|kubernetes|aws|azure|gcp|linux|ubuntu|windows|'
                             r'function|class|method|async|await|promise|callback|'
                             r'server|client|frontend|backend|fullstack)\b', all_text)
        
        return list(set(keywords))  # Remove duplicates
    
    def _update_domain_patterns(self, query: str, domain: str, document_context: Optional[list[str]] = None):
        """Update domain detection patterns based on successful detections."""
        if domain not in self.domain_patterns:
            self.domain_patterns[domain] = {
                'keywords': set(),
                'phrases': set(),
                'context_keywords': set(),
                'confidence_score': 1.0
            }
        
        # Extract query keywords
        query_words = set(word.lower() for word in re.findall(r'\b\w+\b', query) if len(word) > 2)
        self.domain_patterns[domain]['keywords'].update(query_words)
        
        # Extract phrases (2-3 word combinations)
        words = query.lower().split()
        for i in range(len(words) - 1):
            phrase = ' '.join(words[i:i+2])
            self.domain_patterns[domain]['phrases'].add(phrase)
            
            if i < len(words) - 2:
                phrase3 = ' '.join(words[i:i+3])
                self.domain_patterns[domain]['phrases'].add(phrase3)
        
        # Add context keywords
        if document_context:
            context_keywords = self._extract_context_keywords(document_context)
            self.domain_patterns[domain]['context_keywords'].update(context_keywords)
        
        # Convert sets to lists for JSON serialization
        for pattern_type in ['keywords', 'phrases', 'context_keywords']:
            if isinstance(self.domain_patterns[domain][pattern_type], set):
                self.domain_patterns[domain][pattern_type] = list(self.domain_patterns[domain][pattern_type])
    
    def get_improved_domain_detection(self, query: str, document_context: Optional[list[str]] = None) -> tuple[str, float]:
        """
        Get improved domain detection based on historical patterns.
        
        Args:
            query: The search query
            document_context: Optional document snippets for additional context
            
        Returns:
            Tuple of (detected_domain, confidence_score)
        """
        query_lower = query.lower()
        domain_scores = {}
        
        # Score each known domain based on patterns
        for domain, patterns in self.domain_patterns.items():
            score = 0.0
            
            # Check keyword matches
            query_words = set(word.lower() for word in re.findall(r'\b\w+\b', query) if len(word) > 2)
            keyword_matches = len(query_words.intersection(set(patterns.get('keywords', []))))
            score += keyword_matches * 2.0  # Weight keyword matches highly
            
            # Check phrase matches
            for phrase in patterns.get('phrases', []):
                if phrase in query_lower:
                    score += 3.0  # Phrases are even more important
            
            # Check context keywords if available
            if document_context:
                context_keywords = self._extract_context_keywords(document_context)
                context_matches = len(set(context_keywords).intersection(set(patterns.get('context_keywords', []))))
                score += context_matches * 1.5
            
            # Apply base confidence
            score *= patterns.get('confidence_score', 1.0)
            
            domain_scores[domain] = score
        
        # Get the highest scoring domain
        if domain_scores:
            best_domain = max(domain_scores.items(), key=lambda x: x[1])
            if best_domain[1] > 0.5:  # Minimum confidence threshold
                return best_domain[0], min(best_domain[1] / 10.0, 1.0)  # Normalize confidence
        
        # Fall back to original detection
        fallback_domain = detect_query_domain(query, document_context)
        return fallback_domain, 0.3  # Lower confidence for fallback
    
    def get_similar_instructions(self, query: str, domain: str, limit: int = 3) -> list[dict]:
        """
        Find similar historical instructions for reference.
        
        Args:
            query: Current query
            domain: Detected domain
            limit: Maximum number of similar instructions to return
            
        Returns:
            List of similar instruction records
        """
        similar = []
        query_words = set(query.lower().split())
        
        for record in self.history_cache.values():
            if record['detected_domain'] == domain:
                record_words = set(record['query'].lower().split())
                similarity = len(query_words.intersection(record_words)) / len(query_words.union(record_words))
                
                if similarity > 0.2:  # Minimum similarity threshold
                    similar.append({
                        'record': record,
                        'similarity': similarity
                    })
        
        # Sort by similarity and return top results
        similar.sort(key=lambda x: x['similarity'], reverse=True)
        return [item['record'] for item in similar[:limit]]
    
    def get_statistics(self) -> dict:
        """Get statistics about instruction history."""
        if not self.history_cache:
            return {'total_records': 0}
        
        domain_counts = {}
        success_scores = []
        
        for record in self.history_cache.values():
            domain = record['detected_domain']
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
            
            if record.get('success_score'):
                success_scores.append(record['success_score'])
        
        stats = {
            'total_records': len(self.history_cache),
            'domain_distribution': domain_counts,
            'known_domains': len(self.domain_patterns),
            'avg_success_score': sum(success_scores) / len(success_scores) if success_scores else None
        }
        
        return stats


# Global instance
_instruction_history_service = None

def get_instruction_history_service() -> InstructionHistoryService:
    """Get the global instruction history service instance."""
    global _instruction_history_service
    if _instruction_history_service is None:
        _instruction_history_service = InstructionHistoryService()
    return _instruction_history_service
def detect_query_domain(query: str, document_context: list[str] = None) -> str:
    """
    Automatically detect the domain/context of a query for instruction generation.
    
    Args:
        query: The search query
        document_context: Optional list of document snippets for additional context
        
    Returns:
        Detected domain (e.g., 'code', 'technical', 'security', 'general')
    """
    query_lower = query.lower()
    
    # Programming/code related
    code_keywords = [
        'function', 'method', 'class', 'api', 'library', 'framework', 
        'syntax', 'code', 'programming', 'javascript', 'python', 'java',
        'react', 'vue', 'angular', 'node', 'express', 'django', 'flask',
        'implementation', 'algorithm', 'data structure', 'debugging'
    ]
    
    # Security related
    security_keywords = [
        'security', 'vulnerability', 'attack', 'authentication', 'authorization',
        'encryption', 'jwt', 'oauth', 'cors', 'xss', 'sql injection', 'csrf',
        'ssl', 'tls', 'certificate', 'firewall', 'penetration', 'exploit'
    ]
    
    # Technical/system administration
    technical_keywords = [
        'server', 'database', 'deployment', 'docker', 'kubernetes', 'aws',
        'cloud', 'infrastructure', 'monitoring', 'logging', 'performance',
        'optimization', 'scaling', 'load balancer', 'nginx', 'apache'
    ]
    
    # Documentation/tutorial
    tutorial_keywords = [
        'how to', 'tutorial', 'guide', 'setup', 'install', 'configure',
        'getting started', 'beginner', 'step by step', 'introduction'
    ]
    
    # Troubleshooting
    troubleshoot_keywords = [
        'error', 'problem', 'issue', 'fix', 'debug', 'troubleshoot',
        'not working', 'fails', 'crash', 'exception', 'bug'
    ]
    
    # Check query against keyword categories
    if any(keyword in query_lower for keyword in code_keywords):
        return 'code'
    elif any(keyword in query_lower for keyword in security_keywords):
        return 'security'  
    elif any(keyword in query_lower for keyword in technical_keywords):
        return 'technical'
    elif any(keyword in query_lower for keyword in tutorial_keywords):
        return 'tutorial'
    elif any(keyword in query_lower for keyword in troubleshoot_keywords):
        return 'troubleshooting'
    
    # Check document context if available
    if document_context:
        context_text = ' '.join(document_context).lower()
        if any(keyword in context_text for keyword in code_keywords):
            return 'code'
        elif any(keyword in context_text for keyword in security_keywords):
            return 'security'
        elif any(keyword in context_text for keyword in technical_keywords):
            return 'technical'
    
    return 'general'
:
                    logger.warning(f"Failed to move model to {self.device}, using CPU: {e}")
                    self.device = "cpu"
            
            # Setup token IDs
            self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
            self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
            
            # Setup prefix and suffix tokens
            prefix = ("﻿<|im_start|>system\\nJudge whether the Document meets the requirements "
                     "based on the Query and the Instruct provided. Note that the answer can "
                     "only be \\\"yes\\\" or \\\"no\\\".<|im_end|>\\n<|im_start|>user\\n")
            suffix = "<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n"
            
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

    async def get_instruction(self, query: str, domain: Optional[str] = None) -> str:
        """
        Get instruction for the query, either static or dynamically generated.
        
        Args:
            query: Search query
            domain: Optional domain context
            
        Returns:
            Instruction to use for reranking
        """
        # Use static instruction if provided
        if self.static_instruction:
            return self.static_instruction
        
        # Use dynamic instruction generation if enabled
        if self.enable_dynamic_instructions and self.instruction_generator:
            try:
                return await self.instruction_generator.generate_instruction(query, domain)
            except Exception as e:
                logger.warning(f"Dynamic instruction generation failed: {e}")
        
        # Fallback to default
        return DEFAULT_INSTRUCTION

    async def format_instruction(self, query: str, document: str, domain: Optional[str] = None) -> str:
        """
        Format the instruction, query, and document according to Qwen3 format.
        
        Args:
            query: Search query
            document: Document to evaluate
            domain: Optional domain context for dynamic instructions
            
        Returns:
            Formatted instruction string
        """
        instruction = await self.get_instruction(query, domain)
        return f"<Instruct>: {instruction}\\n<Query>: {query}\\n<Document>: {document}"

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

    async def predict(self, query_doc_pairs: list[list[str]], domain: Optional[str] = None) -> list[float]:
        """
        Predict relevance scores for query-document pairs with dynamic instructions.
        
        Args:
            query_doc_pairs: List of [query, document] pairs
            domain: Optional domain context for instruction generation
            
        Returns:
            List of relevance scores (higher = more relevant)
        """
        if not self.is_available():
            logger.warning("Qwen3 reranker not available, returning zero scores")
            return [0.0] * len(query_doc_pairs)
        
        if not query_doc_pairs:
            return []
        
        try:
            # Format all pairs with dynamic instructions
            formatted_pairs = []
            query = query_doc_pairs[0][0] if query_doc_pairs else ""  # Get query from first pair
            
            for query_text, document in query_doc_pairs:
                formatted_pair = await self.format_instruction(query_text, document, domain)
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
            "dynamic_instructions_enabled": self.enable_dynamic_instructions,
            "static_instruction": self.static_instruction,
        }

class InstructionGenerator:
    """Generates dynamic reranking instructions using the configured LLM."""
    
    def __init__(self):
        self.instruction_cache = {}
        
    async def generate_instruction(self, query: str, domain: Optional[str] = None) -> str:
        """
        Generate a custom reranking instruction based on the query context.
        
        Args:
            query: The search query
            domain: Optional domain context (e.g., 'code', 'technical', 'general')
            
        Returns:
            Custom instruction tailored to the query
        """
        # Create cache key
        cache_key = f"{query}_{domain or 'general'}"
        
        # Check cache first
        if cache_key in self.instruction_cache:
            return self.instruction_cache[cache_key]
        
        try:
            from ..llm_provider_service import get_llm_client
            
            # Create instruction generation prompt
            system_prompt = """You are an expert at creating reranking instructions for search systems. Your task is to analyze a search query and generate a precise, specific instruction that will help a reranker model identify the most relevant documents.

Guidelines:
- Keep instructions concise (1-2 sentences)
- Focus on what makes a document relevant to the specific query
- Consider the query type (factual, technical, code, conceptual, etc.)
- Be specific about relevance criteria
- Use action verbs like "rank", "prioritize", "identify", "evaluate"

Examples:
Query: "React useEffect cleanup function" -> "Rank code examples and documentation by how clearly they explain React useEffect cleanup patterns and implementation details"
Query: "machine learning overfitting" -> "Prioritize documents that explain overfitting concepts, causes, and practical prevention techniques in machine learning"
Query: "JWT security best practices" -> "Identify documents that provide comprehensive JWT security guidelines, vulnerability explanations, and implementation recommendations"
"""
            
            user_prompt = f"""Generate a reranking instruction for this search query:

Query: "{query}"
Domain: {domain or "general"}

Return only the instruction text, no explanations."""

            async with get_llm_client() as client:
                response = await client.chat.completions.create(
                    model="gpt-3.5-turbo",  # Use fast model for instruction generation
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.3,  # Lower temperature for consistent instructions
                    max_tokens=100
                )
                
                instruction = response.choices[0].message.content.strip()
                
                # Cache the instruction
                self.instruction_cache[cache_key] = instruction
                
                logger.debug(f"Generated custom instruction for query '{query}': {instruction}")
                return instruction
                
        except Exception as e:
            logger.warning(f"Failed to generate custom instruction: {e}")
            # Fallback to domain-based defaults
            return self._get_fallback_instruction(query, domain)
    
    def _get_fallback_instruction(self, query: str, domain: Optional[str] = None) -> str:
        """Fallback instruction generation based on query patterns and domain."""
        query_lower = query.lower()
        
        # Code-related queries
        if any(keyword in query_lower for keyword in ['code', 'function', 'class', 'method', 'api', 'syntax', 'implementation']):
            return "Rank code examples and technical documentation by implementation relevance and code quality"
        
        # Technical/how-to queries
        elif any(keyword in query_lower for keyword in ['how to', 'tutorial', 'guide', 'setup', 'install', 'configure']):
            return "Prioritize step-by-step guides and tutorials that directly address the specific task or problem"
        
        # Conceptual/definition queries
        elif any(keyword in query_lower for keyword in ['what is', 'define', 'explain', 'concept', 'theory']):
            return "Identify documents that provide clear explanations and comprehensive coverage of the requested concept"
        
        # Problem/troubleshooting queries
        elif any(keyword in query_lower for keyword in ['error', 'problem', 'issue', 'fix', 'debug', 'troubleshoot']):
            return "Rank solutions and troubleshooting guides by practical applicability to the specific problem"
        
        # Security-related queries
        elif any(keyword in query_lower for keyword in ['security', 'vulnerability', 'attack', 'authentication', 'authorization']):
            return "Prioritize security documentation, best practices, and vulnerability analysis relevant to the query"
        
        # Default instruction
        else:
            return "Rank documents by relevance and comprehensiveness in addressing the search query"
    
    def clear_cache(self):
        """Clear the instruction cache."""
        self.instruction_cache.clear()



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
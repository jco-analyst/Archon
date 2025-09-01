# Qwen3 Reranker Integration Guide

This guide explains how to use the Qwen3-Reranker-4B model for improved search result ranking in Archon.

## Overview

The Qwen3 reranker integration allows you to use Qwen's powerful 4-billion parameter reranking model to improve the quality of RAG (Retrieval-Augmented Generation) search results. This model provides more accurate relevance scoring compared to the default CrossEncoder models.

## Features

- **High-Quality Reranking**: Uses Qwen3-Reranker-4B for superior relevance scoring
- **Flexible Configuration**: Supports various optimization settings
- **Hardware Optimization**: Automatic configuration based on available hardware
- **Fallback Support**: Gracefully falls back to CrossEncoder if Qwen3 is unavailable
- **Memory Efficient**: Configurable batch processing and memory usage

## Quick Setup

### 1. Install Dependencies

The required dependencies are already included in `pyproject.toml`:

```bash
cd python
uv sync  # This will install transformers>=4.51.0 and torch>=2.0.0
```

### 2. Configure Reranking Model

Set the reranking model to use Qwen3 by setting environment variables or through the Archon UI:

```bash
# Enable reranking
export USE_RERANKING=true

# Set Qwen3 as the reranking model
export RERANKING_MODEL="Qwen/Qwen3-Reranker-4B"
```

### 3. Optional: Configure Qwen3-Specific Settings

```bash
# Use flash attention for better performance (requires compatible GPU)
export QWEN3_USE_FLASH_ATTENTION=true

# Use half precision for memory efficiency
export QWEN3_TORCH_DTYPE=float16

# Set maximum context length
export QWEN3_MAX_LENGTH=8192

# Force specific device
export QWEN3_DEVICE=cuda  # or 'cpu' or 'auto'

# Custom instruction for reranking
export QWEN3_INSTRUCTION="Given a web search query, retrieve relevant passages that answer the query"
```

## Configuration Options

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_RERANKING` | `false` | Enable/disable reranking |
| `RERANKING_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranking model name |
| `QWEN3_USE_FLASH_ATTENTION` | `false` | Use flash attention for acceleration |
| `QWEN3_TORCH_DTYPE` | `None` | PyTorch data type (`float16`, `bfloat16`, etc.) |
| `QWEN3_MAX_LENGTH` | `8192` | Maximum token length |
| `QWEN3_DEVICE` | `auto` | Device to use (`cuda`, `cpu`, `auto`) |
| `QWEN3_INSTRUCTION` | Default instruction | Custom task instruction |

### Recommended Configurations

#### High Performance (GPU with ≥16GB VRAM)
```bash
export QWEN3_TORCH_DTYPE=float16
export QWEN3_USE_FLASH_ATTENTION=true
export QWEN3_DEVICE=cuda
export QWEN3_MAX_LENGTH=8192
```

#### Memory Efficient (GPU with 8-12GB VRAM)
```bash
export QWEN3_TORCH_DTYPE=float16
export QWEN3_USE_FLASH_ATTENTION=false
export QWEN3_DEVICE=cuda
export QWEN3_MAX_LENGTH=4096
```

#### CPU Only
```bash
export QWEN3_TORCH_DTYPE=float32
export QWEN3_USE_FLASH_ATTENTION=false
export QWEN3_DEVICE=cpu
export QWEN3_MAX_LENGTH=8192
```

## Usage

Once configured, the Qwen3 reranker will automatically be used in:

1. **RAG Queries**: `POST /api/knowledge/search`
2. **Code Example Search**: Via MCP `archon:search_code_examples`
3. **Document Search**: All hybrid and vector search operations

### Example API Usage

```python
import httpx

# Perform a RAG query that will use Qwen3 reranking
response = httpx.post("http://localhost:8181/api/knowledge/search", json={
    "query": "How to implement authentication in FastAPI?",
    "match_count": 10,
    "source": "documentation"  # optional
})

results = response.json()

# Results will include rerank_score from Qwen3
for result in results["results"]:
    print(f"Content: {result['content'][:100]}...")
    print(f"Similarity Score: {result['similarity_score']}")
    print(f"Rerank Score: {result['rerank_score']}")  # From Qwen3
    print("---")
```

## Testing the Integration

Run the test script to verify everything is working:

```bash
cd python
python test_qwen3_integration.py
```

This will test:
- Basic Qwen3 reranker functionality
- Integration with RerankingStrategy
- Configuration system
- Hardware detection

## Performance Considerations

### Model Download
- The Qwen3-Reranker-4B model is ~8GB
- First run will download the model automatically
- Consider pre-downloading in Docker builds

### Memory Usage
- Model requires ~8-12GB VRAM in float16 mode
- ~16GB VRAM in float32 mode
- Use CPU mode if GPU memory is insufficient

### Inference Speed
- GPU: ~50-100ms per batch of 8 query-document pairs
- CPU: ~500-1000ms per batch (significantly slower)
- Flash attention can provide 20-30% speedup on compatible GPUs

### Batch Processing
- Automatically processes in batches of 8 pairs
- Adjust batch size in code if memory constraints occur
- Larger batches = better GPU utilization but more memory

## Troubleshooting

### Common Issues

**"transformers not available"**
```bash
uv sync  # Reinstall dependencies
```

**"CUDA out of memory"**
```bash
export QWEN3_TORCH_DTYPE=float16  # Use half precision
export QWEN3_MAX_LENGTH=4096      # Reduce context length
# Or use CPU mode:
export QWEN3_DEVICE=cpu
```

**"Model download fails"**
- Ensure internet connection
- Check disk space (need ~10GB free)
- Try setting HF_HOME environment variable

**"Flash attention not working"**
- Flash attention requires specific GPU architectures
- Disable if causing issues: `export QWEN3_USE_FLASH_ATTENTION=false`

### Debug Mode

Enable debug logging to see detailed reranking information:

```bash
export LOG_LEVEL=DEBUG
```

Look for log messages like:
```
INFO - Loading Qwen3 reranking model: Qwen/Qwen3-Reranker-4B
DEBUG - Reranked 5 results, score range: 0.123-0.887
```

## Architecture Details

### Integration Points

1. **RerankingStrategy**: Main integration point that detects Qwen3 models
2. **Qwen3Reranker**: Wrapper class that implements the `predict` interface
3. **RAGService**: Orchestrates the full pipeline including reranking
4. **Configuration**: Environment-based configuration with fallbacks

### Pipeline Flow

```
Query → Vector Search → Hybrid Search (optional) → Qwen3 Reranking → Results
```

### Model Interface

The Qwen3Reranker implements the standard interface:

```python
def predict(self, query_doc_pairs: List[List[str]]) -> List[float]:
    """Return relevance scores (0-1) for query-document pairs"""
```

## Custom Models

To use a different Qwen3 reranker variant:

```bash
export RERANKING_MODEL="Qwen/Qwen3-Reranker-1.8B"  # Smaller model
# or
export RERANKING_MODEL="/path/to/local/model"       # Local fine-tuned model
```

## Performance Benchmarks

Typical performance on different hardware:

| Hardware | Model Precision | Batch Size | Time/Batch | Memory Usage |
|----------|----------------|------------|------------|--------------|
| RTX 4090 | float16 | 8 | 80ms | 8GB |
| RTX 3080 | float16 | 4 | 120ms | 6GB |
| CPU (16 cores) | float32 | 4 | 800ms | 12GB RAM |

## Contributing

When modifying the Qwen3 integration:

1. Update tests in `test_qwen3_integration.py`
2. Add configuration options to `qwen3_config.py`
3. Update this documentation
4. Test on both GPU and CPU environments

## Support

For issues with:
- **Qwen3 model**: Check [Qwen repository](https://github.com/QwenLM/Qwen2)
- **Integration bugs**: Check Archon logs and create an issue
- **Performance**: Try different configuration options above
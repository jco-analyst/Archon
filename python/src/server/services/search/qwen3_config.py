"""
Qwen3 Reranker Configuration

This module provides configuration constants and helper functions for the Qwen3 reranker.
"""

import os
from typing import Dict, Any, Optional
import torch

# Default configuration values
DEFAULT_QWEN3_MODEL = "Qwen/Qwen3-Reranker-4B"
DEFAULT_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
DEFAULT_MAX_LENGTH = 8192
DEFAULT_USE_FLASH_ATTENTION = False

# Environment variable names
ENV_QWEN3_MODEL = "QWEN3_RERANKER_MODEL"
ENV_QWEN3_INSTRUCTION = "QWEN3_INSTRUCTION"
ENV_QWEN3_MAX_LENGTH = "QWEN3_MAX_LENGTH"
ENV_QWEN3_USE_FLASH_ATTENTION = "QWEN3_USE_FLASH_ATTENTION"
ENV_QWEN3_TORCH_DTYPE = "QWEN3_TORCH_DTYPE"
ENV_QWEN3_DEVICE = "QWEN3_DEVICE"


def get_qwen3_config() -> Dict[str, Any]:
    """
    Get Qwen3 configuration from environment variables with sensible defaults.
    
    Environment Variables:
    - QWEN3_RERANKER_MODEL: Model name/path (default: Qwen/Qwen3-Reranker-4B)
    - QWEN3_INSTRUCTION: Custom instruction for reranking
    - QWEN3_MAX_LENGTH: Maximum token length (default: 8192)
    - QWEN3_USE_FLASH_ATTENTION: Use flash attention (default: false)
    - QWEN3_TORCH_DTYPE: PyTorch data type (e.g., float16, bfloat16)
    - QWEN3_DEVICE: Device to use (cuda, cpu, auto)
    
    Returns:
        Configuration dictionary for Qwen3Reranker
    """
    config = {
        "model_name": os.getenv(ENV_QWEN3_MODEL, DEFAULT_QWEN3_MODEL),
        "instruction": os.getenv(ENV_QWEN3_INSTRUCTION, DEFAULT_INSTRUCTION),
        "max_length": int(os.getenv(ENV_QWEN3_MAX_LENGTH, str(DEFAULT_MAX_LENGTH))),
        "use_flash_attention": os.getenv(ENV_QWEN3_USE_FLASH_ATTENTION, "false").lower() == "true",
    }
    
    # Handle torch dtype
    torch_dtype_str = os.getenv(ENV_QWEN3_TORCH_DTYPE)
    if torch_dtype_str:
        try:
            config["torch_dtype"] = getattr(torch, torch_dtype_str)
        except AttributeError:
            config["torch_dtype"] = None
    else:
        config["torch_dtype"] = None
    
    # Handle device
    device = os.getenv(ENV_QWEN3_DEVICE)
    if device == "auto":
        config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    elif device:
        config["device"] = device
    else:
        config["device"] = None  # Let Qwen3Reranker decide
    
    return config


def get_recommended_config_for_hardware() -> Dict[str, Any]:
    """
    Get recommended Qwen3 configuration based on available hardware.
    
    Returns:
        Recommended configuration dictionary
    """
    config = get_qwen3_config()
    
    # Adjust based on available hardware
    if torch.cuda.is_available():
        # GPU available - use optimized settings
        if not config.get("torch_dtype"):
            config["torch_dtype"] = torch.float16  # Faster on GPU
        if not config.get("device"):
            config["device"] = "cuda"
        
        # Enable flash attention if supported (requires compatible GPU)
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else ""
        if "A100" in gpu_name or "H100" in gpu_name or "RTX" in gpu_name:
            config["use_flash_attention"] = True
    else:
        # CPU only - conservative settings
        if not config.get("torch_dtype"):
            config["torch_dtype"] = torch.float32  # More stable on CPU
        config["device"] = "cpu"
        config["use_flash_attention"] = False  # Not supported on CPU
    
    return config


# Example configurations for different use cases
PERFORMANCE_CONFIG = {
    "model_name": DEFAULT_QWEN3_MODEL,
    "torch_dtype": torch.float16,
    "use_flash_attention": True,
    "device": "cuda",
    "max_length": 8192,
}

MEMORY_EFFICIENT_CONFIG = {
    "model_name": DEFAULT_QWEN3_MODEL,
    "torch_dtype": torch.float16,
    "use_flash_attention": True,
    "device": "cuda",
    "max_length": 4096,  # Reduced for memory efficiency
}

CPU_CONFIG = {
    "model_name": DEFAULT_QWEN3_MODEL,
    "torch_dtype": torch.float32,
    "use_flash_attention": False,
    "device": "cpu",
    "max_length": 8192,
}
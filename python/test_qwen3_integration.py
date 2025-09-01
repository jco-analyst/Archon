#!/usr/bin/env python3
"""
Test script for Qwen3 Reranker integration with Archon.

This script tests the Qwen3 reranker integration without requiring the full Archon setup.
It can be run independently to verify that the Qwen3 reranker works correctly.

Usage:
    python test_qwen3_integration.py

Note: This requires the Qwen3 model to be downloaded, which may take some time on first run.
"""

import sys
import os
from pathlib import Path

# Add the src directory to Python path for imports
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

from server.services.search.qwen3_reranker import Qwen3Reranker, create_qwen3_reranker
from server.services.search.reranking_strategy import RerankingStrategy
from server.services.search.qwen3_config import get_qwen3_config, get_recommended_config_for_hardware


def test_qwen3_reranker_basic():
    """Test basic Qwen3Reranker functionality."""
    print("🧪 Testing Qwen3Reranker basic functionality...")
    
    try:
        # Create a basic Qwen3 reranker (this will try to download the model)
        print("📥 Loading Qwen3 model (this may take a while on first run)...")
        reranker = create_qwen3_reranker()
        
        if not reranker.is_available():
            print("❌ Qwen3 reranker is not available")
            return False
        
        print("✅ Qwen3 reranker loaded successfully")
        
        # Test basic prediction
        query = "What is machine learning?"
        documents = [
            "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
            "Cooking is the art of preparing food using heat and various techniques.",
            "Deep learning uses neural networks with multiple layers to learn patterns.",
        ]
        
        query_doc_pairs = [[query, doc] for doc in documents]
        
        print("🔍 Testing reranking prediction...")
        scores = reranker.predict(query_doc_pairs)
        
        print(f"📊 Reranking scores: {scores}")
        
        # Verify scores are reasonable
        if len(scores) != len(documents):
            print(f"❌ Expected {len(documents)} scores, got {len(scores)}")
            return False
        
        # Check that scores are between 0 and 1
        for i, score in enumerate(scores):
            if not (0 <= score <= 1):
                print(f"❌ Score {i} is out of range [0,1]: {score}")
                return False
        
        # The first and third documents should score higher than cooking
        if scores[1] >= max(scores[0], scores[2]):
            print("⚠️  Warning: Cooking document scored higher than ML documents")
        
        print("✅ Basic Qwen3 reranker test passed")
        return True
        
    except Exception as e:
        print(f"❌ Qwen3 reranker test failed: {e}")
        return False


def test_reranking_strategy_integration():
    """Test Qwen3 integration with RerankingStrategy."""
    print("\n🧪 Testing RerankingStrategy integration...")
    
    try:
        # Test with Qwen3 model name
        print("🔄 Creating RerankingStrategy with Qwen3 model...")
        strategy = RerankingStrategy(model_name="Qwen/Qwen3-Reranker-4B")
        
        if not strategy.is_available():
            print("❌ RerankingStrategy with Qwen3 is not available")
            return False
        
        print("✅ RerankingStrategy with Qwen3 created successfully")
        
        # Test reranking
        query = "Python programming"
        results = [
            {"content": "Python is a high-level programming language", "id": 1},
            {"content": "Snakes are reptiles that slither on the ground", "id": 2},
            {"content": "Django is a Python web framework", "id": 3},
        ]
        
        print("🔍 Testing result reranking...")
        reranked = await strategy.rerank_results(query, results)
        
        print(f"📊 Reranked results: {[r.get('rerank_score', 'N/A') for r in reranked]}")
        
        # Verify rerank scores were added
        for result in reranked:
            if 'rerank_score' not in result:
                print("❌ Rerank score not added to results")
                return False
        
        # Check that programming-related content scores higher than snake content
        programming_scores = [r['rerank_score'] for r in reranked if r['id'] in [1, 3]]
        snake_score = next(r['rerank_score'] for r in reranked if r['id'] == 2)
        
        if max(programming_scores) <= snake_score:
            print("⚠️  Warning: Snake content scored higher than programming content")
        
        print("✅ RerankingStrategy integration test passed")
        return True
        
    except Exception as e:
        print(f"❌ RerankingStrategy integration test failed: {e}")
        return False


def test_configuration():
    """Test Qwen3 configuration system."""
    print("\n🧪 Testing Qwen3 configuration...")
    
    try:
        # Test basic config
        config = get_qwen3_config()
        print(f"📋 Basic config: {config}")
        
        # Test recommended config
        recommended = get_recommended_config_for_hardware()
        print(f"🎯 Recommended config: {recommended}")
        
        # Test environment variable override
        os.environ["QWEN3_MAX_LENGTH"] = "4096"
        os.environ["QWEN3_USE_FLASH_ATTENTION"] = "true"
        
        env_config = get_qwen3_config()
        
        if env_config["max_length"] != 4096:
            print(f"❌ Environment variable override failed: expected 4096, got {env_config['max_length']}")
            return False
        
        if not env_config["use_flash_attention"]:
            print("❌ Environment variable override failed: flash attention should be True")
            return False
        
        # Clean up
        del os.environ["QWEN3_MAX_LENGTH"]
        del os.environ["QWEN3_USE_FLASH_ATTENTION"]
        
        print("✅ Configuration test passed")
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


async def main():
    """Run all tests."""
    print("🚀 Starting Qwen3 Reranker Integration Tests\n")
    
    tests = [
        ("Configuration", test_configuration),
        ("Basic Qwen3Reranker", test_qwen3_reranker_basic),
        ("RerankingStrategy Integration", test_reranking_strategy_integration),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print(f"{'='*50}")
        
        try:
            if await test_func() if hasattr(test_func, '__call__') and 'async' in str(test_func) else test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
    
    print(f"\n{'='*50}")
    print(f"🏁 Test Results: {passed}/{total} tests passed")
    print(f"{'='*50}")
    
    if passed == total:
        print("🎉 All tests passed! Qwen3 integration is working correctly.")
        return True
    else:
        print("💥 Some tests failed. Check the output above for details.")
        return False


if __name__ == "__main__":
    import asyncio
    
    print("📋 System Information:")
    print(f"Python: {sys.version}")
    print(f"Working Directory: {os.getcwd()}")
    
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("PyTorch: Not installed")
    
    try:
        import transformers
        print(f"Transformers: {transformers.__version__}")
    except ImportError:
        print("Transformers: Not installed")
    
    print()
    
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
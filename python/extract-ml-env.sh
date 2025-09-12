#!/bin/bash
# Extract ML environment to mounted volume

set -e

echo "🔄 Extracting ML environment to host storage..."

# Copy entire .local directory (contains all pip --user packages)
echo "📦 Copying Python ML packages..."
cp -r /root/.local/* /output/

echo "🧠 Pre-downloading Qwen3-Reranker-0.6B model..."
export TRANSFORMERS_CACHE=/models
export HF_HOME=/models
python -c "
from sentence_transformers import CrossEncoder
print('Downloading Qwen3-Reranker-0.6B...')
model = CrossEncoder('Qwen/Qwen3-Reranker-0.6B')
print('Model download complete!')
"

echo "✅ ML environment extraction complete!"
echo "📊 Extracted sizes:"
du -sh /output
du -sh /models
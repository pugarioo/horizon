#!/bin/bash

# Create 'models' directory if it doesn't exist
mkdir -p models
cd models

echo "==================================================="
echo "   DOWNLOADING THESIS MODEL SUITE (4GB VRAM)     "
echo "==================================================="

# 1. THE JUDGE: DeepSeek R1 Distill 1.5B (Q8_0)
# Size: ~1.89 GB
echo ""
echo "[1/3] Downloading DeepSeek R1 1.5B (Q8_0)..."
wget -O deepseek_engine.gguf \
  "https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf"

# 2. THE PROVER: Qwen 2.5 3B Instruct (Q4_K_M)
# Size: ~1.93 GB
echo ""
echo "[2/3] Downloading Qwen 2.5 3B (Q4_K_M)..."
wget -O qwen_engine.gguf \
  "https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF/resolve/main/Qwen2.5-3B-Instruct-Q4_K_M.gguf"

# 3. THE CRITIC: Llama 3.2 3B Instruct (Q4_K_M)
# Size: ~2.02 GB
echo ""
echo "[3/3] Downloading Llama 3.2 3B (Q4_K_M)..."
wget -O llama_engine.gguf \
  "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf"

echo ""
echo "==================================================="
echo "   DOWNLOAD COMPLETE. FILES READY IN ./models/     "
echo "==================================================="
ls -lh
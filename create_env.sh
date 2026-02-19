#!/bin/bash

# ==========================================
# Script: create_env.sh
# Purpose: Creates 'horizon_env' and force-compiles llama-cpp-python with CUDA
# Usage: bash creat_env.sh
# ==========================================

# Initialize Conda for this script session
eval "$(conda shell.bash hook)"

ENV_NAME="horizon_env"

echo "🚀 [1/3] Creating Environment & Installing Compilers..."
# Your cleaner one-liner (added 'pip' just to be safe)
conda create -n $ENV_NAME \
    python=3.10 \
    pip \
    gxx_linux-64=12 \
    gcc_linux-64=12 \
    sysroot_linux-64=2.17 \
    -c conda-forge -y

echo "🔌 [2/3] Activating Environment..."
conda activate $ENV_NAME

echo "⚡ [3/3] Force-Compiling Llama-CPP with CUDA..."

# --- THE "NUCLEAR OPTION" COMPILE BLOCK ---

# 1. Point to System CUDA (The Bridge)
export CUDACXX=/usr/local/cuda/bin/nvcc

# 2. Point to the "Time Capsule" Compilers (The Override)
# We use the compilers we just installed in Step 1
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++

# 3. Force the Compile
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python \
    --upgrade \
    --force-reinstall \
    --no-cache-dir

echo "✅ DONE. Environment '$ENV_NAME' is ready."

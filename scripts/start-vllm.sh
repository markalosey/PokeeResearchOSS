#!/bin/bash
# Copyright 2025 Pokee AI Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Startup script for vLLM server
# Configures and starts vLLM with appropriate settings for T4 GPUs

set -e

# Default values (can be overridden via environment variables)
MODEL=${MODEL:-PokeeAI/pokee_research_7b}
PORT=${PORT:-9999}
QUANTIZATION=${QUANTIZATION:-none}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.60}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-2048}
HF_TOKEN=${HF_TOKEN:-}

# Display configuration
echo "=========================================="
echo "vLLM Server Configuration"
echo "=========================================="
echo "Model: $MODEL"
echo "Port: $PORT"
echo "Quantization: $QUANTIZATION"
echo "GPU Memory Utilization: $GPU_MEMORY_UTILIZATION"
echo "Max Model Length: $MAX_MODEL_LEN"
echo "Tensor Parallel Size: 2 (using both GPUs)"
echo "=========================================="

# Check GPU availability
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. GPU access may not be available."
    exit 1
fi

echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader

# Build vLLM command arguments
VLLM_ARGS=(
    "serve" "$MODEL"
    "--port" "$PORT"
    "--dtype" "auto"
    "--max-model-len" "$MAX_MODEL_LEN"
    "--gpu-memory-utilization" "$GPU_MEMORY_UTILIZATION"
    "--host" "0.0.0.0"
    "--enforce-eager"
    "--tensor-parallel-size" "2"
)

# Add quantization only if specified and not empty
# Note: Only specify quantization if the model repository actually contains quantized weights/config
if [ -n "$QUANTIZATION" ] && [ "$QUANTIZATION" != "none" ]; then
    VLLM_ARGS+=("--quantization" "$QUANTIZATION")
fi

# Note: HuggingFace token is passed via HF_TOKEN environment variable
# vLLM will automatically use it if set - no need to pass via command line

# Start vLLM server
echo "Starting vLLM server..."
exec vllm "${VLLM_ARGS[@]}"


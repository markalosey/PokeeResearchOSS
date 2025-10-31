# Memory Constraints and Solutions

## Problem Analysis

The PokeeResearch-7B model is too large for a single T4 GPU (15.36 GB) when attempting to use reasonable context windows:

- **Model size**: ~14.3 GB (fixed)
- **Available memory**: ~1 GB free (after model load)
- **KV cache requirement**: Scales with context length
  - 4k tokens: ~0.4-0.5 GB
  - 6k tokens: ~0.6-0.8 GB  
  - 8k tokens: ~0.8-1.2 GB
  - 16k tokens: ~1.6-2.4 GB

**Current Status**: Even with 6144 context, we're hitting OOM during profiling.

## Solutions Implemented

### 1. Memory Optimizations (Just Applied)
- Reduced `GPU_MEMORY_UTILIZATION` from 0.75 → 0.65
- Added `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- Reduced `MAX_MODEL_LEN` from 8192 → 6144

**Try this first** - rebuild and restart:
```bash
docker compose build vllm-server
docker compose down
docker compose up -d
```

### 2. If Still OOM: Reduce to 4096 Context

If 6144 still fails, reduce to 4096 (still 2x improvement):

```bash
# In .env file:
MAX_MODEL_LEN=4096
GPU_MEMORY_UTILIZATION=0.60
```

Then restart:
```bash
docker compose down
docker compose up -d
```

### 3. Alternative: Use SimpleDeepResearchAgent

If vLLM continues to fail, switch to transformers-based agent:

```python
# In gradio_app.py or CLI, use:
SimpleDeepResearchAgent(model_path="PokeeAI/pokee_research_7b")
```

This uses transformers directly, which may handle memory differently:
- More flexible memory management
- Can use gradient checkpointing
- Slower but might work where vLLM fails

**Trade-off**: Much slower inference, but might complete research tasks.

### 4. Long-term Solutions

#### Option A: Quantization (Best if available)
- Use AWQ or GPTQ quantized weights
- Reduces model from ~14GB to ~4-7GB
- Frees up 7-10GB for KV cache
- Could support 8k-16k context easily

**Check if quantized weights exist**:
```bash
# Check HuggingFace
huggingface-cli scan-cache
# Or search for: PokeeAI/pokee-research-7b-awq
```

If quantized weights don't exist, creating them requires:
- Significant compute/time
- Quantization tools (AutoAWQ, AutoGPTQ)
- Model-specific calibration data

#### Option B: Accept 4096 Context Limit
- Still 2x improvement over original 2048
- With better truncation logic, might be sufficient
- Focus on optimizing truncation to preserve most important information

#### Option C: Hardware Upgrade
- More GPU memory (H100, A100)
- Or multiple GPUs with better memory management

## Recommended Next Steps

1. **Try the current optimizations** (6144 context with memory optimizations)
2. **If fails, reduce to 4096** context
3. **If still fails, switch to SimpleDeepResearchAgent** (transformers)
4. **If that works but is too slow**, investigate quantization

## Current Configuration

- **Context**: 6144 tokens (try 4096 if OOM)
- **GPU Memory Utilization**: 0.65
- **Input Tokens**: ~5176 (with 4096 context: ~3128)
- **Generation Tokens**: 768

This is still a **2-3x improvement** over the original 2048 context limit, giving the agent significantly more room for conversation history.


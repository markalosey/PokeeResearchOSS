# Context Limit Configuration Guide

## Overview

The PokeeResearch-7B model supports up to **32,768 tokens** of context, but we currently limit it to **4,096 tokens** (2x improvement from original 2048) to balance memory usage and performance on T4 GPUs with tensor parallelism.

## Current Configuration

- **Default MAX_MODEL_LEN**: 4096 tokens (2x improvement from original 2048)
- **GPU Memory Utilization**: 0.60 (60%)
- **Tensor Parallelism**: 2 GPUs
- **Generation Tokens**: 768
- **Available Input Tokens**: ~3128 (4096 - 768 - 200 buffer)
- **KV Cache Memory Available**: ~0.17 GiB per GPU (with tensor parallelism)

## How to Increase Context Limit

### Option 1: Via Environment Variable (Recommended)

Set `MAX_MODEL_LEN` in your `.env` file:

```bash
# For 6k context (test if memory allows)
MAX_MODEL_LEN=6144
GPU_MEMORY_UTILIZATION=0.65

# For 8k context (may require GPU_MEMORY_UTILIZATION=0.70)
MAX_MODEL_LEN=8192
GPU_MEMORY_UTILIZATION=0.70
```

**Note**: With tensor parallelism on T4 GPUs, each GPU only has ~0.17 GiB KV cache available. 8k tokens requires ~0.22 GiB, so you'll need to increase `GPU_MEMORY_UTILIZATION` to free up more memory.

Then restart the vLLM server:

```bash
cd /datapool/PokeeResearchOSS
docker compose down
docker compose up -d
```

### Option 2: Gradual Increase Strategy

If you want to try increasing beyond 4096, increase gradually:

1. **Current: 4096** (safe, works with GPU_MEMORY_UTILIZATION=0.60):
   ```bash
   MAX_MODEL_LEN=4096
   GPU_MEMORY_UTILIZATION=0.60
   ```

2. **Try 6144** (may require higher GPU_MEMORY_UTILIZATION):
   ```bash
   MAX_MODEL_LEN=6144
   GPU_MEMORY_UTILIZATION=0.65
   ```

3. **Try 8192** (will likely require GPU_MEMORY_UTILIZATION=0.70):
   ```bash
   MAX_MODEL_LEN=8192
   GPU_MEMORY_UTILIZATION=0.70
   ```

**Warning**: Higher context limits may cause OOM errors. Monitor GPU memory usage carefully.

### Option 3: Reduce GPU Memory Utilization

If you hit OOM at higher context limits, reduce `GPU_MEMORY_UTILIZATION`:

```bash
# Reduce from 0.60 to 0.50 to free up memory for KV cache
GPU_MEMORY_UTILIZATION=0.50
MAX_MODEL_LEN=16384
```

## Memory Requirements

Context length affects KV (Key-Value) cache memory. **With tensor parallelism on 2x T4 GPUs**, each GPU has limited KV cache memory:

- **4k tokens**: ~0.15 GiB per GPU ✅ (current, safe)
- **6k tokens**: ~0.20 GiB per GPU ⚠️ (may work if you increase GPU_MEMORY_UTILIZATION)
- **8k tokens**: ~0.22 GiB per GPU ❌ (requires more than available 0.17 GiB)

**Important**: With tensor parallelism, the model is split across both GPUs, so each GPU only gets a portion of the total memory for KV cache.

**To increase context limit beyond 4k**, you need to:
1. Increase `GPU_MEMORY_UTILIZATION` (e.g., 0.65-0.70)
2. Or use a single GPU (remove tensor parallelism) - but this may cause OOM
3. Or use quantization (AWQ/GPTQ) to reduce model memory usage

## Monitoring

After increasing, monitor:

1. **GPU Memory Usage**:
   ```bash
   watch -n 1 nvidia-smi
   ```

2. **vLLM Logs**:
   ```bash
   docker logs -f pokee-vllm
   ```

3. **Look for OOM errors**:
   - `CUDA out of memory`
   - `No available memory for the cache blocks`

## Troubleshooting

### OOM Errors

If you see "CUDA out of memory" or "No available memory for the cache blocks":

1. **Reduce context limit**:
   ```bash
   MAX_MODEL_LEN=4096  # or lower
   ```

2. **Reduce GPU memory utilization**:
   ```bash
   GPU_MEMORY_UTILIZATION=0.50  # or lower
   ```

3. **Check for other GPU processes**:
   ```bash
   nvidia-smi
   ```

### Agent Still Truncating

The agent automatically adapts to the context limit. If you see truncation warnings:

- Check that `MAX_MODEL_LEN` environment variable is set correctly
- Verify the vLLM server restarted with the new limit
- Check logs: `docker logs pokee-vllm | grep "Max Model Length"`

## Benefits of Larger Context

- **More conversation history**: Agent can see more previous searches/results
- **Less truncation**: Fewer messages need to be dropped
- **Better continuity**: Agent maintains context across more turns
- **Longer research**: Can handle more complex multi-turn research tasks

## Trade-offs

- **Memory usage**: Higher context = more GPU memory
- **Latency**: Slightly slower inference with larger context
- **Throughput**: Fewer concurrent requests possible

## Recommended Settings

For **Dell R720 with 2x T4 GPUs** (with tensor parallelism):

- **Conservative**: `MAX_MODEL_LEN=4096`, `GPU_MEMORY_UTILIZATION=0.60` ✅ (current default, stable)
- **Moderate**: `MAX_MODEL_LEN=6144`, `GPU_MEMORY_UTILIZATION=0.65` ⚠️ (test carefully)
- **Aggressive**: `MAX_MODEL_LEN=8192`, `GPU_MEMORY_UTILIZATION=0.70` ⚠️ (may cause OOM)

**Note**: Due to tensor parallelism constraints, each GPU only has ~0.17 GiB KV cache available. Higher context limits require increasing `GPU_MEMORY_UTILIZATION` to free up more memory, which may cause OOM if you go too high.

Start with conservative (4096), then increase gradually if needed.


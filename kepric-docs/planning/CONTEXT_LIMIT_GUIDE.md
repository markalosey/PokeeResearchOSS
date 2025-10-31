# Context Limit Configuration Guide

## Overview

The PokeeResearch-7B model supports up to **32,768 tokens** of context, but we currently limit it to **8,192 tokens** (4x increase) to balance memory usage and performance on T4 GPUs.

## Current Configuration

- **Default MAX_MODEL_LEN**: 8192 tokens
- **GPU Memory Utilization**: 0.60 (60%)
- **Tensor Parallelism**: 2 GPUs
- **Generation Tokens**: 1024
- **Available Input Tokens**: ~6970 (8192 - 1024 - 200 buffer)

## How to Increase Context Limit

### Option 1: Via Environment Variable (Recommended)

Set `MAX_MODEL_LEN` in your `.env` file:

```bash
# For 16k context (2x increase)
MAX_MODEL_LEN=16384

# For 32k context (4x increase, maximum)
MAX_MODEL_LEN=32768
```

Then restart the vLLM server:

```bash
cd /datapool/PokeeResearchOSS
docker compose down
docker compose up -d
```

### Option 2: Gradual Increase Strategy

If you hit OOM (Out of Memory) errors, increase gradually:

1. **Start with 4096** (2x current):
   ```bash
   MAX_MODEL_LEN=4096
   ```

2. **If stable, try 8192** (current default):
   ```bash
   MAX_MODEL_LEN=8192
   ```

3. **If stable, try 16384** (2x):
   ```bash
   MAX_MODEL_LEN=16384
   ```

4. **Maximum: 32768** (if memory allows):
   ```bash
   MAX_MODEL_LEN=32768
   ```

### Option 3: Reduce GPU Memory Utilization

If you hit OOM at higher context limits, reduce `GPU_MEMORY_UTILIZATION`:

```bash
# Reduce from 0.60 to 0.50 to free up memory for KV cache
GPU_MEMORY_UTILIZATION=0.50
MAX_MODEL_LEN=16384
```

## Memory Requirements

Context length affects KV (Key-Value) cache memory:

- **8k tokens**: ~2-3 GB per GPU
- **16k tokens**: ~4-6 GB per GPU
- **32k tokens**: ~8-12 GB per GPU

With 2x T4 GPUs (15GB each):
- **8k**: Safe ✅
- **16k**: Should work ✅
- **32k**: May require reduced GPU_MEMORY_UTILIZATION ⚠️

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

For **Dell R720 with 2x T4 GPUs**:

- **Conservative**: `MAX_MODEL_LEN=8192` (current default)
- **Balanced**: `MAX_MODEL_LEN=16384` (recommended)
- **Maximum**: `MAX_MODEL_LEN=32768` (if memory allows)

Start with conservative, then increase if needed.


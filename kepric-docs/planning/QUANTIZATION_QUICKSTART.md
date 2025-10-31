# Quantization Implementation - Quick Start

## ✅ Immediate Solution: BitsAndBytes 4-bit Quantization

**Status**: Implemented and enabled by default!

### What Changed

1. **SimpleDeepResearchAgent** now supports 4-bit quantization via bitsandbytes
   - Reduces model from ~14GB to ~4-5GB
   - Frees up ~9-10GB for KV cache
   - Supports much larger context windows (up to 32k tokens)

2. **Enabled by default** in both CLI and Gradio apps
   - Automatically uses quantization when using `--serving-mode local`

### How to Use (Immediate)

**CLI:**
```bash
cd /datapool/PokeeResearchOSS
source .venv/bin/activate

# Install bitsandbytes if not already installed
pip install bitsandbytes

# Run with quantization (enabled by default)
python cli_app.py --serving-mode local
```

**Gradio:**
```bash
# Quantization is enabled by default
python gradio_app.py --serving-mode local
```

### Expected Results

- **Model size**: ~4-5GB (down from ~14GB)
- **Available memory**: ~10-11GB for KV cache
- **Context window**: Up to 32k tokens supported
- **Speed**: Slower than vLLM, but much faster than hitting OOM errors

### If You See Import Errors

If you get `ImportError: bitsandbytes`, install it:
```bash
pip install bitsandbytes
```

---

## 🔄 Long-term Solution: AWQ Quantization for vLLM

For best performance with vLLM, create AWQ quantized weights:

### Step 1: Create AWQ Weights

```bash
cd /datapool/PokeeResearchOSS
source .venv/bin/activate

# Install AutoAWQ
pip install autoawq

# Create quantized weights (takes 1-4 hours)
python scripts/quantize_awq.py \
    --model PokeeAI/pokee_research_7b \
    --output ./quantized/pokee-research-7b-awq \
    --bits 4
```

### Step 2: Use AWQ with vLLM

```bash
# In .env file:
MODEL=./quantized/pokee-research-7b-awq
QUANTIZATION=awq
MAX_MODEL_LEN=16384  # Can now use much larger context!
GPU_MEMORY_UTILIZATION=0.75

# Restart vLLM
docker compose down
docker compose up -d
```

### Benefits of AWQ

- **Best performance**: Fastest inference with vLLM
- **Large context**: Supports 16k-32k tokens easily
- **Lower memory**: Model ~4-5GB, ~10GB free for KV cache
- **Production ready**: Optimized for serving

---

## Comparison

| Method | Model Size | KV Cache | Max Context | Speed | Status |
|--------|------------|----------|-------------|-------|--------|
| **Original (FP16)** | ~14.3 GB | ~1 GB | 4k-6k | Fast | ❌ OOM |
| **BitsAndBytes 4-bit** | ~4-5 GB | ~10-11 GB | 16k-32k | Medium | ✅ **Ready Now** |
| **AWQ 4-bit** | ~4-5 GB | ~10-11 GB | 16k-32k | Fast | 🔄 Create weights |

---

## Next Steps

1. **Try BitsAndBytes quantization now** (immediate solution)
   ```bash
   # Just run with local mode - quantization is enabled by default
   python gradio_app.py --serving-mode local
   ```

2. **If that works**, you can:
   - Use it immediately for research
   - Create AWQ weights in background for better performance later

3. **If you want AWQ weights**, run the quantization script (1-4 hours)

The BitsAndBytes solution should work immediately and give you the large context windows you need!


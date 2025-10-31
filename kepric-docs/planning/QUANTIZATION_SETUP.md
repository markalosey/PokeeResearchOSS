# Quantization Setup Guide

## Problem

The PokeeResearch-7B model (~14.3 GB) is too large for a single T4 GPU when trying to use reasonable context windows. Quantization can reduce model size by 2-4x, freeing up memory for larger context windows.

## Quantization Options

### Option 1: AWQ Quantization (Recommended for vLLM)

**What it is**: AWQ (Activation-aware Weight Quantization) reduces model to ~4-5 GB while maintaining high quality.

**Benefits**:
- Reduces model from ~14GB to ~4-5GB
- Frees up ~9-10GB for KV cache
- Can support 16k-32k context easily
- Fast inference with vLLM

**Requirements**:
- Pre-quantized AWQ weights (need to create if they don't exist)
- vLLM supports AWQ natively

**Setup**:
1. Create AWQ weights (see scripts below)
2. Set `QUANTIZATION=awq` in `.env`
3. Update `MODEL` to point to quantized weights

### Option 2: GPTQ Quantization

**What it is**: GPTQ (Generalized Post-Training Quantization) alternative to AWQ.

**Similar benefits** to AWQ, but different quantization method.

**Setup**: Same as AWQ, but use `QUANTIZATION=gptq`

### Option 3: BitsAndBytes 4-bit (Fallback)

**What it is**: 4-bit quantization that works with any model without pre-quantization.

**Benefits**:
- Works immediately (no pre-quantization needed)
- Reduces model to ~4-5 GB
- Works with transformers library

**Trade-offs**:
- Requires using SimpleDeepResearchAgent (not vLLM)
- Slower than AWQ/GPTQ
- May have slightly lower quality

**Setup**: Use transformers with bitsandbytes (see SimpleDeepResearchAgent modifications)

## Step-by-Step: Create AWQ Weights

### Prerequisites

```bash
# Install AutoAWQ
pip install autoawq

# Or in Dockerfile, add:
RUN pip install autoawq
```

### Create AWQ Weights

```bash
# Activate virtual environment
source .venv/bin/activate

# Run quantization script (see scripts/quantize-awq.sh)
python scripts/quantize_awq.py \
    --model PokeeAI/pokee_research_7b \
    --output ./quantized/pokee-research-7b-awq \
    --bits 4 \
    --zero_point True \
    --calib_dataset wikitext
```

This will:
- Load the original model
- Quantize weights to 4-bit AWQ
- Save quantized weights to output directory
- Take 1-4 hours depending on hardware

### Use AWQ Weights with vLLM

```bash
# In .env file:
MODEL=./quantized/pokee-research-7b-awq
QUANTIZATION=awq
MAX_MODEL_LEN=16384  # Can now use much larger context!
GPU_MEMORY_UTILIZATION=0.75
```

## Step-by-Step: Use BitsAndBytes (Immediate Solution)

If AWQ weights don't exist and you need immediate quantization:

### Modify SimpleDeepResearchAgent

Add bitsandbytes quantization support to `agent/simple_agent.py`:

```python
from transformers import BitsAndBytesConfig

# In _load_model method:
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=quantization_config,
    device_map=device,
    torch_dtype=torch.float16,
)
```

Then use `SimpleDeepResearchAgent` instead of `VLLMDeepResearchAgent`.

## Recommended Approach

1. **Short-term**: Modify SimpleDeepResearchAgent to use bitsandbytes 4-bit quantization
   - Works immediately
   - Reduces model to ~4-5 GB
   - Can support larger context windows
   - Trade-off: Slower than vLLM

2. **Long-term**: Create AWQ weights and use vLLM
   - Best performance
   - Requires time to create weights
   - Better for production

## Memory Comparison

| Method | Model Size | Available for KV Cache | Max Context |
|--------|------------|----------------------|-------------|
| Original (FP16) | ~14.3 GB | ~1 GB | 4k-6k tokens |
| AWQ 4-bit | ~4-5 GB | ~10-11 GB | 16k-32k tokens |
| GPTQ 4-bit | ~4-5 GB | ~10-11 GB | 16k-32k tokens |
| BitsAndBytes 4-bit | ~4-5 GB | ~10-11 GB | 16k-32k tokens |

## Next Steps

1. Try bitsandbytes quantization first (immediate solution)
2. Create AWQ weights in background (long-term solution)
3. Switch to AWQ when ready for best performance


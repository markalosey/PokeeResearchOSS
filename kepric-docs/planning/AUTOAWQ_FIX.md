# AutoAWQ Compatibility Fix

## Problem
AutoAWQ is incompatible with Transformers 4.57.1 (requires 4.51.3).

## Solution Options

### Option 1: Downgrade Transformers (Quick Fix)

```bash
cd /datapool/PokeeResearchOSS
source .venv/bin/activate

# Downgrade transformers to compatible version
pip install transformers==4.51.3

# Then run quantization
python scripts/quantize_awq.py \
    --model_path PokeeAI/pokee_research_7b \
    --quantized_model_path ./quantized/pokee-research-7b-awq \
    --bits 4 \
    --group_size 128
```

**Note**: This may break other parts of the codebase that need newer transformers.

### Option 2: Use BitsAndBytes Instead (Recommended)

BitsAndBytes quantization is already implemented and works with current transformers:

```bash
# No quantization script needed - just use SimpleDeepResearchAgent
# It's already enabled by default!

python gradio_app.py --serving-mode local
# or
python cli_app.py --serving-mode local --question "Your question"
```

**Benefits**:
- ✅ Already working
- ✅ Compatible with current transformers
- ✅ No pre-quantization needed
- ✅ Same memory savings (~4-5GB model)

**Trade-off**: Slower than vLLM, but faster than OOM errors!

### Option 3: Use vLLM's llm-compressor (Future)

vLLM has adopted AWQ quantization. We'd need to:
1. Install `llm-compressor` from vLLM
2. Update the quantization script to use their API

This is more complex but would give best performance.

## Recommendation

**For immediate use**: Use Option 2 (BitsAndBytes) - it's already implemented and working!

**For best performance later**: Consider Option 3 (vLLM llm-compressor) or Option 1 if you can isolate the quantization environment.


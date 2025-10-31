# Docker Compose Deployment Steps - Quantization

## Current Situation

- ✅ Docker Compose is configured for vLLM + Tool Server
- ✅ vLLM currently running with `MAX_MODEL_LEN=6144` (limited context)
- ⚠️ Need AWQ quantization to enable larger context windows

## Option 1: Use Current Setup (Works Now, Limited Context)

**Status**: Already deployed, just start services

```bash
cd /datapool/PokeeResearchOSS

# Pull latest changes
git pull origin migration/tavily-playwright-gpt5

# Ensure .env file has API keys
cat .env  # Verify TAVILY_API_KEY and OPENAI_API_KEY are set

# Start Docker Compose services
docker compose up -d

# Check status
docker compose ps
docker compose logs -f vllm-server  # Monitor vLLM startup
docker compose logs -f tool-server  # Monitor tool server

# Verify services are healthy
curl http://localhost:9999/health  # vLLM
curl http://localhost:8888/health  # Tool Server
```

**Limitations**:
- Context window: ~6144 tokens (may still truncate long research)
- Model size: ~14.3 GB (uses most of GPU memory)

---

## Option 2: Create AWQ Weights, Then Use with Docker Compose (Best Performance)

**This enables 16k-32k context windows!**

### Step 1: Create AWQ Quantized Weights (On Server, Takes 1-4 Hours)

```bash
cd /datapool/PokeeResearchOSS

# Activate virtual environment
source .venv/bin/activate

# Install AutoAWQ if not already installed
pip install autoawq

# Create output directory
mkdir -p ./quantized

# Run quantization (this will take 1-4 hours)
# Uses GPU 1 (so GPU 0 is free for other tasks)
CUDA_VISIBLE_DEVICES=1 python scripts/quantize_awq.py \
    --model_path PokeeAI/pokee_research_7b \
    --quantized_model_path ./quantized/pokee-research-7b-awq \
    --bits 4 \
    --group_size 128

# Wait for completion (check progress in terminal)
# Expected output: "Quantized model saved successfully to ./quantized/pokee-research-7b-awq"
```

### Step 2: Update .env File

```bash
# Edit .env file
nano .env

# Update these lines:
MODEL=./quantized/pokee-research-7b-awq
QUANTIZATION=awq
MAX_MODEL_LEN=16384  # Can now use 16k+ tokens!
GPU_MEMORY_UTILIZATION=0.75
```

### Step 3: Restart Docker Compose Services

```bash
cd /datapool/PokeeResearchOSS

# Rebuild vLLM server (to pick up new config)
docker compose build vllm-server

# Restart services
docker compose down
docker compose up -d

# Monitor logs
docker compose logs -f vllm-server
```

**Expected Results**:
- ✅ Model size: ~4-5 GB (down from ~14.3 GB)
- ✅ Available memory: ~10-11 GB for KV cache
- ✅ Context window: 16k-32k tokens (no more truncation!)
- ✅ Performance: Fast inference with vLLM

---

## Option 3: Quick Test - Use Gradio Locally, Connect to Docker Services

If you want to test before committing to AWQ quantization:

```bash
cd /datapool/PokeeResearchOSS

# Ensure Docker Compose services are running
docker compose ps  # Should show vllm-server and tool-server UP

# Activate virtual environment
source .venv/bin/activate

# Run Gradio app (connects to Docker services)
python gradio_app.py \
    --serving-mode vllm \
    --vllm-url http://localhost:9999/v1 \
    --port 7777 \
    --server-name 0.0.0.0

# Access UI at http://<server-ip>:7777
```

---

## Recommended Workflow

**Immediate (Now)**:
1. Use Option 1 - verify everything works with current setup
2. Run a test research query to see if 6144 context is sufficient

**If truncation is still an issue**:
1. Start Option 2 (AWQ quantization) in background
2. Use Option 1 while quantization runs
3. Switch to Option 2 after quantization completes

**Long-term**:
- Use Option 2 with AWQ quantization for best performance

---

## Verification Commands

```bash
# Check Docker services
docker compose ps

# Check GPU usage
nvidia-smi

# Check vLLM health
curl http://localhost:9999/health

# Check tool server health
curl http://localhost:8888/health

# View logs
docker compose logs vllm-server --tail 50
docker compose logs tool-server --tail 50
```

---

## Troubleshooting

**If vLLM won't start**:
- Check GPU memory: `nvidia-smi`
- Check logs: `docker compose logs vllm-server`
- Reduce `MAX_MODEL_LEN` if OOM errors

**If tool server won't start**:
- Check API keys in `.env`
- Check logs: `docker compose logs tool-server`
- Verify port 8888 is not in use: `netstat -tuln | grep 8888`

**If quantization fails**:
- Ensure GPU 1 is available: `CUDA_VISIBLE_DEVICES=1 nvidia-smi`
- Check available disk space: `df -h`
- Verify AutoAWQ is installed: `pip show autoawq`


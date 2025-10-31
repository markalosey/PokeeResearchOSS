# Deployment Plan Quick Start Summary

**Purpose:** Quick reference for the comprehensive deployment plan  
**Full Plan:** See `deployment_plan.md` for detailed hierarchical checklists

---

## Quick Overview

This deployment migrates PokeeResearch from:

- **Serper** → **Tavily** (web search)
- **Jina** → **Playwright** (web reading)
- **Gemini** → **GPT-5** (summarization)

Deployment method: **Docker Compose** on Dell R720 with 2x NVIDIA T4 GPUs

---

## Critical Path (Minimum Steps)

### 1. Server Setup (30 minutes)

```bash
# Verify GPUs
nvidia-smi

# Install Docker & NVIDIA Container Toolkit
curl -fsSL https://get.docker.com -o get-docker.sh && sudo sh get-docker.sh
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 2. Clone Repository (5 minutes)

```bash
cd /opt/pokee-research
git clone https://github.com/YOUR_USERNAME/PokeeResearchOSS.git
cd PokeeResearchOSS
git checkout migration/tavily-playwright-gpt5
```

### 3. Code Modifications (2-3 hours)

- Modify `tool_server/search.py` (Serper → Tavily)
- Modify `tool_server/read.py` (Jina → Playwright)
- Modify `tool_server/utils.py` (Gemini → GPT-5)
- Update `requirements.txt`

### 4. Docker Setup (1 hour)

- Create `Dockerfile.vllm`
- Create `Dockerfile.tool-server`
- Create `docker-compose.yml`
- Create `.env` file

### 5. Deploy (30 minutes)

```bash
# Configure environment
cp .env.example .env
# Edit .env with actual API keys

# Build and start
docker compose build
docker compose up -d

# Verify
docker compose ps
docker compose logs -f
```

---

## Key Files to Modify

| File                     | Changes                                                                         |
| ------------------------ | ------------------------------------------------------------------------------- |
| `tool_server/search.py`  | Replace `serper_search()` with `tavily_search()`, update `WebSearchAgent`       |
| `tool_server/read.py`    | Replace `jina_read()` with `playwright_read()`, update `WebReadAgent`           |
| `tool_server/utils.py`   | Replace `get_genai_client()` with `get_openai_client()`, update `llm_summary()` |
| `requirements.txt`       | Add `tavily-python`, `playwright`, `openai`, remove `google-genai`              |
| `start_tool_server.py`   | Update environment variable references (if any)                                 |
| `docker-compose.yml`     | Create complete Docker Compose configuration                                    |
| `Dockerfile.vllm`        | Create vLLM server Dockerfile                                                   |
| `Dockerfile.tool-server` | Create tool server Dockerfile                                                   |

---

## Environment Variables Required

```bash
# Required
TAVILY_API_KEY=your_tavily_key
OPENAI_API_KEY=your_openai_key
HUGGINGFACE_TOKEN=your_hf_token

# Optional
OPENAI_MODEL=gpt-5-pro  # Default: gpt-5-pro
VLLM_URL=http://localhost:9999/v1
TOOL_SERVER_PORT=8888
```

---

## Validation Commands

### Pre-Deployment

```bash
# Test Tavily API
python3 -c "from tavily import TavilyClient; client = TavilyClient(api_key='KEY'); print(client.search('test'))"

# Test Playwright
python3 -c "from playwright.async_api import async_playwright; print('OK')"

# Test OpenAI/GPT-5
python3 -c "from openai import AsyncOpenAI; import asyncio; client = AsyncOpenAI(api_key='KEY'); print([m.id for m in asyncio.run(client.models.list()).data if 'gpt-5' in m.id])"
```

### Post-Deployment

```bash
# Test tool-server search
curl -X POST http://localhost:8888/search \
  -H "Content-Type: application/json" \
  -d '{"query": "test"}'

# Test tool-server read
curl -X POST http://localhost:8888/read \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com", "question": "test"}'

# Check GPU usage
nvidia-smi

# Check container status
docker compose ps
```

---

## Troubleshooting Quick Reference

| Issue              | Solution                                                                              |
| ------------------ | ------------------------------------------------------------------------------------- |
| GPU not accessible | `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker` |
| Build fails        | Check disk space, verify network connectivity                                         |
| API calls fail     | Verify API keys in `.env`, test APIs directly                                         |
| vLLM won't start   | Check GPU memory, verify model name, check HF token                                   |

---

## Estimated Timeline

| Phase              | Duration      | Priority |
| ------------------ | ------------- | -------- |
| Server Setup       | 30 min        | Critical |
| Code Modifications | 2-3 hours     | Critical |
| Docker Setup       | 1 hour        | Critical |
| Testing            | 1-2 hours     | Critical |
| Deployment         | 30 min        | Critical |
| **Total**          | **5-7 hours** |          |

---

## Phase Completion Checklist

- [ ] Phase 1: Pre-Deployment Preparation (✅ Complete)
- [ ] Phase 2: Code Modifications (⏳ In Progress)
- [ ] Phase 3: Docker Setup (⏳ Pending)
- [ ] Phase 4: Validation & Testing (⏳ Pending)
- [ ] Phase 5: Deployment (⏳ Pending)

---

**For detailed step-by-step instructions, see:** `deployment_plan.md`

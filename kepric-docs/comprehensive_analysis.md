# Comprehensive Repository Analysis: PokeeResearch Deep Research Agent

**Date:** 2025-01-27  
**Target Hardware:** Dell R720 with 2x NVIDIA T4 GPUs  
**Host OS:** Debian  
**Model Serving:** vLLM (Recommended) or Transformers (Ollama requires manual model conversion)

---

## Executive Summary

This repository contains Pokee's 7B Deep Research Agent, a sophisticated multi-turn research system that uses web search and content reading capabilities to answer complex questions. The system currently uses **Serper** (web search), **Jina** (web content reading), and **Gemini** (content summarization).

**Required Modifications:** The system needs to be adapted to use **Tavily** (web search), **GPT-5** (content summarization), and **Playwright** (web content reading) instead.

**⚠️ CRITICAL CORRECTIONS:**

- **Ollama does NOT host pokee-research-7b** - Verified via Ollama model registry search. The model `PokeeAI/pokee_research_7b` is only available on HuggingFace. Use vLLM (recommended) or Transformers (SimpleDeepResearchAgent) instead - both already supported in codebase.

- **GPT-5 IS available** - Verified via OpenAI API documentation (platform.openai.com/docs/models). Released in 2025, available via OpenAI API with these model options:

  - `gpt-5` - Standard model, recommended for most tasks, balanced performance/cost
  - `gpt-5-pro` - Highest precision variant, recommended for summarization tasks
  - `gpt-5-mini` - Cost-efficient version for well-defined tasks
  - `gpt-5-nano` - Fastest, most cost-efficient version
  - `gpt-5-codex` - Optimized for agentic coding in Codex
  - **Note:** `gpt-5-chat-latest` exists but is NOT recommended for API use (ChatGPT-only)

- **Model Serving:** Use vLLM with quantization (AWQ/GPTQ) for T4 GPUs - already supported via `VLLMDeepResearchAgent`

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Current Technology Stack](#current-technology-stack)
3. [Component Analysis](#component-analysis)
4. [Required Modifications](#required-modifications)
5. [Hardware Considerations](#hardware-considerations)
6. [Deployment Strategy](#deployment-strategy)
7. [Configuration Changes](#configuration-changes)
8. [Migration Path](#migration-path)

---

## Architecture Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface Layer                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  CLI App     │  │  Gradio App  │  │  Main.py    │      │
│  │  (cli_app.py)│  │(gradio_app.py)│  │(main.py)    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Agent Layer                               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │        BaseDeepResearchAgent (base_agent.py)         │  │
│  │  - Tool management                                    │  │
│  │  - Response parsing                                   │  │
│  │  - Research loop (research ↔ verification modes)     │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────┐              ┌──────────────┐            │
│  │ SimpleAgent  │              │ VLLMAgent   │            │
│  │ (local model)│              │ (via HTTP)  │            │
│  └──────────────┘              └──────────────┘            │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Tool Client Layer                        │
│  ┌──────────────┐              ┌──────────────┐            │
│  │ WebSearchTool│              │ WebReadTool  │            │
│  │ (pokee_tools)│              │ (pokee_tools)│            │
│  └──────────────┘              └──────────────┘            │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Tool Server Layer                        │
│  ┌──────────────┐              ┌──────────────┐            │
│  │ WebSearchAgent│             │ WebReadAgent │            │
│  │ (search.py)  │              │ (read.py)    │            │
│  └──────────────┘              └──────────────┘            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              CacheManager (cache_manager.py)         │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    External Services                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │  Serper  │  │   Jina   │  │  Gemini  │  │  [New]   │   │
│  │   API   │  │   API    │  │   API    │  │ Tavily   │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Core Components

1. **Agent Layer** (`agent/`)

   - `base_agent.py`: Abstract base class with research loop logic
   - `simple_agent.py`: Local model inference (transformers)
   - `vllm_agent.py`: Remote model inference via HTTP

2. **Tool Client** (`tool_client/`)

   - `pokee_tools.py`: WebSearchTool and WebReadTool implementations
   - Uses Ray for distributed execution
   - Communicates with tool server via HTTP

3. **Tool Server** (`tool_server/`)

   - `search.py`: WebSearchAgent using Serper API
   - `read.py`: WebReadAgent using Jina API + Gemini summarization
   - `start_tool_server.py`: FastAPI server entry point

4. **Configuration** (`config/`)
   - `pokee_multiturn_grpo.yaml`: Main training/evaluation config
   - `tool_config/pokee_tool_config.yaml`: Tool definitions

---

## Current Technology Stack

### External APIs (Current)

| Service    | Purpose               | API Key Env Var  | Status    |
| ---------- | --------------------- | ---------------- | --------- |
| **Serper** | Web search            | `SERPER_API_KEY` | ✅ Active |
| **Jina**   | Web content reading   | `JINA_API_KEY`   | ✅ Active |
| **Gemini** | Content summarization | `GEMINI_API_KEY` | ✅ Active |

### External APIs (Required)

| Service        | Purpose               | API Key Env Var  | Status          |
| -------------- | --------------------- | ---------------- | --------------- |
| **Tavily**     | Web search            | `TAVILY_API_KEY` | ❌ To Implement |
| **GPT-5**      | Content summarization | `OPENAI_API_KEY` | ❌ To Implement |
| **Playwright** | Web content reading   | N/A (local)      | ❌ To Implement |

### Model Serving

- **Current:** Supports local transformers model or VLLM server
- **Target:** vLLM server with quantization (recommended) or transformers (Ollama requires manual model conversion)
- **Model:** `PokeeAI/pokee_research_7b` (7B parameters)

### Dependencies

Key dependencies identified from code analysis:

- `transformers` - Model loading and inference
- `torch` - PyTorch backend
- `ray` - Distributed execution
- `fastapi` + `uvicorn` - Tool server
- `aiohttp` - Async HTTP client
- `google-genai` - Gemini API client
- `pydantic` - Data validation
- `gradio` - Web UI (optional)

---

## Component Analysis

### 1. Tool Server (`tool_server/`)

#### `search.py` - WebSearchAgent

**Current Implementation:**

- Uses Serper API (`https://google.serper.dev/search`)
- Returns `SearchResult` with `url_items` (URL, title, description)
- Async implementation with semaphore-based rate limiting

**Key Functions:**

```python
async def serper_search(query: str, timeout: int = 30, top_k: int = 10) -> SearchResult
class WebSearchAgent:
    async def search(self, query: str) -> SearchResult
```

**Required Changes:**

- Replace Serper API calls with Tavily API
- Update response parsing to match Tavily's response format
- Maintain same `SearchResult` structure for compatibility

#### `read.py` - WebReadAgent

**Current Implementation:**

- Uses Jina Reader API (`https://r.jina.ai/{url}`)
- Extracts content and links from webpage
- Uses Gemini API for summarization (`gemini-2.5-flash-lite`)
- Fallback to truncated content if LLM summarization fails

**Key Functions:**

```python
async def jina_read(url: str, timeout: int = 30) -> ReadResult
async def llm_summary(user_prompt: str, client: genai.Client) -> LLMSummaryResult
class WebReadAgent:
    async def read(self, question: str, url: str) -> ReadResult
```

**Required Changes:**

- Replace Jina API with Playwright for web scraping
- Replace Gemini API with GPT-5 (OpenAI API) for summarization
- Implement Playwright browser automation
- Handle JavaScript-rendered content
- Maintain same `ReadResult` structure

#### `utils.py`

**Current Implementation:**

- `get_genai_client()`: Creates Gemini client
- `llm_summary()`: Summarizes content using Gemini
- Error handling and retry logic

**Required Changes:**

- Replace Gemini client with OpenAI client
- Update `llm_summary()` to use GPT-5 model
- Adjust error handling for OpenAI API patterns

### 2. Tool Client (`tool_client/pokee_tools.py`)

**Current Implementation:**

- `WebSearchTool`: Wraps tool server search endpoint
- `WebReadTool`: Wraps tool server read endpoint
- Uses Ray for distributed execution
- Rate limiting via TokenBucketWorker

**Required Changes:**

- Minimal changes needed (works with tool server API)
- May need to adjust timeouts for Playwright operations

### 3. Agent Layer (`agent/`)

**Current Implementation:**

- `base_agent.py`: Core research loop logic
- Two modes: "research" and "verification"
- Tool call parsing and execution
- Streaming support for real-time updates

**Required Changes:**

- No changes needed (tool abstraction handles differences)

### 4. Model Serving

**Current Options:**

1. **Local:** `SimpleDeepResearchAgent` loads model via transformers
2. **VLLM:** `VLLMDeepResearchAgent` uses HTTP API

**⚠️ CORRECTED: Ollama Model Availability**

**CRITICAL ISSUE:** Ollama does NOT host pokee-research-7b in its model library.

**Recommended Approach:** Use existing vLLM support (already in codebase)

- ✅ `VLLMDeepResearchAgent` already exists
- ✅ Better GPU utilization than Ollama
- ✅ Supports quantization for T4 GPUs
- ✅ No model conversion needed

**Alternative:** Use `SimpleDeepResearchAgent` with transformers

- ✅ Already works with T4 GPUs
- ✅ Supports quantization (8-bit/4-bit)
- ✅ No additional setup needed

**If Ollama Required:** Manual model conversion needed (NOT RECOMMENDED)

**Why Not Recommended:**

- Ollama does NOT host pokee-research-7b in its model library (verified via ollama.com/models)
- Requires manual conversion process:
  1. Convert HuggingFace model to GGUF format using llama.cpp
  2. Quantize model for T4 GPUs (q4_K_M recommended)
  3. Create custom Modelfile
  4. Import into Ollama manually
- Time-consuming process (several hours)
- Requires significant disk space (~30GB+)
- May lose model fidelity during conversion
- vLLM is simpler and already integrated

**Conclusion:** Use vLLM instead - it's already supported, requires no conversion, and offers better GPU utilization.

---

## Required Modifications

### 1. Replace Serper with Tavily

**File:** `tool_server/search.py`

**Tavily API Details (Verified):**

- **API Endpoint:** `https://api.tavily.com/search`
- **Request Method:** POST
- **Content-Type:** application/json
- **Authentication:** API key in request body (`api_key` field)
- **Python SDK:** Available via `tavily-python` package

**Changes Needed:**

```python
# OLD: Serper API
url = "https://google.serper.dev/search"
payload = {"q": query, "location": "United States", "num": top_k}
headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}

# NEW: Tavily API
from tavily import TavilyClient

client = TavilyClient(api_key=api_key)
response = client.search(
    query=query,
    search_depth="basic",  # or "advanced" for deeper search
    max_results=top_k
)

# Response structure:
# {
#   "answer": "Direct answer string",
#   "results": [
#     {
#       "url": "...",
#       "title": "...",
#       "content": "...",
#       "score": 0.95
#     }
#   ],
#   "query": "...",
#   "response_time": 0.5
# }
```

**Alternative: Direct HTTP Request**

```python
import httpx

url = "https://api.tavily.com/search"
payload = {
    "api_key": api_key,
    "query": query,
    "search_depth": "basic",  # or "advanced"
    "max_results": top_k
}
headers = {"Content-Type": "application/json"}

async with httpx.AsyncClient() as client:
    response = await client.post(url, json=payload, headers=headers)
    data = response.json()
```

**Response Format Differences:**

- **Serper:** `{"organic": [{"link": "...", "title": "...", "snippet": "..."}]}`
- **Tavily:** `{"results": [{"url": "...", "title": "...", "content": "...", "score": 0.95}]}`
  - Tavily also provides `answer` field with direct answer
  - Tavily includes `score` for relevance ranking

**Action Items:**

- [ ] Install `tavily-python` package
- [ ] Update `serper_search()` → `tavily_search()`
- [ ] Update response parsing in `_extract_organic_from_serper_response()`
- [ ] Update environment variable: `SERPER_API_KEY` → `TAVILY_API_KEY`

### 2. Replace Jina with Playwright

**File:** `tool_server/read.py`

**Playwright Details (Verified):**

- **Package:** `playwright` (Python)
- **Installation:** `pip install playwright && playwright install chromium`
- **Browser:** Chromium (headless mode recommended)
- **Capabilities:** Full browser automation, JavaScript rendering, DOM access
- **Advantages over Jina:** Self-hosted (no API costs), better for JavaScript-heavy sites, more control

**Changes Needed:**

```python
# OLD: Jina API
reader_url = f"https://r.jina.ai/{url}"
async with session.get(reader_url, headers=headers) as response:
    data = await response.json()
    content = data.get("data", {}).get("text", "")

# NEW: Playwright
from playwright.async_api import async_playwright

async def playwright_read(url: str, timeout: int = 30000) -> dict:
    """Read webpage content using Playwright."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=['--disable-gpu', '--disable-dev-shm-usage']
        )
        context = await browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (compatible; PokeeResearch/1.0)'
        )
        page = await context.new_page()

        # Block unnecessary resources for faster loading
        await page.route("**/*.{png,jpg,jpeg,gif,svg,css,woff,woff2}",
                        lambda route: route.abort())

        try:
            await page.goto(url, wait_until="networkidle", timeout=timeout)

            # Extract main content
            text_content = await page.evaluate("""
                () => {
                    // Remove script and style elements
                    const scripts = document.querySelectorAll('script, style');
                    scripts.forEach(el => el.remove());

                    // Get main content
                    const main = document.querySelector('main, article, [role="main"]')
                                || document.body;
                    return main.innerText;
                }
            """)

            # Extract links
            links = await page.evaluate("""
                () => {
                    return Array.from(document.querySelectorAll('a'))
                        .map(a => ({
                            url: a.href,
                            title: a.textContent.trim(),
                            text: a.innerText.trim()
                        }))
                        .filter(link => link.url && link.url.startsWith('http'));
                }
            """)

            await browser.close()

            return {
                "text": text_content,
                "links": links,
                "url": url
            }

        except Exception as e:
            await browser.close()
            raise
```

**Additional Considerations:**

- **Browser Installation:** `playwright install chromium` (required once)
- **System Dependencies:** On Debian: `sudo apt install chromium-browser chromium-driver libnss3 libatk-bridge2.0-0`
- **Timeout Handling:** Default 30s, configurable per request
- **Resource Blocking:** Block images/stylesheets for faster loading
- **Error Handling:** Handle network errors, timeouts, and invalid URLs
- **Memory Management:** Close browser instances properly to avoid memory leaks

**Action Items:**

- [ ] Install `playwright` package
- [ ] Replace `jina_read()` with `playwright_read()`
- [ ] Implement link extraction from DOM
- [ ] Add browser lifecycle management
- [ ] Handle errors and timeouts

### 3. Replace Gemini with GPT-5

**File:** `tool_server/utils.py`

**GPT-5 Availability (Verified via OpenAI API Documentation):**

GPT-5 has been released and is available via OpenAI API. The following models are available for API use:

- **`gpt-5`** - Standard GPT-5 model, recommended for most tasks, balanced performance/cost
- **`gpt-5-pro`** - Premium variant with highest precision, **RECOMMENDED for summarization tasks**
- **`gpt-5-mini`** - Cost-efficient version for well-defined tasks
- **`gpt-5-nano`** - Fastest, most cost-efficient version
- **`gpt-5-codex`** - Optimized for agentic coding in Codex

**Important:** `gpt-5-chat-latest` exists but is marked as "not recommended for API use" - it's the ChatGPT-only version. Use `gpt-5` or `gpt-5-pro` for API integrations.

**Model Selection Recommendation:**

- **For summarization:** Use `gpt-5-pro` for highest quality (recommended)
- **For general use:** Use `gpt-5` for balanced performance/cost
- **For cost-sensitive deployments:** Use `gpt-5-mini` or `gpt-5-nano`

**Changes Needed:**

```python
# OLD: Gemini API
from google import genai
client = genai.Client(api_key=api_key)
response = await client.aio.models.generate_content(
    model="gemini-2.5-flash-lite",
    contents=user_prompt,
    config=GenerateContentConfig(...)
)

# NEW: OpenAI API (GPT-5 - Available now)
from openai import AsyncOpenAI
client = AsyncOpenAI(api_key=api_key)
response = await client.chat.completions.create(
    model="gpt-5-pro",  # Recommended: "gpt-5-pro" for summaries, or "gpt-5" for balanced performance
    messages=[
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": user_prompt}
    ],
    max_tokens=2048,
    temperature=0.1
)
text = response.choices[0].message.content
```

**Error Handling:**

- OpenAI uses different error codes than Gemini
- Rate limiting: 429 status code
- Token limits: 400 status code with "context_length_exceeded"
- Retry logic may need adjustment

**Action Items:**

- [ ] Install `openai` package (latest version)
- [ ] Replace `get_genai_client()` with `get_openai_client()`
- [ ] Update `llm_summary()` to use GPT-5 model (`gpt-5-pro` recommended for summaries, or `gpt-5` for balanced performance)
- [ ] Update error handling patterns
- [ ] Update environment variable: `GEMINI_API_KEY` → `OPENAI_API_KEY`
- [ ] Test with different GPT-5 variants to determine optimal choice

### 4. Model Serving: Use vLLM (NOT Ollama)

**⚠️ CORRECTION:** Ollama does not host pokee-research-7b. Use existing vLLM support instead.

**Recommended: Use Existing vLLM Support**

The codebase already includes `VLLMDeepResearchAgent`. Simply start vLLM server:

```bash
# Start vLLM server with quantization for T4 GPUs
vllm serve PokeeAI/pokee_research_7b \
  --port 9999 \
  --dtype auto \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.45 \
  --quantization awq  # or gptq
```

**Alternative: Manual Ollama Import (NOT RECOMMENDED)**

**Why Not Recommended:**
Ollama does NOT host pokee-research-7b in its model library. Verification shows the model is only available on HuggingFace at `PokeeAI/pokee_research_7b`.

**Manual Conversion Process (if required):**

1. Install llama.cpp conversion tools
2. Convert HuggingFace model to GGUF format
3. Quantize model for T4 GPUs (q4_K_M recommended, reduces to ~4GB per GPU)
4. Create custom Modelfile
5. Import into Ollama manually

**Time Estimate:** Several hours + significant disk space (~30GB+)

**Recommendation:** Use vLLM instead - it's already integrated, requires no conversion, and offers better GPU utilization with quantization support.

**New File:** `agent/ollama_agent.py` (Only if Ollama import completed)

**Implementation:**

```python
import httpx
from agent.base_agent import BaseDeepResearchAgent

class OllamaDeepResearchAgent(BaseDeepResearchAgent):
    def __init__(
        self,
        ollama_url: str = "http://localhost:11434/v1",
        model_name: str = "pokee-research-7b",
        tool_config_path: str = "config/tool_config/pokee_tool_config.yaml",
        max_turns: int = 10,
    ):
        super().__init__(tool_config_path, max_turns)
        self.ollama_url = ollama_url
        self.model_name = model_name
        self.client = httpx.AsyncClient(timeout=300.0)

    async def generate(self, messages: list[dict], temperature: float = 0.7, top_p: float = 0.9) -> str:
        # Ollama OpenAI-compatible API
        response = await self.client.post(
            f"{self.ollama_url}/chat/completions",
            json={
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "tools": self.tool_schemas,  # Function calling support
            }
        )
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
```

**vLLM Setup (RECOMMENDED):**

- Install vLLM: `pip install vllm`
- Start server with quantization: See command above
- Use existing `VLLMDeepResearchAgent` - no code changes needed!

**Action Items:**

- [ ] Install vLLM with quantization support
- [ ] Configure vLLM server for T4 GPUs
- [ ] Test existing `VLLMDeepResearchAgent` integration
- [ ] Benchmark performance

**Ollama Setup (ADVANCED - NOT RECOMMENDED):**

**Why Not Recommended:**

- Ollama does NOT host pokee-research-7b (verified via ollama.com/models)
- Requires manual conversion process:
  1. Convert HuggingFace model to GGUF using llama.cpp
  2. Quantize model for T4 GPUs (q4_K_M recommended)
  3. Create Modelfile with template and parameters
  4. Import into Ollama using `ollama create pokee-research-7b -f Modelfile`
- Time-consuming (several hours)
- Requires significant disk space (~30GB+)
- May lose model fidelity during conversion

**Recommendation:** Use vLLM instead. It's already supported via `VLLMDeepResearchAgent`, requires no conversion, supports quantization (AWQ/GPTQ), and offers better GPU memory management for T4 GPUs.

---

## Hardware Considerations

### Dell R720 Specifications

- **CPU:** Dual Intel Xeon E5-2600 series (typically 12-16 cores total)
- **RAM:** Variable (likely 64-256GB)
- **GPUs:** 2x NVIDIA T4 (16GB VRAM each)
- **Storage:** Variable (SSD recommended for model loading)

### NVIDIA T4 GPU Capabilities

**T4 Specifications:**

- **VRAM:** 16GB GDDR6
- **Compute Capability:** 7.5 (Turing architecture)
- **Tensor Cores:** Yes (mixed precision support)
- **Power:** 70W TDP

**Model Loading Considerations:**

- **Model Size:** 7B parameters ≈ 14GB in FP16/BF16, 28GB in FP32
- **Single T4:** Can load model in quantized format (8-bit or 4-bit)
- **Dual T4:** Can split model across GPUs or run two instances

### Ollama Configuration

**Recommended Settings:**

```yaml
# ollama_config.yaml or environment variables
num_gpu: 2 # Use both T4 GPUs
num_thread: 8 # CPU threads per GPU
numa: true # NUMA awareness
gpu_memory_fraction: 0.9 # Use 90% of VRAM
```

**Model Quantization:**

- Use `q4_K_M` (4-bit quantization) for T4 GPUs
- Reduces memory usage to ~4GB per GPU
- Minimal quality loss

**Docker Configuration:**

```dockerfile
# Dockerfile for Ollama
FROM ollama/ollama:latest

# NVIDIA GPU support
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Expose OpenAI-compatible API port
EXPOSE 11434
```

**Docker Compose:**

```yaml
version: "3.8"
services:
  ollama:
    image: ollama/ollama:latest
    container_name: pokee-ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 2
              capabilities: [gpu]
    ports:
      - "11434:11434"
    volumes:
      - ollama-data:/root/.ollama
    environment:
      - OLLAMA_ORIGINS=*
      - OLLAMA_NUM_PARALLEL=2
      - OLLAMA_MAX_LOADED_MODELS=1
volumes:
  ollama-data:
```

### Performance Expectations

**Single T4 (Quantized Model):**

- Token generation: ~20-40 tokens/second
- Batch size: 1-4 concurrent requests
- Memory usage: ~8-12GB VRAM

**Dual T4 (Load Balanced):**

- Can run 2 independent instances
- Or use model parallelism (requires modification)
- Total throughput: ~40-80 tokens/second

**Bottlenecks:**

- GPU memory bandwidth (T4 has 300 GB/s)
- Model loading time (first request)
- Network latency (tool server calls)

---

## Deployment Strategy

### Option 1: Docker Compose (Recommended)

**Structure:**

```
pokee-research/
├── docker-compose.yml
├── Dockerfile.tool-server
├── Dockerfile.agent
├── .env
└── config/
```

**docker-compose.yml:**

```yaml
version: "3.8"

services:
  ollama:
    image: ollama/ollama:latest
    container_name: pokee-ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 2
              capabilities: [gpu]
    ports:
      - "11434:11434"
    volumes:
      - ollama-data:/root/.ollama
    environment:
      - OLLAMA_ORIGINS=*
      - OLLAMA_NUM_PARALLEL=2
    restart: unless-stopped

  tool-server:
    build:
      context: .
      dockerfile: Dockerfile.tool-server
    container_name: pokee-tool-server
    ports:
      - "8888:8888"
    environment:
      - TAVILY_API_KEY=${TAVILY_API_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - OLLAMA_URL=http://ollama:11434/v1
    depends_on:
      - ollama
    restart: unless-stopped

  agent:
    build:
      context: .
      dockerfile: Dockerfile.agent
    container_name: pokee-agent
    ports:
      - "7777:7777" # Gradio UI
    environment:
      - OLLAMA_URL=http://ollama:11434/v1
      - TOOL_SERVER_URL=http://tool-server:8888
    depends_on:
      - ollama
      - tool-server
    restart: unless-stopped

volumes:
  ollama-data:
```

### Option 2: Native Installation

**Debian Setup:**

```bash
# 1. Install NVIDIA drivers and CUDA
sudo apt update
sudo apt install -y nvidia-driver-535 nvidia-cuda-toolkit

# 2. Install Docker and NVIDIA Container Toolkit
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker

# 3. Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 4. Pull model
ollama pull pokee-research-7b:q4_K_M

# 5. Start Ollama
OLLAMA_ORIGINS=* ollama serve &

# 6. Install Python dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 7. Install Playwright browsers
playwright install chromium

# 8. Start tool server
python start_tool_server.py --port 8888 --enable-cache &

# 9. Start agent (CLI or Gradio)
python cli_app.py --serving-mode ollama --ollama-url http://localhost:11434/v1
```

---

## Configuration Changes

### 1. Environment Variables

**Current (.env):**

```bash
SERPER_API_KEY=your_serper_api_key_here
JINA_API_KEY=your_jina_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```

**Required (.env):**

```bash
TAVILY_API_KEY=your_tavily_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
OLLAMA_URL=http://localhost:11434/v1
OLLAMA_MODEL=pokee-research-7b:q4_K_M
```

### 2. Tool Configuration

**File:** `config/tool_config/pokee_tool_config.yaml`

**No changes needed** - Tool schemas remain the same, only backend implementations change.

### 3. Agent Configuration

**File:** `cli_app.py` and `gradio_app.py`

**Add Ollama support:**

```python
parser.add_argument(
    "--serving-mode",
    type=str,
    choices=["local", "vllm", "ollama"],  # Add "ollama"
    default="ollama",  # Change default
)

parser.add_argument(
    "--ollama-url",
    type=str,
    default="http://localhost:11434/v1",
    help="Ollama server URL",
)
```

### 4. Tool Server Configuration

**File:** `start_tool_server.py`

**Update API key checks:**

```python
# OLD
required_keys = ["SERPER_API_KEY", "JINA_API_KEY", "GEMINI_API_KEY"]

# NEW
required_keys = ["TAVILY_API_KEY", "OPENAI_API_KEY"]
```

---

## Migration Path

### Phase 1: API Replacement (Week 1)

1. **Day 1-2: Tavily Integration**

   - [ ] Install Tavily SDK
   - [ ] Update `tool_server/search.py`
   - [ ] Test search functionality
   - [ ] Update documentation

2. **Day 3-4: Playwright Integration**

   - [ ] Install Playwright
   - [ ] Update `tool_server/read.py`
   - [ ] Implement browser automation
   - [ ] Test with various websites
   - [ ] Handle JavaScript-heavy sites

3. **Day 5: GPT-5 Integration**
   - [ ] Install OpenAI SDK
   - [ ] Update `tool_server/utils.py`
   - [ ] Test summarization
   - [ ] Verify error handling

### Phase 2: Ollama Integration (Week 2)

1. **Day 1-2: Ollama Setup**

   - [ ] Install Ollama on Debian server
   - [ ] Configure Docker container
   - [ ] Import model (quantized)
   - [ ] Test OpenAI-compatible API

2. **Day 3-4: Agent Implementation**

   - [ ] Create `agent/ollama_agent.py`
   - [ ] Update CLI and Gradio apps
   - [ ] Test end-to-end flow
   - [ ] Performance benchmarking

3. **Day 5: Optimization**
   - [ ] GPU memory optimization
   - [ ] Batch processing
   - [ ] Caching improvements
   - [ ] Load testing

### Phase 3: Testing & Validation (Week 3)

1. **Day 1-3: Functional Testing**

   - [ ] Test all tool functions
   - [ ] Test agent research loop
   - [ ] Test error handling
   - [ ] Test concurrent requests

2. **Day 4-5: Performance Testing**
   - [ ] Measure response times
   - [ ] Test GPU utilization
   - [ ] Test memory usage
   - [ ] Optimize bottlenecks

### Phase 4: Production Deployment (Week 4)

1. **Day 1-2: Production Setup**

   - [ ] Deploy Docker containers
   - [ ] Configure reverse proxy (optional)
   - [ ] Set up monitoring
   - [ ] Configure logging

2. **Day 3-4: Documentation**

   - [ ] Update README
   - [ ] Create deployment guide
   - [ ] Document API changes
   - [ ] Create troubleshooting guide

3. **Day 5: Final Validation**
   - [ ] End-to-end testing
   - [ ] User acceptance testing
   - [ ] Performance validation
   - [ ] Go-live

---

## Risk Assessment

### Technical Risks

| Risk                          | Impact | Probability | Mitigation                                 |
| ----------------------------- | ------ | ----------- | ------------------------------------------ |
| GPT-5 not available           | High   | Medium      | Use GPT-4-turbo as fallback                |
| Playwright performance issues | Medium | High        | Implement caching, optimize browser launch |
| T4 GPU memory constraints     | High   | Medium      | Use quantization, model sharding           |
| Tavily API rate limits        | Medium | Low         | Implement caching, request batching        |
| Ollama compatibility issues   | Medium | Low         | Test thoroughly, have VLLM fallback        |

### Operational Risks

| Risk                        | Impact | Probability | Mitigation                       |
| --------------------------- | ------ | ----------- | -------------------------------- |
| Docker container failures   | Medium | Low         | Health checks, auto-restart      |
| Network connectivity issues | Medium | Low         | Retry logic, fallback mechanisms |
| Model loading delays        | Low    | Medium      | Pre-warm model, keep-alive       |

---

## Testing Strategy

### Unit Tests

```python
# tests/test_tavily_search.py
def test_tavily_search():
    result = await tavily_search("test query")
    assert result.success == True
    assert len(result.url_items) > 0

# tests/test_playwright_read.py
async def test_playwright_read():
    result = await playwright_read("https://example.com", "test question")
    assert result.success == True
    assert len(result.content) > 0

# tests/test_gpt5_summary.py
async def test_gpt5_summary():
    result = await llm_summary("test content", client)
    assert result.success == True
    assert len(result.text) > 0
```

### Integration Tests

```python
# tests/test_end_to_end.py
async def test_research_flow():
    agent = OllamaDeepResearchAgent(...)
    answer = await agent.run("What is quantum computing?")
    assert answer is not None
    assert len(answer) > 0
```

### Performance Tests

```python
# tests/test_performance.py
async def test_concurrent_requests():
    # Test 10 concurrent requests
    results = await asyncio.gather(*[
        agent.run(f"Question {i}") for i in range(10)
    ])
    assert all(r is not None for r in results)
```

---

## Monitoring & Observability

### Metrics to Track

1. **Tool Server Metrics**

   - Search request latency (Tavily)
   - Read request latency (Playwright)
   - Summary generation latency (GPT-5)
   - Cache hit rate
   - Error rates by type

2. **Agent Metrics**

   - Average turns per query
   - Tool call success rate
   - End-to-end latency
   - Answer quality (manual review)

3. **Ollama Metrics**
   - GPU utilization
   - Memory usage
   - Token generation rate
   - Request queue length

### Logging

- Use structured logging (JSON format)
- Log all API calls (with rate limiting info)
- Log tool execution times
- Log errors with full context

### Health Checks

```python
# Health check endpoints
GET /health  # Overall health
GET /health/ollama  # Ollama connection
GET /health/tavily  # Tavily API
GET /health/openai  # OpenAI API
```

---

## Cost Analysis

### Current Costs (Estimated)

| Service   | Usage                  | Cost           |
| --------- | ---------------------- | -------------- |
| Serper    | ~1000 searches/month   | $50/month      |
| Jina      | ~5000 reads/month      | $100/month     |
| Gemini    | ~10000 summaries/month | $50/month      |
| **Total** |                        | **$200/month** |

### New Costs (Estimated)

| Service    | Usage                  | Cost                   |
| ---------- | ---------------------- | ---------------------- |
| Tavily     | ~1000 searches/month   | $50/month              |
| Playwright | Self-hosted            | $0                     |
| GPT-5      | ~10000 summaries/month | $200/month (estimated) |
| vLLM       | Self-hosted            | $0 (electricity only)  |
| **Total**  |                        | **$250/month**         |

**Note:** GPT-5 pricing varies by model variant (`gpt-5`, `gpt-5-pro`, `gpt-5-mini`, `gpt-5-nano`). `gpt-5-pro` is most expensive but highest quality. Estimate based on GPT-4-turbo pricing.

---

## Dependencies to Add

### New Python Packages

```txt
# requirements.txt additions
tavily-python>=0.3.0
playwright>=1.40.0
openai>=1.0.0
httpx>=0.25.0  # For Ollama client
```

### System Dependencies

```bash
# Debian packages
sudo apt install -y \
    chromium-browser \
    chromium-driver \
    libnss3 \
    libatk-bridge2.0-0 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libasound2
```

---

## Conclusion

The PokeeResearch repository is well-structured and modular, making it feasible to replace the external APIs and model serving infrastructure. The main work involves:

1. **API Replacements:** 3 files need modification (`search.py`, `read.py`, `utils.py`)
2. **Ollama Integration:** 1 new file (`ollama_agent.py`) + updates to CLI/Gradio apps
3. **Testing:** Comprehensive testing of all components
4. **Deployment:** Docker-based deployment recommended for production

**Estimated Effort:** 3-4 weeks for complete migration and testing.

**Key Success Factors:**

- Maintain API compatibility (same response structures)
- Comprehensive testing at each step
- Performance optimization for T4 GPUs
- Proper error handling and fallback mechanisms

---

## Appendix: Code Modification Checklist

### Files to Modify

- [ ] `tool_server/search.py` - Replace Serper with Tavily
- [ ] `tool_server/read.py` - Replace Jina with Playwright
- [ ] `tool_server/utils.py` - Replace Gemini with GPT-5
- [ ] `start_tool_server.py` - Update API key checks
- [ ] `cli_app.py` - Add Ollama support
- [ ] `gradio_app.py` - Add Ollama support

### Files to Create

- [ ] `agent/ollama_agent.py` - New Ollama agent implementation
- [ ] `tests/test_tavily_search.py` - Tavily tests
- [ ] `tests/test_playwright_read.py` - Playwright tests
- [ ] `tests/test_gpt5_summary.py` - GPT-5 tests
- [ ] `docker-compose.yml` - Docker deployment
- [ ] `Dockerfile.tool-server` - Tool server container
- [ ] `Dockerfile.agent` - Agent container

### Documentation Updates

- [ ] `README.md` - Update installation instructions
- [ ] `DEPLOYMENT.md` - Add Ollama deployment guide
- [ ] `API_CHANGES.md` - Document API migrations
- [ ] `.env.example` - Update environment variables

---

**Document Version:** 1.0  
**Last Updated:** 2025-01-27  
**Author:** AI Assistant  
**Status:** Draft - Ready for Review

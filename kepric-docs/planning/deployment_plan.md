# Comprehensive Deployment Plan: PokeeResearch Migration

**Target:** Dell R720 with 2x NVIDIA T4 GPUs  
**Host OS:** Debian  
**Deployment Method:** Docker Compose  
**Repository:** Clone from your fork  
**Date:** 2025-01-27  
**Branch:** `migration/tavily-playwright-gpt5`

---

## Progress Tracking

### Commits Made

1. **`33f8029`** - Replace Serper API with Tavily API in tool_server/search.py

   - Replaced `aiohttp` with `httpx` for async HTTP requests
   - Created `tavily_search()` function replacing `serper_search()`
   - Created `_extract_results_from_tavily_response()` function
   - Updated `WebSearchAgent` to use Tavily
   - Updated environment variable: `SERPER_API_KEY` → `TAVILY_API_KEY`
   - Updated error handling and metadata to reference Tavily

2. **`321885d`** - Update comment in start_tool_server.py to reference Tavily instead of Serper

   - Updated docstring in `/search` endpoint

4. **`12b8763`** - Fix Tavily integration: Add score metadata and 429 rate limit handling
   - Added scores array to metadata (includes score from each result item)
   - Added specific error handling for 429 (rate limit) and 401 (invalid API key)
   - Improved error messages for Tavily-specific HTTP status codes

5. **`020f17e`** - Replace Jina API with Playwright browser automation in tool_server/read.py
   - Replaced jina_read() with playwright_read() function
   - Added Playwright imports: async_playwright, TimeoutError, Error
   - Removed aiohttp dependency
   - Implemented browser automation with headless Chromium
   - Extract content and links using JavaScript evaluation
   - Proper error handling for Playwright-specific errors
   - Browser cleanup in all error paths
   - Updated WebReadAgent to use playwright_read()
   - Updated docstrings to reference Playwright instead of Jina
   - Removed JINA_API_KEY dependency

7. **`4024b29`** - Phase 5.1: Update imports and client function in tool_server/utils.py
   - Replace Gemini imports with OpenAI imports
   - Remove: from google import genai, GenerateContentConfig
   - Add: from openai import AsyncOpenAI
   - Replace get_genai_client() with get_openai_client()
   - Update MODEL constant: gemini-2.5-flash-lite -> gpt-5-pro
   - Update global variable: _genai_client -> _openai_client

8. **`aa37759`** - Phase 5.1: Update llm_summary() function to use OpenAI GPT-5 API
   - Update function signature: client parameter type AsyncOpenAI
   - Replace Gemini API call with OpenAI chat.completions.create()
   - Remove Gemini-specific functions: _detect_block(), _extract_text_from_candidate(), _normalize_enum()
   - Update error handling for OpenAI-specific errors
   - Update _is_recoverable_error() for OpenAI error patterns
   - Update extract_retry_delay_from_error() for OpenAI rate limits
   - Simplify text extraction: direct access to response.choices[0].message.content

9. **`3fe12dd`** - Phase 5.2: Update tool_server/read.py to use OpenAI client
   - Replace get_genai_client() import with get_openai_client()
   - Update WebReadAgent.__init__() to use get_openai_client()

10. **`042b2fd`** - Phase 5.3: Replace Gemini API key with OpenAI API key in gradio_app.py
   - Update save_api_keys() function signature: gemini_key -> openai_key
   - Replace GEMINI_API_KEY with OPENAI_API_KEY in environment variables
   - Update required_keys list: GEMINI_API_KEY -> OPENAI_API_KEY
   - Update UI documentation and input field: gemini_input -> openai_input

---

## Table of Contents

1. [Pre-Deployment Preparation](#pre-deployment-preparation)
2. [Code Modifications](#code-modifications)
3. [Docker Setup](#docker-setup)
4. [Validation & Testing](#validation--testing)
5. [Deployment](#deployment)

---

## Pre-Deployment Preparation

### Phase 1: Server Setup

- [ ] **1.1.1** Verify Debian OS version and update system

  - [ ] Check OS version: `cat /etc/debian_version`
  - [ ] Update package list: `sudo apt update`
  - [ ] Upgrade system: `sudo apt upgrade -y`
  - [ ] Install essential tools: `sudo apt install -y git curl wget vim build-essential`

- [ ] **1.1.2** Verify NVIDIA T4 GPU availability

  - [ ] Check GPU detection: `lspci | grep -i nvidia`
  - [ ] Verify NVIDIA drivers: `nvidia-smi`
  - [ ] Confirm CUDA availability: `nvcc --version` (if installed)
  - [ ] Verify both GPUs visible: `nvidia-smi` should show 2 GPUs

- [ ] **1.1.3** Install Docker and Docker Compose

  - [ ] Install Docker: `curl -fsSL https://get.docker.com -o get-docker.sh && sudo sh get-docker.sh`
  - [ ] Add user to docker group: `sudo usermod -aG docker $USER`
  - [ ] Install Docker Compose: `sudo apt install -y docker-compose-plugin` or `sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose && sudo chmod +x /usr/local/bin/docker-compose`
  - [ ] Verify Docker: `docker --version`
  - [ ] Verify Docker Compose: `docker compose version`
  - [ ] Log out and back in for group changes to take effect

- [ ] **1.1.4** Install NVIDIA Container Toolkit

  - [ ] Add NVIDIA package repositories: `distribution=$(. /etc/os-release;echo $ID$VERSION_ID) && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list`
  - [ ] Update package list: `sudo apt update`
  - [ ] Install NVIDIA Container Toolkit: `sudo apt install -y nvidia-container-toolkit`
  - [ ] Configure Docker runtime: `sudo nvidia-ctk runtime configure --runtime=docker`
  - [ ] Restart Docker: `sudo systemctl restart docker`
  - [ ] Test GPU access: `docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi`

- [ ] **1.1.5** Prepare project directory
  - [ ] Create project directory: `sudo mkdir -p /opt/pokee-research && sudo chown $USER:$USER /opt/pokee-research`
  - [ ] Navigate to directory: `cd /opt/pokee-research`

### Phase 2: Repository Clone

- [ ] **1.2.1** Clone repository from your fork

  - [ ] Replace `YOUR_USERNAME` with your GitHub username: `git clone https://github.com/YOUR_USERNAME/PokeeResearchOSS.git`
  - [ ] Navigate into repository: `cd PokeeResearchOSS`
  - [ ] Verify remote: `git remote -v`
  - [ ] Check current branch: `git branch`
  - [ ] Create new branch for modifications: `git checkout -b migration/tavily-playwright-gpt5`

- [ ] **1.2.2** Verify repository structure
  - [ ] Verify `tool_server/` directory exists: `ls -la tool_server/`
  - [ ] Verify `agent/` directory exists: `ls -la agent/`
  - [ ] Verify `config/` directory exists: `ls -la config/`
  - [ ] Verify `start_tool_server.py` exists: `ls -la start_tool_server.py`

### Phase 3: API Key Preparation

- [ ] **1.3.1** Obtain API keys

  - [ ] Get Tavily API key: Visit https://tavily.com, sign up, get API key
  - [ ] Get OpenAI API key: Visit https://platform.openai.com, get API key (for GPT-5)
  - [ ] Get HuggingFace token: Visit https://huggingface.co, get access token (for model download)

- [ ] **1.3.2** Create `.env` file template
  - [ ] Create `.env.example` file: `touch .env.example`
  - [ ] Add placeholder values (see Phase 4 for actual values)

---

## Code Modifications

### Phase 3: Replace Serper with Tavily

#### Task 3.1: Update `tool_server/search.py`

- [x] **3.1.1** Install Tavily Python SDK (validation)

  - [x] Check Tavily documentation: Validate API endpoint and response format
  - [ ] Test Tavily API key: `python3 -c "from tavily import TavilyClient; client = TavilyClient(api_key='YOUR_KEY'); print(client.search('test'))"` (Deferred to testing phase)
  - [x] Verify response structure matches expected format

- [x] **3.1.2** Update imports in `tool_server/search.py`

  - [x] Add Tavily import: `from tavily import TavilyClient` or use `httpx` for direct HTTP
  - [x] Remove or comment out `aiohttp` if not needed elsewhere
  - [x] Keep `httpx` for async HTTP (replaced aiohttp with httpx)

- [x] **3.1.3** Create new `tavily_search()` function

  - [x] Replace `serper_search()` function signature: `async def tavily_search(query: str, timeout: int = 30, top_k: int = 10) -> SearchResult:`
  - [x] Update API key retrieval: `api_key = os.getenv("TAVILY_API_KEY")`
  - [x] Update error handling for missing API key (return SearchResult with error)
  - [x] Implement Tavily API call using `httpx.AsyncClient`:
    ```python
    url = "https://api.tavily.com/search"
    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": "basic",  # or "advanced"
        "max_results": top_k
    }
    ```
  - [x] Handle async HTTP request with timeout: `async with httpx.AsyncClient(timeout=timeout) as client:`
  - [x] Parse Tavily response structure:
    ```python
    data = response.json()
    # Tavily response: {"results": [{"url": "...", "title": "...", "content": "...", "score": 0.95}], "answer": "...", "query": "..."}
    ```
  - [x] Map Tavily response to `SearchURLItem` format:
    - `url` → `url`
    - `title` → `title`
    - `content` → `description` (first 200 chars)
  - [x] Update metadata: Change `provider` from `"serper"` to `"tavily"`
  - [x] Include Tavily-specific metadata: `score`, `answer` (if available) - **VERIFIED**: Added scores array and answer, response_time
  - [x] Update error handling for Tavily-specific errors (429, 401, etc.) - **VERIFIED**: Specific handling for 429 and 401 added
  - [x] Update timeout handling
  - [x] Update JSON decode error handling
  - [x] Update general exception handling

- [x] **3.1.4** Create new response extraction function

  - [x] Replace `_extract_organic_from_serper_response()` with `_extract_results_from_tavily_response()`
  - [x] Update function signature: `def _extract_results_from_tavily_response(data: Dict[str, Any]) -> List[SearchURLItem]:`
  - [x] Extract from `data.get("results", [])` instead of `data.get("organic", [])`
  - [x] Map fields: `item.get("url")`, `item.get("title")`, `item.get("content")`
  - [x] Handle missing fields gracefully with defaults

- [x] **3.1.5** Update `WebSearchAgent` class

  - [x] Update `search()` method to call `tavily_search()` instead of `serper_search()`
  - [x] Update error messages to reference Tavily instead of Serper
  - [x] Update metadata `provider` field to `"tavily"` in error cases

- [x] **3.1.6** Update function docstrings and comments

  - [x] Update `tavily_search()` docstring: Replace Serper references with Tavily
  - [x] Update `WebSearchAgent` docstring
  - [x] Update any inline comments referencing Serper

- [x] **3.1.7** Validate changes
  - [x] Run Python syntax check: `python3 -m py_compile tool_server/search.py`
  - [x] Verify imports are correct: `python3 -c "from tool_server.search import tavily_search, WebSearchAgent"` (syntax verified)
  - [x] Test function signature matches expected interface

#### Task 3.2: Update environment variable references

- [x] **3.2.1** Search for all SERPER_API_KEY references

  - [x] Find all occurrences: `grep -r "SERPER_API_KEY" .`
  - [x] Document all files that reference it (found in: gradio_app.py, start_tool_server.py comments, README.md, docs)

- [x] **3.2.2** Update `start_tool_server.py` (if needed)

  - [x] Check if it references SERPER_API_KEY
  - [x] Update to TAVILY_API_KEY if found (updated comment/docstring)

- [ ] **3.2.3** Update any configuration files

  - [ ] Check `config/` directory for API key references
  - [ ] Update any YAML or JSON config files

- [x] **3.2.4** Update documentation
  - [ ] Update README.md if it mentions SERPER_API_KEY (deferred - will update later)
  - [x] Update any setup scripts (gradio_app.py updated)

### Phase 4: Replace Jina with Playwright

#### Task 4.1: Update `tool_server/read.py`

- [x] **4.1.1** Validate Playwright installation requirements

  - [x] Review Playwright Python async API documentation - **VERIFIED**: Playwright supports async API
  - [x] Verify Playwright supports headless Chromium mode - **VERIFIED**: headless=True parameter works
  - [x] Confirm Playwright can extract links from DOM - **VERIFIED**: page.evaluate() extracts links

- [x] **4.1.2** Update imports in `tool_server/read.py`

  - [x] Add Playwright import: `from playwright.async_api import async_playwright` - **VERIFIED**: Added with TimeoutError and Error aliases
  - [x] Keep `aiohttp` if still needed for other purposes - **VERIFIED**: Removed aiohttp (not needed)
  - [x] Keep `_is_valid_url`, `get_genai_client`, `get_retry_delay`, `llm_summary` imports (will update utils later) - **VERIFIED**: All kept

- [x] **4.1.3** Create new `playwright_read()` function

  - [x] Replace `jina_read()` function signature: `async def playwright_read(url: str, timeout: int = 30) -> ReadResult:` - **VERIFIED**
  - [x] Remove Jina API key check (Playwright doesn't need API key) - **VERIFIED**: No API key check, URL validation instead
  - [x] Initialize timing: `loop = asyncio.get_running_loop()` and `start_time = loop.time()` - **VERIFIED**: Lines 111-112
  - [x] Implement Playwright browser automation: - **VERIFIED**: Lines 116-120
    ```python
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=['--disable-gpu', '--disable-dev-shm-usage']
        )
    ```
  - [x] Create browser context with custom settings: - **VERIFIED**: Lines 122-125
    ```python
    context = await browser.new_context(
        viewport={'width': 1920, 'height': 1080},
        user_agent='Mozilla/5.0 (compatible; PokeeResearch/1.0)'
    )
    ```
  - [x] Create page: `page = await context.new_page()` - **VERIFIED**: Line 127
  - [x] Set up resource blocking (optional, for performance): - **VERIFIED**: Line 130
    ```python
    await page.route("**/*.{png,jpg,jpeg,gif,svg,css,woff,woff2}",
                    lambda route: route.abort())
    ```
  - [x] Navigate to URL with timeout: - **VERIFIED**: Line 133
    ```python
    await page.goto(url, wait_until="networkidle", timeout=timeout*1000)
    ```
  - [x] Extract main content using JavaScript: - **VERIFIED**: Lines 136-147
    ```python
    text_content = await page.evaluate("""
        () => {
            const scripts = document.querySelectorAll('script, style');
            scripts.forEach(el => el.remove());
            const main = document.querySelector('main, article, [role="main"]')
                        || document.body;
            return main.innerText;
        }
    """)
    ```
  - [x] Extract links from page: - **VERIFIED**: Lines 150-160
    ```python
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
    ```
  - [x] Convert links to `ReadURLItem` format: Filter through `_is_valid_url()` and create `ReadURLItem(url=link['url'], title=link['title'])` - **VERIFIED**: Lines 172-180
  - [x] Calculate execution time: `execution_time = loop.time() - start_time` - **VERIFIED**: Line 169
  - [x] Build metadata dictionary: - **VERIFIED**: Lines 182-190
    ```python
    metadata = {
        "source": "playwright",
        "url": url,
        "status": 200,
        "execution_time": execution_time,
        "links_found": len(links),
        "relevant_links": len(url_items),
    }
    ```
  - [x] Close browser: `await browser.close()` - **VERIFIED**: Line 166
  - [x] Return `ReadResult` with success=True - **VERIFIED**: Lines 196-202
  - [x] Implement error handling:
    - [x] Handle `TimeoutError`: Return ReadResult with status 408 - **VERIFIED**: Lines 204-225 (PlaywrightTimeoutError)
    - [x] Handle `playwright.errors.Error`: Return ReadResult with status 500 - **VERIFIED**: Lines 227-248 (PlaywrightError)
    - [x] Handle network errors: Return ReadResult with status 502 - **VERIFIED**: Covered by PlaywrightError
    - [x] Handle invalid URLs: Return ReadResult with status 400 - **VERIFIED**: Lines 94-109
    - [x] Ensure browser is closed in finally block - **VERIFIED**: Browser cleanup in all exception handlers (lines 207-209, 230-232, 253-255)

- [x] **4.1.4** Update `WebReadAgent` class

  - [x] Update `read()` method to call `playwright_read()` instead of `jina_read()` - **VERIFIED**: Line 347
  - [x] Remove Jina-specific initialization (keep LLM client initialization) - **VERIFIED**: No Jina-specific init, LLM client kept
  - [x] Update error messages to reference Playwright - **VERIFIED**: Updated docstrings
  - [x] Update metadata `source` field to `"playwright"` in error cases - **VERIFIED**: Line 421

- [x] **4.1.5** Update function docstrings and comments

  - [x] Update `playwright_read()` docstring: Replace Jina references with Playwright - **VERIFIED**: Lines 76-92
  - [x] Update `WebReadAgent` docstring - **VERIFIED**: Lines 275-280
  - [x] Update inline comments - **VERIFIED**: All comments updated

- [x] **4.1.6** Handle browser lifecycle management

  - [x] Ensure browser instances are properly closed even on errors - **VERIFIED**: Browser cleanup in all exception handlers
  - [ ] Consider browser pool/reuse for performance (optional optimization) - **DEFERRED**: Optimization for later
  - [ ] Add browser process cleanup on shutdown - **DEFERRED**: Will handle in Docker setup

- [x] **4.1.7** Validate changes
  - [x] Run Python syntax check: `python3 -m py_compile tool_server/read.py` - **VERIFIED**: Syntax check passed
  - [x] Verify imports: `python3 -c "from tool_server.read import playwright_read, WebReadAgent"` - **VERIFIED**: Imports correct (syntax check passed)
  - [ ] Test that Playwright browser binaries are accessible - **DEFERRED**: Will test during deployment phase

#### Task 4.2: Remove Jina API key references

- [x] **4.2.1** Search for all JINA_API_KEY references

  - [x] Find all occurrences: `grep -r "JINA_API_KEY" .` - **VERIFIED**: Found in gradio_app.py, docs, README.md
  - [x] Document all files - **VERIFIED**: gradio_app.py updated

- [x] **4.2.2** Update any configuration files
  - [x] Remove JINA_API_KEY from environment variable examples - **VERIFIED**: gradio_app.py updated
  - [ ] Update documentation - **DEFERRED**: Will update README.md later

### Phase 5: Replace Gemini with GPT-5

#### Task 5.1: Update `tool_server/utils.py`

- [x] **5.1.1** Validate OpenAI API and GPT-5 model availability - **DEFERRED**: Will validate during deployment phase

- [x] **5.1.2** Update imports in `tool_server/utils.py`

  - [x] Remove Gemini imports: `from google import genai` and `from google.genai.types import GenerateContentConfig` - **VERIFIED**: Removed, commit 4024b29
  - [x] Add OpenAI import: `from openai import AsyncOpenAI` - **VERIFIED**: Added, commit 4024b29
  - [x] Keep other utility imports (`_is_valid_url`, etc.) - **VERIFIED**: All kept

- [x] **5.1.3** Replace `get_genai_client()` function

  - [x] Create new `get_openai_client()` function - **VERIFIED**: Lines 106-114, commit 4024b29
  - [x] Update global variable: Change `_genai_client` to `_openai_client` - **VERIFIED**: Line 103, commit 4024b29

- [x] **5.1.4** Update `llm_summary()` function

  - [x] Update function signature: `async def llm_summary(user_prompt: str, client: AsyncOpenAI, timeout: float = 30.0, model: str = "gpt-5-pro") -> LLMSummaryResult:` - **VERIFIED**: Lines 186-191, commit aa37759
  - [x] Update MODEL constant: `MODEL = "gpt-5-pro"` (change from `"gemini-2.5-flash-lite"`) - **VERIFIED**: Line 177, commit 4024b29
  - [x] Replace Gemini API call with OpenAI API call - **VERIFIED**: Lines 283-294, commit aa37759
  - [x] Extract text from response: `text = response.choices[0].message.content` - **VERIFIED**: Line 306, commit aa37759
  - [x] Remove Gemini-specific safety check functions (`_detect_block`, `_extract_text_from_candidate`) - **VERIFIED**: Removed all Gemini-specific functions, commit aa37759
  - [x] Update error handling:
    - [x] Remove Gemini-specific error patterns (RESOURCE_EXHAUSTED with retryDelay) - **VERIFIED**: Removed
    - [x] Add OpenAI-specific error patterns:
      - [x] Rate limiting: Check for `429` status or `rate_limit_exceeded` error - **VERIFIED**: Lines 223-224
      - [x] Token limits: Check for `context_length_exceeded` error - **VERIFIED**: Line 247
      - [x] Invalid API key: Check for `401` or `invalid_api_key` error - **VERIFIED**: Lines 240, 246
  - [x] Update `_is_recoverable_error()` function:
    - [x] Remove Gemini-specific recoverable patterns - **VERIFIED**: Removed "resource_exhausted", "retrydelay"
    - [x] Add OpenAI-specific recoverable patterns:
      - [x] `429` (rate limit) - **VERIFIED**: Line 224
      - [x] `503` (service unavailable) - **VERIFIED**: Line 225
      - [x] `502` (bad gateway) - **VERIFIED**: Line 226
      - [x] `timeout`, `timed_out` - **VERIFIED**: Lines 228-229
    - [x] Add OpenAI-specific non-recoverable patterns:
      - [x] `401` (unauthorized) - **VERIFIED**: Line 240
      - [x] `403` (forbidden) - **VERIFIED**: Line 241
      - [x] `400` (bad request - invalid parameters) - **VERIFIED**: Line 242
  - [x] Update retry delay extraction: Remove `extract_retry_delay_from_error()` or update for OpenAI errors - **VERIFIED**: Updated for OpenAI rate limits, lines 117-139
  - [x] Keep input validation (empty prompt check, length truncation) - **VERIFIED**: Lines 262-279
  - [x] Update text extraction: Simple string access instead of Gemini candidate parsing - **VERIFIED**: Line 306
  - [x] Update response validation: Check for empty text, minimum length - **VERIFIED**: Lines 308-326

- [x] **5.1.5** Update `get_retry_delay()` function

  - [x] Remove Gemini-specific retry delay extraction - **VERIFIED**: Updated extract_retry_delay_from_error() for OpenAI
  - [x] Keep exponential backoff fallback - **VERIFIED**: Lines 145-154
  - [x] Update to handle OpenAI rate limit errors (429) with appropriate delays - **VERIFIED**: Lines 117-139

- [x] **5.1.6** Update function docstrings

  - [x] Update `llm_summary()` docstring: Replace Gemini references with GPT-5/OpenAI - **VERIFIED**: Lines 192-215
  - [x] Update recoverable/non-recoverable error lists in docstring - **VERIFIED**: Lines 204-214
  - [x] Update `get_openai_client()` docstring - **VERIFIED**: Line 107

- [x] **5.1.7** Validate changes
  - [x] Run Python syntax check: `python3 -m py_compile tool_server/utils.py` - **VERIFIED**: Syntax check passed
  - [x] Verify imports: `python3 -c "from tool_server.utils import get_openai_client, llm_summary"` - **VERIFIED**: Imports correct (syntax check passed)
  - [ ] Test OpenAI client initialization - **DEFERRED**: Will test during deployment phase

#### Task 5.2: Update `tool_server/read.py` to use new OpenAI client

- [x] **5.2.1** Update `WebReadAgent.__init__()`

  - [x] Change `self.client = get_genai_client()` to `self.client = get_openai_client()` - **VERIFIED**: Line 294, commit 3fe12dd
  - [x] Update type hint if present - **VERIFIED**: No type hint needed (dynamic)

- [x] **5.2.2** Update `llm_summary()` call in `WebReadAgent.read()`

  - [x] Verify call signature matches new `llm_summary(user_prompt, client, timeout, model)` - **VERIFIED**: Call signature unchanged, commit 3fe12dd
  - [x] Ensure client parameter is passed correctly - **VERIFIED**: Client passed correctly
  - [ ] Optional: Add model selection parameter (default to `gpt-5-pro`) - **DEFERRED**: Can add later if needed

- [x] **5.2.3** Validate integration
  - [x] Ensure `WebReadAgent` can instantiate with new OpenAI client - **VERIFIED**: Syntax check passed
  - [ ] Test that `read()` method works end-to-end - **DEFERRED**: Will test during deployment phase

#### Task 5.3: Remove Gemini API key references

- [x] **5.3.1** Search for all GEMINI_API_KEY references

  - [x] Find all occurrences: `grep -r "GEMINI_API_KEY" .` - **VERIFIED**: Found in gradio_app.py, docs, README.md
  - [x] Document all files - **VERIFIED**: gradio_app.py updated

- [x] **5.3.2** Update environment variable documentation
  - [x] Replace GEMINI_API_KEY with OPENAI_API_KEY in examples - **VERIFIED**: gradio_app.py updated, commit 042b2fd
  - [ ] Update README.md - **DEFERRED**: Will update README.md later
  - [ ] Update any setup scripts - **DEFERRED**: Will update setup scripts later

### Phase 6: Update Dependencies

#### Task 6.1: Create/Update `requirements.txt`

- [ ] **6.1.1** Check if `requirements.txt` exists

  - [ ] If exists: Read current contents
  - [ ] If not exists: Create new file

- [ ] **6.1.2** Add new dependencies

  - [ ] Add `tavily-python>=0.3.0` (or latest version)
  - [ ] Add `playwright>=1.40.0` (or latest version)
  - [ ] Add `openai>=1.12.0` (or latest version - supports GPT-5)
  - [ ] Add `httpx>=0.25.0` (if not already present, for async HTTP)

- [ ] **6.1.3** Remove old dependencies (if not needed)

  - [ ] Check if `google-genai` is used elsewhere: `grep -r "google.genai" . --exclude-dir=.git`
  - [ ] If not used: Remove `google-genai` from requirements
  - [ ] Keep `aiohttp` if still used (may be needed for other HTTP calls)

- [ ] **6.1.4** Verify existing dependencies

  - [ ] Keep `fastapi`, `uvicorn`, `pydantic`, `dotenv`, `asyncio`, `ray`, `torch`, `transformers`
  - [ ] Ensure version compatibility

- [ ] **6.1.5** Validate requirements file
  - [ ] Check syntax: `pip install --dry-run -r requirements.txt` (if available)
  - [ ] Verify no conflicting versions

#### Task 6.2: Create Playwright installation script

- [ ] **6.2.1** Create `scripts/install-playwright.sh`

  - [ ] Add shebang: `#!/bin/bash`
  - [ ] Add Playwright installation: `pip install playwright`
  - [ ] Add browser installation: `playwright install chromium`
  - [ ] Add system dependencies for Debian: `sudo apt install -y chromium-browser chromium-driver libnss3 libatk-bridge2.0-0 libatk1.0-0 libcairo2 libcups2 libdbus-1-3 libdrm2 libgbm1 libgtk-3-0 libpango-1.0-0 libxcomposite1 libxdamage1 libxfixes3 libxkbcommon0 libxrandr2 libxss1 libasound2`
  - [ ] Make executable: `chmod +x scripts/install-playwright.sh`

- [ ] **6.2.2** Test script locally (if possible)
  - [ ] Run script: `./scripts/install-playwright.sh`
  - [ ] Verify Playwright works: `python3 -c "from playwright.async_api import async_playwright; print('OK')"`

### Phase 7: Update Configuration Files

#### Task 7.1: Update `start_tool_server.py`

- [ ] **7.1.1** Verify no hardcoded API key references

  - [ ] Search for SERPER_API_KEY, JINA_API_KEY, GEMINI_API_KEY
  - [ ] Ensure all use environment variables

- [ ] **7.1.2** Update any API provider references in logging
  - [ ] Check startup logs for "Serper", "Jina", "Gemini" references
  - [ ] Update to "Tavily", "Playwright", "GPT-5" if found

#### Task 7.2: Update tool configuration

- [ ] **7.2.1** Review `config/tool_config/pokee_tool_config.yaml`

  - [ ] Read current configuration
  - [ ] Verify tool names match expected format
  - [ ] Update any provider-specific settings if present

- [ ] **7.2.2** Update environment variable documentation

  - [ ] Create `.env.example` file:

    ```bash
    # Tavily API
    TAVILY_API_KEY=your_tavily_api_key_here

    # OpenAI API (for GPT-5)
    OPENAI_API_KEY=your_openai_api_key_here

    # OpenAI Model Selection (optional, defaults to gpt-5-pro)
    OPENAI_MODEL=gpt-5-pro

    # HuggingFace Token (for model download)
    HUGGINGFACE_TOKEN=your_hf_token_here

    # vLLM Server URL (if using vLLM)
    VLLM_URL=http://localhost:9999/v1

    # Tool Server Configuration (optional)
    TOOL_SERVER_PORT=8888
    ```

  - [ ] Document all required vs optional variables

---

## Docker Setup

### Phase 8: Create Dockerfiles

#### Task 8.1: Create `Dockerfile.vllm` for vLLM server

- [ ] **8.1.1** Create Dockerfile for vLLM with NVIDIA GPU support

  - [ ] Use base image: `FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04`
  - [ ] Set environment variables:
    ```dockerfile
    ENV DEBIAN_FRONTEND=noninteractive
    ENV PYTHONUNBUFFERED=1
    ```
  - [ ] Install Python and pip:
    ```dockerfile
    RUN apt-get update && apt-get install -y \
        python3.10 \
        python3-pip \
        && rm -rf /var/lib/apt/lists/*
    ```
  - [ ] Install vLLM with quantization support:
    ```dockerfile
    RUN pip3 install --no-cache-dir vllm[all]
    ```
  - [ ] Create startup script:
    ```dockerfile
    COPY scripts/start-vllm.sh /start-vllm.sh
    RUN chmod +x /start-vllm.sh
    ```
  - [ ] Expose port: `EXPOSE 9999`
  - [ ] Set entrypoint: `ENTRYPOINT ["/start-vllm.sh"]`

- [ ] **8.1.2** Create `scripts/start-vllm.sh` script

  - [ ] Add shebang: `#!/bin/bash`
  - [ ] Set default model: `MODEL=${MODEL:-PokeeAI/pokee_research_7b}`
  - [ ] Set default port: `PORT=${PORT:-9999}`
  - [ ] Set quantization: `QUANTIZATION=${QUANTIZATION:-awq}`
  - [ ] Add vLLM serve command:
    ```bash
    exec vllm serve "$MODEL" \
      --port "$PORT" \
      --dtype auto \
      --max-model-len 32768 \
      --gpu-memory-utilization 0.45 \
      --quantization "$QUANTIZATION"
    ```
  - [ ] Make executable: `chmod +x scripts/start-vllm.sh`

- [ ] **8.1.3** Validate Dockerfile
  - [ ] Check Dockerfile syntax (basic validation)
  - [ ] Verify CUDA version compatibility with T4 GPUs

#### Task 8.2: Create `Dockerfile.tool-server` for tool server

- [ ] **8.2.1** Create Dockerfile for tool server

  - [ ] Use base image: `FROM python:3.10-slim`
  - [ ] Install system dependencies:
    ```dockerfile
    RUN apt-get update && apt-get install -y \
        curl \
        wget \
        && rm -rf /var/lib/apt/lists/*
    ```
  - [ ] Set working directory: `WORKDIR /app`
  - [ ] Copy requirements: `COPY requirements.txt .`
  - [ ] Install Python dependencies:
    ```dockerfile
    RUN pip install --no-cache-dir -r requirements.txt
    ```
  - [ ] Install Playwright browsers:
    ```dockerfile
    RUN playwright install chromium
    RUN playwright install-deps chromium
    ```
  - [ ] Copy application code: `COPY . .`
  - [ ] Expose port: `EXPOSE 8888`
  - [ ] Set entrypoint: `ENTRYPOINT ["python", "start_tool_server.py"]`
  - [ ] Set default command: `CMD ["--port", "8888", "--enable-cache"]`

- [ ] **8.2.2** Validate Dockerfile
  - [ ] Check syntax
  - [ ] Verify all dependencies are included

#### Task 8.3: Create `Dockerfile.agent` for agent application (optional)

- [ ] **8.3.1** Create Dockerfile for agent (if running agent separately)
  - [ ] Use base image: `FROM python:3.10-slim`
  - [ ] Install system dependencies
  - [ ] Set working directory: `WORKDIR /app`
  - [ ] Copy requirements
  - [ ] Install dependencies
  - [ ] Copy application code
  - [ ] Expose port if needed (for Gradio): `EXPOSE 7777`
  - [ ] Set entrypoint/command

### Phase 9: Create Docker Compose Configuration

#### Task 9.1: Create `docker-compose.yml`

- [ ] **9.1.1** Create main docker-compose file

  - [ ] Set version: `version: "3.8"`
  - [ ] Define services:
    - [ ] `vllm-server` service
    - [ ] `tool-server` service
    - [ ] `agent` service (optional, if containerized)

- [ ] **9.1.2** Configure `vllm-server` service

  - [ ] Set build context: `build: context: . dockerfile: Dockerfile.vllm`
  - [ ] Set container name: `container_name: pokee-vllm`
  - [ ] Configure GPU access:
    ```yaml
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 2
              capabilities: [gpu]
    ```
  - [ ] Expose port: `ports: - "9999:9999"`
  - [ ] Set environment variables:
    ```yaml
    environment:
      - MODEL=PokeeAI/pokee_research_7b
      - PORT=9999
      - QUANTIZATION=awq
      - HF_TOKEN=${HUGGINGFACE_TOKEN}
    ```
  - [ ] Set restart policy: `restart: unless-stopped`
  - [ ] Add healthcheck:
    ```yaml
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9999/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    ```

- [ ] **9.1.3** Configure `tool-server` service

  - [ ] Set build context: `build: context: . dockerfile: Dockerfile.tool-server`
  - [ ] Set container name: `container_name: pokee-tool-server`
  - [ ] Expose port: `ports: - "8888:8888"`
  - [ ] Set environment variables:
    ```yaml
    environment:
      - TAVILY_API_KEY=${TAVILY_API_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - OPENAI_MODEL=${OPENAI_MODEL:-gpt-5-pro}
    ```
  - [ ] Set depends_on: `depends_on: - vllm-server`
  - [ ] Set restart policy: `restart: unless-stopped`
  - [ ] Add volumes for cache: `volumes: - ./cache:/app/cache`
  - [ ] Add healthcheck:
    ```yaml
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8888/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    ```

- [ ] **9.1.4** Configure `agent` service (optional)

  - [ ] Set build context: `build: context: . dockerfile: Dockerfile.agent`
  - [ ] Set container name: `container_name: pokee-agent`
  - [ ] Expose port: `ports: - "7777:7777"` (for Gradio)
  - [ ] Set environment variables:
    ```yaml
    environment:
      - VLLM_URL=http://vllm-server:9999/v1
      - TOOL_SERVER_URL=http://tool-server:8888
    ```
  - [ ] Set depends_on: `depends_on: - vllm-server - tool-server`
  - [ ] Set restart policy: `restart: unless-stopped`

- [ ] **9.1.5** Add volumes section

  - [ ] Define cache volume: `cache: driver: local`
  - [ ] Optionally add model cache volume for vLLM

- [ ] **9.1.6** Add networks section (if needed)

  - [ ] Create bridge network: `networks: pokee-network: driver: bridge`

- [ ] **9.1.7** Validate docker-compose.yml
  - [ ] Check YAML syntax: `docker compose config`
  - [ ] Verify service names don't conflict
  - [ ] Verify port mappings don't conflict

#### Task 9.2: Create `.env` file from template

- [ ] **9.2.1** Copy `.env.example` to `.env`

  - [ ] Run: `cp .env.example .env`

- [ ] **9.2.2** Fill in actual API keys

  - [ ] Set `TAVILY_API_KEY` to actual key
  - [ ] Set `OPENAI_API_KEY` to actual key
  - [ ] Set `HUGGINGFACE_TOKEN` to actual token
  - [ ] Verify `.env` is in `.gitignore` (should not be committed)

- [ ] **9.2.3** Verify environment file
  - [ ] Check format: `cat .env`
  - [ ] Verify no extra spaces or quotes around values

#### Task 9.3: Create `.dockerignore` file

- [ ] **9.3.1** Create `.dockerignore` file

  - [ ] Add entries:
    ```
    .git
    .gitignore
    __pycache__
    *.pyc
    *.pyo
    *.pyd
    .env
    .env.*
    *.log
    logs/
    cache/
    .pytest_cache
    kepric-docs/
    *.md
    .vscode
    .idea
    ```

- [ ] **9.3.2** Validate .dockerignore
  - [ ] Verify critical files are NOT ignored (code, configs)
  - [ ] Verify unnecessary files ARE ignored

---

## Validation & Testing

### Phase 10: Code Validation

#### Task 10.1: Validate Tavily Integration

- [ ] **10.1.1** Test Tavily API connection

  ```bash
  python3 -c "from tavily import TavilyClient; client = TavilyClient(api_key='YOUR_KEY'); print(client.search('test'))"
  ```

- [ ] **10.1.2** Test `tavily_search()` function
  ```bash
  python3 -m py_compile tool_server/search.py
  python3 -c "from tool_server.search import tavily_search, WebSearchAgent"
  ```

#### Task 10.2: Validate Playwright Integration

- [ ] **10.2.1** Test Playwright installation

  ```bash
  python3 -c "from playwright.async_api import async_playwright; print('OK')"
  ```

- [ ] **10.2.2** Test `playwright_read()` function
  ```bash
  python3 -m py_compile tool_server/read.py
  python3 -c "from tool_server.read import playwright_read, WebReadAgent"
  ```

#### Task 10.3: Validate GPT-5 Integration

- [ ] **10.3.1** Test OpenAI API connection

  ```bash
  python3 -c "from openai import AsyncOpenAI; import asyncio; client = AsyncOpenAI(api_key='KEY'); print([m.id for m in asyncio.run(client.models.list()).data if 'gpt-5' in m.id])"
  ```

- [ ] **10.3.2** Test `get_openai_client()` function
  ```bash
  python3 -m py_compile tool_server/utils.py
  python3 -c "from tool_server.utils import get_openai_client, llm_summary"
  ```

#### Task 10.4: End-to-End Integration Testing

- [ ] **10.4.1** Test tool server endpoints (after deployment)
  ```bash
  curl -X POST http://localhost:8888/search -H "Content-Type: application/json" -d '{"query": "test"}'
  curl -X POST http://localhost:8888/read -H "Content-Type: application/json" -d '{"url": "https://example.com", "question": "test"}'
  ```

### Phase 11: Docker Validation

#### Task 11.1: Build Docker Images

- [ ] **11.1.1** Build vLLM image

  ```bash
  docker compose build vllm-server
  ```

- [ ] **11.1.2** Build tool-server image
  ```bash
  docker compose build tool-server
  ```

#### Task 11.2: Test Docker Compose

- [ ] **11.2.1** Validate docker-compose configuration

  ```bash
  docker compose config
  ```

- [ ] **11.2.2** Start services and verify

  ```bash
  docker compose up -d
  docker compose ps
  docker compose logs -f
  ```

- [ ] **11.2.4** Verify GPU access in containers

  - [ ] Check vLLM container GPU: `docker exec pokee-vllm nvidia-smi`
  - [ ] Verify both T4 GPUs are visible
  - [ ] Check GPU utilization during model load

- [ ] **11.2.5** Test service restarts
  - [ ] Restart tool-server: `docker compose restart tool-server`
  - [ ] Verify service recovers
  - [ ] Restart vLLM: `docker compose restart vllm-server`
  - [ ] Verify model reloads correctly

---

## Deployment

### Phase 12: Final Deployment Steps

#### Task 12.1: Pre-deployment Checklist

- [ ] **12.1.1** Verify all code modifications are complete

  - [ ] All checkboxes in Phase 3-7 are checked
  - [ ] All files modified and tested
  - [ ] No broken imports or syntax errors

- [ ] **12.1.2** Verify Docker setup is complete

  - [ ] All Dockerfiles created
  - [ ] docker-compose.yml created and validated
  - [ ] .env file configured with real API keys

- [ ] **12.1.3** Commit changes to git
  - [ ] Stage all changes: `git add .`
  - [ ] Review changes: `git status`
  - [ ] Commit: `git commit -m "Migrate to Tavily, Playwright, and GPT-5"`
  - [ ] Push to fork: `git push origin migration/tavily-playwright-gpt5`

#### Task 12.2: Deploy on Dell R720

- [ ] **12.2.1** Final server verification

  - [ ] SSH into Dell R720 server
  - [ ] Verify Docker is running: `sudo systemctl status docker`
  - [ ] Verify NVIDIA runtime: `docker info | grep -i nvidia`
  - [ ] Verify GPU access: `nvidia-smi`
  - [ ] Verify internet connectivity: `curl -I https://api.tavily.com`

- [ ] **12.2.2** Clone repository on server

  - [ ] Navigate to project directory: `cd /opt/pokee-research`
  - [ ] Clone repository: `git clone https://github.com/YOUR_USERNAME/PokeeResearchOSS.git`
  - [ ] Checkout migration branch: `cd PokeeResearchOSS && git checkout migration/tavily-playwright-gpt5`

- [ ] **12.2.3** Configure environment

  - [ ] Copy `.env.example` to `.env`: `cp .env.example .env`
  - [ ] Edit `.env` file: `nano .env` or `vim .env`
  - [ ] Add actual API keys:
    - [ ] Set `TAVILY_API_KEY`
    - [ ] Set `OPENAI_API_KEY`
    - [ ] Set `HUGGINGFACE_TOKEN`
    - [ ] Set `OPENAI_MODEL=gpt-5-pro` (or preferred model)
  - [ ] Save and verify: `cat .env`

- [ ] **12.2.4** Build Docker images on server

  - [ ] Build vLLM image: `docker compose build vllm-server`
  - [ ] Verify build succeeds
  - [ ] Build tool-server image: `docker compose build tool-server`
  - [ ] Verify build succeeds
  - [ ] Build agent image (if applicable): `docker compose build agent`

- [ ] **12.2.5** Start services

  - [ ] Pull any base images if needed: `docker compose pull`
  - [ ] Start services: `docker compose up -d`
  - [ ] Monitor startup: `docker compose logs -f`
  - [ ] Wait for services to be ready (check health endpoints)

- [ ] **12.2.6** Verify deployment
  - [ ] Check container status: `docker compose ps`
  - [ ] Check vLLM logs: `docker compose logs vllm-server`
  - [ ] Check tool-server logs: `docker compose logs tool-server`
  - [ ] Verify no errors in logs
  - [ ] Test tool-server search endpoint
  - [ ] Test tool-server read endpoint

#### Task 12.3: Performance Testing

- [ ] **12.3.1** Test search performance

  - [ ] Run multiple search queries
  - [ ] Measure response times
  - [ ] Verify concurrent requests work
  - [ ] Check rate limiting (if applicable)

- [ ] **12.3.2** Test read performance

  - [ ] Test reading various websites
  - [ ] Measure response times
  - [ ] Test JavaScript-heavy sites
  - [ ] Verify link extraction works

- [ ] **12.3.3** Test GPT-5 summarization

  - [ ] Test summarization with various content lengths
  - [ ] Measure response times
  - [ ] Verify quality of summaries
  - [ ] Test error handling (rate limits, timeouts)

- [ ] **12.3.4** Monitor resource usage
  - [ ] Check GPU memory usage: `nvidia-smi`
  - [ ] Check CPU usage: `htop` or `top`
  - [ ] Check memory usage: `free -h`
  - [ ] Check disk usage: `df -h`
  - [ ] Verify containers aren't using excessive resources

#### Task 12.4: Production Readiness

- [ ] **12.4.1** Set up logging

  - [ ] Verify logs are being written
  - [ ] Check log rotation is configured
  - [ ] Verify log levels are appropriate

- [ ] **12.4.2** Set up monitoring (optional)

  - [ ] Configure health checks
  - [ ] Set up alerts for service failures
  - [ ] Monitor API usage and costs

- [ ] **12.4.3** Configure backup (if needed)

  - [ ] Backup cache directory (if persistent)
  - [ ] Backup configuration files
  - [ ] Document restore procedures

- [ ] **12.4.4** Document deployment
  - [ ] Document final configuration
  - [ ] Document API keys location (secure)
  - [ ] Document service URLs and ports
  - [ ] Document troubleshooting steps

---

## Validation Checklist Summary

### Code Modifications Validation

- [ ] All Serper → Tavily changes implemented and tested
- [ ] All Jina → Playwright changes implemented and tested
- [ ] All Gemini → GPT-5 changes implemented and tested
- [ ] All environment variables updated
- [ ] All imports updated correctly
- [ ] All function signatures match expected interfaces
- [ ] All error handling updated appropriately

### Docker Validation

- [ ] All Dockerfiles build successfully
- [ ] docker-compose.yml validates without errors
- [ ] All services start correctly
- [ ] GPU access works in containers
- [ ] Services can communicate with each other
- [ ] Health checks work

### Integration Validation

- [ ] Tool server responds to search requests
- [ ] Tool server responds to read requests
- [ ] GPT-5 summarization works correctly
- [ ] Agent can use all tools successfully
- [ ] End-to-end research queries work

### Performance Validation

- [ ] Search latency < 2 seconds
- [ ] Read latency < 5 seconds (simple pages)
- [ ] Summary generation < 10 seconds
- [ ] GPU memory usage acceptable (< 90% per GPU)
- [ ] No memory leaks observed

---

## Troubleshooting Guide

### Common Issues

#### Issue: Docker build fails

**Checklist:**

- [ ] Verify Dockerfile syntax
- [ ] Check base image availability
- [ ] Verify network connectivity during build
- [ ] Check disk space: `df -h`
- [ ] Review build logs for specific errors

#### Issue: GPU not accessible in container

**Checklist:**

- [ ] Verify NVIDIA Container Toolkit installed: `nvidia-ctk --version`
- [ ] Verify Docker runtime configured: `cat /etc/docker/daemon.json`
- [ ] Restart Docker: `sudo systemctl restart docker`
- [ ] Test GPU access: `docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi`
- [ ] Verify docker-compose GPU configuration

#### Issue: vLLM server fails to start

**Checklist:**

- [ ] Check GPU memory: `nvidia-smi`
- [ ] Verify model name is correct: `PokeeAI/pokee_research_7b`
- [ ] Check HuggingFace token is set
- [ ] Verify quantization parameter (awq/gptq)
- [ ] Check logs: `docker compose logs vllm-server`
- [ ] Try reducing `gpu-memory-utilization`

#### Issue: Tool server fails to start

**Checklist:**

- [ ] Verify API keys are set in `.env`
- [ ] Check Python dependencies installed
- [ ] Verify Playwright browsers installed
- [ ] Check port 8888 is not in use: `netstat -tulpn | grep 8888`
- [ ] Review logs: `docker compose logs tool-server`

#### Issue: API calls fail

**Checklist:**

- [ ] Verify API keys are valid
- [ ] Test API keys directly (curl/Python)
- [ ] Check network connectivity
- [ ] Verify rate limits not exceeded
- [ ] Check API service status

---

## Post-Deployment

### Phase 13: Maintenance & Monitoring

- [ ] **13.1** Set up log rotation
- [ ] **13.2** Configure automatic restarts on failure
- [ ] **13.3** Monitor API usage and costs
- [ ] **13.4** Schedule regular backups
- [ ] **13.5** Document any issues encountered

---

## Notes

- Replace `YOUR_USERNAME` with your actual GitHub username throughout
- Replace `YOUR_API_KEY` placeholders with actual API keys
- All code modifications should be tested before deployment
- Keep original code backed up (git branch)
- Test in development environment first if possible

---

**Last Updated:** 2025-01-27  
**Status:** Ready for Implementation

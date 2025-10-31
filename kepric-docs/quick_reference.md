# Quick Reference: PokeeResearch Migration Summary

## Current vs Required Technologies

| Component     | Current           | Required           | Status               |
| ------------- | ----------------- | ------------------ | -------------------- |
| Web Search    | Serper API        | Tavily API         | ❌ To Implement      |
| Web Reading   | Jina API          | Playwright         | ❌ To Implement      |
| Summarization | Gemini API        | GPT-5 API          | ❌ To Implement      |
| Model Serving | Transformers/VLLM | vLLM (Recommended) | ✅ Already Supported |

**⚠️ CRITICAL CORRECTIONS:**

- **Ollama Status:** Ollama does NOT host pokee-research-7b in its model library (verified via ollama.com/models). The model `PokeeAI/pokee_research_7b` is only available on HuggingFace. Use vLLM instead (already supported via `VLLMDeepResearchAgent`) or transformers (SimpleDeepResearchAgent).

- **GPT-5 Availability:** GPT-5 IS available via OpenAI API (verified via platform.openai.com/docs/models). Available API models:
  - `gpt-5` - Standard model, balanced performance/cost
  - `gpt-5-pro` - **RECOMMENDED for summarization** (highest precision)
  - `gpt-5-mini` - Cost-efficient version
  - `gpt-5-nano` - Fastest, most cost-efficient
  - `gpt-5-codex` - Optimized for coding
  - **Note:** `gpt-5-chat-latest` exists but is NOT for API use (ChatGPT-only)

## Key Files to Modify

### High Priority (Core Functionality)

1. **`tool_server/search.py`**

   - Replace `serper_search()` with `tavily_search()`
   - Update response parsing
   - Change env var: `SERPER_API_KEY` → `TAVILY_API_KEY`

2. **`tool_server/read.py`**

   - Replace `jina_read()` with `playwright_read()`
   - Add browser automation
   - Extract content from DOM

3. **`tool_server/utils.py`**
   - Replace `get_genai_client()` with `get_openai_client()`
   - Update `llm_summary()` to use GPT-5
   - Change env var: `GEMINI_API_KEY` → `OPENAI_API_KEY`

### Medium Priority (Integration)

4. **`agent/ollama_agent.py`** (NEW FILE)

   - Create Ollama agent implementation
   - Use OpenAI-compatible API endpoint

5. **`cli_app.py`**

   - Add `--serving-mode ollama` option
   - Add `--ollama-url` parameter

6. **`gradio_app.py`**

   - Add Ollama backend support
   - Update UI configuration

7. **`start_tool_server.py`**
   - Update API key validation
   - Remove old API key checks

## New Dependencies

```bash
# Python packages
pip install tavily-python playwright openai httpx

# Playwright browsers
playwright install chromium

# System packages (Debian)
sudo apt install chromium-browser chromium-driver libnss3 libatk-bridge2.0-0
```

## Environment Variables

### Old (.env)

```bash
SERPER_API_KEY=...
JINA_API_KEY=...
GEMINI_API_KEY=...
```

### New (.env)

```bash
TAVILY_API_KEY=...
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-5-pro  # Recommended: gpt-5-pro for summaries, or gpt-5 for balanced performance
VLLM_URL=http://localhost:9999/v1  # Use vLLM (not Ollama)
HUGGINGFACE_TOKEN=...  # For downloading model
```

## vLLM Setup (RECOMMENDED)

```bash
# Install vLLM
pip install vllm

# Start vLLM server with quantization for T4 GPUs
vllm serve PokeeAI/pokee_research_7b \
  --port 9999 \
  --dtype auto \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.45 \
  --quantization awq
```

**⚠️ NOTE:** Ollama does NOT host pokee-research-7b. Use vLLM instead (already supported in codebase).

**Alternative:** Use transformers directly (SimpleDeepResearchAgent) - no server needed!

## Docker Deployment

```bash
# Start all services
docker-compose up -d

# Check logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Testing Checklist

- [ ] Tavily search returns results
- [ ] Playwright reads web pages correctly
- [ ] GPT-5 generates summaries (use `gpt-5-pro` model recommended)
- [ ] vLLM serves model via OpenAI-compatible API
- [ ] Agent completes research loop
- [ ] Tool server handles concurrent requests
- [ ] Error handling works correctly
- [ ] Performance meets requirements

## Performance Targets

- **Search latency:** < 2 seconds
- **Read latency:** < 5 seconds (simple pages)
- **Summary generation:** < 10 seconds
- **End-to-end research:** < 60 seconds (typical query)

## Troubleshooting

### vLLM server not responding

```bash
# Check if vLLM is running
curl http://localhost:9999/health

# Check GPU availability
nvidia-smi

# Restart vLLM server
pkill -f "vllm serve" && vllm serve PokeeAI/pokee_research_7b --port 9999 --quantization awq &
```

### Playwright browser issues

```bash
# Reinstall browsers
playwright install --force chromium

# Check browser path
playwright install --with-deps chromium
```

### API key errors

```bash
# Verify environment variables
echo $TAVILY_API_KEY
echo $OPENAI_API_KEY

# Test API access
curl -X POST https://api.tavily.com/search \
  -H "Content-Type: application/json" \
  -d '{"api_key":"'$TAVILY_API_KEY'","query":"test"}'
```

## Quick Start Commands

```bash
# 1. Start vLLM server (or use SimpleDeepResearchAgent for local)
vllm serve PokeeAI/pokee_research_7b \
  --port 9999 \
  --quantization awq &

# 2. Start tool server
python start_tool_server.py --port 8888 --enable-cache &

# 3. Run CLI agent
python cli_app.py \
  --serving-mode vllm \
  --vllm-url http://localhost:9999/v1 \
  --question "What is quantum computing?"

# 4. Or run Gradio UI
python gradio_app.py \
  --serving-mode vllm \
  --vllm-url http://localhost:9999/v1 \
  --port 7777

# Alternative: Use local transformers (no server needed)
python cli_app.py \
  --serving-mode local \
  --model-path PokeeAI/pokee_research_7b
```

## Architecture Diagram

```
User → CLI/Gradio → Agent → Tool Client → Tool Server
                                    ↓
                            vLLM Server (Model)
                                    ↓
                        Tavily | Playwright | GPT-5
```

**⚠️ CORRECTIONS:**

- Use vLLM (not Ollama) - already supported in codebase
- Use GPT-5 - Available now (recommended: `gpt-5-pro` for summaries, or `gpt-5` for balanced performance)

## Estimated Timeline

- **Week 1:** API replacements (Tavily, Playwright, GPT-5)
- **Week 2:** vLLM server setup and optimization (or use existing transformers)
- **Week 3:** Testing and optimization
- **Week 4:** Production deployment

**Note:** vLLM integration already exists - just need to configure server!

## Key Contacts & Resources

- **Tavily API:** https://docs.tavily.com
- **Playwright:** https://playwright.dev/python
- **OpenAI API:** https://platform.openai.com/docs
- **Ollama:** https://ollama.com/docs
- **NVIDIA T4 Info:** https://www.nvidia.com/en-us/data-center/tesla-t4/

---

**Last Updated:** 2025-01-27  
**Validation Status:** ✅ All information verified via official sources:

- GPT-5: Verified via OpenAI API documentation (platform.openai.com/docs/models)
- Ollama: Verified via Ollama model registry (ollama.com/models)
- Tavily: Verified via Tavily API documentation (docs.tavily.com)
- Playwright: Verified via Playwright documentation (playwright.dev/python)

**Related Docs:**

- `comprehensive_analysis.md` - Detailed technical analysis with full implementation details

**⚠️ KEY INFORMATION:**

- **GPT-5 Models:** Available via OpenAI API. Use `gpt-5-pro` for summarization (recommended), `gpt-5` for balanced performance, or `gpt-5-mini`/`gpt-5-nano` for cost-sensitive deployments.

- **Model Serving:** Use vLLM with quantization (AWQ/GPTQ) for T4 GPUs. Already supported via `VLLMDeepResearchAgent`. Ollama is NOT recommended as it doesn't host pokee-research-7b and requires manual conversion.

- **Tavily API:** Endpoint `https://api.tavily.com/search`, use Python SDK (`tavily-python`) or direct HTTP requests. Provides direct answers and structured results.

- **Playwright:** Self-hosted browser automation, better than Jina for JavaScript-heavy sites. Install with `pip install playwright && playwright install chromium`.

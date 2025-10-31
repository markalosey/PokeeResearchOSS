# Testing Guide: CLI and Gradio Apps

**Target:** Dell R720 Server  
**Date:** 2025-10-31  
**Status:** Ready for Testing

---

## Prerequisites

✅ **All services must be running:**
- vLLM server: `http://localhost:9999`
- Tool server: `http://localhost:8888`

**Verify services are running:**
```bash
docker compose ps
curl http://localhost:9999/v1/models
curl http://localhost:8888/health
```

---

## Testing CLI App

### Quick Test (Single Query)

**Test a simple question:**
```bash
cd /datapool/PokeeResearchOSS
python3 cli_app.py \
  --serving-mode vllm \
  --vllm-url http://localhost:9999/v1 \
  --question "What is the capital of France?"
```

**Test a research question:**
```bash
python3 cli_app.py \
  --serving-mode vllm \
  --vllm-url http://localhost:9999/v1 \
  --question "What are the latest developments in quantum computing?"
```

**With verbose output:**
```bash
python3 cli_app.py \
  --serving-mode vllm \
  --vllm-url http://localhost:9999/v1 \
  --question "Explain how neural networks work" \
  --verbose
```

### Interactive Mode

**Start interactive CLI:**
```bash
python3 cli_app.py \
  --serving-mode vllm \
  --vllm-url http://localhost:9999/v1 \
  --verbose
```

**In interactive mode:**
- Type your questions and press Enter
- Type `exit` or `quit` to exit
- Press `Ctrl+C` to interrupt

**Example session:**
```
You: What is Python programming used for?
Agent: Researching...
[Searching web...]
[Reading pages...]
Agent: Python is a versatile programming language used for...
Time taken: 15.23 seconds
```

### CLI Options

```bash
python3 cli_app.py --help
```

**Available options:**
- `--serving-mode`: `vllm` (use vLLM server) or `local` (not recommended on server)
- `--vllm-url`: vLLM server URL (default: `http://localhost:9999/v1`)
- `--model-path`: Model name (default: `PokeeAI/pokee_research_7b`)
- `--tool-config`: Tool config path (default: `config/tool_config/pokee_tool_config.yaml`)
- `--question`: Single question (non-interactive mode)
- `--max-turns`: Maximum agent turns (default: 10)
- `--temperature`: Sampling temperature (default: 0.7)
- `--top-p`: Nucleus sampling (default: 0.9)
- `--verbose`: Enable verbose logging

---

## Testing Gradio Web Interface

### Start Gradio App

**Basic start (local access only):**
```bash
cd /datapool/PokeeResearchOSS
python3 gradio_app.py \
  --serving-mode vllm \
  --vllm-url http://localhost:9999/v1 \
  --port 7777
```

**Start with public share link (for remote access):**
```bash
python3 gradio_app.py \
  --serving-mode vllm \
  --vllm-url http://localhost:9999/v1 \
  --port 7777 \
  --share
```

**Access the interface:**
- **Local:** `http://localhost:7777`
- **Network:** `http://<server-ip>:7777`
- **Public share:** Use the share link provided in terminal

### Gradio Interface Features

1. **API Key Configuration**
   - Tavily API Key
   - OpenAI API Key (for GPT-5)
   - Keys are saved to `.env` file

2. **Tool Server Management**
   - Start/Stop tool server
   - View tool server status

3. **Research Interface**
   - Enter your research question
   - Adjust parameters:
     - Temperature (0.0-1.0)
     - Top-p (0.0-1.0)
     - Max turns (1-20)
   - View real-time progress
   - See tool calls (search/read)
   - Get final research answer

4. **Progress Tracking**
   - Real-time updates
   - Tool call visibility
   - Execution time

### Example Research Questions

**Test queries:**
1. "What are the latest developments in AI?"
2. "Explain how transformers work in machine learning"
3. "What is the current state of renewable energy?"
4. "Compare Python and JavaScript programming languages"
5. "What are the benefits of Docker containers?"

---

## Troubleshooting

### Issue: CLI app can't connect to vLLM server

**Check:**
```bash
curl http://localhost:9999/v1/models
```

**Fix:**
- Ensure vLLM container is running: `docker compose ps`
- Check vLLM logs: `docker compose logs vllm-server`
- Verify URL format: `http://localhost:9999/v1` (note `/v1` suffix)

### Issue: CLI app can't connect to tool server

**Check:**
```bash
curl http://localhost:8888/health
```

**Fix:**
- Ensure tool server container is running: `docker compose ps`
- Check tool server logs: `docker compose logs tool-server`
- Verify API keys are set in `.env` file

### Issue: Gradio app won't start

**Check:**
- Python dependencies installed: `pip list | grep gradio`
- Port 7777 available: `netstat -tulpn | grep 7777`
- Python version: `python3 --version` (should be 3.10+)

**Fix:**
- Install dependencies: `pip install gradio`
- Use different port: `--port 7778`
- Check Python version

### Issue: "Module not found" errors

**Fix:**
```bash
cd /datapool/PokeeResearchOSS
pip install -r requirements.txt
```

### Issue: Tool server not starting in Gradio

**Check:**
- Tool server port 8888 is already in use by Docker container
- Gradio tries to start its own tool server

**Fix:**
- Use Docker tool server (already running)
- In Gradio, click "Start Tool Server" only if needed
- Better: Use Docker tool server (recommended)

### Issue: API keys not working

**Check:**
```bash
cat .env | grep -E "TAVILY|OPENAI"
```

**Fix:**
- Verify API keys are correct
- Check API key permissions
- Test keys directly:
  ```bash
  curl -X POST http://localhost:8888/search \
    -H "Content-Type: application/json" \
    -d '{"query": "test", "max_results": 1}'
  ```

---

## Expected Performance

### CLI App
- **Simple queries:** 5-15 seconds
- **Research queries:** 30-90 seconds
- **Tool calls:** 2-5 per query (typical)

### Gradio App
- **Startup time:** 5-10 seconds
- **Query response:** Same as CLI
- **UI updates:** Real-time (streaming)

---

## Testing Checklist

- [ ] **CLI Single Query Test**
  - [ ] Simple question works
  - [ ] Research question works
  - [ ] Verbose mode shows tool calls

- [ ] **CLI Interactive Mode**
  - [ ] Can enter questions
  - [ ] Gets responses
  - [ ] Can exit cleanly

- [ ] **Gradio Interface**
  - [ ] Starts successfully
  - [ ] Can configure API keys
  - [ ] Can start research
  - [ ] Progress updates in real-time
  - [ ] Gets research results

- [ ] **End-to-End Research**
  - [ ] Search tool works
  - [ ] Read tool works
  - [ ] GPT-5 summarization works
  - [ ] Final answer is coherent

---

## Next Steps After Testing

1. **Performance Optimization**
   - Monitor GPU utilization
   - Adjust max_turns if needed
   - Tune temperature/top_p

2. **Production Readiness**
   - Set up reverse proxy (nginx)
   - Configure SSL/TLS
   - Set up monitoring
   - Configure log rotation

3. **Documentation**
   - Document any issues found
   - Update deployment plan
   - Create user guide

---

**Last Updated:** 2025-10-31  
**Status:** Ready for Testing


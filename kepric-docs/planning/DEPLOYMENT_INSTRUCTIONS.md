# Deployment Instructions - Dell R720 Server

**Date:** 2025-01-27  
**Target:** Dell R720 with 2x NVIDIA T4 GPUs running Debian 12  
**Purpose:** Complete step-by-step deployment guide

---

## Prerequisites Checklist

Before starting, ensure you have:
- [ ] SSH access to the Dell R720 server (user: `mlosey`)
- [ ] Your GitHub username: `markalosey`
- [ ] Tavily API key (from https://tavily.com)
- [ ] OpenAI API key (from https://platform.openai.com/api-keys)
- [ ] HuggingFace token (if using private models, from https://huggingface.co/settings/tokens)

---

## Step 1: SSH to the Dell R720 Server

```bash
ssh mlosey@<your-server-ip-or-hostname>
```

**Expected Result:** You should be logged into the server and see a prompt like:
```
mlosey@r720-zfs:~$
```

---

## Step 2: Navigate to Deployment Directory

Choose ONE of the following locations:

**Option A: Use /datapool (Recommended - 1.7TB available)**
```bash
cd /datapool
```

**Option B: Use /opt/pokee-research**
```bash
sudo mkdir -p /opt/pokee-research
sudo chown $USER:$USER /opt/pokee-research
cd /opt/pokee-research
```

**Verify:** Run `pwd` to confirm you're in the correct directory.

---

## Step 3: Clone Your Forked Repository

```bash
git clone https://github.com/markalosey/PokeeResearchOSS.git
```

**Expected Result:** You should see:
```
Cloning into 'PokeeResearchOSS'...
remote: Enumerating objects: ...
remote: Counting objects: ...
...
Receiving objects: 100% (...), done.
```

**Navigate into the repository:**
```bash
cd PokeeResearchOSS
```

**Verify:** Run `git branch` and ensure you see `migration/tavily-playwright-gpt5`

**Switch to the migration branch:**
```bash
git checkout migration/tavily-playwright-gpt5
```

**Verify:** Run `git branch` again and confirm `* migration/tavily-playwright-gpt5` is shown.

---

## Step 4: Create Environment Variables File

**Create `.env` file:**
```bash
cat > .env << 'EOF'
# Tavily API - Required for web search functionality
TAVILY_API_KEY=your_tavily_api_key_here

# OpenAI API - Required for GPT-5 content summarization
OPENAI_API_KEY=your_openai_api_key_here

# OpenAI Model Selection (optional, defaults to gpt-5-pro)
OPENAI_MODEL=gpt-5-pro

# HuggingFace Token - Required for model download
HUGGINGFACE_TOKEN=your_hf_token_here

# vLLM Server Configuration (optional)
MODEL=PokeeAI/pokee_research_7b
PORT=9999
QUANTIZATION=awq
GPU_MEMORY_UTILIZATION=0.45
MAX_MODEL_LEN=32768

# Tool Server Configuration (optional)
TOOL_SERVER_PORT=8888
EOF
```

**Edit the `.env` file with your actual API keys:**
```bash
nano .env
```

**Replace the following placeholders:**
- `your_tavily_api_key_here` → Your actual Tavily API key
- `your_openai_api_key_here` → Your actual OpenAI API key
- `your_hf_token_here` → Your actual HuggingFace token (if needed)

**Save and exit nano:** Press `Ctrl+X`, then `Y`, then `Enter`

**Verify `.env` file:**
```bash
cat .env | grep -v "your_.*_here" | grep "="
```

**Expected Result:** You should see your actual API keys (partially masked).

**Verify `.env` is NOT tracked by git:**
```bash
git check-ignore .env
```

**Expected Result:** Should output `.env` (meaning it's ignored, which is correct).

---

## Step 5: Verify Docker and GPU Access

**Verify Docker is running:**
```bash
docker ps
```

**Expected Result:** Should show existing containers (ollama, open-webui, etc.) or empty list.

**Verify GPU access:**
```bash
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

**Expected Result:** Should show both NVIDIA T4 GPUs.

**Verify Docker Compose:**
```bash
docker compose version
```

**Expected Result:** Should show `Docker Compose version v2.38.2` (or similar).

---

## Step 6: Build Docker Images

**Build vLLM server image (this will take 10-30 minutes):**
```bash
docker compose build vllm-server
```

**Expected Result:** You should see:
```
[+] Building ... 
...
Successfully built <image-id>
Successfully tagged pokee-research-vllm-server:latest
```

**Build tool server image (this will take 5-15 minutes):**
```bash
docker compose build tool-server
```

**Expected Result:** You should see:
```
[+] Building ...
...
Successfully built <image-id>
Successfully tagged pokee-research-tool-server:latest
```

**Note:** The first build will take longer as it downloads base images and installs dependencies.

---

## Step 7: Verify Images Were Created

```bash
docker images | grep pokee
```

**Expected Result:** Should show:
```
pokee-research-vllm-server   latest    <id>    <time>    <size>
pokee-research-tool-server   latest    <id>    <time>    <size>
```

---

## Step 8: Start Services with Docker Compose

**Start both services:**
```bash
docker compose up -d
```

**Expected Result:** You should see:
```
[+] Running 2/2
 ✔ Container pokee-vllm          Started
 ✔ Container pokee-tool-server   Started
```

**Check service status:**
```bash
docker compose ps
```

**Expected Result:** Should show both services as `Up`:
```
NAME                  IMAGE                           STATUS
pokee-vllm           pokee-research-vllm-server      Up
pokee-tool-server    pokee-research-tool-server      Up
```

---

## Step 9: Monitor Service Logs

**View vLLM server logs:**
```bash
docker compose logs -f vllm-server
```

**What to look for:**
- GPU detection messages
- Model loading progress
- "Uvicorn running on http://0.0.0.0:9999"

**Press `Ctrl+C` to exit log viewer**

**View tool server logs:**
```bash
docker compose logs -f tool-server
```

**What to look for:**
- "Tool Server Starting"
- "Server ready at http://0.0.0.0:8888"
- No API key errors

**Press `Ctrl+C` to exit log viewer**

---

## Step 10: Verify Services Are Running

**Check vLLM server health:**
```bash
curl -f http://localhost:9999/health || echo "Health check endpoint may not exist, checking if server responds"
curl http://localhost:9999/v1/models
```

**Expected Result:** Should return JSON with available models or model information.

**Check tool server health:**
```bash
curl -f http://localhost:8888/health || echo "Health check endpoint may not exist, checking if server responds"
curl http://localhost:8888/
```

**Expected Result:** Should return a response (may be JSON or HTML).

**Check GPU utilization:**
```bash
nvidia-smi
```

**Expected Result:** Should show GPU memory usage for the vLLM container.

---

## Step 11: Test Tool Server Endpoints

**Test Tavily search:**
```bash
curl -X POST http://localhost:8888/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Python programming language"}'
```

**Expected Result:** Should return JSON with search results from Tavily.

**Test Playwright read (if search worked):**
```bash
# First, get a URL from the search results, then:
curl -X POST http://localhost:8888/read \
  -H "Content-Type: application/json" \
  -d '{"url": "https://python.org", "question": "What is Python?"}'
```

**Expected Result:** Should return JSON with page content and GPT-5 summary.

---

## Step 12: Verify Container Logs for Errors

**Check for any errors in vLLM logs:**
```bash
docker compose logs vllm-server | grep -i error
```

**Check for any errors in tool server logs:**
```bash
docker compose logs tool-server | grep -i error
```

**Expected Result:** Should show no critical errors (warnings are okay).

---

## Step 13: Verify Network Connectivity

**Test Tavily API from container:**
```bash
docker exec pokee-tool-server curl -I https://api.tavily.com
```

**Expected Result:** Should return HTTP headers (405 is expected for HEAD request).

**Test OpenAI API from container:**
```bash
docker exec pokee-tool-server curl -I https://api.openai.com
```

**Expected Result:** Should return HTTP headers (connection successful).

---

## Step 14: Stop Services (Optional - for testing)

**Stop all services:**
```bash
docker compose down
```

**Expected Result:** Should show:
```
[+] Running 2/2
 ✔ Container pokee-tool-server   Stopped
 ✔ Container pokee-vllm          Stopped
```

**Restart services:**
```bash
docker compose up -d
```

---

## Step 15: Configure Auto-Start (Optional)

**Create systemd service (if you want services to start on boot):**

Create the service file:
```bash
sudo nano /etc/systemd/system/pokee-research.service
```

Add the following content:
```ini
[Unit]
Description=Pokee Research Agent Docker Compose
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/datapool/PokeeResearchOSS
ExecStart=/usr/bin/docker compose up -d
ExecStop=/usr/bin/docker compose down
User=mlosey
Group=docker

[Install]
WantedBy=multi-user.target
```

**Enable the service:**
```bash
sudo systemctl daemon-reload
sudo systemctl enable pokee-research.service
```

**Start the service:**
```bash
sudo systemctl start pokee-research.service
```

**Check status:**
```bash
sudo systemctl status pokee-research.service
```

---

## Troubleshooting

### Issue: Docker build fails

**Check disk space:**
```bash
df -h /datapool
```

**Clean up Docker:**
```bash
docker system prune -a
```

### Issue: GPU not accessible

**Verify NVIDIA runtime:**
```bash
docker info | grep -i nvidia
```

**Restart Docker:**
```bash
sudo systemctl restart docker
```

### Issue: Port conflicts

**Check what's using ports:**
```bash
sudo netstat -tulpn | grep -E "8888|9999"
```

**Modify ports in docker-compose.yml if needed**

### Issue: API key errors

**Verify environment variables:**
```bash
docker exec pokee-tool-server env | grep -E "TAVILY|OPENAI"
```

**Check .env file:**
```bash
cat .env
```

---

## Verification Checklist

- [ ] Repository cloned successfully
- [ ] Branch `migration/tavily-playwright-gpt5` checked out
- [ ] `.env` file created with actual API keys
- [ ] Docker images built successfully
- [ ] Both containers started (`docker compose ps` shows Up)
- [ ] vLLM server responding on port 9999
- [ ] Tool server responding on port 8888
- [ ] GPU visible in `nvidia-smi` for vLLM container
- [ ] Tavily search endpoint working
- [ ] Playwright read endpoint working
- [ ] No critical errors in logs

---

## Next Steps After Deployment

1. **Test the agent via Gradio (if running locally):**
   ```bash
   python gradio_app.py --serving-mode vllm --vllm-url http://localhost:9999/v1
   ```

2. **Monitor resource usage:**
   ```bash
   watch -n 1 nvidia-smi
   ```

3. **View real-time logs:**
   ```bash
   docker compose logs -f
   ```

---

## Quick Reference Commands

**View all container status:**
```bash
docker compose ps
```

**View logs:**
```bash
docker compose logs -f [service-name]
```

**Restart a service:**
```bash
docker compose restart [service-name]
```

**Stop all services:**
```bash
docker compose down
```

**Start all services:**
```bash
docker compose up -d
```

**Rebuild and restart:**
```bash
docker compose up -d --build
```

**Access container shell:**
```bash
docker exec -it pokee-tool-server /bin/bash
docker exec -it pokee-vllm /bin/bash
```

---

**Last Updated:** 2025-01-27  
**Status:** Ready for deployment


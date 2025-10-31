# Server Discovery Results - Dell R720

**Date:** 2025-01-27  
**Purpose:** Document current server state before deployment  
**Target:** Dell R720 with 2x NVIDIA T4 GPUs

---

## Operating System

**Question 1:** What operating system and version is running?

**Answer:**
- **OS:** Debian GNU/Linux 12 (bookworm)
- **Version ID:** 12
- **Codename:** bookworm
- **ID:** debian

**Status:** ✅ Debian 12 (bookworm) - Compatible with Docker and NVIDIA Container Toolkit

---

## GPU Detection

**Question 2:** Are NVIDIA T4 GPUs detected?

**Command:** `lspci | grep -i nvidia`

**Answer:**
- **GPU 1:** PCI 04:00.0 - NVIDIA Corporation TU104GL [Tesla T4] (rev a1)
- **GPU 2:** PCI 42:00.0 - NVIDIA Corporation TU104GL [Tesla T4] (rev a1)

**Status:** ✅ Both NVIDIA T4 GPUs detected successfully

---

## NVIDIA Drivers

**Question 3:** Are NVIDIA drivers installed?

**Command:** `nvidia-smi`

**Answer:**
- **Driver Version:** 535.247.01
- **CUDA Version:** 12.2
- **GPU 0:** Tesla T4 (PCI 04:00.0) - 5MB/15360MB memory, 0% utilization, 34°C, idle
- **GPU 1:** Tesla T4 (PCI 42:00.0) - 5MB/15360MB memory, 0% utilization, 34°C, idle
- **Status:** No processes running (clean state)

**Status:** ✅ NVIDIA drivers installed and working perfectly. Both GPUs idle and ready.

---

## Docker Installation

**Question 4:** Is Docker installed?

**Command:** `docker --version`

**Answer:**
- **Docker Version:** 28.3.2 (build 578ccf6)

**Status:** ✅ Docker is installed and up to date

---

## Docker Compose

**Question 5:** Is Docker Compose installed?

**Command:** `docker compose version`

**Answer:**
- **Docker Compose Version:** v2.38.2

**Status:** ✅ Docker Compose is installed (v2 plugin version)

---

## NVIDIA Container Toolkit

**Question 6:** Is NVIDIA Container Toolkit installed?

**Command:** `nvidia-ctk --version`

**Answer:**
- **NVIDIA Container Toolkit CLI Version:** 1.17.8
- **Commit:** f202b80a9b9d0db00d9b1d73c0128c8962c55f4d

**Status:** ✅ NVIDIA Container Toolkit is installed

---

## Docker NVIDIA Runtime Configuration

**Question 7:** Is Docker configured to use the NVIDIA runtime?

**Command:** `docker info | grep -i nvidia`

**Answer:**
- **Runtimes:** `io.containerd.runc.v2 nvidia runc` (NVIDIA runtime is available)

**Status:** ✅ Docker is configured with NVIDIA runtime - GPU access enabled

---

## Docker GPU Access Test

**Question 8:** Can Docker containers access the GPUs?

**Command:** `docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi`

**Answer:**
- ✅ Container successfully accessed both GPUs
- Both Tesla T4 GPUs visible from within container
- GPU 0: 5MB/15360MB memory, 0% utilization
- GPU 1: 5MB/15360MB memory, 0% utilization
- Image pulled successfully: `nvidia/cuda:12.2.0-base-ubuntu22.04`

**Status:** ✅ Docker GPU access is fully functional - Ready for vLLM deployment

---

## Git Installation

**Question 9:** Is Git installed?

**Command:** `git --version`

**Answer:**
- **Git Version:** 2.39.5

**Status:** ✅ Git is installed - Ready to clone repository

---

## Disk Space

**Question 10:** What is available disk space?

**Command:** `df -h /`

**Answer:**
- **Filesystem:** ZFS (rpool/ROOT/debian)
- **Total Size:** 900GB
- **Used:** 12GB (2%)
- **Available:** 888GB
- **Mount Point:** /

**Status:** ✅ Plenty of disk space available (888GB free) - More than sufficient for Docker images and model storage

---

## Memory

**Question 11:** What is available memory?

**Command:** `free -h`

**Answer:**
- **Total Memory:** 251GB
- **Used:** 108GB
- **Free:** 86GB
- **Available:** 143GB
- **Buff/Cache:** 59GB
- **Swap:** None configured (0B)

**Status:** ✅ Excellent memory availability (143GB available) - More than sufficient for containers and model operations

---

## Network Connectivity

**Question 12:** Can we reach external APIs?

**Commands:** 
- `curl -I https://api.tavily.com`
- `curl -I https://api.openai.com`

**Answer - Tavily API:**
- ✅ **Reachable:** HTTP/2 405 (Method Not Allowed - expected for HEAD request)
- Server: uvicorn
- Date: Fri, 31 Oct 2025 02:02:31 GMT
- **Status:** Tavily API is accessible from server

**Answer - OpenAI API:**
- ✅ **Reachable:** HTTP/2 421 (Misdirected Request - expected for HEAD request)
- Server: cloudflare
- Date: Fri, 31 Oct 2025 02:03:04 GMT
- CF-Ray: 996f92e88a2cf865-ORD
- **Status:** OpenAI API is accessible from server

**Note:** Separate ZFS datapool mounted (not on /) - Will need to verify mount point for Docker volumes

---

## ZFS Datapool Mount Point

**Question 13-14:** Where is the separate ZFS datapool mounted?

**Commands:** 
- `df -h | grep datapool`
- `mount | grep datapool`

**Answer:**
- **Mount Point:** `/datapool`
- **Filesystem:** ZFS (datapool)
- **Total Size:** 1.8TB
- **Used:** 133GB (8%)
- **Available:** 1.7TB
- **Mount Options:** rw, xattr, noacl

**Status:** ✅ Datapool available at `/datapool` with 1.7TB free - Excellent for Docker volumes and model storage

---

## Python Installation

**Question 15:** What Python version is installed?

**Command:** `python3 --version`

**Answer:**
- **Python Version:** 3.11.2

**Status:** ✅ Python 3.11.2 installed - Compatible with all required packages

---

## Docker Group Membership

**Question 16:** Can you run Docker commands without sudo?

**Command:** `groups`

**Answer:**
- **User Groups:** mlosey sudo users docker
- **Docker Group:** ✅ User is member of docker group

**Status:** ✅ Can run Docker commands without sudo - Ready for deployment

---

## Existing Docker Containers

**Question 17:** Are there any existing Docker containers running?

**Command:** `docker ps -a`

**Answer:**
Existing containers:
- **open-webui** - Port 8080 (healthy, running 2 months)
- **ollama** - Port 11434 (running 2 months)
- **nginx-proxy-manager** - Ports 80-81, 443 (healthy, running 2 months)
- **gitea** - Ports 3000, 2222 (running 2 months)
- **npm-db** (MariaDB) - No exposed ports (healthy, running 2 months)
- **postgres** - Port 5432 (running 2 months)

**Port Conflicts Check:**
- Port 8888 (tool-server): ✅ Available
- Port 9999 (vLLM): ✅ Available
- Port 7777 (Gradio): ✅ Available

**Status:** ✅ No port conflicts - All required ports are available

---

## Repository Deployment

**Question 18:** Has the repository been cloned and deployed?

**Answer:**
- **Repository Location:** `/datapool/PokeeResearchOSS`
- **Branch:** `migration/tavily-playwright-gpt5`
- **Clone Status:** ✅ Successfully cloned from GitHub fork
- **Working Directory:** `/datapool/PokeeResearchOSS`

**Status:** ✅ Repository deployed and ready

---

## Python Virtual Environment

**Question 19:** Is Python virtual environment configured?

**Answer:**
- **Python Version:** 3.11.2
- **venv Package:** ✅ `python3.11-venv` installed via apt
- **Virtual Environment:** ✅ Created at `/datapool/PokeeResearchOSS/.venv`
- **Activation:** `source .venv/bin/activate`
- **Dependencies:** To be installed via `pip install -r requirements.txt`

**Status:** ✅ Virtual environment ready for CLI and Gradio apps

---

## Docker Services Deployment

**Question 20:** Are Docker services running?

**Command:** `docker compose ps`

**Answer:**

### vLLM Server
- **Container:** `pokee-vllm`
- **Image:** `pokeeresearchoss-vllm-server:latest`
- **Status:** ✅ Healthy (Up and running)
- **Port:** 9999 (0.0.0.0:9999->9999/tcp)
- **GPU Access:** ✅ Both GPUs accessible (0,1)
- **Health Check:** ✅ Passing

### Tool Server
- **Container:** `pokee-tool-server`
- **Image:** `pokeeresearchoss-tool-server:latest`
- **Status:** ✅ Healthy (Up and running)
- **Port:** 8888 (0.0.0.0:8888->8888/tcp)
- **Health Check:** ✅ Passing

**Status:** ✅ Both services deployed and healthy

---

## vLLM Server Configuration

**Question 21:** What is the vLLM server configuration?

**Answer:**
- **Model:** `PokeeAI/pokee_research_7b`
- **Tensor Parallelism:** ✅ Enabled (`tensor_parallel_size=2`)
- **GPU Memory Utilization:** 0.60 (60% per GPU)
- **Max Model Length:** 2048 tokens
- **Quantization:** None (FP16)
- **Enforce Eager:** ✅ Enabled (disables torch.compile to prevent OOM)
- **GPUs Used:** Both T4 GPUs (split model across GPUs)
- **Memory per GPU:** ~9.4GB used (model weights: ~7.12GB per GPU)

**NCCL Configuration:**
- **NCCL_IB_DISABLE:** 1 (InfiniBand disabled)
- **NCCL_P2P_DISABLE:** 0 (P2P enabled)
- **NCCL_DEBUG:** WARN
- **IPC:** Host namespace enabled
- **Shared Memory:** 1GB

**Docker Configuration:**
- **Base Image:** `nvidia/cuda:12.2.0-devel-ubuntu22.04`
- **Python:** 3.10
- **vLLM Version:** 0.11.0
- **IPC Mode:** `host`
- **Shared Memory:** `1gb`

**Status:** ✅ Optimized for dual T4 GPUs with tensor parallelism

---

## Tool Server Configuration

**Question 22:** What is the tool server configuration?

**Answer:**
- **Base Image:** `python:3.10-slim`
- **Port:** 8888
- **API Integration:**
  - ✅ Tavily API (web search)
  - ✅ Playwright (web reading)
  - ✅ OpenAI GPT-5 (summarization)
- **Cache:** Enabled
- **Max Concurrent Requests:**
  - Search: 300
  - Read: 500
- **Timeout:** 30 seconds

**Status:** ✅ Fully configured with new API integrations

---

## GPU Utilization

**Question 23:** What is the current GPU utilization?

**Command:** `docker exec pokee-vllm nvidia-smi`

**Answer:**
- **GPU 0:** 
  - Memory: 9390MiB / 15360MiB (~61% used)
  - Utilization: 0% (idle, ready for requests)
  - Temperature: 59°C
  - Power: 29W / 70W
- **GPU 1:**
  - Memory: 9390MiB / 15360MiB (~61% used)
  - Utilization: 0% (idle, ready for requests)
  - Temperature: 58°C
  - Power: 30W / 70W

**Status:** ✅ Model loaded successfully, both GPUs at ~61% memory usage

---

## Service Validation

**Question 24:** Have services been validated?

**Answer:**

### vLLM Server Tests
- ✅ Models endpoint: `curl http://localhost:9999/v1/models` - Working
- ✅ Chat completions: Successfully tested with "Hello, how are you?"
- ✅ Health check: Passing

### Tool Server Tests
- ✅ Health endpoint: `curl http://localhost:8888/health` - Working
- ✅ Search endpoint: Successfully tested with Tavily API
- ✅ Read endpoint: Successfully tested with Playwright
- ✅ Both agents operational (search and read)

**Status:** ✅ All services validated and working correctly

---

## Deployment Issues Resolved

**Question 25:** What issues were encountered and resolved during deployment?

**Answer:**

1. **CUDA Base Image:**
   - Issue: `nvidia/cuda:12.2.0-cudnn8-devel-ubuntu22.04` not found
   - Resolution: Changed to `nvidia/cuda:12.2.0-devel-ubuntu22.04`

2. **vLLM Token Flag:**
   - Issue: Ambiguous `--token` flag error
   - Resolution: Removed flag (vLLM uses `HF_TOKEN` environment variable)

3. **AWQ Quantization:**
   - Issue: AWQ config not found
   - Resolution: Made quantization optional, default to `none`

4. **CUDA Out of Memory:**
   - Issue: OOM during model compilation/profiling
   - Resolution: 
     - Added `--enforce-eager` to disable torch.compile
     - Enabled tensor parallelism (`--tensor-parallel-size 2`)
     - Adjusted memory settings (0.60 utilization, 2048 max length)
     - Added NCCL configuration for multi-GPU communication

5. **NCCL Communication:**
   - Issue: NCCL errors in tensor parallelism
   - Resolution: Added Docker IPC host mode and shared memory

6. **Docker Compose Version Warning:**
   - Issue: Obsolete `version` attribute warning
   - Resolution: Removed version field (Docker Compose v2 doesn't require it)

**Status:** ✅ All deployment issues resolved

---

## Environment Variables

**Question 26:** What environment variables are configured?

**Answer:**

### vLLM Server (.env)
- `MODEL=PokeeAI/pokee_research_7b`
- `QUANTIZATION=none`
- `GPU_MEMORY_UTILIZATION=0.60`
- `MAX_MODEL_LEN=2048`
- `HUGGINGFACE_TOKEN=<set>`

### Tool Server (.env)
- `TAVILY_API_KEY=<set>`
- `OPENAI_API_KEY=<set>`
- `OPENAI_MODEL=gpt-5-pro`

**Status:** ✅ All required environment variables configured

---

## Discovery Summary

### ✅ Deployment Complete

**Operating System:**
- Debian 12 (bookworm) ✅

**Hardware:**
- 2x NVIDIA Tesla T4 GPUs detected ✅
- Both GPUs actively used (tensor parallelism) ✅
- 251GB RAM (143GB available) ✅
- 900GB root filesystem (888GB free) ✅
- 1.8TB datapool at `/datapool` (1.7TB free) ✅

**Software Stack:**
- Docker 28.3.2 ✅
- Docker Compose v2.38.2 ✅
- NVIDIA Drivers 535.247.01 (CUDA 12.2) ✅
- NVIDIA Container Toolkit 1.17.8 ✅
- Docker GPU access verified ✅
- Git 2.39.5 ✅
- Python 3.11.2 ✅
- Python venv package installed ✅

**Network:**
- Tavily API accessible ✅
- OpenAI API accessible ✅

**Permissions:**
- User in docker group (no sudo needed) ✅

**Ports:**
- 8888 (tool-server): ✅ In use (healthy)
- 9999 (vLLM): ✅ In use (healthy)
- 7777 (Gradio): ✅ Available for web interface

**Deployment:**
- Repository cloned at `/datapool/PokeeResearchOSS` ✅
- Branch: `migration/tavily-playwright-gpt5` ✅
- Virtual environment created at `.venv` ✅
- Docker services running and healthy ✅
- vLLM server configured with tensor parallelism ✅
- Tool server configured with Tavily/Playwright/GPT-5 ✅
- All services validated ✅

**Configuration:**
- Tensor parallelism: 2 GPUs ✅
- Memory optimization: 0.60 utilization, 2048 max length ✅
- NCCL communication configured ✅
- IPC and shared memory configured ✅

### 🎯 Current Status

**Production Ready:**
- ✅ All Docker services deployed and healthy
- ✅ GPU utilization optimized for dual T4 GPUs
- ✅ API integrations tested and working
- ✅ Virtual environment ready for CLI/Gradio apps
- ✅ All validation tests passed

**Next Steps:**
1. Test CLI app (with virtual environment)
2. Test Gradio web interface
3. Monitor performance and GPU utilization
4. Set up production monitoring and logging

### 📝 Notes

- Separate ZFS datapool used at `/datapool` for repository and Docker volumes
- Existing containers running (Ollama, Open WebUI, etc.) - no conflicts
- Tensor parallelism successfully enabled for dual GPU setup
- Memory settings optimized for T4 GPUs (15GB each)
- All deployment issues resolved and documented

---

**Last Updated:** 2025-10-31  
**Status:** ✅ Deployment complete - All services operational and validated


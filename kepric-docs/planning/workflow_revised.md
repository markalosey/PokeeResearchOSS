# Revised Deployment Workflow

**Workflow:** Local Development → Commit/Push → Clone on Server → Deploy  
**Purpose:** Make all modifications locally, then deploy to Dell R720  
**Date:** 2025-01-27

---

## Workflow Overview

```
┌─────────────────────────────────────────────────────────────┐
│  LOCAL DEVELOPMENT (Your Machine)                           │
│  1. Clone/Pull repository                                   │
│  2. Create migration branch                                 │
│  3. Make all code modifications                             │
│  4. Create Docker files                                     │
│  5. Test locally (if possible)                              │
│  6. Commit and push to fork                                 │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
                    [GitHub Fork]
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  SERVER DEPLOYMENT (Dell R720)                              │
│  1. Clone repository                                        │
│  2. Checkout migration branch                               │
│  3. Configure environment (.env)                            │
│  4. Build Docker images                                     │
│  5. Deploy with docker-compose                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Local Development Setup

### 1.1: Verify Local Environment

- [ ] **1.1.1** Verify Git is installed
  ```bash
  git --version
  ```

- [ ] **1.1.2** Verify Python 3.8+ is installed
  ```bash
  python3 --version
  ```

- [ ] **1.1.3** Verify GitHub access
  ```bash
  git config --global user.name "Your Name"
  git config --global user.email "your.email@example.com"
  ```

### 1.2: Clone Repository Locally

- [ ] **1.2.1** Clone your fork (if not already cloned)
  ```bash
  cd ~/workspace  # or your preferred directory
  git clone https://github.com/YOUR_USERNAME/PokeeResearchOSS.git
  cd PokeeResearchOSS
  ```

- [ ] **1.2.2** Verify remote configuration
  ```bash
  git remote -v
  # Should show your fork as origin
  ```

- [ ] **1.2.3** Create migration branch
  ```bash
  git checkout -b migration/tavily-playwright-gpt5
  git branch  # Verify you're on the new branch
  ```

- [ ] **1.2.4** Verify repository structure
  ```bash
  ls -la tool_server/
  ls -la agent/
  ls -la config/
  ls -la start_tool_server.py
  ```

---

## Phase 2: Local Code Modifications

### 2.1: Replace Serper with Tavily

- [ ] **2.1.1** Follow all tasks from `deployment_plan.md` Phase 3 (Task 3.1-3.2)
  - Modify `tool_server/search.py`
  - Update environment variable references
  - Test locally if possible

### 2.2: Replace Jina with Playwright

- [ ] **2.2.1** Follow all tasks from `deployment_plan.md` Phase 4 (Task 4.1-4.2)
  - Modify `tool_server/read.py`
  - Remove Jina API key references

### 2.3: Replace Gemini with GPT-5

- [ ] **2.3.1** Follow all tasks from `deployment_plan.md` Phase 5 (Task 5.1-5.3)
  - Modify `tool_server/utils.py`
  - Update `tool_server/read.py` to use new OpenAI client

### 2.4: Update Dependencies

- [ ] **2.4.1** Follow all tasks from `deployment_plan.md` Phase 6 (Task 6.1-6.2)
  - Update `requirements.txt`
  - Create Playwright installation script

### 2.5: Update Configuration

- [ ] **2.5.1** Follow all tasks from `deployment_plan.md` Phase 7 (Task 7.1-7.2)
  - Update configuration files
  - Create `.env.example` template

---

## Phase 3: Docker Setup (Local)

### 3.1: Create Dockerfiles

- [ ] **3.1.1** Follow all tasks from `deployment_plan.md` Phase 8 (Task 8.1-8.3)
  - Create `Dockerfile.vllm`
  - Create `Dockerfile.tool-server`
  - Create `Dockerfile.agent` (optional)

### 3.2: Create Docker Compose

- [ ] **3.2.1** Follow all tasks from `deployment_plan.md` Phase 9 (Task 9.1-9.3)
  - Create `docker-compose.yml`
  - Create `.env.example`
  - Create `.dockerignore`

---

## Phase 4: Local Validation

### 4.1: Code Validation

- [ ] **4.1.1** Syntax checks
  ```bash
  python3 -m py_compile tool_server/search.py
  python3 -m py_compile tool_server/read.py
  python3 -m py_compile tool_server/utils.py
  ```

- [ ] **4.1.2** Import checks
  ```bash
  python3 -c "from tool_server.search import tavily_search, WebSearchAgent"
  python3 -c "from tool_server.read import playwright_read, WebReadAgent"
  python3 -c "from tool_server.utils import get_openai_client, llm_summary"
  ```

- [ ] **4.1.3** Follow validation tasks from `deployment_plan.md` Phase 10
  - Use `validation_reference.md` for validation commands

### 4.2: Docker Validation (Optional)

- [ ] **4.2.1** Validate docker-compose.yml syntax
  ```bash
  docker compose config
  ```

- [ ] **4.2.2** Check Dockerfiles for syntax errors
  ```bash
  # Review Dockerfiles manually
  ```

---

## Phase 5: Commit and Push

### 5.1: Review Changes

- [ ] **5.1.1** Check what files have changed
  ```bash
  git status
  ```

- [ ] **5.1.2** Review modified files
  ```bash
  git diff
  # Review each file carefully
  ```

- [ ] **5.1.3** Verify all modifications are complete
  - [ ] All code changes done
  - [ ] All Docker files created
  - [ ] All configuration files updated
  - [ ] `.env.example` created (without real keys)

### 5.2: Stage Changes

- [ ] **5.2.1** Stage all modified files
  ```bash
  git add tool_server/search.py
  git add tool_server/read.py
  git add tool_server/utils.py
  git add requirements.txt
  git add Dockerfile.*
  git add docker-compose.yml
  git add .env.example
  git add .dockerignore
  git add scripts/
  git add kepric-docs/  # Documentation updates
  ```

- [ ] **5.2.2** Verify staged files
  ```bash
  git status
  ```

- [ ] **5.2.3** Ensure `.env` is NOT staged (should be in .gitignore)
  ```bash
  git check-ignore .env  # Should output: .env
  ```

### 5.3: Commit Changes

- [ ] **5.3.1** Create commit with descriptive message
  ```bash
  git commit -m "Migrate to Tavily, Playwright, and GPT-5

  - Replace Serper API with Tavily API (tool_server/search.py)
  - Replace Jina API with Playwright browser automation (tool_server/read.py)
  - Replace Gemini API with GPT-5 OpenAI API (tool_server/utils.py)
  - Update requirements.txt with new dependencies
  - Add Dockerfiles for vLLM and tool-server
  - Add docker-compose.yml for deployment
  - Add .env.example template
  - Update documentation"
  ```

- [ ] **5.3.2** Verify commit
  ```bash
  git log -1
  git show --stat
  ```

### 5.4: Push to Fork

- [ ] **5.4.1** Push branch to GitHub
  ```bash
  git push origin migration/tavily-playwright-gpt5
  ```

- [ ] **5.4.2** Verify push succeeded
  ```bash
  git log origin/migration/tavily-playwright-gpt5
  # Or check on GitHub web interface
  ```

- [ ] **5.4.3** Create Pull Request (optional, for review)
  - Go to GitHub: `https://github.com/YOUR_USERNAME/PokeeResearchOSS`
  - Create PR from `migration/tavily-playwright-gpt5` to `main` (or your base branch)

---

## Phase 6: Server Deployment

### 6.1: Server Verification (Already Done ✅)

From `server_discovery.md`, we know:
- ✅ Docker installed and configured
- ✅ NVIDIA drivers and Container Toolkit working
- ✅ GPU access verified
- ✅ Network connectivity confirmed
- ✅ Ports available

### 6.2: Clone Repository on Server

- [ ] **6.2.1** Choose deployment location
  ```bash
  # Option A: Use /datapool (recommended - 1.7TB available)
  cd /datapool
  mkdir -p pokee-research
  cd pokee-research
  
  # Option B: Use /opt/pokee-research
  # cd /opt/pokee-research
  ```

- [ ] **6.2.2** Clone repository
  ```bash
  git clone https://github.com/YOUR_USERNAME/PokeeResearchOSS.git
  cd PokeeResearchOSS
  ```

- [ ] **6.2.3** Checkout migration branch
  ```bash
  git checkout migration/tavily-playwright-gpt5
  git branch  # Verify correct branch
  ```

- [ ] **6.2.4** Verify all files are present
  ```bash
  ls -la tool_server/
  ls -la Dockerfile.*
  ls -la docker-compose.yml
  ls -la .env.example
  ```

### 6.3: Configure Environment

- [ ] **6.3.1** Create .env file from template
  ```bash
  cp .env.example .env
  ```

- [ ] **6.3.2** Edit .env with actual API keys
  ```bash
  nano .env  # or vim .env
  ```

- [ ] **6.3.3** Set required variables:
  ```bash
  TAVILY_API_KEY=your_actual_tavily_key
  OPENAI_API_KEY=your_actual_openai_key
  OPENAI_MODEL=gpt-5-pro
  HUGGINGFACE_TOKEN=your_actual_hf_token
  VLLM_URL=http://localhost:9999/v1
  ```

- [ ] **6.3.4** Verify .env file (don't commit!)
  ```bash
  cat .env | grep -v "KEY\|TOKEN"  # Show non-sensitive values
  ```

### 6.4: Build Docker Images

- [ ] **6.4.1** Build vLLM server image
  ```bash
  docker compose build vllm-server
  ```

- [ ] **6.4.2** Build tool-server image
  ```bash
  docker compose build tool-server
  ```

- [ ] **6.4.3** Build agent image (if applicable)
  ```bash
  docker compose build agent
  ```

- [ ] **6.4.4** Verify images created
  ```bash
  docker images | grep pokee
  ```

### 6.5: Deploy Services

- [ ] **6.5.1** Start services
  ```bash
  docker compose up -d
  ```

- [ ] **6.5.2** Monitor startup logs
  ```bash
  docker compose logs -f
  # Wait for services to start, then Ctrl+C
  ```

- [ ] **6.5.3** Check service status
  ```bash
  docker compose ps
  ```

### 6.6: Verify Deployment

- [ ] **6.6.1** Test tool-server search endpoint
  ```bash
  curl -X POST http://localhost:8888/search \
    -H "Content-Type: application/json" \
    -d '{"query": "test"}'
  ```

- [ ] **6.6.2** Test tool-server read endpoint
  ```bash
  curl -X POST http://localhost:8888/read \
    -H "Content-Type: application/json" \
    -d '{"url": "https://example.com", "question": "test"}'
  ```

- [ ] **6.6.3** Check GPU usage
  ```bash
  nvidia-smi
  ```

- [ ] **6.6.4** Check logs for errors
  ```bash
  docker compose logs vllm-server
  docker compose logs tool-server
  ```

---

## Validation Checklist

### Before Committing

- [ ] All code modifications complete
- [ ] All Docker files created
- [ ] `.env.example` created (no real keys)
- [ ] `.env` is in `.gitignore`
- [ ] Syntax checks pass
- [ ] Import checks pass
- [ ] docker-compose.yml validates

### Before Pushing

- [ ] Git status shows only intended files
- [ ] Commit message is descriptive
- [ ] Ready to push to fork

### After Cloning on Server

- [ ] Correct branch checked out
- [ ] All files present
- [ ] `.env` file configured with real keys
- [ ] Docker images build successfully
- [ ] Services start without errors
- [ ] Endpoints respond correctly

---

## Quick Reference Commands

### Local Development

```bash
# Setup
git checkout -b migration/tavily-playwright-gpt5

# After modifications
git add .
git status  # Review changes
git commit -m "Descriptive commit message"
git push origin migration/tavily-playwright-gpt5
```

### Server Deployment

```bash
# Clone and setup
cd /datapool/pokee-research
git clone https://github.com/YOUR_USERNAME/PokeeResearchOSS.git
cd PokeeResearchOSS
git checkout migration/tavily-playwright-gpt5

# Configure
cp .env.example .env
nano .env  # Add API keys

# Deploy
docker compose build
docker compose up -d
docker compose logs -f
```

---

**Last Updated:** 2025-01-27  
**Workflow:** Local → Commit → Push → Clone → Deploy


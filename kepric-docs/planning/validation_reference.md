# Deployment Validation Reference

**Purpose:** Validation commands and checks using Tavily and Context7  
**Use Case:** Validate each modification step before marking checkboxes complete

---

## Validation Tools

### 1. Tavily API Validation

**Purpose:** Validate Tavily integration code matches official API

**Commands:**

```bash
# Test Tavily API connection
python3 -c "
from tavily import TavilyClient
client = TavilyClient(api_key='YOUR_KEY')
result = client.search('test query')
print(result)
print('Keys:', result.keys())
print('Results count:', len(result.get('results', [])))
"

# Validate response structure
python3 -c "
from tavily import TavilyClient
import json
client = TavilyClient(api_key='YOUR_KEY')
result = client.search('test', max_results=5)
expected_keys = ['query', 'results', 'answer']
assert all(k in result for k in expected_keys), 'Missing keys'
assert isinstance(result['results'], list), 'Results must be list'
for r in result['results']:
    assert 'url' in r, 'Result missing url'
    assert 'title' in r, 'Result missing title'
    assert 'content' in r, 'Result missing content'
print('✅ Tavily response structure validated')
"
```

**What to Check:**

- [ ] API key is valid
- [ ] Response contains `results` array
- [ ] Each result has `url`, `title`, `content`, `score`
- [ ] Error handling works for invalid keys

### 2. Playwright Validation

**Purpose:** Validate Playwright installation and basic functionality

**Commands:**

```bash
# Test Playwright installation
python3 -c "
from playwright.async_api import async_playwright
import asyncio

async def test():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto('https://example.com')
        title = await page.title()
        await browser.close()
        assert title == 'Example Domain', f'Expected Example Domain, got {title}'
        print('✅ Playwright works correctly')

asyncio.run(test())
"

# Test link extraction
python3 -c "
from playwright.async_api import async_playwright
import asyncio

async def test():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto('https://example.com')
        links = await page.evaluate('''
            () => Array.from(document.querySelectorAll('a'))
                .map(a => ({url: a.href, title: a.textContent.trim()}))
                .filter(l => l.url.startsWith('http'))
        ''')
        await browser.close()
        print(f'✅ Extracted {len(links)} links')
        for link in links[:3]:
            print(f'  - {link[\"title\"]}: {link[\"url\"]}')

asyncio.run(test())
"
```

**What to Check:**

- [ ] Browser launches successfully
- [ ] Can navigate to URLs
- [ ] Can extract content
- [ ] Can extract links
- [ ] Headless mode works

### 3. OpenAI GPT-5 Validation

**Purpose:** Validate GPT-5 API access and model availability

**Commands:**

```bash
# List available GPT-5 models
python3 -c "
from openai import OpenAI
import os
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
models = client.models.list()
gpt5_models = [m.id for m in models.data if 'gpt-5' in m.id]
print('Available GPT-5 models:')
for model in gpt5_models:
    print(f'  - {model}')
assert len(gpt5_models) > 0, 'No GPT-5 models found'
print('✅ GPT-5 models available')
"

# Test GPT-5 API call
python3 -c "
from openai import AsyncOpenAI
import asyncio
import os

async def test():
    client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    response = await client.chat.completions.create(
        model='gpt-5-pro',
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': 'Say hello in one word.'}
        ],
        max_tokens=10
    )
    content = response.choices[0].message.content
    print(f'✅ GPT-5 response: {content}')
    assert content is not None and len(content) > 0

asyncio.run(test())
"

# Test error handling
python3 -c "
from openai import AsyncOpenAI
import asyncio
import os

async def test():
    client = AsyncOpenAI(api_key='invalid_key')
    try:
        await client.chat.completions.create(
            model='gpt-5-pro',
            messages=[{'role': 'user', 'content': 'test'}]
        )
        assert False, 'Should have raised error'
    except Exception as e:
        assert '401' in str(e) or 'invalid' in str(e).lower(), f'Unexpected error: {e}'
        print('✅ Error handling works correctly')

asyncio.run(test())
"
```

**What to Check:**

- [ ] GPT-5 models are accessible (`gpt-5`, `gpt-5-pro`, `gpt-5-mini`, `gpt-5-nano`)
- [ ] Can make API calls successfully
- [ ] Error handling works for invalid keys
- [ ] Rate limiting is handled appropriately

### 4. Code Validation

**Purpose:** Validate modified code matches expected patterns

**Tavily Integration Check:**

```bash
# Validate tavily_search function exists
python3 -c "
from tool_server.search import tavily_search, WebSearchAgent
import inspect

# Check function signature
sig = inspect.signature(tavily_search)
params = list(sig.parameters.keys())
assert 'query' in params, 'Missing query parameter'
assert 'timeout' in params, 'Missing timeout parameter'
assert 'top_k' in params, 'Missing top_k parameter'
print('✅ tavily_search signature correct')

# Check return type annotation
assert sig.return_annotation.__name__ == 'SearchResult', 'Wrong return type'
print('✅ tavily_search return type correct')
"

# Validate response mapping
python3 -c "
# Mock Tavily response
tavily_response = {
    'results': [
        {'url': 'https://example.com', 'title': 'Example', 'content': 'Test content', 'score': 0.95}
    ],
    'query': 'test',
    'answer': 'Test answer'
}

# Test extraction function
from tool_server.search import _extract_results_from_tavily_response
items = _extract_results_from_tavily_response(tavily_response)
assert len(items) == 1, 'Should extract one item'
assert items[0].url == 'https://example.com', 'URL mismatch'
assert items[0].title == 'Example', 'Title mismatch'
print('✅ Response extraction works correctly')
"
```

**Playwright Integration Check:**

```bash
# Validate playwright_read function
python3 -c "
from tool_server.read import playwright_read, WebReadAgent
import inspect

sig = inspect.signature(playwright_read)
params = list(sig.parameters.keys())
assert 'url' in params, 'Missing url parameter'
assert 'timeout' in params, 'Missing timeout parameter'
assert sig.return_annotation.__name__ == 'ReadResult', 'Wrong return type'
print('✅ playwright_read signature correct')
"
```

**GPT-5 Integration Check:**

```bash
# Validate OpenAI client function
python3 -c "
from tool_server.utils import get_openai_client
from openai import AsyncOpenAI

client = get_openai_client()
assert isinstance(client, AsyncOpenAI), 'Should return AsyncOpenAI instance'
print('✅ get_openai_client returns correct type')
"

# Validate llm_summary function
python3 -c "
from tool_server.utils import llm_summary
import inspect

sig = inspect.signature(llm_summary)
params = list(sig.parameters.keys())
assert 'user_prompt' in params, 'Missing user_prompt'
assert 'client' in params, 'Missing client'
assert 'timeout' in params, 'Missing timeout'
assert 'model' in params, 'Missing model'
print('✅ llm_summary signature correct')
"
```

### 5. Integration Testing

**Purpose:** Validate end-to-end functionality

```bash
# Test search endpoint
curl -X POST http://localhost:8888/search \
  -H "Content-Type: application/json" \
  -d '{"query": "test query"}' \
  | python3 -m json.tool

# Expected response structure:
# {
#   "query": "test query",
#   "url_items": [...],
#   "success": true,
#   "metadata": {"provider": "tavily", ...},
#   "error": ""
# }

# Test read endpoint
curl -X POST http://localhost:8888/read \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com", "question": "What is this page about?"}' \
  | python3 -m json.tool

# Expected response structure:
# {
#   "success": true,
#   "content": "...",
#   "summary": "...",
#   "url_items": [...],
#   "metadata": {"source": "playwright", ...},
#   "error": ""
# }
```

---

## Step-by-Step Validation Process

### For Each Code Modification:

1. **Before Modification:**

   - [ ] Understand current implementation
   - [ ] Review API documentation (Tavily/Playwright/OpenAI)
   - [ ] Test current functionality

2. **During Modification:**

   - [ ] Make changes incrementally
   - [ ] Run syntax check: `python3 -m py_compile <file>`
   - [ ] Test imports: `python3 -c "from <module> import <function>"`

3. **After Modification:**
   - [ ] Run unit tests (if available)
   - [ ] Test function directly
   - [ ] Test integration with other components
   - [ ] Verify error handling

### Validation Checklist Template:

For each task in `deployment_plan.md`:

- [ ] **Code Syntax:** No syntax errors
- [ ] **Imports:** All imports resolve correctly
- [ ] **Function Signature:** Matches expected interface
- [ ] **Response Format:** Matches expected structure
- [ ] **Error Handling:** Handles errors gracefully
- [ ] **Performance:** Acceptable latency
- [ ] **Integration:** Works with other components

---

## Using Sequential Thinking for Validation

When validating complex changes:

1. **Break down into sub-tasks**

   - Validate individual functions first
   - Then validate integration
   - Finally validate end-to-end

2. **Use sequential thinking for each step**

   - Think through expected behavior
   - Test actual behavior
   - Compare and identify discrepancies

3. **Document findings**
   - Note any issues found
   - Document solutions
   - Update validation checklist

---

## Context7 Documentation References

Use Context7 to validate implementation against official documentation:

**Tavily:**

- Library ID: `/tavily-ai/tavily-python`
- Validate: API usage, response structure, error handling

**Playwright:**

- Library ID: `/microsoft/playwright-python`
- Validate: Async API usage, browser automation, DOM access

**OpenAI:**

- Library ID: `/openai/openai-python`
- Validate: Async client usage, GPT-5 model names, error handling

---

## Quick Validation Script

Create `scripts/validate_changes.sh`:

```bash
#!/bin/bash
set -e

echo "Validating Tavily integration..."
python3 -c "from tool_server.search import tavily_search; print('✅ Tavily OK')"

echo "Validating Playwright integration..."
python3 -c "from tool_server.read import playwright_read; print('✅ Playwright OK')"

echo "Validating GPT-5 integration..."
python3 -c "from tool_server.utils import get_openai_client, llm_summary; print('✅ GPT-5 OK')"

echo "All validations passed!"
```

---

**Last Updated:** 2025-01-27  
**Related:** `deployment_plan.md`, `quick_start.md`

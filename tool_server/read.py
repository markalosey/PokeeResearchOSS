# Copyright 2025 Pokee AI Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import asyncio
import json
import os
from typing import Any, Dict, List

from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError, Error as PlaywrightError
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from logging_utils import setup_colored_logger
from tool_server.utils import (
    _is_valid_url,
    get_genai_client,
    get_retry_delay,
    llm_summary,
)

load_dotenv()

logger = setup_colored_logger(__name__)


class ReadURLItem(BaseModel):
    """
    A single URL extracted from a webpage with contextual information.

    Attributes:
        url: The complete URL found on the page
        title: Descriptive title explaining what the URL links to
    """

    url: str = Field(description="The full URL found on the page")
    title: str = Field(description="Title or description of the linked content")


class ReadResult(BaseModel):
    """
    Results from reading and analyzing a webpage.

    Attributes:
        success: Whether the read operation completed successfully
        content: Raw text content extracted from the webpage
        summary: LLM-generated summary based on the question, or truncated
                content if LLM summarization fails
        raw_response: Original API response (truncated to 500 chars)
        url_items: Relevant URLs discovered on the page
        metadata: API usage statistics and response metadata
        error: Error message if operation failed, empty string otherwise
    """

    success: bool
    content: str
    summary: str = ""
    raw_response: str = ""
    url_items: List[ReadURLItem] = []
    metadata: Dict[str, Any] = {}
    error: str = ""


async def playwright_read(url: str, timeout: int = 30) -> ReadResult:
    """
    Read and extract content from a webpage using Playwright browser automation.

    Args:
        url: The URL of the webpage to read
        timeout: Maximum time in seconds to wait (default: 30)

    Returns:
        ReadResult containing extracted content, links, and metadata

    Example:
        >>> result = await playwright_read("https://example.com")
        >>> if result.success:
        ...     print(result.content)
        ...     for item in result.url_items:
        ...         print(f"{item.title}: {item.url}")
    """
    # Validate URL before processing
    if not _is_valid_url(url):
        return ReadResult(
            success=False,
            content="",
            url_items=[],
            raw_response="Invalid URL",
            metadata={
                "source": "playwright",
                "url": url,
                "status": 400,
                "execution_time": 0.0,
                "links_found": 0,
                "relevant_links": 0,
            },
            error="Invalid URL format",
        )

    loop = asyncio.get_running_loop()
    start_time = loop.time()
    browser = None

    try:
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
            await page.route("**/*.{png,jpg,jpeg,gif,svg,css,woff,woff2}", lambda route: route.abort())
            
            # Navigate to URL with timeout
            await page.goto(url, wait_until="networkidle", timeout=timeout * 1000)
            
            # Extract main content using JavaScript
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
            
            # Extract links from page
            links = await page.evaluate("""
                () => {
                    return Array.from(document.querySelectorAll('a'))
                        .map(a => ({
                            url: a.href,
                            title: a.textContent.trim() || a.innerText.trim() || 'No Title',
                            text: a.innerText.trim()
                        }))
                        .filter(link => link.url && link.url.startsWith('http'));
                }
            """)
            
            # Get page title
            page_title = await page.title()
            
            # Close browser
            await browser.close()
            browser = None
            
            execution_time = loop.time() - start_time
            
            # Convert links to ReadURLItem format
            url_items = []
            for link in links:
                link_url = link.get("url", "").strip()
                link_title = link.get("title", "No Title").strip()
                
                if link_url and _is_valid_url(link_url):
                    url_items.append(
                        ReadURLItem(url=link_url, title=link_title)
                    )
            
            metadata = {
                "source": "playwright",
                "url": url,
                "status": 200,
                "title": page_title,
                "execution_time": execution_time,
                "links_found": len(links),
                "relevant_links": len(url_items),
            }
            
            logger.info(
                f"Successfully read '{url}', found {len(url_items)} relevant links"
            )
            
            return ReadResult(
                success=True,
                content=text_content,
                url_items=url_items,
                raw_response=f"Page title: {page_title}",
                metadata=metadata,
            )

    except PlaywrightTimeoutError:
        logger.warning(f"Read request timed out after {timeout}s")
        if browser:
            try:
                await browser.close()
            except Exception:
                pass
        return ReadResult(
            success=False,
            content="",
            url_items=[],
            raw_response="Request timed out",
            metadata={
                "source": "playwright",
                "url": url,
                "status": 408,
                "execution_time": loop.time() - start_time,
                "links_found": 0,
                "relevant_links": 0,
            },
            error=f"Request timed out after {timeout}s",
        )

    except PlaywrightError as e:
        logger.warning(f"Playwright error during read: {str(e)}")
        if browser:
            try:
                await browser.close()
            except Exception:
                pass
        return ReadResult(
            success=False,
            content="",
            url_items=[],
            raw_response=str(e)[:500],
            metadata={
                "source": "playwright",
                "url": url,
                "status": 500,
                "execution_time": loop.time() - start_time,
                "links_found": 0,
                "relevant_links": 0,
            },
            error=f"Playwright error: {str(e)[:200]}",
        )

    except Exception as e:
        logger.error(f"Unexpected error during read: {str(e)}")
        if browser:
            try:
                await browser.close()
            except Exception:
                pass
        return ReadResult(
            success=False,
            content="",
            url_items=[],
            raw_response=str(e)[:500],
            metadata={
                "source": "playwright",
                "url": url,
                "status": 500,
                "execution_time": loop.time() - start_time,
                "links_found": 0,
                "relevant_links": 0,
            },
            error=f"Unexpected error: {str(e)[:200]}",
        )


class WebReadAgent:
    """
    Agent for reading web content with LLM summarization and concurrency control.

    This agent reads webpages using Playwright browser automation, extracts content,
    and generates summaries using an LLM. It includes retry logic for recoverable
    errors and falls back to truncated content if LLM summarization fails.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the WebReadAgent.

        Args:
            config: Configuration dictionary with optional keys:
                - max_concurrent_requests: Max concurrent reads (default: 500)
                - max_content_words: Max words to send to LLM (default: 10000)
                - max_summary_words: Max words for fallback summary (default: 2048)
                - max_summary_retries: Max retries for LLM (default: 3)
        """
        self.client = get_genai_client()
        self._timeout = config.get("timeout", 30)
        self._semaphore = asyncio.Semaphore(config.get("max_concurrent_requests", 500))
        self.max_content_words = config.get("max_content_words", 10000)
        self.max_summary_words = config.get("max_summary_words", 2048)
        self.max_summary_retries = config.get("max_summary_retries", 3)

    def _truncate_content_to_words(self, content: str, max_words: int) -> str:
        """
        Truncate content to max words, keeping beginning and end.

        Args:
            content: Content to truncate
            max_words: Maximum number of words to keep

        Returns:
            Truncated content with " ... " in the middle if exceeded
        """
        words = content.split()
        if len(words) <= max_words:
            return content

        half_words = max_words // 2
        return " ".join(words[:half_words]) + " ... " + " ".join(words[-half_words:])

    async def read(self, question: str, url: str) -> ReadResult:
        """
        Read a webpage and generate a summary based on the question.

        This method:
        1. Reads webpage content using Playwright browser automation
        2. Truncates content if too long
        3. Generates LLM summary with up to 3 retries for recoverable errors
        4. Falls back to truncated content if LLM fails

        Args:
            question: Question or context for summarization
            url: URL of the webpage to read

        Returns:
            ReadResult with content and summary (either LLM-generated or
            truncated content as fallback)

        Example:
            >>> agent = WebReadAgent(config={})
            >>> result = await agent.read("What is Python?", "https://python.org")
            >>> if result.success:
            ...     print(result.summary)
        """
        logger.info(f"Reading '{url}' with question: '{question[:100]}...'")

        try:
            async with self._semaphore:
                result = await playwright_read(url.strip(), timeout=self._timeout)

            if not result.success:
                logger.warning(
                    f"Read failed for '{url}' with status "
                    f"{result.metadata.get('status', 'unknown')}: {result.error}"
                )
                return result

            logger.info(
                f"Read successful for '{url}', content: {len(result.content)} chars"
            )

            # Truncate content if too long
            original_content = result.content
            words = result.content.split()
            if len(words) > self.max_content_words:
                result.content = self._truncate_content_to_words(
                    result.content, self.max_content_words
                )
                logger.info(
                    f"Truncated content from {len(words)} to {self.max_content_words} words"
                )

            # Generate summary with retry logic
            for attempt in range(self.max_summary_retries):
                summary_result = await llm_summary(
                    user_prompt=f"<question>{question}</question><content>{result.content}</content>",
                    client=self.client,
                )

                if summary_result.success:
                    logger.info(
                        f"Summary generated for '{url}' on attempt {attempt + 1}, "
                        f"length: {len(summary_result.text)} chars"
                    )
                    result.summary = summary_result.text
                    return result

                if (
                    summary_result.recoverable
                    and attempt < self.max_summary_retries - 1
                ):
                    retry_delay = get_retry_delay(attempt, summary_result.error or "")
                    logger.warning(
                        f"Recoverable error for '{url}' "
                        f"(attempt {attempt + 1}/{self.max_summary_retries}): "
                        f"{summary_result.error}. Retrying in {retry_delay}s..."
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    logger.warning(
                        f"Summary failed for '{url}': {summary_result.error}"
                    )
                    break

            # Fallback to truncated content
            logger.info(f"Using truncated content as fallback for '{url}'")
            result.summary = self._truncate_content_to_words(
                original_content, self.max_summary_words
            )
            return result

        except Exception as e:
            logger.error(
                f"Unexpected error in WebReadAgent for '{url}': {str(e)}",
                exc_info=True,
            )
            return ReadResult(
                success=False,
                content="",
                url_items=[],
                raw_response="",
                metadata={
                    "source": "playwright",
                    "url": url,
                    "status": 500,
                    "execution_time": 0.0,
                    "links_found": 0,
                    "relevant_links": 0,
                },
                error=f"WebReadAgent error: {str(e)[:200]}",
            )

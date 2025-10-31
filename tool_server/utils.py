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
import logging
import os
import re
from typing import Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel

from logging_utils import setup_colored_logger

load_dotenv()


logger = setup_colored_logger(__name__, level=logging.INFO)


def _is_valid_url(url: str) -> bool:
    """
    Check if a URL is valid and not a common non-informative link.

    Args:
        url (str): The URL to validate

    Returns:
        bool: True if the URL is valid and informative
    """
    if not url or not isinstance(url, str):
        return False

    url_lower = url.lower()

    # Skip common non-informative link patterns
    skip_patterns = [
        "javascript:",
        "mailto:",
        "#",
        "tel:",
        "data:",
        "blob:",
    ]

    # Skip social media and common non-content domains
    skip_domains = [
        "facebook.com",
        "twitter.com",
        "x.com",  # Twitter's new domain
        "instagram.com",
        "youtube.com",
        "tiktok.com",
        "pinterest.com",
        "snapchat.com",
        "discord.com",
        "telegram.org",
        "whatsapp.com",
        "wechat.com",
        "weibo.com",
        "douyin.com",
        "substack.com",  # Newsletter platform
        "patreon.com",
        "onlyfans.com",
        "twitch.tv",
        "vimeo.com",
        "dailymotion.com",
        "rumble.com",
        "bitchute.com",
    ]

    # Check skip patterns
    if any(pattern in url_lower for pattern in skip_patterns):
        return False

    # Check skip domains (including subdomains)
    for domain in skip_domains:
        if domain in url_lower:
            return False

    # Must start with http/https or be a relative URL starting with /
    if not (url.startswith(("http://", "https://")) or url.startswith("/")):
        return False

    return True


logger = setup_colored_logger(__name__, level=logging.INFO)
_openai_client = None


def get_openai_client():
    """Get or create the global OpenAI client instance."""
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not found")
        _openai_client = AsyncOpenAI(api_key=api_key)
    return _openai_client


def extract_retry_delay_from_error(error_str: str) -> Optional[float]:
    """
    Extract retry delay from OpenAI API error response.
    Returns the delay in seconds if found, None otherwise.
    """
    try:
        # OpenAI rate limit errors may include retry-after header information
        # Check for common patterns
        if "rate_limit_exceeded" in error_str.lower() or "429" in error_str:
            # Look for retry-after pattern in error message
            retry_after_match = re.search(
                r'retry[_-]after["\']?\s*:\s*["\']?(\d+(?:\.\d+)?)', error_str, re.IGNORECASE
            )
            if retry_after_match:
                return float(retry_after_match.group(1))
            
            # Default delay for rate limits: 2 seconds
            return 2.0

    except Exception as e:
        logger.warning(f"Could not extract retry delay from error: {e}")

    return None


def get_retry_delay(try_cnt: int, error_str: str) -> float:
    """
    Calculate appropriate retry delay based on error type and attempt number.

    First tries to extract explicit retry delay from API error response.
    Falls back to exponential backoff if no delay is specified.

    Args:
        try_cnt: Current attempt number (0-indexed)
        error_str: Error message to check for retry delay hints

    Returns:
        Delay in seconds before next retry (always returns a value)
    """
    # Try to extract API-specified retry delay
    retry_delay = extract_retry_delay_from_error(error_str)
    if retry_delay is None:
        # Fallback to exponential backoff: 1s, 4s, 16s, 64s, etc.
        retry_delay = min(4**try_cnt, 60)  # Cap at 60 seconds
    return retry_delay


SYSTEM_INSTRUCTION = """You are a helpful deep research assistant. I will provide you:
* A complex question that requires a deep research to answer.
* The content of a webpage returned by our web reader.

Your task is to read the webpage content carefully and extract all information that could help answer the question. Provide detailed information including numbers, dates, facts, examples, and explanations when available. Remove the irrelevant parts to reduce noise. Note that there could be no useful information on the webpage.

Important note: Use the same language as the user's main question for the summary. For example, if the question is in Chinese, then the summary should also be in Chinese.

Now think and extract the information that could help answer the question."""

MODEL = "gpt-5-pro"


class LLMSummaryResult(BaseModel):
    """Result from LLM summarization attempt."""

    success: bool
    text: str
    error: Optional[str] = None
    recoverable: bool = False  # Whether the error is recoverable by retrying


async def llm_summary(
    user_prompt: str,
    client: AsyncOpenAI,
    timeout: float = 30.0,
    model: str = MODEL,
) -> LLMSummaryResult:
    """
    Generate a summary using GPT-5 with robust error handling.

    Args:
        user_prompt: The prompt to send to the LLM (includes question and content)
        client: The OpenAI AsyncOpenAI client instance
        timeout: Request timeout in seconds (default: 30.0)
        model: Model to use for generation (default: MODEL constant, "gpt-5-pro")

    Returns:
        LLMSummaryResult with success status, text, and error information

    Recoverable errors (worth retrying):
        - Rate limiting (429, rate_limit_exceeded)
        - Timeout errors
        - Transient network errors (503, 502, connection errors)
        - Internal server errors (500)

    Non-recoverable errors (not worth retrying):
        - Empty or invalid prompts
        - Authentication errors (401, 403, invalid_api_key)
        - Invalid request errors (400, context_length_exceeded)
        - Empty responses (likely a consistent issue)
    """

    def _is_recoverable_error(error_msg: str) -> bool:
        """Check if an error is recoverable by retrying."""
        error_lower = error_msg.lower()

        # Recoverable patterns (OpenAI-specific)
        recoverable_patterns = [
            "rate_limit_exceeded",  # Rate limiting
            "429",  # Too Many Requests
            "503",  # Service Unavailable
            "502",  # Bad Gateway
            "500",  # Internal Server Error (may be transient)
            "timeout",
            "timed out",
            "connection",
            "network",
            "temporary",
            "transient",
            "unavailable",
            "overloaded",
        ]

        # Non-recoverable patterns (OpenAI-specific)
        non_recoverable_patterns = [
            "401",  # Unauthorized
            "403",  # Forbidden
            "400",  # Bad Request
            "invalid",
            "authentication",
            "permission",
            "invalid_api_key",
            "context_length_exceeded",  # Token limit exceeded
            "quota_exceeded",  # Permanent quota issue
        ]

        # Check non-recoverable first (higher priority)
        if any(pattern in error_lower for pattern in non_recoverable_patterns):
            return False

        # Check recoverable patterns
        if any(pattern in error_lower for pattern in recoverable_patterns):
            return True

        # Default: not recoverable for unknown errors
        return False

    # Input validation - NOT RECOVERABLE
    if not user_prompt or not user_prompt.strip():
        logger.warning("Empty or whitespace-only prompt provided to llm_summary")
        return LLMSummaryResult(
            success=False,
            text="",
            error="Empty or whitespace-only prompt",
            recoverable=False,
        )

    original_length = len(user_prompt)
    MAX_PROMPT_LENGTH = 1_000_000  # 1MB limit

    if original_length > MAX_PROMPT_LENGTH:
        logger.warning(
            f"Prompt too long ({original_length:,} chars), truncating to {MAX_PROMPT_LENGTH:,} chars"
        )
        user_prompt = user_prompt[:MAX_PROMPT_LENGTH]

    try:
        # Make API request with timeout
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_INSTRUCTION},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=2048,
                temperature=0.1,
            ),
            timeout=timeout,
        )

        # Extract text from response
        if not response.choices or len(response.choices) == 0:
            logger.warning("No choices in response")
            return LLMSummaryResult(
                success=False,
                text="",
                error="No choices in response",
                recoverable=False,
            )

        text = response.choices[0].message.content

        # Validate extracted text - NOT RECOVERABLE
        if text is None:
            logger.warning("No text in response")
            return LLMSummaryResult(
                success=False,
                text="",
                error="No text in response",
                recoverable=False,
            )

        text = text.strip()
        if not text:
            logger.warning("LLM returned empty text response")
            return LLMSummaryResult(
                success=False,
                text="",
                error="Empty text response",
                recoverable=False,
            )

        if len(text) < 10:
            logger.warning(
                f"LLM returned very short response ({len(text)} chars): '{text}'"
            )

        logger.info(
            f"LLM summary successful, "
            f"input: {original_length:,} chars, output: {len(text):,} chars"
        )
        return LLMSummaryResult(success=True, text=text, recoverable=False)

    except asyncio.TimeoutError:
        # RECOVERABLE - might succeed with retry
        error_msg = f"Request timed out after {timeout}s"
        logger.warning(error_msg)
        return LLMSummaryResult(
            success=False,
            text="",
            error=error_msg,
            recoverable=True,
        )

    except Exception as e:
        error_msg = str(e)
        is_recoverable = _is_recoverable_error(error_msg)

        if is_recoverable:
            logger.warning(f"Recoverable LLM error: {e}")
        else:
            logger.error(f"Non-recoverable LLM error: {e}", exc_info=True)

        return LLMSummaryResult(
            success=False,
            text="",
            error=error_msg,
            recoverable=is_recoverable,
        )

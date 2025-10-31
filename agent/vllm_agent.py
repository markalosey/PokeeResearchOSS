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

"""
VLLM Deep Research Agent

This agent uses a VLLM-served model for faster inference with true concurrent support.
The singleton HTTP client pattern enables efficient connection pooling and reuse across
multiple agent instances.
"""

import asyncio
import os
import threading

import httpx

from agent.base_agent import BaseDeepResearchAgent
from logging_utils import setup_colored_logger

logger = setup_colored_logger(__name__)


def _estimate_tokens(text: str) -> int:
    """Rough estimate of token count (1 token ≈ 4 characters).
    
    This is a simple heuristic. For accurate counting, use a tokenizer,
    but this is fast and sufficient for truncation purposes.
    
    Args:
        text: Text to estimate tokens for
        
    Returns:
        Estimated token count
    """
    return len(text) // 4


def _truncate_messages(messages: list[dict], max_tokens: int = 5176) -> list[dict]:
    """Truncate messages to fit within token limit.
    
    Preserves:
    - System message (first message) - ALWAYS kept
    - Most recent user message/question - ALWAYS kept
    - Last 2-3 complete turn cycles (assistant + tool response pairs)
    - This ensures agent can see recent searches/results and make progress
    
    Args:
        messages: List of message dicts with 'role' and 'content'
        max_tokens: Maximum tokens to keep (default: 5176 for 6144 context - 768 gen - 200 buffer)
        
    Returns:
        Truncated message list (always includes system + latest user message + recent history)
    """
    if not messages:
        return messages
    
    # Always keep the first message (system prompt)
    system_message = messages[0]
    remaining_messages = messages[1:]
    
    if not remaining_messages:
        return [system_message]
    
    # Find the most recent user message (this is the current question)
    recent_user_msgs = [msg for msg in reversed(remaining_messages) if msg.get("role") == "user"]
    most_recent_user_msg = recent_user_msgs[0] if recent_user_msgs else None
    
    # Estimate tokens for system message and user question
    system_tokens = _estimate_tokens(system_message.get("content", ""))
    user_question_tokens = _estimate_tokens(most_recent_user_msg.get("content", "")) if most_recent_user_msg else 0
    
    # Reserve tokens for system + user question + buffer
    reserved_tokens = system_tokens + user_question_tokens + 200  # 200 token buffer
    available_tokens = max_tokens - reserved_tokens
    
    if available_tokens <= 0:
        # If system + user question itself is too large, return just those
        if most_recent_user_msg:
            return [system_message, most_recent_user_msg]
        return [system_message]
    
    # Build truncated list preserving recent conversation history
    # We need to keep tool responses so agent can see what it already searched
    truncated = []
    current_tokens = 0
    
    # Process messages in reverse order (most recent first), but skip the most recent user msg
    messages_to_process = [msg for msg in reversed(remaining_messages) if msg != most_recent_user_msg]
    
    # Keep at least the last complete turn: assistant response + tool response
    # Then add more if space allows
    for msg in messages_to_process:
        content = msg.get("content", "")
        msg_tokens = _estimate_tokens(content)
        
        # If we can fit this message, add it
        if current_tokens + msg_tokens <= available_tokens:
            truncated.insert(0, msg)  # Insert at beginning to maintain order
            current_tokens += msg_tokens
        else:
            # Can't fit this message
            # If we have at least one assistant message, that's good enough
            # Otherwise try to truncate the content to fit at least something
            if not any(m.get("role") == "assistant" for m in truncated):
                # No assistant messages yet - try to fit at least a truncated version
                available_chars = (available_tokens - current_tokens) * 4
                if available_chars > 100 and msg.get("role") == "assistant":
                    truncated_content = content[:available_chars] + "...[truncated]"
                    truncated.insert(0, {"role": msg.get("role"), "content": truncated_content})
            break
    
    # Always include system message at the start, then conversation history, then the current question
    if most_recent_user_msg:
        return [system_message] + truncated + [most_recent_user_msg]
    else:
        return [system_message] + truncated


class VLLMDeepResearchAgent(BaseDeepResearchAgent):
    """Deep research agent using VLLM server for inference.

    This agent communicates with a VLLM server via HTTP API for model inference.
    Benefits over local inference:
    - True concurrent request handling
    - Optimized batching and scheduling
    - Better GPU utilization
    - Lower memory overhead per request

    The HTTP client is shared across all instances (singleton pattern) for efficient
    connection pooling and reuse.
    """

    # Class-level (singleton) HTTP client for connection pooling
    _client = None
    _vllm_url = None
    _timeout = None
    _client_lock = None

    def __init__(
        self,
        vllm_url: str = "http://localhost:9090/v1",
        model_name: str = "PokeeAI/pokee_research_7b",
        tool_config_path: str = "config/tool_config/pokee_tool_config.yaml",
        max_turns: int = 10,
        max_tool_response_length: int = 32768,
        max_tokens: int = 768,  # Generation tokens (default context: 6144, adjust if MAX_MODEL_LEN changes)
        timeout: float = 300.0,
    ):
        """Initialize the VLLM agent.

        Args:
            vllm_url: Base URL of the VLLM server (e.g., http://localhost:9090/v1)
            model_name: Model name for reference (must match VLLM server's model)
            tool_config_path: Path to tool configuration YAML file
            max_turns: Maximum conversation turns before giving up
            max_tool_response_length: Maximum length for tool response text
            max_tokens: Maximum tokens to generate (default: 768, adjust based on MAX_MODEL_LEN)
            timeout: HTTP request timeout in seconds (default: 300s = 5 minutes)
        """
        # Initialize base class (tools, regex patterns, etc.)
        super().__init__(
            tool_config_path=tool_config_path,
            max_turns=max_turns,
            max_tool_response_length=max_tool_response_length,
        )

        self.vllm_url = vllm_url
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.timeout = timeout

        # Initialize thread lock on first use
        if VLLMDeepResearchAgent._client_lock is None:
            VLLMDeepResearchAgent._client_lock = threading.Lock()

        # Create or reuse singleton HTTP client
        with VLLMDeepResearchAgent._client_lock:
            if VLLMDeepResearchAgent._client is None:
                self._create_client(vllm_url, timeout)
            elif VLLMDeepResearchAgent._vllm_url != vllm_url:
                logger.warning(
                    f"VLLM URL mismatch: expected {VLLMDeepResearchAgent._vllm_url}, "
                    f"got {vllm_url}. Using existing client."
                )

        # Reference the shared client
        self.client = VLLMDeepResearchAgent._client
        logger.debug(f"VLLM agent ready (server: {vllm_url})")

    @classmethod
    def _create_client(cls, vllm_url: str, timeout: float):
        """Create singleton HTTP client with optimized connection pooling.

        The client is configured for:
        - Connection pooling and reuse
        - HTTP/2 support for multiplexing
        - Separate connect and read timeouts
        - Large connection limits for high concurrency

        Args:
            vllm_url: Base URL of the VLLM server
            timeout: Request timeout in seconds
        """
        logger.info(f"Creating HTTP client for VLLM server at {vllm_url}...")

        # Create async HTTP client with optimized settings
        cls._client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                timeout=timeout,  # Overall request timeout
                connect=10.0,  # Separate connect timeout (fail fast if server down)
            ),
            limits=httpx.Limits(
                max_connections=100,  # Max total connections across all hosts
                max_keepalive_connections=20,  # Keep this many connections alive
            ),
            http2=True,  # Enable HTTP/2 for request multiplexing
        )
        cls._vllm_url = vllm_url
        cls._timeout = timeout
        logger.info("HTTP client created successfully!")

    async def generate(
        self, messages: list[dict], temperature: float = 0.7, top_p: float = 0.9
    ) -> str:
        """Generate response from messages using VLLM server.

        Makes an HTTP POST request to the VLLM chat completions endpoint.
        Supports cancellation via asyncio.CancelledError.
        Automatically truncates messages if they exceed context limit.

        Args:
            messages: Conversation messages in chat format
            temperature: Sampling temperature (0.0 = deterministic, higher = more random)
            top_p: Nucleus sampling parameter (0.0-1.0)

        Returns:
            Generated text response from the model

        Raises:
            asyncio.CancelledError: If the request is cancelled
            httpx.HTTPStatusError: If the server returns an error status
            httpx.TimeoutException: If the request times out
            httpx.RequestError: If there's a network/connection error
            ValueError: If the response format is unexpected
        """
        # Truncate messages to fit within context limit
        # Default: 6144 context - 768 generation - 200 buffer = ~5176 input tokens
        # Adjust if MAX_MODEL_LEN environment variable is different
        context_limit = int(os.getenv("MAX_MODEL_LEN", "6144"))
        generation_tokens = self.max_tokens
        input_tokens = context_limit - generation_tokens - 200  # Reserve buffer
        truncated_messages = _truncate_messages(messages, max_tokens=input_tokens)
        
        if len(truncated_messages) < len(messages):
            preserved_assistant = sum(1 for m in truncated_messages if m.get("role") == "assistant")
            preserved_tool = sum(1 for m in truncated_messages if m.get("role") == "tool")
            logger.warning(
                f"Truncated messages from {len(messages)} to {len(truncated_messages)} "
                f"(preserved {preserved_assistant} assistant + {preserved_tool} tool responses) "
                f"to fit context limit"
            )
        
        # Log the user question being sent to vLLM
        user_msgs = [msg for msg in truncated_messages if msg.get("role") == "user"]
        if user_msgs:
            logger.info(f"Sending to vLLM - User question: {user_msgs[0].get('content', '')[:200]}...")
        
        # Prepare chat completions request
        request_data = {
            "model": self.model_name,
            "messages": truncated_messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": self.max_tokens,
            "stream": False,  # Non-streaming for simpler handling
        }
        
        # Log request data (first message content)
        if truncated_messages:
            logger.debug(f"vLLM request - First message: {truncated_messages[0].get('content', '')[:200]}...")

        try:
            # Make HTTP request (cancellable)
            response = await self.client.post(
                f"{self.vllm_url}/chat/completions",
                json=request_data,
            )
            response.raise_for_status()

            # Parse JSON response
            result = response.json()

            # Extract generated content
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
                if not content:
                    logger.warning("VLLM returned empty content")
                    return ""
                return content
            else:
                raise ValueError(f"Unexpected VLLM response format: {result}")

        except asyncio.CancelledError:
            logger.info("VLLM generation cancelled")
            raise  # Re-raise to propagate cancellation

        except httpx.HTTPStatusError as e:
            logger.error(f"VLLM HTTP error {e.response.status_code}: {e.response.text}")
            raise

        except httpx.TimeoutException:
            logger.error(f"VLLM request timeout after {self.timeout}s")
            raise

        except httpx.RequestError as e:
            logger.error(f"VLLM request error: {e}")
            raise

        except Exception as e:
            logger.error(f"Unexpected error calling VLLM: {e}")
            raise

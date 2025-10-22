# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
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

# This file is different from verl/verl/tools/search_tool.py in that we use our own tools.

import asyncio
import json
import threading
import time
from contextlib import ExitStack
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar
from uuid import uuid4

import ray
import requests
from dotenv import load_dotenv

from logging_utils import setup_colored_logger
from tool_client.base_tool import BaseTool
from tool_client.schemas import OpenAIFunctionToolSchema, ToolResponse
from verl.utils.rollout_trace import rollout_trace_op

load_dotenv()

# Read port from .server_port file if it exists
DEFAULT_PORT = 8888
port_file = Path(".server_port")
if port_file.exists():
    try:
        port = int(port_file.read_text().strip())
    except (ValueError, IOError) as e:
        logger = setup_colored_logger("port_reader")
        logger.warning(
            f"Failed to read port from .server_port file: {e}. Using default port {DEFAULT_PORT}."
        )
        port = DEFAULT_PORT
else:
    port = DEFAULT_PORT

# Set BASE_URL with the port from the file
BASE_URL = f"http://localhost:{port}"

log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / "pokee_tools.log"
logger = setup_colored_logger(__name__, level="WARNING", log_file=log_file)

logger.info(f"Using Tool Server URL: {BASE_URL}")

T = TypeVar("T")


class PoolMode(Enum):
    """Execution pool mode enumeration."""

    ThreadMode = 1
    ProcessMode = 2


@ray.remote(concurrency_groups={"acquire": 1, "release": 10})
class TokenBucketWorker:
    """
    Ray actor for rate limiting using token bucket algorithm.

    Provides semaphore-based rate limiting with concurrent acquire/release operations.
    """

    def __init__(self, rate_limit: int):
        """
        Initialize the token bucket with specified limit.

        Args:
            rate_limit: Maximum number of concurrent operations
        """
        self.rate_limit = rate_limit
        self.current_count = 0  # For observability
        self._semaphore = threading.Semaphore(rate_limit)

    @ray.method(concurrency_group="acquire")
    def acquire(self):
        """Acquire a token from the bucket."""
        self._semaphore.acquire()
        self.current_count += 1

    @ray.method(concurrency_group="release")
    def release(self):
        """Release a token back to the bucket."""
        self._semaphore.release()
        self.current_count -= 1

    def get_current_count(self):
        """Get current number of acquired tokens."""
        return self.current_count


class SearchExecutionWorker:
    """
    Worker for executing operations with optional rate limiting.

    Provides a wrapper around function execution that applies rate limiting
    through the TokenBucketWorker.
    """

    def __init__(self, enable_global_rate_limit=True, rate_limit=10):
        """
        Initialize execution worker with optional rate limiting.

        Args:
            enable_global_rate_limit: Whether to enable rate limiting
            rate_limit: Maximum concurrent operations if enabled
        """
        self.rate_limit_worker = (
            self._init_rate_limit(rate_limit) if enable_global_rate_limit else None
        )

    def _init_rate_limit(self, rate_limit):
        """
        Initialize singleton rate limiter.

        Args:
            rate_limit: Maximum concurrent operations

        Returns:
            Ray actor reference to TokenBucketWorker
        """
        return TokenBucketWorker.options(
            name="rate-limiter", get_if_exists=True
        ).remote(rate_limit)

    def ping(self):
        """Health check method."""
        return True

    def execute(self, fn: Callable[..., T], *fn_args, **fn_kwargs) -> T:
        """
        Execute function with optional rate limiting.

        Args:
            fn: Function to execute
            *fn_args: Positional arguments to pass to fn
            **fn_kwargs: Keyword arguments to pass to fn

        Returns:
            Return value from fn
        """
        if self.rate_limit_worker:
            with ExitStack() as stack:
                stack.callback(self.rate_limit_worker.release.remote)
                ray.get(self.rate_limit_worker.acquire.remote())
                try:
                    return fn(*fn_args, **fn_kwargs)
                except Exception as e:
                    logger.warning(f"Error when executing operation: {e}")
                    raise
        else:
            return fn(*fn_args, **fn_kwargs)


def init_execution_pool(
    num_workers: int,
    enable_global_rate_limit=True,
    rate_limit=10,
    mode: PoolMode = PoolMode.ThreadMode,
):
    """
    Initialize execution pool with specified workers and rate limit.

    Args:
        num_workers: Number of concurrent workers
        enable_global_rate_limit: Whether to enable rate limiting
        rate_limit: Maximum concurrent operations if enabled
        mode: Execution mode (ThreadMode or ProcessMode)

    Returns:
        Ray actor reference to execution worker

    Raises:
        NotImplementedError: If mode is not ThreadMode
    """
    if mode == PoolMode.ThreadMode:
        return (
            ray.remote(SearchExecutionWorker)
            .options(max_concurrency=num_workers)
            .remote(
                enable_global_rate_limit=enable_global_rate_limit, rate_limit=rate_limit
            )
        )
    else:
        raise NotImplementedError("Process mode is not implemented yet")


def make_request(
    tool_name, payload, timeout=300, retries=5, delay_between_retries=10, result=None
):
    """
    Make HTTP request to tool server with retry logic.

    Args:
        tool_name: Name of the tool to call
        payload: Request payload to send
        timeout: Request timeout in seconds
        retries: Maximum number of retry attempts
        delay_between_retries: Delay in seconds between retries
        result: Optional pre-computed result to return (bypasses request)

    Returns:
        Response dictionary or error message string
    """
    if result is not None:
        return result

    for attempt in range(retries + 1):
        try:
            resp = requests.post(
                f"{BASE_URL}/{tool_name}",
                json=payload,
                timeout=timeout,
            )

            # Handle successful response
            if resp.status_code == 200:
                result = resp.json()
                if result["success"]:
                    break
                if (
                    "status_code" in result["metadata"]
                    and result["metadata"]["status_code"] in (400, 422)
                ):
                    error_reason = (
                        "bad request"
                        if result["metadata"]["status_code"] == 400
                        else "unprocessable content"
                    )
                    result = f"Request failed because {error_reason} in metadata. Tool name {tool_name}. Will stop retrying. The last resp is: {resp.text[:200]}...{resp.text[-200:]}"
                    logger.warning(result + f" for payload: {payload}")
                    break

            # Handle HTTP-level errors where retrying won't help
            if resp.status_code in (400, 422):
                error_reason = (
                    "bad request"
                    if resp.status_code == 400
                    else "unprocessable content"
                )
                result = f"Request failed because {error_reason}. Tool name {tool_name}. Will stop retrying. The last resp is: {resp.text[:200]}...{resp.text[-200:]}"
                logger.warning(result + f" for payload: {payload}")
                break

            if resp.status_code == 400:
                # bad request
                result = f"Request failed because bad request. Tool name {tool_name}. Will stop retrying. The last resp is: {resp.text[:200]}...{resp.text[-200:]}"
                logger.warning(result + f" for payload: {payload}")
                break

            # in other cases, we need to retry
            if attempt < retries:
                time.sleep(delay_between_retries)
            else:
                result = f"Request failed after all {retries} retries. Tool name {tool_name}. Will stop retrying. The last resp is: {resp.text[:200]}...{resp.text[-200:]}"
                logger.warning(result + f" for payload: {payload}")

        except Exception as e:
            if attempt < retries:
                time.sleep(delay_between_retries)
            else:
                logger.warning(
                    f"Exception requesting for payload '{payload}' after all {retries} retries. Tool name {tool_name}. Will stop retrying. The last exception is: {e}"
                )
                result = f"Request failed with exception: {e}. Tool name {tool_name}."

    assert isinstance(result, dict) or isinstance(result, str), (
        "result should be a dict or a string"
    )
    return result


class WebSearchTool(BaseTool):
    """
    Search tool for retrieving information using external search services.

    This tool performs web searches with rate limiting and concurrent execution
    support through Ray. Search queries are executed against external services
    and results are returned in structured format.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        Initialize WebSearchTool with configuration and schema.

        Args:
            config: Configuration dictionary containing:
                - num_workers: Max concurrent worker threads (default: 200)
                - rate_limit: Max concurrent requests (default: 200)
                - timeout: Request timeout in seconds (default: 300)
                - retries: Max retry attempts (default: 3)
                - delay_between_retries: Delay between retries in seconds (default: 5)
                - max_num_queries_in_query_list: Max queries per request (default: 5)
                - strategy_to_handle_too_many_queries: How to handle excess queries
                - enable_global_rate_limit: Whether to apply rate limiting (default: True)

            tool_schema: OpenAI function tool schema definition
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}

        # Worker and rate limiting configuration
        self.num_workers = config.get("num_workers", 200)
        self.rate_limit = config.get("rate_limit", 200)
        self.timeout = config.get("timeout", 300)
        self.retries = config.get("retries", 3)
        self.delay_between_retries = config.get("delay_between_retries", 5)
        self.max_queries = config.get("max_num_queries_in_query_list", 5)
        self.strategy_to_handle_too_many_queries = config.get(
            "strategy_to_handle_too_many_queries", "cap_at_max_queries"
        )

        self.enable_global_rate_limit = config.get("enable_global_rate_limit", True)

        self.execution_pool = init_execution_pool(
            num_workers=self.num_workers,
            enable_global_rate_limit=self.enable_global_rate_limit,
            rate_limit=self.rate_limit,
            mode=PoolMode.ThreadMode,
        )

        logger.info(f"Initialized WebSearchTool with config: {config}")

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        """
        Get the OpenAI tool schema.

        Returns:
            Tool schema in OpenAI format
        """
        return self.tool_schema

    async def create(
        self, instance_id: Optional[str] = None, **kwargs
    ) -> tuple[str, ToolResponse]:
        """
        Create a tool instance with the provided question.

        Args:
            instance_id: Optional instance ID (generated if not provided)
            **kwargs: Additional arguments containing:
                - create_kwargs.idx: Required instance index
                - create_kwargs.question: Required question string

        Returns:
            Tuple of (instance_id, empty tool response)

        Raises:
            ValueError: If idx or question is not provided
        """
        idx = kwargs.get("create_kwargs", {}).get("idx", None)
        if idx is None:
            raise ValueError("idx is not set")

        question = kwargs.get("create_kwargs", {}).get("question", None)
        if question is None:
            raise ValueError("question is not set")

        if instance_id is None:
            instance_id = str(uuid4())

        self._instance_dict[instance_id] = {
            "question": question,
        }
        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        """
        Execute web search queries with the provided parameters.

        Args:
            instance_id: The instance ID of the tool
            parameters: Dictionary containing:
                - query_list: List of search query strings
            **kwargs: Additional arguments

        Returns:
            Tuple of (tool_response, reward_score, metrics) where:
                - tool_response: ToolResponse containing search results
                - reward_score: Reward score (always 0.0)
                - metrics: Dictionary with execution metrics
        """
        # Input validation
        if not isinstance(parameters, dict):
            error_message = f"Parsing failed: web_search tool call arguments should be a dict. Arguments: {parameters}."
            logger.warning(error_message)
            return ToolResponse(text=error_message), 0.0, {"success": False}

        if "query_list" not in parameters:
            error_message = f"Parsing failed: web_search tool call arguments should have a key 'query_list', but not found. Arguments: {parameters}."
            logger.warning(error_message)
            return ToolResponse(text=error_message), 0.0, {"success": False}

        if not isinstance(parameters["query_list"], list) or not all(
            isinstance(q, str) for q in parameters["query_list"]
        ):
            error_message = f"Parsing failed: web_search tool call 'query_list' should be a list of strings. Arguments: {parameters}."
            logger.warning(error_message)
            return ToolResponse(text=error_message), 0.0, {"success": False}

        # Process query list
        query_list_from_params = [query.strip() for query in parameters["query_list"]]

        # Handle query limits based on strategy
        if self.strategy_to_handle_too_many_queries == "cap_at_max_queries":
            query_list_from_params = query_list_from_params[: self.max_queries]
        elif self.strategy_to_handle_too_many_queries == "reject":
            if len(query_list_from_params) > self.max_queries:
                error_message = f"Parsing failed: web_search tool call 'query_list' should have at most {self.max_queries} queries but got {len(query_list_from_params)} queries. Arguments: {parameters}."
                logger.warning(error_message)
                return ToolResponse(text=error_message), 0.0, {"success": False}
        elif self.strategy_to_handle_too_many_queries == "do_all_queries":
            pass
        else:
            raise ValueError(
                f"Invalid strategy to handle too many queries: {self.strategy_to_handle_too_many_queries}"
            )

        # Execute search using Ray execution pool
        try:
            # Create Ray futures for each query
            futures = [
                self.execution_pool.execute.remote(
                    make_request,
                    "search",
                    {
                        "query": query,
                        "retries": 0,  # no retries in the server side
                    },
                    self.timeout,
                    self.retries,
                    self.delay_between_retries,
                )
                for query in query_list_from_params
            ]

            # Wait for Ray workers to complete
            results = await asyncio.gather(*futures)

            # Format results
            return_list = [
                {
                    "query": query,
                    "search_results": result.get("url_items", []),
                }
                if isinstance(result, dict)
                else {"query": query, "error message": result}
                for query, result in zip(query_list_from_params, results)
            ]
            return_message = json.dumps(return_list)
            logger.debug(return_message)

            return ToolResponse(text=return_message), 0.0, {"success": True}

        except Exception as e:
            return_message = f"[web_search] Execution failed: {e}"
            logger.warning(return_message)
            return (
                ToolResponse(text=return_message),
                0.0,
                {"error": str(e), "success": False},
            )

    async def release(self, instance_id: str, **kwargs) -> None:
        """
        Release resources for the specified instance.

        Args:
            instance_id: Instance ID to release
            **kwargs: Additional arguments
        """
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]


class WebReadTool(BaseTool):
    """
    Read tool for retrieving and summarizing web page content.

    This tool fetches content from web pages, processes the content to extract
    key information relevant to a question, and returns summarized results with
    optional nested URLs.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        Initialize WebReadTool with configuration and schema.

        Args:
            config: Configuration dictionary containing:
                - num_workers: Max concurrent worker threads (default: 200)
                - rate_limit: Max concurrent requests (default: 200)
                - timeout: Request timeout in seconds (default: 300)
                - retries: Max retry attempts (default: 3)
                - delay_between_retries: Delay between retries in seconds (default: 5)
                - expose_nested_urls: Whether to include nested URLs in response (default: False)
                - max_num_urls_in_url_list: Max URLs per request (default: 5)
                - strategy_to_handle_too_many_urls: How to handle excess URLs
                - enable_global_rate_limit: Whether to apply rate limiting (default: True)

            tool_schema: OpenAI function tool schema definition
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}

        # Worker and rate limiting configuration
        self.num_workers = config.get("num_workers", 200)
        self.rate_limit = config.get("rate_limit", 200)
        self.timeout = config.get("timeout", 300)
        self.retries = config.get("retries", 3)
        self.delay_between_retries = config.get("delay_between_retries", 5)
        self.expose_nested_urls = config.get("expose_nested_urls", False)
        self.max_urls = config.get("max_num_urls_in_url_list", 5)
        self.strategy_to_handle_too_many_urls = config.get(
            "strategy_to_handle_too_many_urls", "cap_at_max_urls"
        )
        self.enable_global_rate_limit = config.get("enable_global_rate_limit", True)

        self.execution_pool = init_execution_pool(
            num_workers=self.num_workers,
            enable_global_rate_limit=self.enable_global_rate_limit,
            rate_limit=self.rate_limit,
            mode=PoolMode.ThreadMode,
        )

        logger.info(f"Initialized WebReadTool with config: {config}")

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        """
        Get the OpenAI tool schema.

        Returns:
            Tool schema in OpenAI format
        """
        return self.tool_schema

    async def create(
        self, instance_id: Optional[str] = None, **kwargs
    ) -> tuple[str, ToolResponse]:
        """
        Create a tool instance with the provided question.

        Args:
            instance_id: Optional instance ID (generated if not provided)
            **kwargs: Additional arguments containing:
                - create_kwargs.idx: Required instance index
                - create_kwargs.question: Required question string

        Returns:
            Tuple of (instance_id, empty tool response)

        Raises:
            ValueError: If idx or question is not provided
        """
        idx = kwargs.get("create_kwargs", {}).get("idx", None)
        if idx is None:
            raise ValueError("idx is not set")

        question = kwargs.get("create_kwargs", {}).get("question", None)
        if question is None:
            raise ValueError("question is not set")

        if instance_id is None:
            instance_id = str(uuid4())

        self._instance_dict[instance_id] = {
            "question": question,
        }
        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        """
        Read and summarize web pages with the provided parameters.

        Args:
            instance_id: The instance ID of the tool
            parameters: Dictionary containing:
                - url_list: List of URLs to read and summarize
            **kwargs: Additional arguments

        Returns:
            Tuple of (tool_response, reward_score, metrics) where:
                - tool_response: ToolResponse containing page summaries
                - reward_score: Reward score (always 0.0)
                - metrics: Dictionary with execution metrics
        """
        # Input validation
        if not isinstance(parameters, dict):
            error_message = f"Parsing failed: web_read tool call arguments should be a dict. Arguments: {parameters}."
            logger.warning(error_message)
            return ToolResponse(text=error_message), 0.0, {"success": False}

        if "url_list" not in parameters:
            error_message = f"Parsing failed: web_read tool call arguments should have a key 'url_list', but not found. Arguments: {parameters}."
            logger.warning(error_message)
            return ToolResponse(text=error_message), 0.0, {"success": False}

        if not isinstance(parameters["url_list"], list) or not all(
            isinstance(u, str) for u in parameters["url_list"]
        ):
            error_message = f"Parsing failed: web_read tool call 'url_list' should be a list of strings. Arguments: {parameters}."
            logger.warning(error_message)
            return ToolResponse(text=error_message), 0.0, {"success": False}

        # Process URL list
        url_list_from_params = [url.strip() for url in parameters["url_list"]]

        # Handle URL limits based on strategy
        if self.strategy_to_handle_too_many_urls == "cap_at_max_urls":
            url_list_from_params = url_list_from_params[: self.max_urls]
        elif self.strategy_to_handle_too_many_urls == "reject":
            if len(url_list_from_params) > self.max_urls:
                error_message = f"Parsing failed: web_read tool call 'url_list' should have at most {self.max_urls} urls but got {len(url_list_from_params)} urls. Arguments: {parameters}."
                logger.warning(error_message)
                return ToolResponse(text=error_message), 0.0, {"success": False}
        elif self.strategy_to_handle_too_many_urls == "do_all_urls":
            pass
        else:
            raise ValueError(
                f"Invalid strategy to handle too many urls: {self.strategy_to_handle_too_many_urls}"
            )

        # Execute read using Ray execution pool
        try:
            # Create Ray futures for each URL
            futures = [
                self.execution_pool.execute.remote(
                    make_request,
                    "read",
                    {
                        "url": url,
                        "question": self._instance_dict[instance_id]["question"],
                        "retries": 0,  # no retries in the server side
                    },
                    self.timeout,
                    self.retries,
                    self.delay_between_retries,
                )
                for url in url_list_from_params
            ]

            # Wait for Ray workers to complete
            results = await asyncio.gather(*futures)

            # Format results
            return_list = json.dumps(
                [
                    {
                        "url": url,
                        "information": result.get("summary", "No summary available"),
                        "nested_urls": result.get("url_items", [])
                        if self.expose_nested_urls
                        else [],
                    }
                    if isinstance(result, dict)
                    else {"url": url, "error message": result}
                    for url, result in zip(url_list_from_params, results)
                ]
            )

            logger.debug(return_list)
            return ToolResponse(text=return_list), 0.0, {"success": True}

        except Exception as e:
            return_message = f"[web_read] Execution failed: {e}"
            logger.warning(return_message)
            return (
                ToolResponse(text=return_message),
                0.0,
                {"error": str(e), "success": False},
            )

    async def release(self, instance_id: str, **kwargs) -> None:
        """
        Release resources for the specified instance.

        Args:
            instance_id: Instance ID to release
            **kwargs: Additional arguments
        """
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]

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

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel

from logging_utils import setup_colored_logger

load_dotenv()

logger = setup_colored_logger(__name__)


class SearchURLItem(BaseModel):
    """
    A single URL item from search results.

    Attributes:
        url: The full URL of the search result
        title: The title/headline of the webpage
        description: A brief description or snippet from the webpage
    """

    url: str
    title: str
    description: str


class SearchResult(BaseModel):
    """
    Structured results from a search query.

    Attributes:
        query: The original search query
        url_items: List of search result items with URL, title, and description
        raw_response: Original API response (truncated to 500 chars)
        success: Whether the search operation completed successfully
        metadata: Additional metadata about the search (provider, query, etc.)
        error: Error message if the search failed, empty string otherwise
    """

    query: str
    url_items: List[SearchURLItem]
    raw_response: str = ""
    success: bool
    metadata: Dict[str, Any]
    error: str = ""


def _extract_results_from_tavily_response(data: Dict[str, Any]) -> List[SearchURLItem]:
    """
    Extract URLs from Tavily API response.

    Args:
        data: The JSON response from Tavily API

    Returns:
        List of SearchURLItem objects from search results
    """
    results = data.get("results", [])
    return [
        SearchURLItem(
            url=item.get("url", ""),
            title=item.get("title", "No Title"),
            description=item.get("content", "No Description")[:200]
            if item.get("content")
            else "No Description",
        )
        for item in results
    ]


async def tavily_search(query: str, timeout: int = 30, top_k: int = 10) -> SearchResult:
    """
    Perform a search using Tavily API.

    Args:
        query: The search query string
        timeout: Maximum time in seconds to wait (default: 30)
        top_k: Maximum number of results to return (default: 10)

    Returns:
        SearchResult containing search results and metadata

    Example:
        >>> result = await tavily_search("Python programming")
        >>> if result.success:
        ...     for item in result.url_items:
        ...         print(f"{item.title}: {item.url}")
    """
    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        return SearchResult(
            query=query,
            url_items=[],
            success=False,
            metadata={
                "provider": "tavily",
                "status": 401,
                "execution_time": 0.0,
                "total_results": 0,
            },
            error="TAVILY_API_KEY environment variable not found",
        )

    url = "https://api.tavily.com/search"
    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": "basic",
        "max_results": top_k,
    }
    headers = {"Content-Type": "application/json"}

    loop = asyncio.get_running_loop()
    start_time = loop.time()

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=payload, headers=headers)
            execution_time = loop.time() - start_time

            if response.status_code == 200:
                data = response.json()
                raw_response = response.text
                url_items = _extract_results_from_tavily_response(data)

                logger.info(
                    f"Search successful for '{query}', found {len(url_items)} results"
                )

                return SearchResult(
                    query=query,
                    url_items=url_items,
                    raw_response=raw_response[:500],
                    success=True,
                    metadata={
                        "provider": "tavily",
                        "status": 200,
                        "execution_time": execution_time,
                        "total_results": len(url_items),
                        "answer": data.get("answer", ""),
                        "query": data.get("query", query),
                        "response_time": data.get("response_time", 0.0),
                        "scores": [
                            item.get("score", 0.0) for item in data.get("results", [])
                        ],
                    },
                )

            else:
                error_text = response.text
                logger.warning(
                    f"Search failed with HTTP {response.status_code}: {error_text[:100]}"
                )

                # Handle Tavily-specific errors
                if response.status_code == 429:
                    error_msg = "Rate limit exceeded. Please try again later."
                elif response.status_code == 401:
                    error_msg = "Invalid API key."
                else:
                    error_msg = error_text[:200]

                return SearchResult(
                    query=query,
                    url_items=[],
                    success=False,
                    metadata={
                        "provider": "tavily",
                        "status": response.status_code,
                        "execution_time": execution_time,
                        "total_results": 0,
                    },
                    error=f"HTTP {response.status_code}: {error_msg}",
                )

    except httpx.TimeoutException:
        logger.warning(f"Search request timed out after {timeout}s")
        return SearchResult(
            query=query,
            url_items=[],
            success=False,
            metadata={
                "provider": "tavily",
                "status": 408,
                "execution_time": loop.time() - start_time,
                "total_results": 0,
            },
            error=f"Search request timed out after {timeout}s",
        )

    except httpx.RequestError as e:
        logger.warning(f"Client error during search: {str(e)}")
        return SearchResult(
            query=query,
            url_items=[],
            success=False,
            metadata={
                "provider": "tavily",
                "status": 502,
                "execution_time": loop.time() - start_time,
                "total_results": 0,
            },
            error=f"Client error: {str(e)[:200]}",
        )

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON response: {e}")
        return SearchResult(
            query=query,
            url_items=[],
            success=False,
            metadata={
                "provider": "tavily",
                "status": 502,
                "execution_time": loop.time() - start_time,
                "total_results": 0,
            },
            error=f"Failed to parse JSON response: {str(e)[:200]}",
        )

    except Exception as e:
        logger.error(f"Unexpected error during search: {str(e)}")
        return SearchResult(
            query=query,
            url_items=[],
            success=False,
            metadata={
                "provider": "tavily",
                "status": 500,
                "execution_time": loop.time() - start_time,
                "total_results": 0,
            },
            error=f"Unexpected error: {str(e)[:200]}",
        )


class WebSearchAgent:
    """
    Agent for performing web searches with concurrency control.

    This agent handles search queries with semaphore-based rate limiting
    to prevent overwhelming the Tavily search API.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the WebSearchAgent.

        Args:
            config: Configuration dictionary with optional keys:
                - max_concurrent_requests: Max concurrent searches (default: 300)
        """
        self._timeout = config.get("timeout", 30)
        self._top_k = config.get("top_k", 10)
        self._semaphore = asyncio.Semaphore(config.get("max_concurrent_requests", 300))

    async def search(self, query: str) -> SearchResult:
        """
        Perform a web search with concurrency control.

        Args:
            query: Search query string to execute

        Returns:
            SearchResult containing search results and metadata

        Example:
            >>> agent = WebSearchAgent(config={})
            >>> result = await agent.search("Python programming")
            >>> if result.success:
            ...     for item in result.url_items:
            ...         print(item.title)
        """
        logger.info(f"Searching for '{query}'")

        try:
            async with self._semaphore:
                result = await tavily_search(
                    query, timeout=self._timeout, top_k=self._top_k
                )

                if not result.success:
                    logger.warning(
                        f"Search failed for '{query}' with status "
                        f"{result.metadata.get('status', 'unknown')}: {result.error}"
                    )

                return result

        except Exception as e:
            logger.error(
                f"Unexpected error in WebSearchAgent for '{query}': {str(e)}",
                exc_info=True,
            )
            return SearchResult(
                query=query,
                url_items=[],
                success=False,
                metadata={
                    "provider": "tavily",
                    "status": 500,
                    "execution_time": 0.0,
                    "total_results": 0,
                },
                error=f"WebSearchAgent error: {str(e)[:200]}",
            )

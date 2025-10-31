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
Tool Server Main Module

FastAPI server providing web reading and search capabilities
with caching and graceful shutdown handling.
"""

import argparse
import atexit
import signal
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from logging_utils import setup_colored_logger
from tool_server.cache_manager import CacheManager
from tool_server.read import ReadResult, WebReadAgent
from tool_server.search import SearchResult, WebSearchAgent

load_dotenv()

# Create logs directory if it doesn't exist
log_dir = Path("logs")
try:
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "tool_server.log"
    logger = setup_colored_logger("tool_server", log_file=log_file)
except Exception as e:
    # Fallback to console-only logging if log file creation fails
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("tool_server")
    logger.error(f"Failed to create log directory/file: {e}")
    logger.info("Falling back to console-only logging")


class SearchRequest(BaseModel):
    """Request model for web search."""

    query: str


class ReadRequest(BaseModel):
    """Request model for reading a webpage."""

    url: str
    question: str


# Parse command-line arguments
argparser = argparse.ArgumentParser(description="Tool Server for web operations")
# search args
argparser.add_argument(
    "--search-timeout", type=int, default=30, help="Default search timeout in seconds"
)
argparser.add_argument(
    "--max-concurrent-search-requests",
    type=int,
    default=300,
    help="Max concurrent search requests per agent",
)
argparser.add_argument(
    "--top-k", type=int, default=10, help="Default number of search results to return"
)

# read args
argparser.add_argument(
    "--read-timeout", type=int, default=30, help="Default read timeout in seconds"
)
argparser.add_argument(
    "--max-concurrent-read-requests",
    type=int,
    default=500,
    help="Max concurrent read requests per agent",
)
argparser.add_argument(
    "--max-content-words",
    type=int,
    default=10000,
    help="Max words to extract from webpage content",
)
argparser.add_argument(
    "--max-summary-words",
    type=int,
    default=2048,
    help="Max words in the generated summary",
)
argparser.add_argument(
    "--max-summary-retries",
    type=int,
    default=3,
    help="Max retries for summary generation on failure",
)

# cache args
argparser.add_argument(
    "--cache-dir",
    type=str,
    default="cache",
    help="Directory to store cache files",
)
argparser.add_argument(
    "--cache-save-frequency",
    type=int,
    default=100,
    help="Save cache to disk every N operations",
)
argparser.add_argument(
    "--cache-stats-interval",
    type=int,
    default=1200,
    help="Report cache statistics every N seconds (0 to disable)",
)
argparser.add_argument(
    "--enable-cache",
    action="store_true",
    help="Enable caching for search and read operations",
)

# server args
argparser.add_argument(
    "--port",
    type=int,
    default=8888,
    help="Port to run the server on",
)

args = argparser.parse_args()

# Initialize agents with configuration

search_config = {
    "timeout": args.search_timeout,
    "top_k": args.top_k,
    "max_concurrent_requests": args.max_concurrent_search_requests,
}
read_config = {
    "timeout": args.read_timeout,
    "max_concurrent_requests": args.max_concurrent_read_requests,
    "max_content_words": args.max_content_words,
    "max_summary_words": args.max_summary_words,
    "max_summary_retries": args.max_summary_retries,
}

# Initialize cache manager
cache_manager = None
if args.enable_cache:
    try:
        cache_manager = CacheManager(
            cache_dir=args.cache_dir,
            search_config=search_config,
            read_config=read_config,
            save_frequency=args.cache_save_frequency,
            stats_report_interval=args.cache_stats_interval,
        )
        logger.info("Cache manager enabled")
    except Exception as e:
        logger.error(f"Failed to initialize cache manager: {e}", exc_info=True)
        logger.warning("Continuing without cache support")
        cache_manager = None
else:
    logger.info("Cache manager disabled")

try:
    search_agent = WebSearchAgent(config=search_config)
    read_agent = WebReadAgent(config=read_config)
except Exception as e:
    logger.error(f"Failed to initialize agents: {e}", exc_info=True)
    raise


def cleanup_handler(signum=None, frame=None):
    """Handle cleanup on shutdown signals"""
    if signum is not None:
        signal_name = signal.Signals(signum).name
        logger.info(
            f"Received signal {signal_name} ({signum}), initiating graceful shutdown..."
        )
    else:
        logger.info("Initiating graceful shutdown...")

    if cache_manager:
        try:
            logger.info("Flushing cache to disk...")
            cache_manager.shutdown()
            logger.info("Cache successfully flushed to disk")
        except Exception as e:
            logger.error(f"Error during cache shutdown: {e}", exc_info=True)

    logger.info("Cleanup complete")


# Register signal handlers for graceful shutdown
def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown"""
    # Handle Ctrl+C (SIGINT) and kill (SIGTERM)
    signal.signal(signal.SIGINT, cleanup_handler)
    signal.signal(signal.SIGTERM, cleanup_handler)

    # Register atexit handler as fallback
    atexit.register(cleanup_handler)

    logger.debug("Signal handlers registered for graceful shutdown")


# Setup signal handlers
try:
    setup_signal_handlers()
except Exception as e:
    logger.error(f"Failed to setup signal handlers: {e}", exc_info=True)
    logger.warning("Continuing without signal handlers")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    try:
        logger.info("=" * 60)
        logger.info("Tool Server Starting")
        logger.info("=" * 60)
        logger.info("Server Configuration:")
        logger.info(f"  Port: {args.port}")
        logger.info("  Authentication: Disabled")
        logger.info(f"  Cache: {'Enabled' if args.enable_cache else 'Disabled'}")
        logger.info("")

        if args.enable_cache:
            logger.info("Cache Configuration:")
            logger.info(f"  Cache Directory: {args.cache_dir}")
            logger.info(
                f"  Save Frequency: every {args.cache_save_frequency} operations"
            )
            if args.cache_stats_interval > 0:
                logger.info(
                    f"  Stats Report Interval: every {args.cache_stats_interval} seconds"
                )
            else:
                logger.info("  Stats Report: Disabled")
            logger.info("")

        logger.info("Search Agent Configuration:")
        logger.info(f"  Max Concurrent Requests: {args.max_concurrent_search_requests}")
        logger.info(f"  Timeout: {args.search_timeout}s")
        logger.info(f"  Top K Results: {args.top_k}")
        logger.info("")
        logger.info("Read Agent Configuration:")
        logger.info(f"  Max Concurrent Requests: {args.max_concurrent_read_requests}")
        logger.info(f"  Timeout: {args.read_timeout}s")
        logger.info(f"  Max Content Words: {args.max_content_words}")
        logger.info(f"  Max Summary Words: {args.max_summary_words}")
        logger.info(f"  Max Summary Retries: {args.max_summary_retries}")
        logger.info("=" * 60)
        logger.info(f"Tool Server ready at http://0.0.0.0:{args.port}")
        logger.info("=" * 60)
    except Exception as e:
        logger.error(f"Error during startup logging: {e}", exc_info=True)
        # Continue anyway - logging errors shouldn't prevent startup

    try:
        yield  # <-- run the app
    except Exception as e:
        logger.error(f"Error during app lifecycle: {e}", exc_info=True)
        raise
    finally:
        # --- Shutdown ---
        try:
            logger.info("=" * 60)
            logger.info("Tool Server shutting down...")
            logger.info("=" * 60)

            # Shutdown cache manager
            if cache_manager:
                logger.info("Shutting down cache manager...")
                try:
                    cache_manager.shutdown()
                    logger.info("Cache manager shutdown complete")
                except Exception as e:
                    logger.error(
                        f"Error shutting down cache manager: {e}", exc_info=True
                    )
        except Exception as e:
            logger.error(f"Error during shutdown logging: {e}", exc_info=True)


# FastAPI app
app = FastAPI(
    title="Web Tools API",
    description="AI-powered web reading and search service with caching",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "message": "Tool Server is running",
        "status": "healthy",
        "services": ["read", "search"],
        "cache_enabled": args.enable_cache,
        "version": "1.0.0",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    health_data = {
        "status": "healthy",
        "service": "Tool Server",
        "agents": {
            "read": "operational",
            "search": "operational",
        },
    }

    # Add cache stats if enabled
    if cache_manager:
        health_data["cache"] = cache_manager.get_stats()

    return health_data


@app.get("/cache/stats")
async def cache_stats():
    """Get cache statistics"""
    if not cache_manager:
        raise HTTPException(
            status_code=400,
            detail="Cache is not enabled. Start server with --enable-cache",
        )

    return cache_manager.get_stats()


@app.post("/cache/flush")
async def cache_flush():
    """Force flush all caches to disk"""
    if not cache_manager:
        raise HTTPException(
            status_code=400,
            detail="Cache is not enabled. Start server with --enable-cache",
        )

    cache_manager.flush_all()
    return {"message": "All caches flushed to disk", "success": True}


@app.post("/cache/clear")
async def cache_clear():
    """Clear all caches (use with caution!)"""
    if not cache_manager:
        raise HTTPException(
            status_code=400,
            detail="Cache is not enabled. Start server with --enable-cache",
        )

    cache_manager.clear_all()
    return {"message": "All caches cleared", "success": True}


@app.post("/cache/invalidate")
async def cache_invalidate(max_age_days: int = 30):
    """Invalidate cache entries older than specified days"""
    if not cache_manager:
        raise HTTPException(
            status_code=400,
            detail="Cache is not enabled. Start server with --enable-cache",
        )

    removed = cache_manager.invalidate_old_entries(max_age_days=max_age_days)
    return {
        "message": f"Removed {removed} entries older than {max_age_days} days",
        "removed_count": removed,
        "success": True,
    }


@app.post("/search", response_model=SearchResult)
async def search_web(request: SearchRequest) -> SearchResult:
    """
    Perform web search and return structured results.

    This endpoint uses the Tavily API to perform web searches and returns
    structured results with URLs, titles, and descriptions.

    Args:
        request: SearchRequest containing query

    Returns:
        SearchResult with search results and metadata

    Raises:
        HTTPException: If search fails with status 500
    """
    try:
        logger.debug(f"Search request: {request.query}")

        # Check cache first
        if cache_manager:
            cached_result = cache_manager.get_search(request.query)
            if cached_result:
                logger.debug(f"Returning cached search result for: {request.query}")
                return SearchResult(**cached_result)

        # Perform search
        result = await search_agent.search(query=request.query)

        # Cache successful results
        if cache_manager and result.success:
            cache_manager.set_search(request.query, result.model_dump())

        if not result.success:
            logger.warning(
                f"Search failed for '{request.query}': {result.error}",
            )

        return result

    except Exception as e:
        logger.error(f"Error searching for '{request.query}': {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "success": False,
                "query": request.query,
            },
        )


@app.post("/read", response_model=ReadResult)
async def read_webpage(request: ReadRequest) -> ReadResult:
    """
    Read and analyze a webpage with LLM-powered summarization.

    This endpoint fetches webpage content and generates a context-aware summary
    based on the provided question.

    Args:
        request: ReadRequest containing url and question

    Returns:
        ReadResult with content, summary, and discovered URLs

    Raises:
        HTTPException: If reading fails with status 500
    """
    try:
        logger.debug(f"Read request: {request.url} - {request.question[:50]}...")

        # Check cache first
        if cache_manager:
            cached_result = cache_manager.get_read(request.url, request.question)
            if cached_result:
                logger.debug(f"Returning cached read result for: {request.url}")
                return ReadResult(**cached_result)

        # Perform read
        result = await read_agent.read(
            question=request.question,
            url=request.url,
        )

        # Cache successful results
        if cache_manager and result.success:
            cache_manager.set_read(request.url, request.question, result.model_dump())

        if not result.success:
            logger.warning(
                f"Read failed for {request.url}: {result.error}",
            )

        return result

    except Exception as e:
        logger.error(f"Error reading {request.url}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "success": False,
                "url": request.url,
            },
        )


if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting Tool Server on port {args.port}")

    # Write the actual port to a file that clients can read
    port_file = Path(".server_port")
    try:
        port_file.write_text(str(args.port))
        logger.info(f"Server port {args.port} written to {port_file}")
    except Exception as e:
        logger.error(f"Could not write port file: {e}", exc_info=True)

    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=args.port,
            log_level="info",
            access_log=True,
        )
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        raise  # Re-raise to ensure proper exit code
    finally:
        try:
            cleanup_handler()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}", exc_info=True)

        # Clean up port file on shutdown
        if port_file.exists():
            try:
                port_file.unlink()
                logger.debug("Port file cleaned up")
            except Exception as e:
                logger.warning(f"Could not delete port file on shutdown: {e}")

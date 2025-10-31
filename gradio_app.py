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
Gradio Web Interface for Pokee Deep Research Agent

This module provides a web-based interface for interacting with the deep research agent.
It supports both local model loading and VLLM server-based inference with concurrent
user sessions and real-time progress streaming.

Features:
    - Secure API key configuration with environment variable storage
    - Manual tool server lifecycle management
    - Multiple concurrent user sessions with independent state management
    - Real-time research progress updates with tool call visibility
    - Graceful cancellation and cleanup of research tasks
    - Support for both local and VLLM-based agent backends
    - Customizable research parameters (temperature, top_p, max_turns)

Usage:
    # Start with VLLM backend (recommended for faster inference)
    python gradio_app.py --serving-mode vllm --vllm-url http://localhost:9999/v1

    # Start with local backend (single GPU)
    python gradio_app.py --serving-mode local --model-path path/to/model

    # Enable public sharing
    python gradio_app.py --share --port 7777
"""

import argparse
import asyncio
import atexit
import os
import socket
import subprocess
import sys
import threading
import time

import gradio as gr
import requests
import torch

from logging_utils import setup_colored_logger

logger = setup_colored_logger(__name__)

# Global configuration (read-only after initialization, safe for concurrent access)
serving_mode = None
agent_config = {}

# Track running research tasks per session
running_tasks: dict[str, asyncio.Task] = {}

# Global tool server process and configuration
tool_server_proc = None
tool_server_port = 8888


def is_port_available(port: int) -> bool:
    """Check if a TCP port is available for binding.

    Args:
        port: Port number to check (1024-65535)

    Returns:
        bool: True if port is available, False otherwise
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("localhost", port))
            return True
    except OSError:
        return False


def start_tool_server(port: int, timeout: int = 30) -> subprocess.Popen:
    """Start the tool server as a background subprocess.

    Launches the tool server process and waits for it to become healthy
    by polling the health endpoint. Logs stderr output in a background thread.

    Args:
        port: Port number for the tool server (1024-65535)
        timeout: Maximum seconds to wait for server readiness

    Returns:
        subprocess.Popen: The running server process

    Raises:
        RuntimeError: If server fails to start, crashes, or doesn't become ready within timeout
    """
    logger.info(f"Starting tool server on port {port}...")

    # Start the server process with proper output handling
    proc = subprocess.Popen(
        [sys.executable, "start_tool_server.py", "--port", str(port), "--enable-cache"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=os.environ.copy(),  # Pass current environment with API keys
    )

    # Start background thread to log stderr
    def log_stderr():
        for line in proc.stderr:
            logger.debug(f"[Tool Server] {line.rstrip()}")

    stderr_thread = threading.Thread(target=log_stderr, daemon=True)
    stderr_thread.start()

    # Check if process started successfully
    time.sleep(0.5)
    if proc.poll() is not None:
        remaining_stderr = proc.stderr.read()
        logger.error(f"Tool server failed to start: {remaining_stderr}")
        raise RuntimeError(
            f"Tool server process terminated immediately: {remaining_stderr}"
        )

    # Wait for server to become ready by polling health endpoint
    server_url = f"http://localhost:{port}"
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            response = requests.get(server_url, timeout=1.0)
            if response.status_code == 200:
                health_data = response.json()
                if health_data.get("status") == "healthy":
                    logger.info(f"✅ Tool server is ready at {server_url}")
                    return proc
        except requests.exceptions.RequestException:
            pass

        # Check if process crashed
        if proc.poll() is not None:
            logger.error("Tool server crashed during startup")
            raise RuntimeError("Tool server crashed during startup")

        time.sleep(0.5)

    # Timeout reached - terminate process
    logger.error("Tool server failed to become ready within timeout")
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()

    raise RuntimeError(f"Tool server failed to become ready within {timeout}s")


def cleanup_tool_server(proc: subprocess.Popen):
    """Gracefully shutdown the tool server process.

    Attempts graceful termination first, then forces kill if necessary.

    Args:
        proc: The server process to shutdown (can be None)
    """
    if proc is None:
        return

    if proc.poll() is None:  # Process still running
        logger.info("Shutting down tool server...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
            logger.info("Tool server shut down gracefully")
        except subprocess.TimeoutExpired:
            logger.warning("Tool server did not terminate, forcing kill...")
            proc.kill()
            proc.wait()
            logger.info("Tool server killed")


def get_tool_server_status() -> dict:
    """Check if tool server is running and healthy.

    Checks both process status and health endpoint response.

    Returns:
        dict: Status information with keys:
            - 'running' (bool): True if server is healthy
            - 'message' (str): Human-readable status message
    """
    global tool_server_proc

    if tool_server_proc is None:
        return {"running": False, "message": "Tool server not started"}

    if tool_server_proc.poll() is not None:
        return {"running": False, "message": "Tool server has stopped"}

    try:
        response = requests.get(f"http://localhost:{tool_server_port}", timeout=2.0)
        if response.status_code == 200:
            health_data = response.json()
            if health_data.get("status") == "healthy":
                return {"running": True, "message": "Tool server is healthy"}
    except requests.exceptions.RequestException:
        pass

    return {"running": False, "message": "Tool server not responding"}


def save_api_keys(tavily_key: str, openai_key: str) -> str:
    """Save API keys to environment variables.

    Validates that all keys are provided and stores them in the current process
    environment for use by the tool server.

    Args:
        tavily_key: Tavily API key for web search functionality
        openai_key: OpenAI API key for content summarization with GPT-5

    Returns:
        str: Status message for UI display (includes ✅/❌ emoji)
    """
    try:
        # Validate that all keys are provided
        if not all([tavily_key, openai_key]):
            return "❌ Please provide all API keys"

        # Set environment variables
        if tavily_key:
            os.environ["TAVILY_API_KEY"] = tavily_key.strip()
            logger.info("✅ Tavily API key configured")

        if openai_key:
            os.environ["OPENAI_API_KEY"] = openai_key.strip()
            logger.info("✅ OpenAI API key configured")

        return "✅ API keys saved successfully! You can now start the tool server."

    except Exception as e:
        logger.error(f"Failed to save API keys: {e}")
        return f"❌ Failed to save API keys: {str(e)}"


def start_tool_server_ui(port: int) -> tuple[str, str]:
    """Start the tool server from UI button click.

    Validates port availability and API keys, then starts the server process.

    Args:
        port: Port number to use for the tool server (1024-65535)

    Returns:
        tuple[str, str]: (status_message, server_status_display) for UI updates
    """
    global tool_server_proc, tool_server_port

    # Check if already running
    if tool_server_proc is not None and tool_server_proc.poll() is None:
        return "⚠️ Tool server is already running", "🟢 Running"

    # Validate port range
    if not (1024 <= port <= 65535):
        return "❌ Invalid port number. Must be between 1024 and 65535", "🔴 Stopped"

    # Check port availability
    if not is_port_available(port):
        return (
            f"❌ Port {port} is already in use. Please choose another port.",
            "🔴 Stopped",
        )

    # Update global port
    tool_server_port = port

    # Check if API keys are configured
    required_keys = ["TAVILY_API_KEY", "OPENAI_API_KEY"]
    missing_keys = [key for key in required_keys if not os.environ.get(key)]

    if missing_keys:
        logger.warning(f"Missing API keys: {', '.join(missing_keys)}")
        return (
            f"⚠️ Warning: Missing API keys: {', '.join(missing_keys)}\n"
            "Tool server will start but some features may not work.",
            "🟡 Starting...",
        )

    try:
        tool_server_proc = start_tool_server(tool_server_port)
        return (
            f"✅ Tool server started successfully on port {tool_server_port}!",
            "🟢 Running",
        )
    except Exception as e:
        logger.error(f"Failed to start tool server: {e}")
        return f"❌ Failed to start tool server: {str(e)}", "🔴 Stopped"


def stop_tool_server_ui() -> tuple[str, str]:
    """Stop the tool server from UI button click.

    Returns:
        tuple[str, str]: (status_message, server_status_display) for UI updates
    """
    global tool_server_proc

    if tool_server_proc is None or tool_server_proc.poll() is not None:
        return "⚠️ Tool server is not running", "🔴 Stopped"

    try:
        cleanup_tool_server(tool_server_proc)
        tool_server_proc = None
        return "✅ Tool server stopped successfully", "🔴 Stopped"
    except Exception as e:
        logger.error(f"Failed to stop tool server: {e}")
        return f"❌ Failed to stop tool server: {str(e)}", "🔴 Stopped"


def refresh_server_status() -> str:
    """Refresh and return current tool server status.

    Polls the server and returns formatted status string for UI display.

    Returns:
        str: Server status display with emoji indicator (🟢/🔴)
    """
    status = get_tool_server_status()
    if status["running"]:
        return f"🟢 Running - {status['message']}"
    else:
        return f"🔴 Stopped - {status['message']}"


def create_agent():
    """Create a new agent instance for a user session.

    Factory function that creates the appropriate agent type based on
    the global serving_mode configuration.

    Returns:
        VLLMDeepResearchAgent | SimpleDeepResearchAgent: Configured agent instance
    """
    if serving_mode == "vllm":
        from agent.vllm_agent import VLLMDeepResearchAgent

        return VLLMDeepResearchAgent(
            vllm_url=agent_config["vllm_url"],
            model_name=agent_config["model_path"],
            tool_config_path=agent_config["tool_config_path"],
            max_turns=agent_config["max_turns"],
        )
    else:
        from agent.simple_agent import SimpleDeepResearchAgent

        return SimpleDeepResearchAgent(
            model_path=agent_config["model_path"],
            tool_config_path=agent_config["tool_config_path"],
            device=agent_config["device"],
            max_turns=agent_config["max_turns"],
        )


def get_session_id(request: gr.Request) -> str:
    """Get unique session ID for the current user.

    Extracts session hash from Gradio request for session-specific state management.

    Args:
        request: Gradio request object containing session information

    Returns:
        str: Unique session identifier (or "default" if unavailable)
    """
    if request and hasattr(request, "session_hash"):
        return request.session_hash
    return "default"


async def research_stream(
    question: str,
    temperature: float,
    top_p: float,
    max_turns: int,
    request: gr.Request,
):
    """Execute research with real-time step-by-step updates.

    This async generator streams research progress updates to the UI, showing:
    - Tool calls (web searches, web reads)
    - Agent thinking process
    - Verification steps
    - Final answer or errors

    Manages session-specific task tracking for cancellation support.

    Args:
        question: Research question from user
        temperature: Sampling temperature for generation (0.0-1.0, lower = more focused)
        top_p: Nucleus sampling parameter (0.0-1.0)
        max_turns: Maximum number of agent iterations allowed
        request: Gradio request object for session tracking

    Yields:
        tuple[str, str]: (thinking_log, answer) for progressive UI updates
    """
    session_id = get_session_id(request)

    # Check if tool server is running
    status = get_tool_server_status()
    if not status["running"]:
        error_msg = (
            "❌ Tool server is not running! Please start it in the Setup tab first."
        )
        yield error_msg, error_msg
        return

    logger.info(f"Starting research for session {session_id[:8]}...")

    agent = None
    thinking_log = ""

    try:
        agent = create_agent()
        current_task = asyncio.current_task()
        running_tasks[session_id] = current_task

        async for update in agent.run_stream(
            question_raw=question,
            temperature=temperature,
            top_p=top_p,
            max_turns=int(max_turns),
        ):
            update_type = update["type"]

            if update_type == "answer_found":
                think = update.get("think", "")
                if think:
                    thinking_log += f"\n\n💭 **Thinking:**\n\n{think}\n"
                thinking_log += "\n\n🔎 **Verifying answer...**\n"
                yield thinking_log, ""

            elif update_type == "tool_call":
                think = update.get("think", "")
                if think:
                    thinking_log += f"\n\n💭 **Thinking:**\n\n{think}\n"

                tool_name = update["tool_name"]
                if tool_name == "web_search":
                    queries = update["queries"]
                    thinking_log += (
                        f"\n\n🔍 **Searching:** {len(queries)} querie(s)\n\n"
                    )
                    for q in queries:
                        thinking_log += f"- {q}\n"

                elif tool_name == "web_read":
                    urls = update["urls"]
                    thinking_log += f"\n\n📖 **Reading:** {len(urls)} URL(s)\n\n"
                    for url in urls[:3]:
                        thinking_log += f"- {url}\n"
                    if len(urls) > 3:
                        thinking_log += f"- ... and {len(urls) - 3} more\n"

                yield thinking_log, ""

            elif update_type == "done":
                thinking_log += "\n\n✅ **Research completed!**\n"
                yield thinking_log, update["answer"]
                break

            elif update_type == "error":
                logger.error(
                    f"Research error for session {session_id[:8]}: {update['message']}",
                    exc_info=True,
                )
                thinking_log += "\n\n❌ **Error occurred during research.**\n"
                yield thinking_log, "❌ Error occurred during research."
                break

    except asyncio.CancelledError:
        logger.info(f"Research cancelled for session {session_id[:8]}")
        if agent:
            await agent.cleanup_tool_instances()
        thinking_log += "\n\n❌ **Cancelled by user**\n"
        yield thinking_log, "❌ Research cancelled by user."
        raise

    except Exception as e:
        logger.error(f"Research error for session {session_id[:8]}: {e}", exc_info=True)
        thinking_log += "\n\n❌ **Unexpected error occurred.**"
        yield thinking_log, "❌ Error occurred during research."

    finally:
        if session_id in running_tasks:
            del running_tasks[session_id]


def cancel_research(request: gr.Request):
    """Cancel the currently running research task for this session.

    Looks up the session's task and requests cancellation via asyncio.

    Args:
        request: Gradio request object for session tracking

    Returns:
        tuple[gr.update, gr.update]: UI update objects for progress and answer outputs
    """
    session_id = get_session_id(request)

    if session_id not in running_tasks:
        logger.debug(f"No running research for session {session_id[:8]}")
        return gr.update(), gr.update()

    task = running_tasks[session_id]
    if task.done():
        logger.debug(f"Task already done for session {session_id[:8]}")
        return gr.update(), gr.update()

    try:
        task.cancel()
        logger.info(f"Cancellation requested for session {session_id[:8]}")
        return "\n\n🛑 **Cancellation requested...**", gr.update()
    except Exception as e:
        logger.error(f"Failed to cancel session {session_id[:8]}: {e}")
        return f"\n\n❌ **Failed to cancel:** {e}", gr.update()


def create_demo():
    """Create and configure the Gradio interface.

    Builds a multi-tab interface with:
    - Setup tab: API configuration and tool server management
    - Research tab: Question input and real-time progress display
    - About tab: Documentation and usage tips

    Returns:
        gr.Blocks: Configured Gradio interface ready to launch
    """
    serving_mode_display = "VLLM" if serving_mode == "vllm" else "Local"

    with gr.Blocks(
        title="Pokee Deep Research Agent",
        css="""
        .progress-box {
            height: 500px !important;
            max-height: 500px !important;
            overflow-y: auto !important;
            border: 2px solid var(--border-color-primary) !important;
            border-radius: 8px !important;
            padding: 16px !important;
            background-color: var(--background-fill-secondary) !important;
        }
        .progress-box::-webkit-scrollbar {
            width: 8px;
        }
        .progress-box::-webkit-scrollbar-track {
            background: var(--background-fill-primary);
            border-radius: 4px;
        }
        .progress-box::-webkit-scrollbar-thumb {
            background: var(--border-color-primary);
            border-radius: 4px;
        }
        .progress-box::-webkit-scrollbar-thumb:hover {
            background: var(--border-color-accent);
        }
        .status-indicator {
            font-size: 1.2em;
            font-weight: bold;
            padding: 10px;
            border-radius: 5px;
            text-align: center;
        }
        """,
    ) as demo:
        gr.Markdown(
            f"""
            # 🔬 Pokee Deep Research Agent
            
            An AI agent that performs deep research using multiple tools to answer complex questions.
            
            **LLM Serving Mode**: {serving_mode_display}
            """
        )

        with gr.Tab("🔧 Setup"):
            gr.Markdown(
                """
                ## API Configuration & Tool Server Setup
                
                **⚠️ Important**: Configure your API keys and start the tool server before conducting research.
                
                ### Required API Keys:
                - **Tavily API**: For web search functionality ([Get key](https://tavily.com))
                - **OpenAI API**: For read content summarization with GPT-5 ([Get key](https://platform.openai.com/api-keys))
                
                **Note:** Web content reading uses Playwright browser automation (no API key required).
                """
            )

            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### 1️⃣ Configure API Keys")

                    tavily_input = gr.Textbox(
                        label="Tavily API Key",
                        placeholder="Enter your Tavily API key...",
                        type="password",
                        value=os.environ.get("TAVILY_API_KEY", ""),
                    )

                    openai_input = gr.Textbox(
                        label="OpenAI API Key",
                        placeholder="Enter your OpenAI API key...",
                        type="password",
                        value=os.environ.get("OPENAI_API_KEY", ""),
                    )

                    save_keys_btn = gr.Button(
                        "💾 Save API Keys", variant="primary", size="lg"
                    )
                    save_status = gr.Markdown("")

                with gr.Column(scale=1):
                    gr.Markdown("### 2️⃣ Tool Server Control")

                    port_input = gr.Number(
                        label="Tool Server Port",
                        value=tool_server_port,
                        minimum=1024,
                        maximum=65535,
                        step=1,
                        info="Port number for the tool server (1024-65535)",
                    )

                    server_status_display = gr.Markdown(
                        refresh_server_status(), elem_classes=["status-indicator"]
                    )

                    with gr.Row():
                        start_server_btn = gr.Button(
                            "▶️ Start Server", variant="primary", size="lg"
                        )
                        stop_server_btn = gr.Button(
                            "⏹️ Stop Server", variant="stop", size="lg"
                        )

                    refresh_status_btn = gr.Button("🔄 Refresh Status", size="sm")

                    server_message = gr.Markdown("")

            gr.Markdown(
                """
                ---
                ### 📝 Setup Checklist:
                
                1. ✅ Enter your API keys above
                2. ✅ Click "Save API Keys"
                3. ✅ Configure tool server port (if needed)
                4. ✅ Click "Start Server" and wait for green status
                5. ✅ Go to "Research" tab to start asking questions
                """
            )

        with gr.Tab("🔍 Research"):
            with gr.Row():
                with gr.Column(scale=2):
                    question_input = gr.Textbox(
                        label="Question",
                        placeholder="Enter your research question here...",
                        lines=3,
                    )

                    with gr.Accordion("Advanced Settings", open=False):
                        temperature_slider = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.1,
                            step=0.1,
                            label="Temperature",
                            info="Higher values = more creative, lower = more focused",
                        )
                        top_p_slider = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.1,
                            step=0.05,
                            label="Top P",
                            info="Nucleus sampling threshold",
                        )
                        max_turns_research = gr.Slider(
                            minimum=1,
                            maximum=30,
                            value=10,
                            step=1,
                            label="Max Turns",
                            info="Maximum research iterations",
                        )

                    with gr.Row():
                        submit_btn = gr.Button(
                            "🔍 Research", variant="primary", size="lg"
                        )
                        cancel_btn = gr.Button("⛔ Cancel", variant="stop", size="lg")

                with gr.Column(scale=2):
                    with gr.Group(elem_classes=["progress-box-container"]):
                        progress_output = gr.Markdown(
                            value="***Research progress will appear here...***",
                            elem_classes=["progress-box"],
                        )

                    answer_output = gr.Textbox(
                        label="Answer",
                        lines=10,
                        max_lines=20,
                        show_copy_button=True,
                    )

            gr.Examples(
                examples=[
                    ["What is the capital of France?"],
                    ["Who won the 2024 Nobel Prize in Physics?"],
                    ["What are the latest developments in quantum computing?"],
                ],
                inputs=question_input,
            )

        with gr.Tab("ℹ️ About"):
            gr.Markdown(
                """
                ## About Deep Research Agent
                
                This AI agent performs comprehensive research on your questions by:
                1. **Searching** for relevant information using web search
                2. **Reading** and analyzing web content
                3. **Verifying** answers for accuracy
                4. **Iterating** until a satisfactory answer is found
                
                ### Usage Tips
                
                - **Setup First**: Always configure API keys and start the tool server in the Setup tab
                - **Complex questions** may require more turns (adjust in Advanced Settings)
                - **Temperature**: Lower (0.1) = more focused, Higher (1.0) = more creative
                - **Cancellation**: Use the Cancel button to stop long-running research
                - **Progress tracking**: Watch the Research Progress panel for real-time updates
                
                ### Features
                
                - ✅ Secure API key management
                - ✅ Manual tool server control
                - ✅ Multiple concurrent users supported
                - ✅ Independent session management
                - ✅ Real-time progress updates
                - ✅ Graceful cancellation and cleanup
                - ✅ Live research progress tracking with tool visibility
                """
            )

        # Event handlers for Setup tab
        save_keys_btn.click(
            fn=save_api_keys,
            inputs=[tavily_input, openai_input],
            outputs=[save_status],
        )

        start_server_btn.click(
            fn=start_tool_server_ui,
            inputs=[port_input],
            outputs=[server_message, server_status_display],
        )

        stop_server_btn.click(
            fn=stop_tool_server_ui,
            inputs=[],
            outputs=[server_message, server_status_display],
        )

        refresh_status_btn.click(
            fn=refresh_server_status,
            inputs=[],
            outputs=[server_status_display],
        )

        # Event handlers for Research tab
        submit_event = submit_btn.click(
            fn=research_stream,
            inputs=[
                question_input,
                temperature_slider,
                top_p_slider,
                max_turns_research,
            ],
            outputs=[progress_output, answer_output],
            concurrency_limit=30,
        )

        cancel_btn.click(
            fn=cancel_research,
            inputs=[],
            outputs=[progress_output, answer_output],
            cancels=[submit_event],
            concurrency_limit=30,
        )

    return demo


def main():
    """Main entry point for the Gradio application.

    Parses command-line arguments, configures the agent backend, optionally
    pre-loads models (for local mode), and launches the Gradio interface.
    Registers cleanup handlers for graceful shutdown.
    """
    global tool_server_proc

    parser = argparse.ArgumentParser(
        description="Pokee Deep Research Agent - Web Interface"
    )
    parser.add_argument(
        "--serving-mode",
        type=str,
        choices=["local", "vllm"],
        default="local",
        help="Serving mode: 'local' (single GPU) or 'vllm' (server-based)",
    )
    parser.add_argument(
        "--vllm-url",
        type=str,
        default="http://localhost:9999/v1",
        help="VLLM server URL (required for --serving-mode vllm)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="PokeeAI/pokee_research_7b",
        help="Model path or HuggingFace model ID",
    )
    parser.add_argument(
        "--tool-config",
        type=str,
        default="config/tool_config/pokee_tool_config.yaml",
        help="Tool configuration file path",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio share link",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7777,
        help="Port to run the web interface on",
    )
    parser.add_argument(
        "--server-name",
        type=str,
        default="0.0.0.0",
        help="Server name/address to bind to (0.0.0.0 for all interfaces, 127.0.0.1 for local only)",
    )

    args = parser.parse_args()

    # Configure global settings
    global serving_mode, agent_config
    serving_mode = args.serving_mode
    agent_config = {
        "model_path": args.model_path,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "max_turns": 10,
        "tool_config_path": args.tool_config,
        "vllm_url": args.vllm_url if args.serving_mode == "vllm" else None,
    }

    # Validate configuration
    if args.serving_mode == "vllm" and not args.vllm_url:
        parser.error("--vllm-url is required when using --serving-mode vllm")

    logger.info(f"Starting Gradio app with {args.serving_mode.upper()} serving mode")
    logger.info(f"Configuration: {agent_config}")

    # Pre-initialize resources for local agent
    if args.serving_mode == "local":
        logger.info("Pre-loading model...")
        create_agent()
        logger.info("Model loaded successfully!")
    else:
        logger.info("Using VLLM server (no pre-loading needed)")

    # Register cleanup handler for tool server
    atexit.register(cleanup_tool_server, tool_server_proc)

    # Launch interface
    try:
        demo = create_demo()
        demo.launch(share=args.share, server_port=args.port, server_name=args.server_name)
    finally:
        if tool_server_proc is not None:
            cleanup_tool_server(tool_server_proc)


if __name__ == "__main__":
    main()

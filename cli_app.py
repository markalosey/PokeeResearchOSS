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
Deep Research Agent - User Interface
This script provides a simple CLI interface to interact with the trained deep research agent.
"""

import argparse
import asyncio
import time

import torch

from logging_utils import setup_colored_logger

logger = setup_colored_logger("cli_app")


async def interactive_mode_async(
    agent,
    temperature: float,
    top_p: float,
    verbose: bool,
):
    """Async interactive mode loop."""
    while True:
        try:
            question = input("\nYou: ").strip()

            if not question:
                continue

            if question.lower() in ["exit", "quit"]:
                print("\nGoodbye!")
                break

            print("\nAgent: Researching...\n")
            start_time = time.time()
            answer = await agent.run(
                question_raw=question,
                temperature=temperature,
                top_p=top_p,
                verbose=verbose,
            )

            print(f"\nAgent: {answer}\n")
            print("Time taken: {:.2f} seconds".format(time.time() - start_time))
            print("-" * 80)

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            logger.error(f"\nError: {e}")
            if verbose:
                import traceback

                traceback.print_exc()


def interactive_mode(
    serving_mode: str,
    model_path: str,
    tool_config_path: str,
    device: str,
    max_turns: int,
    temperature: float,
    top_p: float,
    verbose: bool,
    vllm_url: str = None,
):
    """Run interactive mode."""
    # Create agent based on type
    if serving_mode == "vllm":
        if not vllm_url:
            raise ValueError("VLLM URL must be provided when using VLLM agent")
        from agent.vllm_agent import VLLMDeepResearchAgent

        logger.info(f"Using VLLM agent at {vllm_url}")
        agent = VLLMDeepResearchAgent(
            vllm_url=vllm_url,
            model_name=model_path,
            tool_config_path=tool_config_path,
            max_turns=max_turns,
        )
    else:
        from agent.simple_agent import SimpleDeepResearchAgent

        logger.info("Using local model agent")
        agent = SimpleDeepResearchAgent(
            model_path=model_path,
            tool_config_path=tool_config_path,
            device=device,
            max_turns=max_turns,
            use_quantization=True,  # Enable quantization by default for memory efficiency
            quantization_bits=4,  # 4-bit quantization reduces model to ~4-5GB
        )

    print("\n" + "=" * 80)
    print("Deep Research Agent - Interactive Mode")
    print(f"Serving Mode: {serving_mode.upper()}")
    print(f"Model: {model_path}")
    print("=" * 80)
    print("Type 'exit' or 'quit' to end the session")
    print("=" * 80 + "\n")

    # Run entire interactive session in single event loop
    asyncio.run(interactive_mode_async(agent, temperature, top_p, verbose))


def single_query_mode(
    question: str,
    serving_mode: str,
    model_path: str,
    tool_config_path: str,
    device: str,
    max_turns: int,
    temperature: float,
    top_p: float,
    verbose: bool,
    vllm_url: str = None,
) -> str:
    """Run single query."""
    # Create agent based on type
    if serving_mode == "vllm":
        if not vllm_url:
            raise ValueError("VLLM URL must be provided when using VLLM agent")
        from agent.vllm_agent import VLLMDeepResearchAgent

        logger.info(f"Using VLLM agent at {vllm_url}")
        agent = VLLMDeepResearchAgent(
            vllm_url=vllm_url,
            model_name=model_path,
            tool_config_path=tool_config_path,
            max_turns=max_turns,
        )
    else:
        from agent.simple_agent import SimpleDeepResearchAgent

        logger.info("Using local model agent")
        agent = SimpleDeepResearchAgent(
            model_path=model_path,
            tool_config_path=tool_config_path,
            device=device,
            max_turns=max_turns,
            use_quantization=True,  # Enable quantization by default for memory efficiency
            quantization_bits=4,  # 4-bit quantization reduces model to ~4-5GB
        )

    start_time = time.time()
    try:
        answer = asyncio.run(
            agent.run(
                question_raw=question,
                temperature=temperature,
                top_p=top_p,
                verbose=verbose,
            )
        )
        print("Time taken: {:.2f} seconds".format(time.time() - start_time))
        return answer
    except Exception as e:
        logger.error(f"\nError: {e}")
        if verbose:
            import traceback

            traceback.print_exc()
        return "Error occurred while processing the query."


def main():
    parser = argparse.ArgumentParser(
        description="Deep Research Agent - User Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode with local model
  python cli_app.py --serving-mode local
  
  # Interactive mode with VLLM
  python cli_app.py --serving-mode vllm --vllm-url http://localhost:9999/v1
  
  # Single query with VLLM
  python cli_app.py --serving-mode vllm --vllm-url http://localhost:9999/v1 --question "What is the capital of France?"
        """,
    )
    parser.add_argument(
        "--serving-mode",
        type=str,
        choices=["local", "vllm"],
        default="local",
        help="Serving mode to use: 'local' for local model loading, 'vllm' for VLLM server",
    )
    parser.add_argument(
        "--vllm-url",
        type=str,
        default="http://localhost:9999/v1",
        help="URL of the VLLM server (required when using --serving-mode vllm)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="PokeeAI/pokee_research_7b",
        help="Path to model or HuggingFace model ID",
    )
    parser.add_argument(
        "--tool-config",
        type=str,
        default="config/tool_config/pokee_tool_config.yaml",
        help="Path to tool configuration file",
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Single question to answer (non-interactive mode)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu) - only used with local agent",
    )
    parser.add_argument(
        "--max-turns", type=int, default=10, help="Maximum number of agent turns"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature"
    )
    parser.add_argument(
        "--top-p", type=float, default=0.9, help="Nucleus sampling parameter"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Validate VLLM URL if using VLLM
    if args.serving_mode == "vllm" and not args.vllm_url:
        parser.error("--vllm-url is required when using --serving-mode vllm")

    if args.question:
        # Single query mode
        answer = single_query_mode(
            question=args.question,
            serving_mode=args.serving_mode,
            model_path=args.model_path,
            tool_config_path=args.tool_config,
            device=args.device,
            max_turns=args.max_turns,
            temperature=args.temperature,
            top_p=args.top_p,
            verbose=args.verbose,
            vllm_url=args.vllm_url,
        )
        print(f"\nQuestion: {args.question}")
        print(f"\nAnswer: {answer}\n")
    else:
        # Interactive mode
        interactive_mode(
            serving_mode=args.serving_mode,
            model_path=args.model_path,
            tool_config_path=args.tool_config,
            device=args.device,
            max_turns=args.max_turns,
            temperature=args.temperature,
            top_p=args.top_p,
            verbose=args.verbose,
            vllm_url=args.vllm_url,
        )


if __name__ == "__main__":
    main()

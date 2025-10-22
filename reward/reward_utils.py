
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



import json
import logging
import re
import string
from typing import Optional

logger = logging.getLogger("reward_func")
logger.setLevel(logging.WARNING)


def preprocess_text(text: str) -> str:
    """Preprocess text for dataset scoring

    Processing steps:
    1. Remove punctuation (.,!?;:'"()[]{}...)
    2. Remove extra spaces
    """
    # Replace punctuation with spaces
    for punct in string.punctuation:
        text = text.replace(punct, " ")

    # Replace multiple spaces with a single space
    text = re.sub(r"\s+", " ", text)

    # Remove leading and trailing spaces
    text = text.strip()
    return text


def matches_pattern(s: str, use_answer_verification: bool = False) -> bool:
    # s starts without assistant\n and ends without user\n, so we add them here
    format_check_passed, err = check_agent_response_format(
        "assistant\n" + s + "user\n", use_answer_verification
    )
    return format_check_passed


def check_agent_response_format(
    full_response: str, use_answer_verification: bool = False
):
    """
    Validates the structure of an entire multi-turn agent response sequence.

    Requirements:
      - Split by assistant\n ... user\n boundaries.
      - Each assistant message must be one of:
          <think>...</think><tool_call>...</tool_call>
          <think>...</think><answer>...</answer>
          <verification>...</verification><verification_result>...</verification_result> (if use_answer_verification=True)
      - tool_call JSON must strictly follow one of three forms:
          1. {"name": "web_search", "arguments": {"query_list": [str, ...]}}
          2. {"name": "web_read", "arguments": {"url_list": [str, ...]}}
          3. {"name": "web_browse", "arguments": {"browsing_job_list": [{"url": str, "task": str}, ...]}}
      - If use_answer_verification=True:
          - There must be at least one <think>...</think><answer>...</answer> before verification.
          - Verification block must immediately follow an answer.
          - <verification_result> must be CORRECT or INCORRECT.
          - If CORRECT → must be final block.
          - If INCORRECT → process restarts from the next assistant block.
      - If use_answer_verification=False:
          - Response ends with <think>...</think><answer>...</answer>
    """

    # --- Split into assistant messages ---
    blocks = re.split(r"user\n", full_response)
    assistant_blocks = []
    for b in blocks:
        parts = re.split(r"assistant\n", b)
        if len(parts) > 1:
            assistant_blocks.append(parts[-1].strip())

    if not assistant_blocks:
        return False, f"No assistant messages found. {full_response}"

    # --- Regex patterns ---
    pattern_tool_call = re.compile(
        r"^\s*<think>.*?</think>\s*<tool_call>(?P<tool_call>.*?)</tool_call>\s*$",
        re.DOTALL,
    )
    pattern_answer = re.compile(
        r"^\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$", re.DOTALL
    )
    pattern_verification = re.compile(
        r"^\s*<verification>.*?</verification>\s*<verification_result>(?P<result>.*?)</verification_result>\s*$",
        re.DOTALL,
    )

    # --- Helper: check tool_call validity ---
    def check_tool_call_json(tool_call_str: str):
        try:
            data = json.loads(tool_call_str.strip())
        except Exception as e:
            return False, f"Tool call is not valid JSON. Tool call: {tool_call_str}"

        if not isinstance(data, dict):
            return False, f"Tool call must be a JSON object. Tool call: {tool_call_str}"

        if set(data.keys()) != {"name", "arguments"}:
            return (
                False,
                f"Tool call must contain only 'name' and 'arguments' keys. Tool call: {tool_call_str}",
            )

        name = data["name"]
        args = data["arguments"]

        if name == "web_search":
            if not isinstance(args, dict) or set(args.keys()) != {"query_list"}:
                return (
                    False,
                    f"web_search tool call arguments should be a dict and have only one key 'query_list'. Arguments: {args}.",
                )
            if not isinstance(args["query_list"], list) or not all(
                isinstance(q, str) for q in args["query_list"]
            ):
                return (
                    False,
                    f"web_search tool call 'query_list' should be a list of strings. Arguments: {args}.",
                )

        elif name == "web_read":
            if not isinstance(args, dict) or set(args.keys()) != {"url_list"}:
                return (
                    False,
                    f"web_read tool call arguments should be a dict and have only one key 'url_list'. Arguments: {args}.",
                )
            if not isinstance(args["url_list"], list) or not all(
                isinstance(u, str) for u in args["url_list"]
            ):
                return (
                    False,
                    f"web_read tool call 'url_list' should be a list of strings. Arguments: {args}.",
                )

        elif name == "web_browse":
            if not isinstance(args, dict) or set(args.keys()) != {"browsing_job_list"}:
                return (
                    False,
                    f"web_browse tool call arguments should be a dict and have only one key 'browsing_job_list'. Arguments: {args}.",
                )
            if not isinstance(args["browsing_job_list"], list) or not all(
                isinstance(j, dict) for j in args["browsing_job_list"]
            ):
                return (
                    False,
                    f"web_browse tool call 'browsing_job_list' should be a list of dictionaries. Arguments: {args}.",
                )
            if not all(
                set(j.keys()) == {"url", "task"} for j in args["browsing_job_list"]
            ):
                return (
                    False,
                    f"web_browse tool call 'browsing_job_list' should be a list of dictionaries with 'url' and 'task' keys. Arguments: {args}.",
                )

        else:
            return (
                False,
                f"Invalid tool name '{name}'. Only 'web_search', 'web_read', and 'web_browse' are allowed.",
            )

        return True, None

    # --- Main validation loop ---
    i = 0
    while i < len(assistant_blocks):
        found_answer = False

        # Research mode loop
        while i < len(assistant_blocks):
            block = assistant_blocks[i]

            if m := pattern_tool_call.match(block):
                ok, err = check_tool_call_json(m.group("tool_call"))
                if not ok:
                    return False, f"Invalid tool_call JSON at block {block}: {err}"
                i += 1
                continue

            elif pattern_answer.match(block):
                found_answer = True
                i += 1
                break

            else:
                return (
                    False,
                    f"Unexpected format in block {i + 1}. Expected tool_call or answer but got {block}.",
                )

        if not found_answer:
            return False, f"No <answer> block found. {full_response}"

        # If answer verification is disabled, answer should be the final block
        if not use_answer_verification:
            if i != len(assistant_blocks):
                return (
                    False,
                    f"Answer found but extra blocks follow (verification disabled). {full_response}",
                )
            return True, "Format valid — ends with answer (no verification)."

        # Verification phase (only when use_answer_verification=True)
        if i >= len(assistant_blocks):
            return (
                False,
                f"Expected verification block after final answer. {full_response}",
            )

        block = assistant_blocks[i]
        if not (m := pattern_verification.match(block)):
            return False, f"Expected verification block after answer at block {block}."

        result_text = m.group("result").strip()
        if result_text.lower() not in ("correct", "incorrect"):
            return (
                False,
                f"Invalid verification result '{result_text}'. Must be CORRECT or INCORRECT.",
            )

        if result_text.lower() == "correct":
            if i != len(assistant_blocks) - 1:
                return (
                    False,
                    f"Verification result is CORRECT but extra components follow. {full_response}",
                )
            return True, "Format valid — ends with correct verification."

        # INCORRECT → restart
        i += 1

    return False, f"Unexpected end of response. {full_response}"


def extract_answer(solution_str: str) -> Optional[str]:
    """Extract the content within the <answer> tags

    Args:
        solution_str: The string containing <answer> tags

    Returns:
        str: The content within the <answer> tags, or None if not found or if there are multiple answers
    """
    try:
        # Find all answer tags
        answer_matches = re.findall(r"<answer>(.*?)</answer>", solution_str, re.DOTALL)

        if len(answer_matches) == 0:
            logger.debug(
                "No <answer> tag found. Last 10 characters: " + solution_str[-10:]
            )
            return None

        # Return the last answer content, stripped of whitespace
        return answer_matches[-1].strip()

    except Exception as e:
        logger.error(f"Error extracting answer content: {e}")
        return None


def compute_process_reward(solution_str: str) -> float:
    return 0.0

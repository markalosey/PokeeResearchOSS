# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

# This file is based on verl/verl/experimental/agent_loop/tool_agent_loop.py. We implement more sophisticated agent-environment interaction logic here.

# This file is based on verl/verl/experimental/agent_loop/tool_agent_loop.py. We implement more sophisticated agent-environment interaction logic here.

import asyncio
import copy
import json
import logging
import os
from typing import Any
from uuid import uuid4
import regex as re

from agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.experimental.agent_loop.tool_parser import FunctionCall
from tool_client.schemas import ToolResponse
from tool_client.tool_registry import initialize_tools_from_config
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from functools import partial
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("tool_agent")
class ToolAgentLoop(AgentLoopBase):
    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        print("Performing class-level ToolAgentLoop initialization")

        # Initialize tools from config file
        cls.tokenizer = tokenizer
        cls.processor = processor
        cls.max_user_turns = config.actor_rollout_ref.rollout.multi_turn.max_user_turns
        cls.max_assistant_turns = config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns
        cls.max_parallel_calls = config.actor_rollout_ref.rollout.multi_turn.max_parallel_calls
        cls.max_tool_response_length = config.actor_rollout_ref.rollout.multi_turn.max_tool_response_length
        cls.tool_response_truncate_side = config.actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side
        tool_config_path = config.actor_rollout_ref.rollout.multi_turn.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        cls.tools = {tool.name: tool for tool in tool_list}
        cls.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]
        print(f"Initialized tools: {cls.tools}")

        cls.apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})
        cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.actor_rollout_ref.rollout.response_length
        cls.system_prompt = tokenizer.apply_chat_template(
            [{}], add_generation_prompt=False, tokenize=True, **cls.apply_chat_template_kwargs
        )
        cls.tool_call_regex = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
        cls.answer_regex = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
        cls.verification_result_regex = re.compile(r"<verification_result>(.*?)</verification_result>", re.DOTALL)

        cls.think_tool_call_regex = re.compile(r'''
        ^<think>
            ((?:(?!<think>|</think>).)*?)          # stops at any nested <think> or </think>
        </think>\s*
        <tool_call>
            ((?:(?!<tool_call>|</tool_call>).)*?)  # stops at any nested <tool_call> or </tool_call>
        </tool_call>$''', re.DOTALL | re.VERBOSE)

        cls.think_answer_regex = re.compile(r'''
        ^<think>
            ((?:(?!<think>|</think>).)*?)          # stops at any nested <think> or </think>
        </think>\s*
        <answer>
            ((?:(?!<answer>|</answer>).)*?)  # stops at any nested <answer> or </answer>
        </answer>$''', re.DOTALL | re.VERBOSE)

        cls.verification_verification_result_regex = re.compile(r'''
        ^<verification>
            ((?:(?!<verification>|</verification>).)*?)          # stops at any nested <verification> or </verification>
        </verification>\s*
        <verification_result>
            ((?:(?!<verification_result>|</verification_result>).)*?)  # stops at any nested <verification_result> or </verification_result>
        </verification_result>$''', re.DOTALL | re.VERBOSE)

        print(f"check_think_tags: {config.actor_rollout_ref.rollout.multi_turn.check_think_tags}")
        print(f"strict_generation: {config.actor_rollout_ref.rollout.multi_turn.strict_generation}")
        cls.check_think_tags = config.actor_rollout_ref.rollout.multi_turn.check_think_tags
        cls.strict_generation = config.actor_rollout_ref.rollout.multi_turn.strict_generation
        cls.use_answer_verification = config.actor_rollout_ref.rollout.multi_turn.use_answer_verification
    

    def parse_without_answer_verification(self, text: str) -> tuple[list[dict], list[str], bool, bool]:
        tool_messages = []
        tool_calls = []
        format_check_passed = True
        generation_finished = False
        if self.check_think_tags:
            think_answer_matches = self.think_answer_regex.findall(text)
            if len(think_answer_matches) == 1:
                generation_finished = True
                return tool_messages, tool_calls, format_check_passed, generation_finished
            else:
                think_tool_call_matches = self.think_tool_call_regex.findall(text)
                if len(think_tool_call_matches) == 1:
                    tool_calls = [think_tool_call_matches[0][1].strip()]
                else:
                    tool_messages = [{"role": "user", "content": "You didn't generate a think-tool_call response or a think-answer response in the right format. Please refer to the format instructions in the system prompt and try again."}]
                    format_check_passed = False

        else:
            answer_matches = self.answer_regex.findall(text)
            if len(answer_matches) == 1:
                generation_finished = True
                return tool_messages, tool_calls, format_check_passed, generation_finished
            elif len(answer_matches) > 1:
                tool_messages = [{"role": "user", "content": "You are only allowed to generate one answer but you generated more than one. Please regenerate the answer."}]
                format_check_passed = False
            else:
                tool_call_matches = self.tool_call_regex.findall(text)
                if len(tool_call_matches) == 0:
                    tool_messages = [{"role": "user", "content": "You didn't generate a <tool_call> block. Please try again."}]
                    format_check_passed = False
                elif len(tool_call_matches) > 1:
                    tool_messages = [{"role": "user", "content": "You are only allowed to generate one tool call but you generated more than one. Please regenerate the tool call."}]
                    format_check_passed = False
                else:
                    tool_calls = [tool_call_matches[0].strip()]
        return tool_messages, tool_calls, format_check_passed, generation_finished
    

    def parse_with_answer_verification(self, text: str, current_mode: str) -> tuple[list[dict], list[str], bool, bool, str]:
        tool_messages = []
        tool_calls = []
        format_check_passed = True
        generation_finished = False
        if self.check_think_tags:
            verification_verification_result_matches = self.verification_verification_result_regex.findall(text)
            if len(verification_verification_result_matches) == 1:
                if verification_verification_result_matches[0][1].strip().lower() == "correct":
                    generation_finished = True
                    return tool_messages, tool_calls, format_check_passed, generation_finished, current_mode
                elif verification_verification_result_matches[0][1].strip().lower() == "incorrect":
                    tool_messages = [{"role": "user", "content": "The answer is verified to be incorrect. Please incorporate the feedback from the verification mode and re-enter the research mode. ##The research mode starts##."}]
                    current_mode = "research"
                else:
                    tool_messages = [{"role": "user", "content": "Your verification result should be either CORRECT or INCORRECT, without any other text. Please redo the verification mode."}]
                    format_check_passed = False
            else:
                think_answer_matches = self.think_answer_regex.findall(text)
                if len(think_answer_matches) == 1:
                    tool_messages = [{"role": "user", "content": "You have provided an answer. ##The verification mode starts##."}]
                    current_mode = "verification"
                else:
                    think_tool_call_matches = self.think_tool_call_regex.findall(text)
                    if len(think_tool_call_matches) == 1:
                        tool_calls = [think_tool_call_matches[0][1].strip()]
                    else:
                        if current_mode == "research":
                            tool_messages = [{"role": "user", "content": "You are in the research mode but didn't generate a think-tool_call response or a think-answer response in the right format. Please refer to the format instructions in the system prompt and try again."}]
                        else:
                            tool_messages = [{"role": "user", "content": "You are in the verification mode but didn't generate a verification-verification_result response in the right format. Please refer to the format instructions in the system prompt and try again."}]
                        format_check_passed = False

        else:
            # check if there is answer, if so, break
            verification_result_matches = self.verification_result_regex.findall(text)
            if len(verification_result_matches) == 1:
                if verification_result_matches[0].strip().lower() == "correct":
                    generation_finished = True
                    return tool_messages, tool_calls, format_check_passed, generation_finished, current_mode
                elif verification_result_matches[0].strip().lower() == "incorrect":
                    tool_messages = [{"role": "user", "content": "The answer is verified to be incorrect. Please incorporate the feedback from the verification mode and re-enter the research mode. ##The research mode starts##."}]
                    current_mode = "research"
                else:
                    tool_messages = [{"role": "user", "content": "Your verification result should be either CORRECT or INCORRECT, without any other text. Please redo the verification mode."}]
                    format_check_passed = False
            elif len(verification_result_matches) > 1:
                tool_messages = [{"role": "user", "content": "You are only allowed to generate one verification result but you generated more than one. Please redo the verification mode and only generate one verification result."}]
                format_check_passed = False
            else:
                answer_matches = self.answer_regex.findall(text)
                if len(answer_matches) == 1:
                    tool_messages = [{"role": "user", "content": "You have provided an answer. ##The verification mode starts##."}]
                    current_mode = "verification"
                elif len(answer_matches) > 1:
                    tool_messages = [{"role": "user", "content": "You are only allowed to generate one answer but you generated more than one. Please regenerate the answer."}]
                    format_check_passed = False
                else:
                    tool_call_matches = self.tool_call_regex.findall(text)
                    if len(tool_call_matches) == 0:
                        if current_mode == "research":
                            tool_messages = [{"role": "user", "content": "You are in the research mode but didn't generate a <tool_call> block. Please try again."}]
                        else:
                            tool_messages = [{"role": "user", "content": "You are in the verification mode but didn't generate a <verification_result> block. Please try again."}]
                        format_check_passed = False
                    elif len(tool_call_matches) > 1:
                        tool_messages = [{"role": "user", "content": "You are only allowed to generate one tool call but you generated more than one. Please regenerate the tool call."}]
                        format_check_passed = False
                    else:
                        tool_calls = [tool_call_matches[0].strip()]
        return tool_messages, tool_calls, format_check_passed, generation_finished, current_mode

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        image_data = copy.deepcopy(kwargs.get("multi_modal_data", {}).get("image", None))
        metrics = {}
        request_id = uuid4().hex
        if self.processor is not None:
            raw_prompt = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    messages,
                    tools=self.tool_schemas,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            model_inputs = self.processor(text=[raw_prompt], images=image_data, return_tensors="pt")
            prompt_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    messages,
                    tools=self.tool_schemas,
                    add_generation_prompt=True,
                    tokenize=True,
                    **self.apply_chat_template_kwargs,
                ),
            )
        response_mask, response_logprobs = [], []
        tools_kwargs = kwargs.get("tools_kwargs", {})
        metrics["at_least_one_failed_parse"] = 0
        user_turns, assistant_turns = 0, 0
        current_mode = "research"
        while True:
            with simple_timer("generate_sequences", metrics):
                output = await self.server_manager.generate(
                    request_id=request_id, prompt_ids=prompt_ids, sampling_params=sampling_params, image_data=image_data
                )
            response_ids = output.token_ids
            prompt_ids += response_ids
            response_mask += [1] * len(response_ids)
            if output.log_probs:
                response_logprobs += output.log_probs
            assistant_turns += 1

            # reach max response length
            if len(response_mask) >= self.response_length:
                break

            # reach max assistant turns
            if self.max_assistant_turns and assistant_turns >= self.max_assistant_turns:
                break

            # reach max user turns
            if self.max_user_turns and user_turns >= self.max_user_turns:
                break

            # no tool calls
            loop = asyncio.get_running_loop()

            decode_fn = partial(self.tokenizer.decode, skip_special_tokens=True)
            text = await loop.run_in_executor(None, decode_fn, response_ids)
            assert self.max_parallel_calls == 1, "Only one tool call is supported for now"

            if self.use_answer_verification:
                tool_messages, tool_calls, format_check_passed, generation_finished, current_mode = self.parse_with_answer_verification(text, current_mode)
            else:
                tool_messages, tool_calls, format_check_passed, generation_finished = self.parse_without_answer_verification(text)

            if generation_finished:
                break

            if not format_check_passed:
                metrics["at_least_one_failed_parse"] = 1 
                if self.strict_generation:
                    break

            # call tools
            if len(tool_calls) > 0:
                tasks = []
                for tool_call in tool_calls[: self.max_parallel_calls]:
                    tasks.append(self._call_tool(tool_call, tools_kwargs))
                with simple_timer("tool_calls", metrics):
                    tool_responses = await asyncio.gather(*tasks)

                assert len(tool_responses) == 1, "Only one tool call is supported for now"

                # Extract messages and update multi_modal_data
                
                new_images_this_turn = []
                for tool_response, tool_success in tool_responses:
                    # Create message from tool response
                    if tool_response.image or tool_response.video:
                        # Multi-modal content with structured format
                        content = []
                        if tool_response.image:
                            content.append({"type": "image"})
                        if tool_response.video:
                            content.append({"type": "video"})
                        if tool_response.text:
                            content.append({"type": "text", "text": tool_response.text})
                        message = {"role": "tool", "content": content}
                    else:
                        # Text-only content
                        message = {"role": "tool", "content": tool_response.text or ""}

                    tool_messages.append(message)

                    # Handle image data
                    if tool_response.image:
                        if image_data is None:
                            image_data = []
                        elif not isinstance(image_data, list):
                            image_data = [image_data]

                        # Add new image data
                        if isinstance(tool_response.image, list):
                            image_data.extend(tool_response.image)
                            new_images_this_turn.extend(tool_response.image)
                        else:
                            image_data.append(tool_response.image)
                            new_images_this_turn.append(tool_response.image)

                    # Handle video data
                    if tool_response.video:
                        # Currently not supported, raise informative error
                        logger.warning("Multimedia type 'video' is not currently supported. Only 'image' is supported.")
                        raise NotImplementedError(
                            "Multimedia type 'video' is not currently supported. Only 'image' is supported."
                        )
            
            assert len(tool_messages) > 0, "No tool messages generated. This should not happen. There is something wrong in the code."

            # append tool_response_ids
            if self.processor is not None:
                raw_tool_response = await self.loop.run_in_executor(
                    None,
                    lambda messages=tool_messages: self.processor.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
                    ),
                )
                # Use only the new images from this turn for processing tool responses
                current_images = new_images_this_turn if new_images_this_turn else None
                model_inputs = self.processor(text=[raw_tool_response], images=current_images, return_tensors="pt")
                tool_response_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
            else:
                tool_response_ids = await self.loop.run_in_executor(
                    None,
                    lambda messages=tool_messages: self.tokenizer.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=True, **self.apply_chat_template_kwargs
                    ),
                )
            tool_response_ids = tool_response_ids[len(self.system_prompt) :]

            # NOTE: last turn should not be user turn, or the EOS token reward
            # can't be propagated to previous token in GAE.
            if len(response_mask) + len(tool_response_ids) >= self.response_length:
                break

            prompt_ids += tool_response_ids
            response_mask += [0] * len(tool_response_ids)
            if response_logprobs:
                response_logprobs += [0.0] * len(tool_response_ids)
            user_turns += 1

        response_ids = prompt_ids[-len(response_mask) :]
        prompt_ids = prompt_ids[: len(prompt_ids) - len(response_mask)]

        multi_modal_data = {"image": image_data} if image_data is not None else {}

        # print("id: ", kwargs["id"], "prompt_texts: ", self.tokenizer.decode(prompt_ids))
        # print("id: ", kwargs["id"], "response_texts: ", self.tokenizer.decode(response_ids))

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            multi_modal_data=multi_modal_data,
            response_logprobs=response_logprobs[: self.response_length] if response_logprobs else None,
            num_turns=user_turns + assistant_turns + 1,
            metrics=metrics,
        )
        print("The research process for one query is finished. The query is: ", kwargs["extra_info"]["question"])
        return output

    async def _call_tool(self, tool_call: FunctionCall, tools_kwargs: dict[str, Any]) -> tuple[ToolResponse, bool]:
        """Call tool and return tool response."""
        tool, instance_id = None, None
        try:
            tool_call = json.loads(tool_call)
        except json.JSONDecodeError as e:
            # If parsing fails, try escaping backslashes and parsing again
            # This handles cases like \mu, \alpha, etc. in LaTeX/math expressions
            tool_call_escaped = tool_call.replace("\\", "\\\\")
            try:
                tool_call = json.loads(tool_call_escaped)
                logger.warning(f"Tool call required backslash escaping: {tool_call}...")
            except json.JSONDecodeError:
                # If still fails, raise the original error
                error_message = f"Parsing failed: tool call is not valid dict object. Tool call: {tool_call}"
                logger.warning(error_message)
                return ToolResponse(text=error_message), False
        if not isinstance(tool_call, dict):
            error_message = f"Parsing failed: tool call must be a dict object. Tool call: {tool_call}"
            logger.warning(error_message)
            return ToolResponse(text=error_message), False

        if "name" not in tool_call or "arguments" not in tool_call:
            error_message = f"Parsing failed: 'name' and 'arguments' are required in the tool call, but not found. Tool call: {tool_call}"
            logger.warning(error_message)
            return ToolResponse(text=error_message), False
        tool_name = tool_call["name"]
        tool_args = tool_call["arguments"]
        if tool_name not in self.tools:
            error_message = f"Parsing failed: tool '{tool_name}' is not in the tool list {list(self.tools.keys())}. Tool call: {tool_call}"
            logger.warning(error_message)
            return ToolResponse(text=error_message), False
        tool = self.tools[tool_name]
        kwargs = tools_kwargs.get(tool_name, {})
        try:
            instance_id, _ = await tool.create(create_kwargs=kwargs.get("create_kwargs", {}))
            tool_execution_response, _, tool_metrics = await tool.execute(instance_id, tool_args)
        except Exception as e:
            error_message = f"Tool execution failed: {type(e).__name__}: {str(e)}. Tool call: {tool_call}"
            logger.warning(error_message)
            return ToolResponse(text=error_message), False
        finally:
            if tool and instance_id:
                await tool.release(instance_id)

        tool_response_text = tool_execution_response.text
        if tool_response_text and len(tool_response_text) > self.max_tool_response_length:
            if self.tool_response_truncate_side == "left":
                tool_response_text = tool_response_text[: self.max_tool_response_length] + "...(truncated)"
            elif self.tool_response_truncate_side == "right":
                tool_response_text = "(truncated)..." + tool_response_text[-self.max_tool_response_length :]
            else:
                length = self.max_tool_response_length // 2
                tool_response_text = tool_response_text[:length] + "...(truncated)..." + tool_response_text[-length:]

        # Create ToolResponse from tool execution result
        tool_response_kwargs = {"text": tool_response_text}

        # Add multimedia data if present
        for attr_name in ["image", "video"]:
            if hasattr(tool_execution_response, attr_name):
                attr_value = getattr(tool_execution_response, attr_name)
                if attr_value is not None:
                    tool_response_kwargs[attr_name] = attr_value

        return ToolResponse(**tool_response_kwargs), tool_metrics.get("success", False)

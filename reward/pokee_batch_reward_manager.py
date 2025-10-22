
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


from collections import defaultdict
from typing import Any, Dict, List

import torch

from verl import DataProto
from verl.workers.reward_manager.abstract import AbstractRewardManager, RawRewardFn


class PokeeBatchRewardManager(AbstractRewardManager):
    """
    A batch reward manager that computes rewards for a batch of data.

    Args:
        tokenizer (Tokenizer): The tokenizer to use for decoding the responses.
        num_examine (int): The number of responses to examine.
        compute_score (callable): The function to compute the rewards.
        reward_fn_key (str): The key to use for the reward function.
        overlong_buffer_len (int): The length of the overlong buffer.
        max_resp_len (int): The maximum length of the response.
        max_num_turns (int): The maximum number of turns.
        reward_weights (dict): The weights of the rewards.
        reward_kwargs (dict): The keyword arguments to pass to the reward function.
    """

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score: RawRewardFn,
        reward_fn_key="data_source",
        overlong_buffer_len=None,
        max_resp_len=None,
        max_num_turns=None,
        reward_weights=None,
        **reward_kwargs,
    ):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key
        self.overlong_buffer_len = overlong_buffer_len
        self.max_resp_len = max_resp_len
        self.max_num_turns = max_num_turns
        self.reward_weights = reward_weights
        self.reward_kwargs = reward_kwargs

        assert len(self.reward_weights) > 0, "reward_weights is empty"

    def verify(
        self, data: DataProto, reward_types: list[str] = ["gemini_mbe"]
    ) -> List[Dict[str, float]]:
        prompt_ids = data.batch["prompts"]
        response_ids = data.batch["responses"]
        attention_mask = data.batch["attention_mask"]

        list_of_questions = [
            data.non_tensor_batch["extra_info"][idx]["question"]
            for idx in range(len(data.non_tensor_batch["extra_info"]))
        ]

        prompt_len = prompt_ids.shape[-1]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)

        responses_str = []
        for i in range(len(data)):
            valid_len = valid_response_lengths[i]
            valid_response_ids = response_ids[i][:valid_len]
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=True
            )
            responses_str.append(response_str)

        ground_truths = [
            item.non_tensor_batch["reward_model"].get("ground_truth", None)
            for item in data
        ]
        data_sources = data.non_tensor_batch[self.reward_fn_key]

        scores = self.compute_score(
            data_sources=data_sources,
            list_of_questions=list_of_questions,
            solution_strs=responses_str,
            ground_truths=ground_truths,
            reward_types=reward_types,
            **self.reward_kwargs,
        )

        return scores

    def __call__(self, data: DataProto) -> List[dict[str, Any]]:
        reward_types = [
            k for k in self.reward_weights.keys() if k != "overlong_reward"
        ]  # overlong_reward is computed separately
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        prompt_ids = data.batch["prompts"]
        prompt_len = prompt_ids.shape[-1]
        attention_mask = data.batch["attention_mask"]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)
        data_sources = data.non_tensor_batch[self.reward_fn_key]

        scores = self.verify(data=data, reward_types=reward_types)
        already_printed: dict[str, Any] = {}

        for i in range(len(data)):
            length = valid_response_lengths[i].item()
            score = scores[i]

            reward = 0.0
            for k in reward_types:
                reward += score[k] * self.reward_weights[k]
            reward_tensor[i, length - 1] = reward

            if "overlong_reward" in self.reward_weights.keys():
                expected_len = self.max_resp_len - self.overlong_buffer_len
                if length > expected_len:
                    reward_tensor[i, expected_len:length] -= (
                        self.reward_weights["overlong_reward"]
                        / self.overlong_buffer_len
                    )
                reward_extra_info["overlong_reward"].append(
                    min(-(length - expected_len) / self.overlong_buffer_len, 0)
                )

            for key, value in score.items():
                reward_extra_info[key].append(value)

            data_source = data_sources[i]
            if already_printed.get(data_source, 0) < self.num_examine:
                response_str = self.tokenizer.decode(
                    data.batch["responses"][i][:length], skip_special_tokens=True
                )
                prompt_str = self.tokenizer.decode(
                    data.batch["prompts"][i], skip_special_tokens=True
                )
                ground_truth = (
                    data[i].non_tensor_batch["reward_model"].get("ground_truth", None)
                )
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                print("[score]", scores[i])
                already_printed[data_source] = already_printed.get(data_source, 0) + 1

        return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}

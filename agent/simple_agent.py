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
Simple Deep Research Agent
Uses local model loading for inference.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from agent.base_agent import BaseDeepResearchAgent
from logging_utils import setup_colored_logger

logger = setup_colored_logger(__name__)


class SimpleDeepResearchAgent(BaseDeepResearchAgent):
    """Simple standalone deep research agent with local model."""

    # Class-level (singleton) model and tokenizer
    _model = None
    _tokenizer = None
    _model_path = None
    _device = None
    _model_lock = None  # For thread-safe initialization

    def __init__(
        self,
        model_path: str = "PokeeAI/pokee_research_7b",
        tool_config_path: str = "config/tool_config/pokee_tool_config.yaml",
        device: str = "cuda",
        max_turns: int = 10,
        max_tool_response_length: int = 32768,
    ):
        """Initialize the agent.

        Args:
            model_path: Path to model or HuggingFace model ID
            tool_config_path: Path to tool configuration
            device: Device to use
            max_turns: Maximum conversation turns
            max_tool_response_length: Max length for tool responses
        """
        # Initialize base class
        super().__init__(
            tool_config_path=tool_config_path,
            max_turns=max_turns,
            max_tool_response_length=max_tool_response_length,
        )

        # Initialize lock on first use
        if SimpleDeepResearchAgent._model_lock is None:
            import threading

            SimpleDeepResearchAgent._model_lock = threading.Lock()

        # Load model only once (singleton pattern) with thread safety
        with SimpleDeepResearchAgent._model_lock:
            if (
                SimpleDeepResearchAgent._model is None
                or SimpleDeepResearchAgent._model_path != model_path
            ):
                self._load_model(model_path, device)
            elif SimpleDeepResearchAgent._device != device:
                logger.warning(
                    f"Model already loaded on {SimpleDeepResearchAgent._device}, ignoring device={device}"
                )

        # Use the shared model and tokenizer
        self.model = SimpleDeepResearchAgent._model
        self.tokenizer = SimpleDeepResearchAgent._tokenizer

    @classmethod
    def _load_model(cls, model_path: str, device: str):
        """Load model and tokenizer (class method for singleton)."""
        logger.info(f"Loading model from {model_path}...")
        cls._tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=True
        )
        cls._model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        cls._model.eval()
        cls._model_path = model_path
        cls._device = device
        logger.info(f"Model loaded successfully on {device}!")

    async def generate(
        self, messages: list[dict], temperature: float = 0.7, top_p: float = 0.9
    ) -> str:
        """Generate response from messages using local model.

        Args:
            messages: Conversation messages
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter

        Returns:
            Generated text
        """
        # Apply chat template (reuse tool_schemas from base class)
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tools=self.tool_schemas,
            add_generation_prompt=True,
            tokenize=False,
        )

        # Tokenize with padding for efficiency
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=False,  # No padding needed for single sequence
            truncation=True,  # Prevent OOM from overly long prompts
            max_length=32768,  # Match model's max context
        ).to(self.model.device)

        # Generate with optimizations
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,  # Only sample if temp > 0
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,  # Enable KV cache for faster generation
                num_beams=1,  # Greedy decoding (faster than beam search)
            )

        # Decode only the generated part (more efficient)
        generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return response

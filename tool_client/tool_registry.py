# Copyright 2025 Bytedance Ltd. and/or its affiliates
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


# This file is different from the original verl/verl/tools/utils/tool_registry.py
# in that we import our own schemas.


# This file is different from the original verl/verl/tools/utils/tool_registry.py
# in that we import our own schemas.

import asyncio
import importlib
import logging
import os
import sys
import threading
from enum import Enum

from omegaconf import OmegaConf

from tool_client.schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class ToolType(Enum):
    NATIVE = "native"
    MCP = "mcp"


def get_tool_class(cls_name):
    module_name, class_name = cls_name.rsplit(".", 1)
    if module_name not in sys.modules:
        spec = importlib.util.find_spec(module_name)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    else:
        module = sys.modules[module_name]

    tool_cls = getattr(module, class_name)
    return tool_cls


def initialize_tools_from_config(tools_config_file):
    tools_config = OmegaConf.load(tools_config_file)
    tool_list = []

    # Use a temporary event loop in a new thread because event
    # loop may already exist in new async architecture while retaining
    # backwards compatibility
    tmp_event_loop = asyncio.new_event_loop()
    thread = threading.Thread(target=tmp_event_loop.run_forever, name="mcp tool list fetcher", daemon=True)

    async def stop_loop():
        tmp_event_loop.stop()

    try:
        for tool_config in tools_config.tools:
            cls_name = tool_config.class_name
            tool_type = ToolType(tool_config.config.type)
            tool_cls = get_tool_class(cls_name)

            match tool_type:
                case ToolType.NATIVE:
                    if tool_config.get("tool_schema", None) is None:
                        tool_schema = None
                    else:
                        tool_schema_dict = OmegaConf.to_container(tool_config.tool_schema, resolve=True)
                        tool_schema = OpenAIFunctionToolSchema.model_validate(tool_schema_dict)
                    tool = tool_cls(
                        config=OmegaConf.to_container(tool_config.config, resolve=True),
                        tool_schema=tool_schema,
                    )
                    tool_list.append(tool)
                case ToolType.MCP:
                    raise NotImplementedError
                case _:
                    raise NotImplementedError
    finally:
        if thread.is_alive():
            asyncio.run_coroutine_threadsafe(stop_loop(), tmp_event_loop)
            thread.join()

    return tool_list

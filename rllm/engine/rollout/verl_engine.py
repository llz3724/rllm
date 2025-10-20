import base64
import io
import uuid

import requests
from PIL import Image
from verl.experimental.agent_loop.agent_loop import AsyncLLMServerManager

from rllm.engine.rollout.rollout_engine import ModelOutput, RolloutEngine
from rllm.parser import ChatTemplateParser, ToolParser
import logging
import inspect

logger = logging.getLogger(__file__)


def load_image(image_identifier: str) -> Image.Image | None:
    """
    从 URL 或 Base64 字符串加载图片，并将其转换为 PIL.Image 对象。

    Args:
        image_identifier (str): 图片的 URL 或 Base64 字符串。

    Returns:
        Image.Image | None: 成功加载则返回 PIL Image 对象，否则返回 None。
    """
    if image_identifier.startswith("http://") or image_identifier.startswith("https://"):
        try:
            response = requests.get(image_identifier, timeout=10)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content)).convert("RGB")
            return image
        except Exception as e:
            print(f"Error: Failed to load image from URL: {image_identifier}. Reason: {e}")
            return None
    elif image_identifier.startswith("data:image"):
        try:
            # 例如: "data:image/jpeg;base64,iVBO..."
            header, encoded = image_identifier.split(",", 1)
            image_data = base64.b64decode(encoded)
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            return image
        except Exception as e:
            print(f"Error: Failed to load image from Base64 string. Reason: {e}")
            return None
    else:
        # 您也可以在这里扩展对本地文件路径的支持
        from pathlib import Path

        if Path(image_identifier).exists():
            return Image.open(image_identifier).convert("RGB")
        print(f"Warning: Unsupported image identifier format for: {image_identifier}")
        return None


class VerlEngine(RolloutEngine):
    def __init__(self, config, rollout_manager, tokenizer, **kwargs):
        self.config = config
        self.rollout_manager = rollout_manager
        self.server_manager = AsyncLLMServerManager(config, rollout_manager.async_llm_servers)
        self.tokenizer = tokenizer
        self.chat_parser = ChatTemplateParser.get_parser(self.tokenizer, disable_thinking=kwargs.get("disable_thinking", False))

        try:
            self.tool_parser = ToolParser.get_parser(self.tokenizer)
        except Exception:
            print(f"Warning: No tool parser found for {self.tokenizer.name_or_path}. Tool calls not be parsed.")
            self.tool_parser = None

        self.validate = False

    async def get_model_response(self, messages: list[dict], **kwargs) -> ModelOutput:
        application_id = kwargs.pop("application_id", str(uuid.uuid4()))
        validate = self.validate or kwargs.pop("validate", False)

        if validate:
            sampling_params = dict(
                temperature=0.0 if self.config.actor_rollout_ref.rollout.val_kwargs.do_sample is False else self.config.actor_rollout_ref.rollout.val_kwargs.temperature,
                top_k=self.config.actor_rollout_ref.rollout.val_kwargs.top_k,
                top_p=self.config.actor_rollout_ref.rollout.val_kwargs.top_p,
            )
        else:
            sampling_params = dict(
                temperature=0.0 if self.config.actor_rollout_ref.rollout.do_sample is False else self.config.actor_rollout_ref.rollout.temperature,
                top_k=self.config.actor_rollout_ref.rollout.top_k,
                top_p=self.config.actor_rollout_ref.rollout.top_p,
            )
        sampling_params.update(kwargs)

        max_tokens = sampling_params.pop("max_tokens", self.config.data.max_response_length)
        images_from_kwargs = sampling_params.pop("images", None)
        # prompt = self.chat_parser.parse(messages, add_generation_prompt=True, is_first_msg=True)
        # prompt_ids = self.tokenizer.encode(prompt)

        # response_ids = await self.server_manager.generate(request_id=application_id, prompt_ids=prompt_ids, sampling_params=sampling_params)
        # [--- MODIFICATION START ---]
        # 1. 調用 parser，它會返回(字符串, 列表)元組或純字符串
        parsed_result = self.chat_parser.parse(messages, add_generation_prompt=True, is_first_msg=True)

        image_locations = None
        image_data_list = None

        if isinstance(parsed_result, tuple):
            prompt, image_locations = parsed_result
        else:
            prompt = parsed_result

        if images_from_kwargs and not image_locations:
            image_locations = images_from_kwargs
        # 2. 將格式化好的字符串編碼為 token IDs

        prompt_ids = self.tokenizer.encode(prompt)

        # 3. 如果有圖片，則加載它們
        if image_locations:
            image_data_list = [load_image(loc) for loc in image_locations]
            # 過濾掉加載失敗的圖片 (load_image 返回 None)
            image_data_list = [img for img in image_data_list if img is not None]
            if not image_data_list:  # 如果所有圖片都加載失敗，則重置為 None
                image_data_list = None

        # [--- 修正開始 ---]
        # 1. 接收完整的 TokenOutput 物件
        token_output = await self.server_manager.generate(request_id=application_id, prompt_ids=prompt_ids, sampling_params=sampling_params, image_data=image_data_list)

        # 2. 從物件中提取出 token ID 列表
        #    假設儲存 ID 的屬性是 .token_ids
        response_ids = token_output.token_ids
        # [--- 修正結束 ---]

        # verl sets max_tokens as max_model_len - len(prompt_ids), where max_model_len is config.data.max_prompt_length + config.data.max_response_length
        # so we truncate the response to max_tokens if it exceeds max_tokens
        finish_reason = "stop"
        if len(response_ids) >= max_tokens:
            finish_reason = "length"
            response_ids = response_ids[:max_tokens]

        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        tool_calls = None
        if self.tool_parser is not None:
            tool_calls = self.tool_parser.parse(response_text)

        return ModelOutput(text=response_text, tool_calls=tool_calls, finish_reason=finish_reason, completion_tokens=len(response_ids), prompt_tokens=len(prompt_ids))

    def wake_up(self):
        try:
            manager = self.rollout_manager
            manager_type = type(manager)
            wake_up_method = manager.wake_up

            # 1. 打印类型
            logger.error(f"--- [INSPECT] Manager Type: {manager_type}")

            # 2. 打印 manager 对象所属类的源文件路径
            file_path = inspect.getfile(manager_type)
            logger.error(f"--- [INSPECT] Class Definition File: {file_path}")

            # 3. 打印 wake_up 方法本身的确切来源
            method_object = getattr(manager, "wake_up")
            method_file_path = inspect.getfile(method_object)
            source_lines, start_line = inspect.getsourcelines(method_object)

            logger.error(f"--- [INSPECT] wake_up() Method is defined in: {method_file_path}")
            logger.error(f"--- [INSPECT] wake_up() Starts at line: {start_line}")

            # 4. 直接把 wake_up 方法的源代码打印出来！
            logger.error("--- [INSPECT] Source Code of wake_up(): ---\n" + "".join(source_lines))
            logger.error("---------------------------------------------")

        except Exception as e:
            logger.error(f"--- [INSPECT] Failed to inspect rollout_manager: {e}")
        # ====================================================================
        self.rollout_manager.wake_up()

    def sleep(self):
        self.rollout_manager.sleep()

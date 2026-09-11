"""LLM 适配层：同时支持 OpenAI 协议与 Anthropic 协议的客户端。

传入 ``VLCaptcha`` 的 ``llm`` 可以是：

* ``openai.OpenAI`` / ``AzureOpenAI`` 或任何兼容 ``chat.completions.create``
  的客户端（DashScope、vLLM、Ollama、OpenRouter 等 OpenAI 兼容网关）；
* ``anthropic.Anthropic`` / ``AnthropicBedrock`` 等提供 ``messages.create``
  的客户端；
* 一个 ``Backend`` 子类实例；
* 一个可调用对象 ``fn(system: str, user_text: str, image: ImagePayload) -> str``。
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import Any, Callable

from .exceptions import ConfigurationError, ModelResponseError
from .images import ImagePayload

#: 未显式指定 model 时各协议使用的默认模型
DEFAULT_OPENAI_MODEL = "gpt-4o"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"

# Anthropic 只接受这几种 media_type
_ANTHROPIC_MEDIA_TYPES = {"image/png", "image/jpeg", "image/gif", "image/webp"}


class Backend(ABC):
    """把 (系统提示, 用户文本, 图片列表) 变成模型的纯文本回复。"""

    name = "backend"

    def __init__(self, model: str, request_params: dict[str, Any] | None = None):
        self.model = model
        self.request_params = dict(request_params or {})

    @abstractmethod
    def complete(
        self,
        system: str,
        user_text: str,
        image: ImagePayload,
        extra_images: list[ImagePayload] | None = None,
    ) -> str:
        """调用模型，返回文本内容。"""

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{type(self).__name__} model={self.model!r}>"


class OpenAIBackend(Backend):
    """OpenAI Chat Completions 协议。"""

    name = "openai"

    def __init__(
        self,
        client: Any,
        model: str = DEFAULT_OPENAI_MODEL,
        request_params: dict[str, Any] | None = None,
    ):
        super().__init__(model, request_params)
        self.client = client

    def complete(
        self,
        system: str,
        user_text: str,
        image: ImagePayload,
        extra_images: list[ImagePayload] | None = None,
    ) -> str:
        content: list[dict] = [
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": {"url": image.data_url}},
        ]
        for extra in extra_images or []:
            content.append(
                {"type": "image_url", "image_url": {"url": extra.data_url}}
            )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            **self.request_params,
        )
        return _extract_openai_text(response)


class AnthropicBackend(Backend):
    """Anthropic Messages 协议。"""

    name = "anthropic"

    def __init__(
        self,
        client: Any,
        model: str = DEFAULT_ANTHROPIC_MODEL,
        request_params: dict[str, Any] | None = None,
    ):
        super().__init__(model, request_params)
        self.client = client
        self.request_params.setdefault("max_tokens", 1024)

    def complete(
        self,
        system: str,
        user_text: str,
        image: ImagePayload,
        extra_images: list[ImagePayload] | None = None,
    ) -> str:
        # Anthropic: 图片放前面，文本放后面
        parts: list[dict] = []

        def _add_image(img: ImagePayload) -> None:
            media_type = img.mime_type
            if media_type not in _ANTHROPIC_MEDIA_TYPES:
                raise ConfigurationError(
                    f"Anthropic 不支持 {media_type} 图片，请先转成 png / jpeg / gif / webp"
                )
            parts.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": img.base64,
                },
            })

        _add_image(image)
        for extra in extra_images or []:
            _add_image(extra)
        parts.append({"type": "text", "text": user_text})

        response = self.client.messages.create(
            model=self.model,
            system=system,
            messages=[{"role": "user", "content": parts}],
            **self.request_params,
        )
        return _extract_anthropic_text(response)


class CallableBackend(Backend):
    """把任意可调用对象包装成后端，方便测试或自定义协议。"""

    name = "callable"

    def __init__(
        self,
        func: Callable[[str, str, ImagePayload], str],
        model: str = "custom",
        request_params: dict[str, Any] | None = None,
    ):
        super().__init__(model, request_params)
        self.func = func

    def complete(
        self,
        system: str,
        user_text: str,
        image: ImagePayload,
        extra_images: list[ImagePayload] | None = None,
    ) -> str:
        content = self.func(system, user_text, image)
        if not isinstance(content, str):
            raise ModelResponseError(f"自定义后端必须返回字符串，得到 {type(content)!r}")
        return content


def build_backend(llm: Any) -> Backend:
    """根据客户端对象自动选择协议适配器。"""
    if llm is None:
        raise ConfigurationError("必须提供 llm 客户端，例如 OpenAI() 或 Anthropic()")

    if isinstance(llm, Backend):
        return llm

    create = _resolve_create(llm, ("chat", "completions", "create"))
    if create is not None:
        _reject_async(llm, create, "OpenAI")
        return OpenAIBackend(llm)

    create = _resolve_create(llm, ("messages", "create"))
    if create is not None:
        _reject_async(llm, create, "Anthropic")
        return AnthropicBackend(llm)

    if callable(llm):
        return CallableBackend(llm)

    raise ConfigurationError(
        "无法识别的 llm 客户端："
        f"{type(llm)!r} 既没有 chat.completions.create（OpenAI 协议），"
        "也没有 messages.create（Anthropic 协议）"
    )


def _resolve_create(client: Any, path: tuple[str, ...]) -> Any:
    node = client
    for attr in path:
        node = getattr(node, attr, None)
        if node is None:
            return None
    return node if callable(node) else None


def _reject_async(client: Any, create: Any, protocol: str) -> None:
    candidates = [create, getattr(create, "__func__", None), getattr(create, "__call__", None)]
    try:
        candidates.append(inspect.unwrap(create))
    except Exception:  # pragma: no cover
        pass
    is_async = any(inspect.iscoroutinefunction(item) for item in candidates if item)
    is_async = is_async or type(client).__name__.startswith("Async")

    if is_async:
        raise ConfigurationError(
            f"检测到异步 {protocol} 客户端，VLCaptcha 目前只支持同步客户端，"
            f"请改用同步客户端（如 OpenAI() / Anthropic()）"
        )


def _extract_openai_text(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if not choices:
        raise ModelResponseError("模型没有返回任何 choice", raw=repr(response))
    message = choices[0].message
    content = getattr(message, "content", None)

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(item.get("text") or "")
            else:
                parts.append(getattr(item, "text", "") or "")
        content = "".join(parts)

    if not content:
        content = getattr(message, "reasoning_content", None)

    if not content:
        raise ModelResponseError("模型返回内容为空", raw=repr(response))
    return content.strip()


def _extract_anthropic_text(response: Any) -> str:
    blocks = getattr(response, "content", None) or []
    parts = []
    for block in blocks:
        if isinstance(block, dict):
            if block.get("type") == "text":
                parts.append(block.get("text", ""))
        elif getattr(block, "type", None) == "text":
            parts.append(getattr(block, "text", "") or "")
    content = "".join(parts).strip()
    if not content:
        raise ModelResponseError("模型返回内容为空", raw=repr(response))
    return content

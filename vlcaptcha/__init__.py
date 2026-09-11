"""vlcaptcha —— 基于视觉语言模型的通用验证码识别库。

::

    from openai import OpenAI
    from vlcaptcha import VLCaptcha

    llm = OpenAI()
    vlcaptcha = VLCaptcha(llm)

    vlcaptcha.text("text.png")       # -> "aB3d"
    vlcaptcha.calc("calc.png")       # -> 12
    vlcaptcha.click("click.png")     # -> [[123, 88], [210, 96]]
    vlcaptcha.rotate("rotate.png")   # -> 90
    vlcaptcha.trace("trace.png")     # -> [[10, 20], ..., [180, 60]]
    vlcaptcha.slider("slider.png")   # -> 156
"""

from __future__ import annotations

from .backends import (
    AnthropicBackend,
    Backend,
    CallableBackend,
    OpenAIBackend,
    build_backend,
)
from .core import DEFAULT_COORD_SCALE, KINDS, VLCaptcha
from .exceptions import (
    ConfigurationError,
    ImageError,
    ModelResponseError,
    VLCaptchaError,
)
from .images import ImagePayload, load_image
from .prompts import PROMPTS, build_prompt, build_user_text

__version__ = "0.1.0"

__all__ = [
    "VLCaptcha",
    "KINDS",
    "DEFAULT_COORD_SCALE",
    "Backend",
    "OpenAIBackend",
    "AnthropicBackend",
    "CallableBackend",
    "build_backend",
    "ImagePayload",
    "load_image",
    "PROMPTS",
    "build_prompt",
    "build_user_text",
    "VLCaptchaError",
    "ConfigurationError",
    "ImageError",
    "ModelResponseError",
    "visualize",
    "__version__",
]


def visualize(*args, **kwargs):
    """惰性代理到 :func:`vlcaptcha.visualize.visualize`（需要 Pillow）。"""
    from .visualize import visualize as _visualize

    return _visualize(*args, **kwargs)

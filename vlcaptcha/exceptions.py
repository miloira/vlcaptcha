"""vlcaptcha 的异常类型。"""

from __future__ import annotations


class VLCaptchaError(Exception):
    """所有 vlcaptcha 异常的基类。"""


class ConfigurationError(VLCaptchaError):
    """LLM 客户端或参数配置有问题。"""


class ImageError(VLCaptchaError):
    """图片无法读取、解码或不受支持。"""


class ModelResponseError(VLCaptchaError):
    """模型返回的内容无法解析成约定的结构。"""

    def __init__(self, message: str, raw: str | None = None) -> None:
        super().__init__(message)
        self.raw = raw

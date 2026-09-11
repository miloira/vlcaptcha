"""VLCaptcha 主入口。"""

from __future__ import annotations

import json as _json
from typing import Any, Callable

from . import parsing
from .backends import Backend, build_backend
from .exceptions import ModelResponseError, VLCaptchaError
from .images import ImagePayload, load_image
from .prompts import (
    build_prompt,
    build_user_text,
    build_detect_prompt,
    build_click_select_user_text,
)

#: 默认的坐标归一化基准（Qwen-VL 等模型的常见约定）
DEFAULT_COORD_SCALE = 1000

#: 输出无法解析时的默认重试次数
DEFAULT_MAX_RETRIES = 2

_RETRY_HINT = (
    "\n\n注意：你上一次的回复不符合格式要求，无法解析。"
    "请重新作答，只输出一个符合格式的 JSON 对象，不要任何多余内容。"
)

KINDS = ("text", "calc", "click", "rotate", "trace", "slider")


class VLCaptcha:
    """用视觉语言模型识别验证码。

    :param llm: OpenAI / Anthropic 客户端、Backend 实例或可调用对象。

    每个方法的关键字参数：

    - ``extra_images``: 附加图片列表，与主图一起发送给模型。
    - ``detect``: 是否先做目标检测再识别（默认 ``True``）。
      设为 ``False`` 跳过检测，直接单步识别，省一次 API 调用。
    """

    def __init__(self, llm: Any) -> None:
        self.backend: Backend = build_backend(llm)
        self.coord_scale: int = DEFAULT_COORD_SCALE
        self.max_retries: int = DEFAULT_MAX_RETRIES

    # ------------------------------------------------------------ 公开接口

    @property
    def model(self) -> str:
        return self.backend.model

    def text(
        self,
        image: Any,
        question: str | None = None,
        *,
        extra_images: list[Any] | None = None,
        detect: bool = True,
    ) -> dict:
        """文字型验证码。返回字典，``result`` 为识别出的字符串。"""
        return self._solve("text", image, question, extra_images, detect, lambda d, r, i: parsing.parse_text(d, r))

    def calc(
        self,
        image: Any,
        question: str | None = None,
        *,
        extra_images: list[Any] | None = None,
        detect: bool = True,
    ) -> dict:
        """算术型验证码。返回字典，``result`` 为计算结果（int 或 float）。"""
        return self._solve("calc", image, question, extra_images, detect, lambda d, r, i: parsing.parse_calc(d, r))

    def click(
        self,
        image: Any,
        question: str | None = None,
        *,
        extra_images: list[Any] | None = None,
        detect: bool = True,
    ) -> dict:
        """点选型验证码。返回字典，``result`` 为按点击顺序排列的像素坐标。"""
        return self._solve(
            "click", image, question, extra_images, detect,
            lambda d, r, i: parsing.parse_click(d, r, i, self.coord_scale),
        )

    def rotate(
        self,
        image: Any,
        question: str | None = None,
        *,
        extra_images: list[Any] | None = None,
        detect: bool = True,
    ) -> dict:
        """旋转型验证码。返回字典，``result`` 为需要顺时针旋转的角度（0-359）。"""
        return self._solve("rotate", image, question, extra_images, detect, lambda d, r, i: parsing.parse_rotate(d, r))

    def trace(
        self,
        image: Any,
        question: str | None = None,
        *,
        extra_images: list[Any] | None = None,
        detect: bool = True,
    ) -> dict:
        """轨迹型验证码。返回字典，``result`` 为拖动路径的像素点列表。"""
        return self._solve(
            "trace", image, question, extra_images, detect,
            lambda d, r, i: parsing.parse_trace(d, r, i, self.coord_scale),
        )

    def slider(
        self,
        image: Any,
        question: str | None = None,
        *,
        extra_images: list[Any] | None = None,
        detect: bool = True,
    ) -> dict:
        """滑块型验证码。返回字典，``result`` 为滑块需要向右移动的像素距离。"""
        return self._solve(
            "slider", image, question, extra_images, detect,
            lambda d, r, i: parsing.parse_slider(d, r, i, self.coord_scale),
        )

    def solve(
        self,
        kind: str,
        image: Any,
        question: str | None = None,
        *,
        extra_images: list[Any] | None = None,
        detect: bool = True,
    ) -> dict:
        """按类型名调用对应方法，方便批量处理。"""
        if kind not in KINDS:
            raise VLCaptchaError(f"未知的验证码类型 {kind!r}，可选: {', '.join(KINDS)}")
        return getattr(self, kind)(image, question, extra_images=extra_images, detect=detect)

    def __repr__(self) -> str:  # pragma: no cover
        return f"VLCaptcha(backend={self.backend.name!r}, model={self.model!r})"

    # ------------------------------------------------------------ 内部实现

    def _load_extras(self, extra_images: list[Any] | None) -> list[ImagePayload] | None:
        if not extra_images:
            return None
        return [load_image(img) for img in extra_images]

    def _detect(
        self, payload: ImagePayload, extras: list[ImagePayload] | None,
    ) -> tuple[list[dict] | None, str | None]:
        """目标检测，返回 (对象列表, 原始输出)。失败返回 (None, None)。"""
        detect_system = build_detect_prompt(self.coord_scale)
        detect_user = "请检测这张图片中的所有对象。"

        for attempt in range(self.max_retries + 1):
            prompt = detect_user if attempt == 0 else detect_user + _RETRY_HINT
            raw_detect = self.backend.complete(detect_system, prompt, payload, extras)
            try:
                data = parsing.extract_json(raw_detect)
                detected = parsing.parse_detection(data, raw_detect)
                return detected, raw_detect
            except ModelResponseError:
                continue
        return None, None

    def _build_user_text(
        self, kind: str, question: str | None, detected: list[dict] | None,
    ) -> str:
        if kind == "click" and detected is not None:
            objects_json = _json.dumps(detected, ensure_ascii=False, indent=2)
            q = question or "请根据图片自带的题目提示，选出需要点击的目标。"
            return build_click_select_user_text(objects_json, q)

        base_text = build_user_text(kind, question)
        if detected is not None:
            objects_json = _json.dumps(detected, ensure_ascii=False, indent=2)
            return (
                f"以下是预先检测到的图中所有对象（供参考）：\n{objects_json}\n\n"
                f"请结合图片和上述检测结果来完成识别。\n\n"
                f"{base_text}"
            )
        return base_text

    def _build_system_prompt(self, kind: str, detected: list[dict] | None) -> str:
        if kind == "click" and detected is not None:
            return (
                "你是验证码点选专家。根据用户提供的检测结果和题目要求，"
                "从候选对象中选出正确的点击目标。只输出 JSON，不要任何多余内容。"
            )
        return build_prompt(kind, self.coord_scale)

    def _solve(
        self,
        kind: str,
        image: Any,
        question: str | None,
        extra_images: list[Any] | None,
        do_detect: bool,
        parse: Callable[[dict, str, ImagePayload], dict],
    ) -> dict:
        payload = load_image(image)
        extras = self._load_extras(extra_images)

        # ---- 第一步：目标检测（可选）----
        detected: list[dict] | None = None
        raw_detect: str | None = None
        if do_detect:
            detected, raw_detect = self._detect(payload, extras)

        # ---- 第二步：识别 ----
        system = self._build_system_prompt(kind, detected)
        user_text = self._build_user_text(kind, question, detected)

        last_error: ModelResponseError | None = None
        for attempt in range(self.max_retries + 1):
            prompt = user_text if attempt == 0 else user_text + _RETRY_HINT
            raw = self.backend.complete(system, prompt, payload, extras)
            try:
                data = parsing.extract_json(raw)
                result = parse(data, raw, payload)
            except ModelResponseError as exc:
                last_error = exc
                continue
            result["raw"] = raw
            if raw_detect is not None:
                result["raw_detect"] = raw_detect
            if detected is not None:
                result["detected_objects"] = detected
            result["image"] = {"width": payload.width, "height": payload.height}
            result["model"] = self.model
            result["question"] = question
            return result

        assert last_error is not None
        raise last_error

"""把识别结果画到图片上，方便调试。需要安装 Pillow。"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any

from .exceptions import VLCaptchaError
from .images import load_image

_FONT_CANDIDATES = (
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/simsun.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)

RED = (255, 48, 48)
GREEN = (0, 220, 0)
YELLOW = (255, 220, 0)
CYAN = (0, 200, 255)
BLACK = (0, 0, 0)


def _require_pillow():
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:  # pragma: no cover
        raise VLCaptchaError(
            "可视化需要 Pillow，请先安装：pip install pillow"
        ) from exc
    return Image, ImageDraw, ImageFont


def _load_font(size: int):
    _, _, ImageFont = _require_pillow()
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _open(image: Any):
    Image, ImageDraw, _ = _require_pillow()
    payload = load_image(image)
    img = Image.open(io.BytesIO(payload.data)).convert("RGBA")
    return img, ImageDraw.Draw(img)


def _outlined_text(draw, x, y, label, font, fill, outline_color=BLACK, outline_width=2):
    """画带描边的文字，不需要背景色就能在任何底色上看清。"""
    for dx in range(-outline_width, outline_width + 1):
        for dy in range(-outline_width, outline_width + 1):
            if dx == 0 and dy == 0:
                continue
            draw.text((x + dx, y + dy), label, fill=outline_color, font=font)
    draw.text((x, y), label, fill=fill, font=font)


def _banner(draw, img, label: str, color=GREEN) -> None:
    font = _load_font(max(18, img.height // 12))
    box = draw.textbbox((0, 0), label, font=font)
    _outlined_text(draw, 8, 4 - box[1], label, font, fill=color)


def _centered(draw, img, label: str, color=YELLOW) -> None:
    font = _load_font(max(22, img.height // 8))
    box = draw.textbbox((0, 0), label, font=font)
    width, height = box[2] - box[0], box[3] - box[1]
    cx, cy = img.width / 2, img.height / 2
    _outlined_text(
        draw, cx - width / 2 - box[0], cy - height / 2 - box[1],
        label, font, fill=color, outline_width=3,
    )


def _markers(draw, img, points, color, connect: bool) -> None:
    radius = max(6, min(img.width, img.height) // 14)
    line_width = max(2, radius // 5)
    if connect and len(points) > 1:
        draw.line([tuple(p) for p in points], fill=color, width=line_width)
        radius = max(4, radius // 2)
    font = _load_font(max(11, int(radius * 1.2)))
    for index, (x, y) in enumerate(points, start=1):
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius], outline=color, width=line_width
        )
        label = str(index)
        box = draw.textbbox((0, 0), label, font=font)
        draw.text(
            (x - (box[2] - box[0]) / 2 - box[0], y - (box[3] - box[1]) / 2 - box[1]),
            label,
            fill=color,
            font=font,
        )


def _append_label(img, label: str, color=GREEN):
    """在原图下方拼接一条白底标签栏，把文字写在里面。"""
    Image, ImageDraw, _ = _require_pillow()
    font = _load_font(max(18, img.height // 12))
    box = ImageDraw.Draw(img).textbbox((0, 0), label, font=font)
    text_h = box[3] - box[1]
    bar_h = text_h + 16  # 上下各留 8px

    bar = Image.new("RGBA", (img.width, bar_h), (0, 0, 0, 255))
    bar_draw = ImageDraw.Draw(bar)
    bar_draw.text((8, 8 - box[1]), label, fill=color, font=font)

    combined = Image.new("RGBA", (img.width, img.height + bar_h))
    combined.paste(img, (0, 0))
    combined.paste(bar, (0, img.height))
    return combined


def _draw_text(image, result) -> Any:
    img, draw = _open(image)
    return _append_label(img, f"识别: {result}")


def _draw_calc(image, result) -> Any:
    img, draw = _open(image)
    return _append_label(img, f"答案: {result}", color=CYAN)


def _draw_click(image, result) -> Any:
    img, draw = _open(image)
    _markers(draw, img, result, RED, connect=False)
    return img


def _draw_rotate(image, result) -> Any:
    img, draw = _open(image)
    # 把图片顺时针旋转识别出的角度（PIL rotate 是逆时针，取负值）
    rotated = img.rotate(-result, resample=1, expand=False)
    draw = _require_pillow()[1].Draw(rotated)
    _append_label_on = _append_label(rotated, f"顺时针旋转 {result}°", color=YELLOW)
    return _append_label_on


def _draw_trace(image, result) -> Any:
    img, draw = _open(image)
    _markers(draw, img, result, CYAN, connect=True)
    radius = max(5, min(img.width, img.height) // 22)
    for point, color in ((result[0], GREEN), (result[-1], RED)):
        x, y = point
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color)
    return img


def _draw_slider(image, result) -> Any:
    img, draw = _open(image)
    y = img.height // 2
    end_x = max(0, min(int(result), img.width - 1))
    draw.line([0, y, end_x, y], fill=RED, width=3)
    for x in (0, end_x):
        draw.line([x, y - 14, x, y + 14], fill=RED, width=3)
    font = _load_font(max(16, img.height // 14))
    label = f"{result}px"
    box = draw.textbbox((0, 0), label, font=font)
    width = box[2] - box[0]
    lx = min(max(end_x / 2 - width / 2, 0), max(img.width - width, 0))
    _outlined_text(draw, lx, y - 38 - box[1], label, font, fill=YELLOW)
    return img


_DRAWERS = {
    "text": _draw_text,
    "calc": _draw_calc,
    "click": _draw_click,
    "rotate": _draw_rotate,
    "trace": _draw_trace,
    "slider": _draw_slider,
}


def visualize(image: Any, kind: str, result: Any, output: str | Path | None = None):
    """按类型把结果标注到图片上。

    :param image: 原图（路径、bytes、base64 等，与识别时一致）。
    :param kind: ``text`` / ``calc`` / ``click`` / ``rotate`` / ``trace`` / ``slider``。
    :param result: 对应方法的返回值。也可直接传 ``detail=True`` 的字典。
    :param output: 输出路径。为空且 ``image`` 是本地路径时，
        自动保存为 ``原名_类型.扩展名``。
    :return: 保存了文件就返回输出路径，否则返回 PIL Image 对象。
    """
    if kind not in _DRAWERS:
        raise VLCaptchaError(f"未知的验证码类型: {kind!r}")
    if isinstance(result, dict) and "result" in result:
        result = result["result"]

    img = _DRAWERS[kind](image, result)

    if output is None and isinstance(image, (str, Path)):
        path = Path(image)
        if path.is_file():
            output = path.with_name(f"{path.stem}_{kind}{path.suffix or '.png'}")

    if output is None:
        return img

    output = Path(output)
    # PNG 保留透明通道，其他格式转 RGB
    if output.suffix.lower() != ".png" and img.mode == "RGBA":
        img = img.convert("RGB")
    img.save(output)
    return str(output)

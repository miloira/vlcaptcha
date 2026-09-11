"""图片输入的统一处理：任意来源 -> 字节 + 格式 + 尺寸。

尺寸解析只读文件头，不依赖 Pillow；解析失败时才尝试用 Pillow 兜底。
"""

from __future__ import annotations

import base64
import binascii
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .exceptions import ImageError

_DATA_URL_RE = re.compile(
    r"^data:image/(?P<fmt>[A-Za-z0-9.+-]+);base64,(?P<data>.*)$", re.DOTALL
)


@dataclass(frozen=True)
class ImagePayload:
    """一张已经就绪的图片。"""

    data: bytes
    format: str
    width: int
    height: int

    @property
    def base64(self) -> str:
        return base64.b64encode(self.data).decode("ascii")

    @property
    def mime_type(self) -> str:
        return f"image/{self.format}"

    @property
    def data_url(self) -> str:
        return f"data:{self.mime_type};base64,{self.base64}"

    @property
    def size(self) -> tuple[int, int]:
        return self.width, self.height


def load_image(source: Any) -> ImagePayload:
    """把各种形式的图片输入统一成 :class:`ImagePayload`。

    支持：本地路径（str / Path）、bytes、base64 字符串、data URL、
    文件对象（含 read 方法）、PIL.Image 对象、已有的 ImagePayload。
    """
    if isinstance(source, ImagePayload):
        return source

    if isinstance(source, (bytes, bytearray, memoryview)):
        return _from_bytes(bytes(source))

    if isinstance(source, Path):
        return _from_path(source)

    if isinstance(source, str):
        return _from_str(source)

    if hasattr(source, "read"):
        data = source.read()
        if isinstance(data, str):
            raise ImageError("文件对象必须以二进制模式打开（'rb'）")
        return _from_bytes(data)

    # PIL.Image：有 save 和 size 就按 PIL 处理
    if hasattr(source, "save") and hasattr(source, "size"):
        return _from_pil(source)

    raise ImageError(f"不支持的图片输入类型: {type(source)!r}")


def _from_path(path: Path) -> ImagePayload:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ImageError(f"读取图片失败: {path}") from exc
    return _from_bytes(data)


def _from_str(source: str) -> ImagePayload:
    text = source.strip()

    match = _DATA_URL_RE.match(text)
    if match:
        return _from_bytes(_b64decode(match.group("data")))

    # 优先当成路径；路径过长或含非法字符时 OSError，直接忽略
    try:
        path = Path(text)
        if path.is_file():
            return _from_path(path)
    except OSError:
        pass

    if len(text) < 32:
        raise ImageError(f"图片路径不存在: {source[:120]}")

    return _from_bytes(_b64decode(text))


def _from_pil(image: Any) -> ImagePayload:
    import io

    fmt = (getattr(image, "format", None) or "PNG").lower()
    if fmt not in {"png", "jpeg", "gif", "webp", "bmp"}:
        fmt = "png"
    buffer = io.BytesIO()
    target = image
    if fmt == "jpeg" and getattr(image, "mode", "RGB") not in {"RGB", "L"}:
        target = image.convert("RGB")
    target.save(buffer, format=fmt.upper())
    data = buffer.getvalue()
    width, height = image.size
    return ImagePayload(data=data, format=fmt, width=int(width), height=int(height))


def _b64decode(text: str) -> bytes:
    cleaned = "".join(text.split())
    padding = (-len(cleaned)) % 4
    try:
        return base64.b64decode(cleaned + "=" * padding, validate=False)
    except (binascii.Error, ValueError) as exc:
        raise ImageError("base64 图片数据无法解码") from exc


def _from_bytes(data: bytes) -> ImagePayload:
    if not data:
        raise ImageError("图片数据为空")
    fmt = sniff_format(data)
    width, height = image_size(data, fmt)
    return ImagePayload(data=data, format=fmt, width=width, height=height)


def sniff_format(data: bytes) -> str:
    """根据文件头判断图片格式。"""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    if data[:2] == b"BM":
        return "bmp"
    if data[:4] in (b"II*\x00", b"MM\x00*"):
        return "tiff"
    raise ImageError("无法识别的图片格式（支持 png / jpeg / gif / webp / bmp / tiff）")


def image_size(data: bytes, fmt: str | None = None) -> tuple[int, int]:
    """只读文件头解析图片宽高，失败时退回 Pillow。"""
    fmt = fmt or sniff_format(data)
    try:
        if fmt == "png":
            return _png_size(data)
        if fmt == "jpeg":
            return _jpeg_size(data)
        if fmt == "gif":
            return _gif_size(data)
        if fmt == "webp":
            return _webp_size(data)
        if fmt == "bmp":
            return _bmp_size(data)
    except (struct.error, IndexError, ValueError):
        pass
    return _pillow_size(data)


def _png_size(data: bytes) -> tuple[int, int]:
    width, height = struct.unpack(">II", data[16:24])
    return int(width), int(height)


def _gif_size(data: bytes) -> tuple[int, int]:
    width, height = struct.unpack("<HH", data[6:10])
    return int(width), int(height)


def _bmp_size(data: bytes) -> tuple[int, int]:
    width, height = struct.unpack("<ii", data[18:26])
    return abs(int(width)), abs(int(height))


def _webp_size(data: bytes) -> tuple[int, int]:
    chunk = data[12:16]
    if chunk == b"VP8 ":
        width = struct.unpack("<H", data[26:28])[0] & 0x3FFF
        height = struct.unpack("<H", data[28:30])[0] & 0x3FFF
        return int(width), int(height)
    if chunk == b"VP8L":
        bits = struct.unpack("<I", data[21:25])[0]
        return int((bits & 0x3FFF) + 1), int(((bits >> 14) & 0x3FFF) + 1)
    if chunk == b"VP8X":
        width = int.from_bytes(data[24:27], "little") + 1
        height = int.from_bytes(data[27:30], "little") + 1
        return int(width), int(height)
    raise ValueError("未知的 WebP 子格式")


_SOF_MARKERS = {
    0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
    0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF,
}


def _jpeg_size(data: bytes) -> tuple[int, int]:
    index, total = 2, len(data)
    while index < total:
        if data[index] != 0xFF:
            index += 1
            continue
        while index < total and data[index] == 0xFF:
            index += 1
        if index >= total:
            break
        marker = data[index]
        index += 1
        # 无长度字段的标记
        if marker in (0x01, 0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            continue
        if index + 2 > total:
            break
        segment_length = struct.unpack(">H", data[index : index + 2])[0]
        if marker in _SOF_MARKERS:
            height, width = struct.unpack(">HH", data[index + 3 : index + 7])
            return int(width), int(height)
        if segment_length < 2:
            break
        index += segment_length
    raise ValueError("JPEG 中没有找到 SOF 段")


def _pillow_size(data: bytes) -> tuple[int, int]:
    try:
        import io

        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise ImageError(
            "无法解析图片尺寸，请安装 Pillow（pip install pillow）后重试"
        ) from exc
    try:
        with Image.open(io.BytesIO(data)) as image:
            return int(image.width), int(image.height)
    except Exception as exc:  # pragma: no cover
        raise ImageError("图片无法解码") from exc

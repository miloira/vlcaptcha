"""模型输出解析：提取 JSON、校验字段、把归一化坐标换算成像素。"""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Any

from .exceptions import ModelResponseError
from .images import ImagePayload

_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


# ---------------------------------------------------------------- JSON 提取

def extract_json(content: str) -> dict:
    """从模型输出里抠出 JSON 对象，容忍代码块、前后废话、单引号。"""
    if not content or not content.strip():
        raise ModelResponseError("模型返回内容为空", raw=content)

    text = content.strip()
    text = _FENCE_RE.sub("", text).strip()

    for candidate in _json_candidates(text):
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
        if isinstance(data, list):
            # 少数模型直接返回坐标数组
            return {"result": data}

    raise ModelResponseError(f"无法从模型输出中解析 JSON: {content[:400]}", raw=content)


def _json_candidates(text: str):
    yield text

    # 取第一个平衡的 {...}
    balanced = _balanced_slice(text, "{", "}")
    if balanced:
        yield balanced
        yield balanced.replace("'", '"')
        # 去掉尾随逗号
        yield re.sub(r",\s*([}\]])", r"\1", balanced)

    array = _balanced_slice(text, "[", "]")
    if array:
        yield array


def _balanced_slice(text: str, opening: str, closing: str) -> str | None:
    start = text.find(opening)
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


# ------------------------------------------------------------ 基础类型转换

def _pick(data: dict, *keys: str) -> Any:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return None


def _to_number(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ModelResponseError(f"字段 {field} 必须是数字，得到 {value!r}")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        normalized = unicodedata.normalize("NFKC", value).strip()
        normalized = normalized.replace("，", "").replace(",", "")
        match = _NUMBER_RE.search(normalized)
        if match:
            return float(match.group())
    raise ModelResponseError(f"字段 {field} 无法转成数字: {value!r}")


def _to_int(value: Any, field: str) -> int:
    return int(round(_to_number(value, field)))


def _tidy_number(value: float) -> int | float:
    """整数就返回 int，否则保留两位小数。"""
    if abs(value - round(value)) < 1e-9:
        return int(round(value))
    return round(value, 2)


def _as_point(value: Any, field: str) -> tuple[float, float]:
    if isinstance(value, dict):
        x = _pick(value, "x", "X", "left", "cx")
        y = _pick(value, "y", "Y", "top", "cy")
        if x is None or y is None:
            raise ModelResponseError(f"字段 {field} 缺少 x / y: {value!r}")
        return _to_number(x, f"{field}.x"), _to_number(y, f"{field}.y")
    if isinstance(value, (list, tuple)):
        if len(value) < 2:
            raise ModelResponseError(f"字段 {field} 至少需要两个数字: {value!r}")
        return _to_number(value[0], f"{field}[0]"), _to_number(value[1], f"{field}[1]")
    raise ModelResponseError(f"字段 {field} 不是坐标: {value!r}")


def _clamp(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


# ------------------------------------------------------------ 坐标换算

def to_pixel(
    point: tuple[float, float],
    image: ImagePayload,
    coord_scale: int | None,
) -> list[int]:
    """归一化坐标 -> 像素坐标。

    ``coord_scale`` 为 None 时认为模型直接给的就是像素；
    否则当两个分量都不超过 scale 时按归一化换算，
    超出 scale 说明模型忽略了约定、直接给了像素，此时按像素处理。
    """
    x, y = point
    if coord_scale and x <= coord_scale and y <= coord_scale:
        x = x / coord_scale * image.width
        y = y / coord_scale * image.height
    return [
        int(round(_clamp(x, 0, max(image.width - 1, 0)))),
        int(round(_clamp(y, 0, max(image.height - 1, 0)))),
    ]


def to_pixel_distance(
    value: float,
    image: ImagePayload,
    coord_scale: int | None,
) -> int:
    """归一化水平距离 -> 像素距离。"""
    if coord_scale and abs(value) <= coord_scale:
        value = value / coord_scale * image.width
    return int(round(_clamp(value, -image.width, image.width)))


# ------------------------------------------------------------ 各类型解析

def parse_text(data: dict, raw: str) -> dict:
    value = _pick(data, "result", "text", "answer", "content", "captcha")
    if value is None:
        raise ModelResponseError(f"缺少 result 字段: {raw[:200]}", raw=raw)
    if isinstance(value, (list, tuple)):
        value = "".join(str(item) for item in value)
    result = str(value).strip()
    if not result:
        raise ModelResponseError("识别结果为空字符串", raw=raw)

    detail = {"type": "text", "result": result, "length": len(result)}
    claimed = _pick(data, "length", "count")
    if claimed is not None:
        try:
            detail["model_length"] = _to_int(claimed, "length")
        except ModelResponseError:
            pass
    return detail


def parse_calc(data: dict, raw: str) -> dict:
    value = _pick(data, "result", "answer", "value")
    if value is None:
        raise ModelResponseError(f"缺少 result 字段: {raw[:200]}", raw=raw)
    number = _tidy_number(_to_number(value, "result"))
    expression = _pick(data, "expression", "expr", "equation", "formula")
    return {
        "type": "calc",
        "result": number,
        "expression": str(expression).strip() if expression is not None else None,
    }


def parse_click(data: dict, raw: str, image: ImagePayload, coord_scale: int | None) -> dict:
    items = _pick(data, "result", "points", "targets", "clicks", "coordinates")
    if items is None:
        raise ModelResponseError(f"缺少 result 字段: {raw[:200]}", raw=raw)
    if isinstance(items, dict):
        items = [items]
    if not isinstance(items, (list, tuple)) or not items:
        raise ModelResponseError(f"result 不是非空的坐标数组: {items!r}", raw=raw)

    # 兼容模型直接给出单个点 [x, y] 的情况
    if len(items) == 2 and all(isinstance(v, (int, float, str)) for v in items):
        items = [list(items)]

    points: list[list[int]] = []
    normalized: list[list[float]] = []
    labels: list[str | None] = []

    for index, item in enumerate(items):
        label = None
        raw_point: Any = item
        if isinstance(item, dict):
            label = _pick(item, "label", "name", "target", "text", "char")
            raw_point = _pick(item, "point", "position", "coordinate", "coord", "xy")
            if raw_point is None:
                raw_point = item  # 形如 {"x": 1, "y": 2}
        point = _as_point(raw_point, f"result[{index}]")
        normalized.append([_tidy_number(point[0]), _tidy_number(point[1])])
        points.append(to_pixel(point, image, coord_scale))
        labels.append(str(label).strip() if label is not None else None)

    return {
        "type": "click",
        "result": points,
        "normalized": normalized,
        "labels": labels,
        "count": len(points),
    }


def parse_rotate(data: dict, raw: str) -> dict:
    value = _pick(data, "result", "angle", "degree", "degrees", "rotation")
    if value is None:
        raise ModelResponseError(f"缺少 result 字段: {raw[:200]}", raw=raw)
    angle = _to_int(value, "result") % 360

    direction = _pick(data, "direction", "clockwise")
    # 模型明确说是逆时针时换算回顺时针
    if isinstance(direction, str) and re.search(
        r"逆时针|counter|anti", direction, re.IGNORECASE
    ):
        angle = (360 - angle) % 360
    elif direction is False:
        angle = (360 - angle) % 360

    return {
        "type": "rotate",
        "result": angle,
        "clockwise": angle,
        "counter_clockwise": (360 - angle) % 360,
    }


def parse_trace(data: dict, raw: str, image: ImagePayload, coord_scale: int | None) -> dict:
    items = _pick(data, "result", "path", "points", "track", "trace")
    if items is None:
        raise ModelResponseError(f"缺少 result 字段: {raw[:200]}", raw=raw)
    if isinstance(items, dict):
        items = _pick(items, "points", "path") or []
    if not isinstance(items, (list, tuple)) or len(items) < 2:
        raise ModelResponseError(f"轨迹至少需要 2 个点: {items!r}", raw=raw)

    points: list[list[int]] = []
    normalized: list[list[float]] = []
    for index, item in enumerate(items):
        raw_point = item
        if isinstance(item, dict) and _pick(item, "point", "position") is not None:
            raw_point = _pick(item, "point", "position")
        point = _as_point(raw_point, f"result[{index}]")
        normalized.append([_tidy_number(point[0]), _tidy_number(point[1])])
        points.append(to_pixel(point, image, coord_scale))

    # 去掉相邻重复点
    deduped = [points[0]]
    for point in points[1:]:
        if point != deduped[-1]:
            deduped.append(point)

    return {
        "type": "trace",
        "result": deduped,
        "normalized": normalized,
        "start": deduped[0],
        "end": deduped[-1],
        "count": len(deduped),
    }


def parse_slider(data: dict, raw: str, image: ImagePayload, coord_scale: int | None) -> dict:
    slider_x = _pick(data, "slider_x", "sliderX", "piece_x", "start_x", "from_x")
    gap_x = _pick(data, "gap_x", "gapX", "target_x", "hole_x", "end_x", "to_x")
    gap_y = _pick(data, "gap_y", "gapY", "target_y", "hole_y")
    distance = _pick(data, "result", "distance", "offset", "dx", "move")

    if distance is None and (slider_x is None or gap_x is None):
        raise ModelResponseError(f"缺少 result / gap_x 字段: {raw[:200]}", raw=raw)

    if distance is None:
        raw_distance = _to_number(gap_x, "gap_x") - _to_number(slider_x, "slider_x")
    else:
        raw_distance = _to_number(distance, "result")

    detail: dict[str, Any] = {
        "type": "slider",
        "result": to_pixel_distance(raw_distance, image, coord_scale),
        "normalized": _tidy_number(raw_distance),
    }
    if slider_x is not None:
        detail["slider_x"] = to_pixel_distance(
            _to_number(slider_x, "slider_x"), image, coord_scale
        )
    if gap_x is not None:
        gap_px = [to_pixel_distance(_to_number(gap_x, "gap_x"), image, coord_scale), None]
        if gap_y is not None:
            gap_px[1] = to_pixel(
                (_to_number(gap_x, "gap_x"), _to_number(gap_y, "gap_y")),
                image,
                coord_scale,
            )[1]
        detail["gap_x"] = gap_px[0]
        if gap_px[1] is not None:
            detail["gap_y"] = gap_px[1]
    return detail


# ------------------------------------------------------------ 目标检测结果解析

def parse_detection(data: dict, raw: str) -> list[dict]:
    """解析目标检测的 JSON，返回对象列表。

    每个对象至少有 label 和 center，可选 bbox。
    """
    objects = _pick(data, "objects", "items", "targets", "detections", "result")
    if objects is None:
        raise ModelResponseError(f"缺少 objects 字段: {raw[:200]}", raw=raw)
    if isinstance(objects, dict):
        objects = [objects]
    if not isinstance(objects, (list, tuple)) or not objects:
        raise ModelResponseError(f"objects 不是非空数组: {objects!r}", raw=raw)

    result = []
    for idx, obj in enumerate(objects):
        if not isinstance(obj, dict):
            continue
        label = _pick(obj, "label", "name", "text", "description", "content")
        if label is None:
            label = f"object_{idx}"
        label = str(label).strip()

        center = _pick(obj, "center", "point", "position", "coord")
        bbox = _pick(obj, "bbox", "box", "rect", "rectangle", "bounding_box")

        item: dict[str, Any] = {"label": label}

        if center is not None:
            try:
                cx, cy = _as_point(center, f"objects[{idx}].center")
                item["center"] = [_tidy_number(cx), _tidy_number(cy)]
            except ModelResponseError:
                pass

        if bbox is not None and isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            try:
                coords = [_to_number(v, f"objects[{idx}].bbox[{i}]") for i, v in enumerate(bbox[:4])]
                item["bbox"] = [_tidy_number(c) for c in coords]
                # 如果没有 center，从 bbox 算
                if "center" not in item:
                    item["center"] = [
                        _tidy_number((coords[0] + coords[2]) / 2),
                        _tidy_number((coords[1] + coords[3]) / 2),
                    ]
            except ModelResponseError:
                pass

        # 必须有坐标才有意义
        if "center" in item:
            result.append(item)

    if not result:
        raise ModelResponseError(f"未检测到任何有效对象: {raw[:200]}", raw=raw)
    return result

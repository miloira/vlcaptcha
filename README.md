# vlcaptcha

vlcaptcha is a universal CAPTCHA recognition library powered by Vision-Language Models (VLM). It treats CAPTCHA solving as a multimodal understanding task, handling multiple CAPTCHA types through a single unified interface — with no need to train a separate model for each type.

## 安装

```bash
pip install openai        # 或 pip install anthropic
pip install pillow        # 可选，只在需要结果可视化时用
```

核心库本身零依赖，图片尺寸直接从文件头解析。

## 快速开始

```python
from openai import OpenAI
from vlcaptcha import VLCaptcha

llm = OpenAI()
vlcaptcha = VLCaptcha(llm)

vlcaptcha.text("text.png")      # -> "aB3d"        文字型
vlcaptcha.calc("calc.png")      # -> 12            算术型
vlcaptcha.click("click.png")    # -> [[123, 88], [210, 96]]   点选型（像素坐标）
vlcaptcha.rotate("rotate.png")  # -> 90            旋转型（需顺时针旋转的角度）
vlcaptcha.trace("trace.png")    # -> [[10, 20], ..., [180, 60]]  轨迹型（拖动路径）
vlcaptcha.slider("slider.png")  # -> 156           滑块型（向右移动的像素距离）
```

Anthropic 协议只换客户端，其余用法一致：

```python
from anthropic import Anthropic

vlcaptcha = VLCaptcha(Anthropic())
```

OpenAI 兼容网关（DashScope、vLLM、Ollama、OpenRouter 等）传 `base_url` 即可：

```python
client = OpenAI(api_key="sk-xxx", base_url="https://.../compatible-mode/v1")
vlcaptcha = VLCaptcha(client)
```

## 指定模型（高级用法）

默认情况下 OpenAI 协议用 `gpt-4o`，Anthropic 协议用 `claude-sonnet-4-6`。
如需指定模型或传入额外请求参数，直接构造 `Backend` 实例：

```python
from vlcaptcha import VLCaptcha, OpenAIBackend, AnthropicBackend

# OpenAI 协议 + 指定模型
backend = OpenAIBackend(OpenAI(), model="gpt-4o-mini", request_params={"temperature": 0})
vlcaptcha = VLCaptcha(backend)

# Anthropic 协议 + 指定模型
backend = AnthropicBackend(Anthropic(), model="claude-sonnet-4-5-20250929")
vlcaptcha = VLCaptcha(backend)
```

## 六种类型与输出格式

每种类型都有一段专用系统提示词，规定唯一的 JSON 输出格式，解析后再换算成可直接使用的值。

| 方法 | 模型输出的 JSON | 方法返回值 |
| --- | --- | --- |
| `text` | `{"type": "text", "result": "aB3d", "length": 4}` | `str` |
| `calc` | `{"type": "calc", "expression": "3+5*2", "result": 13}` | `int \| float` |
| `click` | `{"type": "click", "result": [{"label": "春", "point": [x, y]}, ...]}` | `list[[x, y]]` 像素坐标，按点击顺序 |
| `rotate` | `{"type": "rotate", "result": 90}` | `int`，需顺时针旋转的角度 0-359 |
| `trace` | `{"type": "trace", "result": [[x, y], ...]}` | `list[[x, y]]` 像素路径，首点按下、末点松开 |
| `slider` | `{"type": "slider", "result": 520, "slider_x": 80, "gap_x": 600, "gap_y": 400}` | `int`，向右移动的像素距离 |

坐标类提示词要求模型输出 0-1000 的归一化相对坐标，库内部按图片实际宽高换算成像素。

## 常用参数

- **题目提示**：点选、轨迹等类型可以带上题干，`vlcaptcha.click(img, "依次点击：春 夏 秋 冬")`。
- **图片入参**：支持路径、`Path`、`bytes`、base64 字符串、data URL、文件对象、`PIL.Image`。
- **完整结果**：`detail=True` 返回结构化字典，含归一化坐标、label、图片尺寸和模型原始输出。

```python
detail = vlcaptcha.click("click.png", "点击图中的苹果", detail=True)
# {'type': 'click', 'result': [[30, 30]], 'normalized': [[100, 200]],
#  'labels': ['苹果'], 'count': 1, 'image': {'width': 300, 'height': 150}, ...}
```

## 结果可视化（需要 Pillow）

```python
from vlcaptcha import visualize

points = vlcaptcha.click("click.png")
visualize("click.png", "click", points)   # -> click_click.png
```

## 命令行

```bash
python -m vlcaptcha text captcha.png
python -m vlcaptcha click click.png -q "依次点击：春 夏 秋 冬" --draw
python -m vlcaptcha rotate rotate.png -p anthropic -m claude-sonnet-4-6 --detail
```

凭据取环境变量：`OPENAI_API_KEY`、`OPENAI_BASE_URL`、`ANTHROPIC_API_KEY`，模型可用 `VLCAPTCHA_MODEL`。

## 异常

- `ConfigurationError`：客户端或参数不对（例如传了异步客户端）。
- `ImageError`：图片读不了或格式不支持。
- `ModelResponseError`：重试后仍无法解析模型输出，`.raw` 里是原始回复。

三者都继承 `VLCaptchaError`。

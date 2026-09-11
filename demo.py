"""
vlcaptcha 使用示例

演示六种验证码类型的调用方式，同时展示 OpenAI 协议和 Anthropic 协议的接入方法。
运行前请设置对应的环境变量（API Key），或直接在代码里修改。

用法：
    python demo.py
"""

import os

from vlcaptcha import VLCaptcha, OpenAIBackend, AnthropicBackend, visualize


# ======================== 1. 创建实例 ========================

# --- OpenAI 协议（也适用于 DashScope、vLLM、Ollama、OpenRouter 等兼容网关）---
from openai import OpenAI

llm = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY", "sk-ws-H.ERYRIMI.BRvH.MEUCIHujpTeWu453cQAc33--KrTLZxGGDXbAhtHm6-x4XAxEAiEAzWtmj_3GkBkn4vJQh6lm84vDLK62zQJD9jmr9u7BXYE"),
    base_url="https://llm-yzzgueic8ey10mne.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
)
# model 不是 OpenAI() 的参数，需要通过 OpenAIBackend 指定
captcha = VLCaptcha(OpenAIBackend(llm, model="qwen3.5-omni-plus", request_params={"modalities": ["text"]}))

# --- Anthropic 协议 ---
# from anthropic import Anthropic
# llm = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", "sk-ant-xxx"))
# captcha = VLCaptcha(llm)

# --- 指定模型 + 请求参数（高级用法）---
# backend = OpenAIBackend(llm, model="qwen-vl-max", request_params={"temperature": 0})
# captcha = VLCaptcha(backend)


# ======================== 2. 六种验证码类型示例 ========================
# 所有方法默认返回完整字典，result 字段是核心结果

def demo_text():
    """文字型：识别图中的字符。"""
    image = "samples/text.png"
    data = captcha.text(image)
    print(f"[text] 识别结果: {data['result']!r}")
    print(f"  字符数: {data['length']}")

    visualize(image, "text", data["result"])
    print(f"  标注图片已保存")


def demo_calc():
    """算术型：识别并计算图中的算式。"""
    image = "samples/calc.png"
    data = captcha.calc(image, question="计算图中的算式")
    print(f"[calc] 计算结果: {data['result']}")
    print(f"  算式: {data.get('expression')}")

    visualize(image, "calc", data["result"])
    print(f"  标注图片已保存")


def demo_click():
    """点选型：给出需要依次点击的坐标。"""
    image = "samples/click15.png"
    data = captcha.click(image, question="选择全部 时钟")
    
    print(data)
    print(f"[click] 点击坐标（像素）: {data['result']}")
    print(f"  归一化坐标: {data['normalized']}")
    print(f"  标签: {data['labels']}")
    print(f"  点击数: {data['count']}")
    if "detected_objects" in data:
        print(f"  检测到对象数: {len(data['detected_objects'])}")

    visualize(image, "click", data["result"])
    print(f"  标注图片已保存")


def demo_rotate():
    """旋转型：判断图片需要顺时针旋转多少度。"""
    image = "samples/rotate.png"
    data = captcha.rotate(image, question="旋转图片到里面的场景正常")
    print(f"[rotate] 需顺时针旋转: {data['result']}°")
    print(f"  顺时针: {data['clockwise']}°")
    print(f"  逆时针: {data['counter_clockwise']}°")

    visualize(image, "rotate", data["result"])
    print(f"  标注图片已保存")


def demo_trace():
    """轨迹型：给出鼠标拖动的完整路径。"""
    image = "samples/trace4.png"
    data = captcha.trace(image, question="用最短线连接蓝色验证")
    print(f"[trace] 路径点数: {len(data['result'])}")
    print(f"  起点: {data['start']}, 终点: {data['end']}")

    visualize(image, "trace", data["result"])
    print(f"  标注图片已保存")


def demo_slider():
    """滑块型：计算滑块需要向右移动的像素距离。"""
    image = "samples/bg.png"
    image2 = "samples/sl.png"
    data = captcha.slider(image, detect=False, extra_images=[image2])
    print(data)
    print(f"[slider] 向右移动: {data['result']}px")
    print(f"  滑块位置: {data.get('slider_x')}px")
    print(f"  缺口位置: {data.get('gap_x')}px")

    visualize(image, "slider", data["result"])
    print(f"  标注图片已保存")


# ======================== 3. 批量处理 ========================

def demo_batch():
    """用 solve() 按类型名统一调度，方便批量处理。"""
    tasks = [
        ("text",   "samples/text.png",   None),
        ("calc",   "samples/calc.png",   "计算图中的算式"),
        ("click",  "samples/click.png",  "依次点击：春 夏 秋 冬"),
        ("rotate", "samples/rotate.png", None),
        ("trace",  "samples/trace.png",  None),
        ("slider", "samples/slider.png", None),
    ]
    for kind, image, question in tasks:
        data = captcha.solve(kind, image, question)
        print(f"[batch] {kind:>6s} -> {data['result']}")


# ======================== 运行 ========================

if __name__ == "__main__":
    # 取消注释你要跑的示例（需要对应的样本图片）：
    # demo_text()
    # demo_calc()
    # demo_click()
    # demo_rotate()
    # demo_trace()
    demo_slider()
    # demo_batch()

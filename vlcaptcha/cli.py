"""命令行入口：``python -m vlcaptcha <类型> <图片>``。"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .backends import AnthropicBackend, OpenAIBackend
from .core import KINDS, VLCaptcha


def build_backend(protocol: str, model: str | None):
    """按协议和模型创建 Backend，凭据从环境变量读取。"""
    if protocol == "anthropic":
        from anthropic import Anthropic

        return AnthropicBackend(
            Anthropic(),
            model=model or "claude-sonnet-4-6",
        )

    from openai import OpenAI

    return OpenAIBackend(
        OpenAI(base_url=os.getenv("OPENAI_BASE_URL") or None),
        model=model or "gpt-4o",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vlcaptcha", description="用视觉语言模型识别验证码"
    )
    parser.add_argument("kind", choices=KINDS, help="验证码类型")
    parser.add_argument("image", help="验证码图片路径")
    parser.add_argument("-q", "--question", default=None, help="验证码题目（可选）")
    parser.add_argument(
        "-p",
        "--protocol",
        choices=("openai", "anthropic"),
        default=os.getenv("VLCAPTCHA_PROTOCOL", "openai"),
        help="LLM 协议，默认 openai",
    )
    parser.add_argument(
        "-m", "--model", default=os.getenv("VLCAPTCHA_MODEL"), help="模型名"
    )
    parser.add_argument("--draw", action="store_true", help="生成标注图片")
    parser.add_argument("--detail", action="store_true", help="打印完整结构化结果")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    backend = build_backend(args.protocol, args.model)
    captcha = VLCaptcha(backend)
    detail = captcha.solve(args.kind, args.image, args.question)

    if args.detail:
        print(
            json.dumps(
                {k: v for k, v in detail.items() if k not in ("raw", "raw_detect")},
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(f"{args.kind} -> {detail['result']}")

    if args.draw:
        from .visualize import visualize

        print("标注图片:", visualize(args.image, args.kind, detail["result"]))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

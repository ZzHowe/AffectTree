"""AffectTree 树状情感推理入口（骨架）。

用法：
    python scripts/run_reason.py --video path/to.mp4 --query "主角情绪何时由平静转为愤怒？"

策略（Policy）需接入 VLM 推理后端；视频 IO 需实现 sample / audio 接口。
"""
from __future__ import annotations

import argparse

from affecttree.reasoning.tree import TreeSearchEngine
from affecttree.tools import base as tools_base  # noqa: F401  # 触发工具注册
from affecttree.tools import navigation, visual_ops, face_au, pose_gesture, prosody, scene_tone, affect_spot  # noqa: F401


def main() -> None:
    parser = argparse.ArgumentParser(description="AffectTree reasoning demo")
    parser.add_argument("--video", required=True, help="视频路径")
    parser.add_argument("--query", required=True, help="情感问题")
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--max-width", type=int, default=3)
    args = parser.parse_args()

    raise NotImplementedError(
        "待实现：1) VideoReader（sample/audio 接口）；2) VLM Policy 后端；"
        "3) tools_base.build_tools() 装配工具并运行 TreeSearchEngine"
    )


if __name__ == "__main__":
    main()

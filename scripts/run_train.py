"""AffectTree 训练入口（骨架）。

Stage 1 (SFT warmup) 与 Stage 2 (Emo-RPO) 的分布式训练复用仓库 train/
目录下的 verl（EasyR1）框架；AffectTree 侧需要提供：
  1. SFT 数据（affecttree/training/sft.py 的样本格式与过滤）
  2. Emo-RPO 奖励函数（affecttree/training/emo_rpo.py 的 EmoRPO.advantages）
  3. 情感工具执行环境（affecttree/tools/ 接入 rollout worker）
"""
from __future__ import annotations


def main() -> None:
    raise NotImplementedError(
        "待实现：1) SFT 数据构建与过滤；2) 将 EmoRPO.advantages 注册为 "
        "verl reward function；3) 工具环境与 rollout worker 对接（参考 train/verl/workers/rollout/tool_video.py）"
    )


if __name__ == "__main__":
    main()

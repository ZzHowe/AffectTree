"""Stage 1：SFT warmup。

用教师模型在情感 QA 数据上生成多轮工具调用轨迹（含工具参数与中间
符号化结果），过滤最终答案错误或弧线偏差过大的样本后，训练模型掌握
工具调用格式与基本情感推理模式。

训练损失：L = -Σ_t log πθ(r_t, a_tool | q, r_<t, F_<t)。

说明：AffectTree 的 RL 训练基座复用仓库 train/ 目录下的 verl
（EasyR1）框架；本模块只定义数据格式与过滤逻辑，实际训练入口见
scripts/run_train.py。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolUseStep:
    """一轮工具调用：推理文本 + 工具动作 + 符号化执行结果。"""

    reasoning: str
    action: dict[str, Any]  # {"tool": name, "params": {...}}
    observation: str = ""


@dataclass
class SFTSample:
    """一条 SFT 轨迹样本。"""

    query: str
    video_id: str
    steps: list[ToolUseStep] = field(default_factory=list)
    final_answer: str = ""
    correct: bool = False
    va_ccc: float = 0.0  # 轨迹预测 VA 与标注的 CCC（过滤用）


def filter_samples(samples: list[SFTSample], ccc_min: float = 0.3) -> list[SFTSample]:
    """SFT 数据过滤：答案错误或弧线偏差过大的样本剔除。"""
    return [s for s in samples if s.correct and s.va_ccc >= ccc_min]

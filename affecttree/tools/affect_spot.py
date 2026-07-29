"""AffectSpot：多模态信号融合与情感转折点检测。

对各工具输出的 VA 信号流归一化融合，用 CUSUM / BOCPD 变点检测
定位情感状态显著变化的时间边界，确认或否决当前假设。
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseTool, ToolOutput, register
from ..reasoning.state import Evidence, VAEstimate


def fuse_va(traces: list[VAEstimate], weights: dict[str, float] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """把多条 VA 轨迹按工具先验权重融合成单一时间轴上的 (t, v, a)。

    TODO: 时间对齐（插值到统一网格）+ 加权平均。面部为主通道，
    肢体 / 语音校准 arousal，场景校准 valence。
    """
    raise NotImplementedError


def cusum(x: np.ndarray, threshold: float = 5.0, drift: float = 0.0) -> list[int]:
    """单边 CUSUM 变点检测，返回变点下标。"""
    x = np.asarray(x, dtype=np.float64)
    mu, s = x.mean(), x.std() + 1e-12
    z = (x - mu) / s
    g, points = 0.0, []
    for i, zi in enumerate(z):
        g = max(0.0, g + zi - drift)
        if g > threshold:
            points.append(i)
            g = 0.0
    return points


@register
class AffectSpot(BaseTool):
    """融合节点累计 VA 信号并检测转折点，输出验证结论。"""

    name = "affect_spot"
    description = "AffectSpot(信号流)：融合 VA 轨迹、转折点候选"

    def __init__(self, threshold: float = 5.0) -> None:
        self.threshold = threshold

    def run(self, video: Any, node: Any, **_: Any) -> ToolOutput:
        if len(node.va_trace) < 3:
            return ToolOutput(text="affect_spot: 信号不足，继续探索", pruned=False)
        v = np.array([e.valence for e in node.va_trace])
        a = np.array([e.arousal for e in node.va_trace])
        points = sorted(set(cusum(v, self.threshold)) | set(cusum(a, self.threshold)))
        if not points:
            return ToolOutput(text="affect_spot: 未检出变点，当前假设证据不足", pruned=True)
        tps = [float(node.va_trace[i].t_start) for i in points]
        ev = [
            Evidence(modality="fused", tool=self.name, t_start=t, t_end=t, payload={"kind": "transition"})
            for t in tps
        ]
        return ToolOutput(text=f"affect_spot: 检出 {len(tps)} 个转折点 @ {tps}", evidence=ev, pruned=False)

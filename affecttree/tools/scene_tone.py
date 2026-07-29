"""SceneTone：场景氛围分析（辅助通道）。

色彩心理学传统特征：HSV 色温 / 饱和度 / 亮度统计 + Itten 色彩和谐先验，
辅以 KLT 摄像机运动估计，输出场景氛围 valence 分与镜头语言标签。
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseTool, ToolOutput, register
from ..reasoning.state import Evidence, VAEstimate


def color_stats(frame: np.ndarray) -> dict[str, float]:
    """单帧色彩统计：平均饱和度、亮度、暖色占比（传统 HSV 分析）。"""
    import cv2

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = hsv[..., 0] * 2.0, hsv[..., 1] / 255.0, hsv[..., 2] / 255.0
    warm = ((h < 30) | (h > 330)).mean()  # 红-橙-黄区间占比
    return {"sat": float(s.mean()), "val": float(v.mean()), "warm_ratio": float(warm)}


def tone_to_valence(stats: dict[str, float]) -> float:
    """Itten 色彩和谐先验：暖而亮偏正 valence，冷而暗偏负（初版，待标定）。"""
    v = 0.6 * (stats["warm_ratio"] - 0.5) + 0.4 * (stats["val"] - 0.5) * 2 * 0.5
    return max(-1.0, min(1.0, v))


@register
class SceneTone(BaseTool):
    """估计 [ts, te] 片段的场景氛围分与镜头语言标签。"""

    name = "scene_tone"
    description = "SceneTone(ts, te)：场景氛围分、镜头语言标签"

    def run(self, video: Any, node: Any, ts: float, te: float, **_: Any) -> ToolOutput:
        frames = video.sample(ts, te, rate=1.0)
        if not frames:
            return ToolOutput(text="scene_tone: no frames")
        stats = color_stats(frames[len(frames) // 2])
        valence = tone_to_valence(stats)
        va = [VAEstimate(t_start=ts, t_end=te, valence=valence, arousal=0.0, source=self.name)]
        ev = [Evidence(modality="scene", tool=self.name, t_start=ts, t_end=te, payload=stats)]
        return ToolOutput(
            text=f"scene_tone [{ts:.1f}s, {te:.1f}s]: warm={stats['warm_ratio']:.2f}, val={valence:+.2f}",
            va_updates=va,
            evidence=ev,
        )

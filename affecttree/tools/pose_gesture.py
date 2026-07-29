"""PoseGesture：肢体语言辅助通道。

传统特征：光流运动能量（MBH 风格）+ 肩线开合等几何规则，
输出肢体激活度与姿态开放性，作为 arousal 校准线索。
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseTool, ToolOutput, register
from ..reasoning.state import Evidence, VAEstimate


def motion_energy(prev: np.ndarray, cur: np.ndarray) -> float:
    """相邻帧稠密光流平均幅值（Farneback，传统方法）。"""
    import cv2

    g0 = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY) if prev.ndim == 3 else prev
    g1 = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY) if cur.ndim == 3 else cur
    flow = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return float(mag.mean())


@register
class PoseGesture(BaseTool):
    """对 BodyCrop 区域序列估计肢体激活度与开放性。"""

    name = "pose_gesture"
    description = "PoseGesture(body_crop)：肢体激活度、姿态开放性"

    def run(self, video: Any, node: Any, track_id: int, ts: float, te: float, **_: Any) -> ToolOutput:
        frames = video.sample(ts, te, rate=2.0)  # TODO: 替换为 BodyCrop 区域序列
        if len(frames) < 2:
            return ToolOutput(text="pose_gesture: insufficient frames")
        energy = float(np.mean([motion_energy(p, c) for p, c in zip(frames, frames[1:])]))
        arousal = max(-1.0, min(1.0, energy / 10.0))  # 初版线性归一，待标定
        va = [VAEstimate(t_start=ts, t_end=te, valence=0.0, arousal=arousal, source=self.name)]
        ev = [Evidence(modality="body", tool=self.name, t_start=ts, t_end=te, payload={"motion_energy": energy})]
        return ToolOutput(text=f"pose_gesture(track {track_id}): motion energy {energy:.2f}", va_updates=va, evidence=ev)

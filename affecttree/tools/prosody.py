"""Prosody：语音韵律情感分析（辅助通道）。

传统信号处理：短时能量、过零率、YIN 基频（F0）、语速与停顿。
输出 F0 / 能量曲线与唤醒线索——arousal 强、valence 弱；
影视场景需 VAD 前置以避开 BGM 与重叠语音。
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseTool, ToolOutput, register
from ..reasoning.state import Evidence, VAEstimate


def short_time_energy(wav: np.ndarray, frame: int = 400, hop: int = 160) -> np.ndarray:
    """分帧短时能量。"""
    n = 1 + max(0, (len(wav) - frame) // hop)
    frames = np.lib.stride_tricks.as_strided(
        wav, shape=(n, frame), strides=(wav.strides[0] * hop, wav.strides[0])
    )
    return (frames**2).sum(axis=1)


def zero_crossing_rate(wav: np.ndarray, frame: int = 400, hop: int = 160) -> np.ndarray:
    """分帧过零率。"""
    signs = np.signbit(wav)
    n = 1 + max(0, (len(wav) - frame) // hop)
    frames = np.lib.stride_tricks.as_strided(
        signs, shape=(n, frame), strides=(signs.strides[0] * hop, signs.strides[0])
    )
    return (frames[:, 1:] != frames[:, :-1]).mean(axis=1)


def estimate_f0_autocorr(wav: np.ndarray, sr: int, fmin: int = 50, fmax: int = 500) -> np.ndarray:
    """自相关基频估计（YIN 的简化版）。

    TODO: 完整 YIN（差分函数 + 累积均值归一 + 抛物线插值）。
    """
    raise NotImplementedError


@register
class Prosody(BaseTool):
    """对 [ts, te] 音轨做韵律分析，输出语音唤醒线索。"""

    name = "prosody"
    description = "Prosody(ts, te)：F0 曲线、能量、停顿、jitter"

    def run(self, video: Any, node: Any, ts: float, te: float, **_: Any) -> ToolOutput:
        wav, sr = video.audio(ts, te)  # TODO: VideoReader 音轨接口
        energy = short_time_energy(wav)
        level = float(np.log10(energy.mean() + 1e-12))
        arousal = max(-1.0, min(1.0, (level + 4.0) / 2.0))  # 初版归一，待标定
        va = [VAEstimate(t_start=ts, t_end=te, valence=0.0, arousal=arousal, source=self.name)]
        ev = [Evidence(modality="voice", tool=self.name, t_start=ts, t_end=te, payload={"log_energy": level})]
        return ToolOutput(text=f"prosody [{ts:.1f}s, {te:.1f}s]: logE={level:.2f}", va_updates=va, evidence=ev)

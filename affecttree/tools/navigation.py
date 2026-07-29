"""时序导航工具：负责"看哪里"——片段级定位。

TemporalZoom / TemporalJump / Sliding 三个工具模拟人类观看视频时的
缩放、跳转与连续浏览行为。video 对象需实现 sample(ts, te, rate) 接口
返回帧序列（如基于 decord / OpenCV 的 VideoReader 封装）。
"""
from __future__ import annotations

from typing import Any

from .base import BaseTool, ToolOutput, register


@register
class TemporalZoom(BaseTool):
    """在当前片段内以更高时序分辨率抽帧。"""

    name = "temporal_zoom"
    description = "TemporalZoom(ts, te, rate)：放大 [ts, te] 区间细查"

    def run(self, video: Any, node: Any, ts: float, te: float, rate: float = 2.0, **_: Any) -> ToolOutput:
        frames = video.sample(ts, te, rate)
        return ToolOutput(text=f"zoom [{ts:.1f}s, {te:.1f}s] @ {rate}x fps → {len(frames)} frames", frames=frames)


@register
class TemporalJump(BaseTool):
    """跳出当前片段，抽取全局视频中其他区间，避免陷入局部。"""

    name = "temporal_jump"
    description = "TemporalJump(ts, te)：跳转到全局 [ts, te] 区间"

    def run(self, video: Any, node: Any, ts: float, te: float, rate: float = 1.0, **_: Any) -> ToolOutput:
        frames = video.sample(ts, te, rate)
        return ToolOutput(text=f"jump to [{ts:.1f}s, {te:.1f}s] → {len(frames)} frames", frames=frames)


@register
class Sliding(BaseTool):
    """以步长 dt 滑动时序窗口，渐进探索相邻片段。"""

    name = "sliding"
    description = "Sliding(ts, te, dt)：从 [ts, te] 起以步长 dt 滑窗浏览"

    def run(self, video: Any, node: Any, ts: float, te: float, dt: float, rate: float = 1.0, **_: Any) -> ToolOutput:
        frames = video.sample(ts, te + dt, rate)
        return ToolOutput(text=f"slide [{ts:.1f}s, {te:.1f}s] +{dt:.1f}s → {len(frames)} frames", frames=frames)

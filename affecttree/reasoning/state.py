"""推理节点状态定义。

每个树节点维护状态 s_t = <q, H, O, R, A>：查询、时序情感假设、
已观测情感证据（VA 轨迹）、文本推理轨迹与动作历史。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class VAEstimate:
    """一个片段的 Valence-Arousal 估计。"""

    t_start: float
    t_end: float
    valence: float
    arousal: float
    source: str = ""  # 产生该估计的工具名（face_au / prosody / fused ...）


@dataclass
class Evidence:
    """一条情感证据：模态来源 + 符号化结果。"""

    modality: str  # face / voice / body / scene / physio / fused
    tool: str
    t_start: float
    t_end: float
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class NodeState:
    """树中一个推理节点的完整状态。"""

    query: str
    hypothesis: str = ""
    va_trace: list[VAEstimate] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    reasoning_text: str = ""
    actions: list[dict[str, Any]] = field(default_factory=list)
    frames: list[Any] = field(default_factory=list)  # 回灌 VLM 上下文的截图
    depth: int = 0
    node_id: str = ""

    def transitions(self) -> list[tuple[float, float, float]]:
        """从 VA 轨迹中提取情感转折点候选 (t, dv, da)。"""
        points = []
        for prev, cur in zip(self.va_trace, self.va_trace[1:]):
            dv = cur.valence - prev.valence
            da = cur.arousal - prev.arousal
            points.append((cur.t_start, dv, da))
        return points

"""FaceAU：面部动作单元符号分析（主通道）。

传统路线：CLM 特征点拟合 + 几何 / Gabor 纹理特征 + FACS 编码规则，
输出 AU 强度、Ekman 六情绪标签、头部姿态与眨眼频率。
全部白盒，不依赖深度模型；输出符号供推理节点交叉验证。
"""
from __future__ import annotations

from typing import Any

from .base import BaseTool, ToolOutput, register
from ..reasoning.state import Evidence, VAEstimate

# FACS -> Ekman 规则表：AU 编号 -> 触发权重（初版，待标定）
EKMAN_RULES: dict[str, dict[int, float]] = {
    "happy": {6: 0.4, 12: 0.6},
    "sad": {1: 0.3, 4: 0.3, 15: 0.5},
    "angry": {4: 0.6, 7: 0.5, 23: 0.4},
    "fear": {1: 0.4, 2: 0.4, 4: 0.3, 20: 0.4},
    "surprise": {1: 0.5, 2: 0.5, 5: 0.4, 26: 0.5},
    "disgust": {9: 0.5, 10: 0.5, 17: 0.3},
}

# AU 对 VA 空间的先验贡献（初版，待数据标定）
AU_VALENCE_PRIOR: dict[int, float] = {6: 0.5, 12: 0.8, 15: -0.6, 4: -0.4, 1: -0.2}
AU_AROUSAL_PRIOR: dict[int, float] = {5: 0.5, 26: 0.6, 4: 0.4, 7: 0.3, 23: 0.3}


def map_to_ekman(au: dict[int, float]) -> str:
    """按规则表将 AU 强度映射到最可能的 Ekman 情绪标签。"""
    best, best_score = "neutral", 0.35  # 触发阈值
    for emo, rules in EKMAN_RULES.items():
        score = sum(min(au.get(k, 0.0), 1.0) * w for k, w in rules.items())
        if score > best_score:
            best, best_score = emo, score
    return best


def au_to_va(au: dict[int, float]) -> tuple[float, float]:
    """由 AU 强度估计 (valence, arousal) 先验。"""
    v = sum(au.get(k, 0.0) * w for k, w in AU_VALENCE_PRIOR.items())
    a = sum(au.get(k, 0.0) * w for k, w in AU_AROUSAL_PRIOR.items())
    return max(-1.0, min(1.0, v)), max(-1.0, min(1.0, a))


@register
class FaceAU(BaseTool):
    """对 FaceZoom 截图序列做 AU 分析，输出符号化面部情绪线索。"""

    name = "face_au"
    description = "FaceAU(track_id)：AU 强度、Ekman 标签、头部姿态、眨眼频率"

    def estimate_au(self, face_frames: list[Any]) -> dict[int, float]:
        """TODO: CLM 特征点 + 几何距离 / Gabor 纹理 -> AU 强度（0-1）。"""
        raise NotImplementedError

    def run(self, video: Any, node: Any, track_id: int, ts: float, te: float, **_: Any) -> ToolOutput:
        au = self.estimate_au(node.frames)  # 占位：实际应取该 track 的人脸序列
        emo = map_to_ekman(au)
        v, a = au_to_va(au)
        va = [VAEstimate(t_start=ts, t_end=te, valence=v, arousal=a, source=self.name)]
        ev = [Evidence(modality="face", tool=self.name, t_start=ts, t_end=te, payload={"au": au, "ekman": emo})]
        return ToolOutput(text=f"face_au(track {track_id}): {emo}, AU={au}", va_updates=va, evidence=ev)

"""主动视觉操作工具：检测、跟踪、裁剪，高清截图回灌 VLM 上下文。

设计动机：经典面部算法对输入分辨率与角度极为敏感。FaceZoom 把
"看哪里、看多清"变成可学习的 action——VLM 主动裁剪放大关键人脸，
截图以图像 token 回灌上下文（thinking with images），同时供下游
符号工具（FaceAU / PoseGesture）消费。
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseTool, ToolOutput, register
from ..reasoning.state import Evidence


def detect_faces(frames: list[np.ndarray], min_size: int = 48) -> list[list[tuple[int, int, int, int]]]:
    """逐帧人脸检测（Haar / LBP 级联）。

    TODO: 实现传统检脸；返回每帧的 bbox 列表 [(x, y, w, h)]。
    """
    raise NotImplementedError


def track_faces(per_frame_boxes: list[list[tuple[int, int, int, int]]], iou_thr: float = 0.3) -> list[dict]:
    """镜头内 IoU / KLT 跟踪关联，输出人脸轨迹。

    TODO: 实现关联；返回 [{"track_id", "boxes", "quality"}]。
    """
    raise NotImplementedError


@register
class FaceTrack(BaseTool):
    """镜头内人脸检测与跟踪，输出人脸轨迹与质量分。"""

    name = "face_track"
    description = "FaceTrack(ts, te)：返回 track_id、bbox 序列与质量分"

    def run(self, video: Any, node: Any, ts: float, te: float, **_: Any) -> ToolOutput:
        frames = video.sample(ts, te, rate=2.0)
        tracks = track_faces(detect_faces(frames))
        summary = "; ".join(f"track {t['track_id']} ({len(t['boxes'])} boxes, q={t['quality']:.2f})" for t in tracks)
        ev = [Evidence(modality="face", tool=self.name, t_start=ts, t_end=te, payload={"tracks": tracks})]
        return ToolOutput(text=f"face tracks in [{ts:.1f}s, {te:.1f}s]: {summary}", evidence=ev)


@register
class FaceZoom(BaseTool):
    """裁剪并放大目标人脸，高清截图回灌 VLM。"""

    name = "face_zoom"
    description = "FaceZoom(track_id, t)：人脸高清截图（图像 token 回灌 VLM）"

    def __init__(self, scale: float = 2.0, pad: float = 0.25) -> None:
        self.scale = scale
        self.pad = pad

    def crop(self, frame: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
        """按 bbox 外扩 pad 裁剪并放大。传统插值，无深度模型。"""
        x, y, w, h = box
        px, py = int(w * self.pad), int(h * self.pad)
        x0, y0 = max(0, x - px), max(0, y - py)
        x1, y1 = min(frame.shape[1], x + w + px), min(frame.shape[0], y + h + py)
        crop = frame[y0:y1, x0:x1]
        import cv2

        return cv2.resize(crop, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_CUBIC)

    def run(self, video: Any, node: Any, track_id: int, t: float, **_: Any) -> ToolOutput:
        # TODO: 依据 node.evidence 中 face_track 轨迹取 t 时刻 bbox，裁剪放大
        raise NotImplementedError


@register
class BodyCrop(BaseTool):
    """由人脸框几何外推上半身区域，供肢体分析。"""

    name = "body_crop"
    description = "BodyCrop(track_id, ts, te)：上半身区域序列"

    def run(self, video: Any, node: Any, track_id: int, ts: float, te: float, **_: Any) -> ToolOutput:
        # TODO: 人脸框向下外推 ~2.5x 头高得上半身框，HOG 校验人体存在
        raise NotImplementedError

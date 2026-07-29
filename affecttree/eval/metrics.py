"""EmoArc-Bench 评测指标实现。

- QA Accuracy：T1/T2/T4/T5 答案正确率
- F1@tIoU：转折点区间时序重合（0.3 / 0.5 阈值）
- CCC / RMSE / Pearson：预测 VA 轨迹 vs 标注均值（情感计算标准口径）
- Evidence F1：预测证据片段与标注证据区间的 IoU
- Paired Accuracy：幻觉对抗成对题（basic 与 false-premise 全对才得分）
"""
from __future__ import annotations

import numpy as np

from ..training.rewards import concordance_cc, transition_f1

__all__ = ["concordance_cc", "transition_f1", "qa_accuracy", "interval_iou", "evidence_f1", "paired_accuracy", "va_metrics"]


def qa_accuracy(preds: list[str], gts: list[str]) -> float:
    """答案正确率（外部判分器判对后的统计）。"""
    assert len(preds) == len(gts) and gts, "empty eval set"
    return float(np.mean([p.strip().lower() == g.strip().lower() for p, g in zip(preds, gts)]))


def interval_iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    """两个时间区间的 IoU。"""
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def evidence_f1(pred: list[tuple[float, float]], gt: list[tuple[float, float]], iou_thr: float = 0.3) -> float:
    """证据链 F1：预测证据片段与标注区间 IoU ≥ 阈值视为命中（贪心匹配）。"""
    if not pred and not gt:
        return 1.0
    if not pred or not gt:
        return 0.0
    used, hit = set(), 0
    for p in pred:
        for j, g in enumerate(gt):
            if j not in used and interval_iou(p, g) >= iou_thr:
                used.add(j)
                hit += 1
                break
    precision, recall = hit / len(pred), hit / len(gt)
    return 0.0 if precision + recall == 0 else float(2 * precision * recall / (precision + recall))


def paired_accuracy(basic_correct: list[bool], foil_correct: list[bool]) -> float:
    """幻觉对抗：成对题全对才得分。"""
    assert len(basic_correct) == len(foil_correct)
    return float(np.mean([b and f for b, f in zip(basic_correct, foil_correct)]))


def va_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    """VA 轨迹质量：CCC / RMSE / Pearson。"""
    pred, gt = np.asarray(pred, np.float64), np.asarray(gt, np.float64)
    rmse = float(np.sqrt(((pred - gt) ** 2).mean()))
    pearson = float(np.corrcoef(pred, gt)[0, 1]) if pred.std() > 0 and gt.std() > 0 else 0.0
    return {"ccc": concordance_cc(pred, gt), "rmse": rmse, "pearson": pearson}

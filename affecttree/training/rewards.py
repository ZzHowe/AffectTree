"""Emo-RPO 层级奖励：分支奖励 + 树级奖励 + 情感证据质量奖励。

R_branch：答对且路径越短奖励越高，答错且路径越长惩罚越重；
R_tree：多样性、成功率、深度效率三方面评估整棵推理树；
R_evidence：预测 VA 轨迹与标注的 CCC + 预测转折点的 F1@tIoU。
总优势 E(τ) = R_branch + k·R_tree + R_evidence，各项 batch 内标准化。
"""
from __future__ import annotations

import numpy as np


def branch_reward(n_nodes: int, correct: bool, terminated: bool) -> float:
    """分支轨迹奖励。"""
    if terminated:
        return 0.0
    return 1.0 / max(1, n_nodes) if correct else -float(n_nodes)


def tree_reward(success_rate: float, avg_dissimilarity: float, depth: int, max_depth: int) -> float:
    """树级奖励：成功率 + 分支多样性 + 深度效率（初版等权，待调）。"""
    depth_efficiency = 1.0 - depth / max(1, max_depth)
    return success_rate + avg_dissimilarity + depth_efficiency


def concordance_cc(x: np.ndarray, y: np.ndarray) -> float:
    """一致性相关系数 CCC：情感计算中 VA 轨迹评价的标准口径。"""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    cov = ((x - mx) * (y - my)).mean()
    denom = vx + vy + (mx - my) ** 2
    return float(2 * cov / denom) if denom > 1e-12 else 0.0


def evidence_reward(
    va_pred: np.ndarray,
    va_gt: np.ndarray,
    trans_pred: list[float],
    trans_gt: list[float],
    tolerance: float = 2.0,
    lambda_ccc: float = 1.0,
    lambda_tiou: float = 1.0,
) -> float:
    """情感证据质量奖励：lambda1·CCC(轨迹) + lambda2·F1(转折点, 容差)。"""
    ccc = concordance_cc(va_pred, va_gt)
    f1 = transition_f1(trans_pred, trans_gt, tolerance)
    return lambda_ccc * ccc + lambda_tiou * f1


def transition_f1(pred: list[float], gt: list[float], tolerance: float = 2.0) -> float:
    """转折点检测 F1：预测与标注边界在 ±tolerance 秒内视为命中。"""
    if not pred and not gt:
        return 1.0
    if not pred or not gt:
        return 0.0
    matched = set()
    hit = 0
    for p in pred:
        for j, g in enumerate(gt):
            if j not in matched and abs(p - g) <= tolerance:
                matched.add(j)
                hit += 1
                break
    precision = hit / len(pred)
    recall = hit / len(gt)
    return 0.0 if precision + recall == 0 else float(2 * precision * recall / (precision + recall))


def standardize(values: list[float]) -> np.ndarray:
    """batch 内标准化（均值 0、方差 1），用于各项奖励的组合。"""
    arr = np.asarray(values, dtype=np.float64)
    return (arr - arr.mean()) / (arr.std() + 1e-8)

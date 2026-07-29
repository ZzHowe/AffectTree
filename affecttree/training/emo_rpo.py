"""Stage 2：Emo-RPO——面向树状情感推理轨迹的相对策略优化。

流程：每个 video-query 采样一组推理树（group rollout）→ 计算层级奖励
（分支 + 树级 + 证据质量）→ batch 内标准化组合为优势 → 按 GRPO 风格
目标更新（重要性采样裁剪 + KL 正则）。

实际分布式训练复用 train/ 目录的 verl 框架；本文件给出奖励聚合与
优势计算的最小参考实现，接入点为 verl 的 reward function。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..reasoning.tree import ReasoningTree
from .rewards import branch_reward, evidence_reward, standardize, tree_reward


@dataclass
class RolloutSample:
    """一棵 rollout 推理树及其标注。"""

    tree: ReasoningTree
    correct: bool
    n_nodes: int
    terminated_early: bool
    va_pred: np.ndarray
    va_gt: np.ndarray
    trans_pred: list[float]
    trans_gt: list[float]


class EmoRPO:
    """Emo-RPO 优势计算器（供 verl reward function 调用）。"""

    def __init__(self, tree_coef: float = 0.5, lambda_ccc: float = 1.0, lambda_tiou: float = 1.0, tolerance: float = 2.0):
        self.tree_coef = tree_coef  # 总优势中 R_tree 的系数 k
        self.lambda_ccc = lambda_ccc
        self.lambda_tiou = lambda_tiou
        self.tolerance = tolerance

    def advantages(self, group: list[RolloutSample]) -> np.ndarray:
        """对一组（同一 video-query 的）推理树计算标准化组合优势。"""
        rb = [branch_reward(s.n_nodes, s.correct, s.terminated_early) for s in group]
        rt = [self._tree_level(s.tree) for s in group]
        re = [
            evidence_reward(s.va_pred, s.va_gt, s.trans_pred, s.trans_gt, self.tolerance, self.lambda_ccc, self.lambda_tiou)
            for s in group
        ]
        rb_hat, rt_hat, re_hat = standardize(rb), standardize(rt), standardize(re)
        return rb_hat + self.tree_coef * rt_hat + re_hat

    def _tree_level(self, tree: ReasoningTree) -> float:
        """树级奖励：成功率 + 分支多样性 + 深度效率。"""
        # TODO: 分支多样性（动作序列编辑距离 / VA 轨迹差异）按实际树结构计算
        return tree_reward(success_rate=0.0, avg_dissimilarity=0.0, depth=tree.max_depth, max_depth=tree.max_depth)

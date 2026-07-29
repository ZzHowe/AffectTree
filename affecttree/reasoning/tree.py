"""树状情感搜索引擎：广度优先假设展开、工具执行与验证剪枝。"""
from __future__ import annotations

import itertools
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Protocol

from .state import NodeState


class Policy(Protocol):
    """VLM 推理策略接口：给定节点状态，提出假设与动作。"""

    def propose(self, node: NodeState) -> list[dict[str, Any]]:
        """返回动作列表。动作类型：
        - {"type": "tool", "tool": <name>, "params": {...}, "hypothesis": str}
        - {"type": "answer", "content": str}
        """
        ...


@dataclass
class ReasoningTree:
    """一次推理 episode 产生的完整搜索树。"""

    query: str
    max_depth: int
    max_width: int
    nodes: list[NodeState] = field(default_factory=list)
    answer: str | None = None

    @property
    def branches(self) -> list[NodeState]:
        """叶节点集合（含答案节点与被剪枝节点）。"""
        return [n for n in self.nodes if n.depth == self.max_depth or n.node_id == "answer"]


class TreeSearchEngine:
    """BFS 树搜索：假设生成 -> 工具执行 -> 证据聚合 -> 剪枝/扩展。"""

    _ids = itertools.count()

    def __init__(self, policy: Policy, tools: dict, max_depth: int = 6, max_width: int = 3):
        self.policy = policy
        self.tools = tools  # name -> BaseTool
        self.max_depth = max_depth
        self.max_width = max_width

    def run(self, video: Any, query: str) -> ReasoningTree:
        tree = ReasoningTree(query=query, max_depth=self.max_depth, max_width=self.max_width)
        root = NodeState(query=query, node_id="root")
        queue: deque[NodeState] = deque([root])

        while queue and tree.answer is None:
            node = queue.popleft()
            if node.depth >= self.max_depth:
                continue
            for action in self.policy.propose(node)[: self.max_width]:
                if action.get("type") == "answer":
                    tree.answer = action.get("content", "")
                    break
                child = self._expand(video, node, action)
                tree.nodes.append(child)
                if not child.actions[-1].get("pruned", False):
                    queue.append(child)
        return tree

    def _expand(self, video: Any, node: NodeState, action: dict[str, Any]) -> NodeState:
        """执行单个工具动作并生成子节点（继承父节点 VA 轨迹并增量更新）。"""
        tool = self.tools[action["tool"]]
        obs = tool.run(video=video, node=node, **action.get("params", {}))
        action = {**action, "pruned": obs.pruned}
        return NodeState(
            query=node.query,
            hypothesis=action.get("hypothesis", ""),
            va_trace=[*node.va_trace, *obs.va_updates],
            evidence=[*node.evidence, *obs.evidence],
            reasoning_text=obs.text,
            actions=[*node.actions, action],
            frames=[*node.frames, *obs.frames],
            depth=node.depth + 1,
            node_id=f"n{next(self._ids)}",
        )

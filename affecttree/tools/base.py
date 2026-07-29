"""工具基类、输出协议与注册表。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..reasoning.state import Evidence, VAEstimate


@dataclass
class ToolOutput:
    """工具执行结果：文本说明、VA 增量、证据、回灌帧与剪枝标记。"""

    text: str = ""
    va_updates: list[VAEstimate] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    frames: list[Any] = field(default_factory=list)  # 需回灌 VLM 的图像（numpy）
    pruned: bool = False  # 当前假设是否被本工具结果否决


class BaseTool:
    """所有可调用工具的统一接口。"""

    name: str = ""
    description: str = ""

    def run(self, video: Any, node: Any, **params: Any) -> ToolOutput:
        raise NotImplementedError


_REGISTRY: dict[str, type[BaseTool]] = {}


def register(cls: type[BaseTool]) -> type[BaseTool]:
    _REGISTRY[cls.name] = cls
    return cls


def build_tools() -> dict[str, BaseTool]:
    """实例化全部已注册工具。"""
    return {name: cls() for name, cls in _REGISTRY.items()}

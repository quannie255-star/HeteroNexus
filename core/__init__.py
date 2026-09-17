# -*- coding: utf-8 -*-
"""
HeteroNexus Core —— 统一决策内核

本包是整个项目的内核层，负责把「算力调度」与「模型路由」两类问题
统一到同一套抽象下，使得上层策略可以跨场景复用。

设计原点（一句话）：
    把任务映射到候选执行资源上，在成本 / 延迟 / 质量 的多目标约束下求最优。
    算力侧与模型侧的差别，只在于候选资源的具体形态与「质量」是否为常量。
"""

from .resource import (
    TaskProfile,
    Estimate,
    Executor,
    ComputeExecutor,
    ModelExecutor,
    cap_logit,
    quality_at_difficulty,
)
from .objective import Objective, scalarize, dominates, pareto_front

__all__ = [
    'TaskProfile', 'Estimate', 'Executor', 'ComputeExecutor', 'ModelExecutor',
    'cap_logit', 'quality_at_difficulty',
    'Objective', 'scalarize', 'dominates', 'pareto_front',
]

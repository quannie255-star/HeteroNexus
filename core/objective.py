# -*- coding: utf-8 -*-
"""
多目标统一建模与标量化

算力侧的目标曾是 5 维：时间 / 利用率 / 能耗 / 公平 / 碎片。
跨到 token 侧后，目标空间变成 3 维：成本 / 延迟 / 质量。

两者的关系是：
    - 成本、延迟    → 两侧同义，可直接对齐
    - 质量          → **token 侧独有**。这是本项目的核心增量
    - 利用率/能耗等 → 算力侧独有，token 侧不关心（对 API 调用而言不可见）

因此统一的做法是：把「质量」作为一等目标引入，并区分它在两侧的语义——
算力侧恒为 1（可退化），token 侧随难度变化（不可退化、损失不可逆）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

from .resource import Estimate

# 成本对数化时的下限保护：避免 log(0) 与负成本
_FLOOR = 1e-9


@dataclass(frozen=True)
class Objective:
    """用户对一次决策的偏好。

    w_*  为权重，用于把多目标压成单目标；
    min_quality 为**硬约束** —— 质量低于该值的候选资源直接淘汰。
    硬约束的存在是 token 侧区别于算力侧的关键：
    调度选错节点只是慢一点（软惩罚），模型选错就是答错（不可接受）。
    """

    name: str = 'balanced'
    w_cost: float = 1.0
    w_latency: float = 0.0
    w_quality: float = 1.0
    min_quality: float = 0.0        # 质量下限（0~1）
    max_latency: float = math.inf   # 延迟上限（秒）

    def __post_init__(self):
        s = self.w_cost + self.w_latency + self.w_quality
        if s <= 0:
            raise ValueError('目标权重不能全为 0')

    def feasible(self, est: Estimate) -> bool:
        return est.feasible and est.quality >= self.min_quality - 1e-12 \
            and est.latency <= self.max_latency + 1e-12


@dataclass(frozen=True)
class Reference:
    """标量化用的参考尺度。

    成本采用 **对数 min-max 归一化**，原因有两层，都是被实际结果逼出来的：

    1. 候选成本呈重尾分布（最便宜 ¥1/M、最贵 ¥105/M，跨两个数量级）。
       若用线性 min-max，除头部外的所有候选都会挤在 0~0.03，
       而质量差异普遍在 0.1~0.2 —— 成本项永远压不过质量项，怎么调权重都不省钱。

    2. 决策者对成本的感知本来就是对数的（韦伯—费希纳定律）：
       "贵 3 倍"是显著差别，"贵 5%"通常不是。对数尺度与这种直觉一致。

    参考区间内的语义因此清晰：最便宜的候选成本项为 0，每贵一个数量级，
    成本项按比例上升；λ=1 表示「愿意用一个数量级的成本换取 1 个单位的质量」。
    """

    cost_min: float = 0.0
    cost_max: float = 1.0
    lat_min: float = 0.0
    lat_max: float = 1.0

    @classmethod
    def from_estimates(cls, estimates: Sequence[Estimate]) -> 'Reference':
        costs = [e.cost for e in estimates if e.feasible] or [0.0]
        lat = [e.latency for e in estimates if e.feasible] or [0.0]
        return cls(cost_min=max(min(costs), _FLOOR), cost_max=max(max(costs), _FLOOR),
                   lat_min=min(lat), lat_max=max(lat))


def _log_norm(value: float, lo: float, hi: float) -> float:
    """对数尺度下的 min-max 归一化，返回 [0,1]。"""
    v = max(value, _FLOOR)
    lo = max(lo, _FLOOR)
    hi = max(hi, lo * (1.0 + 1e-12))
    if hi <= lo:
        return 0.0
    return (math.log(v) - math.log(lo)) / (math.log(hi) - math.log(lo))


def scalarize(est: Estimate, obj: Objective, ref: Reference) -> float:
    """多目标 → 单目标标量（越大越好）。

    质量按绝对比例计入收益；成本取对数归一化、延迟取线性归一化后计入代价。
    """
    if not est.feasible:
        return -math.inf
    norm_cost = _log_norm(est.cost, ref.cost_min, ref.cost_max)
    lspan = max(ref.lat_max - ref.lat_min, 1e-12)
    norm_lat = (est.latency - ref.lat_min) / lspan
    return (obj.w_quality * est.quality
            - obj.w_cost * norm_cost
            - obj.w_latency * norm_lat)


def dominates(a: Tuple[float, float, float], b: Tuple[float, float, float],
              tol: float = 1e-9) -> bool:
    """帕累托支配判定。输入为 (cost, latency, quality)。

    a 支配 b 当且仅当：a 的每一项代价都不劣于 b，质量不劣于 b，且至少一项严格更优。
    """
    a_cost, a_lat, a_q = a
    b_cost, b_lat, b_q = b
    not_worse = (a_cost <= b_cost + tol) and (a_lat <= b_lat + tol) and (a_q >= b_q - tol)
    better = (a_cost < b_cost - tol) or (a_lat < b_lat - tol) or (a_q > b_q + tol)
    return not_worse and better


def pareto_front(points: Iterable[Tuple[str, float, float, float]]
                 ) -> List[Tuple[str, float, float, float]]:
    """求成本-延迟-质量三维下的帕累托前沿。

    points: (label, cost, latency, quality)
    返回非支配点列表，保持输入顺序。
    """
    pts = list(points)
    front = []
    for i, (li, ci, lai, qi) in enumerate(pts):
        dominated = False
        for j, (lj, cj, laj, qj) in enumerate(pts):
            if i == j:
                continue
            if dominates((cj, laj, qj), (ci, lai, qi)):
                dominated = True
                break
        if not dominated:
            front.append((li, ci, lai, qi))
    return front

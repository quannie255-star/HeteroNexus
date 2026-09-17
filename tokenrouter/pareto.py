# -*- coding: utf-8 -*-
"""
成本 — 质量帕累托前沿与敏感性分析

两条分析线：

1. **质量下限扫描**：min_quality 从低到高扫一遍，得到一串 (成本, 正确率) 组合。
   它们构成一条帕累托前沿 —— 不存在既更便宜又更准的方案。
   这条曲线的意义是：**「省多少」和「接受多差」必须由用户自己定**，
   我们的角色是把这条曲线画出来并给出推荐点，而不是替用户决定。

2. **GPU 利用率敏感性**：自建模型的单位 token 成本反比于负载率。
   扫一遍利用率就知道「自建到底划不划算」的临界点在哪，
   而这个临界点恰恰取决于算力侧调度能把 GPU 喂到多满。

这与 Q1 算力侧「多目标奖励权重敏感性 + 帕累托前沿」是**同一套方法论**，
只是把目标从「延迟 / 利用率 / 公平」换成了「成本 / 质量」。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from core.resource import ModelExecutor, TaskProfile
from .policies import DifficultyRouterPolicy
from .engine import PolicyResult, run_policy


@dataclass
class SweepPoint:
    """前沿上的一个可交付配置点。"""
    cost_weight: float                       # 调节点：用户对成本的在意程度
    min_quality: float = 0.0                 # 附加硬约束（0 表示不启用）
    accuracy: float = 0.0
    total_cost: float = 0.0
    cost_saving: float = 0.0
    quality_retention: float = 0.0
    recommended_table: Dict[str, str] = field(default_factory=dict)


def sweep_cost_weight(tasks: Sequence[TaskProfile], executors: Sequence[ModelExecutor],
                      grid: Sequence[float] = None, master_seed: int = 42,
                      min_quality: float = 0.0,
                      baseline: PolicyResult = None) -> List[SweepPoint]:
    """扫描成本权重 λ，产出成本—质量帕累托前沿上的点集。

    为什么扫 λ 而不是扫「质量下限」：
        λ 是**连续**的调节旋钮，λ 增大 → 策略单调地更省钱、质量单调地略降，
        扫出来的曲线天然单调，符合帕累托前沿的定义。

        而二值的质量下限会产生非单调：约束一旦跨过某个模型的临界点，
        策略会在候选集之间跳变，甚至出现「要求更严反而更便宜」的反常识结果。
        （这一坑在初版里真实踩到，保留在这里作为设计记录。）

    min_quality 作为**硬安全网**保留：某些环节错了后果严重时必须设。
    但它应当是业务上确有必要才开的开关，而不是调省钱幅度的旋钮。
    """
    if grid is None:
        # 加密集中在 0.05~0.5：这才是业务上有意义的区间（质量保有 ≥ 95%）。
        # λ > 1 时质量快速崩塌到 70% 附近，基本没有实用价值，仅作前沿的延伸。
        grid = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 3.0]
    avg_in = max(1, int(sum(t.input_tokens for t in tasks) / len(tasks)))
    avg_out = max(1, int(sum(t.output_tokens for t in tasks) / len(tasks)))

    pts: List[SweepPoint] = []
    for lam in grid:
        pol = DifficultyRouterPolicy(executors, min_quality=min_quality,
                                     cost_weight=lam, avg_in=avg_in, avg_out=avg_out)
        r = run_policy(pol, tasks, executors, master_seed)
        table = {f'{d}/{b}': m for (d, b), m in sorted(pol.table.items())}
        pts.append(SweepPoint(
            cost_weight=lam,
            min_quality=min_quality,
            accuracy=r.accuracy,
            total_cost=r.total_cost,
            recommended_table=table,
        ))

    # 以「全部走最强模型」的用户现状为参照补相对指标
    base_cost = baseline.total_cost if baseline is not None else (pts[0].total_cost if pts else 0.0)
    base_acc = baseline.accuracy if baseline is not None else (pts[0].accuracy if pts else 0.0)
    for p in pts:
        p.cost_saving = round(1 - p.total_cost / base_cost, 4) if base_cost else 0.0
        p.quality_retention = round(p.accuracy / base_acc, 4) if base_acc else 0.0
    return pts


def knee_point(points: Sequence[SweepPoint]) -> SweepPoint:
    """前沿上的「膝点」：归一化后距离理想点（最低成本、最高正确率）最近的点。

    这是给用户推荐的默认配置 —— 边际收益开始明显递减的那个拐点。
    """
    if not points:
        raise ValueError('空的点集')
    costs = [p.total_cost for p in points]
    accs = [p.accuracy for p in points]
    cmin, cmax = min(costs), max(costs)
    amin, amax = min(accs), max(accs)
    cspan = (cmax - cmin) or 1.0
    aspan = (amax - amin) or 1.0
    best, best_d = None, None
    for p in points:
        # 理想点：成本最小 + 正确率最大
        d = math.hypot((p.total_cost - cmin) / cspan, (amax - p.accuracy) / aspan)
        if best_d is None or d < best_d:
            best, best_d = p, d
    return best


def utilization_sensitivity(raw_catalog: Dict, gpu_specs: Dict,
                           grid: Sequence[float] = None) -> List[Dict]:
    """GPU 利用率对自建经济性的影响 —— 直接基于 catalog 原始数据重算，
    避免依赖已加载对象里缓存的成本，保证扫描结果自洽。

    返回字段：
        utilization                 负载率
        model                       代表性自建模型（能力最强的那个）
        self_cost_per_mtok          该利用率下自建的 output token 成本（元/百万）
        ref_model                   能力最接近的商用 API，作为参照
        ref_cost_per_mtok           参照模型的 output token 成本
        self_hosted_wins            自建在该利用率下是否仍然更便宜
    """
    if grid is None:
        grid = [1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

    entries = raw_catalog.get('models', [])
    sh = [e for e in entries if e.get('kind') == 'self_hosted_model']
    api = [e for e in entries if e.get('kind') == 'api_model']
    if not sh or not api:
        return []

    def score(e):
        c = e.get('capability', {})
        return sum(c.values()) / max(len(c), 1)

    rep = max(sh, key=score)
    gpu_price = float(gpu_specs[rep['gpu']]['hourly_price'])
    gpu_count = int(rep.get('gpu_count', 1))
    tps = float(rep['throughput_tps'])

    # 能力最接近的高端 API 作为参照
    ref = min(api, key=lambda e: abs(score(e) - score(rep)))

    out: List[Dict] = []
    for u in grid:
        per_token = gpu_price * gpu_count / 3600.0 / (tps * max(u, 1e-3))
        per_mtok = per_token * 1_000_000.0
        out.append({
            'utilization': u,
            'model': rep['name'],
            'self_cost_per_mtok': round(per_mtok, 4),
            'ref_model': ref['name'],
            'ref_cost_per_mtok': round(float(ref['price_out']), 4),
            'self_hosted_wins': per_mtok < float(ref['price_out']),
        })
    return out

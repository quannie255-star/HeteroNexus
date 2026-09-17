# -*- coding: utf-8 -*-
"""
Token 侧路由离线仿真引擎

在**不消耗任何真实 token** 的前提下，评估各类路由策略的成本与质量表现。

为什么可以离线做：
    每个 (任务, 模型) 的调用结果由「确定性正确性」决定 —— quality 是该次答对的
    概率，用确定性哈希实现 Bernoulli 采样。整个评估过程相当于在一个固定业务流上
    重放不同花钱方式，不依赖任何真实 API 调用，也不需要真实用户数据。

    这也是本项目能被验证的原因：没有 GPU、没有 API 预算、没有真实流量，
    依然能给出可复现、可进行敏感性分析的定量结论。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from core.resource import ModelExecutor, TaskProfile, quality_at_difficulty
from .policies import Policy, build_policies
from .workload import DIFFICULTY_BANDS


@dataclass
class PolicyResult:
    """单个策略在一批任务上的聚合表现。"""
    policy: str
    description: str = ''
    description_zh: str = ''
    n_tasks: int = 0
    total_cost: float = 0.0            # 元
    correct_count: int = 0
    accuracy: float = 0.0
    avg_latency: float = 0.0           # 秒
    p95_latency: float = 0.0
    total_escalations: int = 0
    escalation_rate: float = 0.0       # 发生升级的任务占比
    calls_per_task: float = 1.0
    model_usage: Dict[str, int] = field(default_factory=dict)
    band_usage: Dict[str, Dict[str, int]] = field(default_factory=dict)  # band -> model -> count

    # 相对用户现状基线（strongest）的派生指标，由 attach_relative_metrics 填充
    cost_saving: float = 0.0            # 省了多少钱的比例
    quality_retention: float = 0.0      # 保住了多少正确率
    latency_change: float = 0.0         # 延迟变化比例
    efficiency_gain: float = 0.0        # 「每个正确答案的成本」改善比例

    @property
    def cost_per_correct_answer(self) -> float:
        """每拿到一个正确答案花的钱 —— 同时刻画成本与质量的关键效率指标。"""
        return self.total_cost / self.correct_count if self.correct_count else float('inf')


def _band_of(difficulty: float) -> str:
    for band, (lo, hi) in DIFFICULTY_BANDS.items():
        if lo <= difficulty <= hi:
            return band
    return 'medium'


def run_policy(policy: Policy, tasks: Sequence[TaskProfile],
               executors: Sequence[ModelExecutor], master_seed: int = 42) -> PolicyResult:
    """在给定负载上跑通一个策略，返回聚合结果。"""
    total_cost = 0.0
    correct = 0
    latencies: List[float] = []
    escalations = 0
    calls = 0
    usage: Dict[str, int] = {}
    band_usage: Dict[str, Dict[str, int]] = {}

    for task in tasks:
        out = policy.route(task, executors, master_seed)
        total_cost += out.cost
        latencies.append(out.latency)
        calls += len(out.attempts)
        if out.final_correct:
            correct += 1
        if out.escalations:
            escalations += 1
        if out.attempts:
            final_id = out.attempts[-1].executor_id
            usage[final_id] = usage.get(final_id, 0) + 1
            band = _band_of(task.difficulty)
            band_usage.setdefault(band, {})
            band_usage[band][final_id] = band_usage[band].get(final_id, 0) + 1

    n = len(tasks) or 1
    lat_sorted = sorted(latencies)
    p95 = lat_sorted[int(len(lat_sorted) * 0.95)] if lat_sorted else 0.0

    return PolicyResult(
        policy=policy.name,
        description=policy.description,
        n_tasks=len(tasks),
        total_cost=round(total_cost, 6),
        correct_count=correct,
        accuracy=round(correct / n, 4),
        avg_latency=round(sum(latencies) / n, 3),
        p95_latency=round(p95, 3),
        total_escalations=escalations,
        escalation_rate=round(escalations / n, 4),
        calls_per_task=round(calls / n, 3),
        model_usage=dict(sorted(usage.items(), key=lambda kv: -kv[1])),
        band_usage={k: dict(sorted(v.items(), key=lambda kv: -kv[1]))
                    for k, v in sorted(band_usage.items())},
    )


def attach_relative_metrics(results: Dict[str, PolicyResult], baseline: str = 'strongest') -> None:
    """给每个结果补上相对基线（用户现状）的相对指标。"""
    base = results.get(baseline)
    if base is None:
        return
    for r in results.values():
        r.cost_saving = round(1 - r.total_cost / base.total_cost, 4) if base.total_cost else 0.0
        r.quality_retention = round(r.accuracy / base.accuracy, 4) if base.accuracy else 0.0
        r.latency_change = round(r.avg_latency / base.avg_latency - 1, 4) if base.avg_latency else 0.0
        r.efficiency_gain = (round(1 - r.cost_per_correct_answer / base.cost_per_correct_answer, 4)
                             if base.correct_count and r.correct_count else 0.0)


def run_experiment(tasks: Sequence[TaskProfile], executors: Sequence[ModelExecutor],
                   min_quality: float = 0.0, master_seed: int = 42,
                   avg_in: int = 1000, avg_out: int = 500,
                   cost_weight: float = 1.0) -> Dict[str, PolicyResult]:
    """跑完整组策略对比。"""
    policies = build_policies(executors, min_quality=min_quality, cost_weight=cost_weight,
                              avg_in=avg_in, avg_out=avg_out)
    results: Dict[str, PolicyResult] = {}
    for p in policies:
        results[p.name] = run_policy(p, tasks, executors, master_seed)
    attach_relative_metrics(results)
    return results


def run_multi_seed(tasks_list: List[Sequence[TaskProfile]],
                   executors: Sequence[ModelExecutor],
                   min_quality: float = 0.0,
                   master_seed: int = 42,
                   cost_weight: float = 1.0) -> Dict[str, PolicyResult]:
    """跨多份负载取平均 —— 与 Q1 算力侧实验的取平均口径保持一致。"""
    acc: Dict[str, List[PolicyResult]] = {}
    first = tasks_list[0]
    per_task_in = max(1, int(sum(t.input_tokens for t in first) / len(first)))
    per_task_out = max(1, int(sum(t.output_tokens for t in first) / len(first)))
    for tasks in tasks_list:
        res = run_experiment(tasks, executors, min_quality, master_seed,
                             per_task_in, per_task_out, cost_weight)
        for name, r in res.items():
            acc.setdefault(name, []).append(r)

    merged: Dict[str, PolicyResult] = {}
    for name, rs in acc.items():
        n = len(rs)
        m = PolicyResult(
            policy=name,
            description=rs[0].description,
            n_tasks=sum(x.n_tasks for x in rs),
            total_cost=round(sum(x.total_cost for x in rs) / n, 6),
            correct_count=sum(x.correct_count for x in rs),
            accuracy=round(sum(x.accuracy for x in rs) / n, 4),
            avg_latency=round(sum(x.avg_latency for x in rs) / n, 3),
            p95_latency=round(sum(x.p95_latency for x in rs) / n, 3),
            total_escalations=sum(x.total_escalations for x in rs),
            escalation_rate=round(sum(x.escalation_rate for x in rs) / n, 4),
            calls_per_task=round(sum(x.calls_per_task for x in rs) / n, 3),
        )
        usage: Dict[str, int] = {}
        for x in rs:
            for k, v in x.model_usage.items():
                usage[k] = usage.get(k, 0) + v
        m.model_usage = dict(sorted(usage.items(), key=lambda kv: -kv[1]))
        merged[name] = m

    # 总成本来自多份负载求和后再比较，因此相对指标基于合并后的绝对值重算
    base = merged.get('strongest')
    if base and base.total_cost:
        for m in merged.values():
            m.cost_saving = round(1 - m.total_cost / base.total_cost, 4)
            m.quality_retention = round(m.accuracy / base.accuracy, 4) if base.accuracy else 0.0
            m.latency_change = round(m.avg_latency / base.avg_latency - 1, 4) if base.avg_latency else 0.0
            if base.correct_count and m.correct_count:
                m.efficiency_gain = round(1 - m.cost_per_correct_answer / base.cost_per_correct_answer, 4)
    return merged

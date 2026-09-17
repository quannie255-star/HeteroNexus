# -*- coding: utf-8 -*-
"""
把「方案」从页面上的数字变成一个**可被执行的文件**。

why this file exists:
    OfflineTablePolicy 生成的推荐表目前只活在内存和 JSON 里。用户拿到的是
    「省 50.7%」这个结论，而不是能直接用的东西。这中间差一步 —— 而这一步
    恰恰是本项目的交付物本体：一张可以被网关读取、被人工执行、被版本管理的
    路由规则文件。

设计上刻意做了两件事：

1. **导出的文件必须能被装回来执行**（load_plan + apply_plan），并且执行结果
   必须与生成它的策略**逐位一致**。否则「方案」只是漂亮的文本 —— 谁都可以写
   一个 YAML，但没人能证明照它执行真的能省 50.7%。这个闭环由
   tests/test_tokenrouter.py::TestPlanFile 守住。

2. **不做序列化魔法**。YAML 手写生成（不引第三方模板），解析只依赖 PyYAML。
   文件里带 meta（场景/λ/负载率/catalog 版本）与期望收益，让拿到文件的人
   不需要上下文就能判断它是否适用于自己的业务。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import yaml

from .policies import (DifficultyRouterPolicy, Policy, RouteOutcome, _by_index,
                       _make_attempt, global_strongest)
from core.resource import ModelExecutor, TaskProfile
from .workload import DIFFICULTY_BANDS

BANDS_ORDER = ['easy', 'medium', 'hard']
DOMAINS_ORDER = ['general', 'knowledge', 'coding', 'reasoning']


def band_of(difficulty: float) -> str:
    for b in BANDS_ORDER:
        lo, hi = DIFFICULTY_BANDS[b]
        if lo <= difficulty <= hi:
            return b
    return 'medium'


def build_plan_dict(policy: DifficultyRouterPolicy, meta: Dict[str, Any]) -> Dict[str, Any]:
    """把策略的推荐表 + 关键元信息组装成可序列化的字典。"""
    table = {f'{d}/{b}': m for (d, b), m in sorted(policy.table.items())}
    fallback = table.get('general/medium')
    rules = []
    for d in DOMAINS_ORDER:
        for b in BANDS_ORDER:
            key = f'{d}/{b}'
            if key not in table:
                continue
            diag = next((x for x in policy.diagnostics
                         if x['domain'] == d and x['band'] == b), None)
            rules.append({
                'when': {'domain': d, 'difficulty_band': b,
                         'difficulty_range': [DIFFICULTY_BANDS[b][0], DIFFICULTY_BANDS[b][1]]},
                'use': table[key],
                'expected_quality': diag['expected_quality'] if diag else None,
                'quality_target_met': diag['quality_target_met'] if diag else True,
            })
    return {
        'meta': dict(meta),
        'fallback': fallback,
        'rules': rules,
    }


def render_plan_yaml(plan: Dict[str, Any]) -> str:
    """渲染为 YAML 文本。手写头部注释，让文件脱离上下文也读得懂。"""
    m = plan['meta']
    head = [
        '# HeteroNexus 模型路由方案',
        '#',
        '# 用法：按请求的「能力域 + 难度档」查 rules，use 即推荐模型；',
        '#      难度无法判定或没有命中任何规则时，用 fallback。',
        '#',
        f"# 生成条件：场景={m.get('scenario')}  成本权重 λ={m.get('cost_weight')}  "
        f"GPU负载率={m.get('gpu_utilization')}",
        f"# 预期收益：省钱 {m.get('expected_cost_saving')}  正确率 {m.get('expected_accuracy')}  "
        f"（基线「全用最强模型」正确率 {m.get('baseline_accuracy')}）",
        '#',
        '# ⚠ 省钱比例必须与质量代价同时引用。只讲省了多少而不讲掉了多少正确率，等同于误导。',
        '',
    ]
    body = yaml.safe_dump(plan, allow_unicode=True, sort_keys=False, default_flow_style=False)
    return '\n'.join(head) + body


def load_plan(text: str) -> Dict[str, Any]:
    """装回一个方案文件。"""
    plan = yaml.safe_load(text)
    if not isinstance(plan, dict) or 'rules' not in plan:
        raise ValueError('不是合法的 HeteroNexus 方案文件：缺少 rules')
    return plan


def apply_plan(plan: Dict[str, Any], domain: str, difficulty: float) -> Optional[str]:
    """照方案执行：给定能力域与难度，返回应使用的模型 id。"""
    band = band_of(difficulty)
    for r in plan.get('rules', []):
        w = r.get('when', {})
        if w.get('domain') == domain and w.get('difficulty_band') == band:
            return r.get('use')
    return plan.get('fallback')


class PlanFilePolicy(Policy):
    """照一个**已导出的方案文件**执行路由。

    存在的唯一目的是闭环验证：把导出的 YAML 装回来跑一遍，结果必须与生成它的
    OfflineTablePolicy 完全一致。若这里对不上，说明导出或解析有一处撒了谎 ——
    那种「看起来能落地」的方案比没有方案更糟。
    """

    name = 'plan_file'
    description = '按导出的方案文件（YAML）查表路由（验证导出—执行闭环）'

    def __init__(self, plan: Dict[str, Any]):
        self.plan = plan
        self.lookup = {(r['when']['domain'], r['when']['difficulty_band']): r['use']
                       for r in plan.get('rules', [])}

    def route(self, task: TaskProfile, executors: Sequence[ModelExecutor],
              master_seed: int) -> RouteOutcome:
        pool = _by_index(executors)
        eid = self.lookup.get((task.domain, band_of(task.difficulty))) \
            or self.plan.get('fallback')
        ex = pool.get(eid) or global_strongest(list(pool.values()))
        att = _make_attempt(task, ex, master_seed)
        att.accepted = True
        return RouteOutcome(task.task_id, [att])

    def plan_for(self, difficulty: float, domain: str) -> str:
        return self.lookup.get((domain, band_of(difficulty))) or self.plan.get('fallback')


def plan_rules_as_table(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    """把方案文件摊平成表格，便于测试与展示。"""
    return [{'domain': r['when']['domain'], 'band': r['when']['difficulty_band'],
             'model': r['use'], 'expected_quality': r.get('expected_quality')}
            for r in plan.get('rules', [])]

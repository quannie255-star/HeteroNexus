# -*- coding: utf-8 -*-
"""
任务流生成

生成一批带「难度」与「能力域」标注的任务画像，用于在离线环境下评估路由策略。

难度是关键变量：真实业务里大量请求（改写、摘要、格式转换、简单问答）根本用不上
顶尖模型，但用户不清楚这一点，于是全部走最强模型付费。可省空间就来自这里的错配。

难度分布是可以按场景替换的参数 —— 企业用户可以描述自己的业务构成，
我们就用对应的分布跑评估，这也是「输入现状 → 输出方案」的输入端。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from core.resource import TaskProfile

DEFAULT_SEED = 42

DOMAINS = ('general', 'knowledge', 'coding', 'reasoning')

# 三档难度区间。0.5 是 benchmark 基准难度（公开分数在这一档上可还原），
# 低于 0.5 属于「强模型能力过剩」区间。
DIFFICULTY_BANDS: Dict[str, Tuple[float, float]] = {
    'easy':   (0.10, 0.35),
    'medium': (0.35, 0.65),
    'hard':   (0.65, 0.95),
}


@dataclass
class WorkloadSpec:
    """一组可描述的业务负载画像。"""
    name: str = 'mixed'
    n_tasks: int = 2000
    seed: int = DEFAULT_SEED
    domain_weights: Dict[str, float] = field(
        default_factory=lambda: {'general': 0.50, 'knowledge': 0.20, 'coding': 0.20, 'reasoning': 0.10})
    difficulty_weights: Dict[str, float] = field(
        default_factory=lambda: {'easy': 0.55, 'medium': 0.30, 'hard': 0.15})
    input_tokens: Tuple[int, int] = (400, 3000)
    output_tokens: Tuple[int, int] = (150, 1200)


# 预置场景：覆盖「不是所有任务都需要顶尖模型」这一主张的几类典型业务形态
PRESETS: Dict[str, WorkloadSpec] = {
    # 日常助手型：大量轻量问答与文本处理，省 token 空间最大
    'daily_assistant': WorkloadSpec(
        name='daily_assistant', n_tasks=2000, seed=DEFAULT_SEED,
        domain_weights={'general': 0.70, 'knowledge': 0.20, 'coding': 0.05, 'reasoning': 0.05},
        difficulty_weights={'easy': 0.70, 'medium': 0.22, 'hard': 0.08},
        input_tokens=(300, 1500), output_tokens=(120, 600)),
    # 研发 Copilot 型：代码占比高，难度重心上移
    'dev_copilot': WorkloadSpec(
        name='dev_copilot', n_tasks=2000, seed=DEFAULT_SEED,
        domain_weights={'general': 0.20, 'knowledge': 0.15, 'coding': 0.55, 'reasoning': 0.10},
        difficulty_weights={'easy': 0.35, 'medium': 0.40, 'hard': 0.25},
        input_tokens=(800, 4000), output_tokens=(200, 1600)),
    # 分析推理型：高难度占比高，省钱空间最小（用于展示边界）
    'analytics': WorkloadSpec(
        name='analytics', n_tasks=2000, seed=DEFAULT_SEED,
        domain_weights={'general': 0.15, 'knowledge': 0.25, 'coding': 0.15, 'reasoning': 0.45},
        difficulty_weights={'easy': 0.20, 'medium': 0.35, 'hard': 0.45},
        input_tokens=(1000, 5000), output_tokens=(400, 2000)),
    'mixed': WorkloadSpec(name='mixed'),
}


def get_preset(name: str) -> WorkloadSpec:
    if name not in PRESETS:
        raise KeyError(f'未知场景预设: {name}，可选 {list(PRESETS)}')
    return PRESETS[name]


def generate_workload(spec: WorkloadSpec) -> List[TaskProfile]:
    """生成可完全复现的任务流（自持 Random 实例，不受全局随机状态影响）。"""
    rng = random.Random(spec.seed)
    domains = list(spec.domain_weights.keys())
    dw = [spec.domain_weights[d] for d in domains]
    bands = list(spec.difficulty_weights.keys())
    bw = [spec.difficulty_weights[b] for b in bands]

    tasks: List[TaskProfile] = []
    for tid in range(spec.n_tasks):
        domain = rng.choices(domains, weights=dw, k=1)[0]
        band = rng.choices(bands, weights=bw, k=1)[0]
        lo, hi = DIFFICULTY_BANDS[band]
        difficulty = rng.uniform(lo, hi)
        tasks.append(TaskProfile(
            task_id=tid,
            difficulty=round(difficulty, 4),
            domain=domain,
            input_tokens=rng.randint(*spec.input_tokens),
            output_tokens=rng.randint(*spec.output_tokens),
            task_type=f'{domain}/{band}',
        ))
    return tasks


def summarize_workload(tasks: List[TaskProfile]) -> Dict:
    """输出负载画像摘要，用于方案书里说明「评估所基于的业务构成」。"""
    by_domain: Dict[str, int] = {}
    by_band: Dict[str, int] = {}
    for t in tasks:
        by_domain[t.domain] = by_domain.get(t.domain, 0) + 1
        band = next((b for b, (lo, hi) in DIFFICULTY_BANDS.items() if lo <= t.difficulty <= hi), 'medium')
        by_band[band] = by_band.get(band, 0) + 1
    n = len(tasks) or 1
    return {
        'n_tasks': len(tasks),
        'total_input_tokens': sum(t.input_tokens for t in tasks),
        'total_output_tokens': sum(t.output_tokens for t in tasks),
        'avg_difficulty': round(sum(t.difficulty for t in tasks) / n, 4),
        'domain_ratio': {k: round(v / n, 4) for k, v in sorted(by_domain.items())},
        'difficulty_ratio': {k: round(v / n, 4) for k, v in sorted(by_band.items())},
    }

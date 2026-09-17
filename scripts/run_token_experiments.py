# -*- coding: utf-8 -*-
"""
Token 侧场景实验主入口

产出：
    data/token_experiments.json   全量结果（策略对比 / 帕累托前沿 / 敏感性分析）

用法：
    python scripts/run_token_experiments.py
    python scripts/run_token_experiments.py --scenario dev_copilot
    python scripts/run_token_experiments.py --gpu-utilization 0.5
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tokenrouter import (  # noqa: E402
    load_catalog, get_preset, PRESETS, generate_workload, summarize_workload,
    run_experiment, run_multi_seed, sweep_cost_weight, knee_point,
    utilization_sensitivity, api_models, self_hosted_models,
)
from tokenrouter.policies import DifficultyRouterPolicy  # noqa: E402

OUT_DIR = ROOT / 'data'
OUT_FILE = OUT_DIR / 'token_experiments.json'

SEEDS = [1, 7, 42, 999, 2026]   # 与算力侧 Q1 实验同一套种子口径


def _avg_tokens(tasks):
    avg_in = max(1, int(sum(t.input_tokens for t in tasks) / len(tasks)))
    avg_out = max(1, int(sum(t.output_tokens for t in tasks) / len(tasks)))
    return avg_in, avg_out


def build_router_plan(cat, tasks, min_quality: float, cost_weight: float) -> Dict:
    """生成可直接交付给用户的「方案」：一张按能力域×难度档的推荐表。"""
    avg_in, avg_out = _avg_tokens(tasks)
    pol = DifficultyRouterPolicy(cat.executors, min_quality=min_quality,
                                 cost_weight=cost_weight, avg_in=avg_in, avg_out=avg_out)
    return {
        'min_quality': min_quality,
        'cost_weight': cost_weight,
        'recommendation_table': pol.export_table(),
        'diagnostics': pol.diagnostics,
    }


def run(args) -> Dict:
    scenario = args.scenario
    min_quality = args.min_quality
    utilization = args.gpu_utilization

    cat = load_catalog(utilization=utilization)
    if args.api_only:
        # 没有自建 GPU 的用户：候选池里只剩商用 API。
        # 这组实验的意义在于 —— 「省 token」的结论不能依赖「你有一张闲着的卡」。
        cat.executors = [e for e in cat.executors if e.kind == 'api_model']
    spec = get_preset(scenario) if scenario in PRESETS else PRESETS['mixed']

    # 多份负载：不同 seed 生成不同业务流，用于跨样本平均
    specs = [spec]
    task_sets = []
    for s in SEEDS:
        from dataclasses import replace
        task_sets.append(generate_workload(replace(spec, seed=s)))

    tasks = task_sets[2] if len(task_sets) > 2 else task_sets[0]   # seed=42 主样本
    workload_summary = summarize_workload(tasks)

    # 全报告统一用「5 份业务流拼接」的单一任务流：策略对比、前沿扫描、控制台
    # 三者必须是同一把尺子，否则同一份报告里 λ=0.2 会印出两个不同的省钱比例。
    all_tasks = [t for ts in task_sets for t in ts]
    avg_in, avg_out = _avg_tokens(all_tasks)
    results = run_experiment(all_tasks, cat.executors, min_quality=min_quality,
                             master_seed=args.seed, avg_in=avg_in, avg_out=avg_out,
                             cost_weight=args.cost_weight)

    # 跨样本离散度：证明结论不是某一份业务流的偶然（只作稳健性证据，不影响主结论）
    spread = {}
    for ts in task_sets:
        ai, ao = _avg_tokens(ts)
        r = run_experiment(ts, cat.executors, min_quality=min_quality, master_seed=args.seed,
                           avg_in=ai, avg_out=ao, cost_weight=args.cost_weight)
        for name, pr in r.items():
            spread.setdefault(name, []).append(pr.cost_saving)

    # 用户现状基线：全部走最强模型。所有「省了多少」都以它为分母
    baseline = results.get('strongest')
    sweep = sweep_cost_weight(all_tasks, cat.executors, master_seed=args.seed,
                              min_quality=min_quality, baseline=baseline)

    raw_catalog = json.loads((ROOT / 'tokenrouter' / 'data' / 'catalog.json').read_text(encoding='utf-8'))
    util_sens = utilization_sensitivity(raw_catalog, cat.gpu_specs)

    plan = build_router_plan(cat, tasks, min_quality, args.cost_weight)
    active_ids = {e.id for e in cat.executors}

    return {
        'meta': {
            'generated_at': datetime.now().isoformat(timespec='seconds'),
            'scenario': scenario,
            'seeds': SEEDS,
            'n_tasks_per_sample': len(tasks),
            'min_quality_constraint': min_quality,
            'cost_weight': args.cost_weight,
            'gpu_utilization': utilization,
            'api_only': bool(args.api_only),
            'executor_ids': [e.id for e in cat.executors],
            'master_seed': args.seed,
            'catalog_version': cat.meta.get('catalog_version'),
            'catalog_sources': cat.meta.get('sources'),
            'calibration_note': cat.meta.get('calibration_note'),
            'caveats': cat.meta.get('known_caveats'),
        },
        'workload_summary': workload_summary,
        'candidate_pool': {
            'n_executors': len(cat.executors),
            'api_models': [e.id for e in api_models(cat)],
            'self_hosted_models': [e.id for e in self_hosted_models(cat)],
            'unit_cost_yuan_per_mtok': {k: round(v, 4) for k, v in sorted(cat.unit_costs.items())
                                        if k in active_ids},
        },
        'policy_comparison': {
            name: {
                'description': r.description,
                'total_cost_yuan': round(r.total_cost, 4),
                'accuracy': r.accuracy,
                'cost_saving': getattr(r, 'cost_saving', 0.0),
                'quality_retention': getattr(r, 'quality_retention', 0.0),
                'efficiency_gain': getattr(r, 'efficiency_gain', 0.0),
                'latency_change': getattr(r, 'latency_change', 0.0),
                'avg_latency_s': r.avg_latency,
                'p95_latency_s': r.p95_latency,
                'calls_per_task': r.calls_per_task,
                'escalation_rate': r.escalation_rate,
                'model_usage': r.model_usage,
                'band_usage': r.band_usage,
            } for name, r in results.items()
        },
        'pareto_front': [
            {
                'cost_weight': p.cost_weight,
                'min_quality': p.min_quality,
                'accuracy': p.accuracy,
                'total_cost_yuan': round(p.total_cost, 4),
                'cost_saving': p.cost_saving,
                'quality_retention': p.quality_retention,
                'recommended_table': p.recommended_table,
            } for p in sweep
        ],
        'knee_point': (
            {
                'cost_weight': knee.cost_weight,
                'min_quality': knee.min_quality,
                'accuracy': knee.accuracy,
                'total_cost_yuan': round(knee.total_cost, 4),
                'cost_saving': knee.cost_saving,
                'quality_retention': knee.quality_retention,
            } if (knee := knee_point(sweep)) else None
        ),
        'deliverable_plan': plan,
        'utilization_sensitivity': util_sens,
        # 结论稳健性的唯一证据：同一策略在 5 份独立业务流上的省钱比例极差
        'per_sample_spread': {
            name: {
                'cost_saving_min': round(min(v), 4),
                'cost_saving_max': round(max(v), 4),
                'cost_saving_mean': round(sum(v) / len(v), 4),
                'n_samples': len(v),
            } for name, v in sorted(spread.items())
        },
    }


def main():
    ap = argparse.ArgumentParser(description='HeteroNexus Token 侧路由实验')
    ap.add_argument('--scenario', default='mixed', help=f'场景预设，可选: {list(PRESETS)}')
    ap.add_argument('--min-quality', type=float, default=0.0,
                    help='硬质量下限（安全网），0 表示不启用')
    ap.add_argument('--cost-weight', type=float, default=0.2,
                    help='路由策略的成本权重 λ，越大越省钱、质量相应下降；'
                         '0.2 为推荐默认（约保住 98%% 质量、省一半钱）')
    ap.add_argument('--gpu-utilization', type=float, default=0.35,
                    help='自建 GPU 有效负载率；0.35 为保守默认，1.0 会显著低估自建成本')
    ap.add_argument('--api-only', action='store_true',
                    help='只使用商用 API（模拟没有自建 GPU 的个人/小企业用户）')
    ap.add_argument('--seed', type=int, default=42, help='呼叫结果的确定性随机主种子')
    args = ap.parse_args()

    payload = run(args)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

    # ---- 控制台摘要 ----
    m = payload['meta']
    print('=' * 72)
    print('HeteroNexus · Token 侧路由离线评估')
    print(f"场景={m['scenario']}  任务/样本={m['n_tasks_per_sample']}  "
          f"种子={m['seeds']}  GPU负载率={m['gpu_utilization']}")
    print('=' * 72)

    w = payload['workload_summary']
    print(f"\n业务负载画像：平均难度 {w['avg_difficulty']}  "
          f"简单/中等/困难 = {w['difficulty_ratio']}")
    print(f"  token 总量: input {w['total_input_tokens']:,} / output {w['total_output_tokens']:,}")

    print('\n候选资源单位成本（元 / 百万 output token）:')
    for k, v in payload['candidate_pool']['unit_cost_yuan_per_mtok'].items():
        print(f'  {v:>8.2f}   {k}')

    print('\n策略对比（基线 = 全部走最强模型）:')
    print(f"  {'策略':<20}{'总成本':>12}{'省钱':>10}{'正确率':>10}{'质量保有':>10}{'每答对成本改善':>14}")
    for name, r in payload['policy_comparison'].items():
        print(f"  {name:<20}{r['total_cost_yuan']:>12.2f}{r['cost_saving']*100:>9.1f}%"
              f"{r['accuracy']*100:>9.1f}%{r['quality_retention']*100:>9.1f}%"
              f"{r['efficiency_gain']*100:>13.1f}%")

    print('\n成本—质量帕累托前沿（成本权重 λ 扫描）:')
    print(f"  {'λ':>8}{'实际正确率':>12}{'总成本':>12}{'省钱':>10}")
    for p in payload['pareto_front']:
        print(f"  {p['cost_weight']:>8.2f}{p['accuracy']*100:>11.1f}%"
              f"{p['total_cost_yuan']:>12.2f}{p['cost_saving']*100:>9.1f}%")

    kp = payload['knee_point']
    if kp:
        print(f"\n推荐配置（膝点）: λ={kp['cost_weight']}  "
              f"正确率 {kp['accuracy']*100:.1f}%  省钱 {kp['cost_saving']*100:.1f}%")

    print(f"\n跨 {len(m['seeds'])} 份独立业务流的离散度（省钱比例 %）:")
    for name, s in payload['per_sample_spread'].items():
        print(f"  {name:<20} {s['cost_saving_min']*100:>6.1f} ~ {s['cost_saving_max']*100:>6.1f}  "
              f"均值 {s['cost_saving_mean']*100:>6.1f}")

    print('\nGPU 利用率敏感性（自建 vs 商用 API）:')
    for r in payload['utilization_sensitivity']:
        flag = '自建更省' if r['self_hosted_wins'] else 'API 更省'
        print(f"  负载率 {r['utilization']*100:>5.0f}%  自建 {r['self_cost_per_mtok']:>7.2f}  "
              f"vs {r['ref_model']} {r['ref_cost_per_mtok']:>7.2f}   → {flag}")

    print(f'\n完整结果已写入: {OUT_FILE}')


if __name__ == '__main__':
    main()

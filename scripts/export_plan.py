# -*- coding: utf-8 -*-
"""
导出可执行的模型路由方案文件，并**当场验证导出—执行闭环**。

导出只是上半步。下半步是把生成的 YAML 装回来，在同样的任务流上跑一遍，
确认花费与正确率与生成它的策略逐位一致 —— 否则「方案」只是一段看起来能用的文本。

用法：
    python scripts/export_plan.py                        # 默认 mixed / λ=0.2
    python scripts/export_plan.py --scenario dev_copilot --out plan_dev.yaml
    python scripts/export_plan.py --cost-weight 0.3 --format json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tokenrouter import (  # noqa: E402
    load_catalog, get_preset, PRESETS, generate_workload, run_experiment,
    sweep_cost_weight, knee_point,
)
from tokenrouter.policies import DifficultyRouterPolicy, OfflineTablePolicy  # noqa: E402
from tokenrouter.planfile import (  # noqa: E402
    build_plan_dict, render_plan_yaml, load_plan, apply_plan, PlanFilePolicy,
)
from tokenrouter.engine import run_policy  # noqa: E402

SEEDS = [1, 7, 42, 999, 2026]


def _avg_tokens(tasks):
    return (max(1, int(sum(t.input_tokens for t in tasks) / len(tasks))),
            max(1, int(sum(t.output_tokens for t in tasks) / len(tasks))))


def export(args) -> Dict:
    from dataclasses import replace
    cat = load_catalog(utilization=args.gpu_utilization)
    if args.api_only:
        cat.executors = [e for e in cat.executors if e.kind == 'api_model']
    spec = get_preset(args.scenario) if args.scenario in PRESETS else PRESETS['mixed']
    tasks = [t for s in SEEDS for t in generate_workload(replace(spec, seed=s))]
    ai, ao = _avg_tokens(tasks)

    pol = DifficultyRouterPolicy(cat.executors, min_quality=args.min_quality,
                                 cost_weight=args.cost_weight, avg_in=ai, avg_out=ao)
    res = run_experiment(tasks, cat.executors, args.min_quality, args.seed, ai, ao,
                         args.cost_weight)
    base, offline = res['strongest'], res['offline_table']

    meta = {
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'scenario': args.scenario,
        'cost_weight': args.cost_weight,
        'min_quality': args.min_quality,
        'gpu_utilization': args.gpu_utilization,
        'api_only': bool(args.api_only),
        'catalog_version': cat.meta.get('catalog_version'),
        'n_tasks_evaluated': len(tasks),
        'expected_cost_saving': round(float(offline.cost_saving), 4),
        'expected_accuracy': round(float(offline.accuracy), 4),
        'baseline_accuracy': round(float(base.accuracy), 4),
        'baseline_cost_yuan': round(float(base.total_cost), 4),
        'expected_cost_yuan': round(float(offline.total_cost), 4),
        'caveat': '省钱比例与质量代价必须同时引用；本表按难度档中点决策，非逐任务最优。',
    }
    plan = build_plan_dict(pol, meta)
    return {'plan': plan, 'tasks': tasks, 'executors': cat.executors,
            'expected': {'cost': offline.total_cost, 'accuracy': offline.accuracy},
            'baseline': {'cost': base.total_cost, 'accuracy': base.accuracy}}


def verify_roundtrip(plan: Dict, tasks, executors, expected, master_seed) -> Dict:
    """把导出的方案装回来执行，结果必须与生成它的策略一致。"""
    text = render_plan_yaml(plan)
    loaded = load_plan(text)
    r = run_policy(PlanFilePolicy(loaded), tasks, executors, master_seed)
    ok_cost = abs(r.total_cost - expected['cost']) < max(1e-9, abs(expected['cost']) * 1e-12)
    ok_acc = abs(r.accuracy - expected['accuracy']) < 1e-12
    # 抽样确认查表路径本身没走偏（而不是恰好总数相同）
    sample_ok = True
    for t in tasks[:50]:
        picked = apply_plan(loaded, t.domain, t.difficulty)
        expect = next((x['use'] for x in plan['rules']
                       if x['when']['domain'] == t.domain
                       and x['when']['difficulty_band'] == _band(t.difficulty)), None)
        if picked != expect:
            sample_ok = False
            break
    return {'cost': r.total_cost, 'accuracy': r.accuracy,
            'cost_ok': ok_cost, 'accuracy_ok': ok_acc, 'sample_ok': sample_ok,
            'all_ok': ok_cost and ok_acc and sample_ok}


def _band(d: float) -> str:
    from tokenrouter.planfile import band_of
    return band_of(d)


def main():
    ap = argparse.ArgumentParser(description='导出可执行的 HeteroNexus 模型路由方案')
    ap.add_argument('--scenario', default='mixed', help=f'场景预设，可选: {list(PRESETS)}')
    ap.add_argument('--cost-weight', type=float, default=0.2, help='成本权重 λ')
    ap.add_argument('--min-quality', type=float, default=0.0, help='硬质量下限')
    ap.add_argument('--gpu-utilization', type=float, default=0.35, help='自建 GPU 负载率')
    ap.add_argument('--api-only', action='store_true', help='只使用商用 API')
    ap.add_argument('--seed', type=int, default=42, help='主种子')
    ap.add_argument('--format', choices=['yaml', 'json'], default='yaml')
    ap.add_argument('--out', default=None, help='输出路径，默认 plans/plan_<scenario>.yaml')
    args = ap.parse_args()

    out = export(args)
    plan = out['plan']
    text = render_plan_yaml(plan)
    if args.format == 'json':
        text = json.dumps(plan, ensure_ascii=False, indent=2)

    path = Path(args.out) if args.out else ROOT / 'plans' / f"plan_{args.scenario}.{args.format}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')

    v = verify_roundtrip(plan, out['tasks'], out['executors'], out['expected'], args.seed)

    print('=' * 68)
    print('HeteroNexus 路由方案导出')
    print(f"场景={args.scenario}  λ={args.cost_weight}  "
          f"候选={len(out['executors'])}  任务={len(out['tasks'])}")
    print('=' * 68)
    b, e = out['baseline'], out['expected']
    print(f"  用户现状（全用最强）  ¥{b['cost']:.2f} / {b['accuracy']*100:.1f}%")
    print(f"  照本方案执行          ¥{e['cost']:.2f} / {e['accuracy']*100:.1f}%"
          f"   省 {(1-e['cost']/b['cost'])*100:.1f}%")
    print(f"\n  规则数 {len(plan['rules'])}  fallback={plan['fallback']}")
    print(f"  已写入: {path}")

    print('\n导出—执行闭环校验:')
    print(f"  装回后总花费 ¥{v['cost']:.4f}  {'✓' if v['cost_ok'] else '✗ 与策略不一致'}")
    print(f"  装回后正确率 {v['accuracy']*100:.2f}%  {'✓' if v['accuracy_ok'] else '✗ 与策略不一致'}")
    print(f"  查表路径抽样 {'✓' if v['sample_ok'] else '✗ 命中了错误的规则'}")
    if not v['all_ok']:
        raise SystemExit('✗ 方案文件与生成它的策略不一致 —— 不能交付')
    print('\n✓ 导出的文件装回来执行，与页面上的数字逐位一致')


if __name__ == '__main__':
    main()

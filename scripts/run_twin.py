# -*- coding: utf-8 -*-
"""
Token 消耗数字孪生实验

回答三个批量静态仿真回答不了的问题：
    1. 加了真实的 TPM / RPM 配额与并发上限之后，「省钱方案」还成立吗？
    2. 便宜的自建模型容量有限，全切过去会丢掉多少请求？
    3. 配额档位不同（初创 / 团队 / 企业），结论会翻转吗？

**口径红线（本脚本的输出必须一起读）**
排队超时与预算耗尽的请求既不算花费也不算答对，所以「总花费最低」完全可以通过
大量丢请求来实现。本脚本因此：
    - 主表把 **服务率** 与 **每千次成功服务的花费** 并列印出；
    - 摘要里的省钱比例只在 **服务率 ≥ 99% 的策略之间** 比较，
      不拿「丢掉一半请求」的策略去跟「全服务」的策略比谁更省钱。

用法：
    python scripts/run_twin.py                       # 默认：团队档，一个工作日
    python scripts/run_twin.py --tier startup        # 看配额不足时会怎样
    python scripts/run_twin.py --budget 500          # 加预算约束
    python scripts/run_twin.py --overflow reroute    # 容量感知二次路由
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tokenrouter.catalog import load_catalog, load_capacity          # noqa: E402
from tokenrouter.engine import run_policy                            # noqa: E402
from tokenrouter.policies import build_policies                      # noqa: E402
from tokenrouter.twin import TwinSpec, generate_arrivals, run_twin   # noqa: E402
from tokenrouter.workload import generate_workload, get_preset, summarize_workload  # noqa: E402

OUT_PATH = ROOT / 'data' / 'twin.json'

# 档位敏感性只跑这几个：全最强（用户现状）、两个路由方案、全最便宜（极端对照）
SWEEP_POLICIES = ('strongest', 'difficulty_router', 'offline_table', 'cheapest')
MIN_SERVE_RATE_FOR_HEADLINE = 0.99


def _fmt_row(name: str, r) -> str:
    return (f"| {name:<18s} | {r.n_served:>7d} | {r.serve_rate*100:>6.1f}% | "
            f"{r.total_cost:>9.2f} | {r.cost_per_1k_served:>8.2f} | "
            f"{r.wait_p95:>7.1f}s | {r.n_timeout:>7d} | {r.queue_rate*100:>6.1f}% |")


def run_one(cat, capacity, spec, tasks, arrivals, names=None):
    pols = build_policies(cat.executors, cost_weight=spec.cost_weight)
    out = {}
    for p in pols:
        if names and p.name not in names:
            continue
        out[p.name] = run_twin(p, tasks, arrivals, cat.executors, capacity, spec)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description='Token 消耗数字孪生实验')
    ap.add_argument('--scenario', default='mixed')
    ap.add_argument('--rate', type=float, default=150.0, help='基准到达率（请求/分钟）')
    ap.add_argument('--duration', type=float, default=8 * 3600.0, help='仿真时长（秒）')
    ap.add_argument('--start-hour', type=int, default=9)
    ap.add_argument('--tier', default='team',
                    choices=['startup', 'team', 'enterprise', 'unlimited'])
    ap.add_argument('--overflow', default='wait', choices=['wait', 'reroute'])
    ap.add_argument('--budget', type=float, default=None, help='预算（元）')
    ap.add_argument('--cost-weight', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--master-seed', type=int, default=42)
    ap.add_argument('--gpu-utilization', type=float, default=0.35)
    ap.add_argument('--api-only', action='store_true')
    ap.add_argument('--no-sweep', action='store_true', help='跳过配额档位敏感性扫描')
    args = ap.parse_args()

    cat = load_catalog(utilization=args.gpu_utilization)
    if args.api_only:
        cat.executors = [e for e in cat.executors if e.kind == 'api_model']
    capacity = load_capacity(args.tier)

    spec = TwinSpec(duration_s=args.duration, rate_rpm=args.rate,
                    start_hour=args.start_hour, seed=args.seed,
                    master_seed=args.master_seed, budget=args.budget,
                    overflow=args.overflow, cost_weight=args.cost_weight)
    arrivals = generate_arrivals(spec)
    if not arrivals:
        print('到达流为空：请提高 --rate 或 --duration')
        return 1

    wspec = get_preset(args.scenario)
    wspec.n_tasks = len(arrivals)
    wspec.seed = args.seed
    tasks = generate_workload(wspec)

    print(f"场景 {args.scenario} · 到达 {len(arrivals)} 条 / {args.duration/3600:.1f}h "
          f"（均 {len(arrivals)/(args.duration/60):.0f} 条每分钟）· 配额档 {args.tier} · "
          f"溢出策略 {args.overflow}" + (f" · 预算 ¥{args.budget}" if args.budget else ''))
    print()

    results = run_one(cat, capacity, spec, tasks, arrivals)

    print('| 策略 | 成功服务 | 服务率 | 总花费(¥) | 每千次(¥) | P95等待 | 超时丢弃 | 排队率 |')
    print('|---|---|---|---|---|---|---|---|')
    order = ['strongest', 'cheapest', 'cascade', 'difficulty_router', 'offline_table', 'oracle']
    for name in order:
        if name in results:
            print(_fmt_row(name, results[name]))
    print()

    # ---- 摘要：只在「服务得住」的策略之间比省钱 ----
    base = results.get('strongest')
    eligible = {k: v for k, v in results.items()
                if v.serve_rate >= MIN_SERVE_RATE_FOR_HEADLINE}
    if base and len(eligible) >= 2:
        print(f"【只在服务率 ≥ {MIN_SERVE_RATE_FOR_HEADLINE*100:.0f}% 的策略之间比较省钱】")
        for k, v in sorted(eligible.items(), key=lambda kv: kv[1].cost_per_1k_served):
            if k == 'strongest':
                print(f"  {k:<18s} 每千次 ¥{v.cost_per_1k_served:>7.2f} · 服务率 {v.serve_rate*100:.1f}%  （基线）")
            else:
                saving = 1 - v.cost_per_1k_served / base.cost_per_1k_served
                acc_gap = v.accuracy_over_served - base.accuracy_over_served
                print(f"  {k:<18s} 每千次 ¥{v.cost_per_1k_served:>7.2f} · 服务率 {v.serve_rate*100:.1f}% · "
                      f"省 {saving*100:.1f}% · 正确率 {v.accuracy_over_served*100:.1f}% "
                      f"({acc_gap*100:+.1f}pp)")
        dropped = [k for k, v in results.items() if v.serve_rate < MIN_SERVE_RATE_FOR_HEADLINE]
        if dropped:
            print()
            print('  以下策略服务率不足，其「总花费低」主要来自丢请求，不参与省钱比较：')
            for k in dropped:
                v = results[k]
                print(f"    {k:<16s} 服务率 {v.serve_rate*100:.1f}% · 丢弃 "
                      f"{v.n_arrivals - v.n_served} 条 · 总花费 ¥{v.total_cost:.2f}")
    print()

    # ---- 配额档位敏感性 ----
    sweep = {}
    if not args.no_sweep:
        print('【配额档位敏感性】路由方案 vs 全用最强模型')
        print('| 档位 | 路由·服务率 | 路由·每千次 | 最强·服务率 | 最强·每千次 | 省钱 |')
        print('|---|---|---|---|---|---|')
        for tier in ('startup', 'team', 'enterprise', 'unlimited'):
            cap_t = load_capacity(tier)
            sub = run_one(cat, cap_t, spec, tasks, arrivals, names=SWEEP_POLICIES)
            rt = sub.get('difficulty_router')
            st = sub.get('strongest')
            if rt is None or st is None:
                continue
            # 只在两者都服务得住时才谈省钱；否则标注服务率差距
            if rt.serve_rate >= MIN_SERVE_RATE_FOR_HEADLINE and st.cost_per_1k_served:
                saving = 1 - rt.cost_per_1k_served / st.cost_per_1k_served
                sv = f'{saving*100:.1f}%'
            else:
                sv = 'n/a'
            print(f"| {tier:<10s} | {rt.serve_rate*100:>6.1f}% | ¥{rt.cost_per_1k_served:>7.2f} | "
                  f"{st.serve_rate*100:>6.1f}% | ¥{st.cost_per_1k_served:>7.2f} | {sv} |")
            sweep[tier] = {k: v.__dict__ for k, v in sub.items()}
        print()

    # 时间序列采样（每 30 秒一个点）只服务于绘图，控制台自己在浏览器里算同一条
    # 曲线。8 小时 × 6 策略写进 JSON 是 3 MB 多，塞进 git 没有意义。
    def _slim(d):
        d = dict(d)
        d.pop('samples', None)
        return d

    payload = {
        'meta': {
            'scenario': args.scenario,
            'tier': args.tier,
            'rate_rpm': args.rate,
            'duration_s': args.duration,
            'start_hour': args.start_hour,
            'overflow': args.overflow,
            'budget': args.budget,
            'cost_weight': args.cost_weight,
            'seed': args.seed,
            'master_seed': args.master_seed,
            'gpu_utilization': args.gpu_utilization,
            'api_only': args.api_only,
            'n_arrivals': len(arrivals),
            'workload': summarize_workload(tasks),
            'baseline_note': ('省钱比例只在服务率 ≥ %.0f%% 的策略之间比较；'
                              '排队超时与预算耗尽的请求不计花费也不计答对，'
                              '故「总花费最低」可通过大量丢请求实现。'
                              % (MIN_SERVE_RATE_FOR_HEADLINE * 100)),
        },
        'results': {k: _slim(v.__dict__) for k, v in results.items()},
        'tier_sweep': {t: {k: _slim(v) for k, v in sub.items()}
                       for t, sub in sweep.items()},
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f"已写入 {OUT_PATH}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

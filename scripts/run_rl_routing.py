# -*- coding: utf-8 -*-
"""
预算约束下的 PPO 路由策略：训练 + 与全部基线同口径对比

要回答的问题：**给你这么多钱，怎么花最值？**

前面的实验（run_token_experiments.py）回答的是「每个请求各花多少」——
在那个问题下逐任务贪心已经是最优解，RL 没有用武之地。这里换成企业真正
面对的约束：一个月就这么多预算，花完就没了。预算把任务耦合起来，
「这一轮该不该上好模型」才变成一个需要规划的序贯决策。

用法：
    python scripts/run_rl_routing.py                 # 默认预算 50%
    python scripts/run_rl_routing.py --budget 0.3    # 更紧的预算
    python scripts/run_rl_routing.py --updates 400   # 训练更久

需要 numpy（训练用）；其余部分零依赖。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from tokenrouter import load_catalog, get_preset, generate_workload  # noqa: E402
from tokenrouter.rl_env import (  # noqa: E402
    BudgetRouteEnv, build_matrices, make_episodes, evaluate_budgeted, DOMAINS,
)
from tokenrouter.ppo import ActorCritic, train_ppo  # noqa: E402

TRAIN_SEEDS = [1, 7, 42]
EVAL_SEEDS = [999, 2026]          # 刻意与训练分开，看是否只是记住了训练批次
OUT_FILE = ROOT / 'data' / 'rl_routing.json'


def state_for(task, remaining: float, budget: float, n_left: int, total: int) -> np.ndarray:
    s = np.zeros(9)
    s[0] = task.difficulty
    if task.domain in DOMAINS:
        s[1 + DOMAINS.index(task.domain)] = 1.0
    s[5] = math.log1p(task.input_tokens / 1000.0)
    s[6] = math.log1p(task.output_tokens / 1000.0)
    s[7] = remaining / max(budget, 1e-12)
    s[8] = n_left / max(total, 1)
    return s


def _log_norm(v, lo, hi):
    v = max(v, 1e-9)
    lo = max(lo, 1e-9)
    hi = max(hi, lo * (1 + 1e-12))
    if hi <= lo:
        return 0.0
    return (math.log(v) - math.log(lo)) / (math.log(hi) - math.log(lo))


def baseline_choosers(executors, lam: float):
    """构造全部基线策略的选模函数。"""
    strongest = int(np.argmax(
        [sum(e.capability.values()) / max(len(e.capability), 1) for e in executors]))

    def strongest_fn(i, t, rem, budget, left, C, Q):
        return strongest

    def cheapest_fn(i, t, rem, budget, left, C, Q):
        return int(np.argmin(C))

    def utility_fn(i, t, rem, budget, left, C, Q):
        lo, hi = float(C.min()), float(C.max())
        sc = Q - lam * np.array([_log_norm(float(c), lo, hi) for c in C])
        return int(np.argmax(sc))

    def budget_greedy_fn(i, t, rem, budget, left, C, Q):
        """按「人均剩余预算」挑质量最高的模型 —— 预算感知的贪心，PPO 要超越的对象。"""
        share = rem / max(left, 1)
        ok = np.where(C <= share + 1e-12)[0]
        if len(ok):
            return int(ok[int(np.argmax(Q[ok]))])
        return int(np.argmin(C))

    return {
        'strongest': strongest_fn,
        'cheapest': cheapest_fn,
        'utility_greedy': utility_fn,
        'budget_greedy': budget_greedy_fn,
    }


def ppo_chooser(net):
    def fn(i, t, rem, budget, left, C, Q):
        mask = C <= rem + 1e-12
        if not mask.any():
            mask[np.argmin(C)] = True
        s = state_for(t, rem, budget, left, left + i)   # 本批总数 = 剩余 + 已过
        return int(net.greedy(s[None, :], mask.astype(bool)[None, :])[0])
    return fn


def main():
    ap = argparse.ArgumentParser(description='预算约束下的 PPO 路由策略')
    ap.add_argument('--budget', type=float, default=0.5,
                    help='预算 = 「全用最强模型跑完」花费的百分之多少')
    ap.add_argument('--scenario', default='mixed')
    ap.add_argument('--episode-len', type=int, default=150, help='每批任务数')
    ap.add_argument('--train-episodes', type=int, default=60)
    ap.add_argument('--eval-episodes', type=int, default=20)
    ap.add_argument('--updates', type=int, default=200, help='PPO 更新轮数')
    ap.add_argument('--ent-coef', type=float, default=0.01,
                    help='熵正则系数：太大会让策略停在随机附近，太小会过早收敛')
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--n-envs', type=int, default=8, help='并行轨迹数')
    ap.add_argument('--lam', type=float, default=0.2, help='基线贪心策略的 λ')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--gpu-utilization', type=float, default=0.35)
    ap.add_argument('--api-only', action='store_true')
    args = ap.parse_args()

    cat = load_catalog(utilization=args.gpu_utilization)
    if args.api_only:
        cat.executors = [e for e in cat.executors if e.kind == 'api_model']
    spec = get_preset(args.scenario)
    train_pool = [t for s in TRAIN_SEEDS for t in generate_workload(replace(spec, seed=s))]
    eval_pool = [t for s in EVAL_SEEDS for t in generate_workload(replace(spec, seed=s))]

    train_eps = make_episodes(train_pool, args.train_episodes, args.episode_len, seed=args.seed)
    eval_eps = make_episodes(eval_pool, args.eval_episodes, args.episode_len,
                             seed=args.seed + 1000)

    executors = cat.executors
    print('=' * 72)
    print('HeteroNexus · 预算约束下的路由（PPO vs 基线）')
    print(f"场景={args.scenario}  预算={args.budget:.0%}  候选={len(executors)}  "
          f"每批={args.episode_len} 任务")
    print(f"训练批次={len(train_eps)}（种子 {TRAIN_SEEDS}）  评估批次={len(eval_eps)}"
          f"（种子 {EVAL_SEEDS}，与训练不重叠）")
    print('=' * 72)

    # ---- 训练 ----
    env = BudgetRouteEnv(train_eps, executors, budget_ratio=args.budget,
                         n_envs=args.n_envs, seed=args.seed)
    net = ActorCritic(obs_dim=9, n_actions=len(executors), hidden=64, seed=args.seed)
    before = evaluate_budgeted(ppo_chooser(net), eval_eps, executors, args.budget)
    logs = train_ppo(env, net, n_updates=args.updates, horizon=args.episode_len,
                     seed=args.seed, ent_coef=args.ent_coef, lr=args.lr,
                     log_every=max(1, args.updates // 6))
    after = evaluate_budgeted(ppo_chooser(net), eval_eps, executors, args.budget)

    # ---- 基线 ----
    choosers = baseline_choosers(executors, args.lam)
    results = {k: evaluate_budgeted(f, eval_eps, executors, args.budget)
               for k, f in choosers.items()}
    results['ppo_untrained'] = before
    results['ppo'] = after

    print(f"\n{'策略':<18}{'答对数':>10}{'服务率':>10}{'花费':>12}{'每元答对':>12}")
    for k, r in sorted(results.items(), key=lambda kv: -kv[1]['correct']):
        print(f"  {k:<16}{r['correct']:>10}{r['served']/r['n_tasks']:>9.1%}"
              f"{r['cost']:>12.2f}{r['correct_per_yuan']:>12.2f}")

    best_base = max(results['budget_greedy']['correct'], results['utility_greedy']['correct'])
    gain = after['correct'] - best_base
    print(f"\nPPO 相对最强基线的答对数增量: {gain:+d}  "
          f"({gain / max(best_base, 1) * 100:+.1f}%)")
    print(f"PPO 相对未训练（随机初始策略）的增量: "
          f"{after['correct'] - before['correct']:+d}  → 学习是否真的发生了")

    print('\n训练曲线（每轮均值）:')
    for lg in logs:
        print(f"  update {lg['update']:>4}  答对 {lg['mean_correct']:.1f}  "
              f"花费 {lg['mean_spent']:.3f}  服务 {lg['mean_served']:.0f}")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'meta': {'scenario': args.scenario, 'budget_ratio': args.budget,
                 'episode_len': args.episode_len, 'train_seeds': TRAIN_SEEDS,
                 'eval_seeds': EVAL_SEEDS, 'n_train_episodes': len(train_eps),
                 'n_eval_episodes': len(eval_eps), 'updates': args.updates,
                 'n_envs': args.n_envs, 'seed': args.seed,
                 'gpu_utilization': args.gpu_utilization, 'api_only': bool(args.api_only),
                 'executor_ids': [e.id for e in executors]},
        'results': results,
        'training_logs': logs,
        'delta_vs_best_baseline': gain,
    }
    OUT_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\n完整结果已写入: {OUT_FILE}')


if __name__ == '__main__':
    main()

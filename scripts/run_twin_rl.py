# -*- coding: utf-8 -*-
"""
孪生环境里的 PPO 路由：训练 + 与全部基线同口径对比

要回答的问题：**预算有限、而且每个模型此刻不一定有空位时，怎么排最值？**

`run_rl_routing.py` 只有预算一维约束；`run_twin.py` 证明了容量才是省钱方案
最先撞上的墙（全用最便宜模型只能服务 45% 的请求）。这里把两件事放在一起：
智能体既要看「钱还剩多少」，也要看「哪些后端此刻有空位」，还得看「这个任务
有多难」。三者耦合之后，「逐任务贪心」不再是最优解 —— 这一版才真正有 RL 的位置。

用法：
    python scripts/run_twin_rl.py                      # 默认 team 档 / 20 分钟片段
    python scripts/run_twin_rl.py --tier startup       # 配额最紧的一档
    python scripts/run_twin_rl.py --budget 0.3         # 更紧的预算

需要 numpy（训练用）；其余部分零依赖。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Callable, List, Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from core.resource import ModelExecutor  # noqa: E402
from tokenrouter import load_catalog, load_capacity, get_preset  # noqa: E402
from tokenrouter.ppo import ActorCritic, train_ppo  # noqa: E402
from tokenrouter.twin import TwinSpec  # noqa: E402
from tokenrouter.twin_env import (  # noqa: E402
    TwinRouteEnv, baseline_choosers, evaluate_twin_agent, make_twin_episodes,
    obs_from, strongest_index,
)

TRAIN_SEEDS = [1, 7, 42]
EVAL_SEEDS = [999, 2026]          # 刻意与训练分开，看是否只是记住了训练片段
OUT_FILE = ROOT / 'data' / 'twin_rl.json'

# 只与「服务率达标」的策略比省钱。丢请求永远最便宜，不加这道闸就会得出
# 「什么都不做省 100%」这种结论。
SERVE_GATE = 0.99


def ppo_chooser(net: ActorCritic, ids: Sequence[str], mask_capacity: bool = True) -> Callable:
    """把训练好的网络包成与基线同签名的决策函数。

    两处容易写错、且写错后不报错的地方：

    - 返回的是**模型 id** 而不是下标。下标会被 `_Sim` 静默回退成「第一个可用
      后端」，于是训练好的网络与随机初始网络跑出逐位相同的结果（twin_env 里
      已加了显式报错兜底，但这里不要触发它）。
    - 掩码必须与训练时一致。训练时若屏蔽了没空位的后端，评估时也必须屏蔽，
      否则策略在被禁止的动作上做 argmax，等于换了一个策略。
    """
    m = len(ids)

    def fn(task, adm, info, C_row, Q_row, ref):
        obs = obs_from(task, info, C_row, ref, m)
        left = info.get('budget_left')
        mask = np.ones(m, dtype=bool) if left is None else (C_row <= float(left) + 1e-12)
        if mask_capacity:
            mask &= np.array(info.get('adm_bits') or [True] * m, dtype=bool)
        if not mask.any():
            mask = np.zeros(m, dtype=bool)
            mask[int(np.argmin(C_row))] = True
        return ids[int(net.greedy(obs[None, :], mask[None, :])[0])]
    return fn


def main() -> None:
    ap = argparse.ArgumentParser(description='孪生（预算 + 实时容量）约束下的 PPO 路由')
    ap.add_argument('--tier', default='startup',
                    help='配额档位：startup / team / enterprise / unlimited。'
                         '默认取 startup：只有在配额与预算同时咬紧的区间里，'
                         '「容量」才真正进入决策；team 档下容量几乎不紧张，'
                         '问题会退化成 run_rl_routing.py 那个纯预算问题。')
    ap.add_argument('--scenario', default='mixed')
    ap.add_argument('--rate', type=float, default=60.0, help='基准到达率（请求/分钟）')
    ap.add_argument('--duration', type=float, default=300.0, help='每段片段时长（秒）')
    ap.add_argument('--start-hour', type=int, default=9)
    ap.add_argument('--budget', type=float, default=0.5,
                    help='预算 = 「这段流量全交给最强模型」花费的百分之多少')
    ap.add_argument('--lam', type=float, default=0.2, help='奖励里的成本权重（也是基线贪心的 λ）')
    ap.add_argument('--queue-penalty', type=float, default=0.25, help='服务不了时的负奖励')
    ap.add_argument('--cost-norm', choices=('share', 'max'), default='max',
                    help="奖励里成本项的归一化基准：share=人均预算（默认，把预算约束"
                         "直接写进奖励），max=该任务交给最贵模型的花费（只消量纲）")
    ap.add_argument('--no-mask-capacity', action='store_true',
                    help='不屏蔽「此刻没空位」的后端：让智能体自己学值不值得排队。'
                         '信用分配显著变难，需要更多轮数才可能收敛。')
    ap.add_argument('--updates', type=int, default=300, help='PPO 更新轮数')
    ap.add_argument('--n-envs', type=int, default=4, help='并行轨迹数')
    ap.add_argument('--train-episodes', type=int, default=12)
    ap.add_argument('--eval-episodes', type=int, default=6)
    ap.add_argument('--ent-coef', type=float, default=0.01)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--gpu-utilization', type=float, default=0.35)
    args = ap.parse_args()

    cat = load_catalog(utilization=args.gpu_utilization)
    cap = load_capacity(args.tier)
    executors: List[ModelExecutor] = cat.executors
    m = len(executors)
    wspec = get_preset(args.scenario)

    def base_for(seed: int) -> TwinSpec:
        return TwinSpec(duration_s=args.duration, rate_rpm=args.rate,
                        start_hour=args.start_hour, master_seed=42, seed=seed,
                        wait_timeout=30.0)

    train_eps = []
    for s in TRAIN_SEEDS:
        k = max(1, args.train_episodes // len(TRAIN_SEEDS))
        train_eps += make_twin_episodes(wspec, base_for(s), k, base_seed=s * 100003)
    eval_eps = []
    for s in EVAL_SEEDS:
        k = max(1, args.eval_episodes // len(EVAL_SEEDS))
        eval_eps += make_twin_episodes(wspec, base_for(s), k, base_seed=s * 100003)

    # 训练环境用其中一段作为「规格模板」（时长/到达率/等待超时），episode 自带到达流
    base_spec = base_for(TRAIN_SEEDS[0])

    print('=' * 78)
    print('HeteroNexus · 孪生环境里的路由（预算 + 实时容量，PPO vs 基线）')
    print(f"档位={args.tier}  场景={args.scenario}  到达={args.rate:.0f} req/min  "
          f"片段={args.duration:.0f}s  候选={m} 个模型")
    print(f"预算={args.budget:.0%}（基准：全用最强模型 {executors[strongest_index(executors)].id}）"
          f"  λ={args.lam}  丢请求惩罚={args.queue_penalty}")
    print(f"训练片段={len(train_eps)}（种子 {TRAIN_SEEDS}）  "
          f"评估片段={len(eval_eps)}（种子 {EVAL_SEEDS}，与训练不重叠）")
    print('=' * 78)

    # ---- 训练 ----
    mask_capacity = not args.no_mask_capacity
    env = TwinRouteEnv(train_eps, executors, cap, base_spec, budget_ratio=args.budget,
                       lam=args.lam, queue_penalty=args.queue_penalty, n_envs=args.n_envs,
                       mask_capacity=mask_capacity, cost_norm=args.cost_norm)
    net = ActorCritic(obs_dim=env.obs_dim, n_actions=m, hidden=64, seed=args.seed)
    ids = [e.id for e in executors]
    before = evaluate_twin_agent(ppo_chooser(net, ids, mask_capacity), eval_eps, executors, cap,
                                 base_spec, args.budget, args.cost_norm)
    logs = train_ppo(env, net, n_updates=args.updates, horizon=env.horizon,
                     seed=args.seed, ent_coef=args.ent_coef, lr=args.lr,
                     log_every=max(1, args.updates // 6))
    after = evaluate_twin_agent(ppo_chooser(net, ids, mask_capacity), eval_eps, executors, cap,
                                base_spec, args.budget, args.cost_norm)

    # ---- 基线 ----
    results = {k: evaluate_twin_agent(f, eval_eps, executors, cap, base_spec, args.budget, args.cost_norm)
               for k, f in baseline_choosers(executors, args.lam).items()}
    results['ppo_untrained'] = before
    results['ppo'] = after

    print(f"\n{'策略':<18}{'服务数':>9}{'服务率':>9}{'答对数':>9}{'花费':>11}"
          f"{'每千次':>10}{'超时丢':>9}{'超预算丢':>9}")
    for k, r in sorted(results.items(), key=lambda kv: -kv[1]['correct']):
        print(f"  {k:<16}{r['served']:>9}{r['serve_rate']:>8.1%}{r['correct']:>9}"
              f"{r['cost']:>11.2f}{r['cost_per_1k']:>10.2f}"
              f"{r['timeout']:>9}{r['budget_drop']:>9}")
    print('  （"每元答对"这一列刻意不印：它单调偏向更便宜的模型，'
          '本表里从来都是「全用最便宜模型」最高，但它只服务了个位数的请求。）')

    # ---- 结论只从「服务率达标」的策略里挑 ----
    qualified = {k: r for k, r in results.items() if r['serve_rate'] >= SERVE_GATE}
    unqualified = {k: r for k, r in results.items() if r['serve_rate'] < SERVE_GATE}
    print(f"\n服务率 ≥ {SERVE_GATE:.0%} 的策略: {sorted(qualified) or '无'}")
    if unqualified:
        print('  ' + '；'.join(f"{k} 只服务 {r['serve_rate']:.1%}（丢 {r['dropped']} 个）"
                               for k, r in sorted(unqualified.items())))
        print('  → 这些策略的「总花费低」主要来自丢请求，**不参与省钱比较**。')

    # ---- 三条结论，按「能不能被反驳」的顺序排 ----
    base_only = {k: r for k, r in results.items() if k not in ('ppo', 'ppo_untrained')}
    cap_g = results.get('capacity_greedy', {})
    util_g = results.get('utility_greedy', {})
    print()
    print('1) 学习确实发生了（随机初始 → 训练后，同一批未见过的片段）:')
    print(f"     答对 {before['correct']} → {after['correct']} "
          f"({(after['correct'] - before['correct']) / max(before['correct'], 1) * 100:+.0f}%)"
          f"   服务 {before['served']} → {after['served']}"
          f"   花费 ¥{before['cost']:.2f} → ¥{after['cost']:.2f}")

    print('2) 容量感知 vs 不感知（这一条是孪生才问得出的问题）:')
    if cap_g and util_g:
        print(f"     capacity_greedy 服务 {cap_g['serve_rate']:.1%} / 答对 {cap_g['correct']}，"
              f"utility_greedy 服务 {util_g['serve_rate']:.1%} / 答对 {util_g['correct']}"
              f"  → 差 {(cap_g['serve_rate'] - util_g['serve_rate']) * 100:+.1f} 个百分点服务率")
        print(f"     看不见容量的贪心把 {util_g['timeout']} 个请求排死在队列里；"
              f"看得见的只丢 {cap_g['timeout']} 个。")

    print('3) PPO 相对容量感知贪心（如实报，包括没赢的时候）:')
    if cap_g:
        d = after['correct'] - cap_g['correct']
        print(f"     答对 {after['correct']} vs {cap_g['correct']}（{d:+d}，"
              f"{d / max(cap_g['correct'], 1) * 100:+.1f}%）"
              f"   服务率 {after['serve_rate']:.1%} vs {cap_g['serve_rate']:.1%}"
              f"   花费 ¥{after['cost']:.2f} vs ¥{cap_g['cost']:.2f}")
        if d >= 0:
            print('     → PPO 在容量约束下拿到了增量。')
        else:
            print('     → **PPO 没赢**：在「有空位就在可用集合里按效用挑」这一层，'
                  '固定 λ 的贪心已经接近最优。')
            print('       RL 的增量来自预算跨时间分配，而本档位下预算不是主要瓶颈'
                  f"（超预算丢 {after['budget_drop']} vs 排队丢 {after['timeout']}）"
                  '。这条要写进结论，不能只报赢的那半边。')
    # 同花费水平上的帕累托对比：只比「花的钱不超过 PPO」的基线
    same_cost = {k: r for k, r in base_only.items() if r['cost'] <= after['cost'] * 1.05}
    if same_cost:
        bk = max(same_cost, key=lambda k: same_cost[k]['correct'])
        print(f"     花费不超过 PPO（¥{after['cost']:.2f}）的基线里，答对最多的是 {bk}"
              f"（{same_cost[bk]['correct']}，¥{same_cost[bk]['cost']:.2f}）")
    if not qualified:
        print(f"\n  注：本档位没有任何策略服务率 ≥ {SERVE_GATE:.0%}（配额太紧），"
              f'省钱比例按口径不予计算 —— 这是结论，不是失败。')

    print('\n训练曲线（每轮各 env 均值）:')
    for lg in logs:
        print(f"  update {lg['update']:>4}  答对 {lg['mean_correct']:.1f}  "
              f"花费 {lg['mean_spent']:.3f}  服务 {lg['mean_served']:.0f}")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'meta': {'tier': args.tier, 'scenario': args.scenario, 'rate_rpm': args.rate,
                 'duration_s': args.duration, 'start_hour': args.start_hour,
                 'budget_ratio': args.budget, 'lam': args.lam,
                 'queue_penalty': args.queue_penalty, 'mask_capacity': mask_capacity,
                 'cost_norm': args.cost_norm,
                 'updates': args.updates,
                 'n_envs': args.n_envs, 'seed': args.seed,
                 'train_seeds': TRAIN_SEEDS, 'eval_seeds': EVAL_SEEDS,
                 'n_train_episodes': len(train_eps), 'n_eval_episodes': len(eval_eps),
                 'gpu_utilization': args.gpu_utilization,
                 'executor_ids': [e.id for e in executors]},
        'results': results,
        'training_logs': logs,
    }
    OUT_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\n完整结果已写入: {OUT_FILE}')


if __name__ == '__main__':
    main()

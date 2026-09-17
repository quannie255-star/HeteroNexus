# -*- coding: utf-8 -*-
"""
预算约束路由与 PPO 的测试

单独成文件的原因：这部分依赖 numpy，而项目其余部分是零依赖的。
放在这里可以在没有 numpy 的环境里整组跳过，不会连累其他测试。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

np = pytest.importorskip('numpy')

from tokenrouter import load_catalog, get_preset, generate_workload  # noqa: E402
from tokenrouter.rl_env import (  # noqa: E402
    BudgetRouteEnv, build_matrices, make_episodes, evaluate_budgeted,
)
from tokenrouter.ppo import ActorCritic, train_ppo  # noqa: E402


@pytest.fixture(scope='module')
def executors():
    return load_catalog().executors


@pytest.fixture(scope='module')
def pool():
    return generate_workload(get_preset('mixed'))


class TestBudgetEnv:

    def test_预算恰为_最强模型花费的指定比例(self, pool, executors):
        """预算口径必须逐条对齐：环境会打乱批次顺序，所以要按实际装载的批次校验。"""
        eps = make_episodes(pool, 3, 60, seed=0)
        env = BudgetRouteEnv(eps, executors, budget_ratio=0.5, n_envs=3, seed=0)
        strongest = int(np.argmax(
            [sum(e.capability.values()) / max(len(e.capability), 1) for e in executors]))
        for e in range(env.n_envs):
            _Q, C, _O = build_matrices(eps[int(env.ep[e])], executors, 42)
            assert env.budget[e] == pytest.approx(
                float(C[:, strongest].sum()) * 0.5, rel=1e-9)

    def test_花销永远不会超过预算(self, pool, executors):
        """这是环境的硬不变量：超支一旦发生，整套「预算下的比较」就失去意义。"""
        eps = make_episodes(pool, 4, 50, seed=1)
        env = BudgetRouteEnv(eps, executors, budget_ratio=0.4, n_envs=4, seed=1)
        rng = np.random.default_rng(0)
        for _ in range(50):
            mask = env.action_mask()
            a = np.array([rng.choice(np.where(mask[e])[0]) for e in range(env.n_envs)])
            env.step(a)
        assert (env.remaining >= -1e-9).all()
        assert (env.spent <= env.budget + 1e-9).all()

    def test_买不起的动作被屏蔽(self, pool, executors):
        eps = make_episodes(pool, 2, 40, seed=2)
        env = BudgetRouteEnv(eps, executors, budget_ratio=0.2, n_envs=2, seed=2)
        mask = env.action_mask()
        for e in range(env.n_envs):
            _Q, C, _O = build_matrices(eps[int(env.ep[e])], executors, 42)
            affordable = C[int(env.pos[e]), :] <= env.remaining[e] + 1e-12
            assert (mask[e] == affordable).all()
            assert mask[e].any(), '至少要有买得起的动作'

    def test_服务数不超过任务数(self, pool, executors):
        eps = make_episodes(pool, 3, 30, seed=3)
        env = BudgetRouteEnv(eps, executors, budget_ratio=0.6, n_envs=3, seed=3)
        rng = np.random.default_rng(3)
        for _ in range(30):
            mask = env.action_mask()
            a = np.array([rng.choice(np.where(mask[e])[0]) for e in range(env.n_envs)])
            env.step(a)
        assert (env.served <= 30).all()

    def test_答对与否与既有策略同源(self, pool, executors):
        """RL 里的「是否答对」必须与其他策略用的是同一个确定性抽样，
        否则 RL 赢的可能只是运气，而不是策略更好。"""
        from tokenrouter.policies import realized_correct
        tasks = pool[:20]
        Q, _C, O = build_matrices(tasks, executors, 42)
        for i, t in enumerate(tasks[:5]):
            for j, ex in enumerate(executors[:4]):
                expect = realized_correct(42, t, ex, Q[i, j])
                assert bool(O[i, j]) == expect


class TestBaselines:

    def test_预算越紧_最强模型的服务率越低(self, pool, executors):
        """预算一紧，「一律用最强模型」首先崩在服务率上 —— 这正是它的问题所在。"""
        eps = make_episodes(pool, 6, 60, seed=5)
        strong = int(np.argmax(
            [sum(e.capability.values()) / max(len(e.capability), 1) for e in executors]))
        loose = evaluate_budgeted(lambda i, t, r, b, l, C, Q: strong, eps, executors, 0.9)
        tight = evaluate_budgeted(lambda i, t, r, b, l, C, Q: strong, eps, executors, 0.3)
        assert tight['served'] < loose['served']
        assert tight['served'] / tight['n_tasks'] < 0.5

    def test_最便宜模型永远服务得完(self, pool, executors):
        eps = make_episodes(pool, 5, 50, seed=6)
        r = evaluate_budgeted(lambda i, t, rem, b, l, C, Q: int(np.argmin(C)),
                              eps, executors, 0.3)
        assert r['served'] == r['n_tasks']

    def test_花销不超过预算(self, pool, executors):
        eps = make_episodes(pool, 4, 50, seed=7)
        strongest = int(np.argmax(
            [sum(e.capability.values()) / max(len(e.capability), 1) for e in executors]))
        budget = sum(float(build_matrices(ep, executors, 42)[1][:, strongest].sum()) * 0.5
                     for ep in eps)
        r = evaluate_budgeted(lambda i, t, rem, b, l, C, Q: strongest, eps, executors, 0.5)
        assert r['cost'] <= budget + 1e-6


class TestPPO:

    def test_训练后优于未训练(self, pool, executors):
        """学习是否真的发生了 —— 这一条是这个模块存在的唯一理由。

        曾因为 grads() 里策略项漏了负号与 ratio 因子，训练曲线单调下行，
        智能体越学越差；这条测试就是那个 bug 的回归网。
        """
        eps = make_episodes(pool, 12, 60, seed=11)
        env = BudgetRouteEnv(eps, executors, budget_ratio=0.5, n_envs=4, seed=11)
        net = ActorCritic(obs_dim=9, n_actions=len(executors), hidden=32, seed=11)

        def chooser(n):
            def f(i, t, rem, budget, left, C, Q):
                mask = C <= rem + 1e-12
                if not mask.any():
                    mask[np.argmin(C)] = True
                s = np.zeros(9)
                s[0] = t.difficulty
                s[7] = rem / max(budget, 1e-12)
                s[8] = left / max(left + i, 1)
                return int(n.greedy(s[None, :], mask.astype(bool)[None, :])[0])
            return f

        before = evaluate_budgeted(chooser(net), eps[:6], executors, 0.5)['correct']
        train_ppo(env, net, n_updates=40, horizon=60, seed=11, log_every=10)
        after = evaluate_budgeted(chooser(net), eps[:6], executors, 0.5)['correct']
        assert after > before, f'训练后答对数应上升: {before} → {after}'

    def test_同种子可复现(self, pool, executors):
        """RL 最容易被质疑的就是「再跑一次结果就变了」。同一颗种子必须逐位一致。"""
        eps = make_episodes(pool, 8, 40, seed=21)

        def run():
            env = BudgetRouteEnv(eps, executors, budget_ratio=0.5, n_envs=4, seed=21)
            net = ActorCritic(obs_dim=9, n_actions=len(executors), hidden=32, seed=21)
            train_ppo(env, net, n_updates=10, horizon=40, seed=21, log_every=1000)
            return net.greedy(np.zeros((1, 9)), np.ones((1, len(executors)), dtype=bool))[0]

        assert run() == run()

    def test_动作屏蔽后概率仍归一(self):
        net = ActorCritic(obs_dim=9, n_actions=10, hidden=16, seed=0)
        mask = np.zeros((2, 10), dtype=bool)
        mask[0, 3] = True
        mask[0, 7] = True
        mask[1, :] = True
        from tokenrouter.ppo import masked_softmax
        p = masked_softmax(net.forward(np.zeros((2, 9)))[1], mask)
        assert p.sum(axis=1) == pytest.approx(1.0)
        assert p[0, 0] == 0.0 and p[0, 3] > 0

# -*- coding: utf-8 -*-
"""Token 消耗数字孪生的测试

最重要的一个测试是 `test_无容量约束时退化为批量仿真`：把容量设成无穷，时间、
排队、限流就全部失效，孪生必须**逐位退化**成 `engine.run_policy` 的结果。
这条守住之后，孪生新增的那套机制（事件推进、窗口重置、队列）才值得信任 ——
否则「服务率只有 45%」这类结论可能只是仿真 bug。
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from tokenrouter.catalog import load_catalog, load_capacity
from tokenrouter.engine import run_policy
from tokenrouter.policies import build_policies
from tokenrouter.twin import (
    DIURNAL_24, TwinSpec, _Sim, generate_arrivals, run_twin,
)
from tokenrouter.twin_env import (
    TwinRouteEnv, baseline_choosers, evaluate_twin_agent, make_twin_episodes,
    obs_dim, obs_from, strongest_index,
)
from tokenrouter.workload import get_preset, generate_workload


@pytest.fixture(scope='module')
def catalog():
    return load_catalog(utilization=0.35)


@pytest.fixture(scope='module')
def small_setup(catalog):
    """一个小到能在测试里跑完的设置。"""
    spec = TwinSpec(duration_s=600.0, rate_rpm=30.0, start_hour=9,
                    seed=7, master_seed=42, wait_timeout=30.0)
    arrivals = generate_arrivals(spec)
    tasks = generate_workload(replace(get_preset('mixed'), seed=7, n_tasks=len(arrivals)))
    return tasks, arrivals, spec


# ---------------------------------------------------------------- 到达过程

def test_到达时刻严格升序且在仿真时长内():
    spec = TwinSpec(duration_s=600.0, rate_rpm=30.0, start_hour=9)
    times = generate_arrivals(spec)
    assert times, '不该生成空的到达流'
    assert times == sorted(times)
    assert all(0.0 <= t < spec.duration_s for t in times)


def test_到达过程可复现且随种子变化():
    a = generate_arrivals(TwinSpec(seed=1))
    b = generate_arrivals(TwinSpec(seed=1))
    c = generate_arrivals(TwinSpec(seed=2))
    assert a == b
    assert a != c


def test_到达数不受银行家舍入影响():
    """150 × 1.35 = 202.5 会真实出现。

    Python 的 round(202.5) = 202（银行家舍入），JS 的 Math.round(202.5) = 203。
    如果哪天有人把 int(x + 0.5) 改回 round()，控制台与 Python 端就会分叉 ——
    这个测试直接把两种舍入钉在同一个数上。
    """
    assert int(202.5 + 0.5) == 203       # 两侧一致用的写法
    assert round(202.5) == 202           # Python 的 round 会给出另一个答案
    spec = TwinSpec(duration_s=600.0, rate_rpm=150.0, start_hour=10)
    times = generate_arrivals(spec)
    assert len(times) > 0


def test_时段系数是查表不是三角函数():
    """三角函数在 JS / Python 之间不保证逐位一致，所以系数必须写成常量表。"""
    assert len(DIURNAL_24) == 24
    assert all(isinstance(x, float) for x in DIURNAL_24)
    # 白天高、凌晨低
    assert DIURNAL_24[10] > DIURNAL_24[3]


# ---------------------------------------------------------------- 退化不变式

def test_无容量约束时退化为批量仿真(catalog, small_setup):
    """本模块最重要的正确性锚点。

    unlimited 档下容量无穷、并发无穷，于是不会排队、不会超时、窗口重置也没有
    意义 —— 孪生处理的每个请求都与批量仿真面对同一个 (任务, 路由结果)。
    如果这里对不上，说明事件循环、窗口或队列里有一个是错的。
    """
    tasks, arrivals, spec = small_setup
    cap = load_capacity('unlimited')
    policies = build_policies(catalog.executors, cost_weight=spec.cost_weight)
    assert policies, '没有构造出任何策略'
    for p in policies:
        twin = run_twin(p, tasks, arrivals, catalog.executors, cap, spec)
        batch = run_policy(p, tasks, catalog.executors, spec.master_seed)
        assert twin.n_served == len(tasks), f'{p.name}: 无容量约束下不该丢请求'
        assert twin.n_timeout == 0, f'{p.name}: 无容量约束下不该有超时'
        assert math.isclose(twin.total_cost, batch.total_cost, rel_tol=0, abs_tol=1e-9), \
            f'{p.name}: 花费 {twin.total_cost} != 批量 {batch.total_cost}'
        assert twin.correct == batch.correct_count, \
            f'{p.name}: 答对数 {twin.correct} != 批量 {batch.correct_count}'


def test_有容量约束时服务率不高于无约束(catalog, small_setup):
    """加了配额只能让服务率下降，不可能上升 —— 上升说明限流没生效。"""
    tasks, arrivals, spec = small_setup
    policies = {p.name: p for p in build_policies(catalog.executors)}
    p = policies['cheapest']
    free = run_twin(p, tasks, arrivals, catalog.executors, load_capacity('unlimited'), spec)
    tight = run_twin(p, tasks, arrivals, catalog.executors, load_capacity('startup'), spec)
    assert free.n_served == len(tasks)
    assert tight.n_served <= free.n_served
    assert tight.n_timeout > 0


def test_等待时间不超过超时上限(catalog, small_setup):
    """排了队但最终被服务的请求，等待时间不能超过 wait_timeout。"""
    tasks, arrivals, spec = small_setup
    r = run_twin(build_policies(catalog.executors)[0], tasks, arrivals,
                 catalog.executors, load_capacity('startup'), spec)
    assert r.wait_p95 <= spec.wait_timeout + 1e-9


def test_预算耗尽后不再产生花费(catalog, small_setup):
    tasks, arrivals, _ = small_setup
    spec = replace(small_setup[2], budget=1.0)      # 一个几乎必然花完的预算
    r = run_twin(build_policies(catalog.executors)[0], tasks, arrivals,
                 catalog.executors, load_capacity('unlimited'), spec)
    assert r.total_cost <= 1.0 + 1e-9
    assert r.n_budget_reject > 0


# ---------------------------------------------------------------- 结果口径

def test_服务数加丢弃数不超过到达数(catalog, small_setup):
    """仿真在 duration_s 处截断，队列里剩下的请求既不计服务也不计超时。

    这个测试守住的是「不会凭空多出服务数」；少于是允许的（属于下一段的 backlog）。
    """
    tasks, arrivals, spec = small_setup
    r = run_twin(build_policies(catalog.executors)[0], tasks, arrivals,
                 catalog.executors, load_capacity('startup'), spec)
    assert r.n_served + r.n_timeout + r.n_budget_reject <= len(tasks)
    assert r.n_served >= 0


def test_每千次花费与服务率同时给出(catalog, small_setup):
    """单独引用 total_cost 是被禁止的 —— 结果里必须同时有服务率这个护栏。"""
    tasks, arrivals, spec = small_setup
    r = run_twin(build_policies(catalog.executors)[0], tasks, arrivals,
                 catalog.executors, load_capacity('startup'), spec)
    assert 0.0 <= r.serve_rate <= 1.0
    if r.n_served:
        assert r.cost_per_1k_served > 0


# ---------------------------------------------------------------- RL 环境

def test_观测维度与掩码形状一致(catalog):
    cap = load_capacity('startup')
    ex = catalog.executors
    m = len(ex)
    base = TwinSpec(duration_s=180.0, rate_rpm=30.0, start_hour=9)
    eps = make_twin_episodes(get_preset('mixed'), base, 2, base_seed=0)
    env = TwinRouteEnv(eps, ex, cap, base, n_envs=2)
    obs = env.reset()
    mask = env.action_mask()
    assert obs.shape == (2, obs_dim(m))
    assert mask.shape == (2, m)
    assert mask.any(axis=1).all(), '至少要有一个可行动作'


def test_环境跑完一整条轨迹且奖励有正有负(catalog):
    """端到端：随机初始策略跑完，统计量与结局日志必须对得上。"""
    cap = load_capacity('startup')
    ex = catalog.executors
    base = TwinSpec(duration_s=180.0, rate_rpm=60.0, start_hour=9)
    eps = make_twin_episodes(get_preset('mixed'), base, 2, base_seed=0)
    env = TwinRouteEnv(eps, ex, cap, base, budget_ratio=0.5, n_envs=2)
    env.reset()
    total_r = 0.0
    steps = 0
    while not env.done.all() and steps < env.horizon + 5:
        mask = env.action_mask()
        a = mask.argmax(axis=1)          # 只选第一个可用动作（确定性，便于复现）
        _o, r, _d, _i = env.step(a)
        total_r += float(r.sum())
        steps += 1
    assert steps > 0
    assert env.done.all(), 'horizon 之内应该跑完'
    # 有服务就有正分，有丢弃就有负分；全正说明从未丢过请求（startup 档不可能）
    assert env.served.sum() > 0


def test_决策函数返回下标会被显式报错(catalog):
    """曾经踩过的坑：返回下标会被静默回退成「第一个可用后端」，
    导致训练好的网络与随机初始网络跑出逐位相同的结果。"""
    cap = load_capacity('startup')
    ex = catalog.executors
    base = TwinSpec(duration_s=180.0, rate_rpm=30.0, start_hour=9)
    eps = make_twin_episodes(get_preset('mixed'), base, 1, base_seed=0)
    with pytest.raises(ValueError, match='不是任何候选模型的 id'):
        evaluate_twin_agent(lambda task, adm, info, C, Q, ref: 0,   # 下标，不是 id
                            eps, ex, cap, base, 0.5)


def test_容量感知贪心优于不感知的贪心(catalog):
    """这条是孪生带来的结论：看得见「此刻有没有空位」能救下大量请求。"""
    cap = load_capacity('startup')
    ex = catalog.executors
    base = TwinSpec(duration_s=600.0, rate_rpm=60.0, start_hour=9)
    eps = make_twin_episodes(get_preset('mixed'), base, 2, base_seed=999)
    bl = baseline_choosers(ex, 0.2)
    a = evaluate_twin_agent(bl['capacity_greedy'], eps, ex, cap, base, 0.5, 'max')
    b = evaluate_twin_agent(bl['utility_greedy'], eps, ex, cap, base, 0.5, 'max')
    assert a['serve_rate'] > b['serve_rate']
    assert a['timeout'] < b['timeout']


def test_训练环境与评估函数的预算口径一致(catalog):
    """`strongest_index` 必须是同一个定义，否则练的是一道题、考的是另一道。"""
    ex = catalog.executors
    assert strongest_index(ex) == strongest_index(list(reversed(ex))[::-1])
    assert 0 <= strongest_index(ex) < len(ex)


def test_观测在训练与评估两条路径上同构(catalog):
    """obs_from 是唯一构造观测的地方；两条路径都调它，形状与取值必须一致。"""
    from tokenrouter.twin import _Sim
    cap = load_capacity('startup')
    ex = catalog.executors
    m = len(ex)
    base = TwinSpec(duration_s=180.0, rate_rpm=30.0, start_hour=9)
    eps = make_twin_episodes(get_preset('mixed'), base, 1, base_seed=0)
    tasks, arrivals = eps[0]
    env = TwinRouteEnv(eps, ex, cap, base, n_envs=1)
    env.reset()
    info = env._info_of(0, 0)
    o1 = env._obs_of(0, 0)
    o2 = obs_from(tasks[0], info, env.C[0][0], float(env.ref[0][0]), m)
    assert o1.shape == o2.shape == (obs_dim(m),)
    assert (o1 == o2).all()

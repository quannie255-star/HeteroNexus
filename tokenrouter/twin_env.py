# -*- coding: utf-8 -*-
"""
预算 + 实时容量双重约束下的序贯路由环境（PPO 用）

与 `rl_env.py`（只有预算）的区别
-------------------------------
`rl_env.py` 的耦合只来自预算：钱花完就没了。本环境在它之上再加两层真实约束：

    1. **容量**：每个后端有 TPM / RPM 配额与并发上限，此刻**有没有空位**是状态的一部分。
       选一个已经排满的后端，这个请求很可能排队超时 —— 那一刻的省钱是假的。
    2. **时间**：请求按到达过程进入，队列会积压也会消化。智能体看到的是
       「现在几点、队列多长」，而不是「还剩几个任务」。

于是「这一轮用哪个模型」同时受三个东西约束：预算还剩多少、哪些后端此刻有空位、
以及这个任务有多难。这是一个真正需要序贯决策的问题 —— 贪心在这里不再最优。

奖励按结局结算，不按决策瞬间结算
--------------------------------
决策在到达时刻做出，但结局可能晚很多才发生：请求先排队，几十秒后才被放行，
或者排队超时被丢掉。所以奖励由 `_Sim.settle_log` 驱动 —— **服务了才算答对，
超时了要赔**。否则智能体会学到「选满员的便宜后端」这种账面好看、实际丢请求的策略。

    服务成功   r = 答对(1/0) − λ·(花费 / 该任务交给最贵模型的花费)
    排队超时   r = −queue_penalty
    预算耗尽   r = −queue_penalty

成本项按「该任务交给最贵模型」归一化，使奖励尺度与任务大小无关；服务不了的请求
给负奖励，是因为**什么都不做最省钱**是本环境里最容易学坏的捷径。

动作屏蔽
--------
- **买不起**永远屏蔽：这是硬约束，与 `rl_env.BudgetRouteEnv` 口径一致。
- **此刻没空位**默认也屏蔽（`mask_capacity=True`）。理由不是偷懒，而是真实调度
  器就是这么工作的：向一个已经占满的后端提交，等于把请求塞进它的队列，
  而队列是我们自己加的机制。把「排队」留给系统（满了就换一个），把「在可用
  后端之间怎么权衡质量、成本与剩余预算」留给智能体 —— 后者才是序贯决策的部分。
  设 `mask_capacity=False` 可以放开这个约束，让智能体自己学「值不值得排队」；
  放开后信用分配会显著变难（排队的结果几十步之后才结算），需要更长的训练。
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.resource import ModelExecutor, TaskProfile
from .catalog import LoadedCapacity
from .twin import TwinSpec, _Sim, generate_arrivals
from .workload import WorkloadSpec, generate_workload

DOMAINS = ('general', 'knowledge', 'coding', 'reasoning')

# 观测的前 10 维是任务与系统标量，其后每个候选模型 1 个「此刻有空位」位，
# 最后 1 维是「当前能用的最便宜模型相对最贵模型的成本比」。
BASE_DIM = 10

# 排队 / 预算耗尽的负奖励。取值含义：相当于「答对一题收益的 1/4 被扣掉」，
# 再大智能体会学会宁可全部丢请求。
QUEUE_PENALTY = 0.25


def obs_dim(n_models: int) -> int:
    return BASE_DIM + n_models + 1


def strongest_index(executors: Sequence[ModelExecutor]) -> int:
    """「最强模型」的下标 —— 预算口径的基准。

    训练环境与评估函数必须用**同一个**定义，否则同一个 budget_ratio 在两边
    对应不同的钱数，PPO 练的是一道题、考的是另一道。
    """
    return int(np.argmax([sum(e.capability.values()) / max(len(e.capability), 1)
                          for e in executors]))


def obs_from(task: TaskProfile, info: Dict[str, Any], C_row: np.ndarray,
             ref: float, m: int) -> np.ndarray:
    """把「任务 + 此刻系统状态」拼成观测。

    训练时（环境内部）与评估时（`router` 回调里）都调这一个函数 —— 两边观测
    只要构造方式分叉，训练出来的策略在评估时就会失效，这是最容易踩的坑。
    """
    o = np.zeros(obs_dim(m))
    o[0] = task.difficulty
    if task.domain in DOMAINS:
        o[1 + DOMAINS.index(task.domain)] = 1.0
    o[5] = np.log1p(task.input_tokens / 1000.0)
    o[6] = np.log1p(task.output_tokens / 1000.0)
    b = info.get('budget')
    if b:
        o[7] = max(0.0, min(1.0, float(info.get('budget_left') or 0.0) / b))
    o[8] = min(1.0, float(info.get('t', 0.0)) / max(float(info.get('duration', 1.0)), 1.0))
    cap_total = float(info.get('cap_total') or 1.0)
    o[9] = min(1.0, float(info.get('queued_total', 0.0)) / (cap_total * 4.0))
    bits = info.get('adm_bits') or [False] * m
    for j in range(m):
        o[BASE_DIM + j] = 1.0 if bits[j] else 0.0
    adm = [j for j in range(m) if bits[j]]
    if adm:
        o[BASE_DIM + m] = float(np.min(C_row[adm]) / max(ref, 1e-12))
    return o


def make_twin_episodes(wspec: WorkloadSpec, base_spec: TwinSpec, n_episodes: int,
                       base_seed: int = 0) -> List[Tuple[List[TaskProfile], List[float]]]:
    """为 RL 生成多段独立的「业务流片段」。

    每段用自己的到达种子，任务流长度与该段的到达数一致 —— 与 run_twin 的口径相同。
    """
    out = []
    for k in range(n_episodes):
        s = replace(base_spec, seed=base_seed + 1000 * k)
        arrivals = generate_arrivals(s)
        if not arrivals:
            continue
        tasks = generate_workload(replace(wspec, seed=s.seed, n_tasks=len(arrivals)))
        out.append((tasks, arrivals))
    return out


class TwinRouteEnv:
    """向量化环境：每个 step 让每个 env 处理**一个到达**。

    接口与 `rl_env.BudgetRouteEnv` 对齐，可直接喂给 `ppo.train_ppo`。
    """

    def __init__(self, episodes, executors: Sequence[ModelExecutor],
                 capacity: LoadedCapacity, base_spec: TwinSpec,
                 budget_ratio: float = 0.5, lam: float = 0.2,
                 queue_penalty: float = QUEUE_PENALTY, n_envs: int = 4,
                 mask_capacity: bool = True, cost_norm: str = 'share'):
        if not 0.0 < budget_ratio <= 1.0:
            raise ValueError('budget_ratio 必须在 (0, 1]')
        self.episodes = list(episodes)
        self.executors = list(executors)
        self.m = len(executors)
        self.obs_dim = obs_dim(self.m)
        self.n_envs = n_envs
        self.capacity = capacity
        self.base_spec = base_spec
        self.budget_ratio = budget_ratio
        self.lam = lam
        self.queue_penalty = queue_penalty
        self.mask_capacity = mask_capacity
        if cost_norm not in ('share', 'max'):
            raise ValueError("cost_norm 只能是 'share' 或 'max'")
        self.cost_norm = cost_norm
        self.ids = [e.id for e in executors]
        self.j_of = {e.id: j for j, e in enumerate(executors)}
        self._strongest = strongest_index(executors)
        self.ep_cursor = 0

        self._cache: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._reset_arrays()

    # ---------------- 数据预计算 ----------------

    def _mats(self, ep_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """(任务 × 候选) 的质量 / 成本矩阵。"""
        if ep_idx in self._cache:
            return self._cache[ep_idx]
        tasks, _ = self.episodes[ep_idx]
        Q = np.zeros((len(tasks), self.m))
        C = np.zeros((len(tasks), self.m))
        for i, t in enumerate(tasks):
            for j, e in enumerate(self.executors):
                est = e.estimate(t)
                Q[i, j] = est.quality
                C[i, j] = est.cost
        self._cache[ep_idx] = (Q, C)
        return self._cache[ep_idx]

    def _ref_of(self, C: np.ndarray, budget: float) -> np.ndarray:
        """成本归一化基准 —— 奖励里「花多了」到底是跟什么比。

        'share'：跟「人均预算」比（预算 ÷ 本段请求数）。这一档直接把预算约束写进
        奖励：花掉 5 个人的份就扣 5λ 分。它让智能体能**在决策那一刻**知道自己
        是不是在透支，而不必等预算耗尽时才发现。
        'max' ：跟「交给最贵模型」比，只消除任务大小的量纲差异，不带预算信息。
        """
        if self.cost_norm == 'share':
            return np.full(C.shape[0], max(budget / max(C.shape[0], 1), 1e-12))
        ref = C.max(axis=1)
        ref[ref <= 0] = 1.0
        return ref

    def _reset_arrays(self) -> None:
        self.sims: List[Optional[_Sim]] = [None] * self.n_envs
        self.tasks: List[list] = [[] for _ in range(self.n_envs)]
        self.Q: List[Optional[np.ndarray]] = [None] * self.n_envs
        self.C: List[Optional[np.ndarray]] = [None] * self.n_envs
        self.ref: List[Optional[np.ndarray]] = [None] * self.n_envs
        self.done = np.zeros(self.n_envs, dtype=bool)
        self._choice = np.zeros(self.n_envs, dtype=np.int64)
        self._log_pos = np.zeros(self.n_envs, dtype=np.int64)
        self._last_obs = np.zeros((self.n_envs, self.obs_dim))
        self._last_mask = np.ones((self.n_envs, self.m), dtype=bool)
        self.correct = np.zeros(self.n_envs)
        self.spent = np.zeros(self.n_envs)
        self.served = np.zeros(self.n_envs)

    # ---------------- episode 装载 ----------------

    def _load(self, e: int, ep_idx: int) -> None:
        tasks, arrivals = self.episodes[ep_idx]
        Q, C = self._mats(ep_idx)
        # 预算 = 「这段流量全交给最强模型」花费的一个比例
        budget = self.budget_ratio * float(C[:, self._strongest].sum())
        ref = self._ref_of(C, budget)

        def router(task, adm, info):
            return self.ids[int(self._choice[e])]

        spec = replace(self.base_spec, budget=budget, router=router)
        sim = _Sim(None, tasks, arrivals, self.executors, self.capacity, spec)
        sim.prime()
        self.sims[e] = sim
        self.tasks[e] = tasks
        self.Q[e], self.C[e], self.ref[e] = Q, C, ref
        self._log_pos[e] = 0

    # ---------------- 观测 / 掩码 ----------------

    def _info_of(self, e: int, i: int) -> Dict[str, Any]:
        """当前时刻的系统状态 —— 与 `_Sim._route_externally` 给 router 的字典同构。"""
        sim = self.sims[e]
        task = self.tasks[e][i]
        return {
            't': sim.t,
            'duration': sim.spec.duration_s,
            'budget': sim.spec.budget,
            'budget_left': sim.spec.budget - sim.spent,
            'queued_total': sum(len(b.queue) for b in sim.backends.values()),
            'in_flight_total': sum(b.in_flight for b in sim.backends.values()),
            'cap_total': sum(b.cap.max_concurrency for b in sim.backends.values()),
            'adm_bits': [sim.backends[x.id].can_admit(task) for x in self.executors],
            'served': sim.served,
        }

    def _obs_of(self, e: int, i: int) -> np.ndarray:
        return obs_from(self.tasks[e][i], self._info_of(e, i),
                        self.C[e][i], float(self.ref[e][i]), self.m)

    def _mask_of(self, e: int, i: int) -> np.ndarray:
        """买不起的屏蔽；`mask_capacity=True` 时此刻没空位的也屏蔽。"""
        left = self.sims[e].spec.budget - self.sims[e].spent
        mask = self.C[e][i] <= left + 1e-12
        if self.mask_capacity:
            mask &= np.array(self._info_of(e, i)['adm_bits'], dtype=bool)
        if not mask.any():
            mask[:] = False
            mask[int(np.argmin(self.C[e][i]))] = True
        return mask

    # ---------------- 结算 ----------------

    def _settle(self, e: int) -> float:
        """把自上次结算以来新产生的结局换算成奖励。"""
        sim = self.sims[e]
        log = sim.settle_log
        r = 0.0
        while self._log_pos[e] < len(log):
            kind, i, cost, ok, _wait = log[self._log_pos[e]]
            self._log_pos[e] += 1
            if kind == 'served':
                r += float(ok) - self.lam * float(cost / max(self.ref[e][i], 1e-12))
                self.correct[e] += float(ok)
                self.spent[e] += cost
                self.served[e] += 1
            else:                       # timeout / budget：服务不了就是负收益
                r -= self.queue_penalty
        return r

    def _advance(self, e: int) -> float:
        """推进到下一个到达，顺带结算中途发生的离场与超时。"""
        r = self._settle(e)
        i = self.sims[e].pump_to_arrival()
        if i is None:
            self.done[e] = True
            return r
        self._last_obs[e] = self._obs_of(e, i)
        self._last_mask[e] = self._mask_of(e, i)
        return r

    # ---------------- 标准接口 ----------------

    def reset(self) -> np.ndarray:
        self._reset_arrays()
        for e in range(self.n_envs):
            ep = (self.ep_cursor + e) % len(self.episodes)
            self._load(e, ep)
            self._advance(e)
        self.ep_cursor = (self.ep_cursor + self.n_envs) % len(self.episodes)
        return self._last_obs.copy()

    def action_mask(self) -> np.ndarray:
        return self._last_mask.copy()

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        rewards = np.zeros(self.n_envs)
        for e in range(self.n_envs):
            if self.done[e]:
                continue
            self._choice[e] = int(actions[e])
            self.sims[e]._pop_event()          # 处理堆顶的那个到达 → router 回调
            rewards[e] = self._advance(e)
        return self._last_obs.copy(), rewards, self.done.copy(), {}

    @property
    def horizon(self) -> int:
        return max(len(a) for _, a in self.episodes)


# ---------------- 基线 ----------------

def baseline_choosers(executors: Sequence[ModelExecutor], lam: float = 0.2):
    """四个对照策略。签名统一为 (task, adm_ids, info, C_row, Q_row, ref) -> executor_id。

    `capacity_greedy` 是这里的关键对照：它是「看得见容量」的贪心。
    如果 PPO 打不过它，说明加进来的时间维与容量约束并没有真正制造出序贯决策。
    """
    ids = [e.id for e in executors]
    strongest = strongest_index(executors)

    def strongest_fn(task, adm, info, C_row, Q_row, ref):
        return ids[strongest]

    def cheapest_fn(task, adm, info, C_row, Q_row, ref):
        return ids[int(np.argmin(C_row))]

    def utility_fn(task, adm, info, C_row, Q_row, ref):
        """看不见容量的贪心 —— 与 rl_env 里的 budget_greedy 同族。"""
        sc = Q_row - lam * (C_row / max(ref, 1e-12))
        return ids[int(np.argmax(sc))]

    def capacity_fn(task, adm, info, C_row, Q_row, ref):
        """看得见容量的贪心：只在「此刻有空位」的后端里挑效用最高的。

        一个空位都没有时不能退回「全体」—— 那会退化成看不见容量的贪心，
        这个基线就没意义了。退到最便宜的：既然谁都要排队，就排代价最小的队。
        """
        pool = [j for j in range(len(ids)) if adm and ids[j] in set(adm)]
        if not pool:
            return ids[int(np.argmin(C_row))]
        sc = [Q_row[j] - lam * (C_row[j] / max(ref, 1e-12)) for j in pool]
        return ids[int(pool[int(np.argmax(sc))])]

    return {
        'strongest': strongest_fn,
        'cheapest': cheapest_fn,
        'utility_greedy': utility_fn,
        'capacity_greedy': capacity_fn,
    }


def evaluate_twin_agent(choose_fn: Callable, episodes, executors: Sequence[ModelExecutor],
                        capacity: LoadedCapacity, base_spec: TwinSpec,
                        budget_ratio: float = 0.5, cost_norm: str = 'share') -> Dict[str, float]:
    """用给定决策函数跑完所有 episode，返回聚合指标。

    choose_fn(task, adm_ids, info, C_row, Q_row, ref) -> executor_id
    """
    served = correct = dropped = 0
    timeouts = budget_drops = 0
    spent = 0.0
    n = 0
    for tasks, arrivals in episodes:
        Q = np.array([[e.estimate(t).quality for e in executors] for t in tasks])
        C = np.array([[e.estimate(t).cost for e in executors] for t in tasks])
        strongest = strongest_index(executors)
        budget = budget_ratio * float(C[:, strongest].sum())
        if cost_norm == 'share':
            ref = np.full(len(tasks), max(budget / max(len(tasks), 1), 1e-12))
        else:
            ref = C.max(axis=1)
            ref[ref <= 0] = 1.0
        row_of = {t.task_id: i for i, t in enumerate(tasks)}
        ids = [e.id for e in executors]

        def router(task, adm, info):
            i = row_of[task.task_id]
            eid = choose_fn(task, adm, info, C[i], Q[i], float(ref[i]))
            if eid not in ids:
                # 曾在这里踩过坑：决策函数返回的是**下标**而不是模型 id，
                # _Sim 会静默回退到第一个可用后端，于是「训练好的网络」和
                # 随机初始网络跑出逐位相同的结果、而训练曲线却明明在涨。
                # 静默回退是这类 bug 的温床，这里直接报错。
                raise ValueError(
                    f'决策函数返回了 {eid!r}，它不是任何候选模型的 id '
                    f'（候选: {ids}）。注意要返回 id 字符串，不是下标。')
            return eid

        spec = replace(base_spec, budget=budget, router=router)
        r = _Sim(None, tasks, arrivals, executors, capacity, spec).run()
        served += r.n_served
        correct += r.correct
        spent += r.total_cost
        dropped += r.n_timeout + r.n_budget_reject
        timeouts += r.n_timeout
        budget_drops += r.n_budget_reject
        n += len(tasks)
    return {
        'served': served, 'n': n, 'serve_rate': served / n if n else 0.0,
        'correct': correct, 'accuracy': correct / served if served else 0.0,
        'cost': spent, 'cost_per_1k': spent / served * 1000 if served else 0.0,
        'dropped': dropped, 'timeout': timeouts, 'budget_drop': budget_drops,
        'per_yuan_correct': correct / spent if spent else 0.0,
    }

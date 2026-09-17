# -*- coding: utf-8 -*-
"""
预算约束下的路由决策环境（向量化，供 PPO 训练）

## 为什么必须有预算这一维

任务相互独立时，「逐任务贪心地最大化 q − λ·cost」**本身就已经是最优解** ——
任何 RL 都只能把它重新学一遍，拿不出增量。这不是实现问题，是问题定义问题：
**没有跨任务耦合，就没有序贯决策**。

真实的耦合来自预算：用户面对的是「这个月就这么多钱」，而不是「每个请求
各花多少」。一旦预算有限、且花完之后剩下的请求无法服务，「这一轮该不该上
好模型」就与「后面还剩多少任务、多少钱」耦合起来 —— 这是一个背包式的序贯
决策，贪心不再最优，RL 才有存在的理由。

## 与项目其余部分的对齐

- 答对与否沿用 policies.realized_correct 的确定性抽样（同一 detU + 盐值），
  因此同一个 (任务, 模型) 对在所有策略下结果一致，比较才是公平的。
- 环境是**确定性**的：给定种子，任务序列、成本矩阵、结果矩阵全部可复现。
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from core.resource import ModelExecutor, TaskProfile
from .policies import _det_u, _stable_hash, _SALT_CORRECT

DOMAINS = ['general', 'knowledge', 'coding', 'reasoning']
STATE_DIM = 9          # difficulty + domain(4) + log(in) + log(out) + 预算比 + 剩余任务比


def _quality(executor: ModelExecutor, task: TaskProfile) -> float:
    """难度校准质量模型（与 ModelExecutor.estimate 同源，这里只要质量项）。"""
    from core.resource import quality_at_difficulty
    return quality_at_difficulty(executor.ability(task.domain), task.difficulty,
                                 getattr(executor, 'gamma', 4.0))


def build_matrices(tasks: Sequence[TaskProfile], executors: Sequence[ModelExecutor],
                   master_seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """预计算 (任务 × 候选) 的质量 / 成本 / 是否答对 三个矩阵。

    预计算的意义不只是快：它把「答对与否」固定下来，使得不同策略面对的是
    **同一个平行世界** —— 否则比较 RL 与基线时，赢的可能只是运气。
    """
    n, m = len(tasks), len(executors)
    Q = np.zeros((n, m))
    C = np.zeros((n, m))
    O = np.zeros((n, m))
    for j, ex in enumerate(executors):
        hid = _stable_hash(ex.id)
        for i, t in enumerate(tasks):
            est = ex.estimate(t)
            q = est.quality
            Q[i, j] = q
            C[i, j] = est.cost
            O[i, j] = 1.0 if _det_u(master_seed, t.task_id, hid, _SALT_CORRECT) < q else 0.0
    return Q, C, O


def make_episodes(pool: Sequence[TaskProfile], n_episodes: int, episode_len: int,
                  seed: int) -> List[List[TaskProfile]]:
    """把任务池切成若干「月度批次」，每个批次是一次完整的预算周期。"""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(pool))
    out = []
    for _ in range(n_episodes):
        pick = rng.choice(idx, size=min(episode_len, len(pool)), replace=False)
        out.append([pool[int(k)] for k in pick])
    return out


class BudgetRouteEnv:
    """预算约束路由环境，同时跑 n_envs 条独立轨迹。

    状态（9 维）：
        0        任务难度
        1..4     能力域 one-hot
        5,6      log1p(input/1000), log1p(output/1000)
        7        剩余预算 / 初始预算
        8        剩余任务数 / 本批任务数

    动作：候选模型下标。**买不起的动作会被屏蔽**，否则智能体可以选一个
    永远付不起的模型，把学到的东西全部废掉。

    奖励：答对 +1，答错 0。预算耗尽则本批提前结束，剩余任务计 0 ——
    这就是「花超了」的代价，不需要额外的惩罚项。
    """

    def __init__(self, episodes: List[List[TaskProfile]], executors: Sequence[ModelExecutor],
                 budget_ratio: float = 0.5, master_seed: int = 42,
                 n_envs: int = 8, shuffle: bool = True, seed: int = 0):
        if not 0.0 < budget_ratio <= 1.0:
            raise ValueError('budget_ratio 必须在 (0, 1]')
        self.episodes = episodes
        self.executors = list(executors)
        self.budget_ratio = budget_ratio
        self.master_seed = master_seed
        self.n_envs = n_envs
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.m = len(executors)
        self._strongest = int(np.argmax(
            [sum(e.capability.values()) / max(len(e.capability), 1) for e in executors]))
        self._cursor = 0
        self._cache: dict = {}
        self.reset()

    # ---- 内部 ----
    def _mats(self, ep_idx: int):
        if ep_idx not in self._cache:
            Q, C, O = build_matrices(self.episodes[ep_idx], self.executors, self.master_seed)
            self._cache[ep_idx] = (Q, C, O)
        return self._cache[ep_idx]

    def reset(self) -> np.ndarray:
        self.ep = np.zeros(self.n_envs, dtype=np.int64)
        self.pos = np.zeros(self.n_envs, dtype=np.int64)
        self.done = np.zeros(self.n_envs, dtype=bool)
        self.remaining = np.zeros(self.n_envs)
        self.budget = np.zeros(self.n_envs)
        self.correct = np.zeros(self.n_envs)
        self.spent = np.zeros(self.n_envs)
        self.served = np.zeros(self.n_envs)
        if self.shuffle:
            order = self.rng.permutation(len(self.episodes))
            self._order = list(order)
        else:
            self._order = list(range(len(self.episodes)))
        for e in range(self.n_envs):
            self._load(e, self._order[(self._cursor + e) % len(self._order)])
        self._cursor = (self._cursor + self.n_envs) % max(len(self._order), 1)
        return self._obs()

    def _load(self, e: int, ep_idx: int):
        Q, C, _O = self._mats(ep_idx)
        self.ep[e] = ep_idx
        self.pos[e] = 0
        self.done[e] = False
        # 预算：只给「全用最强模型跑完这一批」所需花费的 budget_ratio 倍
        base = float(C[:, self._strongest].sum())
        self.budget[e] = base * self.budget_ratio
        self.remaining[e] = self.budget[e]
        self.correct[e] = 0.0
        self.spent[e] = 0.0
        self.served[e] = 0.0

    def _obs(self) -> np.ndarray:
        obs = np.zeros((self.n_envs, STATE_DIM))
        for e in range(self.n_envs):
            if self.done[e]:
                continue
            Q, C, _O = self._mats(int(self.ep[e]))
            i = int(self.pos[e])
            t = self.episodes[int(self.ep[e])][i]
            obs[e, 0] = t.difficulty
            obs[e, 1 + DOMAINS.index(t.domain)] = 1.0
            obs[e, 5] = np.log1p(t.input_tokens / 1000.0)
            obs[e, 6] = np.log1p(t.output_tokens / 1000.0)
            obs[e, 7] = self.remaining[e] / max(self.budget[e], 1e-12)
            obs[e, 8] = (len(self.episodes[int(self.ep[e])]) - i) / \
                max(len(self.episodes[int(self.ep[e])]), 1)
        return obs

    def action_mask(self) -> np.ndarray:
        """买不起的动作为 False。"""
        mask = np.ones((self.n_envs, self.m), dtype=bool)
        for e in range(self.n_envs):
            if self.done[e]:
                mask[e, :] = True      # 已结束的轨迹不参与学习，掩码给个全开即可
                continue
            _Q, C, _O = self._mats(int(self.ep[e]))
            mask[e, :] = C[int(self.pos[e]), :] <= self.remaining[e] + 1e-12
            if not mask[e, :].any():
                mask[e, int(np.argmin(C[int(self.pos[e]), :]))] = True
        return mask

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        rewards = np.zeros(self.n_envs)
        for e in range(self.n_envs):
            if self.done[e]:
                continue
            _Q, C, O = self._mats(int(self.ep[e]))
            i = int(self.pos[e])
            a = int(actions[e])
            cost = float(C[i, a])
            if cost > self.remaining[e] + 1e-12:
                a = int(np.argmin(C[i, :]))       # 兜底：买不起就退到最便宜
                cost = float(C[i, a])
            self.remaining[e] -= cost
            self.spent[e] += cost
            self.served[e] += 1
            rewards[e] = float(O[i, a])
            self.correct[e] += rewards[e]
            self.pos[e] = i + 1
            ep_len = len(self.episodes[int(self.ep[e])])
            if self.pos[e] >= ep_len:
                self.done[e] = True
            else:
                # 剩下的钱连最便宜的模型都买不起 → 本批提前结束
                if float(C[int(self.pos[e]), :].min()) > self.remaining[e] + 1e-12:
                    self.done[e] = True
        return self._obs(), rewards, self.done.copy(), {}


def evaluate_budgeted(choose_fn, episodes: List[List[TaskProfile]],
                      executors: Sequence[ModelExecutor], budget_ratio: float,
                      master_seed: int = 42) -> dict:
    """在固定预算下评估任意一个「选模型」的函数。

    choose_fn(i, task, remaining, budget, n_left, C_row, Q_row) -> action
    所有策略共用同一套矩阵与同一个预算口径，比较才成立。
    """
    n_correct = 0
    n_tasks = 0
    n_served = 0
    cost_total = 0.0
    for tasks in episodes:
        Q, C, O = build_matrices(tasks, executors, master_seed)
        strongest = int(np.argmax(
            [sum(e.capability.values()) / max(len(e.capability), 1) for e in executors]))
        budget = float(C[:, strongest].sum()) * budget_ratio
        remaining = budget
        for i in range(len(tasks)):
            if remaining <= 0:
                break
            a = int(choose_fn(i, tasks[i], remaining, budget, len(tasks) - i, C[i], Q[i]))
            cost = float(C[i, a])
            if cost > remaining + 1e-12:
                a = int(np.argmin(C[i, :]))
                cost = float(C[i, a])
                if cost > remaining + 1e-12:
                    break
            remaining -= cost
            cost_total += cost
            n_served += 1
            n_correct += int(O[i, a])
        n_tasks += len(tasks)
    return {
        'correct': n_correct,
        'served': n_served,
        'n_tasks': n_tasks,
        'accuracy_over_all': n_correct / max(n_tasks, 1),
        'accuracy_over_served': n_correct / max(n_served, 1),
        'cost': cost_total,
        'correct_per_yuan': n_correct / max(cost_total, 1e-12),
    }

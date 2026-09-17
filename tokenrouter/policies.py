# -*- coding: utf-8 -*-
"""
路由策略集

每类策略代表一种「用户会怎么花钱」的现状或一种「我们能给的方案」。

对比逻辑统一为：
    在**最终正确率不低于最强基线某个比例**的前提下，比较总花费。
单独看省钱没有意义 —— 全用最便宜的模型能省 95%，但答案全错。

关于随机性（重要）：
    所有「这次是否答对」「judge 是否放行」的判定都用 **确定性哈希** 而不是
    随机数发生器状态。这样不同策略面对的是**同一个平行世界**，
    比较才公平：Oracle 不会因为它抽到了更好的运气而显得更强。
"""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from core.objective import Objective, Reference, scalarize
from core.resource import ModelExecutor, TaskProfile, quality_at_difficulty
from .workload import DIFFICULTY_BANDS

# 用于方差分解：让不同用途的随机判定彼此独立
_SALT_CORRECT = 101
_SALT_JUDGE = 202


def _stable_hash(s: str) -> int:
    """确定性的字符串散列。

    **绝不能用内置 hash()**：Python 对 str 的哈希带随机种子（PYTHONHASHSEED），
    同一段代码两次运行会得到不同的值。一旦用它参与「这次是否答对」的判定，
    整套实验就不可复现了 —— 而这个项目的全部立论都建立在可复现之上。
    """
    return zlib.crc32(s.encode('utf-8'))


def _det_u(*ints: int) -> float:
    """确定性伪均匀随机数 [0,1)：同一组输入永远得到同一输出。"""
    h = 2166136261
    for i in ints:
        h = (h ^ ((int(i) * 2654435761) & 0xFFFFFFFF)) & 0xFFFFFFFF
        h = (h * 16777619) & 0xFFFFFFFF
    return ((h >> 8) & 0xFFFFFF) / float(1 << 24)


def realized_correct(master_seed: int, task: TaskProfile,
                     executor: ModelExecutor, quality: float) -> bool:
    """本次调用是否真的答对（Bernoulli(quality) 的确定性实现）。"""
    return _det_u(master_seed, task.task_id, _stable_hash(executor.id), _SALT_CORRECT) < quality


def judge_accepts(master_seed: int, task: TaskProfile, executor_id: str,
                  correct: bool, tpr: float, fpr: float) -> bool:
    """级联判定器是否放行当前答案。

    tpr: 答案正确时判定为「可信」的概率（正确放行）
    fpr: 答案错误时误判为「可信」的概率（误放行 → 质量损失的唯一来源）
    """
    r = _det_u(master_seed, task.task_id, _stable_hash(executor_id), _SALT_JUDGE)
    return r < (tpr if correct else fpr)


@dataclass
class Attempt:
    """一次模型调用。"""
    executor_id: str
    cost: float = 0.0
    latency: float = 0.0
    correct: bool = False
    accepted: bool = False


@dataclass
class RouteOutcome:
    """一个任务的完整路由结果。"""
    task_id: int
    attempts: List[Attempt] = field(default_factory=list)

    @property
    def cost(self) -> float:
        return sum(a.cost for a in self.attempts)

    @property
    def latency(self) -> float:
        return sum(a.latency for a in self.attempts)

    @property
    def final_correct(self) -> bool:
        return bool(self.attempts) and self.attempts[-1].correct

    @property
    def escalations(self) -> int:
        return max(0, len(self.attempts) - 1)


class Policy:
    """路由策略基类。"""
    name = 'policy'
    description = ''

    def route(self, task: TaskProfile, executors: Sequence[ModelExecutor],
              master_seed: int) -> RouteOutcome:
        raise NotImplementedError

    def plan_for(self, difficulty: float, domain: str) -> str:
        """该策略在这类任务上推荐哪个模型 —— 方案输出的原子单元。"""
        raise NotImplementedError


def _exec_quality(executor: ModelExecutor, task: TaskProfile) -> float:
    return quality_at_difficulty(executor.ability(task.domain), task.difficulty, executor.gamma)


def global_strongest(executors: Sequence[ModelExecutor]) -> ModelExecutor:
    """全局能力最强的模型（各能力域均值最高）。

    注意这里刻意**不用按 domain 的最强**：按域最强会挑到某些专项分高但价格
    贵一个量级的模型（例如 coding 域选 ¥105/M 的闭源旗舰，而 ¥16/M 的模型
    综合只差几个百分点）。回退目标应当与「用户本来会用的那个」对齐，
    这样任何策略在最坏情况下也不会比现状更差（no-regret）。
    """
    return max(executors, key=lambda e: sum(e.capability.values()) / max(len(e.capability), 1))


def _make_attempt(task: TaskProfile, executor: ModelExecutor, master_seed: int) -> Attempt:
    est = executor.estimate(task)
    correct = realized_correct(master_seed, task, executor, est.quality)
    return Attempt(executor_id=executor.id, cost=est.cost,
                   latency=est.latency, correct=correct)


def _by_index(executors: Sequence[ModelExecutor]) -> Dict[str, ModelExecutor]:
    return {e.id: e for e in executors}


class StrongestPolicy(Policy):
    """现状基线：所有任务一律走最强的模型。

    这就是大多数用户的真实行为 —— 不确定该用哪个，那就用最好的。
    """
    name = 'strongest'
    description = '全部走能力最强的模型（用户现状基线）'

    def __init__(self, executors: Sequence[ModelExecutor]):
        self.target = global_strongest(executors)

    def route(self, task, executors, master_seed) -> RouteOutcome:
        ex = _by_index(executors)[self.target.id]
        att = _make_attempt(task, ex, master_seed)
        att.accepted = True
        return RouteOutcome(task.task_id, [att])

    def plan_for(self, difficulty, domain) -> str:
        return self.target.id


class CheapestPolicy(Policy):
    """另一个极端：全走最便宜的。用来划定「省钱但有代价」的下界。"""
    name = 'cheapest'
    description = '全部走单位成本最低的模型（省钱的代价下界）'

    def __init__(self, executors: Sequence[ModelExecutor], avg_in: int = 1000, avg_out: int = 500):
        ref = TaskProfile(task_id=-1, input_tokens=avg_in, output_tokens=avg_out)
        self.target = min(executors, key=lambda e: e.estimate(ref).cost)

    def route(self, task, executors, master_seed) -> RouteOutcome:
        att = _make_attempt(task, _by_index(executors)[self.target.id], master_seed)
        att.accepted = True
        return RouteOutcome(task.task_id, [att])

    def plan_for(self, difficulty, domain) -> str:
        return self.target.id


class DifficultyRouterPolicy(Policy):
    """难度分档路由 —— **本项目输出的「方案」本身**。

    离线阶段按 (能力域 × 难度档) 交叉分组，对每组在所有候选模型里挑出
    「满足质量下限的最便宜者」，形成一张可直接交付的推荐表。

    这张表就是给用户的答案：「你的业务里，这类任务用这个模型就够了」。
    它可解释、可审计、可直接落成一个 YAML 规则 —— 区别于黑盒路由。
    """

    name = 'difficulty_router'
    description = '多目标效用路由：逐任务在候选模型上做成本-质量权衡（可交付方案）'

    def __init__(self, executors: Sequence[ModelExecutor], min_quality: float = 0.0,
                 cost_weight: float = 1.0, latency_weight: float = 0.0,
                 quality_weight: float = 1.0, avg_in: int = 1000, avg_out: int = 500):
        self.min_quality = min_quality
        # 走内核层统一的目标建模：成本/延迟/质量在同一把尺子上加权
        self.objective = Objective(name='utility_router',
                                   w_cost=cost_weight, w_latency=latency_weight,
                                   w_quality=quality_weight, min_quality=min_quality)
        self.executors = list(executors)
        self.table: Dict[Tuple[str, str], str] = {}
        self.diagnostics: List[Dict] = []
        self.avg_in, self.avg_out = avg_in, avg_out

        # 离线推荐表：面向「典型规模的任务」生成，用于交付给用户
        for domain in ('general', 'knowledge', 'coding', 'reasoning'):
            for band, (lo, hi) in DIFFICULTY_BANDS.items():
                mid = (lo + hi) / 2.0
                ref_task = TaskProfile(task_id=-1, difficulty=mid, domain=domain,
                                       input_tokens=avg_in, output_tokens=avg_out)
                ex, q, met = self._choose(ref_task)
                self.table[(domain, band)] = ex.id
                entry = {
                    'domain': domain, 'band': band, 'mid_difficulty': round(mid, 3),
                    'chosen': ex.id, 'expected_quality': round(q, 4),
                    'quality_target_met': met,
                }
                if not met:
                    entry['warning'] = (f'质量下限 {min_quality} 无模型可满足，'
                                        f'已退化为该能力域最强的模型')
                self.diagnostics.append(entry)

    def _choose(self, task: TaskProfile) -> Tuple[ModelExecutor, float, bool]:
        """对**具体任务**挑效用最优的模型。

        两件事都必须做对，缺一个就会得出荒谬的推荐：

        1. 必须 per-task 估算，而不是拿「平均规模的任务」代表全体 ——
           input/output token 比例会改变成本排序。

        2. 必须在候选之间做**多目标效用比较**，而不是「先按质量二值筛选、
           再挑最便宜的」。后者会让策略在临界点上选中「质量只强 3 个百分点、
           价格贵 10 倍」的模型，导致扫描出的帕累托前沿出现非单调
           （约束越严反而越便宜），这在答辩时会被一眼看穿。

        这里直接复用内核层 core.objective.scalarize，保证与算力侧共用同一套决策逻辑。
        """
        ests = [(e, e.estimate(task)) for e in self.executors]
        ref = Reference.from_estimates([est for _, est in ests])
        feasible = [(e, est) for e, est in ests if self.objective.feasible(est)]
        if not feasible:
            # 硬质量约束无解时回退到用户现状模型（no-regret）。
            # 不能顺手按效用挑一个 —— 那样会在「要求更严」时反而推荐更便宜/更差的模型，
            # 把业务语义搞反（详见 pareto.sweep_cost_weight 的注释）。
            fb = global_strongest(self.executors)
            return fb, fb.estimate(task).quality, False
        best_ex, best_est = max(feasible, key=lambda x: scalarize(x[1], self.objective, ref))
        return best_ex, best_est.quality, True

    def plan_for(self, difficulty: float, domain: str) -> str:
        return self.table.get((domain, self.band_of(difficulty)),
                              self.table.get(('general', 'medium')))

    def route(self, task, executors, master_seed) -> RouteOutcome:
        # 注意：优先信任传入的 executors（可能与构造时的不同），per-task 实时决策
        pool = list(executors) if executors else self.executors
        saved, self.executors = self.executors, pool
        try:
            ex, _q, _met = self._choose(task)
        finally:
            self.executors = saved
        att = _make_attempt(task, ex, master_seed)
        att.accepted = True
        return RouteOutcome(task.task_id, [att])

    def band_of(self, difficulty: float) -> str:
        for band, (lo, hi) in DIFFICULTY_BANDS.items():
            if lo <= difficulty <= hi:
                return band
        return 'medium'

    def export_table(self) -> List[Dict]:
        """导出可直接落地为路由配置的推荐表。"""
        return [{'domain': d, 'band': b, 'model': m} for (d, b), m in sorted(self.table.items())]


class OfflineTablePolicy(DifficultyRouterPolicy):
    """**可交付形态**：只用离线生成的 (能力域 × 难度档) 推荐表做路由。

    与父类（在线逐任务决策）的差别只有一处：运行时不重新求解，直接查表。
    为什么要单独测它：

    - 父类是「效果上界」，但它要求调用方把每个任务的难度/规模估出来再实时求解；
    - 本类是「落地版本」，用户拿到的就是一张 4×3 的表，可以写成 YAML 规则、
      塞进网关，甚至人工执行 —— 可解释、可审计、无运行时依赖。

    两者的差距就是**可解释性的代价**，必须在报告里如实给出，不能只报上界。
    """

    name = 'offline_table'
    description = '离线生成的 (能力域 × 难度档) 推荐表查表路由（可交付形态）'

    def route(self, task, executors, master_seed) -> RouteOutcome:
        pool = _by_index(executors) if executors else _by_index(self.executors)
        eid = self.plan_for(task.difficulty, task.domain)
        ex = pool.get(eid) or global_strongest(list(pool.values()))
        att = _make_attempt(task, ex, master_seed)
        att.accepted = True
        return RouteOutcome(task.task_id, [att])


class CascadePolicy(Policy):
    """级联路由：先用便宜模型，判定不可信再升级。

    经典做法（FrugalGPT / RouteLLM 的 cascade 变体）。代价是升级时要付两次钱，
    而且判定器会误放行错误答案 —— 这是它相对「离线分档 + 一次命中」的主要劣势。
    """

    name = 'cascade'
    description = '小模型先行，判定不可信则逐级升级（经典级联，带额外调用开销）'

    def __init__(self, executors: Sequence[ModelExecutor], ladder: Optional[List[str]] = None,
                 tpr: float = 0.80, fpr: float = 0.35, avg_in: int = 1000, avg_out: int = 500,
                 max_hops: int = 4):
        ref = TaskProfile(task_id=-1, input_tokens=avg_in, output_tokens=avg_out)
        ordered = sorted(executors, key=lambda e: e.estimate(ref).cost)
        self.ladder = ladder or [e.id for e in ordered]
        self.tpr = tpr
        self.fpr = fpr
        self.max_hops = max(1, max_hops)

    def route(self, task, executors, master_seed) -> RouteOutcome:
        idx = _by_index(executors)
        # 末位强制为全局最强的模型兜底 —— 否则级联可能在最便宜的一档里空转。
        # 注意必须先剔除最强再截断：直接截断后追加会让长度变成 max_hops + 1。
        strongest = global_strongest(executors)
        ladder = [x for x in self.ladder if x != strongest.id][:self.max_hops - 1]
        ladder.append(strongest.id)

        attempts: List[Attempt] = []
        for pos, eid in enumerate(ladder):
            att = _make_attempt(task, idx[eid], master_seed)
            last = (pos == len(ladder) - 1)
            if last or judge_accepts(master_seed, task, eid, att.correct, self.tpr, self.fpr):
                att.accepted = True
                attempts.append(att)
                break
            attempts.append(att)
        return RouteOutcome(task.task_id, attempts)

    def plan_for(self, difficulty, domain) -> str:
        return self.ladder[0]


class OraclePolicy(Policy):
    """理论最优选：预先知道每个模型「这次是否会答对」，选其中答对的最便宜者。

    用于衡量各策略相对理论最优的效率损失（regret）。同一平行世界里的最强人物。
    """

    name = 'oracle'
    description = '已知每次调用的真实正确性下的最小成本选择（理论上界）'

    def route(self, task, executors, master_seed) -> RouteOutcome:
        best = None
        for e in executors:
            att = _make_attempt(task, e, master_seed)
            if att.correct and (best is None or att.cost < best.cost):
                best = att
        if best is None:
            # 没有任何模型能答对 → 取用户现状模型，按「错误」的实际结果如实记账。
            # 这里刻意不按能力域取最强：那样会挑到专项强但贵一个量级的模型，
            # 与 no-regret 原则冲突。
            best = _make_attempt(task, global_strongest(executors), master_seed)
        best.accepted = True
        return RouteOutcome(task.task_id, [best])

    def plan_for(self, difficulty, domain) -> str:
        return 'oracle(动态)'


def build_policies(executors: Sequence[ModelExecutor], min_quality: float = 0.0,
                   cost_weight: float = 1.0,
                   avg_in: int = 1000, avg_out: int = 500) -> List[Policy]:
    """构造完整的策略对比集。

    min_quality: 硬质量下限（0 表示不启用安全网）
    cost_weight: 路由策略的成本权重 λ —— 调省钱幅度的主旋钮，见 pareto.sweep_cost_weight
    """
    return [
        StrongestPolicy(executors),
        CheapestPolicy(executors, avg_in=avg_in, avg_out=avg_out),
        CascadePolicy(executors, avg_in=avg_in, avg_out=avg_out),
        DifficultyRouterPolicy(executors, min_quality=min_quality, cost_weight=cost_weight,
                               avg_in=avg_in, avg_out=avg_out),
        OfflineTablePolicy(executors, min_quality=min_quality, cost_weight=cost_weight,
                           avg_in=avg_in, avg_out=avg_out),
        OraclePolicy(),
    ]

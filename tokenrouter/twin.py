# -*- coding: utf-8 -*-
"""
Token 消耗数字孪生 —— 有时间、有容量、会排队的离散事件仿真

与 `engine.py`（批量静态仿真）的分工
-----------------------------------
`engine.py` 回答的是「这一批任务一共值多少钱」：给它 2000 条任务，逐条路由，
最后汇总。它**没有时间维度** —— 任务之间没有先后，也不会互相抢资源。

本模块回答的是另一组问题，而这几个问题恰恰是方案落地时最先撞上的墙：

    这一分钟烧掉多少 token？
    什么时候会撞上供应商的 TPM / RPM 配额？
    便宜的自建模型容量有限，高峰期溢出到哪去、多花多少钱？
    预算什么时候烧完，烧完之后还有多少请求服务不了？

关键的设计取舍
--------------
1. **到达过程刻意不用泊松**。指数分布的间隔需要 `-ln(u)`，而 `Math.log`
   在 JS 与 Python 之间**不保证逐位一致**，会让控制台与 Python 端算出不同的
   到达时刻。这里改用「每分钟到达数 = 基准 × 时段系数 × 突发系数 + 分钟内均匀散布」，
   全部是四则运算，两侧可逐位复刻。附带的好处：真实业务流量本来就不是无记忆的
   泊松过程，显式突发比泊松更接近现实。同理，时段系数用**查表**而非 sin 曲线。

2. **正确性判定完全复用策略产出的 RouteOutcome**，不在孪生里另起一套。
   否则同一个任务在批量仿真与孪生里会得出不同的对错，两套数字无法对话。
   代价是：路由发生在**到达时刻**而非服务时刻，二次路由才会重新调用策略。

3. **必须能退化**：把容量设成无穷（`tier='unlimited'`），孪生必须
   **逐位退化**为 `engine.run_policy` 的结果。这是本模块最重要的正确性锚点，
   由 `tests/test_twin.py::test_无容量约束时退化为批量仿真` 守住。

一个必须警惕的陷阱
------------------
排队超时与预算耗尽的请求**既不算花费也不算答对**。于是「把请求丢掉」在账面上
会显得极省钱 —— 这与「全用最便宜模型看起来每元答对数最高」是同一类骗局。
因此本模块**从不单独报告总花费**：凡是省钱比例，一律与服务率并列给出，
并提供「每千次成功服务的花费」这一不鼓励丢请求的口径。
"""

from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

from core.resource import ModelExecutor, TaskProfile
from .catalog import Capacity, LoadedCapacity
from .policies import Attempt, Policy, RouteOutcome, realized_correct
from .prng import Mulberry32

# 一天的时段系数（按小时），索引 = 小时。查表而非 sin —— 三角函数在
# JS 与 Python 之间同样不保证逐位一致。
DIURNAL_24: List[float] = [
    0.25, 0.15, 0.10, 0.10, 0.12, 0.20, 0.35, 0.60,   # 00-07
    0.90, 1.20, 1.50, 1.45, 1.10, 1.00, 1.35, 1.50,   # 08-15
    1.40, 1.20, 1.00, 0.90, 0.80, 0.60, 0.40, 0.30,   # 16-23
]

EV_ARRIVAL, EV_DEPART, EV_WINDOW = 0, 1, 2


@dataclass
class TwinSpec:
    """一次孪生运行的可配置参数。"""
    duration_s: float = 8 * 3600.0      # 仿真时长（默认一个工作日）
    rate_rpm: float = 150.0             # 基准到达率（请求 / 分钟）
    start_hour: int = 9                 # 仿真起始小时（决定取哪段时段系数）
    burst_prob: float = 0.12            # 每分钟触发突发的概率
    burst_mult: float = 3.0             # 突发时的额外倍率上限（1 + U(0, m)）
    seed: int = 42                      # 到达过程随机源
    master_seed: int = 42               # 正确性判定种子，与批量仿真共用
    wait_timeout: float = 30.0          # 排队超时（秒），超时即丢弃
    budget: Optional[float] = None      # 预算（元），耗尽后请求服务不了
    overflow: str = 'wait'              # 'wait' 原地排队 | 'reroute' 容量感知二次路由
    cost_weight: float = 0.2
    sample_interval: float = 30.0       # 时间序列采样间隔（秒）

    # ---- 供强化学习环境驱动（tokenrouter/twin_env.py）----
    # router: (task, admissible_ids, info) -> executor_id。给定时**绕过策略**，
    #         由外部智能体决策；admissible_ids 为空表示当前没有任何后端有空位。
    # on_decision: (task, executor_id, admitted, cost, correct, waited) -> None。
    #         每个到达决策后的回调，用于给智能体发奖励。
    # 两个钩子都只在 RL 场景下使用，默认 None 时行为与批量策略版完全一致。
    router: Optional[Any] = None
    on_decision: Optional[Any] = None


@dataclass
class _Pending:
    """一个已进入系统、等待或正在服务的请求。

    路由结果在**到达时刻**就已确定（成本 / 对错 / 承载模型），
    这样无容量约束时孪生与批量仿真面对的是同一份结果。
    """
    task: TaskProfile
    executor_id: str
    cost: float
    correct: bool
    latency: float
    enqueued_at: float
    idx: int = -1            # 对应 tasks 的下标，用于把结局回传给外部智能体


class _Backend:
    """一个有容量约束的部署实例。"""

    def __init__(self, ex: ModelExecutor, cap: Capacity, window_seconds: float,
                 prefill_efficiency: float):
        self.ex = ex
        self.cap = cap
        self.window_seconds = window_seconds
        self.prefill_efficiency = prefill_efficiency
        self.in_flight = 0
        self.win_tokens = 0.0
        self.win_reqs = 0
        self.queue: Deque[_Pending] = deque()
        self.served = 0
        self.correct = 0
        self.cost = 0.0
        self.tokens = 0.0
        self.timeouts = 0
        self.reroutes_in = 0
        self.queue_max = 0
        self.wait_sum = 0.0

    def work_tokens(self, task: TaskProfile) -> float:
        """本次调用消耗的「容量 token」。

        API 按 in+out 计（与计费口径一致）；自建按 GPU 实际工作量计 —— prefill
        比 decode 快 prefill_efficiency 倍，所以 input 折算后再计入。
        """
        if self.ex.kind == 'self_hosted_model':
            return task.input_tokens / max(self.prefill_efficiency, 1e-6) + task.output_tokens
        return task.input_tokens + task.output_tokens

    def service_duration(self, task: TaskProfile) -> float:
        """本次调用占用一个并发槽位的时长（秒）。"""
        if self.ex.kind == 'self_hosted_model':
            # 处理器共享近似：每槽位速度 = 部署总吞吐 ÷ 槽位数
            tps_per_slot = (self.cap.tpm_limit / self.window_seconds
                            / max(self.cap.max_concurrency, 1))
            work = task.input_tokens / max(self.prefill_efficiency, 1e-6) + task.output_tokens
            return self.cap.ttft + work / max(tps_per_slot, 1e-6)
        return self.cap.ttft + task.output_tokens / max(self.ex.output_speed, 1e-6)

    def can_admit(self, task: TaskProfile) -> bool:
        return (self.in_flight < self.cap.max_concurrency
                and self.win_tokens + self.work_tokens(task) <= self.cap.tpm_limit
                and self.win_reqs + 1 <= self.cap.rpm_limit)

    def admit(self, task: TaskProfile) -> None:
        self.in_flight += 1
        self.win_tokens += self.work_tokens(task)
        self.win_reqs += 1

    def release(self) -> None:
        self.in_flight = max(0, self.in_flight - 1)

    def reset_window(self) -> None:
        self.win_tokens = 0.0
        self.win_reqs = 0


@dataclass
class TwinResult:
    """一次孪生运行的聚合结果。

    `total_cost` 不可单独引用 —— 请连同 `serve_rate` 一起读，或直接用
    `cost_per_1k_served`。理由见模块文档末尾的陷阱说明。
    """
    policy: str = ''
    tier: str = ''
    n_arrivals: int = 0
    n_served: int = 0
    n_timeout: int = 0
    n_budget_reject: int = 0
    serve_rate: float = 0.0
    total_cost: float = 0.0
    total_tokens: float = 0.0
    correct: int = 0
    accuracy_over_served: float = 0.0
    accuracy_over_all: float = 0.0
    cost_per_1k_served: float = 0.0
    cost_per_correct: float = 0.0
    wait_p50: float = 0.0
    wait_p95: float = 0.0
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    queue_rate: float = 0.0          # 排过队的请求占比
    reroute_rate: float = 0.0        # 触发二次路由的请求占比
    budget_exhausted_at: Optional[float] = None
    per_model: Dict[str, Dict[str, float]] = field(default_factory=dict)
    samples: List[Dict[str, float]] = field(default_factory=list)


def _percentile(sorted_vals: Sequence[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = int(round((len(sorted_vals) - 1) * q))
    return float(sorted_vals[max(0, min(idx, len(sorted_vals) - 1))])


def generate_arrivals(spec: TwinSpec) -> List[float]:
    """生成按时间升序的到达时刻（秒）。

    每分钟：n = round(基准 × 时段系数 × 突发系数)，再在分钟内均匀散布。
    纯四则运算，可在浏览器里逐位复刻。
    """
    rng = Mulberry32(spec.seed)
    times: List[float] = []
    n_min = int(spec.duration_s // 60) + 1
    for m in range(n_min):
        hour = (spec.start_hour + m // 60) % 24
        base = spec.rate_rpm * DIURNAL_24[hour]
        shock = 1.0
        if rng.random() < spec.burst_prob:
            shock = 1.0 + rng.uniform(0.0, spec.burst_mult)
        # 不能用 round()：Python 是银行家舍入（round(202.5)=202），JS 的 Math.round
        # 是四舍五入（Math.round(202.5)=203）。150×1.35=202.5 会真实出现。
        # 改成 +0.5 再取整，两侧都是纯算术，逐位一致（控制台会复算同一条流）。
        n_m = int(base * shock + 0.5)
        offsets = sorted(rng.uniform(0.0, 60.0) for _ in range(n_m))
        t0 = m * 60.0
        for o in offsets:
            t = t0 + o
            if t < spec.duration_s:
                times.append(t)
    times.sort()
    return times


def _single_attempt(task: TaskProfile, ex: ModelExecutor, master_seed: int) -> Attempt:
    """「这个任务交给这个模型」的结果 —— 与 policies._make_attempt 同源。

    只在外部智能体（RL）绕开策略直接指定模型时才用到。策略路径走
    `_outcome_to_pending`，保留策略可能产生的多次调用（如 cascade）。
    """
    est = ex.estimate(task)
    ok = realized_correct(master_seed, task, ex, est.quality)
    return Attempt(executor_id=ex.id, cost=est.cost, latency=est.latency,
                   correct=ok, accepted=True)


def _outcome_to_pending(task: TaskProfile, out: RouteOutcome, now: float) -> _Pending:
    """把策略产出的 RouteOutcome 固定成一条待服务请求。"""
    if not out.attempts:
        return _Pending(task, '', 0.0, False, 0.0, now)
    return _Pending(task=task,
                    executor_id=out.attempts[-1].executor_id,
                    cost=out.cost,
                    correct=out.final_correct,
                    latency=out.latency,
                    enqueued_at=now)


class _Sim:
    """离散事件仿真的状态机。写成类而不是闭包，是为了让计数器归属清晰。"""

    def __init__(self, policy: Optional[Policy], tasks: Sequence[TaskProfile],
                 arrivals: Sequence[float], executors: Sequence[ModelExecutor],
                 capacity: LoadedCapacity, spec: TwinSpec):
        # policy 可为 None：外部智能体（RL）通过 spec.router 直接决策时不用策略
        self.policy = policy
        self.tasks = tasks
        self.arrivals = arrivals
        self.executors = list(executors)
        self.spec = spec
        self.by_id = {e.id: e for e in executors}
        self.order = [e.id for e in executors]

        self.backends: Dict[str, _Backend] = {}
        for ex in self.executors:
            cap = capacity.capacities.get(ex.id)
            if cap is None:
                raise KeyError(f'capacity.json 缺少 {ex.id} 的配额配置')
            self.backends[ex.id] = _Backend(ex, cap, capacity.window_seconds, 3.0)

        self.heap: List[Tuple[float, int, int, str, int]] = []
        self.seq = 0
        self.t = 0.0
        # 每个请求的最终结局（种类, 任务下标, 花费, 是否答对, 等待秒数）。
        # 强化学习环境靠它发奖励：决策在到达时刻做出，但结局可能晚很多才发生
        # （先排队、再放行；或者排队超时被丢掉）。没有这份日志，环境只能拿
        # 「此刻有没有空位」当奖励，智能体学到的是假的东西。
        self.settle_log: List[Tuple[str, int, float, bool, float]] = []
        self.spent = 0.0
        self.served = 0
        self.correct = 0
        self.tokens = 0.0
        self.n_timeout = 0
        self.n_budget = 0
        self.n_queued = 0
        self.n_reroute = 0
        self.waits: List[float] = []
        self.latencies: List[float] = []
        self.samples: List[Dict[str, float]] = []
        self.next_sample = 0.0
        self.budget_exhausted_at: Optional[float] = None

    def push(self, t: float, kind: int, bid: str = '', rid: int = -1) -> None:
        heapq.heappush(self.heap, (t, self.seq, kind, bid, rid))
        self.seq += 1

    def drain(self, bid: str, now: float) -> None:
        """FIFO 放行队列：先吐掉超时的，再能放几个放几个。"""
        b = self.backends[bid]
        while b.queue:
            head = b.queue[0]
            waited = now - head.enqueued_at
            if waited > self.spec.wait_timeout:
                b.queue.popleft()
                b.timeouts += 1
                self.n_timeout += 1
                self.settle_log.append(('timeout', head.idx, 0.0, False, waited))
                continue
            if not b.can_admit(head.task):
                break
            b.queue.popleft()
            self.start(bid, head, now)

    def start(self, bid: str, p: _Pending, now: float) -> None:
        b = self.backends[bid]
        if self.spec.budget is not None and self.spent + p.cost > self.spec.budget:
            self.n_budget += 1
            if self.budget_exhausted_at is None:
                self.budget_exhausted_at = now
            self.settle_log.append(('budget', p.idx, 0.0, False, now - p.enqueued_at))
            return
        b.admit(p.task)
        self.spent += p.cost
        self.served += 1
        b.served += 1
        b.cost += p.cost
        wt = b.work_tokens(p.task)
        b.tokens += wt
        self.tokens += wt
        if p.correct:
            self.correct += 1
            b.correct += 1
        wait = now - p.enqueued_at
        dur = b.service_duration(p.task)
        self.waits.append(wait)
        b.wait_sum += wait
        # 端到端延迟用孪生自己的服务时长（含该部署的首 token 延迟与自建的
        # prefill 折算），而不是 estimate.latency —— 后者是批量仿真给用户看的
        # 延迟代理，刻画的不是「占用槽位多久」。
        self.latencies.append(wait + dur)
        self.settle_log.append(('served', p.idx, p.cost, p.correct, wait))
        self.push(now + dur, EV_DEPART, bid)

    def sample(self, now: float) -> None:
        while self.next_sample <= now:
            self.samples.append({
                't': round(self.next_sample, 3),
                'cost': round(self.spent, 6),
                'served': self.served,
                'queued': sum(len(x.queue) for x in self.backends.values()),
                'in_flight': sum(x.in_flight for x in self.backends.values()),
                'dropped': self.n_timeout + self.n_budget,
            })
            self.next_sample += self.spec.sample_interval

    def _pop_event(self) -> bool:
        """弹出并处理一个事件，返回它是不是「到达」。"""
        if not self.heap:
            return False
        t, _, kind, bid, rid = heapq.heappop(self.heap)
        now = t
        self.t = now
        if kind == EV_WINDOW:
            for b in self.backends.values():
                b.reset_window()
            for b_id in self.order:
                self.drain(b_id, now)
        elif kind == EV_DEPART:
            self.backends[bid].release()
            self.drain(bid, now)
        else:  # EV_ARRIVAL
            self.on_arrival(now, rid)
        self.sample(now)
        return kind == EV_ARRIVAL

    def pump_to_arrival(self) -> Optional[int]:
        """推进到下一个到达事件为止（中间的时间窗/离场照常处理）。

        返回该到达对应的任务下标；没有更多到达时返回 None。
        强化学习环境用它来「先看一眼下一个请求，再决定动作」。
        """
        while self.heap:
            if self.heap[0][2] == EV_ARRIVAL:
                return self.heap[0][4]
            self._pop_event()
        return None

    def prime(self) -> None:
        """把到达与时间窗事件装进堆里（不开始跑）。

        `run()` 一次跑完；强化学习环境则先 prime，再用 `pump_to_arrival` /
        `_pop_event` 一步一步驱动，在到达之间插入自己的决策。
        """
        for i, t in enumerate(self.arrivals):
            self.push(t, EV_ARRIVAL, '', i)
        window_s = self.backends[self.order[0]].window_seconds
        for m in range(int(self.spec.duration_s // window_s) + 2):
            self.push((m + 1) * window_s, EV_WINDOW)

    def run(self) -> TwinResult:
        self.prime()
        while self._pop_event():
            pass
        # 最后一个到达之后，队列仍要排空、超时仍要计入
        while self.heap:
            self._pop_event()
        return self.result()

    def on_arrival(self, now: float, rid: int = -1) -> None:
        if rid < 0:
            rid = self._arrival_index(now)
        task = self.tasks[rid]
        if self.spec.router is not None:
            p = self._route_externally(task, now)
        else:
            p = self._route_by_policy(task, now)
        p.idx = rid
        self._dispatch(p, now)

    def _route_by_policy(self, task: TaskProfile, now: float) -> _Pending:
        out = self.policy.route(task, self.executors, self.spec.master_seed)
        p = _outcome_to_pending(task, out, now)

        if self.spec.overflow == 'reroute' and not self.backends[p.executor_id].can_admit(task):
            pool = [e for e in self.executors if self.backends[e.id].can_admit(task)]
            if pool:
                # 固定目标型策略（strongest / cheapest）在构造时就锁定了唯一目标，
                # 给它一个缩小的候选池会直接失效 —— 它们本来就只认一个模型，
                # 「改道」对它们没有语义。捕获后保持原决策，让它去原队列排队。
                try:
                    out2 = self.policy.route(task, pool, self.spec.master_seed)
                    p2 = _outcome_to_pending(task, out2, now)
                except (KeyError, IndexError):
                    p2 = None
                if p2 is not None and p2.executor_id in {e.id for e in pool}:
                    if p2.executor_id != p.executor_id:
                        self.n_reroute += 1
                        self.backends[p2.executor_id].reroutes_in += 1
                    p = p2
        return p

    def _route_externally(self, task: TaskProfile, now: float) -> _Pending:
        """由外部智能体决策（RL 场景）。策略被绕过，其余机制完全一致。"""
        adm = [e.id for e in self.executors if self.backends[e.id].can_admit(task)]
        info = {
            't': now,
            'duration': self.spec.duration_s,
            'budget': self.spec.budget,
            'budget_left': (None if self.spec.budget is None
                            else self.spec.budget - self.spent),
            'queued_total': sum(len(b.queue) for b in self.backends.values()),
            'in_flight_total': sum(b.in_flight for b in self.backends.values()),
            'cap_total': sum(b.cap.max_concurrency for b in self.backends.values()),
            # 按 executors 顺序的「此刻有没有空位」位图 —— 智能体必须能看到
            # 具体是哪一个后端空着，只给一个比例它学不会挑。
            'adm_bits': [self.backends[e.id].can_admit(task) for e in self.executors],
            'served': self.served,
        }
        eid = self.spec.router(task, adm, info)
        ex = self.by_id.get(eid) or self.by_id[adm[0] if adm else self.order[0]]
        att = _single_attempt(task, ex, self.spec.master_seed)
        admitted = self.backends[ex.id].can_admit(task)
        if self.spec.on_decision is not None:
            self.spec.on_decision(task, ex.id, admitted, att.cost, att.correct, info)
        return _Pending(task=task, executor_id=ex.id, cost=att.cost,
                        correct=att.correct, latency=att.latency, enqueued_at=now)

    def _dispatch(self, p: _Pending, now: float) -> None:
        b = self.backends[p.executor_id]
        if b.can_admit(p.task):
            self.start(p.executor_id, p, now)
        else:
            b.queue.append(p)
            self.n_queued += 1
            b.queue_max = max(b.queue_max, len(b.queue))

    def _arrival_index(self, now: float) -> int:
        """到达事件按入堆顺序取任务：与 arrivals 一一对应。"""
        i = getattr(self, '_aidx', 0)
        self._aidx = i + 1
        return i

    def result(self) -> TwinResult:
        n = len(self.tasks)
        ws = sorted(self.waits)
        ls = sorted(self.latencies)
        r = TwinResult(
            policy=(self.policy.name if self.policy else 'external_router'),
            tier='',
            n_arrivals=n,
            n_served=self.served,
            n_timeout=self.n_timeout,
            n_budget_reject=self.n_budget,
            serve_rate=round(self.served / n, 4) if n else 0.0,
            total_cost=round(self.spent, 6),
            total_tokens=round(self.tokens, 2),
            correct=self.correct,
            accuracy_over_served=round(self.correct / self.served, 4) if self.served else 0.0,
            accuracy_over_all=round(self.correct / n, 4) if n else 0.0,
            cost_per_1k_served=round(self.spent / self.served * 1000.0, 4) if self.served else 0.0,
            cost_per_correct=round(self.spent / self.correct, 6) if self.correct else 0.0,
            wait_p50=round(_percentile(ws, 0.50), 3),
            wait_p95=round(_percentile(ws, 0.95), 3),
            latency_p50=round(_percentile(ls, 0.50), 3),
            latency_p95=round(_percentile(ls, 0.95), 3),
            queue_rate=round(self.n_queued / n, 4) if n else 0.0,
            reroute_rate=round(self.n_reroute / n, 4) if n else 0.0,
            budget_exhausted_at=self.budget_exhausted_at,
            samples=self.samples,
        )
        r.per_model = {
            bid: {
                'served': b.served,
                'correct': b.correct,
                'cost': round(b.cost, 6),
                'tokens': round(b.tokens, 2),
                'timeouts': b.timeouts,
                'reroutes_in': b.reroutes_in,
                'queue_max': b.queue_max,
                'avg_wait': round(b.wait_sum / b.served, 3) if b.served else 0.0,
            }
            for bid, b in self.backends.items()
        }
        return r


def run_twin(policy: Policy,
             tasks: Sequence[TaskProfile],
             arrivals: Sequence[float],
             executors: Sequence[ModelExecutor],
             capacity: LoadedCapacity,
             spec: TwinSpec) -> TwinResult:
    """跑一次孪生。

    tasks[i] 与 arrivals[i] 一一对应：第 i 个任务在第 i 个到达时刻进入系统。
    无容量约束时处理顺序即 tasks 顺序，结果必须与 `engine.run_policy` 逐位一致。
    """
    r = _Sim(policy, tasks, arrivals, executors, capacity, spec).run()
    r.tier = capacity.tier
    return r

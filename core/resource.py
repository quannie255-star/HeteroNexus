# -*- coding: utf-8 -*-
"""
统一执行资源抽象层

把「算力节点」与「大模型」统一到同一套 Executor 接口下，使得上层
调度/路由策略可以复用同一套多目标决策逻辑。

两类 Executor 唯一的本质差异：
    算力节点  → 质量恒为 1.0。训练/批算的结果由代码决定，选错节点只会变慢，
                不会产生错误答案，且可以重跑（损失可逆）。
    大模型    → 质量是任务难度的函数。选错模型就是答错，不可回滚
                （损失不可逆），所以必须显式建模质量。

这一个差异决定了：token 侧的奖励函数必须比算力侧多一个「质量/不可逆损失」项，
这是 HeteroNexus 从 Q1（纯算力）扩展到 Q2（算力 + 模型）时真实的技术增量。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional

# 难度敏感度：difficulty 每偏离基准 0.1，logit 变化 DEFAULT_GAMMA * 0.1
# 该值是全模型唯一的自由参数，取值只影响结论的陡峭程度，不影响定性方向，
# 实验环节会做敏感性分析。
DEFAULT_GAMMA = 4.0

# benchmark 的能力基准难度：公开榜单分数视为在这一难度分布上的期望正确率
BENCHMARK_BASE_DIFFICULTY = 0.5


def _clip01(x: float) -> float:
    return min(1.0, max(0.0, x))


def cap_prob(p: float, eps: float = 1e-3) -> float:
    """把正确率限制在 (eps, 1-eps)，避免 logit 发散到无穷。"""
    return min(1.0 - eps, max(eps, p))


def cap_logit(p: float, eps: float = 1e-3) -> float:
    """正确率 → logit 尺度（教育测量中的能力参数 θ）。"""
    q = cap_prob(p, eps)
    return math.log(q / (1.0 - q))


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def quality_at_difficulty(base_prob: float, difficulty: float,
                          gamma: float = DEFAULT_GAMMA) -> float:
    """给定模型在某 benchmark 上的公开正确率，估计它在难度 d 的单个任务上的正确率。

    采用难度校准的 logistic 模型（简化 IRT / Rasch 形式）::

        logit(p) = logit(base_prob) + gamma * (base_difficulty - difficulty)

    关键性质：**当 difficulty == BENCHMARK_BASE_DIFFICULTY 时，p 恰好等于公开
    benchmark 分数**。这保证了仿真不是凭空造数 —— 在难度分布均值为 0.5 的评测集上，
    仿真的期望正确率可以被还原成公开榜单的实际数字。这是本模型可校准性的基础。
    """
    base = cap_logit(base_prob)
    return _clip01(_sigmoid(base + gamma * (BENCHMARK_BASE_DIFFICULTY - difficulty)))


@dataclass(frozen=True)
class TaskProfile:
    """统一任务画像 —— 两个场景共用的输入表示（未用到的维度填 0）。"""
    task_id: int

    # 模型侧语义
    difficulty: float = 0.5        # 0~1，越接近 1 越需要强模型
    domain: str = 'general'        # general / reasoning / coding / knowledge
    input_tokens: int = 0
    output_tokens: int = 0

    # 算力侧语义
    cpu_req: int = 0
    mem_req: int = 0
    gpu_req: int = 0
    duration: float = 0.0          # 预估运行时长（秒）

    task_type: str = 'generic'


@dataclass(frozen=True)
class Estimate:
    """一次「任务 → 资源」映射的度量结果。"""
    cost: float = 0.0        # 元
    latency: float = 0.0     # 秒
    quality: float = 1.0     # 0~1 的正确率/成功率
    feasible: bool = True
    reason: str = ''


@dataclass
class Executor:
    """执行资源基类。"""
    id: str
    name: str
    kind: str                                   # compute | api_model | self_hosted_model
    tags: list = field(default_factory=list)

    def estimate(self, task: TaskProfile) -> Estimate:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f'<Executor {self.id} ({self.kind})>'


@dataclass
class ComputeExecutor(Executor):
    """算力节点：本地 CPU/GPU 或云 GPU。

    质量恒为 1.0 —— 训练作业在哪个节点上跑，最终精度都一样，差别只在多快跑完。
    """

    cpu_total: int = 0
    mem_total: int = 0
    gpu_total: int = 0
    gpu_hourly_price: float = 0.0    # 元 / 卡·小时（自建填折旧，云填租价）
    cpu_hourly_price: float = 0.0    # 元 / 核·小时

    def __post_init__(self):
        if self.kind != 'compute':
            self.kind = 'compute'

    def estimate(self, task: TaskProfile) -> Estimate:
        if self.gpu_total < task.gpu_req or self.cpu_total < task.cpu_req \
                or self.mem_total < task.mem_req:
            return Estimate(feasible=False, reason='资源规格不足')
        hours = max(task.duration, 0.0) / 3600.0
        cost = hours * (task.gpu_req * self.gpu_hourly_price
                        + task.cpu_req * self.cpu_hourly_price)
        return Estimate(cost=cost, latency=task.duration, quality=1.0)


@dataclass
class ModelExecutor(Executor):
    """大模型执行器：按 token 计费的 API，或自建部署的开源模型。

    两者的差别只在 成本模型：
        api_model         → 直接有 price_in / price_out（元 / 百万 token）
        self_hosted_model → 由 GPU 租金与吞吐折算成「每 token 成本」
    后者正是把算力侧与 token 侧缝合起来的地方：自建模型本质上是
    「用 GPU 卡时换 token」，所以两边可以在同一个成本维度上比较。
    """

    capability: Dict[str, float] = field(default_factory=dict)   # mmlu/math/humaneval/gsm8k
    price_in: float = 0.0          # 元 / 百万 input token
    price_out: float = 0.0         # 元 / 百万 output token
    output_speed: float = 1000.0   # tokens / 秒
    first_token_latency: float = 0.4
    gamma: float = DEFAULT_GAMMA

    # 自建形态的成本参数（kind == 'self_hosted_model' 时使用）
    cost_per_output_token: Optional[float] = None   # 元 / token（由 GPU 租金折算）
    prefill_efficiency: float = 3.0                 # prefill 吞吐相对 decode 的倍数

    def ability(self, domain: str) -> float:
        """取该模型在指定能力域上的公开正确率。

        general 域没有对应的单项 benchmark，用所有已知能力的**均值**代表综合水平；
        其余域映射到同性质的公开榜单（知识→MMLU，推理→MATH，代码→HumanEval）。
        """
        if not self.capability:
            return 0.5
        col = {
            'knowledge': 'mmlu',
            'reasoning': 'math',
            'coding': 'humaneval',
        }.get(domain)
        if col is None or col not in self.capability:
            return sum(self.capability.values()) / len(self.capability)
        return self.capability[col]

    def _cost(self, task: TaskProfile) -> float:
        if self.kind == 'self_hosted_model' and self.cost_per_output_token is not None:
            per_out = self.cost_per_output_token
            per_in = per_out / max(self.prefill_efficiency, 1e-6)
        else:
            per_out = self.price_out / 1_000_000.0
            per_in = self.price_in / 1_000_000.0
        return max(task.output_tokens, 0) * per_out + max(task.input_tokens, 0) * per_in

    def estimate(self, task: TaskProfile) -> Estimate:
        quality = quality_at_difficulty(self.ability(task.domain), task.difficulty, self.gamma)
        n_out = max(task.output_tokens, 0)
        latency = self.first_token_latency + n_out / max(self.output_speed, 1e-6)
        return Estimate(cost=self._cost(task), latency=latency, quality=quality)

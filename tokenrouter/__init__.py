# -*- coding: utf-8 -*-
"""
HeteroNexus Token 侧执行体 —— 「场景外扩」新增的场景

与算力侧（simulation/）并列的两个场景之一，共用 core/ 下的同一套资源抽象与
多目标建模。本包负责回答：给定一堆请求，哪个环节该用哪个模型，能省多少，
代价是什么。

核心 evaluable claim（可被验证的主张）：
    不需要真实 GPU、不需要 API 预算、不需要真实用户流量，
    仅凭公开 benchmark 分数 + 公开定价表 + 可复现的任务流，
    就能定量给出「省钱的方案」，并標明质量代价。
"""

from .catalog import LoadedCatalog, load_catalog, api_models, self_hosted_models
from .workload import WorkloadSpec, generate_workload, summarize_workload, get_preset, PRESETS
from .policies import (
    Policy, StrongestPolicy, CheapestPolicy, DifficultyRouterPolicy,
    CascadePolicy, OraclePolicy, build_policies, RouteOutcome, Attempt,
)
from .engine import PolicyResult, run_policy, run_experiment, run_multi_seed
from .pareto import SweepPoint, sweep_cost_weight, knee_point, utilization_sensitivity

__all__ = [
    'LoadedCatalog', 'load_catalog', 'api_models', 'self_hosted_models',
    'WorkloadSpec', 'generate_workload', 'summarize_workload', 'get_preset', 'PRESETS',
    'Policy', 'StrongestPolicy', 'CheapestPolicy', 'DifficultyRouterPolicy',
    'CascadePolicy', 'OraclePolicy', 'build_policies', 'RouteOutcome', 'Attempt',
    'PolicyResult', 'run_policy', 'run_experiment', 'run_multi_seed',
    'SweepPoint', 'sweep_cost_weight', 'knee_point', 'utilization_sensitivity',
]

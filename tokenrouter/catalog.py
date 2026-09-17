# -*- coding: utf-8 -*-
"""
模型清单加载器

把 data/catalog.json 里的声明式数据转成内核层的 ModelExecutor 对象。

本模块承担两件关键事情：

1. **数据溯源**：catalog 里每个数字都带 source 字段，加载后可被实验报告直接引用，
   避免答辩时出现「这个数哪来的」无法回答的情况。

2. **自建成本折算**（核心）：把「GPU 卡时」折算成「每 token 成本」，这是缝合
   算力侧与 token 侧的那道桥::

        cost_per_output_token = gpu_count * gpu_hourly_price
                                / 3600 / (throughput_tps * utilization)

   分母里的 utilization 是**关键变量**：GPU 是按时长计费的，跑不满也要付钱。
   于是「自建模型是否比调 API 划算」不再是一个固定结论，而取决于你的负载
   能否把 GPU 喂满 —— 而「把 GPU 喂满」恰恰是算力侧调度器的职责。

   ⇒ 两个场景不是并列关系，而是耦合关系：
     算力调度决定利用率 → 利用率决定自建 token 成本 → token 成本决定路由选择。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from core.resource import ModelExecutor

DEFAULT_CATALOG = Path(__file__).resolve().parent / 'data' / 'catalog.json'

# 自建 GPU 的默认有效负载率。取值说明见 load_catalog 文档。
# 保守取 0.35：宁可低估自建的吸引力，也不要让「自建更划算」的结论建立在
# 满载假设上。真实的低并发服务场景往往远低于此。
DEFAULT_UTILIZATION = 0.35


@dataclass
class LoadedCatalog:
    executors: List[ModelExecutor] = field(default_factory=list)
    meta: Dict = field(default_factory=dict)
    gpu_specs: Dict = field(default_factory=dict)
    unit_costs: Dict[str, float] = field(default_factory=dict)  # executor_id -> 元/百万 output token


def _self_hosted_unit_cost(entry: Dict, gpu_specs: Dict, utilization: float,
                           prefill_efficiency: float) -> float:
    """折算自建模型的 output token 单位成本，返回 元 / token。"""
    gpu_id = entry['gpu']
    spec = gpu_specs.get(gpu_id)
    if spec is None:
        raise KeyError(f'catalog 缺少 GPU 规格: {gpu_id}')
    hourly = float(spec['hourly_price']) * int(entry.get('gpu_count', 1))
    effective_tps = float(entry['throughput_tps']) * max(utilization, 1e-3)
    # 元/小时 ÷ 秒/小时 ÷ token/秒 = 元/token
    return hourly / 3600.0 / effective_tps


def load_catalog(path: Optional[Path] = None,
                 utilization: float = DEFAULT_UTILIZATION,
                 gamma: float = 4.0) -> LoadedCatalog:
    """加载清单并构建执行器。

    utilization: 自建 GPU 的**有效负载率**（0~1），是自建成本模型里最敏感的参数。

        它不等于 GPU 利用率监控指标，而是「一小时内这张卡真正在产出 token 的
        时间占比」，由三件事共同决定：请求到达是否连续、批处理能否攒够 batch、
        以及是否有运维/预热窗口。取值保守一些更接近真实账单。

        默认值 0.35 表示：按理想 batch 吞吐估算的产出能力，实际只实现约三分之一。
        把它设成 1.0 会让自建看起来便宜到失真 —— 实验里务必连同敏感性分析一起看。

    gamma:       难度敏感度参数，见 core.resource.quality_at_difficulty。
    """
    p = Path(path) if path is not None else DEFAULT_CATALOG
    with open(p, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    meta = raw.get('_meta', {})
    gpu_specs = raw.get('gpu_specs', {})
    prefill_eff = float(raw.get('self_host_defaults', {}).get('prefill_efficiency', 3.0))

    executors: List[ModelExecutor] = []
    unit_costs: Dict[str, float] = {}

    for entry in raw.get('models', []):
        kind = entry['kind']
        cost_per_out_token = None
        if kind == 'self_hosted_model':
            cost_per_out_token = _self_hosted_unit_cost(entry, gpu_specs, utilization, prefill_eff)
            unit_costs[entry['id']] = cost_per_out_token * 1_000_000.0
        else:
            unit_costs[entry['id']] = float(entry['price_out'])

        ex = ModelExecutor(
            id=entry['id'],
            name=entry['name'],
            kind=kind,
            tags=[entry.get('capability_source', 'unknown')],
            capability={k: float(v) for k, v in entry.get('capability', {}).items()},
            price_in=float(entry.get('price_in', 0.0)),
            price_out=float(entry.get('price_out', 0.0)),
            output_speed=float(entry.get('output_speed', 1000.0)),
            gamma=gamma,
            cost_per_output_token=cost_per_out_token,
            prefill_efficiency=prefill_eff,
        )
        executors.append(ex)

    return LoadedCatalog(executors=executors, meta=meta,
                         gpu_specs=gpu_specs, unit_costs=unit_costs)


def api_models(cat: LoadedCatalog) -> List[ModelExecutor]:
    return [e for e in cat.executors if e.kind == 'api_model']


def self_hosted_models(cat: LoadedCatalog) -> List[ModelExecutor]:
    return [e for e in cat.executors if e.kind == 'self_hosted_model']


DEFAULT_CAPACITY = Path(__file__).resolve().parent / 'data' / 'capacity.json'


@dataclass
class Capacity:
    """一个**部署**的容量约束 —— 不是模型固有属性，而是你在这个供应商的配额档位。"""
    tpm_limit: float          # 每个窗口的 token 上限（API 按 in+out；自建按 in/eff+out）
    rpm_limit: float          # 每个窗口的请求数上限
    max_concurrency: int      # 同时在服务的请求数上限
    ttft: float               # 首 token 延迟（秒）


@dataclass
class LoadedCapacity:
    capacities: Dict[str, Capacity] = field(default_factory=dict)
    window_seconds: float = 60.0
    wait_timeout: float = 30.0
    tier: str = 'team'
    scale: Optional[float] = 1.0


def load_capacity(tier: str = 'team', path: Optional[Path] = None) -> LoadedCapacity:
    """加载配额档位。

    tier='unlimited' 时 scale=None，三个上限全部取无穷 —— 用于验证孪生在无容量约束时
    必须逐位退化为批量静态仿真（见 tests/test_twin.py 的退化不变式）。
    """
    p = Path(path) if path is not None else DEFAULT_CAPACITY
    with open(p, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    tiers = raw.get('tiers', {})
    if tier not in tiers:
        raise KeyError(f'未知配额档位: {tier}，可选 {list(tiers)}')
    scale = tiers[tier].get('scale')

    defaults = raw.get('defaults', {})
    out = LoadedCapacity(
        window_seconds=float(defaults.get('window_seconds', 60.0)),
        wait_timeout=float(defaults.get('wait_timeout', 30.0)),
        tier=tier,
        scale=scale,
    )
    inf = float('inf')
    for mid, c in raw.get('models', {}).items():
        if scale is None:
            out.capacities[mid] = Capacity(inf, inf, 10 ** 9, float(c.get('ttft', 0.4)))
        else:
            out.capacities[mid] = Capacity(
                tpm_limit=float(c['tpm_limit']) * scale,
                rpm_limit=float(c['rpm_limit']) * scale,
                # 同 twin.generate_arrivals：+0.5 取整而非 round()，保证 JS 侧一致
                max_concurrency=max(1, int(int(c['max_concurrency']) * scale + 0.5)),
                ttft=float(c.get('ttft', 0.4)),
            )
    return out

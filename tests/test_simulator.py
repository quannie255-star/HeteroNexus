# -*- coding: utf-8 -*-
"""
集群仿真引擎与基线调度算法测试。

覆盖：
1. 集群拓扑与负载生成器的不变量
2. 所有调度算法下任务的完整性与资源守恒（无泄漏、无超额分配）
3. 指标口径的合理性
4. 相对基线的性能断言（防止算法回归）
5. 与已归档实验结果的一致性（保证可复现）
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation.cluster_simulator import (  # noqa: E402
    BestFitScheduler,
    DRFScheduler,
    FIFOScheduler,
    SJFScheduler,
    SimulationEngine,
    Task,
    build_cluster,
    generate_workload,
)

SCHEDULERS = [FIFOScheduler, BestFitScheduler, DRFScheduler, SJFScheduler]
N_TASKS = 200


# --------------------------------------------------------------------------
# fixtures / helpers
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def workload():
    return generate_workload(n_tasks=N_TASKS, sim_horizon=6000)


def run_simulation(sched_cls, tasks):
    """在全新集群上运行一次仿真，返回 (nodes, metrics)。"""
    nodes = build_cluster()
    copies = [
        Task(t.task_id, t.task_type, t.submit_time,
             t.cpu_req, t.mem_req, t.gpu_req, t.duration)
        for t in tasks
    ]
    engine = SimulationEngine(nodes, copies, sched_cls(nodes))
    metrics = engine.run()
    return nodes, metrics


# --------------------------------------------------------------------------
# 1. 拓扑与负载
# --------------------------------------------------------------------------

def test_cluster_topology():
    nodes = build_cluster()
    assert len(nodes) == 6
    gpu_nodes = [n for n in nodes if n.node_type == "gpu"]
    cpu_nodes = [n for n in nodes if n.node_type == "cpu"]
    assert len(gpu_nodes) == 3 and len(cpu_nodes) == 3
    assert sum(n.gpu_total for n in nodes) == 24
    assert all(n.gpu_total == 8 for n in gpu_nodes)
    assert all(n.gpu_total == 0 for n in cpu_nodes)


def test_workload_invariants(workload):
    assert len(workload) == N_TASKS
    # 按提交时间升序排列，仿真引擎的事件推进依赖该不变量
    submits = [t.submit_time for t in workload]
    assert submits == sorted(submits)
    # 所有资源需求为正
    for t in workload:
        assert t.cpu_req > 0 and t.mem_req > 0 and t.duration > 0
        assert t.gpu_req >= 0
        # 单个任务必须能在最大的空节点上放下，否则会产生永久排队
        assert t.cpu_req <= 64 and t.mem_req <= 256 and t.gpu_req <= 8
    # 任务类型分布覆盖全部四类
    kinds = {t.task_type for t in workload}
    assert kinds == {"small_train", "medium_train", "large_train", "inference"}


def test_workload_is_deterministic():
    a = generate_workload(n_tasks=50, sim_horizon=1000)
    b = generate_workload(n_tasks=50, sim_horizon=1000)
    assert [(t.cpu_req, t.mem_req, t.gpu_req, round(t.duration, 6)) for t in a] == \
           [(t.cpu_req, t.mem_req, t.gpu_req, round(t.duration, 6)) for t in b]


# --------------------------------------------------------------------------
# 2. 仿真正确性
# --------------------------------------------------------------------------

@pytest.mark.parametrize("sched_cls", SCHEDULERS, ids=lambda c: c.name)
def test_all_tasks_complete(sched_cls, workload):
    _, m = run_simulation(sched_cls, workload)
    assert m["completed"] == m["total_tasks"] == N_TASKS


@pytest.mark.parametrize("sched_cls", SCHEDULERS, ids=lambda c: c.name)
def test_no_resource_leak(sched_cls, workload):
    """全部任务结束后，每个节点的资源占用必须归零。"""
    nodes, _ = run_simulation(sched_cls, workload)
    for n in nodes:
        assert n.cpu_used == 0, f"{n.node_id} CPU 未释放: {n.cpu_used}"
        assert n.mem_used == 0, f"{n.node_id} 内存未释放: {n.mem_used}"
        assert n.gpu_used == 0, f"{n.node_id} GPU 未释放: {n.gpu_used}"


@pytest.mark.parametrize("sched_cls", SCHEDULERS, ids=lambda c: c.name)
def test_temporal_consistency(sched_cls, workload):
    """任务的时间戳必须满足 submit <= start <= finish。"""
    nodes = build_cluster()
    copies = [
        Task(t.task_id, t.task_type, t.submit_time,
             t.cpu_req, t.mem_req, t.gpu_req, t.duration)
        for t in workload
    ]
    SimulationEngine(nodes, copies, sched_cls(nodes)).run()
    for t in copies:
        assert t.start_time is not None, f"任务 {t.task_id} 未调度"
        assert t.finish_time is not None, f"任务 {t.task_id} 未完成"
        assert t.submit_time - 1e-6 <= t.start_time <= t.finish_time


@pytest.mark.parametrize("sched_cls", SCHEDULERS, ids=lambda c: c.name)
def test_utilization_samples_respect_capacity(sched_cls, workload):
    """利用率采样值必须落在 [0, 1]，否则说明存在超额分配。"""
    _, m = run_simulation(sched_cls, workload)
    assert len(m["util_samples"]) > 10
    for s in m["util_samples"]:
        for key in ("cpu_util", "mem_util", "gpu_util"):
            assert 0.0 <= s[key] <= 1.0, f"{key}={s[key]} 超出容量"


@pytest.mark.parametrize("sched_cls", SCHEDULERS, ids=lambda c: c.name)
def test_metrics_are_ordered(sched_cls, workload):
    """中位 JCT 不应超过 P95 JCT。"""
    _, m = run_simulation(sched_cls, workload)
    assert m["avg_jct"] > 0
    assert m["median_jct"] <= m["p95_jct"]
    assert m["avg_waiting"] >= 0
    assert m["makespan"] > 0


# --------------------------------------------------------------------------
# 3. 相对基线的性能断言（防算法回归）
# --------------------------------------------------------------------------

def test_drf_outperforms_fifo_on_average_jct(workload):
    """DRF 相较 FIFO 应显著降低平均 JCT —— 这是 Q1 的核心实验结论。"""
    _, fifo = run_simulation(FIFOScheduler, workload)
    _, drf = run_simulation(DRFScheduler, workload)
    improvement = (fifo["avg_jct"] - drf["avg_jct"]) / fifo["avg_jct"]
    assert improvement > 0.10, f"DRF 仅降低 {improvement:.1%}，低于 10% 阈值"


def test_sjf_minimizes_median_jct(workload):
    """SJF 应取得最优的中位 JCT（短任务优先的直接结果）。"""
    results = {c.name: run_simulation(c, workload)[1] for c in SCHEDULERS}
    best = min(results.items(), key=lambda kv: kv[1]["median_jct"])
    assert best[0] == "SJF", f"中位 JCT 最优者为 {best[0]}，预期 SJF"


def test_fifo_maximizes_gpu_utilization(workload):
    """FIFO 应取得最高 GPU 利用率与最短 Makespan —— 吞吐与延迟的权衡证据。"""
    results = {c.name: run_simulation(c, workload)[1] for c in SCHEDULERS}
    best_util = max(results.items(), key=lambda kv: kv[1]["avg_gpu_util"])
    assert best_util[0] == "FIFO", f"GPU 利用率最高者为 {best_util[0]}，预期 FIFO"


# --------------------------------------------------------------------------
# 4. 与归档实验结果一致（可复现性）
# --------------------------------------------------------------------------

def test_matches_archived_results():
    archived_path = ROOT / "simulation" / "baseline_results.json"
    if not archived_path.exists():
        pytest.skip("未找到归档实验结果")
    archived = json.loads(archived_path.read_text(encoding="utf-8"))
    if "FIFO" not in archived:
        pytest.skip("归档文件结构不符")

    workload = generate_workload(n_tasks=N_TASKS, sim_horizon=6000)
    _, fifo = run_simulation(FIFOScheduler, workload)
    # random.seed(42) 固定在模块级，结果应完全可复现
    assert fifo["avg_jct"] == pytest.approx(archived["FIFO"]["avg_jct"], rel=1e-6)
    assert fifo["completed"] == archived["FIFO"]["completed"]

# -*- coding: utf-8 -*-
"""
任务特征提取模块与硬件监控模块的冒烟测试。

不依赖 psutil / pynvml 等可选硬件依赖 —— 两个模块在无硬件环境下
应自动降级为模拟采集，CI 中必须可正常运行。
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.hardware_monitor import HardwareMonitor, NodeMetrics  # noqa: E402
from modules.task_feature_extractor import (  # noqa: E402
    TaskFeatureExtractor,
    TaskRuntimeMetrics,
    TaskStaticInfo,
)


# --------------------------------------------------------------------------
# 任务特征提取
# --------------------------------------------------------------------------

def _make_task(task_id="task-001", task_type="train", framework="pytorch"):
    return TaskStaticInfo(
        task_id=task_id,
        task_type=task_type,
        framework=framework,
        model_name="resnet50",
        model_size_mb=102.0,
        declared_cpu=8,
        declared_mem_gb=32,
        declared_gpu=2,
        batch_size=64,
        dataset_size_gb=50.0,
        priority=5,
        submit_time=0.0,
    )


def test_feature_names_are_stable():
    ex = TaskFeatureExtractor()
    names = ex.get_feature_names()
    assert len(names) == len(set(names)), "特征名存在重复"
    # 项目文档中声称的 20 维特征向量
    assert len(names) == 20, f"特征维度为 {len(names)}，与文档声称的 20 维不一致"


def test_extract_returns_consistent_dimensionality():
    ex = TaskFeatureExtractor()
    ex.register_task(_make_task())
    vec = ex.extract("task-001")
    assert vec is not None
    assert len(vec.feature_vector) == len(ex.get_feature_names())
    assert all(isinstance(v, float) for v in vec.feature_vector)


def test_extract_unknown_task_returns_none():
    ex = TaskFeatureExtractor()
    assert ex.extract("does-not-exist") is None


def test_runtime_metrics_update_changes_features():
    ex = TaskFeatureExtractor()
    ex.register_task(_make_task())
    before = ex.extract("task-001").feature_vector
    ex.update_metrics("task-001", TaskRuntimeMetrics(
        cpu_util=0.85, gpu_util=0.92, mem_used_gb=28.0, gpu_mem_used_gb=30.0,
        io_read_mbps=120.0, io_write_mbps=80.0, net_rx_mbps=200.0, net_tx_mbps=150.0,
        nvlink_util=0.4, step_time_sec=0.35, elapsed_sec=600.0, progress=0.5,
    ))
    after = ex.extract("task-001").feature_vector
    assert len(after) == len(before)
    assert before != after, "注入运行时指标后特征向量应发生变化"


def test_batch_extract_covers_all_registered_tasks():
    ex = TaskFeatureExtractor()
    ids = [f"task-{i:03d}" for i in range(5)]
    for i, tid in enumerate(ids):
        ex.register_task(_make_task(tid, task_type="train" if i % 2 else "inference"))
    vecs = ex.batch_extract(ids)
    assert len(vecs) == len(ids)
    assert {v.task_id for v in vecs} == set(ids)


def test_feature_values_are_finite():
    """特征不得出现 NaN / inf —— 否则会污染后续 DRL 训练。"""
    import math

    ex = TaskFeatureExtractor()
    for i, ttype in enumerate(["train", "finetune", "inference", "preprocess"]):
        ex.register_task(_make_task(f"t{i}", task_type=ttype))
    for i in range(4):
        vec = ex.extract(f"t{i}")
        for name, value in zip(ex.get_feature_names(), vec.feature_vector):
            assert math.isfinite(value), f"特征 {name} 非有限值: {value}"


# --------------------------------------------------------------------------
# 硬件监控
# --------------------------------------------------------------------------

def test_monitor_collect_once_without_hardware():
    """无 GPU / 无 psutil 环境下，collect_once 应走模拟分支而非抛异常。"""
    mon = HardwareMonitor(node_ids=["gpu-node-1", "cpu-node-1"], sample_interval=1.0)
    snap = mon.collect_once()
    assert set(snap.keys()) == {"gpu-node-1", "cpu-node-1"}
    for node_id, m in snap.items():
        assert isinstance(m, NodeMetrics)
        assert m.node_id == node_id
        assert 0.0 <= m.cpu_util <= 1.0
        assert 0.0 <= m.mem_util <= 1.0
        assert m.mem_used_gb <= m.mem_total_gb


def test_cluster_state_aggregation():
    mon = HardwareMonitor(node_ids=["gpu-node-1", "gpu-node-2", "cpu-node-1"], sample_interval=1.0)
    mon.collect_once()
    state = mon.get_cluster_state()
    assert state.total_cpu_cores > 0
    assert state.total_mem_gb > 0
    assert 0.0 <= state.avg_cpu_util <= 1.0
    assert 0.0 <= state.avg_gpu_util <= 1.0
    assert len(state.nodes) == 3


def test_node_history_accumulates():
    mon = HardwareMonitor(node_ids=["gpu-node-1"], sample_interval=1.0)
    for _ in range(5):
        mon.collect_once()
    hist = mon.get_node_history("gpu-node-1", "cpu_util", last_n=10)
    assert len(hist) >= 5
    assert all(0.0 <= v <= 1.0 for v in hist)


def test_snapshot_export(tmp_path):
    import json

    mon = HardwareMonitor(node_ids=["gpu-node-1"], sample_interval=1.0)
    mon.collect_once()
    out = tmp_path / "snapshot.json"
    mon.export_snapshot(str(out))
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data  # 非空


def test_anomaly_detection_marks_unhealthy():
    """人为注入过载指标，应被标记为不健康。"""
    mon = HardwareMonitor(node_ids=["gpu-node-1"], sample_interval=1.0)

    def overloaded(node_id):
        return NodeMetrics(
            node_id=node_id, timestamp=0.0,
            cpu_util=0.995, cpu_cores_total=64, cpu_cores_used=63.7, load_avg_1m=120.0,
            mem_total_gb=256.0, mem_used_gb=250.0, mem_util=0.977,
            gpu_count=8, gpu_utils=[1.0] * 8, gpu_mem_total_gb=640.0, gpu_mem_used_gb=638.0,
            gpu_temps=[92.0] * 8, gpu_power_watts=[400.0] * 8,
            power_total_watts=3200.0,
        )

    mon.set_collector(overloaded)
    snap = mon.collect_once()
    m = snap["gpu-node-1"]
    assert not m.is_healthy, "过载节点应被判定为不健康"
    assert m.anomaly_score > 0.0

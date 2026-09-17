# -*- coding: utf-8 -*-
"""
硬件性能监控模块
实时采集集群节点的 CPU、GPU、内存、网络、存储等硬件指标，
为调度算法提供集群状态视图，并支持异常检测与告警。

支持的采集方式：
1. 本地采集：psutil（CPU/内存/网络）、pynvml（GPU）
2. 远程采集：Kubernetes metrics API、Prometheus、node_exporter
3. 模拟采集：用于仿真环境和无硬件时的开发测试
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
from collections import deque
import time
import threading
import json


@dataclass
class NodeMetrics:
    """节点级硬件指标快照"""
    node_id: str
    timestamp: float
    # CPU
    cpu_util: float = 0.0           # 0-1
    cpu_cores_total: int = 0
    cpu_cores_used: float = 0.0
    load_avg_1m: float = 0.0
    # 内存
    mem_total_gb: float = 0.0
    mem_used_gb: float = 0.0
    mem_util: float = 0.0           # 0-1
    swap_used_gb: float = 0.0
    # GPU
    gpu_count: int = 0
    gpu_utils: List[float] = field(default_factory=list)       # 每张卡利用率 0-1
    gpu_mem_total_gb: List[float] = field(default_factory=list)
    gpu_mem_used_gb: List[float] = field(default_factory=list)
    gpu_temps: List[float] = field(default_factory=list)       # 摄氏度
    gpu_power_watts: List[float] = field(default_factory=list)
    # 网络
    net_rx_mbps: float = 0.0
    net_tx_mbps: float = 0.0
    rdma_rx_mbps: float = 0.0
    rdma_tx_mbps: float = 0.0
    # 存储
    disk_read_mbps: float = 0.0
    disk_write_mbps: float = 0.0
    disk_util: float = 0.0         # 0-1
    # 衍生
    power_total_watts: float = 0.0
    is_healthy: bool = True
    anomaly_score: float = 0.0     # 0-1, 越高越异常


@dataclass
class ClusterState:
    """集群全局状态（聚合所有节点）"""
    timestamp: float
    nodes: Dict[str, NodeMetrics]
    # 聚合指标
    total_cpu_cores: int = 0
    used_cpu_cores: float = 0.0
    total_mem_gb: float = 0.0
    used_mem_gb: float = 0.0
    total_gpus: int = 0
    used_gpus: float = 0.0
    avg_cpu_util: float = 0.0
    avg_gpu_util: float = 0.0
    avg_mem_util: float = 0.0
    total_power_watts: float = 0.0
    unhealthy_nodes: List[str] = field(default_factory=list)
    gpu_fragmentation: float = 0.0  # GPU 资源碎片率 0-1


class HardwareMonitor:
    """硬件性能监控器"""

    def __init__(self, node_ids: List[str], sample_interval: float = 5.0,
                 history_size: int = 360):
        """
        Args:
            node_ids: 监控的节点 ID 列表
            sample_interval: 采样间隔（秒）
            history_size: 历史数据保留条数（默认 360 * 5s = 30 分钟）
        """
        self.node_ids = node_ids
        self.sample_interval = sample_interval
        self.history_size = history_size
        self.current_metrics: Dict[str, NodeMetrics] = {}
        self.history: Dict[str, deque] = {nid: deque(maxlen=history_size) for nid in node_ids}
        self._collector_fn: Optional[Callable[[str], NodeMetrics]] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        # 异常检测阈值
        self.thresholds = {
            'cpu_util_high': 0.95,
            'gpu_util_high': 0.98,
            'gpu_temp_high': 85.0,
            'mem_util_high': 0.95,
            'net_error_rate': 0.01,
        }

    def set_collector(self, fn: Callable[[str], NodeMetrics]):
        """设置指标采集函数（接入真实硬件或模拟数据）"""
        self._collector_fn = fn

    def collect_once(self) -> Dict[str, NodeMetrics]:
        """执行一次采集"""
        if self._collector_fn is None:
            # 默认使用模拟采集
            return self._mock_collect()
        results = {}
        for nid in self.node_ids:
            try:
                metrics = self._collector_fn(nid)
                self._detect_anomaly(metrics)
                results[nid] = metrics
            except Exception as e:
                # 采集失败时复用上一次数据
                if nid in self.current_metrics:
                    m = self.current_metrics[nid]
                    m.is_healthy = False
                    results[nid] = m
        with self._lock:
            self.current_metrics = results
            for nid, m in results.items():
                self.history[nid].append(m)
        return results

    def _mock_collect(self) -> Dict[str, NodeMetrics]:
        """模拟采集（用于开发测试）"""
        import random
        results = {}
        for nid in self.node_ids:
            is_gpu = 'gpu' in nid
            gpu_count = 8 if is_gpu else 0
            m = NodeMetrics(
                node_id=nid,
                timestamp=time.time(),
                cpu_util=random.uniform(0.3, 0.85),
                cpu_cores_total=64 if is_gpu else 32,
                cpu_cores_used=random.uniform(10, 50),
                mem_total_gb=256 if is_gpu else 128,
                mem_used_gb=random.uniform(40, 180),
                gpu_count=gpu_count,
                gpu_utils=[random.uniform(0.5, 0.95) for _ in range(gpu_count)],
                gpu_mem_total_gb=[80 for _ in range(gpu_count)],
                gpu_mem_used_gb=[random.uniform(20, 70) for _ in range(gpu_count)],
                gpu_temps=[random.uniform(55, 78) for _ in range(gpu_count)],
                gpu_power_watts=[random.uniform(200, 350) for _ in range(gpu_count)],
                net_rx_mbps=random.uniform(50, 500),
                net_tx_mbps=random.uniform(30, 300),
                disk_read_mbps=random.uniform(20, 200),
                disk_write_mbps=random.uniform(10, 100),
            )
            m.mem_util = m.mem_used_gb / m.mem_total_gb if m.mem_total_gb else 0
            m.power_total_watts = sum(m.gpu_power_watts) + m.cpu_cores_used * 5
            self._detect_anomaly(m)
            results[nid] = m
        with self._lock:
            self.current_metrics = results
            for nid, m in results.items():
                self.history[nid].append(m)
        return results

    def _detect_anomaly(self, metrics: NodeMetrics):
        """简单异常检测"""
        score = 0.0
        if metrics.cpu_util > self.thresholds['cpu_util_high']:
            score += 0.3
        if metrics.mem_util > self.thresholds['mem_util_high']:
            score += 0.2
        for t in metrics.gpu_temps:
            if t > self.thresholds['gpu_temp_high']:
                score += 0.3
                break
        for u in metrics.gpu_utils:
            if u > self.thresholds['gpu_util_high']:
                score += 0.1
                break
        metrics.anomaly_score = min(score, 1.0)
        metrics.is_healthy = score < 0.5

    def get_cluster_state(self) -> ClusterState:
        """获取聚合后的集群状态"""
        with self._lock:
            metrics = dict(self.current_metrics)

        state = ClusterState(timestamp=time.time(), nodes=metrics)
        for nid, m in metrics.items():
            state.total_cpu_cores += m.cpu_cores_total
            state.used_cpu_cores += m.cpu_cores_used
            state.total_mem_gb += m.mem_total_gb
            state.used_mem_gb += m.mem_used_gb
            state.total_gpus += m.gpu_count
            state.used_gpus += sum(u > 0.1 for u in m.gpu_utils)
            state.total_power_watts += m.power_total_watts
            if not m.is_healthy:
                state.unhealthy_nodes.append(nid)

        n = len(metrics) if metrics else 1
        state.avg_cpu_util = state.used_cpu_cores / state.total_cpu_cores if state.total_cpu_cores else 0
        state.avg_mem_util = state.used_mem_gb / state.total_mem_gb if state.total_mem_gb else 0
        state.avg_gpu_util = state.used_gpus / state.total_gpus if state.total_gpus else 0

        # GPU 碎片率：有多少节点的 GPU 是碎片化使用的（用了但没用满）
        fragmented = 0
        for nid, m in metrics.items():
            for u in m.gpu_utils:
                if 0.05 < u < 0.7:
                    fragmented += 1
        state.gpu_fragmentation = fragmented / state.total_gpus if state.total_gpus else 0
        return state

    def get_node_history(self, node_id: str, metric: str, last_n: int = 60) -> List[float]:
        """获取某节点某指标的历史序列"""
        if node_id not in self.history:
            return []
        records = list(self.history[node_id])[-last_n:]
        return [getattr(r, metric, 0.0) for r in records]

    def export_snapshot(self, filepath: str):
        """导出当前集群状态快照为 JSON"""
        state = self.get_cluster_state()
        data = {
            'timestamp': state.timestamp,
            'total_cpu_cores': state.total_cpu_cores,
            'used_cpu_cores': state.used_cpu_cores,
            'total_mem_gb': state.total_mem_gb,
            'used_mem_gb': state.used_mem_gb,
            'total_gpus': state.total_gpus,
            'avg_cpu_util': round(state.avg_cpu_util, 4),
            'avg_gpu_util': round(state.avg_gpu_util, 4),
            'avg_mem_util': round(state.avg_mem_util, 4),
            'gpu_fragmentation': round(state.gpu_fragmentation, 4),
            'unhealthy_nodes': state.unhealthy_nodes,
            'nodes': {
                nid: {
                    'cpu_util': m.cpu_util, 'gpu_utils': m.gpu_utils,
                    'mem_util': m.mem_util, 'power': m.power_total_watts,
                    'healthy': m.is_healthy, 'anomaly': m.anomaly_score,
                } for nid, m in state.nodes.items()
            }
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def start(self):
        """启动后台采集线程"""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """停止采集"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self):
        while self._running:
            self.collect_once()
            time.sleep(self.sample_interval)


# ============================================================
# 使用示例
# ============================================================
if __name__ == '__main__':
    nodes = ['gpu-node-1', 'gpu-node-2', 'cpu-node-1', 'cpu-node-2']
    monitor = HardwareMonitor(nodes, sample_interval=1.0)

    # 采集 3 次
    for i in range(3):
        monitor.collect_once()
        time.sleep(0.2)

    state = monitor.get_cluster_state()
    print(f'集群状态: {len(state.nodes)} 节点')
    print(f'  CPU 利用率: {state.avg_cpu_util*100:.1f}%')
    print(f'  GPU 利用率: {state.avg_gpu_util*100:.1f}%')
    print(f'  内存利用率: {state.avg_mem_util*100:.1f}%')
    print(f'  GPU 碎片率: {state.gpu_fragmentation*100:.1f}%')
    print(f'  总功耗: {state.total_power_watts:.0f}W')
    print(f'  异常节点: {state.unhealthy_nodes}')

    monitor.export_snapshot(r'C:\Users\10393\Desktop\大创\项目代码\modules\sample_snapshot.json')
    print('快照已导出')

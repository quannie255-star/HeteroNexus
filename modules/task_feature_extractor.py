# -*- coding: utf-8 -*-
"""
任务特征提取模块
从训练任务的元信息、运行时指标和历史数据中提取多维特征，
供调度算法（DRL / 启发式）进行任务分类与资源需求预测。

特征维度：
1. 静态特征：任务类型、框架、模型规模、声明资源需求
2. 动态特征：实际 CPU/GPU 利用率、内存占用、IO 频率、通信模式
3. 衍生特征：计算密度、资源均衡度、预计时长、优先级
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import time
import math


@dataclass
class TaskStaticInfo:
    """任务静态信息（提交时已知）"""
    task_id: str
    task_type: str            # 'train', 'finetune', 'inference', 'preprocess'
    framework: str            # 'pytorch', 'tensorflow', 'jax'
    model_name: str = ''
    model_size_mb: float = 0  # 模型参数量对应的权重大小
    declared_cpu: int = 0
    declared_mem_gb: float = 0
    declared_gpu: int = 0
    batch_size: int = 0
    dataset_size_gb: float = 0
    priority: int = 5         # 1-10, 越大优先级越高
    submit_time: float = field(default_factory=time.time)


@dataclass
class TaskRuntimeMetrics:
    """任务运行时指标（采样获取）"""
    cpu_util: float = 0.0        # 0-1
    gpu_util: float = 0.0        # 0-1
    mem_used_gb: float = 0.0
    gpu_mem_used_gb: float = 0.0
    io_read_mbps: float = 0.0
    io_write_mbps: float = 0.0
    net_rx_mbps: float = 0.0
    net_tx_mbps: float = 0.0
    nvlink_util: float = 0.0     # GPU 间通信利用率
    step_time_sec: float = 0.0   # 每个训练 step 的平均耗时
    elapsed_sec: float = 0.0
    progress: float = 0.0        # 0-1, 训练进度


@dataclass
class TaskFeatureVector:
    """提取后的特征向量（供调度算法使用）"""
    task_id: str
    # 静态特征
    task_type_encoded: List[float]   # one-hot: [train, finetune, inference, preprocess]
    framework_encoded: List[float]    # one-hot: [pytorch, tensorflow, jax, other]
    model_size_norm: float            # 归一化模型大小
    declared_cpu_norm: float
    declared_mem_norm: float
    declared_gpu_norm: float
    priority_norm: float
    # 动态特征
    compute_density: float            # 计算密度 = GPU利用率 / CPU利用率
    mem_intensity: float              # 内存强度 = 内存占用 / 声明内存
    io_intensity: float               # IO 强度 = (读+写) / 计算时间
    communication_intensity: float    # 通信强度 = NVLink + 网络 / 计算时间
    resource_balance: float           # 资源均衡度（CPU/GPU/内存利用的标准差倒数）
    # 预测特征
    estimated_duration: float         # 预计剩余时长
    estimated_gpu_demand: float       # 预测实际 GPU 需求
    # 综合
    feature_vector: List[float]       # 拼接后的完整向量


class TaskFeatureExtractor:
    """任务特征提取器"""

    # 归一化参数（根据集群规模设定）
    MAX_CPU = 64
    MAX_MEM_GB = 256
    MAX_GPU = 8
    MAX_MODEL_MB = 100000  # 100GB

    TASK_TYPES = ['train', 'finetune', 'inference', 'preprocess']
    FRAMEWORKS = ['pytorch', 'tensorflow', 'jax', 'other']

    def __init__(self):
        self.history: Dict[str, List[TaskRuntimeMetrics]] = {}
        self.static_info: Dict[str, TaskStaticInfo] = {}

    def register_task(self, info: TaskStaticInfo):
        """注册任务静态信息"""
        self.static_info[info.task_id] = info
        self.history[info.task_id] = []

    def update_metrics(self, task_id: str, metrics: TaskRuntimeMetrics):
        """更新任务运行时指标"""
        if task_id in self.history:
            self.history[task_id].append(metrics)

    def _one_hot(self, value: str, categories: List[str]) -> List[float]:
        return [1.0 if value == c else 0.0 for c in categories]

    def _safe_div(self, a: float, b: float) -> float:
        return a / b if b > 1e-9 else 0.0

    def extract(self, task_id: str) -> Optional[TaskFeatureVector]:
        """提取任务特征向量"""
        if task_id not in self.static_info:
            return None

        info = self.static_info[task_id]
        metrics_list = self.history.get(task_id, [])
        # 取最近若干次采样的平均值
        recent = metrics_list[-5:] if metrics_list else []
        if recent:
            avg = TaskRuntimeMetrics(
                cpu_util=sum(m.cpu_util for m in recent) / len(recent),
                gpu_util=sum(m.gpu_util for m in recent) / len(recent),
                mem_used_gb=sum(m.mem_used_gb for m in recent) / len(recent),
                gpu_mem_used_gb=sum(m.gpu_mem_used_gb for m in recent) / len(recent),
                io_read_mbps=sum(m.io_read_mbps for m in recent) / len(recent),
                io_write_mbps=sum(m.io_write_mbps for m in recent) / len(recent),
                net_rx_mbps=sum(m.net_rx_mbps for m in recent) / len(recent),
                net_tx_mbps=sum(m.net_tx_mbps for m in recent) / len(recent),
                nvlink_util=sum(m.nvlink_util for m in recent) / len(recent),
                step_time_sec=sum(m.step_time_sec for m in recent) / len(recent),
                elapsed_sec=recent[-1].elapsed_sec,
                progress=recent[-1].progress,
            )
        else:
            avg = TaskRuntimeMetrics()

        # 静态特征
        task_type_enc = self._one_hot(info.task_type, self.TASK_TYPES)
        framework_enc = self._one_hot(info.framework, self.FRAMEWORKS)
        model_size_norm = min(info.model_size_mb / self.MAX_MODEL_MB, 1.0)
        cpu_norm = min(info.declared_cpu / self.MAX_CPU, 1.0)
        mem_norm = min(info.declared_mem_gb / self.MAX_MEM_GB, 1.0)
        gpu_norm = min(info.declared_gpu / self.MAX_GPU, 1.0)
        priority_norm = info.priority / 10.0

        # 动态特征
        compute_density = self._safe_div(avg.gpu_util, avg.cpu_util + 0.01)
        compute_density = min(compute_density, 10.0) / 10.0  # 归一化到 0-1
        mem_intensity = self._safe_div(avg.mem_used_gb, max(info.declared_mem_gb, 1))
        io_intensity = (avg.io_read_mbps + avg.io_write_mbps) / 1000.0  # 归一化
        io_intensity = min(io_intensity, 1.0)
        comm_intensity = (avg.nvlink_util + (avg.net_rx_mbps + avg.net_tx_mbps) / 1000.0) / 2.0
        comm_intensity = min(comm_intensity, 1.0)

        # 资源均衡度
        utils = [avg.cpu_util, avg.gpu_util, mem_intensity]
        mean_u = sum(utils) / len(utils)
        std_u = math.sqrt(sum((u - mean_u)**2 for u in utils) / len(utils))
        resource_balance = 1.0 / (1.0 + std_u * 5)  # 越均衡越接近 1

        # 预测：基于历史 step 时间和进度
        if avg.step_time_sec > 0 and info.batch_size > 0 and avg.progress > 0:
            total_steps = info.dataset_size_gb * 1024 / (info.batch_size * 0.01)  # 估算
            remaining_steps = total_steps * (1 - avg.progress)
            estimated_duration = remaining_steps * avg.step_time_sec
        else:
            estimated_duration = 3600.0  # 默认 1 小时

        estimated_gpu_demand = info.declared_gpu * (0.8 + 0.4 * avg.gpu_util)  # 实际需求修正

        # 拼接特征向量
        feature_vector = (
            task_type_enc + framework_enc +
            [model_size_norm, cpu_norm, mem_norm, gpu_norm, priority_norm,
             compute_density, mem_intensity, io_intensity, comm_intensity,
             resource_balance, min(estimated_duration / 86400, 1.0),
             min(estimated_gpu_demand / self.MAX_GPU, 1.0)]
        )

        return TaskFeatureVector(
            task_id=task_id,
            task_type_encoded=task_type_enc,
            framework_encoded=framework_enc,
            model_size_norm=model_size_norm,
            declared_cpu_norm=cpu_norm,
            declared_mem_norm=mem_norm,
            declared_gpu_norm=gpu_norm,
            priority_norm=priority_norm,
            compute_density=compute_density,
            mem_intensity=mem_intensity,
            io_intensity=io_intensity,
            communication_intensity=comm_intensity,
            resource_balance=resource_balance,
            estimated_duration=estimated_duration,
            estimated_gpu_demand=estimated_gpu_demand,
            feature_vector=feature_vector,
        )

    def batch_extract(self, task_ids: List[str]) -> List[TaskFeatureVector]:
        """批量提取特征"""
        results = []
        for tid in task_ids:
            feat = self.extract(tid)
            if feat:
                results.append(feat)
        return results

    def get_feature_names(self) -> List[str]:
        """获取特征名称列表（用于模型可解释性）"""
        names = []
        names += [f'task_type_{t}' for t in self.TASK_TYPES]
        names += [f'framework_{f}' for f in self.FRAMEWORKS]
        names += ['model_size_norm', 'cpu_norm', 'mem_norm', 'gpu_norm', 'priority_norm',
                  'compute_density', 'mem_intensity', 'io_intensity', 'comm_intensity',
                  'resource_balance', 'estimated_duration_norm', 'estimated_gpu_demand_norm']
        return names


# ============================================================
# 使用示例
# ============================================================
if __name__ == '__main__':
    extractor = TaskFeatureExtractor()

    # 注册任务
    info = TaskStaticInfo(
        task_id='job-001',
        task_type='train',
        framework='pytorch',
        model_name='bert-large',
        model_size_mb=1300,
        declared_cpu=8,
        declared_mem_gb=32,
        declared_gpu=2,
        batch_size=32,
        dataset_size_gb=10,
        priority=7,
    )
    extractor.register_task(info)

    # 模拟运行时指标
    for i in range(3):
        extractor.update_metrics('job-001', TaskRuntimeMetrics(
            cpu_util=0.6 + i * 0.05,
            gpu_util=0.85 + i * 0.02,
            mem_used_gb=28,
            gpu_mem_used_gb=14,
            io_read_mbps=150,
            io_write_mbps=50,
            net_rx_mbps=80,
            net_tx_mbps=30,
            nvlink_util=0.4,
            step_time_sec=0.5,
            elapsed_sec=300 + i * 60,
            progress=0.1 + i * 0.05,
        ))

    feat = extractor.extract('job-001')
    print(f'任务 {feat.task_id} 特征向量维度: {len(feat.feature_vector)}')
    print(f'计算密度: {feat.compute_density:.3f}')
    print(f'资源均衡度: {feat.resource_balance:.3f}')
    print(f'预计剩余时长: {feat.estimated_duration:.0f}s')
    print(f'特征名称: {extractor.get_feature_names()}')

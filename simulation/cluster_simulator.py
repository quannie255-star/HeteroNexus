# -*- coding: utf-8 -*-
"""
异构算力协同调度平台 —— 集群仿真环境与基线调度算法（修正版）
复现 FIFO、DRF、BestFit、SJF 四种基线算法
修正：迭代式调度避免资源超额；拥塞负载使算法差异显现
"""

import json
import random
import heapq
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

# 随机源采用「自持种子」而非全局 random 状态：
# 保证 generate_workload 的结果只取决于 seed 参数，与调用顺序、其他模块的随机消耗无关。
DEFAULT_SEED = 42


# ============================================================
# 1. 集群与任务建模
# ============================================================

@dataclass
class Node:
    node_id: str
    node_type: str
    cpu_total: int
    mem_total: int
    gpu_total: int
    cpu_used: int = 0
    mem_used: int = 0
    gpu_used: int = 0

    @property
    def cpu_free(self): return self.cpu_total - self.cpu_used
    @property
    def mem_free(self): return self.mem_total - self.mem_used
    @property
    def gpu_free(self): return self.gpu_total - self.gpu_used

    def can_fit(self, cpu, mem, gpu):
        return self.cpu_free >= cpu and self.mem_free >= mem and self.gpu_free >= gpu

    def allocate(self, cpu, mem, gpu):
        self.cpu_used += cpu
        self.mem_used += mem
        self.gpu_used += gpu

    def release(self, cpu, mem, gpu):
        self.cpu_used -= cpu
        self.mem_used -= mem
        self.gpu_used -= gpu


@dataclass
class Task:
    task_id: int
    task_type: str
    submit_time: float
    cpu_req: int
    mem_req: int
    gpu_req: int
    duration: float
    start_time: Optional[float] = None
    finish_time: Optional[float] = None
    assigned_node: Optional[str] = None

    @property
    def completed(self):
        return self.finish_time is not None

    @property
    def waiting_time(self):
        return self.start_time - self.submit_time if self.start_time is not None else None

    @property
    def jct(self):
        return self.finish_time - self.submit_time if self.finish_time is not None else None


def build_cluster() -> List[Node]:
    """异构集群：3 GPU 节点(8卡) + 3 CPU 节点，资源更紧凑以产生拥塞"""
    nodes = []
    for i in range(3):
        nodes.append(Node(f'gpu-node-{i+1}', 'gpu', cpu_total=64, mem_total=256, gpu_total=8))
    for i in range(3):
        nodes.append(Node(f'cpu-node-{i+1}', 'cpu', cpu_total=32, mem_total=128, gpu_total=0))
    return nodes


def generate_workload(n_tasks: int = 200, sim_horizon: float = 6000.0,
                      seed: int = DEFAULT_SEED) -> List[Task]:
    """生成拥塞负载：任务更密集、大任务比例更高。

    使用独立的 Random(seed) 实例，结果完全可复现，不依赖全局随机状态。
    """
    rng = random.Random(seed)
    tasks = []
    type_weights = [
        ('small_train',  0.30),
        ('medium_train', 0.35),
        ('large_train',  0.20),
        ('inference',    0.15),
    ]
    types = [t for t, _ in type_weights]
    weights = [w for _, w in type_weights]

    for tid in range(n_tasks):
        ttype = rng.choices(types, weights=weights, k=1)[0]
        submit_time = rng.uniform(0, sim_horizon * 0.6)

        if ttype == 'small_train':
            cpu = rng.randint(2, 8)
            mem = rng.randint(4, 16)
            gpu = 0
            duration = rng.uniform(50, 300)
        elif ttype == 'medium_train':
            cpu = rng.randint(4, 16)
            mem = rng.randint(8, 32)
            gpu = rng.choice([1, 2, 2])
            duration = rng.uniform(200, 800)
        elif ttype == 'large_train':
            cpu = rng.randint(16, 48)
            mem = rng.randint(32, 128)
            gpu = rng.choice([4, 4, 8])
            duration = rng.uniform(600, 2000)
        else:
            cpu = rng.randint(2, 8)
            mem = rng.randint(4, 16)
            gpu = 1
            duration = rng.uniform(800, 2500)

        tasks.append(Task(tid, ttype, submit_time, cpu, mem, gpu, duration))

    tasks.sort(key=lambda t: t.submit_time)
    return tasks


# ============================================================
# 2. 调度器
# ============================================================

class Scheduler:
    name = 'base'
    def __init__(self, nodes): self.nodes = nodes
    def select_node(self, task): raise NotImplementedError
    def sort_queue(self, queue): return queue  # 子类可重写排序逻辑


class FIFOScheduler(Scheduler):
    name = 'FIFO'
    def select_node(self, task):
        for node in self.nodes:
            if node.can_fit(task.cpu_req, task.mem_req, task.gpu_req):
                return node
        return None


class BestFitScheduler(Scheduler):
    name = 'BestFit'
    def select_node(self, task):
        best, best_waste = None, float('inf')
        for node in self.nodes:
            if node.can_fit(task.cpu_req, task.mem_req, task.gpu_req):
                waste = (node.cpu_free - task.cpu_req) + (node.mem_free - task.mem_req) + (node.gpu_free - task.gpu_req) * 8
                if waste < best_waste:
                    best_waste, best = waste, node
        return best


class DRFScheduler(Scheduler):
    name = 'DRF'
    def __init__(self, nodes):
        super().__init__(nodes)
        self.total_cpu = sum(n.cpu_total for n in nodes)
        self.total_mem = sum(n.mem_total for n in nodes)
        self.total_gpu = sum(n.gpu_total for n in nodes)

    def dominant_share(self, task):
        c = task.cpu_req / self.total_cpu if self.total_cpu else 0
        m = task.mem_req / self.total_mem if self.total_mem else 0
        g = task.gpu_req / self.total_gpu if self.total_gpu else 0
        return max(c, m, g)

    def sort_queue(self, queue):
        return sorted(queue, key=lambda t: self.dominant_share(t))

    def select_node(self, task):
        best, best_waste = None, float('inf')
        for node in self.nodes:
            if node.can_fit(task.cpu_req, task.mem_req, task.gpu_req):
                waste = (node.cpu_free - task.cpu_req) + (node.mem_free - task.mem_req) + (node.gpu_free - task.gpu_req) * 8
                if waste < best_waste:
                    best_waste, best = waste, node
        return best


class SJFScheduler(Scheduler):
    """最短作业优先：按运行时长排序，短任务优先"""
    name = 'SJF'
    def sort_queue(self, queue):
        return sorted(queue, key=lambda t: t.duration)

    def select_node(self, task):
        for node in self.nodes:
            if node.can_fit(task.cpu_req, task.mem_req, task.gpu_req):
                return node
        return None


# ============================================================
# 3. 仿真引擎（修正：迭代式调度）
# ============================================================

class SimulationEngine:
    def __init__(self, nodes, tasks, scheduler):
        self.nodes = nodes
        self.tasks = tasks
        self.scheduler = scheduler
        self.current_time = 0.0
        self.waiting_queue = []
        self.running = []
        self.completed = []
        self.event_heap = []
        self.task_map = {t.task_id: t for t in tasks}
        self.util_samples = []
        self.sample_interval = 30.0
        self.next_sample = 0.0

    def run(self):
        task_idx = 0
        n = len(self.tasks)

        while task_idx < n or self.waiting_queue or self.running:
            # 1. 到达
            while task_idx < n and self.tasks[task_idx].submit_time <= self.current_time:
                self.waiting_queue.append(self.tasks[task_idx])
                task_idx += 1

            # 2. 迭代式调度：每次调度一个任务后立即分配，避免超额
            changed = True
            while changed:
                changed = False
                sorted_q = self.scheduler.sort_queue(self.waiting_queue)
                for task in sorted_q:
                    node = self.scheduler.select_node(task)
                    if node is not None and node.can_fit(task.cpu_req, task.mem_req, task.gpu_req):
                        task.start_time = self.current_time
                        task.assigned_node = node.node_id
                        node.allocate(task.cpu_req, task.mem_req, task.gpu_req)
                        self.running.append(task)
                        self.waiting_queue.remove(task)
                        heapq.heappush(self.event_heap, (self.current_time + task.duration, task.task_id))
                        changed = True
                        break  # 资源状态变了，重新调度

            # 3. 采样
            if self.current_time >= self.next_sample:
                self._sample()
                self.next_sample += self.sample_interval

            # 4. 推进时间
            next_finish = self.event_heap[0][0] if self.event_heap else float('inf')
            next_arrival = self.tasks[task_idx].submit_time if task_idx < n else float('inf')
            next_time = min(next_finish, next_arrival, self.next_sample)

            if next_time <= self.current_time:
                if not self.running and not self.waiting_queue and task_idx < n:
                    self.current_time = self.tasks[task_idx].submit_time
                    continue
                else:
                    self.current_time += 0.5
                    continue

            self.current_time = next_time

            # 5. 完成事件
            while self.event_heap and self.event_heap[0][0] <= self.current_time + 1e-9:
                _, tid = heapq.heappop(self.event_heap)
                task = self.task_map[tid]
                task.finish_time = self.current_time
                node = next(n for n in self.nodes if n.node_id == task.assigned_node)
                node.release(task.cpu_req, task.mem_req, task.gpu_req)
                self.running.remove(task)
                self.completed.append(task)

        self._sample()
        return self._metrics()

    def _sample(self):
        cu = sum(n.cpu_used for n in self.nodes)
        ct = sum(n.cpu_total for n in self.nodes)
        mu = sum(n.mem_used for n in self.nodes)
        mt = sum(n.mem_total for n in self.nodes)
        gu = sum(n.gpu_used for n in self.nodes)
        gt = sum(n.gpu_total for n in self.nodes)
        self.util_samples.append({
            'time': round(self.current_time, 2),
            'cpu_util': round(cu / ct, 4) if ct else 0,
            'mem_util': round(mu / mt, 4) if mt else 0,
            'gpu_util': round(gu / gt, 4) if gt else 0,
            'running': len(self.running),
            'waiting': len(self.waiting_queue),
        })

    def _metrics(self):
        done = [t for t in self.tasks if t.completed]
        jcts = sorted([t.jct for t in done])
        waits = [t.waiting_time for t in done if t.waiting_time is not None]
        cpu_u = [s['cpu_util'] for s in self.util_samples]
        gpu_u = [s['gpu_util'] for s in self.util_samples]
        mem_u = [s['mem_util'] for s in self.util_samples]
        makespan = max(t.finish_time for t in done) if done else 0

        by_type = defaultdict(list)
        for t in done:
            by_type[t.task_type].append(t.jct)

        return {
            'scheduler': self.scheduler.name,
            'total_tasks': len(self.tasks),
            'completed': len(done),
            'avg_jct': round(sum(jcts)/len(jcts), 1) if jcts else 0,
            'median_jct': round(jcts[len(jcts)//2], 1) if jcts else 0,
            'p95_jct': round(jcts[int(len(jcts)*0.95)], 1) if jcts else 0,
            'avg_waiting': round(sum(waits)/len(waits), 1) if waits else 0,
            'makespan': round(makespan, 1),
            'avg_cpu_util': round(sum(cpu_u)/len(cpu_u)*100, 1) if cpu_u else 0,
            'avg_gpu_util': round(sum(gpu_u)/len(gpu_u)*100, 1) if gpu_u else 0,
            'avg_mem_util': round(sum(mem_u)/len(mem_u)*100, 1) if mem_u else 0,
            'jct_by_type': {k: round(sum(v)/len(v), 1) for k, v in by_type.items()},
            'util_samples': self.util_samples,
            'all_jcts': jcts,
        }


# ============================================================
# 4. 主实验
# ============================================================

def run_experiment(n_tasks: int = 200, seed: int = DEFAULT_SEED):
    print('=' * 60)
    print('异构算力协同调度平台 —— 基线算法对比实验')
    print(f'（任务数={n_tasks}，随机种子={seed}，结果可完全复现）')
    print('=' * 60)

    tasks = generate_workload(n_tasks=n_tasks, sim_horizon=6000, seed=seed)
    print(f'\n任务数: {len(tasks)}')
    tc = defaultdict(int)
    for t in tasks:
        tc[t.task_type] += 1
    for k, v in tc.items():
        print(f'  {k}: {v}')

    sched_classes = [FIFOScheduler, BestFitScheduler, DRFScheduler, SJFScheduler]
    all_results = {}

    for sc in sched_classes:
        nodes = build_cluster()
        sched = sc(nodes)
        task_copies = [Task(t.task_id, t.task_type, t.submit_time, t.cpu_req, t.mem_req, t.gpu_req, t.duration) for t in tasks]
        engine = SimulationEngine(nodes, task_copies, sched)
        r = engine.run()
        all_results[sched.name] = r
        print(f'\n--- {sched.name} ---')
        print(f'  完成: {r["completed"]}/{r["total_tasks"]}  平均JCT: {r["avg_jct"]}  中位JCT: {r["median_jct"]}  P95: {r["p95_jct"]}')
        print(f'  平均等待: {r["avg_waiting"]}  Makespan: {r["makespan"]}')
        print(f'  CPU利用率: {r["avg_cpu_util"]}%  GPU利用率: {r["avg_gpu_util"]}%  内存利用率: {r["avg_mem_util"]}%')

    # 保存
    save = {}
    for name, r in all_results.items():
        save[name] = {k: v for k, v in r.items()}
    out = str(Path(__file__).resolve().parent / 'baseline_results.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(save, f, ensure_ascii=False, indent=2)
    print(f'\n结果已保存: {out}')
    return all_results


if __name__ == '__main__':
    run_experiment()

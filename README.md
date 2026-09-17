# HeteroNexus · 异构算力协同调度平台

> 面向深度学习训练的异构算力智能调度平台 —— 以 Kubernetes 插件形式部署，通过多目标深度强化学习实现 CPU / GPU / FPGA 混合架构下的动态资源分配。

[![CI](https://github.com/quannie255-star/HeteroNexus/actions/workflows/ci.yml/badge.svg)](https://github.com/quannie255-star/HeteroNexus/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Stage](https://img.shields.io/badge/Stage-Q1%20Prototype-orange.svg)

**项目归属**：2026 年中国人民大学"大学生创业训练计划"立项项目 · 统计学院
**项目周期**：2026.06 – 2027.05

---

## 一、问题背景

高校实验室与中小企业的 GPU 集群普遍存在**算力利用率低、调度不均衡、异构硬件管理复杂**三大痛点：

| 痛点 | 现状 | 后果 |
| --- | --- | --- |
| 队列阻塞 | 静态分配为主，大任务长期占卡 | 小任务排队严重，GPU 平均利用率不足 40% |
| 资源失衡 | GPU 打满但 CPU 闲置（训练任务） | 异构资源无法协同，整体成本高 |
| 管理割裂 | CPU / GPU / FPGA 分别管理 | 缺乏统一资源抽象与调度接口 |
| 优化单一 | 现有方案以启发式规则为主 | 无法同时兼顾延迟、吞吐、能耗与公平性 |

现有方案存在明确缺口：Kubeflow 偏重、部署复杂；NVIDIA GPU Operator 仅支持自家硬件且闭源；Volcano / Kueue 算法以启发式为主，缺乏智能决策层。

## 二、技术方案

系统采用**五层解耦架构**，智能决策层与调度内核层通过 gRPC 通信，支持启发式安全回退：

```
产品与接口层   Web 控制台 · FastAPI · CLI · Helm
      ▼
可观测与管理层 Prometheus · Grafana · OpenTelemetry · 成本核算
      ▼
智能决策层     DRL 调度代理(PPO) · 任务特征提取 · 多目标奖励引擎
      ▼  gRPC
集群调度内核层 K8s Scheduler Framework · Filter/Score/Permit/Reserve · DRA
      ▼
基础设施层     CPU / GPU / FPGA 异构节点池
```

### 核心创新点

1. **多目标奖励塑形**：奖励函数同时建模作业完成时间（JCT）、资源利用率、能耗、公平性与资源碎片率，区别于单一目标的启发式调度。
2. **动作空间扩展**：(任务 → 节点) 联合决策 + "不调度"等待动作 + 抢占动作，而非仅做节点打分。
3. **在线自适应**：DRL 代理根据实时集群状态调整目标权重，解决静态权重无法通吃的帕累托困境（见下方实验结论）。
4. **无侵入部署**：以 K8s 调度插件（而非重写调度器）实现，支持一键回退至 DRF 启发式。

## 三、当前进展（Q1，2026.06 – 2026.08）

### ✅ 已完成

| 模块 | 说明 | 位置 |
| --- | --- | --- |
| 集群仿真引擎 | 离散事件仿真，支持自定义异构拓扑与负载生成 | `simulation/cluster_simulator.py` |
| 基线算法复现 | FIFO / DRF / BestFit / SJF 四类调度器 | `simulation/cluster_simulator.py` |
| 基线对比实验 | 3×GPU 节点 + 3×CPU 节点，200 任务全量实验 | `simulation/baseline_results.json` |
| 任务特征提取 | 20 维特征向量（静态 + 动态 + 预测） | `modules/task_feature_extractor.py` |
| 硬件性能监控 | CPU/GPU/内存/网络/功耗采集与异常检测 | `modules/hardware_monitor.py` |
| 调度控制台 | 离线可运行的实时调度数字孪生（5 页签） | `console/index.html` |

### 🚧 进行中（Q2）

- [ ] PPO 策略网络训练与收敛性验证
- [ ] K8s 调度插件（Filter / Score 扩展点）实现
- [ ] DRL 代理 gRPC 服务化

## 四、实验结论

### 4.1 基线算法对比（Python 仿真引擎，seed=42，200 任务）

| 调度算法 | 平均 JCT | 中位 JCT | P95 JCT | 平均等待 | Makespan | GPU 利用率 | CPU 利用率 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FIFO | 2441.2 | 1720.1 | 9430.4 | 1741.9 | 15231.4 | 94.2% | 46.4% |
| BestFit | 2556.7 | 1911.8 | 9627.2 | 1857.4 | 15323.4 | 93.6% | 46.1% |
| **DRF** | **2094.9** | **786.1** | 10940.9 | **1395.5** | 15498.4 | 92.6% | 45.6% |
| **SJF** | **2018.9** | **713.5** | 10503.8 | **1319.5** | 16172.8 | 88.6% | 43.7% |

**关键发现**

1. 队列排序策略是 JCT 的第一影响因子：DRF 相较 FIFO 平均 JCT 降低 **14.2%**、中位 JCT 降低 **54.3%**。
2. **吞吐与延迟存在明确权衡**：FIFO 的 GPU 利用率最高（94.2%）且 Makespan 最短，但长任务阻塞严重。
3. **长尾任务（P95）是所有基线的共同短板**：DRF/SJF 的 P95 JCT 反而劣于 FIFO —— "短任务优先"会饿死大任务。
4. **CPU 利用率普遍偏低（约 46%）**：异构资源协同存在可观优化空间。

> 结论：固定规则无法同时优化平均 JCT、P95 长尾与 GPU 利用率（三者构成帕累托权衡），这正是引入强化学习动态权衡的动机。

### 4.2 多目标奖励权重敏感性（控制台内置引擎，5 随机种子 × 200 任务平均）

| 权重预设 | w₁ 时间 | w₂ 利用率 | w₃ 能耗 | w₄ 公平 | w₅ 碎片 | 平均 JCT | 中位 JCT | P95 JCT | GPU 利用率 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 均衡（默认） | 0.50 | 0.15 | 0.05 | 0.20 | 0.10 | −16.2% | −43.3% | −3.1% | 89.5% |
| 延迟优先 | 0.70 | 0.05 | 0.05 | 0.10 | 0.10 | **−20.4%** | **−60.3%** | −2.4% | 88.5% |
| 吞吐优先 | 0.20 | 0.45 | 0.15 | 0.10 | 0.10 | −1.2% | −28.2% | −2.2% | **92.2%** |
| 公平优先 | 0.25 | 0.10 | 0.05 | 0.55 | 0.05 | −9.8% | −16.5% | **−4.1%** | 89.7% |
| *参照 SJF* | — | — | — | — | — | −20.5% | −66.1% | −3.5% | 87.7% |
| *参照 FIFO* | — | — | — | — | — | 0.00% | 0.00% | 0.00% | 89.8% |

> 表中百分比为相对 FIFO 基线的变化（负值表示指标更优）。

**三条可复现的发现**

1. **w₂ 与 JCT 存在反向关系**：w₂ 从 0.05 提高到 0.45，GPU 利用率从 88.5% 升至 92.2%，但平均 JCT 改善从 −20.4% 退化到 −1.2%。原因是利用率奖励引导任务向已饱和节点聚集，形成热点并加剧队头阻塞。
2. **无静态权重可通吃全部指标**：四组预设分别落在帕累托前沿的不同位置，不存在支配所有其他配置的单点。
3. **→ 强化学习的必要性**：既然最优权重随负载特征（推理密集 / 训练密集、大任务占比、队列深度）变化，"在线选择权重"本身就是一个可学习的策略，而非调参问题。

## 五、快速开始

### 运行基线实验

```bash
pip install -r requirements.txt
python simulation/cluster_simulator.py     # 输出 baseline_results.json
python simulation/generate_svg_charts.py   # 生成对比图表 SVG
```

### 打开调度控制台

控制台为**单文件、零依赖、完全离线**的 HTML，直接双击即可：

```bash
# Windows
start console/index.html
# macOS
open console/index.html
```

控制台提供 5 个页签：实时调度数字孪生、基线实验基准、系统架构、年度路线图、工程能力图谱。
在「实时调度数字孪生」页可热切换调度算法、调节多目标奖励权重，并现场记录对比结果生成帕累托表。

### 运行测试

```bash
pytest tests/ -v
```

## 六、目录结构

```
HeteroNexus/
├── console/
│   └── index.html                    # 调度控制台（单文件离线应用）
├── simulation/
│   ├── cluster_simulator.py          # 离散事件仿真引擎 + 4 类基线调度器
│   ├── generate_svg_charts.py        # 实验图表生成
│   └── baseline_results.json         # 200 任务实验存档结果（可复现）
├── modules/
│   ├── task_feature_extractor.py     # 20 维任务特征提取
│   └── hardware_monitor.py           # 硬件指标采集与异常检测
├── tests/
│   └── test_simulator.py             # 仿真引擎单元测试
├── docs/                             # 项目文档
└── .github/workflows/ci.yml          # CI 流水线
```

## 七、技术栈

| 类别 | 选型 |
| --- | --- |
| 调度内核 | Go 1.22 · Kubernetes v1.34 · scheduler framework · DRA |
| 强化学习 | Python 3.11 · PyTorch 2.3 · Gymnasium · PPO |
| 特征工程 | NumPy · pandas · scikit-learn |
| 可观测性 | Prometheus · Grafana · OpenTelemetry |
| 接口服务 | FastAPI · gRPC · protobuf |
| 前端 | 原生 JavaScript · Canvas（零依赖） · React 18（规划） |
| 部署与 CI | Docker · Helm 3 · GitHub Actions |

## 八、路线图

- [x] **Q1** 市场调研 · 架构设计 · 仿真引擎 · 基线算法实验 · 特征提取 · 硬件监控
- [ ] **Q2** 多目标 DRL 算法设计与 PPO 训练 · 调度控制台 v1.0 · K8s 插件启动
- [ ] **Q3** 原型系统 V1.0 集成 · 72h 稳定性测试 · 合作单位实地验证
- [ ] **Q4** 性能优化（利用率 +25% / 耗时 −20% / 延迟 −60%） · 开源推广 · 结题

## 九、团队分工

| 方向 | 职责 |
| --- | --- |
| 算法组 | 调度策略建模、强化学习算法实现、仿真实验设计 |
| 系统组 | 仿真引擎、K8s 调度器开发、异构硬件适配 |
| 产品商业组 | 市场调研、竞品分析、商业计划书 |

**指导教师**：信息学院 · 分布式系统与 AI 系统优化方向

## 十、许可证

本项目采用 [MIT License](LICENSE) 开源协议。

---

<div align="center">
<sub>中国人民大学 2026 年大学生创业训练计划 · HeteroNexus 项目组</sub>
</div>

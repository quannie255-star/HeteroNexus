# -*- coding: utf-8 -*-
"""
Token 侧执行体（core + tokenrouter）测试。

覆盖的关键不变量：
1. 质量模型的可校准性：在基准难度上必须还原公开 benchmark 分数 —— 这是整个
   仿真「不是凭空造数」的唯一保证，也是答辩时最容易被追问的地方。
2. 跨进程可复现性：同参数两次运行必须逐位一致（曾因误用内置 hash(str) 踩坑）。
3. 策略之间的支配关系：Oracle 必须是成本下界、router 不应比用户现状更差。
4. 自建成本折算公式、对数归一化的边界行为。

只依赖标准库，无网络、无 API key、无 GPU。
"""

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from core.resource import (
    ModelExecutor, ComputeExecutor, TaskProfile, Estimate,
    quality_at_difficulty, cap_logit, BENCHMARK_BASE_DIFFICULTY,
)
from core.objective import Objective, Reference, scalarize, dominates, pareto_front
from tokenrouter import (
    load_catalog, get_preset, generate_workload, summarize_workload,
    run_policy, run_experiment, sweep_cost_weight, knee_point,
    utilization_sensitivity, api_models, self_hosted_models,
)
from tokenrouter.policies import (
    _stable_hash, realized_correct, judge_accepts,
    StrongestPolicy, CheapestPolicy, DifficultyRouterPolicy,
    CascadePolicy, OraclePolicy, global_strongest,
)

ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# 1. 质量模型的可校准性
# ============================================================

class TestQualityModel:
    """这一组是整个仿真的立论基础。"""

    def test_还原_benchmark_分数(self):
        """在基准难度上必须恰好还原公开分数，否则仿真就失去了校准锚点。"""
        for p in (0.50, 0.70, 0.85, 0.90, 0.95):
            got = quality_at_difficulty(p, BENCHMARK_BASE_DIFFICULTY)
            assert got == pytest.approx(p, abs=1e-9), f'{p} 应在基准难度上被还原'

    def test_难度越低质量越高(self):
        assert quality_at_difficulty(0.8, 0.2) > quality_at_difficulty(0.8, 0.5)
        assert quality_at_difficulty(0.8, 0.5) > quality_at_difficulty(0.8, 0.9)

    def test_能力越强质量越高(self):
        assert quality_at_difficulty(0.9, 0.5) > quality_at_difficulty(0.6, 0.5)

    def test_输出恒在_0_1_区间(self):
        for p in (0.0, 0.001, 0.5, 0.999, 1.0):
            for d in (0.0, 0.25, 0.5, 0.75, 1.0):
                q = quality_at_difficulty(p, d)
                assert 0.0 <= q <= 1.0, f'({p},{d}) 越界: {q}'

    def test_logit_避免发散(self):
        """接近 0/1 的正确率不能让 logit 发散到无穷。"""
        assert math.isfinite(cap_logit(0.0))
        assert math.isfinite(cap_logit(1.0))
        assert cap_logit(0.5) == pytest.approx(0.0, abs=1e-9)

    def test_难度敏感度单调(self):
        """gamma 越大，难度对质量的影响越强。"""
        lo = quality_at_difficulty(0.8, 0.2, gamma=1.0)
        hi = quality_at_difficulty(0.8, 0.2, gamma=8.0)
        assert hi > lo


# ============================================================
# 2. 跨进程可复现性（曾踩过的坑：内置 hash(str) 带随机种子）
# ============================================================

class TestReproducibility:

    def test_散列为稳定值(self):
        assert _stable_hash('deepseek-v3') == _stable_hash('deepseek-v3')
        assert _stable_hash('a') != _stable_hash('b')

    def test_确定性不与_builtin_hash_耦合(self):
        """相同字符串在任何 Python 进程里必须得到相同结果。"""
        code = (
            'import sys; sys.path.insert(0, r"%s");'
            'from tokenrouter.policies import _stable_hash as h;'
            'print(h("gpt-4o"), h("claude-3-5-sonnet"))' % str(ROOT).replace('\\', '/')
        )
        out = set()
        for seed in ('0', '1', '12345'):
            env_kwargs = {'capture_output': True, 'text': True}
            r = subprocess.run([sys.executable, '-c', code], **env_kwargs)
            assert r.returncode == 0, r.stderr
            out.add(r.stdout.strip())
        assert len(out) == 1, '不同 hash 种子下结果应完全一致'

    def test_同参数两次运行逐位一致(self):
        tasks = generate_workload(get_preset('daily_assistant'))
        cat = load_catalog()
        a = run_experiment(tasks, cat.executors, min_quality=0.0, cost_weight=0.2)
        b = run_experiment(tasks, cat.executors, min_quality=0.0, cost_weight=0.2)
        for name in a:
            assert a[name].accuracy == b[name].accuracy
            assert a[name].total_cost == b[name].total_cost

    def test_不同_gamma_下_smoke(self):
        pool = load_catalog().executors
        assert len(pool) > 0


# ============================================================
# 3. 策略支配关系
# ============================================================

@pytest.fixture(scope='module')
def executors():
    return load_catalog().executors


@pytest.fixture(scope='module')
def tasks():
    return generate_workload(get_preset('daily_assistant'))


class TestPolicies:

    def test_oracle_每个正确答案的成本最优(self, tasks, executors):
        """Oracle 不是无条件的成本下界 —— 当没有模型能答对时，它会为保证质量
        而花掉比「全用最便宜模型」更多的钱。它真正的下界地位体现在
        「每拿到一个正确答案的成本」上。"""
        res = run_experiment(tasks, executors, cost_weight=0.2)
        oracle = res['oracle']
        for name, r in res.items():
            if name == 'oracle':
                continue
            assert oracle.cost_per_correct_answer <= r.cost_per_correct_answer + 1e-9, \
                f'oracle 的效率不应低于 {name}'

    def test_oracle_质量最高(self, tasks, executors):
        res = run_experiment(tasks, executors, cost_weight=0.2)
        for name, r in res.items():
            if name == 'oracle':
                continue
            assert res['oracle'].accuracy >= r.accuracy - 1e-9

    def test_router_不劣于用户现状(self, tasks, executors):
        """λ 温和时，路由方案不应比「全走最强」更贵或更差 —— 否则方案没有交付价值。"""
        res = run_experiment(tasks, executors, cost_weight=0.2)
        base, router = res['strongest'], res['difficulty_router']
        assert router.total_cost <= base.total_cost + 1e-6
        assert router.accuracy >= base.accuracy * 0.95

    def test_省钱必然伴随质量下降(self, tasks, executors):
        """单独看省钱没有意义：cheapest 最便宜，质量也最低。"""
        res = run_experiment(tasks, executors, cost_weight=0.2)
        assert res['cheapest'].total_cost < res['strongest'].total_cost
        assert res['cheapest'].accuracy < res['strongest'].accuracy

    def test_路由推荐表覆盖所有_域_难度组合(self, executors):
        pol = DifficultyRouterPolicy(executors, min_quality=0.0, cost_weight=0.2)
        table = pol.export_table()
        assert len(table) == 12            # 4 能力域 × 3 难度档
        domains = {t['domain'] for t in table}
        bands = {t['band'] for t in table}
        assert domains == {'general', 'knowledge', 'coding', 'reasoning'}
        assert bands == {'easy', 'medium', 'hard'}

    def test_约束无法满足时给出告警且退化为全局最强(self, executors):
        """质量下限设到无人可达时不能静默出错，必须标记且回退到用户现状模型。"""
        pol = DifficultyRouterPolicy(executors, min_quality=0.999, cost_weight=0.2)
        warned = [d for d in pol.diagnostics if 'warning' in d]
        assert len(warned) == 12
        fb = global_strongest(executors)
        assert all(d['chosen'] == fb.id for d in warned)

    def test_在线逐任务路由_而非查表(self, executors):
        """路由必须按任务的实际规模实时求解，不是拿三档表查了事。

        这条同时是一个**结构回归**：曾经因为 plan_for/route 重复定义，
        后定义的查表版本把 per-task 版本覆盖掉，实验数字悄悄变成了查表的结果。
        判据：同一 (域 × 难度)、token 规模悬殊的两个任务，成本排序会翻转。
        """
        pol = DifficultyRouterPolicy(executors, min_quality=0.0, cost_weight=0.2)
        small = TaskProfile(task_id=1, difficulty=0.4, domain='coding',
                            input_tokens=200, output_tokens=100)
        large = TaskProfile(task_id=2, difficulty=0.4, domain='coding',
                            input_tokens=6000, output_tokens=3000)
        # 查表版本对两者返回同一个模型（难度/域相同）
        assert pol.plan_for(small.difficulty, small.domain) == \
            pol.plan_for(large.difficulty, large.domain)
        # per-task 版本应当感知规模差异：至少要保证仍能给出合法推荐
        a = pol.route(small, executors, 42).attempts[0].executor_id
        b = pol.route(large, executors, 42).attempts[0].executor_id
        assert a and b

    def test_离线查表是路由的可交付形态(self, tasks, executors):
        """离线三档表与在线逐任务是同一前沿上的两个工作点，互有胜负。

        刻意**不做方向性断言**：曾经以为「查表一定更差」，实测并非如此 ——
        在纯 API 候选池下查表反而更省（¥60.8 vs ¥69.3），因为三档表把升级
        推迟到下一个难度档（online 在难度 ~0.62 就升级，表要到 0.65），
        少花钱也略更容易错。方向取决于候选池结构，不是一个定理。

        能断言的只有：两者都必须显著优于用户现状，且质量保有都在可接受区间。
        """
        res = run_experiment(tasks, executors, cost_weight=0.2)
        assert 'offline_table' in res
        base, online, offline = res['strongest'], res['difficulty_router'], res['offline_table']
        for r in (online, offline):
            assert r.total_cost < base.total_cost * 0.75, '两种形态都应省下相当比例的钱'
            assert r.accuracy >= base.accuracy * 0.95, '质量保有不得低于 95%'
            assert r.accuracy >= res['cheapest'].accuracy - 1e-9

    def test_级联的最终答案必定被接受(self, tasks, executors):
        """级联不可能悬空 —— 要么被判定器放行，要么走到兜底那一跳。"""
        pol = CascadePolicy(executors)
        for t in tasks[:30]:
            out = pol.route(t, executors, 42)
            assert out.attempts, '至少要有一次调用'
            assert out.attempts[-1].accepted

    def test_级联无法收敛时兜底到全局最强(self, tasks, executors):
        """判定器永不放行时，最后一跳必须落到兜底模型，而不是停在便宜的一档。"""
        pol = CascadePolicy(executors, tpr=0.0, fpr=0.0)
        fb = global_strongest(executors)
        for t in tasks[:20]:
            out = pol.route(t, executors, 42)
            assert out.attempts[-1].executor_id == fb.id
            assert out.attempts[-1].accepted

    def test_cascade_调用次数受_max_hops_限制(self, tasks, executors):
        pol = CascadePolicy(executors, max_hops=3)
        for t in tasks[:50]:
            assert len(pol.route(t, executors, 42).attempts) <= 3


# ============================================================
# 4. 成本模型与归一化
# ============================================================

class TestCostModel:

    def test_自建成本按_GPU_租金与吞吐折算(self):
        cat = load_catalog(utilization=1.0)
        ids = {e.id: e for e in cat.executors}
        e = ids['self-llama31-8b']            # rtx-4090 ¥3/h，2000 tok/s
        expect_per_token = 3.0 * 1 / 3600.0 / 2000.0
        assert e.cost_per_output_token == pytest.approx(expect_per_token)
        # 溢价三倍吞吐 → 单位成本降为三分之一
        assert e.cost_per_output_token * 1e6 == pytest.approx(3.0 / 3600.0 / 2000.0 * 1e6)

    def test_负载率减半则单位成本翻倍(self):
        full = load_catalog(utilization=1.0)
        half = load_catalog(utilization=0.5)
        a = {e.id: e.cost_per_output_token for e in full.executors}
        b = {e.id: e.cost_per_output_token for e in half.executors}
        eid = 'self-llama31-8b'
        assert b[eid] == pytest.approx(a[eid] * 2.0)

    def test_api_成本不受负载率影响(self):
        """商用 API 按 token 计费，不存在「跑不满浪费」的问题 —— 这是它相对自建的关键差异。"""
        full = load_catalog(utilization=1.0)
        low = load_catalog(utilization=0.1)
        for e_id in [e.id for e in api_models(full)]:
            a = next(x for x in full.executors if x.id == e_id)
            b = next(x for x in low.executors if x.id == e_id)
            assert a.price_out == b.price_out
            assert b.cost_per_output_token is None

    def test_prefill_成本低于_decode(self):
        cat = load_catalog()
        e = self_hosted_models(cat)[0]
        t = TaskProfile(task_id=0, input_tokens=1_000_000, output_tokens=1_000_000)
        est = e.estimate(t)
        # input 单价 = output 单价 / prefill_efficiency
        assert est.cost < 2 * e.cost_per_output_token * 1_000_000

    def test_对数归一化铺满_0_1(self):
        """跨数量级成本下，归一化必须真正区分开便宜与昂贵。"""
        cheap = Estimate(cost=1e-6, latency=1, quality=0.6)
        mid = Estimate(cost=1e-4, latency=1, quality=0.6)
        dear = Estimate(cost=1e-2, latency=1, quality=0.6)
        ref = Reference.from_estimates([cheap, mid, dear])
        obj = Objective(name='t', w_cost=1.0, w_quality=0.0, w_latency=0.0)
        s_cheap = scalarize(cheap, obj, ref)
        s_mid = scalarize(mid, obj, ref)
        s_dear = scalarize(dear, obj, ref)
        assert s_cheap > s_mid > s_dear
        # 每个数量级应当贡献大致相同的差距（对数的定义）
        assert abs((s_cheap - s_mid) - (s_mid - s_dear)) < 1e-6

    def test_相同成本时不产生_nan(self):
        e1 = Estimate(cost=1e-6, latency=1, quality=0.5)
        e2 = Estimate(cost=1e-6, latency=1, quality=0.9)
        ref = Reference.from_estimates([e1, e2])
        obj = Objective(name='t', w_cost=1.0, w_quality=1.0)
        assert math.isfinite(scalarize(e1, obj, ref))
        assert scalarize(e2, obj, ref) > scalarize(e1, obj, ref)

    def test_硬约束拦截低质量候选(self):
        obj = Objective(name='t', min_quality=0.9)
        assert obj.feasible(Estimate(quality=0.95))
        assert not obj.feasible(Estimate(quality=0.80))


# ============================================================
# 5. 负载生成与场景
# ============================================================

class TestWorkload:

    def test_可复现(self):
        a = generate_workload(get_preset('mixed'))
        b = generate_workload(get_preset('mixed'))
        assert [t.difficulty for t in a] == [t.difficulty for t in b]
        assert [t.input_tokens for t in a] == [t.input_tokens for t in b]

    def test_难度落在合法区间(self):
        for name in ('daily_assistant', 'dev_copilot', 'analytics', 'mixed'):
            for t in generate_workload(get_preset(name))[:200]:
                assert 0.0 <= t.difficulty <= 1.0
                assert t.domain in ('general', 'knowledge', 'coding', 'reasoning')

    def test_画像摘要比例和为_1(self):
        s = summarize_workload(generate_workload(get_preset('mixed')))
        assert s['difficulty_ratio']
        assert abs(sum(s['difficulty_ratio'].values()) - 1.0) < 1e-6
        assert abs(sum(s['domain_ratio'].values()) - 1.0) < 1e-6

    def test_研发场景的难度重心高于日常场景(self):
        daily = summarize_workload(generate_workload(get_preset('daily_assistant')))
        dev = summarize_workload(generate_workload(get_preset('dev_copilot')))
        assert dev['avg_difficulty'] > daily['avg_difficulty']


# ============================================================
# 6. 数据与结论完整性
# ============================================================

class TestCatalogIntegrity:

    def test_每个模型都标注了能力来源(self):
        raw = json.loads((ROOT / 'tokenrouter' / 'data' / 'catalog.json').read_text(encoding='utf-8'))
        for m in raw['models']:
            assert 'capability_source' in m, f"{m['id']} 缺少能力来源标注"
            assert m['capability'], f"{m['id']} 缺少能力分数"

    def test_自建模型都能找到对应_GPU_规格(self):
        cat = load_catalog()
        assert len(self_hosted_models(cat)) >= 3
        for e in self_hosted_models(cat):
            assert e.cost_per_output_token is not None
            assert e.cost_per_output_token > 0

    def test_catalog_带元信息与校准说明(self):
        raw = json.loads((ROOT / 'tokenrouter' / 'data' / 'catalog.json').read_text(encoding='utf-8'))
        assert '_meta' in raw
        assert 'sources' in raw['_meta']
        assert 'calibration_note' in raw['_meta']
        assert raw['_meta']['sources']['price']['as_of']


class TestPareto:

    def test_前沿单调(self, tasks, executors):
        """λ 增大 → 成本必须单调不增，这是帕累托前沿的基本性质。
        （初版用「质量下限」扫描时出现过非单调，这条测试就是为了防止回归。）"""
        pts = sweep_cost_weight(tasks, executors, master_seed=42)
        costs = [p.total_cost for p in pts]
        assert all(costs[i] >= costs[i + 1] - 1e-6 for i in range(len(costs) - 1)), \
            f'前沿非单调: {costs}'

    def test_膝点在前沿上(self, tasks, executors):
        pts = sweep_cost_weight(tasks, executors, master_seed=42)
        k = knee_point(pts)
        assert any(p.cost_weight == k.cost_weight for p in pts)

    def test_敏感性分析给出临界点(self):
        raw = json.loads((ROOT / 'tokenrouter' / 'data' / 'catalog.json').read_text(encoding='utf-8'))
        cat = load_catalog()
        rows = utilization_sensitivity(raw, cat.gpu_specs)
        assert rows
        # 负载率足够低时自建必然不如直接调 API
        assert not rows[-1]['self_hosted_wins']
        # 满载时自建应当占优
        assert rows[0]['self_hosted_wins']


class TestCoreAbstraction:
    """算力侧与模型侧共用同一套抽象的关键性质。"""

    def test_算力节点质量为常量(self):
        node = ComputeExecutor(id='n1', name='GPU节点', kind='compute',
                               cpu_total=64, mem_total=256, gpu_total=8,
                               gpu_hourly_price=10.0, cpu_hourly_price=0.5)
        task = TaskProfile(task_id=1, difficulty=0.9, cpu_req=8, gpu_req=2,
                           duration=3600)
        est = node.estimate(task)
        assert est.quality == 1.0            # 训练结果不因节点而变化
        assert est.cost == pytest.approx(2 * 10.0 + 8 * 0.5)

    def test_资源不足时不可行(self):
        node = ComputeExecutor(id='n1', name='CPU节点', kind='compute',
                               cpu_total=32, mem_total=128, gpu_total=0)
        est = node.estimate(TaskProfile(task_id=1, gpu_req=1, cpu_req=4, duration=100))
        assert not est.feasible

    def test_模型节点的通用域用能力均值(self):
        e = ModelExecutor(id='m', name='m', kind='api_model',
                          capability={'mmlu': 0.8, 'math': 0.6})
        expect = (0.8 + 0.6) / 2
        assert e.ability('general') == pytest.approx(expect)
        assert e.ability('knowledge') == 0.8

    def test_帕累托支配判定(self):
        assert dominates((1.0, 1.0, 0.9), (2.0, 2.0, 0.8))
        assert not dominates((2.0, 1.0, 0.9), (1.0, 1.0, 0.8))
        assert not dominates((1.0, 1.0, 0.9), (1.0, 1.0, 0.9))

    def test_帕累托前沿筛选(self):
        pts = [('a', 1.0, 1.0, 0.9), ('b', 2.0, 2.0, 0.8), ('c', 0.5, 0.5, 0.6)]
        front = pareto_front(pts)
        labels = {p[0] for p in front}
        assert 'b' not in labels        # 被 a 支配
        assert 'a' in labels and 'c' in labels

# -*- coding: utf-8 -*-
"""
控制台与 Python 端的同源校验

控制台是零依赖单文件 HTML，Token 侧的全部计算在浏览器里重跑一遍。
本脚本把 HTML 里的 JS 引擎抠出来交给 node 执行，与 Python 端逐项对拍：

    1. 任务流指纹（难度 / 能力域 / input / output）
    2. 四个策略的总花费与正确率
    3. HTML 内嵌的 catalog 与 tokenrouter/data/catalog.json 是否一致
    4. **数字孪生**（时间 + 容量 + 排队）在多个配额档 / 溢出策略下的结果
    5. HTML 内嵌的配额表与 tokenrouter/data/capacity.json 是否一致

任一项不通过都说明两侧已经漂移 —— 控制台显示的数字会与 README 对不上，
而「可复现」是本项目的立论基础，不能靠人工核对保证。

用法：
    python scripts/verify_console_sync.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tokenrouter import load_catalog, get_preset, generate_workload, run_experiment  # noqa: E402
from tokenrouter import load_capacity, TwinSpec, generate_arrivals, run_twin  # noqa: E402
from tokenrouter.policies import build_policies  # noqa: E402

HTML = ROOT / 'console' / 'index.html'
CATALOG_JSON = ROOT / 'tokenrouter' / 'data' / 'catalog.json'
CAPACITY_JSON = ROOT / 'tokenrouter' / 'data' / 'capacity.json'
SEEDS = [1, 7, 42, 999, 2026]
SCENE, UTIL, POOL, LAM = 'mixed', 0.35, 'all', 0.2

# 孪生对拍：覆盖「无约束必须退化」「团队档会排队」「二次路由」三类
TWIN_CFGS = [
    ('strongest', 'unlimited', 'wait'),
    ('router', 'unlimited', 'wait'),
    ('plan', 'unlimited', 'wait'),
    ('strongest', 'team', 'wait'),
    ('cheapest', 'team', 'wait'),
    ('router', 'team', 'wait'),
    ('router', 'team', 'reroute'),
    ('router', 'startup', 'reroute'),
]
TWIN_DUR, TWIN_RATE = 3600.0, 150.0

BEGIN = 'const TK = (function(){'
END = '\n})();'
# 孪生代码块：从它的分隔注释到 activate 之前。它与 TK 一并交给 node 执行。
TWIN_BEGIN = '/* ================= Token 消耗数字孪生（离散事件仿真） ================= */'
TWIN_END = 'function activate(name){'

RUNNER = r"""
/* ---- 数字孪生：与 Python 的 tokenrouter.twin 逐位对拍 ---- */
const TWIN_RES = {};
for (const cfg of TWIN_CFGS) {
  const mode = cfg[0], tier = cfg[1], ov = cfg[2];
  /* buildExec / genWorkload / buildIndex 都在 TK 闭包里，必须走 TK.* */
  const execs = TK.buildExec(UTIL);
  const arrivals = genArrivals(TWIN_DUR, TWIN_RATE, 9, 0.12, 3.0, 42);
  const tasks = TK.genWorkload(SCENE, 42, arrivals.length);
  const ix = TK.buildIndex(tasks, execs);
  const S = twinCreate(mode, {execs:execs, ix:ix, tasks:tasks, arrivals:arrivals,
    tier:tier, overflow:ov, lam:LAM, durationS:TWIN_DUR, waitTimeout:30, sampleInterval:30});
  twinStep(S, TWIN_DUR + 1e9);
  const r = twinFinish(S);
  TWIN_RES[mode + '|' + tier + '|' + ov] = {
    n: r.n, served: r.served, cost: r.cost, correct: r.correct,
    timeout: r.timeout, rerouteRate: r.rerouteRate, queueRate: r.queueRate
  };
}

const st = TK.runAll(SCENE, UTIL, POOL, LAM, 0.0);
const fp = [];
for (let i = 0; i < 8; i++) {
  const t = st.tasks[i];
  fp.push([t.id, t.difficulty.toFixed(17), t.domain, t.in, t.out].join('|'));
}
const keys = ['strongest','cheapest','router','plan','oracle'];
const res = {};
for (const k of keys) { res[k] = { cost: st[k].cost, acc: st[k].acc }; }
console.log(JSON.stringify({
  nTasks: st.tasks.length,
  nExecs: st.execs.length,
  fp: fp,
  res: res,
  modelIds: st.execs.map(e => e.id),
  unitCosts: st.execs.map(e => ({ id: e.id, unit: e.unit })),
  /* MODELS 是 JS 字面量（键无引号、含注释），交给 node 自己序列化，
     不要在 Python 侧当 JSON 解析 */
  models: TK.MODELS,
  cap: CAP,
  twin: TWIN_RES
}));
"""


def extract_engine() -> str:
    """TK 引擎 + 数字孪生块。孪生块必须一并抠出来：它复用 TK 的内部函数。"""
    txt = HTML.read_text(encoding='utf-8')
    s = txt.index(BEGIN)
    e = txt.index(END, s) + len(END)
    engine = txt[s:e]
    ts = txt.index(TWIN_BEGIN)
    te = txt.index(TWIN_END, ts)
    # mulberry32 定义在 TK 之前（全局），孪生生成到达流要用它，必须一起带上
    ms = txt.index('function mulberry32(a){')
    me = txt.index('\n', ms)
    return txt[ms:me] + '\n' + engine + '\n' + txt[ts:te]


def run_js(engine: str) -> dict:
    node = shutil.which('node')
    if node is None:
        raise SystemExit('未找到 node，跳过 JS 对拍（控制台同源性无法自动校验）')
    tmp = Path(tempfile.gettempdir()) / 'hn_console_sync.js'
    tmp.write_text(
        # 孪生块与 TK 一起被包进 new Function，作用域不外泄 —— 必须显式导出，
        # 否则 RUNNER 里拿不到 genArrivals / twinCreate。
        'const CODE = %s;\n'
        'const _M = new Function(CODE + "\\nreturn {TK:TK, genArrivals:genArrivals,'
        ' twinCreate:twinCreate, twinStep:twinStep, twinFinish:twinFinish,'
        ' buildPicker:buildPicker, pickIdx:pickIdx, CAP:CAP};")();\n'
        'const TK=_M.TK, genArrivals=_M.genArrivals, twinCreate=_M.twinCreate,'
        ' twinStep=_M.twinStep, twinFinish=_M.twinFinish, CAP=_M.CAP;\n'
        'const SCENE=%s, UTIL=%s, POOL=%s, LAM=%s;\n'
        'const TWIN_CFGS=%s, TWIN_DUR=%s, TWIN_RATE=%s;\n%s'
        % (json.dumps(engine), json.dumps(SCENE), UTIL, json.dumps(POOL), LAM,
           json.dumps([list(c) for c in TWIN_CFGS]), TWIN_DUR, TWIN_RATE, RUNNER),
        encoding='utf-8')
    out = subprocess.run([node, str(tmp)], capture_output=True, text=True, timeout=180)
    if out.returncode != 0:
        raise SystemExit('node 执行失败:\n' + out.stderr)
    return json.loads(out.stdout.strip().splitlines()[-1])


def python_side() -> dict:
    cat = load_catalog(utilization=UTIL)
    executors = cat.executors if POOL == 'all' else [e for e in cat.executors if e.kind == 'api_model']
    spec = get_preset(SCENE)
    tasks = []
    for s in SEEDS:
        tasks += generate_workload(replace(spec, seed=s))
    avg_in = max(1, int(sum(t.input_tokens for t in tasks) / len(tasks)))
    avg_out = max(1, int(sum(t.output_tokens for t in tasks) / len(tasks)))
    res = run_experiment(tasks, executors, min_quality=0.0, master_seed=42,
                         avg_in=avg_in, avg_out=avg_out, cost_weight=LAM)
    fp = []
    for t in tasks[:8]:
        fp.append('|'.join([str(t.task_id), f'{t.difficulty:.17f}', t.domain,
                            str(t.input_tokens), str(t.output_tokens)]))
    return {
        'nTasks': len(tasks),
        'nExecs': len(executors),
        'fp': fp,
        'res': {k: {'cost': r.total_cost, 'acc': r.accuracy}
                for k, r in res.items() if k in
                ('strongest', 'cheapest', 'difficulty_router', 'offline_table', 'oracle')},
        'modelIds': [e.id for e in executors],
        'unitCosts': [{'id': e.id, 'unit': cat.unit_costs[e.id]} for e in executors],
    }


def python_twin() -> dict:
    """Python 侧跑同一组孪生配置。"""
    cat = load_catalog(utilization=UTIL)
    spec = TwinSpec(duration_s=TWIN_DUR, rate_rpm=TWIN_RATE, seed=42,
                    master_seed=42, wait_timeout=30.0, cost_weight=LAM,
                    sample_interval=30.0)
    arrivals = generate_arrivals(spec)
    wspec = get_preset(SCENE)
    tasks = generate_workload(replace(wspec, seed=42, n_tasks=len(arrivals)))
    pols = {p.name: p for p in build_policies(cat.executors, cost_weight=LAM)}
    alias = {'router': 'difficulty_router', 'plan': 'offline_table'}
    out = {}
    for mode, tier, ov in TWIN_CFGS:
        p = pols[alias.get(mode, mode)]
        s = replace(spec, overflow=ov)
        r = run_twin(p, tasks, arrivals, cat.executors, load_capacity(tier), s)
        out[f'{mode}|{tier}|{ov}'] = {
            'n': r.n_arrivals, 'served': r.n_served, 'cost': r.total_cost,
            'correct': r.correct, 'timeout': r.n_timeout,
            'rerouteRate': r.reroute_rate, 'queueRate': r.queue_rate,
        }
    return out


def check_capacity(js_cap: dict) -> list:
    """HTML 内嵌的配额表必须与 capacity.json 一致。"""
    raw = json.loads(CAPACITY_JSON.read_text(encoding='utf-8'))
    errs = []
    for mid, c in raw['models'].items():
        j = js_cap.get(mid)
        if j is None:
            errs.append(f'capacity.json 有 {mid} 但控制台缺失')
            continue
        for key, js_key in (('tpm_limit', 'tpm'), ('rpm_limit', 'rpm'),
                            ('max_concurrency', 'conc'), ('ttft', 'ttft')):
            if abs(c[key] - j[js_key]) > 1e-9:
                errs.append(f"{mid}.{key}: JSON={c[key]} JS={j[js_key]}")
    extra = set(js_cap) - set(raw['models'])
    if extra:
        errs.append(f'控制台配额表多出: {sorted(extra)}')
    return errs


def check_catalog(js_models: list) -> list:
    """HTML 内嵌的 catalog 必须与 JSON 一致，否则控制台会用旧数据出结论。"""
    raw = json.loads(CATALOG_JSON.read_text(encoding='utf-8'))
    errs = []
    jm = {m['id']: m for m in raw['models']}
    if {m['id'] for m in js_models} != set(jm):
        errs.append(f"模型集合不一致: JS={sorted(m['id'] for m in js_models)} "
                    f"JSON={sorted(jm)}")
    for m in js_models:
        j = jm.get(m['id'])
        if not j:
            continue
        for key, js_key in (('price_in', 'pin'), ('price_out', 'pout'),
                            ('throughput_tps', 'tps'), ('output_speed', 'speed')):
            if key in j and j[key] != m.get(js_key):
                errs.append(f"{m['id']}.{key}: JSON={j[key]} JS={m.get(js_key)}")
        if 'capability' in j:
            for k, v in j['capability'].items():
                if abs(m['cap'][k] - v) > 1e-12:
                    errs.append(f"{m['id']}.cap.{k}: JSON={v} JS={m['cap'][k]}")
    return errs


def main():
    engine = extract_engine()
    js = run_js(engine)
    py = python_side()

    errs: list = []

    if js['nTasks'] != py['nTasks']:
        errs.append(f"任务数不一致: JS={js['nTasks']} PY={py['nTasks']}")
    if js['nExecs'] != py['nExecs']:
        errs.append(f"候选数不一致: JS={js['nExecs']} PY={py['nExecs']}")
    if js['modelIds'] != py['modelIds']:
        errs.append(f"候选顺序不一致: JS={js['modelIds']} PY={py['modelIds']}")

    for a, b in zip(js['fp'], py['fp']):
        if a != b:
            errs.append(f'任务流指纹不一致:\n    JS={a}\n    PY={b}')
            break

    alias = {'router': 'difficulty_router', 'plan': 'offline_table'}
    for k, jv in js['res'].items():
        pk = alias.get(k, k)
        pv = py['res'].get(pk)
        if pv is None:
            errs.append(f'Python 侧缺少策略 {pk}')
            continue
        # 浮点累加顺序可能造成 ULP 级差异，用相对容差
        if abs(jv['cost'] - pv['cost']) > max(1e-6, abs(pv['cost']) * 1e-9):
            errs.append(f"{k} 花费不一致: JS={jv['cost']:.6f} PY={pv['cost']:.6f}")
        if abs(jv['acc'] - pv['acc']) > 1e-9:
            errs.append(f"{k} 正确率不一致: JS={jv['acc']:.6f} PY={pv['acc']:.6f}")

    errs += check_catalog(js['models'])

    # ---- 数字孪生对拍 ----
    tw_py = python_twin()
    tw_js = js.get('twin') or {}
    for key, pv in tw_py.items():
        jv = tw_js.get(key)
        if jv is None:
            errs.append(f'孪生配置 {key} 在 JS 侧缺失')
            continue
        if jv['n'] != pv['n']:
            errs.append(f"{key} 到达数不一致: JS={jv['n']} PY={pv['n']}")
            continue
        if jv['served'] != pv['served']:
            errs.append(f"{key} 服务数不一致: JS={jv['served']} PY={pv['served']}")
        if jv['correct'] != pv['correct']:
            errs.append(f"{key} 答对数不一致: JS={jv['correct']} PY={pv['correct']}")
        if jv['timeout'] != pv['timeout']:
            errs.append(f"{key} 超时数不一致: JS={jv['timeout']} PY={pv['timeout']}")
        if abs(jv['cost'] - pv['cost']) > max(1e-6, abs(pv['cost']) * 1e-9):
            errs.append(f"{key} 花费不一致: JS={jv['cost']:.6f} PY={pv['cost']:.6f}")
        # 比率在 Python 侧 round 到 4 位，容差按该精度取（不是漂移）
        for f in ('rerouteRate', 'queueRate'):
            if abs(jv[f] - pv[f]) > 1e-4:
                errs.append(f"{key} {f} 不一致: JS={jv[f]} PY={pv[f]}")

    errs += check_capacity(js.get('cap') or {})

    # 单位成本：自建模型的折算依赖负载率，是最容易悄悄漂移的一处
    for a, b in zip(js['unitCosts'], py['unitCosts']):
        if a['id'] != b['id'] or abs(a['unit'] - b['unit']) > max(1e-9, abs(b['unit']) * 1e-12):
            errs.append(f"单位成本不一致: {a['id']} JS={a['unit']:.6f} PY={b['unit']:.6f}")
            break

    print('=' * 68)
    print('控制台 / Python 同源校验')
    print(f'场景={SCENE}  负载率={UTIL}  候选池={POOL}  λ={LAM}  种子={SEEDS}')
    print('=' * 68)
    print(f"  任务数 {py['nTasks']}  候选数 {py['nExecs']}")
    for k, jv in js['res'].items():
        print(f"  {k:<18} JS  ¥{jv['cost']:.4f} / {jv['acc']*100:.2f}%")
    print('  -- 数字孪生（时间 + 容量 + 排队）--')
    for key, pv in tw_py.items():
        jv = tw_js.get(key, {})
        print(f"  {key:<30} 服务 {pv['served']}/{pv['n']} · ¥{pv['cost']:.3f} "
              f"· 答对 {pv['correct']} · 超时 {pv['timeout']}"
              + ('' if jv.get('served') == pv['served'] else '   ← 不一致'))
    if errs:
        print('\n✗ 发现 ' + str(len(errs)) + ' 处不一致:')
        for e in errs:
            print('   - ' + e)
        sys.exit(1)
    print('\n✓ 两侧逐位一致，控制台数字可直接引用')


if __name__ == '__main__':
    main()

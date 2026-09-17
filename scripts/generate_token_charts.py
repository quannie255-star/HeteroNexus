# -*- coding: utf-8 -*-
"""
Token 侧实验图表生成（纯标准库输出 SVG，与 Q1 算力侧保持一致的技术风格）

读取 data/token_experiments.json，输出到 figures/：
    token_pareto_front.svg        成本—质量帕累托前沿（含膝点与理论最优）
    token_policy_comparison.svg   策略对比：省了多少钱 / 掉了多少质量
    token_utilization.svg         自建 GPU 负载率 vs 商用 API 的成本交叉点

用法：
    python scripts/run_token_experiments.py && python scripts/generate_token_charts.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data' / 'token_experiments.json'
OUT_DIR = ROOT / 'figures'


# ---------- 基础绘图原语 ----------

def esc(s: str) -> str:
    return (str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;'))


class Canvas:
    def __init__(self, w: int, h: int, title: str):
        self.w, self.h = w, h
        self.parts: List[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}" font-family="Microsoft YaHei, PingFang SC, sans-serif">',
            f'<rect width="{w}" height="{h}" fill="#ffffff"/>',
            f'<text x="30" y="34" font-size="18" font-weight="600" fill="#1a1a1a">{esc(title)}</text>',
        ]

    def rect(self, x, y, w, h, fill, **kw):
        self.parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
                          f'fill="{fill}" ' + ' '.join(f'{k.replace("_", "-")}="{v}"'
                                                       for k, v in kw.items()) + '/>')

    def line(self, x1, y1, x2, y2, stroke='#999', width=1.5, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ''
        self.parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                          f'stroke="{stroke}" stroke-width="{width}"{d}/>')

    def polyline(self, pts: Sequence[Tuple[float, float]], stroke='#378ADD', width=2.0, fill='none'):
        d = ' '.join(f'{x:.1f},{y:.1f}' for x, y in pts)
        self.parts.append(f'<polyline points="{d}" fill="{fill}" stroke="{stroke}" '
                          f'stroke-width="{width}"/>')

    def circle(self, x, y, r, fill, stroke='#fff', sw=1.5):
        self.parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}" '
                          f'stroke="{stroke}" stroke-width="{sw}"/>')

    def text(self, x, y, s, size=12, fill='#333', anchor='start', weight='400', rotate=None):
        tr = f' transform="rotate({rotate} {x:.1f} {y:.1f})"' if rotate else ''
        self.parts.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
                          f'text-anchor="{anchor}" font-weight="{weight}"{tr}>{esc(s)}</text>')

    def save(self, path: Path):
        self.parts.append('</svg>')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('\n'.join(self.parts), encoding='utf-8')
        return path


def nice_ticks(lo: float, hi: float, n: int = 5) -> List[float]:
    import math
    span = hi - lo
    if span <= 0:
        return [lo]
    raw = span / n
    mag = 10 ** math.floor(math.log10(raw))
    step = mag
    for m in (1, 2, 2.5, 5, 10):
        if mag * m >= raw:
            step = mag * m
            break
    start = math.floor(lo / step) * step
    out, v = [], start
    while v <= hi + 1e-9:
        out.append(round(v, 6))
        v += step
    return out


# ---------- 图 1：成本—质量帕累托前沿 ----------

def chart_pareto(payload: Dict) -> Canvas:
    pts = payload['pareto_front']
    knee = payload.get('knee_point') or {}
    pc = payload['policy_comparison']
    meta = payload['meta']

    W, H = 900, 560
    L, R, T, B = 90, 40, 80, 70
    c = Canvas(W, H, '成本—质量帕累托前沿：为每一分省下的钱标出质量代价')

    xs = [p['total_cost_yuan'] for p in pts]
    ys = [p['accuracy'] * 100 for p in pts]
    x_lo, x_hi = 0, max(xs) * 1.05
    y_lo = min(min(ys) - 5, 60)
    y_hi = 100

    def sx(v): return L + (v - x_lo) / (x_hi - x_lo) * (W - L - R)
    def sy(v): return H - B - (v - y_lo) / (y_hi - y_lo) * (H - T - B)

    # 网格与坐标轴
    for t in nice_ticks(x_lo, x_hi, 6):
        if t < 0:
            continue
        x = sx(t)
        c.line(x, T, x, H - B, stroke='#eee')
        c.text(x, H - B + 20, f'{t:.0f}', size=11, fill='#666', anchor='middle')
    for t in nice_ticks(y_lo, y_hi, 6):
        if not (y_lo <= t <= y_hi):
            continue
        y = sy(t)
        c.line(L, y, W - R, y, stroke='#eee')
        c.text(L - 10, y + 4, f'{t:.0f}%', size=11, fill='#666', anchor='end')

    c.line(L, H - B, W - R, H - B, stroke='#333', width=1.5)
    c.line(L, T, L, H - B, stroke='#333', width=1.5)
    c.text((L + W - R) / 2, H - 22, '一批任务的总花费（元，越低越好）', size=12,
           fill='#444', anchor='middle')
    c.text(28, (T + H - B) / 2, '整体正确率（%，越高越好）', size=12, fill='#444',
           anchor='middle', rotate=-90)

    # 用户现状基线（全部走最强模型）
    base_cost = pc['strongest']['total_cost_yuan']
    base_acc = pc['strongest']['accuracy'] * 100
    c.line(sx(base_cost), T, sx(base_cost), H - B, stroke='#D85A30', width=1.5, dash='6 4')
    c.line(L, sy(base_acc), W - R, sy(base_acc), stroke='#D85A30', width=1.5, dash='6 4')
    c.text(sx(base_cost) - 8, T + 18, f'用户现状：¥{base_cost:.0f} / {base_acc:.1f}%',
           size=12, fill='#993C1D', anchor='end', weight='600')

    # 理论最优 Oracle
    oc = pc['oracle']
    c.circle(sx(oc['total_cost_yuan']), sy(oc['accuracy'] * 100), 7, '#639922')
    c.text(sx(oc['total_cost_yuan']) + 12, sy(oc['accuracy'] * 100) + 4,
           f"理论最优 Oracle ¥{oc['total_cost_yuan']:.1f} / {oc['accuracy']*100:.0f}%",
           size=11, fill='#3B6D11', weight='600')

    # 前沿
    ordered = sorted(zip(xs, ys, pts), key=lambda z: z[0])
    c.polyline([(sx(x), sy(y)) for x, y, _ in ordered], stroke='#185FA5', width=2.5)
    for x, y, p in ordered:
        cx, cy = sx(x), sy(y)
        c.circle(cx, cy, 4.5, '#185FA5')
        off = -14 if p['cost_weight'] in (0.15, 0.25, 0.5, 0.1) else 18
        c.text(cx, cy + off, f"λ={p['cost_weight']}", size=10, fill='#0C447C', anchor='middle')

    # 膝点
    if knee:
        kx, ky = sx(knee['total_cost_yuan']), sy(knee['accuracy'] * 100)
        c.circle(kx, ky, 9, '#FAEEDA', stroke='#BA7517', sw=2.5)
        c.circle(kx, ky, 4, '#BA7517', stroke='none', sw=0)
        c.text(kx + 14, ky - 8,
               f"膝点 λ={knee['cost_weight']}：省 {knee['cost_saving']*100:.0f}%，"
               f"质量保有 {knee['quality_retention']*100:.0f}%",
               size=12, fill='#633806', weight='600')

    c.text(L, 56, f"场景 {meta['scenario']} · {meta['n_tasks_per_sample']} 任务/样本 "
                  f"· GPU 负载率 {meta['gpu_utilization']}", size=11, fill='#888')
    return c


# ---------- 图 2：策略对比 ----------

def chart_policies(payload: Dict) -> Canvas:
    pc = payload['policy_comparison']
    # 两种交付形态都画：只画在线版会让人误以为那就是用户拿到的东西
    order = ['strongest', 'cheapest', 'cascade', 'difficulty_router',
             'offline_table', 'oracle']
    labels = {'strongest': '全用最强\n(用户现状)', 'cheapest': '全用最便宜\n(省钱下界)',
              'cascade': '级联升级', 'difficulty_router': '路由方案\n(在线逐任务)',
              'offline_table': '路由方案\n(离线三档表)',
              'oracle': '理论最优\nOracle'}
    colors = {'strongest': '#888780', 'cheapest': '#D4537E', 'cascade': '#7F77DD',
              'difficulty_router': '#185FA5', 'offline_table': '#0F9D58',
              'oracle': '#639922'}

    W, H = 900, 520
    L, R, T, B = 90, 60, 90, 110
    c = Canvas(W, H, '策略对比：省下的钱 vs 付出的质量代价')

    max_cost = max(pc[k]['total_cost_yuan'] for k in order)
    y_hi = 100
    cw = (W - L - R) / len(order)

    def sy(v): return H - B - v / y_hi * (H - T - B)

    for t in nice_ticks(0, y_hi, 5):
        if 0 <= t <= y_hi:
            y = sy(t)
            c.line(L, y, W - R, y, stroke='#eee')
            c.text(L - 10, y + 4, f'{t:.0f}%', size=11, fill='#666', anchor='end')
    c.line(L, H - B, W - R, H - B, stroke='#333', width=1.5)
    c.text(30, (T + H - B) / 2, '正确率 / 相对花费（%）', size=12, fill='#444',
           anchor='middle', rotate=-90)

    for i, k in enumerate(order):
        r = pc[k]
        cx = L + cw * i
        bw = min(cw * 0.42, 54)
        # 相对花费柱
        rel = r['total_cost_yuan'] / max_cost * 100
        hgt = (H - T - B) * rel / 100
        c.rect(cx + cw / 2 - bw - 4, H - B - hgt, bw, hgt, colors[k], opacity='0.85')
        c.text(cx + cw / 2 - bw / 2 - 4, H - B - hgt - 8, f'¥{r["total_cost_yuan"]:.2f}',
               size=11, fill='#222', anchor='middle', weight='600')
        # 正确率柱
        ah = (H - T - B) * r['accuracy'] * 100 / 100
        c.rect(cx + cw / 2 + 4, H - B - ah, bw, ah, colors[k], opacity='0.35')
        c.text(cx + cw / 2 + bw / 2 + 4, H - B - ah - 8, f'{r["accuracy"]*100:.1f}%',
               size=11, fill='#222', anchor='middle')

        for j, line in enumerate(labels[k].split('\n')):
            c.text(cx + cw / 2, H - B + 26 + j * 15, line, size=11, fill='#333',
                   anchor='middle', weight='600')
        saving = r.get('cost_saving', 0.0)
        tag = '基线' if k == 'strongest' else f'省 {saving*100:.0f}%'
        c.text(cx + cw / 2, H - B + 60, tag, size=11, fill=colors[k], anchor='middle', weight='600')

    lx = L + 10
    c.rect(lx, T - 42, 16, 16, '#444', opacity='0.85')
    c.text(lx + 24, T - 30, '总花费', size=12, fill='#333')
    c.rect(lx + 90, T - 42, 16, 16, '#444', opacity='0.35')
    c.text(lx + 114, T - 30, '正确率', size=12, fill='#333')
    return c


# ---------- 图 3：GPU 负载率敏感性 ----------

def chart_utilization(payload: Dict) -> Canvas:
    rows = payload['utilization_sensitivity']
    if not rows:
        return Canvas(600, 200, '无自建模型，跳过负载率敏感性分析')

    W, H = 900, 500
    L, R, T, B = 100, 50, 90, 90
    c = Canvas(W, H, '自建划不划算，取决于能把 GPU 喂到多满')

    xs = [r['utilization'] * 100 for r in rows]
    ys = [r['self_cost_per_mtok'] for r in rows]
    ref = rows[0]['ref_cost_per_mtok']
    ref_name = rows[0]['ref_model']

    x_hi = 100
    y_hi = max(max(ys) * 1.1, ref * 1.2)

    def sx(v): return L + v / x_hi * (W - L - R)
    def sy(v): return H - B - v / y_hi * (H - T - B)

    for t in nice_ticks(0, x_hi, 5):
        if 0 <= t <= x_hi:
            x = sx(t)
            c.line(x, T, x, H - B, stroke='#eee')
            c.text(x, H - B + 20, f'{t:.0f}%', size=11, fill='#666', anchor='middle')
    for t in nice_ticks(0, y_hi, 6):
        if 0 <= t <= y_hi:
            y = sy(t)
            c.line(L, y, W - R, y, stroke='#eee')
            c.text(L - 10, y + 4, f'{t:.0f}', size=11, fill='#666', anchor='end')

    c.line(L, H - B, W - R, H - B, stroke='#333', width=1.5)
    c.line(L, T, L, H - B, stroke='#333', width=1.5)
    c.text((L + W - R) / 2, H - 22, 'GPU 有效负载率', size=12, fill='#444', anchor='middle')
    c.text(30, (T + H - B) / 2, 'output token 成本（元 / 百万）', size=12, fill='#444',
           anchor='middle', rotate=-90)

    # API 参照线
    c.line(L, sy(ref), W - R, sy(ref), stroke='#D4537E', width=2)
    c.text(W - R - 6, sy(ref) - 10, f'{ref_name} API：¥{ref:.0f}/M',
           size=12, fill='#993556', anchor='end', weight='600')

    # 自建成本曲线
    c.polyline([(sx(x), sy(y)) for x, y in zip(xs, ys)], stroke='#534AB7', width=2.5)
    for x, y, r in zip(xs, ys, rows):
        c.circle(sx(x), sy(y), 5, '#534AB7')
    c.text(sx(xs[0]) + 10, sy(ys[0]) - 12, rows[0]['model'], size=12,
           fill='#3C3489', weight='600')

    # 交叉点
    cross = None
    for r in rows:
        if not r['self_hosted_wins']:
            cross = r
            break
    if cross:
        cx, cy = sx(cross['utilization'] * 100), sy(cross['self_cost_per_mtok'])
        c.line(cx, T, cx, H - B, stroke='#D85A30', width=1.5, dash='6 4')
        c.text(cx + 8, T + 20,
               f"临界点 ≈ {cross['utilization']*100:.0f}%：低于此负载，自建反而不如调 API",
               size=12, fill='#993C1D', weight='600')

    c.text(L, 60, '这张图是算力侧与 token 侧的耦合点：负载率由调度决定，'
                  '而负载率反过来决定自建模型的单位成本。', size=11, fill='#888')
    return c


def main():
    if not DATA.exists():
        raise SystemExit(f'找不到 {DATA}，请先运行 scripts/run_token_experiments.py')
    payload = json.loads(DATA.read_text(encoding='utf-8'))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    outs = [
        chart_pareto(payload).save(OUT_DIR / 'token_pareto_front.svg'),
        chart_policies(payload).save(OUT_DIR / 'token_policy_comparison.svg'),
        chart_utilization(payload).save(OUT_DIR / 'token_utilization.svg'),
    ]
    print('已生成图表:')
    for p in outs:
        print(' ', p)


if __name__ == '__main__':
    main()

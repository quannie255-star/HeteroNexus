# -*- coding: utf-8 -*-
"""纯 Python 生成 SVG 对比图表，无需第三方库"""
import json

with open(r'C:\Users\10393\Desktop\大创\项目代码\simulation\baseline_results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

schedulers = ['FIFO', 'BestFit', 'DRF', 'SJF']
colors = {'FIFO': '#94a3b8', 'BestFit': '#60a5fa', 'DRF': '#0d9488', 'SJF': '#f59e0b'}
out_dir = r'C:\Users\10393\Desktop\大创\项目代码\simulation'

W, H = 900, 520
ML, MR, MT, MB = 70, 30, 60, 70  # margins

def svg_header(w, h, title):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" font-family="Microsoft YaHei, SimHei, sans-serif">
<rect width="{w}" height="{h}" fill="#ffffff"/>
<text x="{w//2}" y="32" text-anchor="middle" font-size="18" font-weight="bold" fill="#1f2937">{title}</text>'''

def svg_footer():
    return '</svg>'

def draw_axes(ax, max_val, y_label, n_cats):
    plot_w = W - ML - MR
    plot_h = H - MT - MB
    # grid lines
    lines = ''
    for i in range(6):
        y = MT + plot_h - i * plot_h / 5
        val = max_val * i / 5
        lines += f'<line x1="{ML}" y1="{y:.1f}" x2="{W-MR}" y2="{y:.1f}" stroke="#e5e7eb" stroke-width="1"/>'
        lines += f'<text x="{ML-8}" y="{y+4:.1f}" text-anchor="end" font-size="11" fill="#6b7280">{val:.0f}</text>'
    # axes
    lines += f'<line x1="{ML}" y1="{MT}" x2="{ML}" y2="{MT+plot_h}" stroke="#374151" stroke-width="1.5"/>'
    lines += f'<line x1="{ML}" y1="{MT+plot_h}" x2="{W-MR}" y2="{MT+plot_h}" stroke="#374151" stroke-width="1.5"/>'
    lines += f'<text x="20" y="{MT+plot_h//2}" text-anchor="middle" font-size="12" fill="#374151" transform="rotate(-90,20,{MT+plot_h//2})">{y_label}</text>'
    return lines, plot_w, plot_h

# ===== 图1：JCT 对比分组柱状图 =====
def chart_jct():
    metrics = [('平均 JCT', [data[s]['avg_jct'] for s in schedulers]),
               ('中位 JCT', [data[s]['median_jct'] for s in schedulers]),
               ('P95 JCT', [data[s]['p95_jct'] for s in schedulers])]
    max_val = max(max(v) for _, v in metrics) * 1.15
    svg = svg_header(W, H, '基线调度算法 JCT 对比（200 任务 / 异构集群）')
    grid, plot_w, plot_h = draw_axes(None, max_val, '时间（模拟单位）', len(schedulers))
    svg += grid

    group_w = plot_w / len(schedulers)
    bar_w = group_w * 0.22
    metric_colors = ['#3b82f6', '#0d9488', '#f59e0b']

    for si, sched in enumerate(schedulers):
        cx = ML + group_w * si + group_w / 2
        for mi, (mname, vals) in enumerate(metrics):
            bx = cx - bar_w * 1.5 + mi * bar_w * 1.05
            bh = vals[si] / max_val * plot_h
            by = MT + plot_h - bh
            svg += f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" fill="{metric_colors[mi]}" rx="2"/>'
            svg += f'<text x="{bx+bar_w/2:.1f}" y="{by-5:.1f}" text-anchor="middle" font-size="9" fill="#374151">{vals[si]:.0f}</text>'
        svg += f'<text x="{cx}" y="{MT+plot_h+22}" text-anchor="middle" font-size="13" font-weight="bold" fill="#1f2937">{sched}</text>'

    # legend
    lx = W - MR - 200
    for mi, (mname, _) in enumerate(metrics):
        svg += f'<rect x="{lx+mi*70}" y="{H-30}" width="14" height="14" fill="{metric_colors[mi]}" rx="2"/>'
        svg += f'<text x="{lx+mi*70+18}" y="{H-19}" font-size="11" fill="#374151">{mname}</text>'
    svg += svg_footer()
    with open(f'{out_dir}/chart_jct_comparison.svg', 'w', encoding='utf-8') as f:
        f.write(svg)
    print('图1 已保存: chart_jct_comparison.svg')

# ===== 图2：资源利用率对比 =====
def chart_util():
    metrics = [('CPU 利用率', [data[s]['avg_cpu_util'] for s in schedulers]),
               ('GPU 利用率', [data[s]['avg_gpu_util'] for s in schedulers]),
               ('内存利用率', [data[s]['avg_mem_util'] for s in schedulers])]
    max_val = 110
    svg = svg_header(W, H, '基线调度算法资源利用率对比')
    grid, plot_w, plot_h = draw_axes(None, max_val, '利用率 (%)', len(schedulers))
    svg += grid
    group_w = plot_w / len(schedulers)
    bar_w = group_w * 0.22
    metric_colors = ['#60a5fa', '#0d9488', '#f59e0b']
    for si, sched in enumerate(schedulers):
        cx = ML + group_w * si + group_w / 2
        for mi, (mname, vals) in enumerate(metrics):
            bx = cx - bar_w * 1.5 + mi * bar_w * 1.05
            bh = vals[si] / max_val * plot_h
            by = MT + plot_h - bh
            svg += f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" fill="{metric_colors[mi]}" rx="2"/>'
            svg += f'<text x="{bx+bar_w/2:.1f}" y="{by-5:.1f}" text-anchor="middle" font-size="9" fill="#374151">{vals[si]:.1f}%</text>'
        svg += f'<text x="{cx}" y="{MT+plot_h+22}" text-anchor="middle" font-size="13" font-weight="bold" fill="#1f2937">{sched}</text>'
    lx = W - MR - 210
    for mi, (mname, _) in enumerate(metrics):
        svg += f'<rect x="{lx+mi*72}" y="{H-30}" width="14" height="14" fill="{metric_colors[mi]}" rx="2"/>'
        svg += f'<text x="{lx+mi*72+18}" y="{H-19}" font-size="11" fill="#374151">{mname}</text>'
    svg += svg_footer()
    with open(f'{out_dir}/chart_utilization_comparison.svg', 'w', encoding='utf-8') as f:
        f.write(svg)
    print('图2 已保存: chart_utilization_comparison.svg')

# ===== 图3：综合对比表（横向条形）=====
def chart_summary():
    rows = [
        ('平均 JCT ↓', [data[s]['avg_jct'] for s in schedulers], False),
        ('中位 JCT ↓', [data[s]['median_jct'] for s in schedulers], False),
        ('P95 JCT ↓', [data[s]['p95_jct'] for s in schedulers], False),
        ('平均等待 ↓', [data[s]['avg_waiting'] for s in schedulers], False),
        ('GPU 利用率 ↑', [data[s]['avg_gpu_util'] for s in schedulers], True),
        ('CPU 利用率 ↑', [data[s]['avg_cpu_util'] for s in schedulers], True),
    ]
    n_rows = len(rows)
    rh = 55
    chart_h = MT + n_rows * rh + MB
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{chart_h}" viewBox="0 0 {W} {chart_h}" font-family="Microsoft YaHei, SimHei, sans-serif">
<rect width="{W}" height="{chart_h}" fill="#ffffff"/>
<text x="{W//2}" y="32" text-anchor="middle" font-size="18" font-weight="bold" fill="#1f2937">调度算法综合性能对比（绿色为该指标最优）</text>'''
    label_w = 130
    plot_w = W - ML - MR - label_w
    for ri, (rname, vals, higher_better) in enumerate(rows):
        y = MT + ri * rh
        best_val = max(vals) if higher_better else min(vals)
        mx = max(vals)
        # label
        svg += f'<text x="{ML}" y="{y+22}" font-size="13" font-weight="bold" fill="#374151">{rname}</text>'
        # bars
        bar_h = 18
        gap = 4
        for si, sched in enumerate(schedulers):
            by = y + 30 + si * (bar_h + gap)
            bw = vals[si] / mx * plot_w if mx > 0 else 0
            is_best = vals[si] == best_val
            fill = colors[sched] if not is_best else '#16a34a'
            svg += f'<rect x="{ML+label_w}" y="{by}" width="{bw:.1f}" height="{bar_h}" fill="{fill}" rx="3" opacity="0.85"/>'
            svg += f'<text x="{ML+label_w-5}" y="{by+13}" text-anchor="end" font-size="10" fill="#6b7280">{sched}</text>'
            svg += f'<text x="{ML+label_w+bw+5:.1f}" y="{by+13}" font-size="10" fill="#374151" font-weight="{"bold" if is_best else "normal"}">{vals[si]:.1f}{"%" if "利用率" in rname else ""}</text>'
    svg += '</svg>'
    with open(f'{out_dir}/chart_summary.svg', 'w', encoding='utf-8') as f:
        f.write(svg)
    print('图3 已保存: chart_summary.svg')

# ===== 图4：利用率随时间变化（FIFO vs DRF）=====
def chart_time_series():
    chart_h = 600
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{chart_h}" viewBox="0 0 {W} {chart_h}" font-family="Microsoft YaHei, SimHei, sans-serif">
<rect width="{W}" height="{chart_h}" fill="#ffffff"/>
<text x="{W//2}" y="30" text-anchor="middle" font-size="17" font-weight="bold" fill="#1f2937">FIFO vs DRF —— 资源利用率随时间变化</text>'''
    panel_h = 240
    for pi, (sched, color) in enumerate([('FIFO', '#3b82f6'), ('DRF', '#0d9488')]):
        py = 50 + pi * (panel_h + 30)
        samples = data[sched]['util_samples']
        times = [s['time'] for s in samples]
        gpu = [s['gpu_util']*100 for s in samples]
        cpu = [s['cpu_util']*100 for s in samples]
        t_max = max(times)
        plot_w = W - ML - MR
        # grid
        for i in range(6):
            gy = py + 20 + panel_h - 40 - i * (panel_h - 60) / 5
            svg += f'<line x1="{ML}" y1="{gy:.1f}" x2="{W-MR}" y2="{gy:.1f}" stroke="#f3f4f6" stroke-width="1"/>'
            svg += f'<text x="{ML-8}" y="{gy+4:.1f}" text-anchor="end" font-size="10" fill="#9ca3af">{i*20}%</text>'
        # GPU line
        pts = []
        for ti, t in enumerate(times):
            x = ML + t / t_max * plot_w
            y = py + 20 + panel_h - 40 - gpu[ti] / 100 * (panel_h - 60)
            pts.append(f'{x:.1f},{y:.1f}')
        svg += f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="1.8"/>'
        # CPU line
        pts = []
        for ti, t in enumerate(times):
            x = ML + t / t_max * plot_w
            y = py + 20 + panel_h - 40 - cpu[ti] / 100 * (panel_h - 60)
            pts.append(f'{x:.1f},{y:.1f}')
        svg += f'<polyline points="{" ".join(pts)}" fill="none" stroke="#f59e0b" stroke-width="1.5" opacity="0.8"/>'
        # axes
        svg += f'<line x1="{ML}" y1="{py+20}" x2="{ML}" y2="{py+panel_h-20}" stroke="#374151" stroke-width="1"/>'
        svg += f'<line x1="{ML}" y1="{py+panel_h-20}" x2="{W-MR}" y2="{py+panel_h-20}" stroke="#374151" stroke-width="1"/>'
        svg += f'<text x="{ML}" y="{py+12}" font-size="13" font-weight="bold" fill="{color}">{sched} 调度器</text>'
        svg += f'<text x="{W//2}" y="{py+panel_h-2}" text-anchor="middle" font-size="11" fill="#6b7280">模拟时间 →</text>'
    # legend
    svg += f'<rect x="{W-180}" y="8" width="12" height="12" fill="#3b82f6" rx="2"/><text x="{W-164}" y="18" font-size="11" fill="#374151">GPU 利用率</text>'
    svg += f'<rect x="{W-90}" y="8" width="12" height="12" fill="#f59e0b" rx="2"/><text x="{W-74}" y="18" font-size="11" fill="#374151">CPU 利用率</text>'
    svg += '</svg>'
    with open(f'{out_dir}/chart_utilization_over_time.svg', 'w', encoding='utf-8') as f:
        f.write(svg)
    print('图4 已保存: chart_utilization_over_time.svg')

chart_jct()
chart_util()
chart_summary()
chart_time_series()
print('\n全部 SVG 图表生成完成！')

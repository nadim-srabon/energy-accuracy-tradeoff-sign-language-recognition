"""LaTeX rows of paper Tables 2-3 and the operational-energy values (version 4) from summary.csv, seed_results.csv and runs/.

Usage: python make_rows5.py summary.csv seed_results.csv runs
"""
import csv, json, os, sys, statistics as st

summ, seeds_csv, runs = sys.argv[1:4]
ROWS = [('mobilenet', 'baseline', 'MobileNetV3-Small'), ('resnet', 'baseline', 'ResNet-50'),
        ('r2plus1d', 'baseline', 'R(2+1)D-18'), ('vivit', 'baseline', 'ViViT-Lite'),
        ('timesformer', 'baseline', 'TimeSformer'), ('swin3d', 'baseline', 'Video Swin-T'),
        ('mobilevit', 'baseline', 'MobileViT-S'), ('hybrid', 'baseline', 'Hybrid, learned gate'),
        ('hybrid', 'fixed', 'Hybrid, fixed 50:50')]
SEEDS = [42, 123, 2024, 7, 999]
S = {}
for r in csv.DictReader(open(summ, encoding='utf8')):
    S.setdefault((r['model'], r['variant']), {})[r['metric']] = {k: float(r[k]) for k in ['mean', 'sd', 'ci95_low', 'ci95_high']}
P = {}
for r in csv.DictReader(open(seeds_csv, encoding='utf8')):
    P.setdefault((r['model'], r['variant']), {})[int(r['seed'])] = r
best = max(S[(m, v)]['test_top1_pct']['mean'] for m, v, _ in ROWS)


def neg(x):
    return f'$-${-x:.2f}' if x < 0 else f'{x:.2f}'


print('% Table 2')
for m, v, name in ROWS:
    a, a5, e, i = (S[(m, v)][k] for k in ['test_top1_pct', 'test_top5_pct', 'train_kwh', 'inference_kwh_per_clip'])
    top1 = f"{a['mean']:.2f} $\\pm$ {a['sd']:.2f}"
    if abs(a['mean'] - best) < 1e-9:
        top1 = f'\\textbf{{{top1}}}'
    print(f"    {name:22s}& {top1} & [{neg(a['ci95_low'])}, {neg(a['ci95_high'])}] & {a5['mean']:.2f} $\\pm$ {a5['sd']:.2f} & "
          f"{e['mean']:.3f} $\\pm$ {e['sd']:.3f} & {i['mean']*1e6:.3f} $\\pm$ {i['sd']*1e6:.3f} \\\\")
print('% operational energy (text and Figure 3)')
for m, v, name in ROWS:
    et = st.mean(float(x['train_kwh']) for x in P[(m, v)].values())
    ec = st.mean(float(x['inference_mwh_per_clip']) for x in P[(m, v)].values()) * 1e-6
    ops = [et + n * ec for n in (1e4, 1e5, 1e6)]
    print(f"    {name:22s}& {ops[0]:.3f} & {ops[1]:.3f} & {ops[2]:.3f} & {(et/1e5 + ec)*412.72*1000:.3f} \\\\")
print('% Table 3')
R = {}
for v in ['baseline', 'fixed']:
    for s in SEEDS:
        R[(v, s)] = json.load(open(os.path.join(runs, f'hybrid_{v}_seed{s}_train', 'result.json'), encoding='utf8'))


def line(label, vals, fmt='{:.2f}'):
    print(f"    {label:24s}& " + ' & '.join(fmt.format(x) for x in vals) + ' & ' + fmt.format(st.mean(vals)) + ' \\\\')


for v, lab in [('baseline', 'Learned gate, fused'), ('fixed', 'Fixed 50:50, fused')]:
    line(lab, [R[(v, s)]['test_top1_pct'] for s in SEEDS])
    line('\\quad CNN head alone', [R[(v, s)]['checkpoint_diagnostic_top1_pct']['aux_cnn'] for s in SEEDS])
    line('\\quad Swin head alone', [R[(v, s)]['checkpoint_diagnostic_top1_pct']['aux_swin'] for s in SEEDS])
    if v == 'baseline':
        line('\\quad Gate on CNN (\\%)', [R[(v, s)]['gate_mean'][0] * 100 for s in SEEDS], '{:.1f}')
    print('    \\midrule')
line('Fused, fixed $-$ learned', [R[('fixed', s)]['test_top1_pct'] - R[('baseline', s)]['test_top1_pct'] for s in SEEDS], '{:+.2f}')

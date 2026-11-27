"""Paper Figures 2 and 3 (version 4: 9 configurations x 5 seeds) from the analyzer's summary.csv.

Usage: python figs5.py <figures-dir> summary.csv
Colour = family (CNN blue, Transformer green, hybrid orange); square = Kinetics-pretrained;
hollow = fixed-fusion variant. Figure 1 is drawn in TikZ (sections/figure_architecture.tex).
"""
import csv, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out, summ = sys.argv[1], sys.argv[2]
plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.family': 'serif', 'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#555555', 'axes.linewidth': 0.6, 'xtick.color': '#444444', 'ytick.color': '#444444'})
CNN, TR, HY = '#2a78d6', '#1baf7a', '#eb6834'
KEY = {('mobilenet', 'baseline'): 'MobileNetV3-Small', ('resnet', 'baseline'): 'ResNet-50',
       ('r2plus1d', 'baseline'): 'R(2+1)D-18', ('vivit', 'baseline'): 'ViViT-Lite',
       ('timesformer', 'baseline'): 'TimeSformer', ('swin3d', 'baseline'): 'Video Swin-T',
       ('mobilevit', 'baseline'): 'MobileViT-S', ('hybrid', 'baseline'): 'Hybrid, learned gate',
       ('hybrid', 'fixed'): 'Hybrid, fixed 50:50'}
V = {}
for r in csv.DictReader(open(summ, encoding='utf8')):
    V.setdefault(KEY[(r['model'], r['variant'])], {})[r['metric']] = (float(r['mean']), float(r['sd']))
# name, colour, marker, filled
STYLE = [('MobileNetV3-Small', CNN, 'o', True), ('ResNet-50', CNN, 'o', True), ('R(2+1)D-18', CNN, 's', True),
         ('ViViT-Lite', TR, 'o', True), ('TimeSformer', TR, 's', True), ('Video Swin-T', TR, 's', True),
         ('MobileViT-S', HY, 'o', True), ('Hybrid, learned gate', HY, 'D', True), ('Hybrid, fixed 50:50', HY, 'D', False)]
M = []
for n, c, mk, f in STYLE:
    v = V[n]
    M.append(dict(n=n, c=c, mk=mk, f=f, a=v['test_top1_pct'][0], asd=v['test_top1_pct'][1],
                  e=v['train_kwh'][0], esd=v['train_kwh'][1], i=v['inference_kwh_per_clip'][0] * 1e6))
byname = {m['n']: m for m in M}

# ---------- Figure 2: accuracy vs training energy ----------
fig, ax = plt.subplots(figsize=(3.4, 2.45))
ax.grid(True, color='#e6e6e6', lw=0.5); ax.set_axisbelow(True)
fr = [byname['ViViT-Lite'], byname['Video Swin-T']]
ax.plot([fr[0]['e'], fr[1]['e']], [fr[0]['a'], fr[1]['a']], color='#888888', lw=1, zorder=1)
off = {'MobileNetV3-Small': (2, 9), 'ResNet-50': (-5, 7), 'R(2+1)D-18': (6, -3), 'ViViT-Lite': (4, -11),
       'TimeSformer': (21, -2), 'Video Swin-T': (0, 11), 'MobileViT-S': (-4, 8),
       'Hybrid, learned gate': (-6, -11), 'Hybrid, fixed 50:50': (6, 2)}
ha = {'ResNet-50': 'right', 'MobileViT-S': 'center', 'Hybrid, learned gate': 'right', 'Video Swin-T': 'center'}
for m in M:
    ax.errorbar(m['e'], m['a'], yerr=m['asd'], xerr=m['esd'], fmt='none', ecolor=m['c'], elinewidth=0.9, capsize=1.8, zorder=2)
    ax.scatter(m['e'], m['a'], s=30, marker=m['mk'], facecolor=m['c'] if m['f'] else 'white',
               edgecolor='white' if m['f'] else m['c'], linewidth=1.1 if m['f'] else 1.3, zorder=3)
    ax.annotate(m['n'], (m['e'], m['a']), xytext=off[m['n']], textcoords='offset points', fontsize=6.5,
                color='#222222', ha=ha.get(m['n'], 'left'))
ax.set_xlabel(r'Training + validation energy (kWh), mean $\pm$ SD')
ax.set_ylabel(r'Test top-1 (%), mean $\pm$ SD')
ax.set_xlim(0, 0.56); ax.set_ylim(0, 74)
fig.tight_layout(pad=0.3)
fig.savefig(f'{out}/figure2_accuracy_vs_energy_a4000.pdf'); fig.savefig(f'{out}/figure2_accuracy_vs_energy_a4000.png', dpi=250)

# ---------- Figure 3: operational energy vs volume ----------
fig, ax = plt.subplots(figsize=(3.4, 2.55))
ax.grid(True, which='major', color='#e6e6e6', lw=0.5); ax.set_axisbelow(True)
N = np.logspace(3, 6, 200)
ls = {'MobileNetV3-Small': '--', 'ResNet-50': '-', 'R(2+1)D-18': '-.', 'ViViT-Lite': '--', 'TimeSformer': '-',
      'Video Swin-T': '-.', 'MobileViT-S': '--', 'Hybrid, learned gate': '-', 'Hybrid, fixed 50:50': ':'}
ends = {}
for m in M:
    y = m['e'] + N * m['i'] * 1e-6
    ax.plot(N, y, color=m['c'], lw=1.4 if ls[m['n']] != ':' else 1.7, ls=ls[m['n']])
    ends[m['n']] = np.log10(y[-1])
# spread end labels to a minimum vertical gap (log10 units), keeping their order
order = sorted(ends, key=ends.get)
pos = [ends[n] for n in order]
gap = 0.085
for _ in range(200):
    for k in range(1, len(pos)):
        if pos[k] - pos[k - 1] < gap:
            d = (gap - (pos[k] - pos[k - 1])) / 2
            pos[k] += d; pos[k - 1] -= d
for n, p in zip(order, pos):
    ax.annotate(n, (1e6, 10 ** ends[n]), xytext=(1.12e6, 10 ** p), textcoords='data', fontsize=6, va='center',
                color='#222222', annotation_clip=False,
                arrowprops=dict(arrowstyle='-', color='#aaaaaa', lw=0.4, shrinkA=0, shrinkB=0))
ax.axvspan(1e4, 1e6, color='#f2f2f2', zorder=0)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlim(1e3, 1e6)
ax.set_xlabel('Assumed number of inferences $N$ (clips)')
ax.set_ylabel('Operational energy (kWh)')
fig.tight_layout(pad=0.3); fig.subplots_adjust(right=0.68)
fig.savefig(f'{out}/figure3_operational_energy_vs_volume.pdf'); fig.savefig(f'{out}/figure3_operational_energy_vs_volume.png', dpi=250)
print('ok')

"""Every A4000 number in the paper (version 4: 9 configurations x 5 seeds), computed
from the machine-generated files: summary.csv and seed_results.csv (analyzer output,
analyze_ext.py) and the 45 run directories in runs/ (result.json, emissions.csv,
training_history.json, provenance.json).

Usage: python derive5.py summary.csv seed_results.csv runs
"""
import csv, json, os, sys, glob, itertools, statistics as st
from scipy.stats import t as tdist

summ, seeds_csv, runs = sys.argv[1], sys.argv[2], sys.argv[3]
NAME = {('mobilenet', 'baseline'): 'MobileNetV3-Small', ('resnet', 'baseline'): 'ResNet-50',
        ('r2plus1d', 'baseline'): 'R(2+1)D-18', ('vivit', 'baseline'): 'ViViT-Lite',
        ('timesformer', 'baseline'): 'TimeSformer', ('swin3d', 'baseline'): 'Video Swin-T',
        ('mobilevit', 'baseline'): 'MobileViT-S', ('hybrid', 'baseline'): 'Hybrid (learned)',
        ('hybrid', 'fixed'): 'Hybrid (fixed)'}
ORDER = list(NAME.values())
SEEDS = [42, 123, 2024, 7, 999]
# Runs from the first extension session (22-23 Nov 2026), which ran in a disturbed machine state
# (see paper Sec. 3.4); used only for the robustness check below.
SESSION1 = {('MobileNetV3-Small', 7), ('ResNet-50', 7), ('ViViT-Lite', 7)}
I_REF = 412.72

S = {}
for r in csv.DictReader(open(summ, encoding='utf8')):
    S.setdefault(NAME[(r['model'], r['variant'])], {})[r['metric']] = {k: float(r[k]) for k in ['mean', 'sd', 'ci95_low', 'ci95_high']}
P = {}
for r in csv.DictReader(open(seeds_csv, encoding='utf8')):
    P.setdefault(NAME[(r['model'], r['variant'])], {})[int(r['seed'])] = dict(
        top1=float(r['top1_pct']), top5=float(r['top5_pct']), kwh=float(r['train_kwh']),
        mwh=float(r['inference_mwh_per_clip']), ms=float(r['ms_per_clip']), mem=float(r['peak_infer_gb']),
        epochs=int(r['epochs']), campaign=r['campaign'])
assert all(sorted(P[m]) == sorted(SEEDS) for m in ORDER), 'expected 5 seeds per configuration'


def mean(m, k, excl=False):
    return st.mean(v[k] for s, v in P[m].items() if not (excl and (m, s) in SESSION1))


print(f'== Table 2 (mean +- SD [95% t-interval], n=5, t={tdist.ppf(.975, 4):.3f})')
for m in ORDER:
    a, a5, e, i = S[m]['test_top1_pct'], S[m]['test_top5_pct'], S[m]['train_kwh'], S[m]['inference_kwh_per_clip']
    print(f"{m:18s} top1 {a['mean']:.2f}+-{a['sd']:.2f} [{a['ci95_low']:.2f},{a['ci95_high']:.2f}] top5 {a5['mean']:.2f}+-{a5['sd']:.2f} "
          f"Etrain {e['mean']:.3f}+-{e['sd']:.3f} (SD {e['sd']/e['mean']*100:.0f}%) Eclip {i['mean']*1e6:.3f}+-{i['sd']*1e6:.3f} (SD {i['sd']/i['mean']*100:.1f}%) "
          f"ms {mean(m, 'ms'):.1f} mem {max(v['mem'] for v in P[m].values()):.2f} epochs {[P[m][s]['epochs'] for s in SEEDS]}")
print('== same, original three seeds only (42/123/2024)')
for m in ORDER:
    v = [P[m][s] for s in (42, 123, 2024)]
    print(f"{m:18s} top1 {st.mean(x['top1'] for x in v):.2f}+-{st.stdev(x['top1'] for x in v):.2f} Etrain {st.mean(x['kwh'] for x in v):.3f} Eclip {st.mean(x['mwh'] for x in v):.3f}")

print('== descriptive Pareto fronts on means (5 seeds)')
for label, key in [('training energy', 'kwh'), ('inference energy per clip', 'mwh')]:
    pts = {m: (mean(m, key), mean(m, 'top1')) for m in ORDER}
    front = [m for m in ORDER if not any(pts[o][0] <= pts[m][0] and pts[o][1] >= pts[m][1] and pts[o] != pts[m] for o in ORDER)]
    print(f"{label}: front = {front}")

print('== per-seed comparisons (paired by seed): d(top1), d(train kWh), d(mWh/clip)')
for a, b in [('Video Swin-T', 'TimeSformer'), ('Video Swin-T', 'R(2+1)D-18'), ('Video Swin-T', 'ResNet-50'),
             ('TimeSformer', 'ResNet-50'), ('R(2+1)D-18', 'ResNet-50'), ('Hybrid (fixed)', 'Hybrid (learned)'),
             ('MobileViT-S', 'MobileNetV3-Small'), ('ViViT-Lite', 'MobileNetV3-Small')]:
    d = [(round(P[a][s]['top1'] - P[b][s]['top1'], 2), round(P[a][s]['kwh'] - P[b][s]['kwh'], 3), round(P[a][s]['mwh'] - P[b][s]['mwh'], 3)) for s in SEEDS]
    allE = all(x['kwh'] < y['kwh'] for x in P[a].values() for y in P[b].values())
    allA = all(x['top1'] > y['top1'] for x in P[a].values() for y in P[b].values())
    print(f"{a} - {b}: {d}; every-run lower train energy: {allE}; every-run higher top-1: {allA}")
for m in ORDER:
    v = P[m].values()
    print(f"  range {m:18s} top1 {min(x['top1'] for x in v):.2f}-{max(x['top1'] for x in v):.2f} kWh {min(x['kwh'] for x in v):.3f}-{max(x['kwh'] for x in v):.3f} mWh {min(x['mwh'] for x in v):.3f}-{max(x['mwh'] for x in v):.3f}")
d = [P['Hybrid (fixed)'][s]['top1'] - P['Hybrid (learned)'][s]['top1'] for s in SEEDS]
print(f"fixed-learned mean diff {st.mean(d):.2f}; exact two-sided sign-flip p over 5 seeds = "
      f"{sum(1 for signs in itertools.product([1, -1], repeat=5) if abs(sum(x*s for x, s in zip(d, signs))) >= abs(sum(d)) - 1e-12) / 32:.4f}")


def crossover(a, b, excl=False):
    """a: lower training energy, higher per-clip energy."""
    pa = [v for s, v in P[a].items() if not (excl and (a, s) in SESSION1)]
    pb = [v for s, v in P[b].items() if not (excl and (b, s) in SESSION1)]
    ns = [(y['kwh'] - x['kwh']) / ((x['mwh'] - y['mwh']) * 1e-6) for x in pa for y in pb]
    m = (mean(b, 'kwh', excl) - mean(a, 'kwh', excl)) / ((mean(a, 'mwh', excl) - mean(b, 'mwh', excl)) * 1e-6)
    return m, min(ns), max(ns), len(ns)


print('== operational crossovers (means; min/max over all seed pairings)')
for a, b in [('TimeSformer', 'ResNet-50'), ('Video Swin-T', 'ResNet-50'), ('R(2+1)D-18', 'ResNet-50'), ('ViViT-Lite', 'MobileNetV3-Small')]:
    m, lo, hi, n = crossover(a, b)
    m2, lo2, hi2, n2 = crossover(a, b, excl=True)
    print(f"{a} vs {b}: N*={m:,.0f} [{lo:,.0f}, {hi:,.0f}] over {n} pairings | excl. session-1 runs: {m2:,.0f} [{lo2:,.0f}, {hi2:,.0f}] ({n2})")

print('== operational energy E_train + N E_clip (kWh) from means; mg/query at 1e5 (412.72); training share; training CO2e')
for m in ORDER:
    et, ec = mean(m, 'kwh'), mean(m, 'mwh') * 1e-6
    ops = [et + n * ec for n in (1e4, 1e5, 1e6)]
    print(f"{m:18s} {ops[0]:.3f} {ops[1]:.3f} {ops[2]:.3f}  mg/q {(et/1e5+ec)*I_REF*1000:.3f}  train share {et/ops[0]*100:.0f}/{et/ops[1]*100:.0f}/{et/ops[2]*100:.0f}%  "
          f"CO2e train {et*I_REF:.1f} g (50: {et*50:.1f}, 800: {et*800:.1f})  eta {mean(m, 'top1')/(et*I_REF):.3f}%/g")
for n in (1e4, 1e5, 1e6):
    pts = {m: (mean(m, 'kwh') + n * mean(m, 'mwh') * 1e-6, mean(m, 'top1')) for m in ORDER}
    front = [m for m in ORDER if not any(pts[o][0] <= pts[m][0] and pts[o][1] >= pts[m][1] and pts[o] != pts[m] for o in ORDER)]
    print(f"operational front at N={n:.0e}: {front}")

print('== robustness: energy means without the three session-1 runs')
for m in ORDER:
    print(f"{m:18s} Etrain {mean(m, 'kwh', True):.3f} (all {mean(m, 'kwh'):.3f})  Eclip {mean(m, 'mwh', True):.3f} (all {mean(m, 'mwh'):.3f})  ms {mean(m, 'ms', True):.1f}")

print('== runs: epochs, epoch duration, energy/epoch, component shares, fit at best-validation epoch, gate/aux')
rows = {}
for dpath in sorted(glob.glob(os.path.join(runs, '*_train'))):
    r = json.load(open(os.path.join(dpath, 'result.json'), encoding='utf8'))
    h = json.load(open(os.path.join(dpath, 'training_history.json'), encoding='utf8'))
    em = [x for x in csv.DictReader(open(os.path.join(dpath, 'emissions.csv'), encoding='utf8')) if 'inference' not in x['project_name']]
    dur = st.mean(float(x['duration']) for x in em)
    tot = sum(float(x['energy_consumed']) for x in em)
    shares = {k: sum(float(x[k + '_energy']) for x in em) / tot * 100 for k in ['gpu', 'cpu', 'ram']}
    vals = [x['val_top1'] for x in h]; be = vals.index(max(vals))
    m = NAME[(r['model'], r['variant'])]
    rows.setdefault(m, []).append(dict(dur=dur, kwh_ep=tot / len(em), fit=h[be]['train_top1'] * 100, val=vals[be] * 100, **shares))
    line = (f"{os.path.basename(dpath):36s} ep {len(h):2d} best {be+1:2d} epoch_s {dur:6.1f} kWh/ep {tot/len(em):.4f} gpu {shares['gpu']:.0f}% "
            f"cpu {shares['cpu']:.0f}% ram {shares['ram']:.0f}% trainTop1@best {h[be]['train_top1']*100:.1f} val@best {vals[be]*100:.1f} test {r['test_top1_pct']:.2f}")
    if 'gate_mean' in r:
        q = r['gate_quantiles']
        line += (f" | gateCNN {r['gate_mean'][0]*100:.1f} H {r['gate_mean_entropy_nats']:.3f} IQR {q[1][0]:.3f}-{q[3][0]:.3f} "
                 f"norms {[round(x, 1) for x in r['branch_feature_l2_norm_mean']]} aux {r['checkpoint_diagnostic_top1_pct']}")
    print(line)
print('== per-configuration ranges over seeds')
for m in ORDER:
    v = rows[m]
    rg = lambda k: f"{min(x[k] for x in v):.1f}-{max(x[k] for x in v):.1f}"
    print(f"{m:18s} epoch_s {rg('dur')} Wh/ep {min(x['kwh_ep'] for x in v)*1000:.2f}-{max(x['kwh_ep'] for x in v)*1000:.2f} "
          f"gpu {rg('gpu')}% cpu {rg('cpu')}% ram {rg('ram')}% fit@best {rg('fit')}% val@best {rg('val')}%")
allv = [x for v in rows.values() for x in v]
print(f"all runs: gpu {min(x['gpu'] for x in allv):.0f}-{max(x['gpu'] for x in allv):.0f}% cpu {min(x['cpu'] for x in allv):.0f}-{max(x['cpu'] for x in allv):.0f}% ram {min(x['ram'] for x in allv):.0f}-{max(x['ram'] for x in allv):.0f}%")

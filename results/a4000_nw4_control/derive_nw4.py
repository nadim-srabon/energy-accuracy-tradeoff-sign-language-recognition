"""Every number of the data-loading control in the paper (Sec. 4.4), from the run folders.

Compares ResNet-50 and TimeSformer trained with num_workers=4 (this folder's runs/) with
the same configurations and seeds trained with num_workers=0 (../a4000_cohort/runs/).

Usage: python derive_nw4.py runs ../a4000_cohort/runs
"""
import csv, glob, json, os, sys, statistics as st
from scipy.stats import t

nw4, nw0 = sys.argv[1], sys.argv[2]
SEEDS = [42, 123, 2024, 7, 999]


def run(root, model, seed):
    d = os.path.join(root, f'{model}_baseline_seed{seed}_train')
    r = json.load(open(os.path.join(d, 'result.json'), encoding='utf8'))
    em = list(csv.DictReader(open(os.path.join(d, 'emissions.csv'), encoding='utf8')))
    tr = [e for e in em if 'inference' not in e['project_name']]
    return dict(kwh=r['train_kwh'], top1=r['test_top1_pct'], epochs=len(tr),
                s_ep=st.mean(float(e['duration']) for e in tr),
                wh_ep=sum(float(e['energy_consumed']) for e in tr) / len(tr) * 1000,
                cpu_w=st.mean(float(e['cpu_power']) for e in tr),
                mwh=r['inference_kwh_per_clip'] * 1e6,
                ms=st.mean(w['batch_amortized_ms_per_clip'] for w in r['inference_windows']),
                start=em[0]['timestamp'][:16], tracked=sum(float(e['energy_consumed']) for e in em))


def ci(v):
    m, s = st.mean(v), st.stdev(v)
    h = t.ppf(.975, len(v) - 1) * s / len(v) ** .5
    return f'{m:.3f} +- {s:.3f} [{m-h:.3f}, {m+h:.3f}]'


R = {(m, w, s): run(root, m, s) for m in ['resnet', 'timesformer'] for w, root in [('nw0', nw0), ('nw4', nw4)] for s in SEEDS}
for m in ['resnet', 'timesformer']:
    print(f'== {m}')
    for w in ['nw0', 'nw4']:
        v = [R[(m, w, s)] for s in SEEDS]
        print(f"  {w}: train kWh {ci([x['kwh'] for x in v])}; top-1 {ci([x['top1'] for x in v])}; epochs {[x['epochs'] for x in v]}; "
              f"s/epoch {min(x['s_ep'] for x in v):.0f}-{max(x['s_ep'] for x in v):.0f}; Wh/epoch {min(x['wh_ep'] for x in v):.2f}-{max(x['wh_ep'] for x in v):.2f}; "
              f"ms/clip {min(x['ms'] for x in v):.1f}-{max(x['ms'] for x in v):.1f}; mWh/clip {min(x['mwh'] for x in v):.3f}-{max(x['mwh'] for x in v):.3f}")
    print('  paired training-energy change (%):', [round((R[(m, 'nw4', s)]['kwh'] / R[(m, 'nw0', s)]['kwh'] - 1) * 100, 1) for s in SEEDS])
    print('  paired top-1 difference (points):', [round(R[(m, 'nw4', s)]['top1'] - R[(m, 'nw0', s)]['top1'], 2) for s in SEEDS])
    print('  nw4 CPU power by run (W):', [(s, R[(m, 'nw4', s)]['start'], round(R[(m, 'nw4', s)]['cpu_w'])) for s in SEEDS])
mean = lambda m, w, k: st.mean(R[(m, w, s)][k] for s in SEEDS)
for w in ['nw0', 'nw4']:
    print(f"{w}: ResNet/TimeSformer training-energy ratio {mean('resnet', w, 'kwh') / mean('timesformer', w, 'kwh'):.2f}")
print('nw4: every ResNet run below every TimeSformer run in training energy:',
      max(R[('resnet', 'nw4', s)]['kwh'] for s in SEEDS) < min(R[('timesformer', 'nw4', s)]['kwh'] for s in SEEDS))
print('nw4: every TimeSformer run more accurate than every ResNet run:',
      min(R[('timesformer', 'nw4', s)]['top1'] for s in SEEDS) > max(R[('resnet', 'nw4', s)]['top1'] for s in SEEDS))
print(f"tracked energy of the 10 control runs: {sum(R[(m, 'nw4', s)]['tracked'] for m in ['resnet', 'timesformer'] for s in SEEDS):.3f} kWh")

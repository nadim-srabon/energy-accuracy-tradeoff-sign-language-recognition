"""Pool the first A4000 campaign and the extension campaign, then summarise.

Same statistics as revision/gpu_package/analyze.py (mean, SD, two-sided 95% t
interval over seeds; descriptive Pareto front; illustrative carbon scenarios),
extended to several input folders and a few more per-run metrics. Pooling is
only allowed within one (cohort, manifest, model, variant, lr) key, and a
seed may appear only once per key.

Usage:
  python analyze_ext.py D:/nsyss50_results/local_pc_a4000_rebuiltcache_v1 \
      D:/nsyss50_results_ext/local_pc_a4000_rebuiltcache_v1 --out analysis_ext
"""
import argparse
from collections import defaultdict
import csv
import itertools
import json
from pathlib import Path
import statistics

ORIGINAL_SEEDS = {42, 123, 2024}
KEY = ['cohort', 'manifest_sha256', 'model', 'variant', 'lr']


def interval(values):
    from scipy.stats import t
    n = len(values)
    mean = statistics.mean(values)
    if n < 2:
        return mean, None, None, None
    sd = statistics.stdev(values)
    half = float(t.ppf(.975, n - 1)) * sd / n ** .5
    return mean, sd, mean - half, mean + half


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_record(file):
    r = json.loads(file.read_text(encoding='utf-8'))
    d = file.parent
    windows = r.get('inference_windows', [])
    epochs = None
    if (d / 'training_history.json').is_file():
        epochs = len(json.loads((d / 'training_history.json').read_text(encoding='utf-8')))
    tracked_kwh = None
    if (d / 'emissions.csv').is_file():
        with (d / 'emissions.csv').open(encoding='utf-8', newline='') as f:
            tracked_kwh = sum(float(row['energy_consumed']) for row in csv.DictReader(f))
    return r, {
        'test_top1_pct': r['test_top1_pct'],
        'test_top5_pct': r.get('test_top5_pct'),
        'train_kwh': r['train_kwh'],
        'inference_kwh_per_clip': r['inference_kwh_per_clip'],
        'ms_per_clip': statistics.mean(w['batch_amortized_ms_per_clip'] for w in windows) if windows else None,
        'peak_infer_gb': max(w['peak_allocated_gb'] for w in windows) if windows else None,
        'epochs': epochs,
        'tracked_kwh_all_windows': tracked_kwh,
    }


def summarise(groups, seeds=None):
    rows = []
    for key, runs in sorted(groups.items()):
        runs = [x for x in runs if seeds is None or x[0]['seed'] in seeds]
        if not runs:
            continue
        base = dict(zip(KEY, key))
        for metric in ['test_top1_pct', 'test_top5_pct', 'train_kwh', 'inference_kwh_per_clip',
                       'ms_per_clip', 'peak_infer_gb', 'epochs']:
            values = [m[metric] for _, m, _ in runs if m[metric] is not None]
            if values:
                mean, sd, low, high = interval(values)
                rows.append({**base, 'metric': metric, 'n_seeds': len(values),
                             'seeds': ' '.join(str(r['seed']) for r, _, _ in sorted(runs, key=lambda x: x[0]['seed'])),
                             'mean': mean, 'sd': sd, 'ci95_low': low, 'ci95_high': high})
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('results', type=Path, nargs='+', help='one or more cohort folders')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--expected-seeds', type=int, default=5)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    groups, raw, seen = defaultdict(list), [], set()
    for root in args.results:
        files = sorted(root.rglob('result.json'))
        if not files:
            raise ValueError(f'No result.json under {root}')
        for file in files:
            r, m = run_record(file)
            key = tuple(r[k] for k in KEY)
            identity = (*key, r['seed'])
            if identity in seen:
                raise ValueError(f'Duplicate seed/recipe/cohort: {identity} ({file})')
            seen.add(identity)
            provenance = json.loads((file.parent / 'provenance.json').read_text(encoding='utf-8'))
            campaign = 'extension' if 'runner' in provenance else 'first'
            groups[key].append((r, m, campaign))
            raw.append({**dict(zip(KEY, key)), 'seed': r['seed'], 'campaign': campaign, 'mode': r['mode'],
                        'top1_pct': m['test_top1_pct'], 'top5_pct': m['test_top5_pct'],
                        'train_kwh': m['train_kwh'], 'inference_mwh_per_clip': m['inference_kwh_per_clip'] * 1e6,
                        'ms_per_clip': m['ms_per_clip'], 'peak_infer_gb': m['peak_infer_gb'],
                        'epochs': m['epochs'], 'tracked_kwh_all_windows': m['tracked_kwh_all_windows'],
                        'source': str(file.resolve())})
    manifests = {k[1] for k in groups}
    if len(manifests) != 1:
        raise ValueError(f'More than one data manifest across inputs: {manifests}')
    raw.sort(key=lambda r: (r['model'], r['variant'], r['seed']))
    write_csv(args.out / 'seed_results.csv', raw)
    write_csv(args.out / 'summary.csv', summarise(groups))
    write_csv(args.out / 'summary_original3seeds.csv', summarise(groups, ORIGINAL_SEEDS))

    # Illustrative operational carbon scenarios (same grid as analyze.py).
    scenarios = []
    for key, runs in sorted(groups.items()):
        for intensity in [50, 200, 412.72, 800]:
            for volume in [10000, 100000, 1000000]:
                values = [(m['train_kwh'] / volume + m['inference_kwh_per_clip']) * intensity * 1000 for _, m, _ in runs]
                mean, sd, low, high = interval(values)
                scenarios.append({**dict(zip(KEY, key)), 'scenario_g_per_kwh': intensity, 'queries': volume,
                                  'n_seeds': len(values), 'operational_mg_per_query': mean, 'sd': sd,
                                  'ci95_low': low, 'ci95_high': high})
    write_csv(args.out / 'illustrative_carbon_scenarios.csv', scenarios)

    # Descriptive Pareto front on mean training energy vs mean top-1.
    means = {key: (statistics.mean(m['train_kwh'] for _, m, _ in runs),
                   statistics.mean(m['test_top1_pct'] for _, m, _ in runs),
                   statistics.mean(m['inference_kwh_per_clip'] for _, m, _ in runs)) for key, runs in groups.items()}
    frontier = []
    for key, (e, a, c) in sorted(means.items()):
        dominated = any(e2 <= e and a2 >= a and (e2 < e or a2 > a) for k2, (e2, a2, _) in means.items() if k2 != key)
        frontier.append({**dict(zip(KEY, key)), 'n_seeds': len(groups[key]), 'mean_train_kwh': e,
                         'mean_top1_pct': a, 'mean_inference_mwh_per_clip': c * 1e6,
                         'descriptive_frontier': not dominated})
    write_csv(args.out / 'pareto.csv', frontier)

    # Operational-energy crossover E_train + N * E_clip for every pair whose orderings disagree.
    cross = []
    for (k1, (e1, a1, c1)), (k2, (e2, a2, c2)) in itertools.combinations(sorted(means.items()), 2):
        if (e1 - e2) * (c1 - c2) < 0:
            n_star = (e2 - e1) / (c1 - c2)
            cross.append({'config_a': f'{k1[2]}/{k1[3]}', 'config_b': f'{k2[2]}/{k2[3]}',
                          'a_train_kwh': e1, 'b_train_kwh': e2, 'a_mwh_per_clip': c1 * 1e6,
                          'b_mwh_per_clip': c2 * 1e6, 'crossover_inferences': n_star})
    write_csv(args.out / 'crossover.csv', cross)

    # Completeness check.
    lines = []
    for key, runs in sorted(groups.items()):
        seeds = sorted(r['seed'] for r, _, _ in runs)
        flag = '' if len(seeds) == args.expected_seeds else f'   <-- expected {args.expected_seeds} seeds'
        lines.append(f'{key[2]:<12} {key[3]:<9} n={len(seeds)} seeds={seeds}{flag}')
    total = sum(m['tracked_kwh_all_windows'] or 0 for runs in groups.values() for _, m, _ in runs)
    ext = sum(m['tracked_kwh_all_windows'] or 0 for runs in groups.values() for _, m, c in runs if c == 'extension')
    lines.append(f'\nTracked energy (training + inference windows), all runs: {total:.3f} kWh; extension campaign: {ext:.3f} kWh')
    (args.out / 'checks.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    for row in frontier:
        key = tuple(row[k] for k in KEY)
        ax.scatter([m['train_kwh'] for _, m, _ in groups[key]], [m['test_top1_pct'] for _, m, _ in groups[key]], alpha=.35, s=14)
        ax.scatter(row['mean_train_kwh'], row['mean_top1_pct'], marker='D', color='black' if row['descriptive_frontier'] else 'grey')
        ax.annotate(f"{row['model']}/{row['variant']}", (row['mean_train_kwh'], row['mean_top1_pct']),
                    xytext=(4, 4), textcoords='offset points', fontsize=8)
    ax.set(xlabel='Training + validation energy (kWh)', ylabel='Test top-1 accuracy (%)',
           title='Individual seeds (dots) and means (diamonds; black = descriptive front)')
    fig.tight_layout()
    fig.savefig(args.out / 'pareto.png', dpi=200)
    fig.savefig(args.out / 'pareto.pdf')
    print('Analysis written:', args.out.resolve())


if __name__ == '__main__':
    main()

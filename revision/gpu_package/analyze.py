"""Summarize completed real runs without pooling hardware/recipe/data cohorts."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics


def interval(values):
    from scipy.stats import t
    n = len(values)
    mean = statistics.mean(values)
    if n < 2:
        return mean, None, None, None
    sd = statistics.stdev(values)
    half = float(t.ppf(.975, n-1)) * sd / n**.5
    return mean, sd, mean-half, mean+half


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('results', type=Path)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    groups, raw, seen = defaultdict(list), [], set()
    for file in args.results.rglob('result.json'):
        r = json.loads(file.read_text(encoding='utf-8'))
        key = (r['cohort'], r['manifest_sha256'], r['model'], r['variant'], r['lr'])
        identity = (*key, r['seed'])
        if identity in seen:
            raise ValueError(f'Duplicate seed/recipe/cohort: {identity}; select one nonoverlapping input directory')
        seen.add(identity)
        groups[key].append(r)
        raw.append({**dict(zip(['cohort', 'manifest_sha256', 'model', 'variant', 'lr'], key)),
                    'seed': r['seed'], 'mode': r['mode'], 'top1_pct': r['test_top1_pct'],
                    'train_kwh': r['train_kwh'],
                    'inference_mwh_per_clip': r['inference_kwh_per_clip'] * 1e6,
                    'source': str(file.resolve())})
    if not raw:
        raise ValueError('No completed result.json files; diagnostics/tuning do not count')
    write_csv(args.out / 'seed_results.csv', raw)
    summaries, scenarios = [], []
    for key, runs in groups.items():
        base = dict(zip(['cohort', 'manifest_sha256', 'model', 'variant', 'lr'], key))
        for metric in ['test_top1_pct', 'train_kwh', 'inference_kwh_per_clip']:
            values = [r[metric] for r in runs if r[metric] is not None]
            if values:
                mean, sd, low, high = interval(values)
                summaries.append({**base, 'metric': metric, 'n_seeds': len(values),
                                  'mean': mean, 'sd': sd, 'ci95_low': low, 'ci95_high': high})
        for intensity in [50, 200, 412.72, 800]:
            for volume in [10000, 100000, 1000000]:
                values = [(r['train_kwh']/volume + r['inference_kwh_per_clip']) * intensity * 1000
                          for r in runs if r['train_kwh'] is not None]
                if values:
                    mean, sd, low, high = interval(values)
                    scenarios.append({**base, 'scenario_g_per_kwh': intensity, 'queries': volume,
                                      'n_seeds': len(values), 'operational_mg_per_query': mean,
                                      'sd': sd, 'ci95_low': low, 'ci95_high': high})
    write_csv(args.out / 'summary.csv', summaries)
    write_csv(args.out / 'illustrative_carbon_scenarios.csv', scenarios)
    # Descriptive frontier on mean training energy and mean accuracy, within cohort/data/recipe LR.
    means = []
    for key, runs in groups.items():
        if all(r['train_kwh'] is not None for r in runs):
            means.append((key, statistics.mean(r['train_kwh'] for r in runs),
                          statistics.mean(r['test_top1_pct'] for r in runs)))
    frontier = []
    for key, energy, acc in means:
        peers = [(k, e, a) for k, e, a in means if (k[0], k[1], k[4]) == (key[0], key[1], key[4])]
        dominated = any(e <= energy and a >= acc and (e < energy or a > acc) for k, e, a in peers)
        frontier.append({'cohort': key[0], 'manifest_sha256': key[1], 'model': key[2],
                         'variant': key[3], 'lr': key[4], 'mean_train_kwh': energy,
                         'mean_top1_pct': acc, 'descriptive_frontier': not dominated})
    write_csv(args.out / 'pareto.csv', frontier)
    if frontier:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        cohorts = sorted(set((r['cohort'], r['manifest_sha256'], r['lr']) for r in frontier))
        for i, cohort in enumerate(cohorts):
            fig, ax = plt.subplots(figsize=(8, 5))
            for row in frontier:
                if (row['cohort'], row['manifest_sha256'], row['lr']) != cohort:
                    continue
                label = row['model'] + '/' + row['variant']
                key = (row['cohort'], row['manifest_sha256'], row['model'], row['variant'], row['lr'])
                xs = [r['train_kwh'] for r in groups[key]]
                ys = [r['test_top1_pct'] for r in groups[key]]
                ax.scatter(xs, ys, alpha=.45)
                ax.scatter(row['mean_train_kwh'], row['mean_top1_pct'], marker='D')
                ax.annotate(label, (row['mean_train_kwh'], row['mean_top1_pct']), xytext=(4,4), textcoords='offset points', fontsize=8)
            ax.set(xlabel='Training + validation energy (kWh)', ylabel='Test top-1 accuracy (%)', title=f'{cohort[0]}: individual seeds; diamonds = means')
            fig.tight_layout()
            fig.savefig(args.out / f'pareto_{i}.png', dpi=200)
            fig.savefig(args.out / f'pareto_{i}.pdf')
            plt.close(fig)
    print('Analysis written:', args.out)
    print('CI: t interval over independent seeds; small-n assumptions apply. Scenarios are illustrative, not observed regional grids.')


if __name__ == '__main__':
    main()

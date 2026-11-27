"""Sequential GPU jobs with durable logs and fail-fast behavior (extension copy:
identical to revision/gpu_package/queue_jobs.py except that it calls run_ext.py).

With --resume: a job whose output directory already contains its completion
marker (result.json / diagnostic_only.json / validation_only.json) is skipped;
a directory WITHOUT a marker is moved aside to `<run_id>.incomplete_<ts>`
(never overwritten, never counted as done) and the job runs fresh from epoch 1,
so single-run training energy is never silently undercounted.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

p = argparse.ArgumentParser()
p.add_argument('jobs', type=Path)
p.add_argument('--config', default='config.json')
p.add_argument('--log-dir', type=Path, default=Path('queue_logs'))
p.add_argument('--resume', action='store_true',
               help='skip completed jobs; move incomplete output dirs aside and rerun them fresh')
args = p.parse_args()
args.log_dir.mkdir(parents=True, exist_ok=False)
print(f'Queue {args.jobs.name}: logs in {args.log_dir}', flush=True)

settings = json.loads(Path(args.config).read_text(encoding='utf-8'))


def run_id_for(job):
    vals = {}
    key = None
    for item in job:
        if key is not None:
            vals[key] = item
            key = None
        elif item.startswith('--'):
            key = item[2:]
    model = vals.get('model', 'mobilenet')
    variant = vals.get('variant', 'baseline')
    seed = vals.get('seed', '42')
    mode = vals.get('mode', 'smoke')
    lr = vals.get('lr')
    rid = f'{model}_{variant}_seed{seed}_{mode}'
    if lr is not None:
        rid += f'_lr{float(lr):g}'
    return rid


def marker_for(job):
    vals = {}
    key = None
    for item in job:
        if key is not None:
            vals[key] = item
            key = None
        elif item.startswith('--'):
            key = item[2:]
    if vals.get('validation-only') is not None:
        return 'validation_only.json'
    if vals.get('mode', 'smoke') in ('smoke', 'overfit'):
        return 'diagnostic_only.json'
    return 'result.json'


for i, job in enumerate(json.loads(args.jobs.read_text(encoding='utf-8'))):
    rid = run_id_for(job)
    out_dir = Path(settings['output_dir']).resolve() / settings['cohort'] / rid
    if args.resume:
        if (out_dir / marker_for(job)).is_file():
            print(f'Job {i+1} [{rid}]: COMPLETE, skipping (marker found)', flush=True)
            continue
        if out_dir.exists():
            aside = out_dir.with_name(out_dir.name + f'.incomplete_{time.strftime("%Y%m%d_%H%M%S")}')
            shutil.move(str(out_dir), str(aside))
            print(f'Job {i+1} [{rid}]: incomplete output moved aside -> {aside.name}', flush=True)
    command = [sys.executable, '-u', str(Path(__file__).with_name('run_ext.py')), '--config', args.config] + job
    print(f'Job {i+1}: {command}', flush=True)
    start = time.time()
    with (args.log_dir / f'{i:02d}.log').open('w', encoding='utf-8') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    print(f'Job {i+1}: exit={result.returncode}, elapsed={(time.time()-start)/60:.1f} min', flush=True)
    if result.returncode:
        raise SystemExit('Stopped on failure. Inspect log; do not silently skip models.')

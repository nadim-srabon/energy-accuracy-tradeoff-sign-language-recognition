"""Preflight for the extension campaign. Run before any GPU job; stops on any FAIL.

Checks that the new runs will be poolable with the 18 completed A4000 runs:
same notebooks, same Python packages, same data cache and selection, same GPU,
and that no planned run already exists. Writes preflight_report.json.
"""
import argparse
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import run_ext

# Packages whose version change would make the new runs a different software cohort.
CRITICAL = {'torch', 'torchvision', 'timm', 'transformers', 'codecarbon', 'numpy',
            'huggingface-hub', 'pillow', 'psutil', 'scipy', 'scikit-learn', 'pandas'}

report = {'pass': [], 'warn': [], 'fail': []}


def ok(msg):
    report['pass'].append(msg)
    print('  PASS ', msg)


def warn(msg):
    report['warn'].append(msg)
    print('  WARN ', msg)


def fail(msg):
    report['fail'].append(msg)
    print('  FAIL ', msg)


def run_id_for(job):
    """Same naming rule as run_ext.py / queue_ext.py."""
    vals, key = {}, None
    for item in job:
        if key is not None:
            vals[key], key = item, None
        elif item.startswith('--'):
            key = item[2:]
            if key == 'validation-only':
                vals[key], key = True, None
    rid = f"{vals.get('model', 'mobilenet')}_{vals.get('variant', 'baseline')}_seed{vals.get('seed', '42')}_{vals.get('mode', 'smoke')}"
    if vals.get('lr') is not None:
        rid += f"_lr{float(vals['lr']):g}"
    return rid


def freeze_map(text):
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if '==' in line:
            name, version = line.split('==', 1)
            out[name.lower().replace('_', '-')] = version
        elif ' @ ' in line:
            name, version = line.split(' @ ', 1)
            out[name.lower().replace('_', '-')] = version
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, default=Path('config.ext.json'))
    p.add_argument('--jobs', type=Path, nargs='+', default=[Path('jobs.newseeds.json'), Path('jobs.newmodels.json')])
    p.add_argument('--report', type=Path, default=Path('preflight_report.json'))
    args = p.parse_args()
    settings = json.loads(args.config.read_text(encoding='utf-8'))

    print('[1] Python and packages')
    ref_dir = Path(settings['reference_results_dir'])
    ref_freeze = ref_dir / 'resnet_baseline_seed42_train' / 'pip-freeze.txt'
    ref_prov = ref_dir / 'resnet_baseline_seed42_train' / 'provenance.json'
    if platform.python_version() != '3.13.17':
        warn(f'Python {platform.python_version()} (first 18 runs used 3.13.17)')
    else:
        ok('Python 3.13.17 (same as first 18 runs)')
    if not ref_freeze.is_file():
        fail(f'Reference pip-freeze.txt not found: {ref_freeze} (check reference_results_dir in config)')
    else:
        now = freeze_map(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
        ref = freeze_map(ref_freeze.read_text(encoding='utf-8', errors='replace'))
        diffs = sorted(k for k in set(now) | set(ref) if now.get(k) != ref.get(k))
        critical = [k for k in diffs if k in CRITICAL]
        for k in critical:
            fail(f'{k}: now {now.get(k)} vs first runs {ref.get(k)}')
        others = [f'{k}: {now.get(k)} vs {ref.get(k)}' for k in diffs if k not in CRITICAL]
        if others:
            warn('non-critical package differences: ' + '; '.join(others))
        if not diffs:
            ok('pip freeze identical to the first 18 runs')
        elif not critical:
            ok('all critical package versions identical to the first 18 runs')

    print('[2] GPU')
    try:
        import torch
        if not torch.cuda.is_available():
            fail('CUDA not available in this Python')
        else:
            name = torch.cuda.get_device_name(settings.get('gpu_id', 0))
            (ok if 'A4000' in name else fail)(f'GPU {settings.get("gpu_id", 0)}: {name}')
            free, total = torch.cuda.mem_get_info(settings.get('gpu_id', 0))
            (ok if free > 14 * 1024**3 else warn)(f'GPU free memory {free/1024**3:.1f} of {total/1024**3:.1f} GiB')
    except Exception as e:  # noqa: BLE001
        fail(f'torch/CUDA check failed: {e}')
    smi = shutil.which('nvidia-smi')
    if smi:
        apps = subprocess.run([smi, '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'],
                              capture_output=True, text=True).stdout.strip()
        (warn if apps else ok)('other GPU compute processes: ' + (apps.replace('\n', ' | ') if apps else 'none'))
        power = subprocess.run([smi, '--query-gpu=power.limit,clocks.max.sm,driver_version', '--format=csv,noheader'],
                               capture_output=True, text=True).stdout.strip()
        report['gpu_power_limit_clock_driver'] = power
        ok(f'nvidia-smi power limit / max SM clock / driver: {power}')
    else:
        warn('nvidia-smi not on PATH; cannot list other GPU processes')
    if ref_prov.is_file():
        hw = json.loads(ref_prov.read_text(encoding='utf-8'))['hardware']
        report['reference_hardware'] = hw

    print('[3] Code')
    for model in run_ext.ALL_MODELS:
        try:
            run_ext.check_notebook(model)
            run_ext.compile_sources(model)
        except Exception as e:  # noqa: BLE001
            fail(f'{model}: {e}')
    ok('notebooks byte-identical to the first 18 runs; all 8 model sources compile')
    original = Path(__file__).resolve().parent.parent / 'gpu_package' / 'run.py'
    if original.is_file():
        import hashlib
        digest = hashlib.sha256(original.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        if digest == '85789f86686ff6568a34d17e3d6ac3f3912668aed7e1d57d13f8f63145d913b6':
            ok('revision/gpu_package/run.py is the version run_ext.py was built from')
        else:
            warn('revision/gpu_package/run.py differs from the copy in revision.zip that run_ext.py was built from; '
                 'send that run.py to the person who built this package before running experiments')
    else:
        warn('revision/gpu_package/run.py not found next to gpu_package_ext (wrong folder?)')

    print('[4] Data and configuration')
    for d in settings['cache_dirs']:
        n = len(list(Path(d).glob('*.npz'))) if Path(d).is_dir() else 0
        (ok if n >= 3544 else fail)(f'cache {d}: {n} .npz files (selection needs 3,544)')
    for name in ['train.csv', 'val.csv', 'test.csv']:
        f = Path(settings['splits_dir']) / name
        (ok if f.is_file() else fail)(f'split file {f}')
    expected = settings.get('expected_manifest_sha256')
    (ok if expected == '62334c358efc8e8c94b09b8f12189d12ce307c3eee2c2f73e940c4ddedcfaa0c' else fail)(
        'expected_manifest_sha256 pinned to the first runs\' selection (62334c35...)')
    if settings['cohort'] != 'local_pc_a4000_rebuiltcache_v1':
        warn(f'cohort is {settings["cohort"]!r}; new seeds will NOT pool with the first 18 runs')
    out_root = (Path(settings['output_dir']) / settings['cohort']).resolve()
    if out_root == ref_dir.resolve():
        fail('output_dir points at the first campaign folder; use a separate folder (D:/nsyss50_results_ext)')
    ref_results = sorted(ref_dir.glob('*/result.json'))
    (ok if len(ref_results) == 18 else warn)(f'{len(ref_results)} completed reference runs in {ref_dir} (expected 18)')

    print('[5] Planned jobs')
    planned = []
    for jobs in args.jobs:
        for job in json.loads(jobs.read_text(encoding='utf-8')):
            planned.append(run_id_for(job))
    dup = sorted({r for r in planned if planned.count(r) > 1})
    (fail if dup else ok)(f'{len(planned)} planned runs, duplicates: {dup or "none"}')
    clash = [r for r in planned if (ref_dir / r).exists()]
    (fail if clash else ok)(f'planned runs already in the first campaign: {clash or "none"}')
    done = [r for r in planned if (out_root / r / 'result.json').is_file()]
    if done:
        ok(f'{len(done)} planned runs already complete in {out_root} (will be skipped by --resume)')

    print('[6] Disk')
    drive = Path(settings['output_dir']).resolve().anchor or str(Path(settings['output_dir']).resolve())
    free_gb = shutil.disk_usage(drive).free / 1024**3
    (ok if free_gb > 20 else fail)(f'free disk on {drive}: {free_gb:.0f} GB (need > 20 GB for checkpoints)')

    args.report.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f"\nPreflight: {len(report['pass'])} pass, {len(report['warn'])} warn, {len(report['fail'])} fail "
          f"-> {args.report}")
    if report['fail']:
        raise SystemExit('Preflight FAILED. Fix the FAIL lines above before running experiments.')


if __name__ == '__main__':
    main()

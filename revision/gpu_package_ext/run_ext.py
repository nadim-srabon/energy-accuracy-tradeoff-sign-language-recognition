"""Extension runner (seed extension + one extra model per family).

Identical to revision/gpu_package/run.py for the five original models; the only
additions are (a) three extra models whose architecture cell comes from
models_extra/*.py while every other cell (CONFIG, data, sampler, training loop,
early stopping, energy tracking, evaluation) comes from a base notebook,
(b) a hard check that the notebooks are byte-identical to the ones used for the
first 18 runs, and (c) extra provenance fields. Run via queue_ext.py.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

FILES = {
    'mobilenet': 'green-computing-over-slr-model-mobilenetv3_v2.ipynb',
    'resnet': 'green-computing-over-slr-model-resnet50_v2.ipynb',
    'vivit': 'green-computing-over-slr-model-small-vivit_v2.ipynb',
    'timesformer': 'green-computing-over-slr-model-timesformer_v2.ipynb',
    'hybrid': 'green-computing-over-slr-model-tinymslr_v2.ipynb',
}


# Notebook SHA-256 recorded in provenance.json of the 18 completed A4000 runs.
EXPECTED_NOTEBOOK_SHA256 = {
    'mobilenet': '9e36816e1b10e874780d9ff547de8a00998364f4693244610b01dd190dd6cf30',
    'resnet': 'cdebfa621a0a38c12102812b3c84d618e52f033ef65f383c09ddf227b0ffc81c',
    'vivit': 'b83daaa8deffc2146fc8393cab2f3bbb4b63f16aa269f090d439ab5f9b7a7532',
    'timesformer': 'b1c79ef8f9e62f7b0da79765d753cdab8b514da4ff1fcd7089ba8fc517d0e707',
    'hybrid': 'd9dcd87bfd7351ce3ffa956b13f4addf06e8ff1798773c992e598fe4ff93d484',
}

# Extra models: architecture cell from models_extra/, everything else from `base`.
EXTRA = {
    'r2plus1d': {'base': 'resnet', 'arch': 'r2plus1d18.py', 'family': 'CNN',
                 'tag': 'r2plus1d18_cnn_video_k400'},
    'swin3d': {'base': 'vivit', 'arch': 'swin3d_t.py', 'family': 'Transformer',
               'tag': 'swin3d_tiny_transformer_video_k400'},
    'mobilevit': {'base': 'mobilenet', 'arch': 'mobilevit_s.py', 'family': 'Hybrid',
                  'tag': 'mobilevit_small_hybrid_light'},
}
MOBILEVIT_ID = 'mobilevit_s.cvnets_in1k'
ALL_MODELS = list(FILES) + list(EXTRA)


def base_of(model):
    return EXTRA[model]['base'] if model in EXTRA else model


def arch_path(model):
    return Path(__file__).parent / 'models_extra' / EXTRA[model]['arch']


def notebook_path(model):
    model = base_of(model)
    local = Path(__file__).parent / 'notebooks' / FILES[model]
    return local if local.exists() else Path(__file__).resolve().parents[2] / FILES[model]


def check_notebook(model):
    digest = hashlib.sha256(notebook_path(model).read_bytes()).hexdigest()
    if digest != EXPECTED_NOTEBOOK_SHA256[base_of(model)]:
        raise ValueError(f'Notebook for {model} differs from the one used by the first 18 runs: '
                         f'{notebook_path(model)} sha256={digest}')
    return digest


def extra_normalization(model):
    """Mean/std the pretrained weights were trained with (read from the library, not guessed)."""
    import torchvision
    if model == 'r2plus1d':
        t = torchvision.models.video.R2Plus1D_18_Weights.KINETICS400_V1.transforms()
        return [float(v) for v in t.mean], [float(v) for v in t.std]
    if model == 'swin3d':
        t = torchvision.models.video.Swin3D_T_Weights.KINETICS400_V1.transforms()
        return [float(v) for v in t.mean], [float(v) for v in t.std]
    if model == 'mobilevit':
        import timm
        data_cfg = timm.data.resolve_model_data_config(timm.create_model(MOBILEVIT_ID, pretrained=False))
        return [float(v) for v in data_cfg['mean']], [float(v) for v in data_cfg['std']]
    raise KeyError(model)


def sources(model):
    nb = json.loads(notebook_path(model).read_text(encoding='utf-8'))
    return [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']


def find(cells, marker):
    hits = [c for c in cells if marker in c]
    if len(hits) != 1:
        raise ValueError(f'Expected one source cell for {marker!r}; found {len(hits)}')
    return hits[0]


def definitions(source):
    tree = ast.parse(source)
    tree.body = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))]
    return ast.unparse(tree)


def compile_sources(model):
    cells = sources(model)
    for i, cell in enumerate(cells):
        compile(cell, f'{model}:cell{i}', 'exec')
    for marker in ('class CONFIG:', 'cache_index = {}', 'class WLASLAppearanceDataset',
                   'n_params_total =', 'def run_epoch('):
        find(cells, marker)
    compile(definitions(find(cells, 'n_params_total =')), model, 'exec')
    if model in EXTRA:
        arch = arch_path(model).read_text(encoding='utf-8')
        compile(arch, f'{model}:arch', 'exec')
        if 'n_params_total =' not in arch or 'model = ' not in arch:
            raise ValueError(f'{arch_path(model)} must define model and n_params_total')
    return cells


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=ALL_MODELS, default='mobilenet')
    p.add_argument('--mode', choices=['smoke', 'overfit', 'train', 'evaluate'], default='smoke')
    p.add_argument('--config', type=Path, default=Path('config.json'))
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--variant', choices=['baseline', 'fixed'], default='baseline')
    p.add_argument('--lr', type=float)
    p.add_argument('--steps', type=int, default=50)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--checkpoint-label-map', type=Path)
    p.add_argument('--validate-source', action='store_true')
    p.add_argument('--validation-only', action='store_true', help='Training/tuning run: do not evaluate test set')
    args = p.parse_args()
    if args.validate_source:
        for model in ALL_MODELS:
            check_notebook(model)
            compile_sources(model)
            print(model, 'notebook hash, source extraction and syntax OK')
        return
    if args.variant != 'baseline' and args.model != 'hybrid':
        p.error('fixed fusion applies only to hybrid')
    if args.mode == 'evaluate' and (not args.checkpoint or not args.checkpoint_label_map):
        p.error('evaluate requires checkpoint and its ORIGINAL class-index map')
    if args.validation_only and args.mode != 'train':
        p.error('validation-only applies to training')
    settings = json.loads(args.config.read_text(encoding='utf-8'))
    if settings.get('inference_repeats', 3) < 3 or settings.get('inference_min_seconds', 60) < 60:
        raise ValueError('Use at least 3 inference windows of at least 60 seconds')
    for d in settings['cache_dirs']:
        if not Path(d).is_dir():
            raise FileNotFoundError(d)
    for name in ['train.csv', 'val.csv', 'test.csv']:
        if not (Path(settings['splits_dir']) / name).is_file():
            raise FileNotFoundError(name)
    check_notebook(args.model)
    cells = compile_sources(args.model)
    run_id = f'{args.model}_{args.variant}_seed{args.seed}_{args.mode}'
    if args.lr is not None:
        run_id += f'_lr{args.lr:g}'
    out = Path(settings['output_dir']).resolve() / settings['cohort'] / run_id
    out.mkdir(parents=True, exist_ok=False)  # Never overwrite an earlier run.
    os.environ['MPLBACKEND'] = 'Agg'
    ns = {'__name__': '__main__'}

    def execute(source):
        exec(compile(source, f'{args.model}:adapter', 'exec'), ns)

    # Skip the notebook's auto-pip-install loop; install requirements explicitly first.
    imports = ast.parse(cells[0])
    imports.body = [n for n in imports.body if not isinstance(n, ast.For)]
    execute(ast.unparse(imports))
    torch, np = ns['torch'], ns['np']
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required. Check torch build and GPU driver.')
    gpu = settings.get('gpu_id', 0)
    torch.cuda.set_device(gpu)
    execute(definitions(find(cells, 'class CONFIG:')))
    cfg = ns['CONFIG']
    cfg.CACHE_DIRS = [Path(x).resolve() for x in settings['cache_dirs']]
    cfg.SPLITS_DIR = Path(settings['splits_dir']).resolve()
    cfg.WORK_DIR = out
    cfg.OUTPUT_DIR = out
    cfg.CHECKPOINT_DIR = out
    cfg.DEVICE = f'cuda:{gpu}'
    cfg.NUM_WORKERS = 0  # Portable on Windows; equal across this new cohort.
    cfg.SEED = args.seed
    cfg.SEEDS = [args.seed]
    if args.lr is not None:
        cfg.LR = args.lr
    if args.model in EXTRA:
        cfg.MODEL_TAG = EXTRA[args.model]['tag']
    ns['seed_everything'](args.seed)
    execute(find(cells, 'HARDWARE_INFO ='))
    # Catch duplicate cache stems before notebook dictionary indexing can hide them.
    stems = set()
    for directory in cfg.CACHE_DIRS:
        for path in directory.glob('*.npz'):
            if path.stem in stems:
                raise ValueError(f'Duplicate cache clip: {path.stem}')
            stems.add(path.stem)
    import csv
    ids = set()
    for name in ['train.csv', 'val.csv', 'test.csv']:
        with (cfg.SPLITS_DIR / name).open(encoding='utf-8', newline='') as f:
            for row in csv.DictReader(f):
                clip = Path(row['Video file']).stem
                if clip in ids:
                    raise ValueError(f'Duplicate clip in split CSVs: {clip}')
                ids.add(clip)
    execute(find(cells, 'cache_index = {}'))
    if ns['NUM_CLASSES'] != 100:
        raise ValueError('Expected exactly 100 selected classes')
    label_map = {str(k): v for k, v in ns['class_list'].items()}
    dump(out / 'label_map.json', label_map)
    manifest = ns['nslt_data']
    dump(out / 'selected_manifest.json', manifest)
    manifest_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    expected = settings.get('expected_manifest_sha256')
    if expected and expected != manifest_hash:
        raise ValueError('Selected manifest differs from the reference cohort')
    # Independently audit explicit signer IDs on the selected subset.
    signer_sets = {s: set() for s in ['train', 'val', 'test']}
    signer_columns = ['Participant ID', 'ParticipantID', 'Participant', 'Signer ID',
                      'SignerID', 'Signer', 'signer_id', 'participant_id']
    for subset in signer_sets:
        with (cfg.SPLITS_DIR / f'{subset}.csv').open(encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            col = next((c for c in signer_columns if c in reader.fieldnames), None)
            if col is None:
                raise ValueError('Explicit signer ID column required; filename heuristic is insufficient')
            for row in reader:
                if Path(row['Video file']).stem in manifest:
                    signer = row[col].strip()
                    if not signer:
                        raise ValueError('Empty signer ID')
                    signer_sets[subset].add(signer)
    if any(signer_sets[a] & signer_sets[b] for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]):
        raise ValueError('Signer overlap in selected subset')
    dump(out / 'signer_audit.json', {k: sorted(v) for k, v in signer_sets.items()})
    dataset_cell = find(cells, 'class WLASLAppearanceDataset')
    normalization = None
    if args.model in EXTRA:
        mean, std = extra_normalization(args.model)
        normalization = {'mean': mean, 'std': std}
        for name, values in [('IMAGENET_MEAN', [0.485, 0.456, 0.406]), ('IMAGENET_STD', [0.229, 0.224, 0.225])]:
            old = f'{name} = torch.tensor({values})'
            if dataset_cell.count(old) != 1:
                raise ValueError(f'Cannot locate {name} in the base dataset cell')
            dataset_cell = dataset_cell.replace(old, f'{name} = torch.tensor({mean if name.endswith("MEAN") else std})')
    execute(dataset_cell)
    for name in ['train_dataset', 'val_dataset', 'test_dataset']:
        if len(ns[name]) == 0:
            raise ValueError(f'Empty {name}')
    if len({label for _, label in ns['train_dataset'].samples}) != 100:
        raise ValueError('Not all selected classes have training clips')
    first_clip = ns['train_dataset'].samples[0][0]
    with np.load(ns['cache_index'][first_clip]) as cache:
        appearance = cache['appearance']
        if appearance.dtype != np.uint8 or appearance.ndim != 5 or appearance.shape[0] != 16 or appearance.shape[-1] != 3:
            raise ValueError('Expected appearance uint8 (16, streams, H, W, 3)')
        import matplotlib.pyplot as plt
        plt.imsave(out / 'training_input_rgb.png', appearance[0, cfg.APPEARANCE_STREAM_IDX])
    arch = find(cells, 'n_params_total =')
    if args.model in EXTRA:
        arch = arch_path(args.model).read_text(encoding='utf-8')
    if args.variant == 'fixed':
        old = 'torch.softmax(self.gate_mlp(f), dim=1)'
        if old not in arch:
            raise ValueError('Cannot locate hybrid gate')
        arch = arch.replace(old, 'torch.full((f.shape[0], 2), 0.5, device=f.device, dtype=f.dtype)')
    execute(arch)
    model = ns['model']
    if args.variant == 'fixed':
        for param in model.gate_mlp.parameters():
            param.requires_grad = False
    config_dump = {k: str(v) if isinstance(v, Path) else v for k, v in vars(cfg).items()
                   if k.isupper() and k != 'CACHE_DIRS'}
    provenance = {'model': args.model, 'variant': args.variant, 'seed': args.seed,
                  'mode': args.mode, 'cohort': settings['cohort'], 'manifest_sha256': manifest_hash,
                  'notebook_sha256': hashlib.sha256(notebook_path(args.model).read_bytes()).hexdigest(),
                  'config': config_dump, 'settings': settings, 'hardware': ns['HARDWARE_INFO'],
                  'trainable_parameters': sum(x.numel() for x in model.parameters() if x.requires_grad),
                  'total_parameters': sum(x.numel() for x in model.parameters()),
                  'trainable_names': [n for n, x in model.named_parameters() if x.requires_grad]}
    provenance['split_clip_counts'] = {s: len(ns[f'{s}_dataset']) for s in ['train', 'val', 'test']}
    provenance['runner'] = 'gpu_package_ext/run_ext.py'
    provenance['runner_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if args.model in EXTRA:
        provenance['extension_model'] = {**EXTRA[args.model],
            'base_notebook': FILES[base_of(args.model)],
            'arch_sha256': hashlib.sha256(arch_path(args.model).read_bytes()).hexdigest(),
            'input_normalization': normalization}
    dump(out / 'provenance.json', provenance)
    (out / 'pip-freeze.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
    train_cell = find(cells, 'def run_epoch(').replace(
        'print("  [warning] non-finite loss, batch skipped")\n                continue',
        'raise FloatingPointError("Non-finite training loss")')
    if args.mode in ['smoke', 'overfit']:
        x, y = next(iter(ns['train_loader']))
        x, y = x.to(cfg.DEVICE), y.to(cfg.DEVICE)
        opt = torch.optim.AdamW(filter(lambda v: v.requires_grad, model.parameters()), lr=cfg.LR)
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=cfg.LABEL_SMOOTHING)
        trace = []
        model.train()
        for step in range(1 if args.mode == 'smoke' else args.steps):
            opt.zero_grad()
            output = model(x)
            logits = output[0] if isinstance(output, tuple) else output
            if tuple(logits.shape) != (len(y), 100) or not torch.isfinite(logits).all():
                raise ValueError('Invalid logits')
            if isinstance(output, tuple):
                gate = output[3]
                if tuple(gate.shape) != (len(y), 2) or not torch.allclose(gate.sum(1), torch.ones(len(y), device=y.device)):
                    raise ValueError('Invalid hybrid gate shape or normalization')
            loss = criterion(logits, y)
            if isinstance(output, tuple):
                loss = loss + cfg.LAMBDA_AUX * (criterion(output[1], y) + criterion(output[2], y))
            loss.backward()
            grads = [v.grad for v in model.parameters() if v.requires_grad and v.grad is not None]
            if not grads or not all(torch.isfinite(g).all() for g in grads):
                raise ValueError('Missing/nonfinite gradients')
            if any(v.grad is not None for v in model.parameters() if not v.requires_grad):
                raise ValueError('Frozen parameter received a gradient')
            tracked = next(v for v in model.parameters() if v.requires_grad and v.grad is not None)
            before = tracked.detach().clone()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP_NORM)
            opt.step()
            if step == 0 and torch.equal(before, tracked):
                raise ValueError('Optimizer did not update a parameter')
            trace.append({'step': step, 'loss': loss.item(), 'train_batch_top1': (logits.argmax(1) == y).float().mean().item()})
        dump(out / 'diagnostic_only.json', trace)
        model.eval()
        with torch.no_grad():
            original = model(x)
            original = original[0] if isinstance(original, tuple) else original
            torch.save(model.state_dict(), out / 'smoke_roundtrip.pt')
            model.load_state_dict(torch.load(out / 'smoke_roundtrip.pt', weights_only=True))
            restored = model(x)
            restored = restored[0] if isinstance(restored, tuple) else restored
            if not torch.allclose(original, restored, rtol=1e-4, atol=1e-5):
                raise ValueError('Checkpoint round-trip changed predictions')
        print('Diagnostic complete; excluded from scientific result tables:', out)
        return
    from codecarbon import EmissionsTracker

    def tracker(**kwargs):
        kwargs.update(gpu_ids=str(gpu), measure_power_secs=1, tracking_mode='machine',
                      save_to_file=True, output_file='emissions.csv', log_level='info')
        return EmissionsTracker(**kwargs)

    ns['EmissionsTracker'], ns['CODECARBON_AVAILABLE'] = tracker, True
    # Per-epoch crash durability: the executed training cell calls _dump_epoch
    # once per epoch (via the anchor patch below). It changes NO computation and
    # NO measurement policy; it writes one JSON line per epoch (~1 KB) so a
    # crashed run still reports its per-epoch energy (reportable wasted energy).
    # Mid-run energy RESUME remains intentionally unsupported: a rerun starts
    # from epoch 1 so training energy is never undercounted.
    ns['_EPOCH_LOG_PATH'] = str(out / 'epoch_log.jsonl')

    def _dump_epoch(path, history, best_val_top1, per_epoch_energy_kwh):
        record = dict(history[-1]) if history else {}
        record['best_val_top1_so_far'] = best_val_top1
        record['n_energy_epochs'] = len(per_epoch_energy_kwh)
        with open(path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, default=str) + '\n')
            f.flush()

    ns['_dump_epoch'] = _dump_epoch
    training_energy = None
    if args.mode == 'train':
        train_cell = find(cells, 'def run_epoch(').replace(
            'print("  [warning] non-finite loss, batch skipped")\n                continue',
            'raise FloatingPointError("Non-finite training loss")')
        anchor = 'is_best = val_metrics["top1"] > best_val_top1'
        if train_cell.count(anchor) != 1:
            raise ValueError('Expected exactly one per-epoch anchor for the epoch-log patch')
        train_cell = train_cell.replace(
            anchor,
            "_dump_epoch(_EPOCH_LOG_PATH, history, best_val_top1, per_epoch_energy_kwh)\n    " + anchor)
        execute(train_cell)
        dump(out / 'training_history.json', ns['history'])
        training_energy = ns['TOTAL_TRAIN_ENERGY_KWH']
        if training_energy is None or training_energy <= 0:
            raise ValueError('No valid training energy')
        if args.validation_only:
            dump(out / 'validation_only.json', {'best_val_top1': ns['best_val_top1'],
                 'train_kwh': training_energy, 'lr': cfg.LR, 'seed': args.seed})
            print('Validation-only tuning complete; test set not evaluated:', out)
            return
    else:
        old_map = json.loads(args.checkpoint_label_map.read_text(encoding='utf-8'))
        if old_map != label_map:
            raise ValueError('Original checkpoint class order differs from current cache selection')
        saved = torch.load(args.checkpoint, map_location=cfg.DEVICE, weights_only=True)
        model.load_state_dict(saved.get('model_state_dict', saved), strict=True)
        dump(out / 'checkpoint_provenance.json', {
            'path': str(args.checkpoint.resolve()),
            'sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            'label_map_path': str(args.checkpoint_label_map.resolve())})
    model.eval()
    # Accuracy and interventions are outside the energy window.
    predictions, labels, gates = [], [], []
    branch_c, branch_s, fixed, feature_norms = [], [], [], []
    with torch.no_grad():
        for x, y in ns['test_loader']:
            x = x.to(cfg.DEVICE)
            output = model(x)
            if isinstance(output, tuple):
                logits, c, s, g = output
                branch_c.extend(c.argmax(1).cpu().tolist())
                branch_s.extend(s.argmax(1).cpu().tolist())
                gates.extend(g.cpu().tolist())
                fc, fs = model.cnn_stream(x), model.transformer_stream(x)
                feature_norms.extend(torch.stack([fc.norm(dim=1), fs.norm(dim=1)], dim=1).cpu().tolist())
                fixed.extend(model.classifier((fc + fs) / 2).argmax(1).cpu().tolist())
            else:
                logits = output
            if not torch.isfinite(logits).all():
                raise ValueError('Nonfinite evaluation logits')
            predictions.extend(logits.topk(5, dim=1).indices.cpu().tolist())
            labels.extend(y.tolist())
    np.savez_compressed(out / 'predictions.npz', labels=labels, top5=predictions,
                        video_ids=[s[0] for s in ns['test_dataset'].samples], gates=gates)
    windows = []
    # Warm real batches, synchronize before and after each full window.
    with torch.no_grad():
        for i, (x, _) in enumerate(ns['test_loader']):
            model(x.to(cfg.DEVICE))
            if i >= 4:
                break
        for repeat in range(settings.get('inference_repeats', 3)):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t = tracker(project_name=f'{run_id}_inference{repeat}', output_dir=str(out), allow_multiple_runs=True)
            t.start()
            start, count = time.perf_counter(), 0
            while time.perf_counter() - start < settings.get('inference_min_seconds', 60):
                for x, _ in ns['test_loader']:
                    model(x.to(cfg.DEVICE))
                    count += len(x)
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            t.stop()
            energy = t.final_emissions_data.energy_consumed
            if energy is None or energy <= 0:
                raise ValueError('Invalid inference energy')
            windows.append({'repeat': repeat, 'clips': count, 'seconds': elapsed,
                            'kwh': energy, 'kwh_per_clip': energy / count,
                            'batch_amortized_ms_per_clip': elapsed * 1000 / count,
                            'peak_allocated_gb': torch.cuda.max_memory_allocated() / 1024**3})
    result = {**{k: provenance[k] for k in ['model', 'variant', 'seed', 'cohort', 'manifest_sha256']},
              'mode': args.mode, 'lr': cfg.LR, 'test_clips': len(labels),
              'test_top1_pct': float(np.mean([p[0] == y for p, y in zip(predictions, labels)]) * 100),
              'test_top5_pct': float(np.mean([y in p for p, y in zip(predictions, labels)]) * 100),
              'train_kwh': training_energy, 'inference_windows': windows,
              'inference_kwh_per_clip': float(np.mean([w['kwh_per_clip'] for w in windows]))}
    if gates:
        result['gate_mean'] = np.mean(gates, axis=0).tolist()
        result['gate_quantiles'] = np.quantile(gates, [0, .25, .5, .75, 1], axis=0).tolist()
        gate_array = np.array(gates)
        result['gate_mean_entropy_nats'] = float(-(gate_array * np.log(gate_array.clip(1e-12))).sum(axis=1).mean())
        result['branch_feature_l2_norm_mean'] = np.mean(feature_norms, axis=0).tolist()
        result['checkpoint_diagnostic_top1_pct'] = {name: float(np.mean(np.array(pred) == labels) * 100)
            for name, pred in [('aux_cnn', branch_c), ('aux_swin', branch_s), ('fixed_fusion', fixed)]}
    dump(out / 'result.json', result)
    print('Complete:', out)


if __name__ == '__main__':
    main()

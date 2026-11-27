"""Print a one-line summary per smoke/overfit diagnostic run (for the human check)."""
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
bad = 0
print(f'{"run":<40} {"params total/trainable":>26} {"loss first -> last":>22} {"batch acc":>9}')
for d in sorted(p for p in root.iterdir() if p.is_dir()):
    diag, prov = d / 'diagnostic_only.json', d / 'provenance.json'
    if not diag.is_file():
        print(f'{d.name:<40} INCOMPLETE (no diagnostic_only.json) -> check the queue log')
        bad += 1
        continue
    trace = json.loads(diag.read_text(encoding='utf-8'))
    p = json.loads(prov.read_text(encoding='utf-8'))
    params = f"{p['total_parameters']:,}/{p['trainable_parameters']:,}"
    loss = f"{trace[0]['loss']:.2f} -> {trace[-1]['loss']:.2f}"
    print(f'{d.name:<40} {params:>26} {loss:>22} {trace[-1]["train_batch_top1"]:>9.2f}')
    if '_overfit' in d.name and trace[-1]['loss'] > 0.5 * trace[0]['loss']:
        print('   ^ overfit loss fell by less than half: inspect before launching experiments')
print('\nSmoke runs: 1 step, loss should be about 4.6-4.9 (ln 100 = 4.61).')
print('Overfit runs: 100 steps on one batch, loss should fall sharply and batch acc approach 1.0.')
print('Also open training_input_rgb.png in any run folder: it must show a signer, in normal colours.')
sys.exit(1 if bad else 0)

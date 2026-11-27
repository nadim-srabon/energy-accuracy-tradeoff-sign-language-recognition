"""CPU tests only: scientific GPU/model validity must be checked on real inputs."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import run
from analyze import interval


class PackageTests(unittest.TestCase):
    def test_original_sources_and_fixed_fusion_compile(self):
        for name in run.FILES:
            cells = run.compile_sources(name)
            training = run.find(cells, 'def run_epoch(')
            self.assertIn('torch.optim.AdamW', training)
            self.assertIn('best_state_dict', training)
        arch = run.find(run.sources('hybrid'), 'n_params_total =')
        needle = 'torch.softmax(self.gate_mlp(f), dim=1)'
        self.assertEqual(arch.count(needle), 1)
        compile(arch.replace(needle, 'torch.full((f.shape[0], 2), 0.5, device=f.device, dtype=f.dtype)'), 'fixed', 'exec')

    def test_ci(self):
        mean, sd, lo, hi = interval([1, 2, 3])
        self.assertEqual(mean, 2)
        self.assertEqual(sd, 1)
        self.assertAlmostEqual(hi-mean, mean-lo)
        self.assertAlmostEqual(hi, 4.4841377117, places=6)
        self.assertEqual(interval([5]), (5, None, None, None))

    def test_synthetic_analysis_and_duplicate_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for model, energy, acc in [('a', .1, 50), ('b', .2, 40)]:
                for seed in [1, 2, 3]:
                    path = root / 'input' / f'{model}_{seed}'
                    path.mkdir(parents=True)
                    result = dict(cohort='SYNTHETIC_TEST_ONLY', manifest_sha256='fixture', model=model,
                                  variant='baseline', lr=.0001, seed=seed, mode='train',
                                  train_kwh=energy, test_top1_pct=acc+seed,
                                  inference_kwh_per_clip=1e-7)
                    run.dump(path / 'result.json', result)
            command = [sys.executable, str(Path(__file__).with_name('analyze.py')), str(root/'input'), '--out', str(root/'analysis')]
            subprocess.run(command, check=True, capture_output=True)
            import csv
            with (root/'analysis'/'pareto.csv').open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual({r['model']: r['descriptive_frontier'] for r in rows}, {'a': 'True', 'b': 'False'})
            duplicate = root/'input'/'duplicate'
            duplicate.mkdir()
            run.dump(duplicate/'result.json', result)
            command[-1] = str(root/'rejected')
            check = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(check.returncode, 0)
            self.assertIn('Duplicate seed', check.stderr)


if __name__ == '__main__':
    unittest.main()

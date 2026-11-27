"""CPU-only checks for the extension package (no GPU, no data needed)."""
import json
from pathlib import Path
import unittest

import run_ext

HERE = Path(__file__).parent


class ExtensionTests(unittest.TestCase):
    def test_notebooks_and_sources(self):
        for model in run_ext.ALL_MODELS:
            run_ext.check_notebook(model)
            cells = run_ext.compile_sources(model)
            self.assertIn('torch.optim.AdamW', run_ext.find(cells, 'def run_epoch('))

    def test_base_dataset_cells_have_normalisation_anchor(self):
        for model in run_ext.EXTRA:
            cell = run_ext.find(run_ext.sources(model), 'class WLASLAppearanceDataset')
            self.assertEqual(cell.count('IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])'), 1)
            self.assertEqual(cell.count('IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])'), 1)

    def test_jobs(self):
        ids = []
        for name in ['jobs.newseeds.json', 'jobs.newmodels.json']:
            for job in json.loads((HERE / name).read_text(encoding='utf-8')):
                vals = dict(zip(job[::2], job[1::2]))
                self.assertIn(vals['--model'], run_ext.ALL_MODELS)
                self.assertEqual(vals['--mode'], 'train')
                ids.append((vals['--model'], vals.get('--variant', 'baseline'), vals['--seed']))
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(ids), 27)
        for model, variant, seed in ids:
            if model in run_ext.FILES:
                self.assertIn(seed, {'7', '999'}, 'old configurations must only get the NEW seeds')


if __name__ == '__main__':
    unittest.main(verbosity=2)

# A4000 cohort: source data for the paper (version 4)

Cohort `local_pc_a4000_rebuiltcache_v1`: NVIDIA RTX A4000, rebuilt full-frame cache,
3,544 clips (1,727/381/1,436), 30/5/11 signers (zero overlap), data-manifest SHA-256
`62334c358efc8e8c94b09b8f12189d12ce307c3eee2c2f73e940c4ddedcfaa0c`.

45 training runs = 9 configurations x 5 seeds (42, 123, 2024, 7, 999):

* First campaign (18-19 Nov 2026, `revision/gpu_package/run.py`): seeds 42/123/2024 of
  mobilenet, resnet, vivit, timesformer, hybrid (learned gate), hybrid (fixed 50:50).
* Extension campaign (22-24 Nov 2026, `revision/gpu_package_ext/run_ext.py`): seeds 7/999
  of those six, and all five seeds of r2plus1d (R(2+1)D-18), swin3d (Video Swin-T) and
  mobilevit (MobileViT-S). Before every run the extension checked that the notebooks were
  byte-identical to the first campaign's, that the pip freeze was identical, and that the
  data manifest hash matched. Extension runs carry `runner` and (for the new models)
  `extension_model` fields in `provenance.json`.

## Files

* `summary.csv`, `seed_results.csv`, `summary_original3seeds.csv`, `pareto.csv`,
  `crossover.csv`, `illustrative_carbon_scenarios.csv`, `checks.txt`: output of
  `analyze_ext.py` run on the GPU PC over both campaign folders (5 seeds per configuration).
  The `source` column of `seed_results.csv` gives each run's original path on the GPU PC.
* `runs/<model>_<variant>_seed<seed>_train/`: per-run `result.json` (top-1/top-5,
  inference windows with clip counts, batch-amortized ms/clip and peak memory, hybrid gate
  statistics and auxiliary-head diagnostics), `training_history.json` (per-epoch accuracy
  and energy), `emissions.csv` (raw CodeCarbon 3.3.2 log, longitude/latitude columns removed for
  publication), `provenance.json`, `signer_audit.json` (signer counts and pairwise overlap;
  participant IDs removed for publication).
* `discarded_runs/`: evidence for the three runs discarded from the extension's first
  session (22-23 Nov): TimeSformer and learned-gate hybrid at seed 7 (completed but with
  per-epoch energy up to 2x and 10x normal) and fixed-fusion hybrid at seed 7 (interrupted
  after 5 epochs). All three were retrained from scratch; the kept versions are in `runs/`.
  Their energy (0.384 + 1.987 + 0.225 = 2.60 kWh) is reported in the paper as overhead.

## Scripts

* `derive5.py summary.csv seed_results.csv runs`: every derived number in the text
  (per-seed comparisons, Pareto fronts, crossovers with seed-pairing ranges, operational
  energy, CO2e scenarios, marginal costs, epoch durations, CPU/GPU/RAM energy shares,
  training fit, gate and auxiliary-head values) and the robustness check without the three
  kept runs from the disturbed session (MobileNetV3-Small, ResNet-50, ViViT-Lite at seed 7).
* `make_rows5.py summary.csv seed_results.csv runs`: the LaTeX rows of Tables 2 and 3.
* `figs5.py <figures-dir> summary.csv`: Figures 2 and 3. Figure 1 is TikZ
  (`sections/figure_architecture.tex`).

Notes: gate index 0 is the CNN (ConvNeXt-GRU) branch. GFLOPs of the three added models in
Table 1 were computed offline with the notebooks' own `thop` call on a (1, 16, 3, 224, 224)
input (the same call reproduces ResNet-50's 132.21 and MobileNetV3-Small's 1.95).
CodeCarbon's IP lookup reported Dhaka Division, Bangladesh for 1,286 of 1,316 log rows and
Quebec, Canada for 30; the paper uses scenario intensities instead of a regional factor.

T4-cohort values come from `../t4_cohort/` (unchanged). Supplementary T4 experiments come
from `../t4_supplementary/`.

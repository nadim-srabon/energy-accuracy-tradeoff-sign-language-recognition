# Quantifying the Carbon Cost of Accuracy in Word-Level Sign Language Recognition

Code, run records and analysis for the paper

> Nadim Srabon, Tonima Islam Lamia, Redoanul Alam Mozumder, and Ahmed Abdal Shafi Rasel. 2026.
> **Quantifying the Carbon Cost of Accuracy: An Energy-Aware Comparison of CNN, Transformer, and Hybrid
> Architectures for Word-Level Sign Language Recognition.** In *13th International Conference on Next
> Generation Computing, Communication, Systems and Security (NSysS 2026)*, December 17–19, 2026,
> Cox's Bazar, Bangladesh. ACM. https://doi.org/10.1145/3872188.3872195

Department of Computer Science and Engineering, East West University, Dhaka, Bangladesh.

---

## What the study does

Word-level sign language recognition (WLSR) is usually evaluated by accuracy alone. This study measures
what that accuracy costs in energy and carbon. Eight configurations are trained on a 100-class,
signer-disjoint pilot of [ASL Citizen](https://www.microsoft.com/en-us/research/project/asl-citizen/)
under one shared recipe, with five seeds each, and their training and inference energy is tracked with
[CodeCarbon](https://github.com/mlco2/codecarbon) on one NVIDIA RTX A4000:

| Family | Light | Heavy | Kinetics-400-pretrained, similar size |
|---|---|---|---|
| CNN | MobileNetV3-Small | ResNet-50 | R(2+1)D-18 |
| Transformer | ViViT-Lite | TimeSformer | Video Swin-T |
| Hybrid | MobileViT-S | ConvNeXt–GRU/Swin (TinyMSLR-inspired, non-distilled; learned or fixed 50:50 fusion) | – |

## Main results (A4000 cohort, mean ± SD over 5 seeds)

| Configuration | Top-1 (%) | Training energy (kWh) | Inference energy (mWh/clip) |
|---|---|---|---|
| MobileNetV3-Small | 6.39 ± 2.04 | 0.093 ± 0.021 | 0.781 ± 0.065 |
| ResNet-50 | 46.55 ± 1.85 | 0.484 ± 0.010 | 1.425 ± 0.127 |
| R(2+1)D-18 | 39.12 ± 1.86 | 0.233 ± 0.011 | 2.796 ± 0.024 |
| ViViT-Lite | 6.48 ± 1.21 | 0.070 ± 0.014 | 1.049 ± 0.004 |
| TimeSformer | 56.74 ± 2.99 | 0.267 ± 0.048 | 3.856 ± 0.144 |
| **Video Swin-T** | **65.46 ± 4.12** | 0.157 ± 0.032 | 2.013 ± 0.019 |
| MobileViT-S | 8.25 ± 2.27 | 0.407 ± 0.068 | 1.425 ± 0.050 |
| Hybrid, learned gate | 26.55 ± 4.88 | 0.289 ± 0.047 | 4.051 ± 0.023 |
| Hybrid, fixed 50:50 | 30.88 ± 1.56 | 0.336 ± 0.018 | 4.099 ± 0.075 |

* ViViT-Lite and Video Swin-T form the accuracy–training-energy Pareto front.
* With matched Kinetics-400 pretraining, Video Swin-T was 16–31 points more accurate than R(2+1)D-18 at every seed.
* TimeSformer needs 55% of ResNet-50's training energy but 2.7× its energy per clip; their combined
  training-plus-inference energy crosses near 9×10⁴ inferences.
* Fixed 50:50 fusion beat the hybrid's learned gate on all five paired seeds.
* With four data-loader workers instead of one process, ResNet-50's training energy fell by 75% at
  unchanged accuracy (data-loading control), so energy rankings depend on the pipeline.

Carbon is reported only under grid-intensity scenarios (50–800 gCO₂e/kWh); see the paper for details,
uncertainty, and limitations.

---

## Repository map

```
.
├── green-computing-over-slr-model-*_v2.ipynb   the five model notebooks (also the model code used by
│                                                every runner below; keep them at the repository root)
├── green-computing-over-slr-model-6-aggregation.ipynb   T4-cohort multi-seed aggregation
├── kaggle_notebook_links.txt                   original Kaggle notebooks (T4 cohort)
├── run_all_experiments.ps1                     driver of the first A4000 campaign
├── requirements.txt
├── revision/                                   A4000 experiment code (see revision/README.md)
│   ├── local_setup/        cache builder and pretrained-weight prefetch
│   ├── gpu_package/        first campaign (seeds 42, 123, 2024)
│   ├── gpu_package_ext/    seeds 7, 999 and the three added models
│   └── gpu_package_nw4/    data-loading control (num_workers = 4)
├── results/                                    run records and analysis (see results/README.md)
│   ├── a4000_cohort/       45 runs + scripts that recompute every number, table and figure
│   ├── a4000_nw4_control/  10 control runs + derive_nw4.py
│   ├── t4_cohort/          earlier Kaggle T4 cohort tables
│   └── t4_supplementary/   exploratory T4 runs
└── figures/                                    Figures 1–3 of the paper (Figure 1 with its TikZ source)
```

## Reproduce the paper's numbers (no GPU needed)

```bash
pip install numpy scipy matplotlib
cd results/a4000_cohort
python derive5.py summary.csv seed_results.csv runs      # every derived number in the text
python make_rows5.py summary.csv seed_results.csv runs   # rows of Tables 2-3
python figs5.py ../../figures summary.csv                # Figures 2 and 3
cd ../a4000_nw4_control
python derive_nw4.py runs ../a4000_cohort/runs           # Section 4.4 (data-loading control)
```

## Rerun the experiments

1. **Hardware/software used:** one NVIDIA RTX A4000 (16 GB), Windows 11, Python 3.13, PyTorch 2.10.0 +
   CUDA 12.8 (`torch==2.10.0+cu128`, `torchvision==0.25.0+cu128`), then `pip install -r requirements.txt`.
2. **Data:** obtain ASL Citizen from its authors (official split CSVs and videos; not redistributed here),
   then build the 16-frame cache with `revision/local_setup/build_cache_rebuilt.py`. The 100-class selection
   is pinned by the manifest SHA-256 `62334c358efc8e8c94b09b8f12189d12ce307c3eee2c2f73e940c4ddedcfaa0c`;
   every run checks it.
3. **Pretrained weights:** `python revision/local_setup/prefetch_weights.py` (and `prefetch_ext.py` for the added models).
4. **Run:** follow `revision/gpu_package_ext/README.md` (`run_all_ext.ps1 -Phase check`, then
   `-Phase experiments`). Edit the paths in the `config*.json` files first.

Each run writes `result.json`, `training_history.json`, a raw CodeCarbon `emissions.csv`, `provenance.json`
(configuration, data-manifest hash, notebook and runner SHA-256, hardware, trainable parameters) and a
signer audit. Model checkpoints are not included (size).

## Notes on the published data

* `emissions.csv` files are CodeCarbon logs with the `longitude`/`latitude` columns removed (not used by
  any result). The paper does not use CodeCarbon's location-based grid factor.
* `signer_audit.json` files give signer counts per split and the pairwise overlap (zero); the ASL Citizen
  participant IDs were removed for redistribution.
* Three runs from a disturbed session were discarded and retrained; their evidence is in
  `results/a4000_cohort/discarded_runs/`. A first attempt of the data-loading control, in which a software
  fault gave all seeds the same initialization, was quarantined and is not included.

## Licence

* Code: MIT (see `LICENSE`).
* Run records and derived results in `results/` and the figures: CC BY 4.0.
* ASL Citizen is subject to its own terms; no videos or cached frames are included.

## Citation

See `CITATION.cff`, or cite the paper above (DOI `10.1145/3872188.3872195`).

## synthetic-data-evaluation

Information-theoretic evaluation of **synthetic**, **simulated**, and **ground-truth** data, with a focus on fidelity metrics (e.g., Jensen–Shannon divergence, MMD, classifier two-sample tests) across multiple data modalities.

### What’s in this repo

- **Core experiment pipelines (CLI)**: under `experiments/`
  - `experiments/adult_tabular`: GT vs SIM vs SYN (SDV CTGAN/TVAE) on UCI Adult.
  - `experiments/time_series_har`: GT vs SIM (DFM-mosaic) on UCI HAR inertial signals.
  - `experiments/bank_marketing`: currently mirrors the Adult pipeline (see note below).
- **Shared utilities**: under `common/` (metrics, I/O, seeding, visualization).
- **Docs**: LaTeX writeups (and built PDFs) under `docs/` documenting methodology and exact run commands.
- **Tutorial notebook**: `notebooks/synthetic_data_tutorial.ipynb` (intro-style SDV walkthrough).

### Quickstart

Run an experiment from the repo root:

```bash
python -m experiments.adult_tabular.run \
  --seed 42 \
  --n_eval 5000 \
  --sim_mode gaussian_copula \
  --synth ctgan \
  --epochs 300 \
  --batch_size 500 \
  --pac 10 \
  --out results_ctgan.json
```

This writes:
- **Figures** to `experiments/adult_tabular/figures/`
- **Results JSON** to `experiments/adult_tabular/<out>`

### Installation

This repository includes a pinned `requirements.txt` (an environment freeze). On some platforms (e.g., macOS), certain pinned wheels (notably CUDA-related packages) may not install.

- **Option A (use pinned file, if it works on your machine)**:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

- **Option B (minimal install for the CLI experiments)**: install the typical scientific stack + SDV:

```bash
pip install numpy pandas scipy scikit-learn matplotlib sdv
```

### Running experiments

#### `experiments/adult_tabular` (tabular: GT vs SIM vs SYN)

- **Entry point**: `python -m experiments.adult_tabular.run`
- **Key flags**:
  - `--sim_mode {independent,gaussian_copula}`
  - `--synth {ctgan,tvae}`
  - `--epochs`, `--batch_size`, `--pac` (CTGAN requires `batch_size % pac == 0`)
  - `--n_eval` (size for SIM/SYN sampling and GT reference)

#### `experiments/time_series_har` (time-series: GT vs SIM)

- **Entry point**: `python -m experiments.time_series_har.run`
- **Example**:

```bash
python -m experiments.time_series_har.run \
  --seed 42 \
  --n_sim 4000 \
  --sg_window 15 --sg_poly 3 \
  --segments_min 3 --segments_max 6 \
  --seglen_min_frac 0.15 --seglen_max_frac 0.35 \
  --residual_scale 1.0 \
  --reconstruct mul \
  --out results_sim.json
```

Outputs:
- `experiments/time_series_har/figures/` (global overlays + per-class grids)
- `experiments/time_series_har/<out>` (results JSON)

#### Note on `experiments/bank_marketing`

The current `experiments/bank_marketing` code is effectively a copy of `experiments/adult_tabular` (including downloading the Adult dataset). Treat it as **work-in-progress / placeholder** unless updated.

### Repository layout

- `common/`: shared utilities
  - `metrics.py`: JS divergence (histogram), RBF-MMD, C2ST AUC
  - `io.py`: JSON writing + runtime metadata helpers
  - `sampling.py`: global seeding
  - `viz.py`: plotting helpers used by experiments
- `experiments/`: reproducible CLI pipelines (see above)
- `docs/`: LaTeX/PDF writeups (method + reproducible commands)
- `notebooks/`: standalone tutorial notebook(s)
- `artifacts/`, `results/`, `scripts/`: present but currently minimal/empty in this branch

### Related project: privacy–utility notebook pipeline (dev branch only)

There is an additional, closely related notebook-based pipeline on the **`dev` branch** under `synth_data_privacy_utility/`. It contains a single pipeline executed on two datasets (Adult and Bank) with notebooks split by pipeline stage, plus cached outputs.

To view it (without merging into `main`):

```bash
git checkout dev
```

### Reproducibility notes

- Experiments expose a `--seed` flag and use consistent preprocessing (fit on train only; evaluate on a held-out split).
- Metrics are computed in a **common numeric representation** (standardized continuous + one-hot categorical for tabular; standardized windows for HAR).

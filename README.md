## synthetic-data-evaluation

Information-theoretic evaluation of **synthetic**, **simulated**, and **ground-truth** data, with a focus on fidelity metrics (e.g., Jensen–Shannon divergence, MMD, classifier two-sample tests) across multiple data modalities.

### Publications

- **[Add publication title]** — *[Venue]*, [Year]. (`[link]`)

```bibtex
@article{add_key_here,
  title   = {Add Title Here},
  author  = {Add Authors Here},
  journal = {Add Venue Here},
  year    = {20XX},
  url     = {Add URL Here}
}
```

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

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
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


### Repository layout

- `common/`: shared utilities
  - `metrics.py`: JS divergence (histogram), RBF-MMD, C2ST AUC
  - `io.py`: JSON writing + runtime metadata helpers
  - `sampling.py`: global seeding
  - `viz.py`: plotting helpers used by experiments
- `experiments/`: reproducible CLI pipelines
- `docs/`: LaTeX/PDF writeups 
- `notebooks/`: standalone tutorial notebook




#!/usr/bin/env python
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from common.io import save_json, now_iso_pst, runtime_versions, ensure_dir
from common.sampling import set_global_seed
from experiments.adult_tabular.data import prepare_splits, _download_and_extract, _load_dataframe
from experiments.adult_tabular.simulate import simulate_smart
from experiments.adult_tabular.synth_sdv import train_and_sample_sdv
from experiments.adult_tabular.evaluate import js_for_continuous_columns, global_mmd_and_c2st
from experiments.adult_tabular.simulate import fit_marginals

def parse_args():
    p = argparse.ArgumentParser(description="Adult Tabular (SDV): GT vs SIM vs SYN")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_eval", type=int, default=5000, help="samples for SIM/SYN evaluation")

    # Simulator settings
    p.add_argument("--sim_mode", type=str, choices=["independent", "gaussian_copula"],
                   default="gaussian_copula",
                   help="How to simulate from fitted per-feature marginals")

    # SDV synthesizer settings
    p.add_argument("--synth", type=str, choices=["ctgan", "tvae"], default="ctgan")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=500,  
                   help="Training batch size (for CTGAN must be divisible by pac)")
    p.add_argument("--pac", type=int, default=10,
                   help="CTGAN 'pack' size; batch_size must be divisible by this")
    p.add_argument("--enforce_rounding", action="store_true", default=True)
    p.add_argument("--no-enforce_rounding", dest="enforce_rounding", action="store_false")

    p.add_argument("--out", type=str, default="results.json")
    return p.parse_args()

def main():
    args = parse_args()
    set_global_seed(args.seed)

    here = Path(__file__).resolve().parent
    cache_dir = here / "data_cache"
    fig_dir = here / "figures"
    ensure_dir(fig_dir)

    # 1) Load & transform GT (encoders for fair comparisons)
    X_train, y_train, X_eval, y_eval, scaler, ohe, meta = prepare_splits(
        cache_dir, test_size=0.25, seed=args.seed
    )
    d_cont = len(meta["continuous"])

    # 1a) Reconstruct RAW train/eval dataframes (for SDV fitting and SIM in raw space)
    members = _download_and_extract(cache_dir)
    full_df = _load_dataframe(members)
    from sklearn.model_selection import train_test_split
    train_df, eval_df = train_test_split(
        full_df, test_size=0.25, random_state=args.seed, stratify=full_df["income"]
    )
    train_df = train_df.dropna().reset_index(drop=True)
    eval_df  = eval_df.dropna().reset_index(drop=True)


    fits = fit_marginals(train_df, exclude=["income"])
    for col in ["education_num", "hours_per_week", "capital_gain", "capital_loss", "age"]:
        fr = fits[col]
        print(f"[FIT] {col}: kind={fr.kind} notes={getattr(fr, 'notes', '')}")


    # 2) SIM: per-feature fitted marginals
    # We exclude the label from generation; metrics/eval operate on features after encoding.
    sim_df_raw = simulate_smart(
        df_train_raw=train_df,
        n=args.n_eval,
        seed=args.seed + 1,
        mode=args.sim_mode,
        exclude=["income"],
    )

    # Encode SIM with the same scaler+OHE as GT train
    cont_cols = meta["continuous"]
    cat_cols  = meta["categorical"]
    sim_cont = scaler.transform(sim_df_raw[cont_cols].to_numpy(dtype=float))
    sim_cat  = ohe.transform(sim_df_raw[cat_cols])
    X_sim = np.hstack([sim_cont, sim_cat])

    # 3) SYN via SDV (CTGAN/TVAE) trained on RAW train_df
    syn_df_raw, syn_info = train_and_sample_sdv(
        df_train=train_df,
        n_samples=args.n_eval,
        synth=args.synth,
        epochs=args.epochs,
        batch_size=args.batch_size,
        enforce_rounding=args.enforce_rounding,
        verbose=False,
        pac=args.pac,  
    )

    # Guard: ensure SYN has all expected columns (fill sensible defaults if missing)
    for col in cont_cols + cat_cols + ["income"]:
        if col not in syn_df_raw.columns:
            syn_df_raw[col] = np.nan if col in cont_cols else "Unknown"

    # Encode SYN with the same scaler+OHE
    syn_cont = scaler.transform(syn_df_raw[cont_cols].to_numpy(dtype=float))
    syn_cat  = ohe.transform(syn_df_raw[cat_cols])
    X_syn = np.hstack([syn_cont, syn_cat])

    # 4) Reference GT sample for evaluation (size-match)
    n_gt = min(args.n_eval, len(X_eval))
    X_gt_ref = X_eval[:n_gt]

    # 5) Evaluate
    cont_js = js_for_continuous_columns(
        X_gt_ref, X_sim, X_syn,
        cont_idx=list(range(d_cont)), names=meta["continuous"],
        out_dir=fig_dir
    )
    global_scores = global_mmd_and_c2st(X_gt_ref, X_sim, X_syn)

    # 6) Persist JSON
    results = {
        "experiment": {
            "name": "adult_tabular",
            "timestamp": now_iso_pst(),
            "seed": args.seed
        },
        "data": {
            "ground_truth": {
                "type": "external",
                "source": "UCI Adult (1994), cleaned, train/eval split",
                "n_train": int(len(X_train)),
                "n_eval_reference": int(n_gt),
                "preprocess": {"scaler": "standard", "cat_encoding": "onehot"},
                "feature_dim": int(X_train.shape[1])
            },
            "simulated": {
                "generator": f"per_feature_{args.sim_mode}",
                "params": {"exclude": ["income"]},
                "n": int(len(X_sim))
            },
            "synthetic": {
                "library": "SDV",
                "model": args.synth.upper(),
                "train": {
                    "epochs": int(args.epochs),
                    "batch_size": int(args.batch_size),
                    "enforce_rounding": bool(args.enforce_rounding),
                    "pac": int(args.pac) if args.synth == "ctgan" else None
                },
                "sampled_n": int(len(X_syn))
            }
        },
        "metrics": {
            "continuous_js": cont_js,
            "global": global_scores
        },
        "figures": sorted([str(p.name) for p in fig_dir.glob("*.png")]),
        "notes": (
            "SIM: per-feature marginals with optional Gaussian copula for continuous dependence; "
            "SYN: SDV (CTGAN/TVAE) trained on raw Adult; evaluation in standardized+one-hot space."
        ),
        "versions": runtime_versions()
    }

    out_path = here / args.out
    save_json(results, out_path)
    print(f"Saved results -> {out_path}")
    print(f"Saved figures -> {fig_dir}")

if __name__ == "__main__":
    main()

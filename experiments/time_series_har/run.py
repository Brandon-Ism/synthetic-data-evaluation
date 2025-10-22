#!/usr/bin/env python
import argparse, json
from pathlib import Path
import numpy as np

from experiments.time_series_har.data import prepare_windows
from experiments.time_series_har.simulate import simulate_dfm_mosaic, DFMParams
from experiments.time_series_har.evaluate import evaluate_ts

def parse_args():
    p = argparse.ArgumentParser(description="HAR Time-Series: GT vs SIM (DFM-mosaic)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_sim", type=int, default=4000)
    # DFM params (you can expose more if you want)
    p.add_argument("--sg_window", type=int, default=15)
    p.add_argument("--sg_poly", type=int, default=3)
    p.add_argument("--segments_min", type=int, default=3)
    p.add_argument("--segments_max", type=int, default=6)
    p.add_argument("--seglen_min_frac", type=float, default=0.15)
    p.add_argument("--seglen_max_frac", type=float, default=0.35)
    p.add_argument("--residual_scale", type=float, default=1.0)
    p.add_argument("--reconstruct", type=str, choices=["mul","add"], default="mul")
    p.add_argument("--out", type=str, default="results_sim.json")
    return p.parse_args()

def main():
    args = parse_args()
    here = Path(__file__).resolve().parent
    fig_dir = here / "figures"
    out_path = here / args.out

    # 1) GT windows (standardized)
    X_train, y_train, X_eval, y_eval, scaler, meta = prepare_windows(use_cache=True)

    # 2) SIM via DFM-mosaic (class-conditional)
    params = DFMParams(
        sg_window=args.sg_window,
        sg_poly=args.sg_poly,
        segments_min=args.segments_min,
        segments_max=args.segments_max,
        seglen_min_frac=args.seglen_min_frac,
        seglen_max_frac=args.seglen_max_frac,
        reconstruct=args.reconstruct,
        residual_scale=args.residual_scale,
    )
    X_sim, y_sim, sim_info = simulate_dfm_mosaic(
        X_train=X_train, y_train=y_train,
        n_samples=args.n_sim, seed=args.seed,
        params=params
    )

    # 3) Evaluate GT (eval split) vs SIM
    # size-match eval
    n = min(len(X_eval), len(X_sim))
    gt_ref = X_eval[:n]
    sim_ref = X_sim[:n]
    metrics_gt_sim = evaluate_ts(gt_ref, sim_ref, fig_dir=fig_dir)

    # 4) Persist JSON
    results = {
        "experiment": {
            "name": "time_series_har",
            "seed": args.seed,
            "window_len": meta["window_len"],
            "channels": meta["channels"],
            "class_names": meta["class_names"],
        },
        "data": {
            "ground_truth": {"n_eval": int(len(gt_ref))},
            "simulated": {
                "generator": "dfm_mosaic",
                "params": vars(params),
                "n": int(len(sim_ref))
            }
        },
        "metrics": {"gt_vs_sim": metrics_gt_sim},
        "figures": ["psd_overlay_ch0.png", "acf_overlay_ch0.png"]
    }
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results -> {out_path}")
    print(f"Saved figures -> {fig_dir}")

if __name__ == "__main__":
    main()

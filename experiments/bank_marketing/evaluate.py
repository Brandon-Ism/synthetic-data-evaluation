import numpy as np
from pathlib import Path
from common.metrics import js_divergence_1d, mmd_rbf, c2st_auc
from common.viz import hist_three

def js_for_continuous_columns(X_gt, X_sim, X_syn, cont_idx, names, out_dir: Path):
    out = {}
    for i, name in zip(cont_idx, names):
        js_gt_sim = js_divergence_1d(X_gt[:, i], X_sim[:, i])
        js_gt_syn = js_divergence_1d(X_gt[:, i], X_syn[:, i])
        js_sim_syn = js_divergence_1d(X_sim[:, i], X_syn[:, i])
        out[name] = {"gt_vs_sim": js_gt_sim, "gt_vs_syn": js_gt_syn, "sim_vs_syn": js_sim_syn}

        # figure
        fig_path = out_dir / f"hist_{name}.png"
        hist_three(X_gt[:, i], X_sim[:, i], X_syn[:, i], f"{name} (standardized)", fig_path)
    return out

def global_mmd_and_c2st(X_gt, X_sim, X_syn):
    return {
        "mmd_rbf_gt_syn": float(mmd_rbf(X_gt, X_syn)),
        "mmd_rbf_gt_sim": float(mmd_rbf(X_gt, X_sim)),
        "mmd_rbf_sim_syn": float(mmd_rbf(X_sim, X_syn)),
        "c2st_auc_gt_syn": float(c2st_auc(X_gt, X_syn)),
        "c2st_auc_gt_sim": float(c2st_auc(X_gt, X_sim)),
        "c2st_auc_sim_syn": float(c2st_auc(X_sim, X_syn)),
    }

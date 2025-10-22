from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
from scipy.signal import welch
import matplotlib.pyplot as plt

EPS = 1e-12

# ----------------------------
# Helpers
# ----------------------------

def _js(p: np.ndarray, q: np.ndarray, eps: float = EPS) -> float:
    p = np.clip(p, eps, None); p /= p.sum()
    q = np.clip(q, eps, None); q /= q.sum()
    m = 0.5 * (p + q)
    return 0.5 * (np.sum(p * np.log(p / m))) + 0.5 * (np.sum(q * np.log(q / m)))

def _acf(x: np.ndarray, L: int) -> np.ndarray:
    # x: [T], zero-mean; returns lags 0..L
    x = x - x.mean()
    c = np.correlate(x, x, mode="full")
    mid = len(c) // 2
    c = c[mid: mid + L + 1]
    c /= (np.var(x) * len(x) + EPS)
    return c  # c[0] ~ 1

def _rbf_mmd2(X: np.ndarray, Y: np.ndarray, subsample: int = 2000) -> float:
    # X,Y: [N,D] float
    n = min(len(X), len(Y), subsample)
    X = X[:n]; Y = Y[:n]
    Z = np.vstack([X, Y])
    # median heuristic
    d2 = np.sum((Z[:, None, :] - Z[None, :, :]) ** 2, axis=-1)
    med = np.median(d2[np.triu_indices_from(d2, k=1)])
    gamma = 1.0 / (2.0 * (med + EPS))
    # kernel
    def k(a, b): return np.exp(-gamma * np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1))
    Kxx = k(X, X); Kyy = k(Y, Y); Kxy = k(X, Y)
    np.fill_diagonal(Kxx, 0.0); np.fill_diagonal(Kyy, 0.0)
    m = n
    mmd2 = (Kxx.sum() / (m * (m - 1))) + (Kyy.sum() / (m * (m - 1))) - (2.0 * Kxy.mean())
    return float(max(mmd2, 0.0))

# ----------------------------
# Metrics
# ----------------------------

def psd_js_per_channel(Xa: np.ndarray, Xb: np.ndarray, fs: float = 50.0, nperseg: int = 64) -> Dict[str, float]:
    # Xa/Xb: [N,T,C]; compute mean PSD per channel then JS
    N, T, C = Xa.shape
    js = {}
    for c in range(C):
        f, Pa = welch(Xa[:, :, c].reshape(-1, T).T, fs=fs, nperseg=nperseg, axis=0)  # [F, N]
        _, Pb = welch(Xb[:, :, c].reshape(-1, T).T, fs=fs, nperseg=nperseg, axis=0)
        Pa = np.maximum(Pa.mean(axis=1), EPS)
        Pb = np.maximum(Pb.mean(axis=1), EPS)
        Pa /= Pa.sum(); Pb /= Pb.sum()
        js[f"ch{c}"] = _js(Pa, Pb)
    return js

def acf_delta_per_channel(Xa: np.ndarray, Xb: np.ndarray, L: int = 20) -> Dict[str, float]:
    # mean absolute difference of ACF (lags 1..L) per channel
    N, T, C = Xa.shape
    deltas = {}
    for c in range(C):
        acf_a = np.stack([_acf(Xa[i, :, c], L) for i in range(min(N, 512))], axis=0).mean(axis=0)  # [L+1]
        acf_b = np.stack([_acf(Xb[i, :, c], L) for i in range(min(len(Xb), 512))], axis=0).mean(axis=0)
        deltas[f"ch{c}"] = float(np.mean(np.abs(acf_a[1:] - acf_b[1:])))
    return deltas

def corr_delta_fro(Xa: np.ndarray, Xb: np.ndarray) -> float:
    # Flatten time and samples; compare channel correlation matrices
    A = Xa.reshape(-1, Xa.shape[2])  # [N*T, C]
    B = Xb.reshape(-1, Xb.shape[2])
    Ca = np.corrcoef(A, rowvar=False)
    Cb = np.corrcoef(B, rowvar=False)
    return float(np.linalg.norm(Ca - Cb, ord="fro"))

def mmd_rbf_windows(Xa: np.ndarray, Xb: np.ndarray) -> float:
    # Flatten each window to D=T*C
    A = Xa.reshape(len(Xa), -1)
    B = Xb.reshape(len(Xb), -1)
    return _rbf_mmd2(A, B, subsample=2000)

# ----------------------------
# Figures (quick sanity)
# ----------------------------

def save_psd_overlay_fig(Xs: Tuple[np.ndarray, np.ndarray], labels: Tuple[str, str], out_path: Path, fs: float = 50.0, nperseg: int = 64):
    (Xa, Xb) = Xs
    a, b = labels
    T = Xa.shape[1]
    f, Pa = welch(Xa[:, :, 0].reshape(-1, T).T, fs=fs, nperseg=nperseg, axis=0)
    _, Pb = welch(Xb[:, :, 0].reshape(-1, T).T, fs=fs, nperseg=nperseg, axis=0)
    Pa = (Pa.mean(axis=1) + EPS); Pb = (Pb.mean(axis=1) + EPS)
    Pa /= Pa.sum(); Pb /= Pb.sum()
    plt.figure()
    plt.plot(f, Pa, label=f"{a}")
    plt.plot(f, Pb, label=f"{b}")
    plt.xlabel("Hz"); plt.ylabel("Normalized PSD"); plt.title("Channel 0 PSD overlay")
    plt.legend(); plt.tight_layout()
    plt.savefig(out_path); plt.close()

def save_acf_overlay_fig(Xs: Tuple[np.ndarray, np.ndarray], labels: Tuple[str, str], out_path: Path, L: int = 20):
    (Xa, Xb) = Xs
    a, b = labels
    acfa = _acf(Xa[0, :, 0], L); acfb = _acf(Xb[0, :, 0], L)
    lags = np.arange(L + 1)
    plt.figure()
    plt.stem(lags, acfa, basefmt=" ", linefmt="-", markerfmt="o", label=f"{a}")
    plt.stem(lags, acfb, basefmt=" ", linefmt="-", markerfmt="x", label=f"{b}")
    plt.xlabel("Lag"); plt.ylabel("ACF"); plt.title("Channel 0 ACF overlay (single window)")
    plt.legend(); plt.tight_layout()
    plt.savefig(out_path); plt.close()



def save_psd_grid_per_class(Xa, Xb, class_name: str, fig_dir: Path, fs: float = 50.0, nperseg: int = 64):
    """
    Save a 3x3 grid of normalized PSD overlays for all 9 channels: GT vs SIM for one class.
    """
    fig_dir.mkdir(parents=True, exist_ok=True)
    C = Xa.shape[2]
    assert C == 9, "HAR grid assumes 9 channels (3x3)."
    fig, axes = plt.subplots(3, 3, figsize=(10, 8), sharex=True, sharey=True)
    axes = axes.ravel()
    T = Xa.shape[1]
    for c in range(C):
        f, Pa = welch(Xa[:, :, c].reshape(-1, T).T, fs=fs, nperseg=nperseg, axis=0)
        _, Pb = welch(Xb[:, :, c].reshape(-1, T).T, fs=fs, nperseg=nperseg, axis=0)
        Pa = (Pa.mean(axis=1) + EPS); Pb = (Pb.mean(axis=1) + EPS)
        Pa /= Pa.sum(); Pb /= Pb.sum()
        ax = axes[c]
        ax.plot(f, Pa, label="GT")
        ax.plot(f, Pb, label="SIM")
        ax.set_title(f"ch{c}")
    axes[0].legend()
    fig.suptitle(f"PSD overlay — {class_name}")
    fig.tight_layout()
    fig.savefig(fig_dir / f"psd_grid_{class_name}.png")
    plt.close(fig)

def save_acf_grid_per_class(Xa, Xb, class_name: str, fig_dir: Path, L: int = 20):
    """
    Save a 3x3 grid of ACF overlays (lags 0..L) for all 9 channels: GT vs SIM for one class.
    """
    fig_dir.mkdir(parents=True, exist_ok=True)
    C = Xa.shape[2]
    assert C == 9, "HAR grid assumes 9 channels (3x3)."
    fig, axes = plt.subplots(3, 3, figsize=(10, 8), sharex=True, sharey=True)
    axes = axes.ravel()
    lags = np.arange(L + 1)
    for c in range(C):
        acfa = np.stack([_acf(x, L) for x in Xa[:, :, c][:min(len(Xa), 64)]], 0).mean(0)
        acfb = np.stack([_acf(x, L) for x in Xb[:, :, c][:min(len(Xb), 64)]], 0).mean(0)
        ax = axes[c]
        ax.plot(lags, acfa, marker="o", label="GT")
        ax.plot(lags, acfb, marker="x", label="SIM")
        ax.set_title(f"ch{c}")
    axes[0].legend()
    fig.suptitle(f"ACF overlay — {class_name}")
    fig.tight_layout()
    fig.savefig(fig_dir / f"acf_grid_{class_name}.png")
    plt.close(fig)

# ----------------------------
# Orchestrator
# ----------------------------

def evaluate_ts(gt: np.ndarray, sim: np.ndarray, fig_dir: Path) -> Dict:
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    out["psd_js"] = psd_js_per_channel(gt, sim)
    out["acf_delta"] = acf_delta_per_channel(gt, sim, L=20)
    out["corr_delta_fro"] = corr_delta_fro(gt, sim)
    out["mmd_rbf"] = mmd_rbf_windows(gt, sim)
    # quick figures
    save_psd_overlay_fig((gt, sim), ("GT", "SIM"), fig_dir / "psd_overlay_ch0.png")
    save_acf_overlay_fig((gt, sim), ("GT", "SIM"), fig_dir / "acf_overlay_ch0.png")
    return out


def evaluate_ts_by_class(
    gt: np.ndarray, gt_y: np.ndarray,
    sim: np.ndarray, sim_y: np.ndarray,
    class_names: list[str],
    fig_dir: Path,
    per_class_figs: bool = True,
) -> Dict:
    """
    Compute fidelity metrics per class. For each class k:
      - size-match GT_k and SIM_k (downsample to min count)
      - compute psd_js, acf_delta, corr_delta_fro, mmd_rbf
      - (optionally) save 3x3 PSD/ACF grids across channels
    Returns dict[class_name] -> metrics dict
    """
    out: Dict[str, Dict] = {}
    fig_dir.mkdir(parents=True, exist_ok=True)
    classes = np.unique(gt_y)
    for k in classes:
        cname = class_names[int(k)] if int(k) < len(class_names) else f"class_{k}"
        gt_k = gt[gt_y == k]
        sim_k = sim[sim_y == k]
        if len(gt_k) == 0 or len(sim_k) == 0:
            continue
        n = min(len(gt_k), len(sim_k))
        gt_ref = gt_k[:n]
        sim_ref = sim_k[:n]
        metrics = {
            "psd_js": psd_js_per_channel(gt_ref, sim_ref),
            "acf_delta": acf_delta_per_channel(gt_ref, sim_ref, L=20),
            "corr_delta_fro": corr_delta_fro(gt_ref, sim_ref),
            "mmd_rbf": mmd_rbf_windows(gt_ref, sim_ref),
            "n": int(n),
        }
        out[cname] = metrics
        if per_class_figs:
            save_psd_grid_per_class(gt_ref, sim_ref, cname, fig_dir)
            save_acf_grid_per_class(gt_ref, sim_ref, cname, fig_dir)
    return out


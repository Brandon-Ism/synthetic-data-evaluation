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
    plt.stem(lags, acfa, basefmt=" ", linefmt="-", markerfmt="o", label=f"{a}", use_line_collection=True)
    plt.stem(lags, acfb, basefmt=" ", linefmt="-", markerfmt="x", label=f"{b}", use_line_collection=True)
    plt.xlabel("Lag"); plt.ylabel("ACF"); plt.title("Channel 0 ACF overlay (single window)")
    plt.legend(); plt.tight_layout()
    plt.savefig(out_path); plt.close()

# ----------------------------
# Orchestrator
# ----------------------------

def evaluate_ts(gt: np.ndarray, other: np.ndarray, fig_dir: Path) -> Dict:
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    out["psd_js"] = psd_js_per_channel(gt, other)
    out["acf_delta"] = acf_delta_per_channel(gt, other, L=20)
    out["corr_delta_fro"] = corr_delta_fro(gt, other)
    out["mmd_rbf"] = mmd_rbf_windows(gt, other)
    # quick figures
    save_psd_overlay_fig((gt, other), ("GT", "OTHER"), fig_dir / "psd_overlay_ch0.png")
    save_acf_overlay_fig((gt, other), ("GT", "OTHER"), fig_dir / "acf_overlay_ch0.png")
    return out

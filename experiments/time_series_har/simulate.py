from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, List, Optional
import numpy as np
from scipy.signal import savgol_filter

# -----------------------------------------------------------------------------
# Params
# -----------------------------------------------------------------------------

@dataclass
class DFMParams:
    # Savitzky–Golay smoothing
    sg_window: int = 15          # odd, >= poly+2; 11–21 are typical for T=128
    sg_poly: int = 3

    # Segmenting the DFM (number and length of pieces to stitch)
    segments_min: int = 3
    segments_max: int = 6
    seglen_min_frac: float = 0.15   # each segment length in [min_frac*T, max_frac*T]
    seglen_max_frac: float = 0.35

    # Reconstruction: "mul" = mean * (1 + dfm_frac), "add" = mean + dfm_add
    reconstruct: str = "mul"        # "mul" or "add"

    # Residual noise
    residual_mode: str = "window"   # "window" or "segment"
    residual_scale: float = 1.0     # scale residual amplitude

    # Misc
    ensure_pos_eps: float = 1e-6    # to avoid division by zero in fractional DFM
    seam_smooth: bool = True        # apply linear ramp at segment seams

# -----------------------------------------------------------------------------
# Smoothing + residuals
# -----------------------------------------------------------------------------

def _savgol_smooth(X: np.ndarray, win: int, poly: int) -> np.ndarray:
    """
    Savitzky–Golay smoothing per (N,T,C).
    """
    N, T, C = X.shape
    # enforce valid window (odd and <= T)
    win = max(3, min(win + (1 - win % 2), T - (1 - T % 2)))
    poly = min(poly, win - 1)
    out = np.empty_like(X, dtype=float)
    for c in range(C):
        out[:, :, c] = savgol_filter(X[:, :, c], window_length=win, polyorder=poly, axis=1, mode="interp")
    return out

# -----------------------------------------------------------------------------
# Class templates (min, max, mean) and DFM banks
# -----------------------------------------------------------------------------

def _build_class_template(smooth_cls: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given smoothed windows for a class [Nc, T, C], return (min_curve, max_curve, mean_curve) with shape [T, C].
    mean_curve is (min+max)/2 as per the method.
    """
    min_curve = smooth_cls.min(axis=0)      # [T, C]
    max_curve = smooth_cls.max(axis=0)      # [T, C]
    mean_curve = 0.5 * (min_curve + max_curve)
    return min_curve, max_curve, mean_curve

def _compute_dfm_banks(smooth_cls: np.ndarray, mean_curve: np.ndarray, eps: float, mode: str) -> Dict[str, np.ndarray]:
    """
    Build banks of deviations:
      - dfm_add:  additive deviation = smooth - mean
      - dfm_frac: fractional deviation = (smooth - mean) / max(mean, eps)
    Return dict with both (shape [Nc, T, C]).
    """
    dfm_add = smooth_cls - mean_curve[None, :, :]
    denom = np.maximum(np.abs(mean_curve), eps)[None, :, :]
    dfm_frac = dfm_add / denom
    return {"add": dfm_add, "frac": dfm_frac}

# -----------------------------------------------------------------------------
# Segment planning + seam smoothing
# -----------------------------------------------------------------------------

def _plan_segments(T: int, rng: np.random.Generator, kmin: int, kmax: int, lmin_frac: float, lmax_frac: float) -> List[Tuple[int, int]]:
    """
    Decide how to split length T into K random segments with lengths in [lmin, lmax].
    Returns list of (start, end) indices covering [0, T).
    """
    k = int(rng.integers(kmin, kmax + 1))
    Lmin = max(1, int(np.floor(lmin_frac * T)))
    Lmax = max(Lmin, int(np.ceil(lmax_frac * T)))
    # greedily sample lengths, adjusting the last one to hit T exactly
    lengths = []
    remaining = T
    for i in range(k - 1):
        # ensure we leave enough for at least 1 segment
        hi = min(Lmax, remaining - Lmin * (k - 1 - i))
        lo = min(Lmin, hi)
        L = int(rng.integers(lo, hi + 1))
        lengths.append(L)
        remaining -= L
    lengths.append(remaining)
    # build segments
    segs = []
    s = 0
    for L in lengths:
        segs.append((s, s + L))
        s += L
    return segs

def _seam_ramp(segment: np.ndarray, target_start: np.ndarray) -> np.ndarray:
    """
    Apply a linear ramp to the segment so that its first value matches target_start (per channel).
    segment: [L, C], target_start: [C]
    """
    L, C = segment.shape
    delta = (target_start - segment[0])  # [C]
    # ramp from delta at t=0 to 0 at t=L-1
    r = np.linspace(1.0, 0.0, L)[:, None] * delta[None, :]
    return segment + r

# -----------------------------------------------------------------------------
# Mosaic + reconstruct
# -----------------------------------------------------------------------------

def _mosaic_dfm(
    dfm_bank: np.ndarray,  # [Nc, T, C] (either "add" or "frac")
    mean_curve: np.ndarray,  # [T, C]
    residual_bank: np.ndarray,  # residuals [Nc, T, C]
    params: DFMParams,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Create ONE simulated sample of shape [T, C] by stitching DFM segments and reconstructing.
    """
    Nc, T, C = dfm_bank.shape

    # pick a random exemplar index for residuals (window-level residual)
    ridx = int(rng.integers(0, Nc))
    base_residual = residual_bank[ridx] * params.residual_scale  # [T, C]

    # segment plan (shared across channels by design)
    segs = _plan_segments(
        T=T,
        rng=rng,
        kmin=params.segments_min,
        kmax=params.segments_max,
        lmin_frac=params.seglen_min_frac,
        lmax_frac=params.seglen_max_frac
    )

    # build stitched DFM (same segmentation for all channels)
    stitched = np.zeros((T, C), dtype=float)
    prev_end_val = None

    pos = 0
    for (s, e) in segs:
        L = e - s
        # choose a random source window and a random cut with the same length
        src_w = int(rng.integers(0, Nc))
        # allow starting at any index, wrap if needed
        if L <= T:
            start_idx = int(rng.integers(0, T))
            # slice with wrap-around
            idx = (np.arange(L) + start_idx) % T
        else:
            # (shouldn't happen with our segment bounds)
            idx = np.arange(T)

        seg = dfm_bank[src_w, idx, :]  # [L, C]

        # seam smoothing: align the segment's first row to prior end value
        if params.seam_smooth and prev_end_val is not None:
            seg = _seam_ramp(seg, prev_end_val)

        stitched[pos: pos + L, :] = seg
        prev_end_val = seg[-1, :]
        pos += L

    # reconstruction
    if params.reconstruct == "mul":
        # stitched holds fractional DFM; mean * (1 + dfm_frac)
        recon = mean_curve * (1.0 + stitched)
    elif params.reconstruct == "add":
        # stitched holds additive DFM; mean + dfm_add
        recon = mean_curve + stitched
    else:
        raise ValueError("params.reconstruct must be 'mul' or 'add'")

    # add residual noise
    recon_noisy = recon + base_residual
    return recon_noisy

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def simulate_dfm_mosaic(
    X_train: np.ndarray,  # [N, T, C]
    y_train: np.ndarray,  # [N], labels {0..K-1}
    n_samples: int,
    seed: int = 42,
    params: Optional[DFMParams] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Class-conditional DFM-mosaic simulator.
    Returns:
      X_sim: [n_samples, T, C]
      y_sim: [n_samples]
      info : dict of template stats + params for provenance
    """
    if params is None:
        params = DFMParams()

    rng = np.random.default_rng(seed)
    N, T, C = X_train.shape
    classes = np.unique(y_train)
    K = len(classes)

    # smooth and build residuals (once)
    smooth_all = _savgol_smooth(X_train, params.sg_window, params.sg_poly)
    residual_all = X_train - smooth_all

    # ensure mean curve stays positive for fractional deviations (mul mode)
    # We compute per-class template and, if needed, add a positive offset to the
    # smoothed signals (and mean) so denominator is safe.
    X_sim_list = []
    y_sim_list = []
    info: Dict = {"params": asdict(params), "classes": {}}

    # allocate how many samples per class (roughly proportional to class freq)
    counts = np.bincount(y_train, minlength=K)
    p = counts / counts.sum()
    ns_per_class = rng.multinomial(n_samples, p)


    for k_idx, k in enumerate(classes):
        mask = (y_train == k)
        Xc = X_train[mask]              # [Nc, T, C]
        Sc = smooth_all[mask]           # [Nc, T, C]
        Rc = residual_all[mask]         # [Nc, T, C]
        Nc = Xc.shape[0]

        # positivity offset (if needed)
        if params.reconstruct == "mul":
            min_val = Sc.min()
            offset = max(0.0, -(min_val) + params.ensure_pos_eps)
            Sc_pos = Sc + offset
        else:
            offset = 0.0
            Sc_pos = Sc

        # class template & DFM bank
        min_curve, max_curve, mean_curve = _build_class_template(Sc_pos)  # [T, C]
        banks = _compute_dfm_banks(Sc_pos, mean_curve, eps=params.ensure_pos_eps, mode=params.reconstruct)
        # choose which bank to use for stitching
        dfm_bank = banks["frac"] if params.reconstruct == "mul" else banks["add"]  # [Nc, T, C]

        # mosaic class samples
        nk = int(ns_per_class[k_idx])
        for _ in range(nk):
            x_sim = _mosaic_dfm(
                dfm_bank=dfm_bank,
                mean_curve=mean_curve,
                residual_bank=Rc,
                params=params,
                rng=rng,
            )
            # undo offset if applied to smooth/template
            if offset > 0 and params.reconstruct == "mul":
                x_sim = x_sim - offset
            X_sim_list.append(x_sim.astype(np.float32))
            y_sim_list.append(k)

        info["classes"][int(k)] = {
            "n_train": int(Nc),
            "min_curve_min": float(min_curve.min()),
            "max_curve_max": float(max_curve.max()),
            "offset_used": float(offset),
        }

    X_sim = np.stack(X_sim_list, axis=0) if X_sim_list else np.zeros((0, T, C), dtype=np.float32)
    y_sim = np.array(y_sim_list, dtype=int)
    return X_sim, y_sim, info

# -----------------------------------------------------------------------------
# Smoke test
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    from .data import prepare_windows
    Xtr, ytr, Xev, yev, scaler, meta = prepare_windows(use_cache=True)
    params = DFMParams()
    Xs, ys, info = simulate_dfm_mosaic(Xtr, ytr, n_samples=10, seed=123, params=params)
    print("Simulated:", Xs.shape, "Labels:", np.bincount(ys))
    print("Info keys:", info.keys())

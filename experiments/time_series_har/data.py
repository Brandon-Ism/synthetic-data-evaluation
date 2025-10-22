from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# ---------------------------
# Constants / paths
# ---------------------------

HERE = Path(__file__).resolve().parent
DATASET_DIR = HERE / "UCI HAR Dataset"  
CACHE_DIR = HERE / "data_cache"

CHANNELS = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]
CLASS_NAMES = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
               "SITTING", "STANDING", "LAYING"]
WINDOW_LEN = 128

# Map channel
FNAME_MAP = {
    "body_acc_x": "body_acc_x",
    "body_acc_y": "body_acc_y",
    "body_acc_z": "body_acc_z",
    "body_gyro_x": "body_gyro_x",
    "body_gyro_y": "body_gyro_y",
    "body_gyro_z": "body_gyro_z",
    "total_acc_x": "total_acc_x",
    "total_acc_y": "total_acc_y",
    "total_acc_z": "total_acc_z",
}

@dataclass
class ChannelScaler:
    mean_: np.ndarray  # shape [C]
    std_: np.ndarray   # shape [C]

    def transform(self, X: np.ndarray) -> np.ndarray:
        # X shape [N, T, C]
        return (X - self.mean_[None, None, :]) / np.clip(self.std_[None, None, :], 1e-12, None)

# ---------------------------
# Low-level readers
# ---------------------------

def _load_inertial_matrix(txt_path: Path) -> np.ndarray:
    """Load one inertial signal text file -> [N, 128] float array."""

    return pd.read_csv(
        txt_path,
        sep=r"\s+",
        header=None,
        engine="python" 
    ).to_numpy(dtype=float)


def _read_split(split: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read one split ('train' or 'test') and stack 9 channels into [N, 128, 9].
    Returns (X, y) where y is 0..5
    """
    assert split in {"train", "test"}
    base = DATASET_DIR / split
    inertial = base / "Inertial Signals"


    y_path = base / f"y_{split}.txt"
    y = pd.read_csv(y_path, header=None).squeeze("columns").to_numpy(dtype=int) - 1  

    mats = []
    for ch in CHANNELS:
        fname = FNAME_MAP[ch] + f"_{split}.txt"
        arr = _load_inertial_matrix(inertial / fname)  # [N, 128]
        mats.append(arr[:, :, None])  # add channel dim
    X = np.concatenate(mats, axis=2)  # [N, 128, 9]
    return X, y


# ---------------------------
# Public API
# ---------------------------

def prepare_windows(use_cache: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, ChannelScaler, Dict]:
    """
    Load UCI HAR inertial signals as windows and standardize per channel (fit on train only).

    Returns:
      X_train, y_train, X_eval, y_eval, scaler, meta
        X_*: float32 arrays [N, 128, 9]
        y_*: int arrays [N] with labels in {0..5}
        scaler: per-channel z-score scaler (mean_, std_)
        meta: dict with dataset metadata
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_npz = CACHE_DIR / "har_inertial_9ch.npz"

    if use_cache and cache_npz.exists():
        data = np.load(cache_npz, allow_pickle=True)
        X_train = data["X_train"]
        y_train = data["y_train"]
        X_eval  = data["X_eval"]
        y_eval  = data["y_eval"]
        mean_   = data["mean_"]
        std_    = data["std_"]
        scaler = ChannelScaler(mean_=mean_, std_=std_)
        meta = {
            "channels": CHANNELS,
            "class_names": CLASS_NAMES,
            "window_len": WINDOW_LEN,
            "n_classes": len(CLASS_NAMES),
            "train_shape": X_train.shape,
            "eval_shape": X_eval.shape,
        }
        return X_train, y_train, X_eval, y_eval, scaler, meta

    # Read raw splits
    X_train, y_train = _read_split("train")
    X_test,  y_test  = _read_split("test")

    X_eval, y_eval = X_test, y_test

    # Per-channel standardization using TRAIN only
    # Compute mean/std over all train windows & timesteps for each channel
    mean_ = X_train.mean(axis=(0, 1))  # shape [C]
    std_  = X_train.std(axis=(0, 1), ddof=0)
    scaler = ChannelScaler(mean_=mean_, std_=std_)

    X_train = scaler.transform(X_train).astype(np.float32)
    X_eval  = scaler.transform(X_eval).astype(np.float32)

    # Save cache
    np.savez_compressed(
        cache_npz,
        X_train=X_train, y_train=y_train,
        X_eval=X_eval,   y_eval=y_eval,
        mean_=mean_, std_=std_
    )

    meta = {
        "channels": CHANNELS,
        "class_names": CLASS_NAMES,
        "window_len": WINDOW_LEN,
        "n_classes": len(CLASS_NAMES),
        "train_shape": X_train.shape,
        "eval_shape": X_eval.shape,
    }
    return X_train, y_train, X_eval, y_eval, scaler, meta

if __name__ == "__main__":

    Xtr, ytr, Xev, yev, sc, meta = prepare_windows(use_cache=False)
    print("Train:", Xtr.shape, "Eval:", Xev.shape)
    print("Labels:", np.bincount(ytr), np.bincount(yev))
    print("Channels:", meta["channels"])

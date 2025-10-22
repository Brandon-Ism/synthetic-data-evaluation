import io
import zipfile
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .utils import COLUMNS, CONTINUOUS, CATEGORICAL, TARGET, UCI_BASE

def _download_and_extract(cache_dir: Path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / "adult.zip"
    if not zip_path.exists():
        r = requests.get(UCI_BASE, timeout=60)
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            f.write(r.content)
    with zipfile.ZipFile(zip_path, "r") as z:
        # adult.data, adult.test, adult.names are inside
        members = {name: z.read(name) for name in z.namelist() if name.endswith((".data",".test"))}
    return members

def _load_dataframe(members: dict):
    # adult.data has 32561 rows; adult.test has header row and a trailing '.' in labels
    def read_text(name, raw):
        df = pd.read_csv(io.BytesIO(raw), header=None, names=COLUMNS, skipinitialspace=True)
        # strip trailing periods in test labels if any
        if "test" in name:
            df[TARGET] = df[TARGET].astype(str).str.replace(".", "", regex=False)
        return df

    frames = []
    for name, raw in members.items():
        if name.endswith(".data") or name.endswith(".test"):
            frames.append(read_text(name, raw))
    df = pd.concat(frames, ignore_index=True)
    # Clean unknowns
    df.replace("?", pd.NA, inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    # Normalize target labels
    df[TARGET] = df[TARGET].map({">50K":1, ">50K":1, ">50K.":1, "<=50K":0, "<=50K.":0}).fillna(
        df[TARGET].apply(lambda s: 1 if ">50K" in str(s) else 0)
    ).astype(int)
    return df

def prepare_splits(cache_dir: Path, test_size=0.25, seed=42):
    members = _download_and_extract(cache_dir)
    df = _load_dataframe(members)

    # Train/val split
    train_df, eval_df = train_test_split(df, test_size=test_size, random_state=seed, stratify=df[TARGET])

    # Fit transforms on TRAIN ONLY
    scaler = StandardScaler()
    train_cont = scaler.fit_transform(train_df[CONTINUOUS].to_numpy().astype(float))
    eval_cont  = scaler.transform(eval_df[CONTINUOUS].to_numpy().astype(float))

    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    train_cat = ohe.fit_transform(train_df[CATEGORICAL])
    eval_cat  = ohe.transform(eval_df[CATEGORICAL])

    X_train = np.hstack([train_cont, train_cat])
    X_eval  = np.hstack([eval_cont,  eval_cat])
    y_train = train_df[TARGET].to_numpy().astype(int)
    y_eval  = eval_df[TARGET].to_numpy().astype(int)

    meta = {
        "continuous": CONTINUOUS,
        "categorical": CATEGORICAL,
        "ohe_categories_": [list(c) for c in ohe.categories_],
        "scaler_mean_": scaler.mean_.tolist(),
        "scaler_scale_": scaler.scale_.tolist(),
        "feature_dim": X_train.shape[1]
    }
    return X_train, y_train, X_eval, y_eval, scaler, ohe, meta

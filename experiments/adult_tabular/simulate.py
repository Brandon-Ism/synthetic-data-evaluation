# experiments/adult_tabular/simulate.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats

# ----------------------------
# Helpers to detect / fit marginals
# ----------------------------

@dataclass
class FitResult:
    kind: str
    params: Dict[str, Any]   # distribution parameters, bounds, etc.
    notes: str = ""

def _is_binary(series: pd.Series) -> bool:
    vals = series.dropna().unique()
    return len(vals) == 2

def _is_categorical(series: pd.Series) -> bool:
    # treat object/string or low-cardinality integer as categorical
    if series.dtype == "O":
        return True
    nunique = series.nunique(dropna=True)
    return nunique <= 20 and not np.issubdtype(series.dtype, np.floating)

def _is_count(series: pd.Series) -> bool:
    s = series.dropna()
    if s.empty:
        return False
    # integers and non-negative
    return np.issubdtype(s.dtype, np.integer) and s.min() >= 0

def _bounded_continuous(series: pd.Series) -> Optional[Tuple[float, float]]:
    s = series.dropna().astype(float)
    if s.empty:
        return None
    lo, hi = float(s.min()), float(s.max())
    if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
        return lo, hi
    return None

def _aic(loglik: float, k_params: int, n: int) -> float:
    return -2*loglik + 2*k_params

def _fit_continuous_candidates(x: np.ndarray) -> FitResult:
    """
    Fit several continuous distributions and choose by AIC.
    Candidates: normal, lognorm, gamma, expon, t; 
    If variable looks bounded -> also try beta on [min,max] scaling.
    """
    x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        return FitResult(kind="constant", params={"value": 0.0}, notes="empty -> 0")

    # Constant?
    if np.allclose(x, x[0]):
        return FitResult(kind="constant", params={"value": float(x[0])})

    candidates = []
    # Unbounded candidates
    # normal
    mu, sigma = np.mean(x), np.std(x, ddof=1)
    if sigma <= 1e-12:
        return FitResult(kind="constant", params={"value": float(mu)})
    ll_norm = np.sum(stats.norm.logpdf(x, loc=mu, scale=sigma))
    candidates.append(("normal", _aic(ll_norm, 2, n), {"mu": mu, "sigma": sigma}))

    # lognormal (only if positive)
    if np.all(x > 0):
        shape, loc, scale = stats.lognorm.fit(x, floc=0)  # constrain loc=0 often stabilizes
        ll = np.sum(stats.lognorm.logpdf(x, s=shape, loc=loc, scale=scale))
        candidates.append(("lognorm", _aic(ll, 2, n), {"shape": shape, "loc": loc, "scale": scale}))

        # gamma
        a, loc_g, scale_g = stats.gamma.fit(x, floc=0)
        ll = np.sum(stats.gamma.logpdf(x, a=a, loc=loc_g, scale=scale_g))
        candidates.append(("gamma", _aic(ll, 2, n), {"a": a, "loc": loc_g, "scale": scale_g}))

        # expon
        loc_e, scale_e = stats.expon.fit(x)
        ll = np.sum(stats.expon.logpdf(x, loc=loc_e, scale=scale_e))
        candidates.append(("expon", _aic(ll, 1, n), {"loc": loc_e, "scale": scale_e}))

    # student-t
    df, loc_t, scale_t = stats.t.fit(x)
    ll = np.sum(stats.t.logpdf(x, df=df, loc=loc_t, scale=scale_t))
    candidates.append(("student_t", _aic(ll, 3, n), {"df": df, "loc": loc_t, "scale": scale_t}))

    # If bounded, also try beta on [min,max]
    bounds = _bounded_continuous(pd.Series(x))
    if bounds is not None:
        lo, hi = bounds
        if hi > lo:
            y = (x - lo) / (hi - lo)
            # clip to (0,1) open interval to avoid inf log-likelihood
            y = np.clip(y, 1e-6, 1 - 1e-6)
            a, b, loc_b, scale_b = stats.beta.fit(y, floc=0, fscale=1)
            ll = np.sum(stats.beta.logpdf(y, a=a, b=b, loc=0, scale=1))
            candidates.append(("beta_scaled", _aic(ll, 2, n), {"a": a, "b": b, "lo": lo, "hi": hi}))

    # pick min AIC
    name, aic, params = sorted(candidates, key=lambda t: t[1])[0]
    return FitResult(kind=name, params=params)

def _fit_binary(s: pd.Series) -> FitResult:
    p = float(s.mean())
    return FitResult(kind="bernoulli", params={"p": p})

def _fit_categorical(s: pd.Series) -> FitResult:
    vc = s.value_counts(normalize=True, dropna=True)
    categories = vc.index.tolist()
    probs = vc.values.astype(float).tolist()
    return FitResult(kind="categorical", params={"categories": categories, "probs": probs})

def _fit_count(s: pd.Series) -> FitResult:
    x = s.dropna().astype(int).to_numpy()
    n = len(x)
    if n == 0:
        return FitResult(kind="poisson", params={"lam": 0.0}, notes="empty -> 0")
    m = x.mean()
    v = x.var(ddof=1) if n > 1 else 0.0
    if v <= m + 1e-6:
        # Poisson
        return FitResult(kind="poisson", params={"lam": float(max(m, 1e-6))})
    # Negative Binomial (method-of-moments)
    # Var = m + m^2/r  => r = m^2 / (v - m), p = r / (r + m)
    r = (m*m) / max(v - m, 1e-6)
    p = r / (r + m)
    r = max(r, 1e-6); p = np.clip(p, 1e-6, 1 - 1e-6)
    return FitResult(kind="neg_binom", params={"r": float(r), "p": float(p)})

# ----------------------------
# Sampling from fitted marginals
# ----------------------------

def _sample_from_fit(fr: FitResult, n: int, rng: np.random.Generator) -> np.ndarray:
    k = fr.kind
    p = fr.params
    if k == "constant":
        return np.full(n, p["value"])
    if k == "normal":
        return rng.normal(p["mu"], p["sigma"], size=n)
    if k == "lognorm":
        return stats.lognorm(s=p["shape"], loc=p["loc"], scale=p["scale"]).rvs(size=n, random_state=rng)
    if k == "gamma":
        return stats.gamma(a=p["a"], loc=p["loc"], scale=p["scale"]).rvs(size=n, random_state=rng)
    if k == "expon":
        return stats.expon(loc=p["loc"], scale=p["scale"]).rvs(size=n, random_state=rng)
    if k == "student_t":
        return stats.t(df=p["df"], loc=p["loc"], scale=p["scale"]).rvs(size=n, random_state=rng)
    if k == "beta_scaled":
        y = stats.beta(a=p["a"], b=p["b"]).rvs(size=n, random_state=rng)
        return p["lo"] + y * (p["hi"] - p["lo"])
    if k == "bernoulli":
        return rng.binomial(1, p["p"], size=n)
    if k == "categorical":
        cats = np.array(p["categories"], dtype=object)
        probs = np.array(p["probs"], dtype=float)
        idx = rng.choice(len(cats), size=n, p=probs)
        return cats[idx]
    if k == "poisson":
        return rng.poisson(p["lam"], size=n)
    if k == "neg_binom":
        r, prob = p["r"], p["p"]
        # numpy uses (n, p) for number of successes? Use gamma-poisson mixture:
        # NB(r,p) ~ Poisson(Gamma(r, (1-p)/p))
        lam = rng.gamma(shape=r, scale=(1 - prob) / prob, size=n)
        return rng.poisson(lam)
    raise ValueError(f"Unknown fit kind: {k}")

# ----------------------------
# Public API
# ----------------------------

def fit_marginals(
    df_train: pd.DataFrame,
    exclude: Optional[List[str]] = None
) -> Dict[str, FitResult]:
    """
    Fit one distribution per column in the *raw* training dataframe.
    """
    exclude = set(exclude or [])
    fits: Dict[str, FitResult] = {}
    for col in df_train.columns:
        if col in exclude:
            continue
        s = df_train[col]
        try:
            if _is_binary(s):
                fit = _fit_binary(s.astype(int))
            elif _is_categorical(s):
                fit = _fit_categorical(s.astype(str))
            elif _is_count(s):
                fit = _fit_count(s)
            else:
                fit = _fit_continuous_candidates(s.to_numpy(dtype=float))
        except Exception as e:
            # fallback: empirical categorical if disaster happens
            fit = _fit_categorical(s.astype(str))
            fit.notes = f"fallback:{type(e).__name__}"
        fits[col] = fit
    return fits

def sample_independent(
    fits: Dict[str, FitResult],
    n: int,
    seed: int = 42
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {}
    for col, fr in fits.items():
        data[col] = _sample_from_fit(fr, n, rng)
    return pd.DataFrame(data)

def sample_gaussian_copula(
    df_train: pd.DataFrame,
    fits: Dict[str, FitResult],
    n: int,
    seed: int = 42
) -> pd.DataFrame:
    """
    Preserve cross-feature dependence among *continuous-like* columns via a Gaussian copula.

    Steps:
      1) For each continuous-like column, convert training values to uniforms via the fitted CDF (PIT).
      2) Map to normal scores via Phi^{-1}.
      3) Fit correlation matrix on those normal scores.
      4) Sample correlated normals, map back to uniforms, then to original scale via PPF of the fitted marginals.
      5) For categorical/binary/count columns, sample independently from their fitted marginals.
    """
    rng = np.random.default_rng(seed)

    # Identify which columns are continuous-like
    cont_cols = [c for c, fr in fits.items() if fr.kind in
                 {"normal","lognorm","gamma","expon","student_t","beta_scaled","constant"}]

    # 1) PIT on train for cont cols
    def _cdf(fr: FitResult, x: np.ndarray) -> np.ndarray:
        p = fr.params; k = fr.kind
        if k == "constant":
            return np.full_like(x, 0.5, dtype=float)
        if k == "normal":
            return stats.norm(loc=p["mu"], scale=p["sigma"]).cdf(x)
        if k == "lognorm":
            return stats.lognorm(s=p["shape"], loc=p["loc"], scale=p["scale"]).cdf(x)
        if k == "gamma":
            return stats.gamma(a=p["a"], loc=p["loc"], scale=p["scale"]).cdf(x)
        if k == "expon":
            return stats.expon(loc=p["loc"], scale=p["scale"]).cdf(x)
        if k == "student_t":
            return stats.t(df=p["df"], loc=p["loc"], scale=p["scale"]).cdf(x)
        if k == "beta_scaled":
            lo, hi = p["lo"], p["hi"]
            y = np.clip((x - lo) / max(hi - lo, 1e-12), 1e-9, 1 - 1e-9)
            return stats.beta(a=p["a"], b=p["b"]).cdf(y)
        raise ValueError(f"cdf not defined for {k}")

    def _ppf(fr: FitResult, u: np.ndarray) -> np.ndarray:
        p = fr.params; k = fr.kind
        u = np.clip(u, 1e-9, 1 - 1e-9)
        if k == "constant":
            return np.full_like(u, p["value"], dtype=float)
        if k == "normal":
            return stats.norm(loc=p["mu"], scale=p["sigma"]).ppf(u)
        if k == "lognorm":
            return stats.lognorm(s=p["shape"], loc=p["loc"], scale=p["scale"]).ppf(u)
        if k == "gamma":
            return stats.gamma(a=p["a"], loc=p["loc"], scale=p["scale"]).ppf(u)
        if k == "expon":
            return stats.expon(loc=p["loc"], scale=p["scale"]).ppf(u)
        if k == "student_t":
            return stats.t(df=p["df"], loc=p["loc"], scale=p["scale"]).ppf(u)
        if k == "beta_scaled":
            lo, hi = p["lo"], p["hi"]
            y = stats.beta(a=p["a"], b=p["b"]).ppf(u)
            return lo + y * (hi - lo)
        raise ValueError(f"ppf not defined for {k}")

    # build training matrix of normal scores
    Z_cols = []
    for c in cont_cols:
        x = df_train[c].to_numpy(dtype=float)
        u = _cdf(fits[c], x)
        z = stats.norm.ppf(np.clip(u, 1e-9, 1 - 1e-9))
        Z_cols.append(z)
    if Z_cols:
        Z = np.vstack(Z_cols).T  # (n_train, d_cont)
        # correlation estimate (Spearman would require ranks; here we use Pearson on z)
        Sigma = np.cov(Z, rowvar=False)
        # sample
        L = np.linalg.cholesky(Sigma + 1e-6*np.eye(Sigma.shape[0]))
        z_new = rng.normal(size=(n, Z.shape[1])) @ L.T
        # map back
        u_new = stats.norm.cdf(z_new)
        cont_samples = {}
        for j, c in enumerate(cont_cols):
            cont_samples[c] = _ppf(fits[c], u_new[:, j])
    else:
        cont_samples = {}

    # discrete columns independently
    data = {}
    for c, fr in fits.items():
        if c in cont_cols:
            data[c] = cont_samples[c]
        else:
            data[c] = _sample_from_fit(fr, n, rng)

    return pd.DataFrame(data)

# ----------------------------
# High-level convenience
# ----------------------------

def simulate_smart(
    df_train_raw: pd.DataFrame,
    n: int,
    seed: int = 42,
    mode: str = "independent",     # "independent" or "gaussian_copula"
    exclude: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Fit per-feature distributions on RAW train dataframe
    and sample n rows either independently or with a Gaussian copula for continuous columns.
    """
    fits = fit_marginals(df_train_raw, exclude=exclude)
    if mode == "independent":
        return sample_independent(fits, n, seed=seed)
    elif mode == "gaussian_copula":
        return sample_gaussian_copula(df_train_raw, fits, n, seed=seed)
    else:
        raise ValueError("mode must be 'independent' or 'gaussian_copula'")

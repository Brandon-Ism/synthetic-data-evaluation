import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import entropy
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

# --- JS divergence (discrete histograms) ---
def _safe_hist(x, bins, range=None, density=True):
    h, edges = np.histogram(x, bins=bins, range=range, density=density)
    h = np.clip(h, 1e-12, None)
    h = h / h.sum()
    return h, edges

def js_divergence_1d(x, y, bins=50, range=None):
    p, _ = _safe_hist(x, bins=bins, range=range, density=True)
    q, _ = _safe_hist(y, bins=bins, range=range, density=True)
    m = 0.5*(p+q)
    return 0.5*entropy(p, m, base=2) + 0.5*entropy(q, m, base=2)

# --- RBF MMD (unbiased) on vectors ---
def _median_heuristic(X):
    # Use subset if large
    n = min(len(X), 2000)
    idx = np.random.choice(len(X), size=n, replace=False)
    D = cdist(X[idx], X[idx], metric="euclidean")
    med = np.median(D[np.triu_indices_from(D, k=1)])
    if med <= 0 or not np.isfinite(med):
        med = 1.0
    return med

def mmd_rbf(X, Y, gamma=None):
    # X, Y: (n,d) arrays
    if gamma is None:
        sigma = _median_heuristic(np.vstack([X, Y]))
        gamma = 1.0/(2*sigma**2)
    XX = np.exp(-gamma*cdist(X, X, "sqeuclidean"))
    YY = np.exp(-gamma*cdist(Y, Y, "sqeuclidean"))
    XY = np.exp(-gamma*cdist(X, Y, "sqeuclidean"))
    n = len(X)
    m = len(Y)
    # Unbiased estimate
    np.fill_diagonal(XX, 0.0)
    np.fill_diagonal(YY, 0.0)
    term_x = XX.sum()/(n*(n-1))
    term_y = YY.sum()/(m*(m-1))
    term_xy = XY.mean()
    return term_x + term_y - 2*term_xy

# --- C2ST (logistic regression AUC) ---
def c2st_auc(X, Y):
    X = np.asarray(X); Y = np.asarray(Y)
    Z = np.vstack([X, Y])
    y = np.hstack([np.zeros(len(X)), np.ones(len(Y))])
    clf = LogisticRegression(max_iter=200, n_jobs=None)
    clf.fit(Z, y)
    scores = clf.predict_proba(Z)[:,1]
    return float(roc_auc_score(y, scores))

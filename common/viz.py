from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from .io import ensure_dir

def hist_three(x_gt, x_sim, x_syn, title, out_path: Path, bins=50):
    ensure_dir(out_path.parent)
    plt.figure()
    plt.hist(x_gt, bins=bins, alpha=0.5, label="GT", density=True)
    plt.hist(x_sim, bins=bins, alpha=0.5, label="SIM", density=True)
    plt.hist(x_syn, bins=bins, alpha=0.5, label="SYN", density=True)
    plt.title(title); plt.legend(); plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

# experiments/time_series_har/synth_timegan.py
from __future__ import annotations
import math, time
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler


@dataclass
class TGParams:
    seq_len: int = 128
    feat_dim: int = 9
    hidden_dim: int = 64
    z_dim: int = 32
    num_layers: int = 2
    batch_size: int = 128
    epochs: int = 2000
    lr: float = 1e-3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # loss weights
    w_recon: float = 10.0
    w_sup: float = 100.0
    w_adv: float = 1.0
    # discriminator training ratio
    d_steps: int = 1
    g_steps: int = 1
    # gradient clipping
    grad_clip: float = 1.0
    # random seed
    seed: int = 42


def _set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LSTMBlock(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, num_layers=2):
        super().__init__()
        self.rnn = nn.LSTM(input_size=in_dim, hidden_size=hid_dim,
                           num_layers=num_layers, batch_first=True)
        self.proj = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        h, _ = self.rnn(x)
        return self.proj(h)  # [B, T, out_dim]


class Discriminator(nn.Module):
    def __init__(self, in_dim, hid_dim, num_layers=2):
        super().__init__()
        self.rnn = nn.LSTM(input_size=in_dim, hidden_size=hid_dim,
                           num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hid_dim, 1)

    def forward(self, x):
        h, _ = self.rnn(x)
        # score per time step, average over time
        logits = self.fc(h)  # [B, T, 1]
        return logits.mean(dim=1)  # [B, 1]


class TimeGAN(nn.Module):
    """
    Components:
      - Embedder: X -> H
      - Recovery: H -> X_hat
      - Generator: Z -> E
      - Supervisor: H -> H_hat (next-step features)  (and also used on E)
      - Discriminator: judge H_tilde vs H (in data space use recovered X)
    """
    def __init__(self, p: TGParams):
        super().__init__()
        self.p = p
        hd, nl = p.hidden_dim, p.num_layers

        # networks
        self.embedder = LSTMBlock(p.feat_dim, hd, hd, nl)
        self.recovery = LSTMBlock(hd, hd, p.feat_dim, nl)

        self.generator = LSTMBlock(p.z_dim, hd, hd, nl)
        self.supervisor = LSTMBlock(hd, hd, hd, nl)

        self.disc_h = Discriminator(hd, hd, nl)

        # losses
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, x, z):
        # Embed & recover
        h = self.embedder(x)
        x_tilde = self.recovery(h)

        # Supervisor: next-step prediction in latent H
        h_hat = self.supervisor(h)

        # Generator: noise -> latent E, then supervised to H-tilde
        e = self.generator(z)
        h_tilde = self.supervisor(e)  # enforce temporal dynamics
        x_hat = self.recovery(h_tilde)

        return h, x_tilde, h_hat, e, h_tilde, x_hat

    # ----- Loss terms -----
    def recon_loss(self, x, x_tilde):
        return self.mse(x, x_tilde)

    def sup_loss(self, h, h_hat):
        # encourage h_hat to match shifted h (next-step supervision)
        return self.mse(h[:, :-1, :], h_hat[:, 1:, :])

    def adv_loss_g(self, real_h_logits, fake_h_logits):
        # generator wants discriminator to say "real" on fake
        target_real = torch.ones_like(fake_h_logits)
        return self.bce(fake_h_logits, target_real)

    def adv_loss_d(self, real_h_logits, fake_h_logits):
        # discriminator wants real->1, fake->0
        ones = torch.ones_like(real_h_logits)
        zeros = torch.zeros_like(fake_h_logits)
        loss_real = self.bce(real_h_logits, ones)
        loss_fake = self.bce(fake_h_logits, zeros)
        return (loss_real + loss_fake) * 0.5

    # ----- Sampling -----
    def sample(self, n, z_sampler):
        self.eval()
        with torch.no_grad():
            z = z_sampler(n, self.p.seq_len, self.p.z_dim, device=self.p.device)
            e = self.generator(z)
            h_tilde = self.supervisor(e)
            x_hat = self.recovery(h_tilde)
        return x_hat  # [n, T, C]


# -------- Utilities --------

def _batch_iter(X, batch_size, device):
    N = X.shape[0]
    idx = np.arange(N)
    np.random.shuffle(idx)
    for i in range(0, N, batch_size):
        b = idx[i:i + batch_size]
        yield torch.tensor(X[b], dtype=torch.float32, device=device)


def _z_sampler(n, T, z_dim, device):
    # Gaussian noise in latent input space
    return torch.randn(n, T, z_dim, device=device)


# -------- Trainer / API --------

def train_timegan(X_tr_01: np.ndarray, p: TGParams) -> TimeGAN:
    """Train TimeGAN on data scaled to [0,1]."""
    _set_seed(p.seed)
    device = p.device
    model = TimeGAN(p).to(device)

    # opt groups
    opt_e = optim.Adam(list(model.embedder.parameters()) + list(model.recovery.parameters()), lr=p.lr)
    opt_g = optim.Adam(list(model.generator.parameters()) + list(model.supervisor.parameters()), lr=p.lr)
    opt_d = optim.Adam(model.disc_h.parameters(), lr=p.lr)

    X = X_tr_01.astype(np.float32)
    n_batches = math.ceil(len(X) / p.batch_size)

    # Pretrain embedder/recovery (reconstruction) a bit for stability
    pre_epochs = min(200, p.epochs // 5)
    for ep in range(pre_epochs):
        model.train()
        total = 0.0
        for xb in _batch_iter(X, p.batch_size, device):
            opt_e.zero_grad()
            h = model.embedder(xb)
            x_tilde = model.recovery(h)
            loss = model.recon_loss(xb, x_tilde)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), p.grad_clip)
            opt_e.step()
            total += loss.item()
        if (ep + 1) % 50 == 0:
            print(f"[Pre] epoch {ep+1}/{pre_epochs} recon={total/n_batches:.4f}")

    # Joint adversarial training
    for ep in range(p.epochs):
        model.train()
        g_total, d_total = 0.0, 0.0
        for xb in _batch_iter(X, p.batch_size, device):
            # ----- Discriminator steps -----
            for _ in range(p.d_steps):
                opt_d.zero_grad()
                # real latent
                h_real = model.embedder(xb).detach()
                # fake latent
                z = _z_sampler(len(xb), p.seq_len, p.z_dim, device)
                e = model.generator(z).detach()
                h_fake = model.supervisor(e).detach()

                # logits
                d_real = model.disc_h(h_real)
                d_fake = model.disc_h(h_fake)
                d_loss = model.adv_loss_d(d_real, d_fake)
                d_loss.backward()
                nn.utils.clip_grad_norm_(model.disc_h.parameters(), p.grad_clip)
                opt_d.step()
                d_total += d_loss.item()

            # ----- Generator (and embedder/supervisor) steps -----
            for _ in range(p.g_steps):
                opt_g.zero_grad(); opt_e.zero_grad()

                # real path
                h_real = model.embedder(xb)
                x_tilde = model.recovery(h_real)
                h_hat = model.supervisor(h_real)

                # fake path
                z = _z_sampler(len(xb), p.seq_len, p.z_dim, device)
                e = model.generator(z)
                h_fake = model.supervisor(e)
                d_fake = model.disc_h(h_fake)

                # losses
                l_recon = model.recon_loss(xb, x_tilde)
                l_sup = model.sup_loss(h_real, h_hat)
                l_gadv = model.adv_loss_g(None, d_fake)

                loss_g = p.w_recon * l_recon + p.w_sup * l_sup + p.w_adv * l_gadv
                loss_g.backward()
                nn.utils.clip_grad_norm_(model.parameters(), p.grad_clip)
                opt_g.step(); opt_e.step()
                g_total += loss_g.item()

        if (ep + 1) % 50 == 0:
            print(f"[Train] epoch {ep+1}/{p.epochs} g={g_total/max(1,n_batches):.4f} d={d_total/max(1,n_batches):.4f}")

    return model


def train_and_sample_timegan(
    X_train_std: np.ndarray,
    n_samples: int,
    params: TGParams,
) -> Tuple[np.ndarray, Dict]:
    """
    X_train_std: standardized train windows [N, T, C] (your pipeline output).
    Returns:
        X_syn_std: synthetic windows in standardized space [n, T, C]
        info: dict with metadata (scaler mins/maxs, timings, params)
    """
    assert X_train_std.ndim == 3
    N, T, C = X_train_std.shape
    assert T == params.seq_len and C == params.feat_dim

    _set_seed(params.seed)

    # MinMax scale to [0,1] per channel using TRAIN ONLY (fits to standardized train)
    scaler = MinMaxScaler(feature_range=(0, 1))
    X2d = X_train_std.reshape(N * T, C)
    scaler.fit(X2d)
    X_tr_01 = scaler.transform(X2d).reshape(N, T, C)

    t0 = time.time()
    model = train_timegan(X_tr_01, params)
    train_secs = time.time() - t0

    # sample
    with torch.no_grad():
        X_hat_01 = model.sample(n_samples, _z_sampler).cpu().numpy()

    # invert to standardized space to be compatible with your evaluation
    X_hat_std = scaler.inverse_transform(X_hat_01.reshape(-1, C)).reshape(n_samples, T, C)

    info = {
        "timegan": {
            "train_seconds": train_secs,
            "params": vars(params),
        },
        "scaler": {
            "min_": scaler.min_.tolist(),
            "scale_": scaler.scale_.tolist(),
            "data_min_": scaler.data_min_.tolist(),
            "data_max_": scaler.data_max_.tolist(),
        }
    }
    return X_hat_std.astype(np.float32), info

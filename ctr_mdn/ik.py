"""Inverse kinematics: tip pose -> distribution over encoded actuator configurations."""
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .fk_train import get_device, set_seed


def load_ik_data(csv_path):
    """Returns tip poses x (N, 7) and encoded joints y (N, 9)."""
    data = np.loadtxt(csv_path, delimiter=",", dtype=np.float32)
    y = np.ascontiguousarray(data[:, :9])
    x = np.ascontiguousarray(data[:, 9:]).copy()
    q = x[:, 3:7] / (np.linalg.norm(x[:, 3:7], axis=1, keepdims=True) + 1e-8)
    q[q[:, 0] < 0] *= -1.0
    x[:, 3:7] = q
    return x, y


def split(x, y, rows_per_seq, val_seq, test_seq):
    seq = np.arange(len(x)) // rows_per_seq
    tr = (seq != val_seq) & (seq != test_seq)
    va, te = seq == val_seq, seq == test_seq
    return (x[tr], y[tr]), (x[va], y[va]), (x[te], y[te])


class CTRInverseMDN(nn.Module):
    """MLP + mixture of diagonal Gaussians over the 9 encoded joint values."""

    def __init__(self, input_dim=7, hidden_dim=128, n_components=7, output_dim=9, sigma_floor=1e-3):
        super().__init__()
        self.K = n_components
        self.D = output_dim
        self.sigma_floor = sigma_floor

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_pi = nn.Linear(hidden_dim, n_components)
        self.fc_mu = nn.Linear(hidden_dim, n_components * output_dim)
        self.fc_sigma = nn.Linear(hidden_dim, n_components * output_dim)
        self.act = nn.ReLU()

        for layer in (self.fc1, self.fc2, self.fc3):
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)
        for layer in (self.fc_pi, self.fc_mu, self.fc_sigma):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        B = x.shape[0]
        h = self.act(self.fc1(x))
        h = self.act(self.fc2(h))
        h = self.act(self.fc3(h))
        logits = self.fc_pi(h)
        mu = self.fc_mu(h).view(B, self.K, self.D)
        sigma = F.softplus(self.fc_sigma(h).view(B, self.K, self.D)) + self.sigma_floor
        return logits, mu, sigma


def ik_log_prob(logits, mu, sigma, y):
    D = mu.shape[-1]
    z = (y.unsqueeze(1) - mu) / sigma
    const = -0.5 * D * torch.log(torch.tensor(2.0 * math.pi, device=mu.device, dtype=mu.dtype))
    log_comp = const - torch.log(sigma).sum(-1) - 0.5 * (z ** 2).sum(-1)
    return torch.logsumexp(F.log_softmax(logits, -1) + log_comp, dim=1)


@dataclass
class IKConfig:
    csv: str = "data/ctr_data.csv"
    out_dir: str = "runs/ik"
    rows_per_seq: int = 12500
    val_seq: int = 4
    test_seq: int = 7
    n_components: int = 7
    hidden_dim: int = 128
    sigma_floor: float = 1e-3
    batch_size: int = 256
    lr: float = 1e-3
    epochs: int = 200
    patience: int = 20
    seed: int = 1
    device: Optional[str] = "cpu"


def mean_nll(model, loader, device):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            total += -ik_log_prob(*model(xb), yb).sum().item()
            n += len(xb)
    return total / n


def train_ik(cfg: IKConfig, verbose=True):
    device = get_device(cfg.device)
    set_seed(cfg.seed)

    x, y = load_ik_data(cfg.csv)
    (x_tr, y_tr), (x_va, y_va), (x_te, y_te) = split(x, y, cfg.rows_per_seq, cfg.val_seq, cfg.test_seq)
    x_mean = x_tr.mean(0, keepdims=True).astype(np.float32)
    x_std = (x_tr.std(0, keepdims=True) + 1e-8).astype(np.float32)

    def loader(xa, ya, shuffle):
        xa = ((xa - x_mean) / x_std).astype(np.float32)
        ds = TensorDataset(torch.from_numpy(np.ascontiguousarray(xa)), torch.from_numpy(np.ascontiguousarray(ya)))
        return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle)

    train_loader = loader(x_tr, y_tr, True)
    val_loader = loader(x_va, y_va, False)
    test_loader = loader(x_te, y_te, False)

    model = CTRInverseMDN(7, cfg.hidden_dim, cfg.n_components, 9, cfg.sigma_floor).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    out = os.path.join(cfg.out_dir, f"K{cfg.n_components}_H{cfg.hidden_dim}")
    os.makedirs(out, exist_ok=True)
    ckpt = os.path.join(out, "best.pt")

    best, best_epoch, wait, log = float("inf"), 0, 0, []
    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total, n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = -ik_log_prob(*model(xb), yb).mean()
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
            n += len(xb)
        val_nll = mean_nll(model, val_loader, device)
        log.append({"epoch": epoch, "train_nll": total / n, "val_nll": val_nll})
        if verbose and (epoch == 1 or epoch % 10 == 0):
            print(f"epoch {epoch:3d}  train {total / n:8.4f}  val {val_nll:8.4f}", flush=True)

        if val_nll < best:
            best, best_epoch, wait = val_nll, epoch, 0
            torch.save({"model": model.state_dict(), "x_mean": x_mean, "x_std": x_std,
                        "best_epoch": epoch, "config": asdict(cfg)}, ckpt)
        else:
            wait += 1
            if wait >= cfg.patience:
                break

    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["model"])
    test_nll = mean_nll(model, test_loader, device)
    result = {"config": asdict(cfg), "best_epoch": best_epoch, "best_val_nll": best,
              "test_nll": test_nll, "train_time_s": time.time() - t0}
    with open(os.path.join(out, "log.json"), "w") as f:
        json.dump(log, f, indent=1)
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(result, f, indent=2)
    if verbose:
        print(f"\nbest epoch {best_epoch}  val NLL {best:.3f}  test NLL {test_nll:.3f}")
    return result

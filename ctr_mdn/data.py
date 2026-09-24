"""Dataset loading, joint encoding and leave-one-sequence-out splits."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset

# tube lengths [mm] used in the translation mapping M_B
TUBE_LENGTHS = np.array([130.0, 95.0, 50.0])
N_SEQ = 8
N_ROWS = 99_979


def build_MB(L=TUBE_LENGTHS):
    L1, L2, L3 = L
    return np.array([[-L1, 0.0, 0.0],
                     [-L1, L1 - L2, 0.0],
                     [-L1, L1 - L2, L2 - L3]])


def encode_beta(beta):
    # beta = 0.5 * M_B (beta_U + 1)  ->  beta_U = 2 M_B^-1 beta - 1
    return 2.0 * np.linalg.solve(build_MB(), beta.T).T - 1.0


def decode_beta(beta_u):
    return 0.5 * (build_MB() @ (beta_u + 1.0).T).T


def encode_joints(q):
    """[a1, b1, a2, b2, a3, b3] -> [cos a1, sin a1, b1_U, ..., cos a3, sin a3, b3_U]"""
    a = q[:, [0, 2, 4]]
    bu = encode_beta(q[:, [1, 3, 5]])
    cols = []
    for i in range(3):
        cols += [np.cos(a[:, i]), np.sin(a[:, i]), bu[:, i]]
    return np.column_stack(cols)


def build_training_csv(raw_csv, out_csv, n_rows=N_ROWS):
    """Convert the CRL-Dataset-CTCR-Pose csv (40 columns) into the 16-column training file:
    encoded joints (9) followed by the tip pose of the innermost tube [x, y, z, qw, qx, qy, qz]."""
    raw = pd.read_csv(raw_csv, header=None).to_numpy(np.float64)
    if raw.shape[1] != 40:
        raise ValueError(f"expected 40 columns, got {raw.shape[1]}")
    if n_rows is not None:
        raw = raw[:n_rows]
    out = pd.DataFrame(np.hstack([encode_joints(raw[:, 0:6]), raw[:, 33:40]]))
    out.to_csv(out_csv, header=False, index=False, float_format="%.10g")
    return out


class CTRFKDataset(Dataset):
    """Tip-pose dataset with a short actuation-history buffer.

    The input at time t is [q_t, q_{t-1}, ..., q_{t-M}] (encoded joints). The buffer never
    reaches across a sequence boundary: at the beginning of a sequence the missing past
    samples are replaced by the first sample of that sequence. With include_dq=True the
    increments q_{t-i} - q_{t-i-1} are appended as well.
    """

    def __init__(self, csv_path, memory_len=3, include_dq=False, n_seq=N_SEQ):
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(csv_path)
        data = pd.read_csv(csv_path, header=None).to_numpy(np.float32)
        if data.shape[1] != 16:
            raise ValueError(f"expected 16 columns, got {data.shape[1]}")

        N = (len(data) // n_seq) * n_seq
        data = data[:N]
        seq_len = N // n_seq
        q, y = data[:, :9], data[:, 9:].copy()

        idx = np.arange(N)
        seq_start = idx - idx % seq_len
        if memory_len > 0:
            lag = np.maximum(idx[:, None] - np.arange(memory_len + 1)[None, :], seq_start[:, None])
            q_hist = q[lag]
            x = q_hist.reshape(N, -1)
            if include_dq:
                dq = q_hist[:, :-1] - q_hist[:, 1:]
                x = np.concatenate([x, dq.reshape(N, -1)], axis=1)
        else:
            x = q

        self.x = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
        self.y = torch.from_numpy(y)
        # unit quaternions with qw >= 0
        flip = self.y[:, 3] < 0
        self.y[flip, 3:7] *= -1.0
        self.y[:, 3:7] /= self.y[:, 3:7].norm(dim=1, keepdim=True).clamp(min=1e-8)

        self.seq_id = torch.from_numpy(idx // seq_len)
        self.n_seq = n_seq
        self.seq_len = seq_len
        self.input_dim = self.x.shape[1]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def loso_folds(n_seq=N_SEQ):
    # test on sequence s, validate on the previous one
    return [(s, (s - 1) % n_seq) for s in range(n_seq)]


def loso_split(ds, test_seq, val_seq):
    sid = ds.seq_id.numpy()
    train = np.where((sid != test_seq) & (sid != val_seq))[0]
    return (Subset(ds, train),
            Subset(ds, np.where(sid == val_seq)[0]),
            Subset(ds, np.where(sid == test_seq)[0]))


def fit_stats(ds, train_idx):
    i = torch.as_tensor(np.asarray(train_idx), dtype=torch.long)
    x, p = ds.x[i], ds.y[i, :3]
    return {"x_mean": x.mean(0), "x_std": x.std(0, unbiased=False).clamp(min=1e-6),
            "y_pos_mean": p.mean(0), "y_pos_std": p.std(0, unbiased=False).clamp(min=1e-6)}


def norm_x(x, s):
    return (x - s["x_mean"]) / s["x_std"]


def norm_y(y, s):
    y = y.clone()
    y[:, :3] = (y[:, :3] - s["y_pos_mean"]) / s["y_pos_std"]
    return y


def denorm_y(y, s):
    y = y.clone()
    y[..., :3] = y[..., :3] * s["y_pos_std"] + s["y_pos_mean"]
    return y

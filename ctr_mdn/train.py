import json
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import CTRFKDataset, fit_stats, loso_split, norm_x, norm_y
from .metrics import evaluate
from .model import CTRForwardMDN, mdn_nll


@dataclass
class Config:
    csv: str = "data/ctr_data.csv"
    out_dir: str = "runs"
    test_seq: int = 7
    val_seq: int = 6
    memory_len: int = 3
    n_components: int = 5
    hidden_dim: int = 256
    include_dq: bool = False
    diag_floor: float = 1e-3
    batch_size: int = 256
    lr: float = 5e-4
    lr_factor: float = 0.5
    lr_patience: int = 10
    clip_grad: float = 5.0
    epochs: int = 200
    patience: int = 30
    seed: int = 42
    device: Optional[str] = None


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_dir(cfg):
    return os.path.join(cfg.out_dir, f"M{cfg.memory_len}_K{cfg.n_components}_H{cfg.hidden_dim}",
                        f"test{cfg.test_seq}_val{cfg.val_seq}")


def get_device(name=None):
    return torch.device(name or ("cuda" if torch.cuda.is_available() else "cpu"))


def load_checkpoint(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    sd = ck["model"]
    K = sd["fc_pi.weight"].shape[0]
    model = CTRForwardMDN(sd["fc1.weight"].shape[1], sd["fc1.weight"].shape[0], K,
                          sd["fc_mu.weight"].shape[0] // K).to(device)
    model.load_state_dict(sd)
    stats = {k: v.to(device) for k, v in ck["stats"].items()}
    return model, stats, ck


def train(cfg: Config, verbose=True):
    device = get_device(cfg.device)
    set_seed(cfg.seed)

    ds = CTRFKDataset(cfg.csv, cfg.memory_len, cfg.include_dq)
    tr, va, te = loso_split(ds, cfg.test_seq, cfg.val_seq)
    train_loader = DataLoader(tr, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(va, batch_size=cfg.batch_size)
    test_loader = DataLoader(te, batch_size=cfg.batch_size)

    stats = fit_stats(ds, tr.indices)
    stats_dev = {k: v.to(device) for k, v in stats.items()}

    model = CTRForwardMDN(ds.input_dim, cfg.hidden_dim, cfg.n_components, 7, cfg.diag_floor).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=cfg.lr_factor, patience=cfg.lr_patience)

    out = run_dir(cfg)
    os.makedirs(out, exist_ok=True)
    ckpt = os.path.join(out, "best.pt")

    best, best_epoch, wait, log = float("inf"), 0, 0, []
    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = mdn_nll(*model(norm_x(x, stats_dev)), norm_y(y, stats_dev))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
            opt.step()
            total += loss.item() * len(x)
            n += len(x)

        val = evaluate(model, val_loader, stats_dev, device)
        sched.step(val["nll"])
        log.append({"epoch": epoch, "lr": opt.param_groups[0]["lr"], "train_nll": total / n,
                    "val_nll": val["nll"], "val_pos_mm": val["pos_mean_mm"], "val_ang_deg": val["ang_mean_deg"]})
        if verbose and (epoch == 1 or epoch % 10 == 0):
            print(f"epoch {epoch:3d}  train {total / n:8.4f}  val {val['nll']:8.4f}  "
                  f"pos {val['pos_mean_mm']:.3f} mm  ang {val['ang_mean_deg']:.3f} deg", flush=True)

        if val["nll"] < best:
            best, best_epoch, wait = val["nll"], epoch, 0
            torch.save({"model": model.state_dict(), "stats": stats, "best_epoch": epoch,
                        "best_val_nll": best, "config": asdict(cfg)}, ckpt)
        else:
            wait += 1
            if cfg.patience and wait >= cfg.patience:
                break

    model, stats_dev, _ = load_checkpoint(ckpt, device)
    test = evaluate(model, test_loader, stats_dev, device)
    result = {"config": asdict(cfg), "best_epoch": best_epoch, "best_val_nll": best,
              "train_time_s": time.time() - t0, "test": test}
    with open(os.path.join(out, "log.json"), "w") as f:
        json.dump(log, f, indent=1)
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(result, f, indent=2)

    if verbose:
        print(f"\nbest epoch {best_epoch}  val NLL {best:.3f}  test NLL {test['nll']:.3f}  "
              f"pos {test['pos_mean_mm']:.3f} mm ({test['pos_mean_pct']:.2f}%)  ang {test['ang_mean_deg']:.3f} deg")
    return result

"""Evaluate a trained model on its test sequence.

    python evaluate.py --ckpt runs/M3_K5_H256/test7_val6/best.pt
"""
import argparse
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from ctr_mdn.data import CTRFKDataset, loso_split, norm_x
from ctr_mdn.metrics import evaluate, position_uncertainty, risk_coverage
from ctr_mdn.train import get_device, load_checkpoint


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--csv", default="data/ctr_data.csv")
    p.add_argument("--device", default=None)
    args = p.parse_args()
    device = get_device(args.device)

    model, stats, ck = load_checkpoint(args.ckpt, device)
    cfg = ck["config"]
    ds = CTRFKDataset(args.csv, cfg["memory_len"], cfg.get("include_dq", False))
    _, _, te = loso_split(ds, cfg["test_seq"], cfg["val_seq"])
    loader = DataLoader(te, batch_size=256)

    res = evaluate(model, loader, stats, device)

    u = position_uncertainty(model, loader, stats, device)
    out_dir = os.path.dirname(args.ckpt)
    plt.figure(figsize=(6, 4))
    for key, label in [("mix", "mixture mean + mixture covariance"), ("map", "MAP component + its covariance")]:
        c, r, aurc = risk_coverage(u[f"err_{key}"], u[f"u_{key}"])
        res[f"aurc_{key}"] = aurc
        plt.plot(c, r, label=f"{label} (AURC={aurc:.3f})")
    plt.xlabel("Coverage")
    plt.ylabel("Mean position error [mm]")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(out_dir, "risk_coverage.png"), dpi=200, bbox_inches="tight")

    # latency of a single forward pass
    x = norm_x(ds.x[te.indices[:1]].to(device), stats)
    times = []
    model.eval()
    with torch.no_grad():
        for i in range(1100):
            t0 = time.perf_counter()
            model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            if i >= 100:
                times.append(1e3 * (time.perf_counter() - t0))
    res["latency_ms"] = float(np.mean(times))
    res["latency_ms_std"] = float(np.std(times))

    with open(os.path.join(out_dir, "eval.json"), "w") as f:
        json.dump(res, f, indent=2)
    for k, v in res.items():
        print(f"{k:16s} {v:.4f}" if isinstance(v, float) else f"{k:16s} {v}")


if __name__ == "__main__":
    main()

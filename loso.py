"""8-fold leave-one-sequence-out cross-validation for one architecture.

    python loso.py --M 3 --K 5 --H 256
"""
import argparse
import os

import pandas as pd

from ctr_mdn.data import loso_folds
from ctr_mdn.train import Config, train


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/ctr_data.csv")
    p.add_argument("--out_dir", default="runs/loso")
    p.add_argument("--M", type=int, default=3)
    p.add_argument("--K", type=int, default=5)
    p.add_argument("--H", type=int, default=256)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    rows = []
    for t, v in loso_folds():
        res = train(Config(csv=args.csv, out_dir=args.out_dir, memory_len=args.M, n_components=args.K,
                           hidden_dim=args.H, test_seq=t, val_seq=v, device=args.device))
        rows.append({"test_seq": t, "val_seq": v, "best_epoch": res["best_epoch"],
                     "val_nll": res["best_val_nll"], **{f"test_{k}": x for k, x in res["test"].items()}})

    df = pd.DataFrame(rows)
    out = os.path.join(args.out_dir, f"M{args.M}_K{args.K}_H{args.H}", "loso.csv")
    df.to_csv(out, index=False)
    cols = ["val_nll", "test_nll", "test_pos_mean_mm", "test_ang_mean_deg"]
    print(df[["test_seq"] + cols].to_string(index=False))
    print("\nmean:\n" + df[cols].mean().to_string())
    print("std:\n" + df[cols].std().to_string())


if __name__ == "__main__":
    main()

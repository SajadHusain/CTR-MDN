"""Grid search over memory length M, number of components K and hidden width H."""
import argparse
import itertools
import json
import os

import pandas as pd

from ctr_mdn.fk_train import FKConfig, run_dir, train_fk


def int_list(s):
    return [int(v) for v in s.split(",")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/ctr_data.csv")
    p.add_argument("--out_dir", default="runs/fk_sweep")
    p.add_argument("--M", default="0,1,3,5")
    p.add_argument("--K", default="1,3,5,7")
    p.add_argument("--H", default="64,128,256,512")
    p.add_argument("--seq_len", type=int, default=None)
    p.add_argument("--test_seq", type=int, default=7)
    p.add_argument("--val_seq", type=int, default=6)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    for M, K, H in itertools.product(int_list(args.M), int_list(args.K), int_list(args.H)):
        cfg = FKConfig(csv=args.csv, out_dir=args.out_dir, seq_len=args.seq_len, memory_len=M,
                       n_components=K, hidden_dim=H, test_seq=args.test_seq, val_seq=args.val_seq,
                       device=args.device)
        res_file = os.path.join(run_dir(cfg), "results.json")
        if os.path.exists(res_file):
            with open(res_file) as f:
                res = json.load(f)
        else:
            print(f"--- M={M} K={K} H={H}")
            res = train_fk(cfg)
        t = res["test"]
        rows.append({"M": M, "K": K, "H": H, "best_epoch": res["best_epoch"],
                     "val_nll": res["best_val_nll"], "test_nll": t["nll"],
                     "pos_mm": t["pos_mean_mm"], "pos_pct": t["pos_mean_pct"], "ang_deg": t["ang_mean_deg"]})
        pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "sweep.csv"), index=False)

    df = pd.DataFrame(rows).sort_values("val_nll")
    print(df.to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()

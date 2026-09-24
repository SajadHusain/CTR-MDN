"""Grid search over memory length M, number of components K and hidden width H.

    python sweep.py --device cpu
    python sweep.py --all_folds
"""
import argparse
import itertools
import json
import os

import matplotlib.pyplot as plt
import pandas as pd

from ctr_mdn.data import loso_folds
from ctr_mdn.train import Config, run_dir, train


def int_list(s):
    return [int(v) for v in s.split(",")]


def plot(df, path):
    Ms, Ks, Hs = sorted(df.M.unique()), sorted(df.K.unique()), sorted(df.H.unique())
    val = df.groupby(["M", "K", "H"])["val_nll"].mean()
    lo, hi = val.min(), val.max()
    ncols = 2 if len(Ms) > 1 else 1
    nrows = -(-len(Ms) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.4 * nrows), squeeze=False)
    for ax, M in zip(axes.flat, Ms):
        Z = [[val.get((M, K, H), float("nan")) for H in Hs] for K in Ks]
        im = ax.imshow(Z, origin="lower", cmap="Blues_r", vmin=lo, vmax=hi, aspect="auto")
        for i, row in enumerate(Z):
            for j, v in enumerate(row):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=9,
                        color="white" if v - lo < 0.5 * (hi - lo) else "black")
        ax.set_xticks(range(len(Hs)), Hs)
        ax.set_yticks(range(len(Ks)), Ks)
        ax.set_xlabel("Hidden dimension H")
        ax.set_ylabel("Mixture components K")
        ax.set_title(f"Memory length M = {M}")
    for ax in list(axes.flat)[len(Ms):]:
        ax.axis("off")
    fig.colorbar(im, ax=axes, label="Validation NLL")
    fig.savefig(path, dpi=200, bbox_inches="tight")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/ctr_data.csv")
    p.add_argument("--out_dir", default="runs/sweep")
    p.add_argument("--M", default="0,1,3,5")
    p.add_argument("--K", default="1,3,5,7")
    p.add_argument("--H", default="64,128,256,512")
    p.add_argument("--test_seq", type=int, default=7)
    p.add_argument("--val_seq", type=int, default=6)
    p.add_argument("--all_folds", action="store_true")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    folds = loso_folds() if args.all_folds else [(args.test_seq, args.val_seq)]
    rows = []
    for M, K, H in itertools.product(int_list(args.M), int_list(args.K), int_list(args.H)):
        for t, v in folds:
            cfg = Config(csv=args.csv, out_dir=args.out_dir, memory_len=M, n_components=K,
                         hidden_dim=H, test_seq=t, val_seq=v, device=args.device)
            res_file = os.path.join(run_dir(cfg), "results.json")
            if os.path.exists(res_file):
                with open(res_file) as f:
                    res = json.load(f)
            else:
                print(f"--- M={M} K={K} H={H} test={t} val={v}")
                res = train(cfg)
            rows.append({"M": M, "K": K, "H": H, "test_seq": t, "val_seq": v,
                         "best_epoch": res["best_epoch"], "val_nll": res["best_val_nll"],
                         "test_nll": res["test"]["nll"], "pos_mm": res["test"]["pos_mean_mm"],
                         "ang_deg": res["test"]["ang_mean_deg"]})
            pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "sweep.csv"), index=False)

    df = pd.DataFrame(rows)
    print(df.groupby(["M", "K", "H"]).mean(numeric_only=True)
            .sort_values("val_nll")[["val_nll", "test_nll", "pos_mm", "ang_deg"]].head(10))
    plot(df, os.path.join(args.out_dir, "heatmap.png"))


if __name__ == "__main__":
    main()

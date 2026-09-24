"""Grid search over the number of mixture components K and the hidden width H of the IK-MDN.
"""
import argparse
import itertools
import json
import os

import pandas as pd

from ctr_mdn.ik import IKConfig, train_ik


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/ctr_data.csv")
    p.add_argument("--out_dir", default="runs/ik_sweep")
    p.add_argument("--K", default="1,3,5,7")
    p.add_argument("--H", default="64,128,256,512")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    for K, H in itertools.product([int(v) for v in args.K.split(",")], [int(v) for v in args.H.split(",")]):
        res_file = os.path.join(args.out_dir, f"K{K}_H{H}", "results.json")
        if os.path.exists(res_file):
            with open(res_file) as f:
                res = json.load(f)
        else:
            print(f"--- K={K} H={H}")
            res = train_ik(IKConfig(csv=args.csv, out_dir=args.out_dir, n_components=K,
                                    hidden_dim=H, device=args.device))
        rows.append({"K": K, "H": H, "best_epoch": res["best_epoch"],
                     "val_nll": res["best_val_nll"], "test_nll": res["test_nll"]})
        pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "sweep.csv"), index=False)

    print(pd.DataFrame(rows).sort_values("val_nll").to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()

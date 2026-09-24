import argparse

from ctr_mdn.data import N_ROWS, build_training_csv


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw", required=True)
    p.add_argument("--out", default="data/ctr_data_rebuilt.csv")
    p.add_argument("--all_rows", action="store_true", help="keep all 100,000 samples")
    args = p.parse_args()
    df = build_training_csv(args.raw, args.out, None if args.all_rows else N_ROWS)
    print(f"saved {args.out} ({len(df)} rows)")

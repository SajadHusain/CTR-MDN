"""Evaluate a trained FK-MDN checkpoint on its validation and test blocks.

    python evaluate_fk.py --ckpt runs/fk/M3_K5_H256/test7_val6/best.pt --memory_len 3 --device cpu
"""
import argparse
import json

from torch.utils.data import DataLoader

from ctr_mdn.data import CTRFKDataset, loso_split
from ctr_mdn.fk_train import evaluate, get_device, load_checkpoint


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--csv", default="data/ctr_data.csv")
    p.add_argument("--memory_len", type=int, required=True)
    p.add_argument("--seq_len", type=int, default=None)
    p.add_argument("--test_seq", type=int, default=7)
    p.add_argument("--val_seq", type=int, default=6)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = get_device(args.device)
    model, stats, _ = load_checkpoint(args.ckpt, device)
    ds = CTRFKDataset(args.csv, args.memory_len, seq_len=args.seq_len)
    _, va, te = loso_split(ds, args.test_seq, args.val_seq)
    res = {"val": evaluate(model, DataLoader(va, batch_size=256), stats, device),
           "test": evaluate(model, DataLoader(te, batch_size=256), stats, device)}
    print(json.dumps(res, indent=2))

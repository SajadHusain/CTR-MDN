"""Train the FK-MDN on one fold.

    python train.py --device cpu
    python train.py --memory_len 0 --n_components 5 --hidden_dim 256
"""
import argparse
from dataclasses import fields

from ctr_mdn.train import Config, train


def parse_args():
    p = argparse.ArgumentParser()
    for f in fields(Config):
        if f.type in (bool, "bool"):
            p.add_argument(f"--{f.name}", action="store_true")
        elif f.type in (int, "int"):
            p.add_argument(f"--{f.name}", type=int, default=f.default)
        elif f.type in (float, "float"):
            p.add_argument(f"--{f.name}", type=float, default=f.default)
        else:
            p.add_argument(f"--{f.name}", type=str, default=f.default)
    return Config(**vars(p.parse_args()))


if __name__ == "__main__":
    train(parse_args())

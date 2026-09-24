"""Train the FK-MDN on one fold.
"""
import argparse
from dataclasses import fields
from typing import Optional

from ctr_mdn.fk_train import FKConfig, train_fk


def parse_args():
    p = argparse.ArgumentParser()
    for f in fields(FKConfig):
        if f.type is bool:
            p.add_argument(f"--{f.name}", action="store_true")
        elif f.type in (int, Optional[int]):
            p.add_argument(f"--{f.name}", type=int, default=f.default)
        elif f.type is float:
            p.add_argument(f"--{f.name}", type=float, default=f.default)
        else:
            p.add_argument(f"--{f.name}", type=str, default=f.default)
    return FKConfig(**vars(p.parse_args()))


if __name__ == "__main__":
    train_fk(parse_args())

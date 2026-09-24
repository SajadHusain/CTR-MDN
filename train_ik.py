"""Train the IK-MDN (tip pose -> actuator configuration).
"""
import argparse
from dataclasses import fields

from ctr_mdn.ik import IKConfig, train_ik


def parse_args():
    p = argparse.ArgumentParser()
    for f in fields(IKConfig):
        if f.type is int:
            p.add_argument(f"--{f.name}", type=int, default=f.default)
        elif f.type is float:
            p.add_argument(f"--{f.name}", type=float, default=f.default)
        else:
            p.add_argument(f"--{f.name}", type=str, default=f.default)
    return IKConfig(**vars(p.parse_args()))


if __name__ == "__main__":
    train_ik(parse_args())

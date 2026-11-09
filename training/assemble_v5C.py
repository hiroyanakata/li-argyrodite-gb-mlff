#!/usr/bin/env python
"""Assemble v5C training set = v5_trainB (clean) + AL train (oversampled).

Both inputs are already on the cohesive-energy zero (ATMENE-subtracted); the
energy_gauge.json in the finetune step shifts them together. AL frames are the
corrective signal for GB-core PS4 decomposition, so they are duplicated
--al_mult times to carry weight against the ~5k clean frames (mirrors the
NEB x3 oversample used when v5_trainB itself was built).
"""
import argparse
import numpy as np
from ase.io import read, write


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trainB", required=True)
    ap.add_argument("--al_train", required=True)
    ap.add_argument("--al_mult", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    base = read(a.trainB, ":")
    al = read(a.al_train, ":")
    frames = list(base) + list(al) * a.al_mult
    write(a.out, frames, format="extxyz")

    eb = np.array([x.get_potential_energy() / len(x) for x in base])
    ea = np.array([x.get_potential_energy() / len(x) for x in al])
    print(f"v5_trainB : {len(base)} frames  E/atom p50 {np.median(eb):.3f}")
    print(f"AL train  : {len(al)} frames x{a.al_mult} = {len(al) * a.al_mult}  "
          f"E/atom p50 {np.median(ea):.3f}")
    print(f"wrote {a.out}: {len(frames)} frames "
          f"(base {len(base)} + AL {len(al) * a.al_mult})")


if __name__ == "__main__":
    main()

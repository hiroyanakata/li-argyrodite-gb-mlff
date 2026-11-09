#!/usr/bin/env python
"""Assemble the Phase B fine-tune training set: v3 train frames (x1, anti-forget)
+ Phase B DFT frames oversampled per-src (emphasise the underestimated hops).

Oversampling == frame duplication, because finetune_mattersim.py reads ONE extxyz
via AtomsAdaptor.from_file (no per-frame weight arg). Duplicating a frame N times
gives it N x the gradient weight.

  python assemble_phaseB_train.py \
     --base data/v3_train_reduce_emb.extxyz \
     --phaseb phaseB_dftk_k26.extxyz \
     --out phaseB_train.extxyz \
     --ratios clrich:8,v4l0_0_50:6,v4l0_18_108:3,brrich:3,interior:1,pre_18_108:1,v3_0_50:1,v3_18_108:1

Round-1 (no NEB, no v3 base) : omit --base to build a phaseB-only smoke set.
Add NEB saddle frames later by appending them to --phaseb (same harvester).
"""
import argparse
from collections import Counter
from ase.io import read, write
from ase.calculators.singlepoint import SinglePointCalculator

ap = argparse.ArgumentParser()
ap.add_argument("--base", default=None, help="v3 train extxyz (x1); omit for phaseB-only")
ap.add_argument("--phaseb", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--ratios", required=True, help="src:mult,...  (missing src -> x1)")
a = ap.parse_args()

ratios = {}
for tok in a.ratios.split(","):
    k, v = tok.split(":")
    ratios[k.strip()] = int(v)

frames = []
nbase = 0
if a.base:
    base = read(a.base, ":")
    for at in base:
        at.info.setdefault("src", "v3train")
    frames += base
    nbase = len(base)
    print(f"base: {nbase} frames (x1)")

pb = read(a.phaseb, ":")
persrc = Counter()
for at in pb:
    src = at.info.get("src", "unknown")
    mult = ratios.get(src, 1)
    e, f = at.get_potential_energy(), at.get_forces()   # preserve labels across copy
    for _ in range(mult):
        c = at.copy()
        c.calc = SinglePointCalculator(c, energy=e, forces=f)
        c.info["src"] = src
        frames.append(c)
    persrc[src] += mult

write(a.out, frames, format="extxyz")
print("phaseB oversample per src:", dict(sorted(persrc.items())))
print(f"wrote {a.out}: {len(frames)} frames (base {nbase} + phaseB {sum(persrc.values())})")

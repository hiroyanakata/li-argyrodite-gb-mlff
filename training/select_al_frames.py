#!/usr/bin/env python
# =============================================================================
# select_al_frames.py -- pick active-learning frames for DFT from the GB-core
# cut-slab trajectories (relax_md_cut.py: rx_<cut>_<model>_md.extxyz).
#
#   * "broken" trajectories (fine-tuned models that decompose PS4): every frame
#     where the PS4 defect count (cn_lt4) changes (onset of a decomposition
#     event), filled up to --n_broken with frames evenly spaced in time;
#   * "healthy" trajectories (pretrained, PS4 intact): --n_healthy frames evenly
#     spaced after --skip steps;
#   * frames hotter than --tmax (thermostat blow-ups) or with any pair closer
#     than --min_dist are dropped.
# Train/test split is by CUT (never by frame), so the test cuts are unseen.
# Writes <out>_train.extxyz / <out>_test.extxyz with info: src, step, role,
# split, T, and the MLIP energy/forces renamed to mlip_energy / mlip_forces.
#
#   python select_al_frames.py --dir . --train_cuts n8_gb1,n8_gb2,uni_gb1 \
#          --test_cuts uni_gb2 --out al1
# =============================================================================
import argparse
import os

import numpy as np
from ase.io import read, write
from ase.neighborlist import neighbor_list

KB = 8.617333e-5


def temperature(at):
    mv = at.arrays.get("momenta")
    if mv is None:
        return 0.0
    m = at.get_masses()[:, None]
    free = at.arrays.get("fixed", np.zeros(len(at), int)) == 0
    ek = 0.5 * ((mv[free] ** 2) / m[free]).sum()
    return 2 * ek / (3 * free.sum() * KB)


def min_dist(at, rc=1.2):
    i, j, d = neighbor_list("ijd", at, rc)
    return float(d.min()) if len(d) else rc


def even(idx, n):
    if len(idx) <= n:
        return list(idx)
    return [idx[int(round(k))] for k in np.linspace(0, len(idx) - 1, n)]


def pick_broken(frames, n):
    cn = [f.info.get("cn_lt4", 0) for f in frames]
    ev = [k for k in range(len(frames)) if k == 0 or cn[k] != cn[k - 1]]
    rest = [k for k in range(len(frames)) if k not in ev]
    need = max(0, n - len(ev))
    sel = sorted(set(even(ev, n)) | set(even(rest, need)))
    return sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".")
    ap.add_argument("--train_cuts", default="n8_gb1,n8_gb2,uni_gb1")
    ap.add_argument("--test_cuts", default="uni_gb2")
    ap.add_argument("--broken_models", default="v5B,phaseB")
    ap.add_argument("--healthy_models", default="pre")
    ap.add_argument("--n_broken", type=int, default=7)
    ap.add_argument("--n_healthy", type=int, default=10)
    ap.add_argument("--n_test", type=int, default=3, help="frames per test trajectory")
    ap.add_argument("--skip", type=int, default=500, help="equilibration steps skipped")
    ap.add_argument("--tmax", type=float, default=1500.0)
    ap.add_argument("--min_dist", type=float, default=0.9)
    ap.add_argument("--extra", default="", help="comma list of extra healthy trajectories (paths)")
    ap.add_argument("--out", default="al1")
    args = ap.parse_args()

    out = {"train": [], "test": []}
    jobs = []
    for split, cuts in (("train", args.train_cuts), ("test", args.test_cuts)):
        for c in filter(None, cuts.split(",")):
            for m in args.broken_models.split(","):
                jobs.append((split, c, m, "broken"))
            for m in args.healthy_models.split(","):
                jobs.append((split, c, m, "healthy"))
    for p in filter(None, args.extra.split(",")):
        jobs.append(("train", p, None, "healthy"))

    for split, c, m, role in jobs:
        path = c if m is None else os.path.join(args.dir, "rx_%s_%s_md.extxyz" % (c, m))
        if not os.path.exists(path):
            print("[miss]", path)
            continue
        fr = read(path, ":")
        ok = [k for k, f in enumerate(fr)
              if f.info.get("step", 0) >= args.skip and temperature(f) <= args.tmax
              and min_dist(f) >= args.min_dist]
        frs = [fr[k] for k in ok]
        if split == "test":
            sel = pick_broken(frs, args.n_test) if role == "broken" else even(list(range(len(frs))), args.n_test)
        elif role == "broken":
            sel = pick_broken(frs, args.n_broken)
        else:
            sel = even(list(range(len(frs))), args.n_healthy)
        for k in sel:
            f = frs[k].copy()
            f.info.update(src=os.path.basename(path), role=role, split=split, T=round(temperature(f), 1))
            f.info["mlip_energy"] = f.info.pop("energy", None)
            if "forces" in f.arrays:
                f.arrays["mlip_forces"] = f.arrays.pop("forces")
            f.calc = None
            out[split].append(f)
        print("%-5s %-34s role=%-7s usable=%3d/%3d picked=%2d steps=%s" % (
            split, os.path.basename(path), role, len(frs), len(fr), len(sel),
            [frs[k].info.get("step") for k in sel]))

    for split, frs in out.items():
        write("%s_%s.extxyz" % (args.out, split), frs)
        print("wrote %s_%s.extxyz (%d frames)" % (args.out, split, len(frs)))


if __name__ == "__main__":
    main()

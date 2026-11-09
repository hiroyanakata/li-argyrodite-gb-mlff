#!/usr/bin/env python
# =============================================================================
# gb_build_uniform.py -- control GB model with Li6PS5X-like grain interiors.
#
# gb_build.py puts every halide on the markers closest to a GB plane, which
# leaves the grain interiors halide-free (and --neutral additionally turns
# PS4-vertex S into halide at the GB).  Here, instead:
#   * only the free-anion markers (Al/Dy) are decorated; PS4 sulfurs are never
#     touched, so no halide sits on a PS4 vertex;
#   * the halide count is fixed by charge neutrality with no Li vacancies
#     (same composition as gbBIG_std);
#   * halides are drawn at random over ALL markers with a mild GB bias,
#     w(d) = 1 + (enrich-1) * exp(-d / lam), so grain interiors keep ~X1.
#
#   python gb_build_uniform.py --xyz test.GB.xyz --repeat 1,1,1 \
#          --enrich 2.0 --lam 3.0 --seed 0 --out gbBIG_uni.extxyz
# =============================================================================
import argparse
from collections import Counter

import numpy as np
from ase.io import write

from gb_build import read_xyz3, dist_to_gb

Q = {"Li": 1, "P": 5, "S": -2, "Cl": -1, "Br": -1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xyz", required=True)
    ap.add_argument("--repeat", default="1,1,1")  # NOT 1,2,1 on gbMID_placeholder (broken, see make_mid.py)
    ap.add_argument("--gb-pos", default="0.25,0.75")
    ap.add_argument("--enrich", type=float, default=2.0, help="halide weight at the GB plane relative to the interior")
    ap.add_argument("--lam", type=float, default=3.0, help="decay length of the GB bias [A]")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="gbBIG_uni.extxyz")
    args = ap.parse_args()

    at = read_xyz3(args.xyz).repeat([int(v) for v in args.repeat.split(",")])
    sym = np.array(at.get_chemical_symbols(), dtype=object)
    L = at.cell.lengths()[0]
    planes = [float(v) for v in args.gb_pos.split(",")]
    markers = np.where((sym == "Al") | (sym == "Dy"))[0]

    # neutrality with all markers filled: nLi + 5 nP - 2 nS(PS4) = 2 nS(free) + nX
    pos_q = (sym == "Li").sum() + 5 * (sym == "P").sum() - 2 * (sym == "S").sum()
    nX = 2 * len(markers) - pos_q
    assert 0 < nX < len(markers), nX

    rng = np.random.default_rng(args.seed)
    d = dist_to_gb(at.positions[markers, 0], L, planes)
    w = 1.0 + (args.enrich - 1.0) * np.exp(-d / args.lam)
    hal = rng.choice(markers, size=nX, replace=False, p=w / w.sum())
    sym[markers] = "S"
    for k, a in enumerate(rng.permutation(hal)):
        sym[a] = "Cl" if k % 2 == 0 else "Br"
    at.set_chemical_symbols(list(sym))

    c = Counter(at.get_chemical_symbols())
    q = sum(Q[e] * n for e, n in c.items())
    print("composition", dict(c), "net charge %+d" % q)
    print("per P: Li %.2f S %.2f X %.2f" % (c["Li"] / c["P"], c["S"] / c["P"], (c["Cl"] + c["Br"]) / c["P"]))
    write(args.out, at)
    print("wrote", args.out)


if __name__ == "__main__":
    main()

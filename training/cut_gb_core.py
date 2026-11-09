#!/usr/bin/env python
# =============================================================================
# cut_gb_core.py -- cut a DFT-sized slab around one GB core of a big GB cell.
#
# The big GB cells (gbV2_*, 69.4 x 44.07 x 9.854 A) cannot be reduced in y:
# the GB cores are not periodic with ly/2 (see bigGB_gpu/make_mid.py).  So we
# keep the full y/z periodicity and cut only along the GB normal x:
#   * window |x - x_gb| <= half_width, then vacuum along x;
#   * a PS4 unit is kept whole iff its P is inside the window (never cut);
#   * free S / halide / Li are kept iff inside the window;
#   * formal charge (Li+1, PS4 -3, S-2, X-1) is brought to 0 by removing the
#     OUTERMOST species (farthest from the GB plane): Li if q>0, X/free S if q<0;
#   * atoms within --fix_width of either cut face get a "fixed" flag
#     (array 'fixed', 1 = frozen) for constrained relaxations.
# Reports, for every P, the number of Li within 5 A (training set minimum = 5).
#
#   python cut_gb_core.py --data gbV2_uni.extxyz --gb-x 17.36 --half_width 7 \
#          --vacuum 15 --out cut_uni_gb1.extxyz
# =============================================================================
import argparse
from collections import Counter

import numpy as np
from ase.io import read, write
from scipy.spatial import cKDTree

Q = {"Li": 1, "P": 5, "S": -2, "Cl": -1, "Br": -1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--gb-x", type=float, required=True, help="GB plane position along x [A]")
    ap.add_argument("--half_width", type=float, default=7.0)
    ap.add_argument("--vacuum", type=float, default=15.0)
    ap.add_argument("--fix_width", type=float, default=2.5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    at = read(args.data)
    L = at.cell.lengths()
    sym = np.array(at.get_chemical_symbols())
    pos = np.mod(at.get_positions(), L) % L
    # x relative to the GB plane, in (-Lx/2, Lx/2]
    dx = (pos[:, 0] - args.gb_x + L[0] / 2) % L[0] - L[0] / 2

    t = cKDTree(pos, boxsize=L)
    P = np.where(sym == "P")[0]
    owner = {}                                   # PS4 sulfur -> its P
    for p in P:
        for j in t.query_ball_point(pos[p], 2.4):
            if sym[j] == "S":
                owner[j] = p

    inwin = np.abs(dx) <= args.half_width
    keep = set()
    for p in P:
        if inwin[p]:
            keep.add(p)
            keep.update(j for j, o in owner.items() if o == p)
    for i in np.where(inwin)[0]:
        if sym[i] != "P" and i not in owner:
            keep.add(i)

    def charge(idx):
        return sum(Q[sym[i]] for i in idx)

    q = charge(keep)
    removed = []
    while q != 0:
        if q > 0:
            cand = [i for i in keep if sym[i] == "Li"]
        else:
            cand = [i for i in keep if i not in owner and sym[i] in ("Cl", "Br")]
            if q <= -2 or not cand:
                cand += [i for i in keep if i not in owner and sym[i] == "S"]
        i = max(cand, key=lambda k: abs(dx[k]))
        if sym[i] == "S" and q == -1:            # S would overshoot to +1
            cand = [k for k in cand if sym[k] != "S"]
            if not cand:
                break
            i = max(cand, key=lambda k: abs(dx[k]))
        keep.discard(i)
        removed.append((sym[i], round(float(dx[i]), 2)))
        q = charge(keep)
    assert q == 0, q

    idx = np.array(sorted(keep))
    new = at[idx]
    p_new = pos[idx].copy()
    p_new[:, 0] = dx[idx] + args.half_width + args.vacuum / 2.0
    wx = 2 * args.half_width + args.vacuum
    new.set_cell([wx, L[1], L[2]])
    new.set_positions(p_new)
    new.pbc = True
    edge = args.half_width - np.abs(dx[idx])
    new.arrays["fixed"] = (edge <= args.fix_width).astype(int)
    new.info["gb_x_in_cut"] = args.half_width + args.vacuum / 2.0
    new.info["src"] = args.data
    write(args.out, new)

    # report
    s2 = np.array(new.get_chemical_symbols())
    t2 = cKDTree(np.mod(p_new, new.cell.lengths()), boxsize=new.cell.lengths())
    licnt = []
    for p in np.where(s2 == "P")[0]:
        nb = t2.query_ball_point(np.mod(p_new[p], new.cell.lengths()), 5.0)
        licnt.append((int((s2[nb] == "Li").sum()), round(float(dx[idx][p]), 2)))
    c = Counter(s2)
    print("%s: nat=%d %s charge=%+d fixed=%d cell=%.2fx%.2fx%.3f" % (
        args.out, len(new), dict(c), charge(idx), new.arrays["fixed"].sum(), wx, L[1], L[2]))
    print("  removed for neutrality:", removed)
    low = sorted(x for x in licnt if x[0] <= 6)
    print("  P with Li@5A<=6 (n_Li, x-x_gb):", low)


if __name__ == "__main__":
    main()

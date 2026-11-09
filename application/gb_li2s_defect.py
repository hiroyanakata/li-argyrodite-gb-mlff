#!/usr/bin/env python
# =============================================================================
# gb_li2s_defect.py -- Li2S-deficient grain boundary on a stoichiometric GB.
#
# Start from a charge-neutral Li6PS5X GB with intact PS4 (gb_build_uniform.py)
# and remove neutral Li2S units at the GB planes:
#   * only FREE S2- is removed (no P within 2.4 A) -- PS4 is never touched;
#   * the free S closest to each GB plane go first, alternating between planes;
#   * for every removed S the 2 nearest Li (not --protect'ed) are removed too.
# The grain interiors keep the Li6PS5X composition; the GB becomes S- and
# Li-poor, so its free-anion sites are left halide-rich.
#
#   python gb_li2s_defect.py --data gbS_seg2.extxyz --n_per_gb 2 \
#          --protect 0,50,43,63 --out gbD_l2.extxyz
# Writes <out> and <out>.map.json (old -> new atom indices).
# =============================================================================
import argparse
import json
from collections import Counter

import numpy as np
from ase.io import read, write

Q = {"Li": 1, "P": 5, "S": -2, "Cl": -1, "Br": -1}


def mic(d, cell):
    f = d @ np.linalg.inv(cell)
    f -= np.round(f)
    return f @ cell


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--n_per_gb", type=int, default=2, help="Li2S units removed per GB plane")
    ap.add_argument("--gb-axis", default="x")
    ap.add_argument("--gb-pos", default="0.25,0.75")
    ap.add_argument("--protect", default="", help="comma list of atom indices never removed (e.g. NEB hop Li)")
    ap.add_argument("--li-max-dgb", type=float, default=0.0,
                    help="if >0, only Li within this distance [A] of a GB plane may be removed "
                         "(keeps the Li2S deficit inside the GB core; 0 = old behaviour)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    at = read(args.data)
    cell = np.array(at.get_cell())
    sym = np.array(at.get_chemical_symbols())
    pos = at.get_positions()
    ax = "xyz".index(args.gb_axis)
    L = at.cell.lengths()[ax]
    planes = [float(v) * L for v in args.gb_pos.split(",")]
    protect = set(int(v) for v in args.protect.split(",") if v.strip())

    P = np.where(sym == "P")[0]
    free_s = [s for s in np.where(sym == "S")[0]
              if np.linalg.norm(mic(pos[P] - pos[s], cell), axis=1).min() > 2.4]

    def dplane(i, x0):
        return abs((pos[i, ax] - x0 + L / 2) % L - L / 2)

    rm_s = []
    for k in range(args.n_per_gb):
        for x0 in planes:
            cand = [s for s in free_s if s not in rm_s]
            rm_s.append(min(cand, key=lambda s: dplane(s, x0)))

    li = [i for i in np.where(sym == "Li")[0] if i not in protect]
    if args.li_max_dgb > 0:
        li = [i for i in li if min(dplane(i, x0) for x0 in planes) <= args.li_max_dgb]
    rm_li = []
    for s in rm_s:
        cand = [i for i in li if i not in rm_li]
        d = np.linalg.norm(mic(pos[cand] - pos[s], cell), axis=1)
        rm_li += [cand[j] for j in np.argsort(d)[:2]]

    drop = set(rm_s) | set(rm_li)
    keep = [i for i in range(len(at)) if i not in drop]
    new = at[keep]
    idx_map = {int(o): n for n, o in enumerate(keep)}

    c = Counter(new.get_chemical_symbols())
    q = sum(Q[e] * n for e, n in c.items())
    assert q == 0, q
    print("removed S %s (d_GB %s A)" % (rm_s, [round(min(dplane(s, x0) for x0 in planes), 2) for s in rm_s]))
    print("removed Li %s" % rm_li)
    print("composition", dict(c), "net charge %+d" % q)
    for i in sorted(protect):
        print("  protected atom %d -> new index %d" % (i, idx_map[i]))
    write(args.out, new)
    json.dump(idx_map, open(args.out + ".map.json", "w"))
    print("wrote", args.out)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# =============================================================================
# gb_build.py -- decorate a placeholder grain-boundary xyz into a real,
#                MatterSim-ready Li6PS5X structure using a recipe.
#
# The GB models in .../genkaimove (l6ps5cl_210.xyz, GB210.big.xyz, ...) carry
# two placeholder anion markers, Al and Dy, on the S/halide sublattice.  A
# recipe file assigns each marker to S / Br / Cl and turns some Li into
# vacancies (V).  Recipe format (whitespace, one rule per line):
#
#     Al  S   10        # 10 of the Al markers -> S
#     Al  Br   5        #  5 -> Br
#     Al  Cl   5        #  5 -> Cl
#     Dy  S    8
#     Dy  Br   4
#     Dy  Cl   4
#     Li  V    6        # remove 6 Li (vacancies)
#
# Placement mimics mod.py: halides (Br/Cl) go to the markers *closest to a GB
# plane* (segregation), S fills the rest; Cl and Br are interleaved to keep the
# requested 1:1.  Li vacancies default to random but can be biased to the GB.
#
#   python gb_build.py --xyz 210/l6ps5cl_210.xyz --recipe reciepe.GB210.4th \
#          --gb-axis x --gb-pos 0.25,0.75 --seed 0 --out gb210_dec.extxyz
#
# !! WARNING -- do NOT use --neutral for production structures.
#    --neutral compensates Li vacancies by turning *any* S into a halide,
#    including PS4-vertex S, which creates PS3X / PS2X2 units that cannot form in
#    Li6PS5X. The GB models of the paper are built with gb_build_uniform.py
#    (halides on free-anion sites only) and, for Li2S-deficient boundaries,
#    gb_li2s_defect.py. This file is still needed: gb_build_uniform.py imports
#    read_xyz3() and dist_to_gb() from it.
# =============================================================================
import argparse
import numpy as np
from ase import Atoms
from ase.io import write

AX = {"x": 0, "y": 1, "z": 2}


def read_xyz3(path):
    """xyz whose comment line is '<lx> <ly> <lz>' (orthorhombic)."""
    lines = open(path).read().splitlines()
    n = int(lines[0].split()[0])
    c = lines[1].split()
    cell = [float(c[0]), float(c[1]), float(c[2])] if len(c) >= 3 else None
    sym, pos = [], []
    for l in lines[2:2 + n]:
        p = l.split()
        sym.append(p[0])
        pos.append([float(p[1]), float(p[2]), float(p[3])])
    at = Atoms(symbols=sym, positions=np.array(pos), cell=cell, pbc=bool(cell))
    return at


def parse_recipe(path):
    rules = []
    for line in open(path):
        p = line.split()
        if len(p) >= 3:
            rules.append((p[0], p[1], int(p[2])))
    return rules


def dist_to_gb(x, L, planes):
    """min distance (with PBC along the axis) from x to any GB plane fraction."""
    d = np.min([np.abs((x - f * L + L / 2) % L - L / 2) for f in planes], axis=0)
    return d


def main():
    ap = argparse.ArgumentParser(description="decorate placeholder GB xyz via a recipe")
    ap.add_argument("--xyz", required=True)
    ap.add_argument("--recipe", default="")
    ap.add_argument("--gb-axis", default="x", choices=list(AX))
    ap.add_argument("--gb-pos", default="0.25,0.75", help="GB plane fractions along the axis")
    ap.add_argument("--anion-ratio", default="",
                    help="target ratio on the disordered (Al/Dy) sublattice, overriding the "
                         "recipe. e.g. 'S:1,Cl:0.5,Br:0.5' for Li6PS5(Cl0.5Br0.5). "
                         "PS4 sulfurs are untouched; halides segregate to the GB.")
    ap.add_argument("--nvac", type=int, default=-1,
                    help="override Li vacancy count (-1 = use the recipe; 0 = none)")
    ap.add_argument("--neutral", action="store_true",
                    help="[DEPRECATED: may convert PS4-vertex S -> X; see header] enforce charge neutrality by compensating Li vacancies with "
                         "S->X substitution (Li6-xPS5-xX1+x). Not used for the paper.")
    ap.add_argument("--scale", action="store_true",
                    help="scale the recipe counts to the cell's actual placeholder counts "
                         "(preserves Cl:Br and defect ratios; use for big cells)")
    ap.add_argument("--vac-region", default="all", choices=["all", "gb", "bulk"],
                    help="where Li vacancies are drawn from")
    ap.add_argument("--gb-width", type=float, default=6.0, help="GB slab half-width [A] for vac-region")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="gb_decorated.extxyz")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    at = read_xyz3(args.xyz)
    sym = np.array(at.get_chemical_symbols(), dtype=object)
    ax = AX[args.gb_axis]
    L = at.cell.lengths()[ax]
    planes = [float(v) for v in args.gb_pos.split(",")]
    x = at.get_positions()[:, ax]
    dgb = dist_to_gb(x, L, planes)

    rules = parse_recipe(args.recipe) if args.recipe else []

    # ---- composition mode: distribute ALL Al/Dy markers by a target ratio ----
    if args.anion_ratio:
        w = {}
        for tok in args.anion_ratio.replace(",", " ").split():
            e, v = tok.split(":")
            w[e] = float(v)
        markers = np.where((sym == "Al") | (sym == "Dy"))[0]
        m = len(markers)
        tot = sum(w.values())
        cnt = {e: int(round(w[e] / tot * m)) for e in w}
        # make the counts sum to m by adjusting S (the framework free-anion)
        cnt["S"] = cnt.get("S", 0) + (m - sum(cnt.values()))
        order = markers[np.argsort(dgb[markers])]      # closest-to-GB first
        halides = [e for e in w if e in ("Cl", "Br", "I")]
        nhal = sum(cnt.get(e, 0) for e in halides)
        # interleave halides on the GB-nearest sites to keep their ratio
        rem = {e: cnt.get(e, 0) for e in halides}
        for a in order[:nhal]:
            e = max(rem, key=lambda k: rem[k])         # assign the most-remaining halide
            sym[a] = e; rem[e] -= 1
        for a in order[nhal:]:
            sym[a] = "S"
        at.set_chemical_symbols(list(sym))
        print("  anion-ratio %s on %d markers -> %s" %
              (args.anion_ratio, m, {e: cnt.get(e, 0) for e in list(w) + ["S"]}))
        rules = [r for r in rules if r[0] == "Li"]     # keep only Li vacancy rule (if any)
        _skip_recipe = True
    else:
        _skip_recipe = False

    # ---- anion markers (everything except the 'Li V' rule) ----
    for marker in ([] if _skip_recipe else sorted(set(r[0] for r in rules if r[0] != "Li"))):
        idx = np.where(sym == marker)[0]
        targets = {t: c for (m, t, c) in rules if m == marker}
        nCl, nBr, nS = targets.get("Cl", 0), targets.get("Br", 0), targets.get("S", 0)
        if args.scale:                      # scale recipe ratios to this cell's count
            t = nCl + nBr + nS
            f = len(idx) / t if t else 0.0
            nCl, nBr = int(round(nCl * f)), int(round(nBr * f))
            nS = len(idx) - nCl - nBr       # absorb rounding into S
            print("  scale %s x%.3f -> S%d Br%d Cl%d" % (marker, f, nS, nBr, nCl))
        nhal = nCl + nBr
        if nhal + nS < len(idx):
            nS = len(idx) - nhal            # remainder -> S (framework)
        if nhal > len(idx):
            raise SystemExit("%s: recipe halides %d > %d markers; use --scale"
                             % (marker, nhal, len(idx)))
        if nhal + nS != len(idx):
            print("WARN %s: recipe assigns %d but %d markers present (extra->S/trim)"
                  % (marker, nhal + nS, len(idx)))
        order = idx[np.argsort(dgb[idx])]   # closest-to-GB first
        hal = order[:nhal]                  # halides segregate to the GB
        ncl = nbr = 0
        for a in hal:
            if (ncl <= nbr and ncl < nCl) or nbr >= nBr:
                sym[a] = "Cl"; ncl += 1
            else:
                sym[a] = "Br"; nbr += 1
        for a in order[nhal:]:              # rest of the markers -> framework S
            sym[a] = "S"

    at.set_chemical_symbols(list(sym))

    # ---- Li vacancies ----
    nvac = sum(c for (m, t, c) in rules if m == "Li" and t.upper() == "V")
    if args.nvac >= 0:                          # explicit override
        nvac = args.nvac
    if args.scale and nvac > 0:
        # scale vacancy count by the anion-sublattice size ratio (cell / recipe)
        rec_anion = sum(c for (m, t, c) in rules if m != "Li")
        act_anion = int(np.isin(np.array(at.get_chemical_symbols()),
                                ["S", "Br", "Cl"]).sum())
        f = act_anion / rec_anion if rec_anion else 1.0
        nvac = int(round(nvac * f))
        print("  scale Li V x%.3f -> %d vacancies" % (f, nvac))
    if nvac > 0:
        li = np.where(np.array(at.get_chemical_symbols()) == "Li")[0]
        xli = at.get_positions()[li, ax]
        dli = dist_to_gb(xli, L, planes)
        if args.vac_region == "gb":
            pool = li[dli <= args.gb_width]
        elif args.vac_region == "bulk":
            pool = li[dli > args.gb_width]
        else:
            pool = li
        if len(pool) < nvac:
            print("WARN: only %d Li in region, requested %d vacancies" % (len(pool), nvac))
            nvac = len(pool)
        drop = rng.choice(pool, size=nvac, replace=False)
        keep = [k for k in range(len(at)) if k not in set(drop.tolist())]
        at = at[keep]

    # ---- charge neutrality: compensate each Li vacancy with one S -> X ----
    # (Li6PS5X -> Li6-x P S5-x X1+x, the standard closed-shell Li-deficient
    #  argyrodite; without this the cell is formally charged and off-distribution
    #  for the MLIP and gives an open-shell/holed DFT reference.)
    Qform = {"Li": 1, "P": 5, "S": -2, "Cl": -1, "Br": -1, "I": -1}

    def net_charge(a):
        s = np.array(a.get_chemical_symbols())
        return int(sum(Qform.get(e, 0) * (s == e).sum() for e in set(s)))

    if args.neutral:
        q = net_charge(at)
        if q < 0:                       # too negative -> turn |q| S into halide (+1 each)
            sym2 = np.array(at.get_chemical_symbols(), dtype=object)
            sidx = np.where(sym2 == "S")[0]
            ds = dist_to_gb(at.get_positions()[sidx, ax], L, planes)
            order = sidx[np.argsort(ds)]        # compensate at the GB first
            need = min(-q, len(order))
            ncl = nbr = 0
            nCl = need // 2 + (need % 2)        # keep Cl:Br ~ 1:1
            for k, a in enumerate(order[:need]):
                if ncl < nCl:
                    sym2[a] = "Cl"; ncl += 1
                else:
                    sym2[a] = "Br"; nbr += 1
            at.set_chemical_symbols(list(sym2))
            print("  neutralize: %d S -> %dCl+%dBr (was q=%+d)" % (need, ncl, nbr, q))
        elif q > 0:
            print("  WARN: net charge %+d > 0; --neutral only compensates S->X" % q)

    from collections import Counter
    print("decorated composition:", dict(Counter(at.get_chemical_symbols())),
          " net charge = %+d e" % net_charge(at))
    write(args.out, at)
    print("wrote", args.out)


if __name__ == "__main__":
    main()

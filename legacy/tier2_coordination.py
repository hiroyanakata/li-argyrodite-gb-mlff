#!/usr/bin/env python
# =============================================================================
# tier2_coordination.py -- static Li environment, GB vs bulk, per force field.
#
# Reads a model-RELAXED full-atom structure (seg_<model>_ref.extxyz, orthorhombic
# cell) and, for the Li in the GB windows vs the bulk window, reports:
#   - Li coordination number (anions within CUT)
#   - Li-S / Li-Cl / Li-Br split, and the halogen fraction of the first shell
#   - nearest-anion distance (free-volume proxy)
#   - free-S vs framework-S(PS4) split of the coordinating S
# This ties the segregation result (halogen expelled / S-rich GB) to the
# trapping (what cage does a GB Li actually sit in) at the atomic level.
#
#   python tier2_coordination.py seg_pretrained_ref.extxyz seg_phaseB_ref.extxyz
# =============================================================================
import sys
import numpy as np

CUT = 3.2          # Li-anion first shell [A]
PS_CUT = 2.6       # S within this of a P == framework (PS4) S
GB_HALF, BULK_HALF = 3.5, 5.0


def read_extxyz(path):
    with open(path) as fh:
        n = int(fh.readline())
        hdr = fh.readline()
        L = hdr.split('Lattice="')[1].split('"')[0].split()
        cell = np.array(L, float).reshape(3, 3)
        sym, pos = [], []
        for _ in range(n):
            p = fh.readline().split()
            sym.append(p[0]); pos.append([float(p[1]), float(p[2]), float(p[3])])
    return np.array(sym), np.array(pos), np.diag(cell)


def mic_dist(a, b, L):
    """all pairwise min-image distances, a:(na,3) b:(nb,3) -> (na,nb)."""
    d = a[:, None, :] - b[None, :, :]
    d -= L * np.round(d / L)
    return np.sqrt((d ** 2).sum(-1))


def analyze(path):
    sym, pos, L = read_extxyz(path)
    Lx = L[0]
    is_ = lambda s: sym == s
    liP, sP, clP, brP, pP = (pos[is_(x)] for x in ("Li", "S", "Cl", "Br", "P"))
    # free vs framework S
    dSP = mic_dist(sP, pP, L)
    frameworkS = (dSP.min(1) <= PS_CUT)
    freeS = sP[~frameworkS]; fwS = sP[frameworkS]

    # region of each Li by x
    wx = np.mod(liP[:, 0], Lx)
    gb = (np.abs(wx - 0.25 * Lx) <= GB_HALF) | (np.abs(wx - 0.75 * Lx) <= GB_HALF)
    bulk = np.abs(wx - 0.50 * Lx) <= BULK_HALF

    dS = mic_dist(liP, sP, L)
    dfree = mic_dist(liP, freeS, L)
    dfw = mic_dist(liP, fwS, L)
    dCl = mic_dist(liP, clP, L)
    dBr = mic_dist(liP, brP, L)
    dhal = np.concatenate([dCl, dBr], axis=1)
    dall = np.concatenate([dS, dCl, dBr], axis=1)

    out = {}
    for name, mask in [("GB", gb), ("BULK", bulk)]:
        if not mask.any():
            continue
        nS = (dS[mask] <= CUT).sum(1).mean()
        nfree = (dfree[mask] <= CUT).sum(1).mean()
        nfw = (dfw[mask] <= CUT).sum(1).mean()
        nCl = (dCl[mask] <= CUT).sum(1).mean()
        nBr = (dBr[mask] <= CUT).sum(1).mean()
        ncoord = (dall[mask] <= CUT).sum(1).mean()
        nhal = nCl + nBr
        halfrac = nhal / ncoord if ncoord else 0.0
        nn = dall[mask].min(1).mean()
        nn_hal = dhal[mask].min(1).mean()
        out[name] = dict(nLi=int(mask.sum()), coord=ncoord, nS=nS, nfreeS=nfree,
                         nfwS=nfw, nCl=nCl, nBr=nBr, nhal=nhal, halfrac=halfrac,
                         nn=nn, nn_hal=nn_hal)
    return out


if __name__ == "__main__":
    hdr = ("%-24s %-5s %4s %6s %6s %7s %6s %6s %6s %7s %6s %7s" %
           ("file", "reg", "nLi", "coord", "Li-S", "freeS", "Li-Cl", "Li-Br",
            "nhal", "halfr", "nn", "nnHal"))
    print(hdr); print("-" * len(hdr))
    for path in sys.argv[1:]:
        r = analyze(path)
        tag = path.replace("_ref.extxyz", "").replace("seg_", "")
        for reg in ("GB", "BULK"):
            if reg not in r:
                continue
            d = r[reg]
            print("%-24s %-5s %4d %6.2f %6.2f %7.2f %6.2f %6.2f %6.2f %7.2f %6.2f %7.2f"
                  % (tag, reg, d["nLi"], d["coord"], d["nS"], d["nfreeS"],
                     d["nCl"], d["nBr"], d["nhal"], d["halfrac"], d["nn"], d["nn_hal"]))

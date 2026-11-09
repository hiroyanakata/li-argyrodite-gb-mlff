#!/usr/bin/env python3
"""Standalone PS4 audit for gbV2_*.extxyz (no ASE needed).

Checks, per structure:
  - P coordination number to S/halogen within r_PS cut  (PS4 => cn == 4)
  - vertex-halogen on P (the gb_build --neutral bug signature)
  - shortest S-S / Li-Li / P-P pairs (decomposition signature)
  - composition / net charge
"""
import re
import sys

import numpy as np

HAL = {"Cl", "Br", "I"}
R_PS = 2.60          # P-X first shell cut (PS4 bond ~2.05 A)
SHORT = {("S", "S"): 2.30, ("Li", "Li"): 1.90, ("P", "P"): 2.60}
CHG = {"Li": 1, "P": 5, "S": -2, "Cl": -1, "Br": -1, "I": -1}


def read_extxyz(path):
    frames = []
    with open(path) as fh:
        while True:
            line = fh.readline()
            if not line.strip():
                break
            nat = int(line)
            comment = fh.readline()
            m = re.search(r'Lattice="([^"]+)"', comment)
            cell = np.array(m.group(1).split(), float).reshape(3, 3) if m else None
            sym, pos = [], []
            for _ in range(nat):
                p = fh.readline().split()
                sym.append(p[0])
                pos.append([float(x) for x in p[1:4]])
            frames.append((sym, np.array(pos), cell))
    return frames


def mic_dist(a, b, cell):
    """Minimum-image distances from one point a to array b (orthorhombic-safe general)."""
    inv = np.linalg.inv(cell)
    d = b - a
    f = d @ inv
    f -= np.round(f)
    return np.linalg.norm(f @ cell, axis=1)


def audit(path):
    frames = read_extxyz(path)
    print(f"\n=== {path}  ({len(frames)} frame(s)) ===")
    for fi, (sym, pos, cell) in enumerate(frames):
        sym = np.array(sym)
        nat = len(sym)
        comp = {s: int((sym == s).sum()) for s in sorted(set(sym))}
        q = sum(CHG.get(s, 0) * n for s, n in comp.items())
        print(f"[frame {fi}] nat={nat} comp={comp} net_charge={q:+d}")
        print(f"           cell diag = {np.diag(cell).round(3).tolist()}")

        anion = np.isin(sym, ["S"] + sorted(HAL))
        ai = np.where(anion)[0]
        pidx = np.where(sym == "P")[0]

        cns, vertex_hal, badP = [], 0, []
        for p in pidx:
            d = mic_dist(pos[p], pos[ai], cell)
            shell = ai[d < R_PS]
            cns.append(len(shell))
            nh = int(np.isin(sym[shell], sorted(HAL)).sum())
            vertex_hal += nh
            if len(shell) != 4:
                badP.append((int(p), len(shell)))
        cns = np.array(cns)
        print(f"           P: n={len(pidx)}  cn==4: {(cns == 4).sum()}  "
              f"cn!=4: {(cns != 4).sum()}  cn hist={dict(zip(*np.unique(cns, return_counts=True)))}")
        print(f"           vertex halogen on P (should be 0): {vertex_hal}")
        if badP:
            print(f"           badP (idx,cn) first 12: {badP[:12]}")

        for (e1, e2), cut in SHORT.items():
            i1 = np.where(sym == e1)[0]
            i2 = np.where(sym == e2)[0]
            mn, nshort = 99.0, 0
            for i in i1:
                d = mic_dist(pos[i], pos[i2], cell)
                d = d[d > 1e-6]
                if d.size:
                    mn = min(mn, d.min())
                    nshort += int((d < cut).sum())
            print(f"           {e1}-{e2}: min={mn:.3f} A  pairs<{cut}: {nshort // 2}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        audit(p)

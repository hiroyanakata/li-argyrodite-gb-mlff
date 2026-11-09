#!/usr/bin/env python
# =============================================================================
# coord_tier2.py -- static Li coordination (Tier-2) GB vs bulk on equilibrated
# gbV2 end states: mean anion coordination of Li within RCUT = 3.2 A, the halide
# fraction of the first shell, <n_S> and <n_halide>. GB = +-3.5 A around the GB
# planes (17.36, 52.07 A), bulk = 30-40 A.
#
#   python coord_tier2.py v5D=nemd_v5D_gbV2_uni_700K_z_F000_s0_ckpt.extxyz #          pretrained=nemd_pretrained_gbV2_uni_700K_z_F000_s0_ckpt.extxyz --out coord_tier2.json
# (the zero-field NEMD end states at 700 K were used in the paper)
# =============================================================================
import numpy as np, re, json
from scipy.spatial import cKDTree

GB = [17.36, 52.07]; WIN = 3.5; BULK = (30.0, 40.0); RCUT = 3.2
HAL = ["Cl", "Br", "I"]

def read_xyz(path):
    with open(path) as f: L = f.read().splitlines()
    n = int(L[0].split()[0])
    cell = np.array(re.search(r'Lattice="([^"]+)"', L[1]).group(1).split(), float).reshape(3, 3)
    sym, pos = [], []
    for l in L[2:2+n]:
        p = l.split(); sym.append(p[0]); pos.append([float(x) for x in p[1:4]])
    return np.array(sym), np.array(pos), np.diag(cell).astype(float)

def analyse(path, tag):
    sym, pos, L = read_xyz(path)
    pos = np.mod(pos, L)
    t = cKDTree(pos, boxsize=L)
    Li = np.where(sym == "Li")[0]
    x = pos[:, 0]
    gbm = np.minimum(np.abs(x - GB[0]), np.abs(x - GB[1])) < WIN
    bkm = (x >= BULK[0]) & (x < BULK[1])
    res = {}
    for rname, mask in [("GB", gbm), ("BULK", bkm)]:
        sel = [i for i in Li if mask[i]]
        coord = []; nS = []; nHal = []
        for i in sel:
            nb = [j for j in t.query_ball_point(pos[i], RCUT) if j != i]
            sp = sym[nb]
            anion = [j for j, s in zip(nb, sp) if s in (["S"] + HAL)]
            coord.append(len(anion))
            nS.append(int(np.sum(sp == "S")))
            nHal.append(int(np.sum(np.isin(sp, HAL))))
        coord = np.array(coord); nS = np.array(nS); nHal = np.array(nHal)
        tot = coord.sum()
        res[rname] = dict(n_Li=len(sel), coord=float(coord.mean()),
                          halfrac=float(nHal.sum()/tot) if tot else 0.0,
                          nS=float(nS.mean()), nHal=float(nHal.mean()))
    print(f"[{tag}]")
    for r in ("GB", "BULK"):
        d = res[r]
        print(f"  {r:5s} n_Li={d['n_Li']:3d}  coord={d['coord']:.2f}  "
              f"hal.frac={d['halfrac']:.2f}  <nS>={d['nS']:.2f} <nHal>={d['nHal']:.2f}")
    return res


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", nargs="+", help="label=file.extxyz")
    ap.add_argument("--out", default="coord_tier2.json")
    a = ap.parse_args()
    out = {}
    for p in a.pairs:
        lab, fn = p.split("=", 1)
        out[lab] = analyse(fn, lab)
    json.dump(out, open(a.out, "w"))
    print("wrote", a.out)

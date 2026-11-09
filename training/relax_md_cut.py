#!/usr/bin/env python
# =============================================================================
# relax_md_cut.py -- constrained relax + short NVT MD of a GB-core cut slab
# (cut_gb_core.py output) with one MatterSim model, tracking PS4 integrity.
#
# Atoms with arrays['fixed']==1 (outer layers next to the vacuum) are frozen.
#   1. FIRE relax to --fmax           -> <out>_relax.extxyz
#   2. Langevin NVT at --temp         -> <out>_md.extxyz (every --every steps)
# PS4 integrity per saved frame: n_P with <4 S within 2.4 A ("cn<4"), with <=2
# ("cn<=2", i.e. broken), and the shortest P-S / S-S / Li-Li distance.
# Frames are the DFT candidates for active learning (energy/forces from the
# model are stored so DFT can be compared pointwise).
#
#   python relax_md_cut.py --data cut_n8_gb1.extxyz --model v5B.pth \
#          --temp 700 --steps 2000 --out rx_n8_gb1_v5B
# =============================================================================
import argparse
import time

import numpy as np

try:                                        # ase>=3.28 moved the stress helpers
    import ase.constraints as _asec
    if not hasattr(_asec, "full_3x3_to_voigt_6_stress"):
        from ase.stress import (full_3x3_to_voigt_6_stress as _f3,
                                voigt_6_to_full_3x3_stress as _v6)
        _asec.full_3x3_to_voigt_6_stress = _f3
        _asec.voigt_6_to_full_3x3_stress = _v6
except Exception:
    pass

from ase import units
from ase.constraints import FixAtoms
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.optimize import FIRE
from scipy.spatial import cKDTree


def make_calc(model, device):
    from mattersim.forcefield import MatterSimCalculator
    if model in ("pretrained", "generic", "default"):
        return MatterSimCalculator(device=device)
    return MatterSimCalculator(load_path=model, device=device)


def ps4_state(at):
    sym = np.array(at.get_chemical_symbols())
    L = at.cell.lengths()
    pos = np.mod(at.get_positions(), L) % L
    t = cKDTree(pos, boxsize=L)
    cn = []
    for p in np.where(sym == "P")[0]:
        cn.append(sum(1 for j in t.query_ball_point(pos[p], 2.4) if sym[j] == "S"))
    cn = np.array(cn)

    def dmin(a, b):
        ia, ib = np.where(sym == a)[0], np.where(sym == b)[0]
        d, _ = cKDTree(pos[ib], boxsize=L).query(pos[ia], k=2 if a == b else 1)
        d = d[:, 1] if a == b else d
        return float(d.min())
    return dict(cn_lt4=int((cn < 4).sum()), cn_le2=int((cn <= 2).sum()),
                dPS=dmin("P", "S"), dSS=dmin("S", "S"), dLiLi=dmin("Li", "Li"))


def fmt(s):
    return "cn<4=%d cn<=2=%d minPS=%.2f minSS=%.2f minLiLi=%.2f" % (
        s["cn_lt4"], s["cn_le2"], s["dPS"], s["dSS"], s["dLiLi"])


def snapshot(at):
    a = at.copy()
    a.calc = None
    a.info["energy"] = float(at.get_potential_energy())
    a.arrays["forces"] = at.get_forces(apply_constraint=False)
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--fmax", type=float, default=0.03)
    ap.add_argument("--relax_steps", type=int, default=1500)
    ap.add_argument("--temp", type=float, default=700.0)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--dt", type=float, default=2.0, help="fs")
    ap.add_argument("--every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    at = read(args.data)
    fixed = at.arrays.get("fixed", np.zeros(len(at), int))
    at.set_constraint(FixAtoms(indices=np.where(fixed == 1)[0]))
    at.calc = make_calc(args.model, args.device)
    print("[init ] %s  E=%.4f" % (fmt(ps4_state(at)), at.get_potential_energy()), flush=True)

    t0 = time.time()
    opt = FIRE(at, logfile=None)
    conv = opt.run(fmax=args.fmax, steps=args.relax_steps)
    print("[relax] %s  E=%.4f steps=%d converged=%s (%.0fs)" % (
        fmt(ps4_state(at)), at.get_potential_energy(), opt.get_number_of_steps(),
        conv, time.time() - t0), flush=True)
    write(args.out + "_relax.extxyz", snapshot(at))

    if args.steps <= 0:
        return
    MaxwellBoltzmannDistribution(at, temperature_K=args.temp,
                                 rng=np.random.default_rng(args.seed))
    md = Langevin(at, args.dt * units.fs, temperature_K=args.temp, friction=0.02,
                  rng=np.random.default_rng(args.seed + 1))
    frames = []
    t0 = time.time()
    for k in range(0, args.steps, args.every):
        md.run(args.every)
        s = ps4_state(at)
        snap = snapshot(at)
        snap.info.update(step=k + args.every, **s)
        frames.append(snap)
        print("[md %5d] %s  T=%.0fK (%.1fs/step)" % (
            k + args.every, fmt(s), at.get_temperature(),
            (time.time() - t0) / (k + args.every)), flush=True)
    write(args.out + "_md.extxyz", frames)
    print("wrote %s_relax.extxyz, %s_md.extxyz (%d frames)" % (args.out, args.out, len(frames)))


if __name__ == "__main__":
    main()

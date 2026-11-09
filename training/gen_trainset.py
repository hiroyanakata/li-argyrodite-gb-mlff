#!/usr/bin/env python
# =============================================================================
# gen_trainset.py -- generate the training data the current set is missing.
#
# The existing store (data/Li094P016S078Br009I000Cl009.*, 49k frames) is 12,000
# MD snapshots of ONE 206-atom GB cell at ONE composition. It therefore contains
#   * no compositional variation (Cl:Br, Li vacancy count)
#   * no configurational variation (halide/vacancy arrangement, other GB planes)
#   * no bulk crystal at all -- every frame contains a grain boundary, so
#     nothing pins the crystalline PS4 stiffness (Raman says the tuned FF
#     softened nu3 by 13 %)
#   * only incidental PS4 distortion (15 % of frames hold one stretched P-S)
#
# This builds structures along exactly those axes and samples frames from each:
#   thermal (NVT MD) | volumetric (cell scaling) | PS4-TARGETED distortions
# The PS4 set is the point of the exercise: symmetric breathing (nu1), single
# bond stretch to dissociation, angular bend (nu2/nu4) and P off-centring (nu3)
# put the P sublattice -- the one the parity breakdown says is worst -- on a
# controlled grid instead of leaving it to thermal luck.
#
# Atoms are canonically sorted (Li P S Br I Cl) so every frame of one
# composition yields an identical type.raw, which is what scf_to_deepmd.py
# requires and what silently breaks a model when it drifts.
#
#   python gen_trainset.py --systems bulk,gb110 --model pretrained --out pool
# =============================================================================
import argparse
import itertools
import os
import subprocess
import sys

import numpy as np

try:
    import ase.constraints as _asec
    if not hasattr(_asec, "full_3x3_to_voigt_6_stress"):
        from ase.stress import (full_3x3_to_voigt_6_stress as _f3,
                                voigt_6_to_full_3x3_stress as _v6)
        _asec.full_3x3_to_voigt_6_stress = _f3
        _asec.voigt_6_to_full_3x3_stress = _v6
except Exception:
    pass

from ase import units
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.optimize import FIRE

ATMTYP = ["Li", "P", "S", "Br", "I", "Cl"]      # scf_to_deepmd.py type order
A_ARGY = 9.859                                   # Li6PS5Cl lattice constant [A]


# --------------------------------------------------------------------------
def make_calc(model, device):
    from mattersim.forcefield import MatterSimCalculator
    if model in ("pretrained", "generic", "default"):
        return MatterSimCalculator(device=device)
    return MatterSimCalculator(load_path=model, device=device)


def canonical_sort(atoms):
    """order atoms by ATMTYP, stable within an element -> identical type.raw."""
    sym = atoms.get_chemical_symbols()
    rank = [ATMTYP.index(s) for s in sym]
    order = np.argsort(np.array(rank), kind="stable")
    return atoms[order]


def mic(d, cell):
    inv = np.linalg.inv(cell)
    f = d @ inv
    f -= np.round(f)
    return f @ cell


# --------------------------------------------------------------------------
def build_bulk(nx=1, halide="Cl", br_frac=0.0, nvac=0, seed=0, min_lili=2.30):
    """ordered bulk Li6PS5X approximant from the F-43m argyrodite structure.

    from_spacegroup fills the Li 48h site completely (48 Li per cell) but only
    half of it is occupied in the real material, so half the Li are removed
    under a minimum Li-Li separation -- which is also a genuine source of
    configurational diversity (each seed is a different Li ordering)."""
    from pymatgen.core import Structure, Lattice
    from pymatgen.io.ase import AseAtomsAdaptor
    st = Structure.from_spacegroup(
        "F-43m", Lattice.cubic(A_ARGY),
        ["P", "S", "S", "Cl", "Li"],
        [[0.5, 0.5, 0.5], [0.6194, 0.6194, 0.6194], [0.0, 0.0, 0.0],
         [0.75, 0.75, 0.75], [0.3182, 0.3182, 0.0163]])
    at = AseAtomsAdaptor.get_atoms(st)
    if nx > 1:
        at = at.repeat((nx, nx, nx))
    at.set_pbc(True)
    rng = np.random.default_rng(seed)
    cell = np.array(at.get_cell())
    sym = np.array(at.get_chemical_symbols())

    # --- Li 48h -> 24h: greedy half-occupancy with a hard-core constraint ---
    li = np.where(sym == "Li")[0]
    pos = at.get_positions()
    target = len(li) // 2 - nvac
    order = rng.permutation(li)
    keep = []
    for i in order:
        if len(keep) >= target:
            break
        if not keep:
            keep.append(i); continue
        d = np.linalg.norm(mic(pos[keep] - pos[i], cell), axis=1)
        if d.min() >= min_lili:
            keep.append(i)
    if len(keep) < target:            # relax the constraint if we ran short
        for i in order:
            if len(keep) >= target:
                break
            if i not in keep:
                keep.append(i)
    drop = sorted(set(li.tolist()) - set(int(k) for k in keep), reverse=True)
    del at[drop]

    # --- halide mixing on the 4d site, and vacancy charge compensation ------
    sym = np.array(at.get_chemical_symbols())
    cl = np.where(sym == "Cl")[0]
    nbr = int(round(br_frac * len(cl)))
    for i in rng.permutation(cl)[:nbr]:
        at[int(i)].symbol = "Br"
    if halide == "Br":
        for i in np.where(np.array(at.get_chemical_symbols()) == "Cl")[0]:
            at[int(i)].symbol = "Br"
    # each Li vacancy removed above is compensated S -> X (Li6-xPS5-xX1+x)
    if nvac:
        sym = np.array(at.get_chemical_symbols())
        s_idx = np.where(sym == "S")[0]
        comp = "Br" if (halide == "Br" or br_frac > 0.5) else "Cl"
        for i in rng.permutation(s_idx)[:nvac]:
            at[int(i)].symbol = comp
    return canonical_sort(at)


def build_gb(gb_build, xyz, ratio, nvac, vac_region, seed, gb_axis, gb_pos,
             gb_width, workdir, neutral=True, scale=False):
    out = os.path.join(workdir, "gb_%s_nv%d_%s_s%d.extxyz"
                       % (ratio.replace(":", "").replace(",", "_"),
                          nvac, vac_region, seed))
    cmd = [sys.executable, gb_build, "--xyz", xyz, "--anion-ratio", ratio,
           "--nvac", str(nvac), "--vac-region", vac_region, "--seed", str(seed),
           "--gb-axis", gb_axis, "--gb-pos", gb_pos, "--gb-width", str(gb_width),
           "--out", out]
    if neutral:
        cmd.append("--neutral")
    if scale:
        cmd.append("--scale")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(out):
        print("  gb_build FAILED:", r.stdout[-300:], r.stderr[-300:])
        return None
    at = read(out)
    at.set_pbc(True)
    return canonical_sort(at)


# --------------------------------------------------------------------------
def ps4_units(atoms, rcut=2.8):
    sym = np.array(atoms.get_chemical_symbols())
    cell = np.array(atoms.get_cell())
    pos = atoms.get_positions()
    P = np.where(sym == "P")[0]
    S = np.where(sym == "S")[0]
    out = []
    for p in P:
        d = np.linalg.norm(mic(pos[S] - pos[p], cell), axis=1)
        o = np.argsort(d)[:4]
        if d[o].max() < rcut:
            out.append((int(p), [int(S[k]) for k in o]))
    return out


def distort_ps4(atoms, mode, amp, rng):
    """Controlled PS4 distortions -- the data the current store lacks.

    breathe : all four P-S scaled by (1+amp)      -> the nu1 curvature
    stretch : ONE P-S extended to 2.05*(1+amp)    -> the dissociation path
    bend    : two S rotated about P by amp rad    -> nu2 / nu4
    offcent : P displaced off the tetrahedron centre by amp A -> nu3
    """
    at = atoms.copy()
    cell = np.array(at.get_cell())
    pos = at.get_positions()
    units_ = ps4_units(at)
    if not units_:
        return None
    for p, ss in units_:
        if mode == "breathe":
            for s in ss:
                v = mic(pos[s] - pos[p], cell)
                pos[s] = pos[p] + v * (1.0 + amp)
        elif mode == "stretch":
            s = ss[rng.integers(len(ss))]
            v = mic(pos[s] - pos[p], cell)
            r = np.linalg.norm(v)
            pos[s] = pos[p] + v / r * (r + amp)
        elif mode == "bend":
            a, b = ss[0], ss[1]
            va, vb = mic(pos[a] - pos[p], cell), mic(pos[b] - pos[p], cell)
            n = np.cross(va, vb)
            n /= (np.linalg.norm(n) + 1e-12)
            th = amp
            for idx, v in ((a, va), (b, -vb)):
                c, s_ = np.cos(th), np.sin(th)
                vr = v * c + np.cross(n, v) * s_ + n * np.dot(n, v) * (1 - c)
                pos[idx] = pos[p] + (vr if idx == a else -vr)
        elif mode == "offcent":
            u = rng.normal(size=3)
            pos[p] = pos[p] + u / np.linalg.norm(u) * amp
    at.set_positions(pos)
    return at


# --------------------------------------------------------------------------
def sample_system(at0, calc, args, tag, rng):
    """thermal + volumetric + PS4-targeted frames from one structure."""
    frames = []
    at = at0.copy()
    at.calc = calc
    FIRE(at, logfile=None).run(fmax=args.relax_fmax, steps=args.relax_steps)
    base = at.copy()

    def stamp(a, kind, detail):
        a = a.copy()
        a.info.update(dict(sys=tag, kind=kind, detail=detail,
                           formula=a.get_chemical_formula()))
        a.calc = None
        return a

    # --- thermal: short NVT at several temperatures ------------------------
    for T in [float(v) for v in args.temps.split(",") if v]:
        a = base.copy(); a.calc = calc
        MaxwellBoltzmannDistribution(a, temperature_K=T, rng=rng)
        dyn = Langevin(a, args.dt * units.fs, temperature_K=T,
                       friction=0.01 / units.fs)
        dyn.run(args.equil)
        for k in range(args.nmd):
            dyn.run(args.stride)
            frames.append(stamp(a, "thermal", "T%.0f_%d" % (T, k)))

    # --- volumetric: isotropic cell scaling (stress / EOS information) -----
    for eps in [float(v) for v in args.strains.split(",") if v]:
        a = base.copy()
        a.set_cell(np.array(base.get_cell()) * (1.0 + eps), scale_atoms=True)
        frames.append(stamp(a, "volume", "eps%+.3f" % eps))

    # --- PS4-targeted: the gap this whole exercise is about ----------------
    for mode, amps in (("breathe", args.breathe), ("stretch", args.stretch),
                       ("bend", args.bend), ("offcent", args.offcent)):
        for amp in [float(v) for v in amps.split(",") if v]:
            a = distort_ps4(base, mode, amp, rng)
            if a is not None:
                frames.append(stamp(a, "ps4_" + mode, "%s%+.3f" % (mode, amp)))
    # PS4 distortion on top of a hot frame, so it is not always the ideal cell
    hot = [f for f in frames if f.info["kind"] == "thermal"]
    if hot:
        for mode in ("stretch", "breathe"):
            amps = (args.stretch if mode == "stretch" else args.breathe)
            for amp in [float(v) for v in amps.split(",") if v][::2]:
                src = hot[rng.integers(len(hot))]
                a = distort_ps4(src, mode, amp, rng)
                if a is not None:
                    frames.append(stamp(a, "ps4_" + mode + "_hot",
                                        "%s%+.3f" % (mode, amp)))
    return frames


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--systems", default="bulk,gb110",
                    help="comma list of bulk,gb110,gb210")
    ap.add_argument("--gb_build", default=os.path.join(here, "gb_build.py"))
    ap.add_argument("--xyz110", default="")
    ap.add_argument("--xyz210", default="")
    ap.add_argument("--gb_axis", default="x")
    ap.add_argument("--gb_pos", default="0.25,0.75")
    ap.add_argument("--gb_width", type=float, default=1.5)
    ap.add_argument("--ratios", default="S:1,Cl:0.5,Br:0.5;S:1,Cl:1,Br:0;S:1,Cl:0,Br:1")
    ap.add_argument("--nvacs", default="0,1,2,4")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--bulk_nx", type=int, default=1)
    ap.add_argument("--bulk_brfracs", default="0.0,0.5,1.0")
    ap.add_argument("--model", default="pretrained")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--relax_fmax", type=float, default=0.05)
    ap.add_argument("--relax_steps", type=int, default=300)
    ap.add_argument("--temps", default="300,600,900")
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--equil", type=int, default=300)
    ap.add_argument("--nmd", type=int, default=4)
    ap.add_argument("--stride", type=int, default=200)
    ap.add_argument("--strains", default="-0.05,-0.03,-0.015,0.015,0.03,0.05")
    ap.add_argument("--breathe", default="-0.06,-0.03,0.03,0.06")
    ap.add_argument("--stretch", default="0.15,0.35,0.60,1.00")
    ap.add_argument("--bend", default="0.10,0.20")
    ap.add_argument("--offcent", default="0.10,0.20")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workdir", default="pool_build")
    ap.add_argument("--out", default="pool")
    args = ap.parse_args()

    os.makedirs(args.workdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    calc = make_calc(args.model, args.device)
    systems = [s for s in args.systems.split(",") if s]
    ratios = [r for r in args.ratios.split(";") if r]
    nvacs = [int(v) for v in args.nvacs.split(",") if v != ""]

    specs = []
    if "bulk" in systems:
        for brf in [float(v) for v in args.bulk_brfracs.split(",") if v != ""]:
            for nv in nvacs:
                for sd in range(args.seeds):
                    specs.append(("bulk_br%.2f_nv%d_s%d" % (brf, nv, sd),
                                  dict(kind="bulk", br_frac=brf, nvac=nv, seed=sd)))
    for name, xyz in (("gb110", args.xyz110), ("gb210", args.xyz210)):
        if name in systems and xyz:
            for r in ratios:
                for nv in nvacs:
                    for reg in ("gb", "bulk"):
                        for sd in range(args.seeds):
                            specs.append(("%s_%s_nv%d_%s_s%d" %
                                          (name, r.replace(":", "").replace(",", "_"),
                                           nv, reg, sd),
                                          dict(kind=name, xyz=xyz, ratio=r,
                                               nvac=nv, vac_region=reg, seed=sd)))

    print("systems to sample: %d" % len(specs))
    allframes = []
    for tag, kw in specs:
        if kw["kind"] == "bulk":
            at0 = build_bulk(nx=args.bulk_nx, br_frac=kw["br_frac"],
                             nvac=kw["nvac"], seed=kw["seed"])
        else:
            at0 = build_gb(args.gb_build, kw["xyz"], kw["ratio"], kw["nvac"],
                           kw["vac_region"], kw["seed"], args.gb_axis,
                           args.gb_pos, args.gb_width, args.workdir)
        if at0 is None:
            continue
        fr = sample_system(at0, calc, args, tag, rng)
        allframes.extend(fr)
        print("  %-40s %-22s %4d frames" %
              (tag, at0.get_chemical_formula(), len(fr)))

    # --- split by composition: one DeePMD system per type.raw --------------
    groups = {}
    for a in allframes:
        groups.setdefault(a.get_chemical_formula(), []).append(a)
    os.makedirs(args.out, exist_ok=True)
    print("\n%-26s %7s %7s  %s" % ("composition", "frames", "natoms", "kinds"))
    for form, fr in sorted(groups.items()):
        write(os.path.join(args.out, form + ".extxyz"), fr)
        kinds = {}
        for a in fr:
            kinds[a.info["kind"]] = kinds.get(a.info["kind"], 0) + 1
        print("%-26s %7d %7d  %s" % (form, len(fr), len(fr[0]),
                                     ",".join("%s:%d" % kv for kv in sorted(kinds.items()))))
    write(os.path.join(args.out, "ALL.extxyz"), allframes)
    print("\ntotal %d frames, %d compositions -> %s/" %
          (len(allframes), len(groups), args.out))


if __name__ == "__main__":
    main()

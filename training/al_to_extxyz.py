#!/usr/bin/env python
"""Convert converged AL single-point pw.x outputs -> extxyz for v5C training.

Numeric conventions are byte-identical to scf_to_deepmd.py / AutoReaction.v17.py:
  ry2ev = 13.6058, bohr2ang = 0.529177 (NOT CODATA).
parse_out() only accepts frames with both "JOB DONE." and
"convergence has been achieved" -> non-converged / still-running jobs are
skipped automatically. The explicit drop list is a belt-and-suspenders guard.

Output: v5C_al_train.extxyz (in_train runs), v5C_al_test.extxyz (in_test runs).
DFT energy/forces go on a SinglePointCalculator; selected AL metadata
(role, src, step, split, T, cn_*, d*) is carried over from the al*_?.extxyz
frame matched by the index in the filename (AL.<run>.<idx> -> <run>.extxyz[idx]).
"""
import os
import re
import sys

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write

BOHR2ANG = 0.529177
RY2EV = 13.6058
RytoeV = 13.605698066819
ATMTYP = ["Li", "P", "S", "Br", "I", "Cl"]
# cohesive-energy zero: byte-identical to assemble_v5_train.py / harvest_qe_to_extxyz.py.
# v5_trainB.extxyz stores energy = raw*RY2EV - ATMENE.sum(); v5C AL frames MUST match.
ATMENE = np.asarray([-14.33801816, -13.66813830, -23.81230945,
                     -46.73796118, -378.81243284, -33.16874338]) * RytoeV

# directory holding al*_{train,test}.extxyz and the AL.<run>.<idx>.out QE outputs
AL = os.environ.get("AL_DIR", ".")
# filename run token -> extxyz basename
RUN2XYZ = {"al1train": "al1_train", "al2train": "al2_train",
           "al1test": "al1_test", "al2test": "al2_test"}
CARRY = ["role", "src", "step", "split", "T", "cn_lt4", "cn_le2",
         "dPS", "dSS", "dLiLi", "gb_x_in_cut"]


def parse_out(path, nat_expect=None):
    with open(path, "r", errors="replace") as fh:
        lines = fh.readlines()
    txt = "".join(lines)
    if "JOB DONE." not in txt:
        return None, "no JOB DONE"
    if "convergence has been achieved" not in txt:
        return None, "scf not converged"
    m = re.search(r"number of atoms/cell\s*=\s*(\d+)", txt)
    if not m:
        return None, "no nat"
    nat = int(m.group(1))
    if nat_expect is not None and nat != nat_expect:
        return None, f"nat {nat} != {nat_expect}"
    alat = None
    cell = np.zeros((3, 3))
    coord = np.zeros((nat, 3))
    force = np.zeros((nat, 3))
    atm = [None] * nat
    energy = None
    got_cell = got_coord = got_force = False
    i, n = 0, len(lines)
    while i < n:
        aa = lines[i].split()
        if len(aa) > 3 and aa[0] == "celldm(1)=" and alat is None:
            alat = float(aa[1]) * BOHR2ANG
        elif len(aa) > 1 and aa[0] == "crystal" and aa[1] == "axes:":
            for k in range(3):
                bb = lines[i + 1 + k].split()
                cell[k] = [float(bb[3]) * alat, float(bb[4]) * alat,
                           float(bb[5]) * alat]
            got_cell = True
            i += 3
        elif len(aa) > 1 and aa[0] == "Cartesian" and aa[1] == "axes":
            for k in range(nat):
                bb = lines[i + 3 + k].split()
                atm[k] = bb[1]
                coord[k] = [float(bb[6]) * alat, float(bb[7]) * alat,
                            float(bb[8]) * alat]
            got_coord = True
            i += 2 + nat
        elif len(aa) > 1 and aa[0] == "!" and aa[1] == "total":
            energy = float(aa[4]) * RY2EV
        elif len(aa) > 1 and aa[0] == "Forces" and aa[1] == "acting":
            for k in range(nat):
                bb = lines[i + 2 + k].split()
                force[k] = [float(bb[6]) * RY2EV / BOHR2ANG,
                            float(bb[7]) * RY2EV / BOHR2ANG,
                            float(bb[8]) * RY2EV / BOHR2ANG]
            got_force = True
            i += 1 + nat
        i += 1
    if not (got_cell and got_coord and got_force) or energy is None:
        miss = [k for k, v in [("cell", got_cell), ("coord", got_coord),
                               ("force", got_force), ("energy", energy is not None)]
                if not v]
        return None, "missing " + ",".join(miss)
    bad = sorted({a for a in atm if a not in ATMTYP})
    if bad:
        return None, f"element(s) not in {ATMTYP}: {bad}"
    return (cell, coord, atm, energy, force), ""


def load_dropped():
    p = os.path.join(AL, "al_dropped_nonconv.txt")
    d = set()
    if os.path.exists(p):
        for ln in open(p):
            ln = ln.strip()
            if ln and not ln.startswith("#"):
                d.add(ln)
    return d


def load_xyz_cache():
    cache = {}
    for xyz in set(RUN2XYZ.values()):
        p = os.path.join(AL, xyz + ".extxyz")
        try:
            cache[xyz] = read(p, index=":")
        except Exception as e:
            print(f"  warn: cannot read {xyz}.extxyz ({e}); metadata skipped")
            cache[xyz] = None
    return cache


def main():
    dropped = load_dropped()
    xyz_cache = load_xyz_cache()
    out = {"train": [], "test": []}
    stats = {"train": [], "test": []}
    skipped = []
    pat = re.compile(r"^AL\.(al[12](?:train|test))\.(\d+)\.out$")

    for sub, split in [("in_train", "train"), ("in_test", "test")]:
        d = os.path.join(AL, sub)
        for f in sorted(os.listdir(d)):
            mobj = pat.match(f)              # skips AL.*.<jobid>.out husks
            if not mobj:
                continue
            run_tok, idx = mobj.group(1), int(mobj.group(2))
            name = f[:-4]                     # AL.al1train.11
            if name in dropped:
                skipped.append((name, "in drop list"))
                continue
            res, why = parse_out(os.path.join(d, f))
            if res is None:
                skipped.append((name, why))
                continue
            cell, coord, atm, energy, force = res
            ezero = ATMENE[[ATMTYP.index(s) for s in atm]].sum()
            e_coh = energy - ezero          # cohesive-E zero, matches v5_trainB
            at = Atoms(symbols=atm, positions=coord, cell=cell, pbc=True)
            at.calc = SinglePointCalculator(at, energy=e_coh, forces=force)
            info = at.info
            info["name"] = name
            info["al_run"] = RUN2XYZ[run_tok]
            info["dft_energy_raw"] = energy       # raw QE total * 13.6058 [eV]
            info["config_type"] = "al"
            # carry metadata from the matching extxyz frame
            frames = xyz_cache.get(RUN2XYZ[run_tok])
            if frames is not None and idx < len(frames):
                src = frames[idx]
                if len(src) == len(at):
                    for k in CARRY:
                        if k in src.info:
                            info[k] = src.info[k]
                else:
                    info["meta_mismatch"] = f"nat {len(at)} vs xyz {len(src)}"
            out[split].append(at)
            stats[split].append((e_coh / len(at),
                                 float(np.linalg.norm(force, axis=1).max())))

    for split in ("train", "test"):
        fn = os.path.join(AL, f"v5C_al_{split}.extxyz")
        if out[split]:
            write(fn, out[split], format="extxyz")
        n = len(out[split])
        if n:
            ea = np.array([s[0] for s in stats[split]])
            fm = np.array([s[1] for s in stats[split]])
            print(f"[{split}] wrote {n} frames -> {os.path.basename(fn)}")
            print(f"        E_coh/atom min {ea.min():.3f}  p50 {np.median(ea):.3f} "
                  f" max {ea.max():.3f} eV")
            print(f"        fmax    p50 {np.median(fm):.3f}  max {fm.max():.3f} eV/A")
        else:
            print(f"[{split}] 0 frames (nothing converged yet)")

    print(f"\nskipped {len(skipped)}:")
    from collections import Counter
    for why, c in Counter(w for _, w in skipped).most_common():
        print(f"  {c:3d}  {why}")


if __name__ == "__main__":
    main()

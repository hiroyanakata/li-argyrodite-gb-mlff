#!/usr/bin/env python
"""Assemble v5 clean fine-tune training sets (no-stress), post PS4-bug.

Base  = v3_train_reduce_emb.extxyz (5271 frames, matches audit_ps4/frames.csv).
  A (control)  = full base + new clean DFT
  B (primary)  = base MINUS frames with n_vertexX>0 (per frames.csv) + new clean DFT
New clean DFT = TRAIN structures/compositions ONLY, harvested from QE SCF *.out
with the SAME cohesive-energy zero as scripts/harvest_qe_to_extxyz.py (NO stress):
  GB  MD  : ps4ok/dft_mdH_{rnd0,seg2,d4}_*
  GB  NEB : ps4ok/dft_nebS_{rnd0,seg2,d4}_*
  bulk MD : bulkdef/dft_mdH_<comp>_*                         (exclude 6 TEST comps)
  bulk NEB: bulkdef/dft_nebS_{Cl8Li48P8S40,Br2Cl10Li44P8S36}_*  (exclude Br6Cl6 TEST)
Excludes v5 TEST per TEST_SPLIT_v5.md: GB {rnd1,seg3,d2}; bulk 6 mixed comps.
The OLD phaseB barrier set (272 frames, PS3Cl-contaminated) is fully replaced.

Oversample via frame duplication (finetune reads one extxyz, no per-frame weight):
  NEB saddle frames x NEB_MULT ; MD frames x MD_MULT.

  python assemble_v5_train.py \
    --base v5build/v3_train_reduce_emb.extxyz --frames_csv v5build/frames.csv \
    --gb_root ps4ok --bulk_root bulkdef \
    --neb_mult 3 --md_mult 1 \
    --outA v5build/v5_trainA.extxyz --outB v5build/v5_trainB.extxyz \
    --out_dft v5build/v5_clean_dft.extxyz
"""
import argparse, os, re, glob
import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write

# --- constants: byte-identical to scripts/harvest_qe_to_extxyz.py ---
BOHR2ANG = 0.529177
RY2EV = 13.6058
RytoeV = 13.605698066819
ATMTYP = ["Li", "P", "S", "Br", "I", "Cl"]
ATMENE = np.asarray([-14.33801816, -13.66813830, -23.81230945,
                     -46.73796118, -378.81243284, -33.16874338]) * RytoeV

GB_TRAIN = {"rnd0", "seg2", "d4"}
GB_TEST = {"rnd1", "seg3", "d2"}
BULK_TEST_COMPS = {"Br2Cl6Li20P4S16", "Br2Cl4Li22P4S18", "Br2Cl3Li23P4S19",
                   "Br6Cl6Li44P8S36", "Br5Cl5Li46P8S38", "Br4Cl5Li47P8S39"}
BULK_NEB_TRAIN = {"Cl8Li48P8S40", "Br2Cl10Li44P8S36"}  # Br6Cl6Li44P8S36 is TEST


def parse_out(path):
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
    alat = None
    cell = np.zeros((3, 3)); coord = np.zeros((nat, 3)); force = np.zeros((nat, 3))
    atm = [None] * nat; energy = None
    got_cell = got_coord = got_force = False
    i, n = 0, len(lines)
    while i < n:
        aa = lines[i].split()
        if len(aa) > 3 and aa[0] == "celldm(1)=" and alat is None:
            alat = float(aa[1]) * BOHR2ANG
        elif len(aa) > 1 and aa[0] == "crystal" and aa[1] == "axes:":
            for k in range(3):
                bb = lines[i + 1 + k].split()
                cell[k] = [float(bb[3]) * alat, float(bb[4]) * alat, float(bb[5]) * alat]
            got_cell = True; i += 3
        elif len(aa) > 1 and aa[0] == "Cartesian" and aa[1] == "axes":
            for k in range(nat):
                bb = lines[i + 3 + k].split()
                atm[k] = bb[1]
                coord[k] = [float(bb[6]) * alat, float(bb[7]) * alat, float(bb[8]) * alat]
            got_coord = True; i += 2 + nat
        elif len(aa) > 1 and aa[0] == "!" and aa[1] == "total":
            energy = float(aa[4]) * RY2EV
        elif len(aa) > 1 and aa[0] == "Forces" and aa[1] == "acting":
            for k in range(nat):
                bb = lines[i + 2 + k].split()
                force[k] = [float(bb[6]) * RY2EV / BOHR2ANG,
                            float(bb[7]) * RY2EV / BOHR2ANG,
                            float(bb[8]) * RY2EV / BOHR2ANG]
            got_force = True; i += 1 + nat
        i += 1
    if not (got_cell and got_coord and got_force) or energy is None:
        return None, "incomplete"
    return (cell, coord, atm, energy, force), ""


def img_index(fname):
    m = re.search(r"\.(\d+)\.out$", fname)
    return int(m.group(1)) if m else -1


def harvest_dir(d, src, config_type):
    frames = []
    outs = sorted(glob.glob(os.path.join(d, "*.out")),
                  key=lambda p: img_index(os.path.basename(p)))
    for p in outs:
        res, _ = parse_out(p)
        if res is None:
            continue
        cell, coord, atm, energy, force = res
        types = [ATMTYP.index(s) for s in atm]
        ezero = ATMENE[types].sum()
        at = Atoms(symbols=atm, positions=coord, cell=cell, pbc=[True, True, True])
        at.calc = SinglePointCalculator(at, energy=energy - ezero, forces=force)
        at.info["src"] = src
        at.info["img"] = img_index(os.path.basename(p))
        at.info["config_type"] = config_type
        frames.append(at)
    return frames


def dup(frames, mult):
    out = []
    for at in frames:
        e = at.get_potential_energy(); f = at.get_forces()
        for _ in range(mult):
            c = at.copy()
            c.calc = SinglePointCalculator(c, energy=e, forces=f)
            c.info.update(at.info)
            out.append(c)
    return out


def tok0(dirname, prefix):
    return os.path.basename(dirname.rstrip("/"))[len(prefix):].split("_")[0]


def load_vertexX_mask(csv_path, base_file="v3_train_reduce_emb.extxyz"):
    """Return list keep[i] for base frames (True = keep in set B: n_vertexX==0)."""
    rows = []
    with open(csv_path) as fh:
        header = fh.readline().rstrip("\n").split(",")
        fi, ff, fv = header.index("file"), header.index("frame"), header.index("n_vertexX")
        for line in fh:
            c = line.rstrip("\n").split(",")
            if c[fi] == base_file:
                rows.append((int(c[ff]), int(c[fv])))
    rows.sort()
    return [v == 0 for _, v in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--frames_csv", required=True)
    ap.add_argument("--gb_root", required=True)
    ap.add_argument("--bulk_root", required=True)
    ap.add_argument("--neb_mult", type=int, default=3)
    ap.add_argument("--md_mult", type=int, default=1)
    ap.add_argument("--outA", required=True)
    ap.add_argument("--outB", required=True)
    ap.add_argument("--out_dft", required=True)
    a = ap.parse_args()

    # ---- harvest new clean DFT (train only) ----
    md, neb = [], []
    acct = {}
    # GB MD
    for d in sorted(glob.glob(os.path.join(a.gb_root, "dft_mdH_*"))):
        st = tok0(d, "dft_mdH_")
        if st not in GB_TRAIN:
            continue
        fr = harvest_dir(d, "gbMD_" + os.path.basename(d)[len("dft_mdH_"):], "gb_md")
        md += fr; acct[os.path.basename(d)] = len(fr)
    # GB NEB
    for d in sorted(glob.glob(os.path.join(a.gb_root, "dft_nebS_*"))):
        st = tok0(d, "dft_nebS_")
        if st not in GB_TRAIN:
            continue
        fr = harvest_dir(d, "gbNEB_" + os.path.basename(d)[len("dft_nebS_"):], "gb_neb")
        neb += fr; acct[os.path.basename(d)] = len(fr)
    # bulk MD
    for d in sorted(glob.glob(os.path.join(a.bulk_root, "dft_mdH_*"))):
        comp = tok0(d, "dft_mdH_")
        if comp in BULK_TEST_COMPS:
            continue
        fr = harvest_dir(d, "bulkMD_" + comp, "bulk_md")
        md += fr; acct[os.path.basename(d)] = len(fr)
    # bulk NEB
    for d in sorted(glob.glob(os.path.join(a.bulk_root, "dft_nebS_*"))):
        comp = tok0(d, "dft_nebS_")
        if comp not in BULK_NEB_TRAIN:
            continue
        fr = harvest_dir(d, "bulkNEB_" + comp, "bulk_neb")
        neb += fr; acct[os.path.basename(d)] = len(fr)

    print("=== per-dir harvested frame counts (train only) ===")
    for k in sorted(acct):
        print(f"  {k:<42} {acct[k]}")
    print(f"\nraw harvest: MD={len(md)}  NEB={len(neb)}")

    md_d = dup(md, a.md_mult)
    neb_d = dup(neb, a.neb_mult)
    dft = md_d + neb_d
    write(a.out_dft, dft, format="extxyz")
    print(f"clean DFT after oversample (MDx{a.md_mult}, NEBx{a.neb_mult}): "
          f"MD={len(md_d)} + NEB={len(neb_d)} = {len(dft)}")

    # ---- base + sets ----
    base = read(a.base, ":")
    for at in base:
        at.info.setdefault("src", "v3train")
    print(f"\nbase (v3 train): {len(base)} frames")

    keepB = load_vertexX_mask(a.frames_csv)
    if len(keepB) != len(base):
        raise SystemExit(f"FATAL: frames.csv rows ({len(keepB)}) != base frames ({len(base)}); wrong base file?")
    baseB = [at for at, k in zip(base, keepB) if k]
    print(f"base B (n_vertexX==0): {len(baseB)} frames (dropped {len(base) - len(baseB)})")

    write(a.outA, list(base) + dft, format="extxyz")
    write(a.outB, baseB + dft, format="extxyz")
    print(f"\nwrote {a.outA}: {len(base) + len(dft)} frames (base {len(base)} + dft {len(dft)})")
    print(f"wrote {a.outB}: {len(baseB) + len(dft)} frames (baseB {len(baseB)} + dft {len(dft)})")

    Eall = np.array([f.get_potential_energy() / len(f) for f in dft])
    print(f"\nclean DFT E_coh/atom  min {Eall.min():.3f}  p50 {np.median(Eall):.3f}  max {Eall.max():.3f} eV")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Pointwise Li-hop barrier eval on the HELD-OUT test paths (TEST_SPLIT_v5 main
criterion). For each DFT-labelled path we read the per-image geometry from the QE
*.out (identical geometry to the DFT single point), evaluate the given MLIP as an
ASE calculator, and compare MLIP barrier = max(E_i - E_0) to the DFT barrier.
Barriers are reference-free (differences along a fixed path), so MLIP/DFT energy
zeros need not match.

TEST paths (never in training):
  GB  : ps4ok/dft_nebS_{rnd1,seg3,d2}_*        (incl unused Li pairs 7_47/84_104/7_45/80_96)
  bulk: bulkdef/dft_nebS_Br6Cl6Li44P8S36_*

  python eval_v5_barriers.py --model models/phaseB.pth --device cuda \
     --gb_root ps4ok --bulk_root bulkdef --out eval_v5/bar_phaseB.csv
"""
import argparse, glob, os, re
import numpy as np
from ase import Atoms

BOHR2ANG = 0.529177
RY2EV = 13.6058
GB_TEST = ("rnd1", "seg3", "d2")
BULK_TEST = ("Br6Cl6Li44P8S36",)


def parse_out(path):
    with open(path, "r", errors="replace") as fh:
        lines = fh.readlines()
    txt = "".join(lines)
    if "JOB DONE." not in txt or "convergence has been achieved" not in txt:
        return None
    m = re.search(r"number of atoms/cell\s*=\s*(\d+)", txt)
    if not m:
        return None
    nat = int(m.group(1)); alat = None
    cell = np.zeros((3, 3)); coord = np.zeros((nat, 3)); atm = [None] * nat; energy = None
    gc = gx = False
    i, n = 0, len(lines)
    while i < n:
        aa = lines[i].split()
        if len(aa) > 3 and aa[0] == "celldm(1)=" and alat is None:
            alat = float(aa[1]) * BOHR2ANG
        elif len(aa) > 1 and aa[0] == "crystal" and aa[1] == "axes:":
            for k in range(3):
                bb = lines[i + 1 + k].split()
                cell[k] = [float(bb[3]) * alat, float(bb[4]) * alat, float(bb[5]) * alat]
            gc = True; i += 3
        elif len(aa) > 1 and aa[0] == "Cartesian" and aa[1] == "axes":
            for k in range(nat):
                bb = lines[i + 3 + k].split()
                atm[k] = bb[1]
                coord[k] = [float(bb[6]) * alat, float(bb[7]) * alat, float(bb[8]) * alat]
            gx = True; i += 2 + nat
        elif len(aa) > 1 and aa[0] == "!" and aa[1] == "total":
            energy = float(aa[4]) * RY2EV
        i += 1
    if not (gc and gx) or energy is None:
        return None
    return cell, coord, atm, energy


def img_index(fname):
    m = re.search(r"\.(\d+)\.out$", fname)
    return int(m.group(1)) if m else -1


def path_barriers(d, calc):
    """Return (barrier_mlip, barrier_dft, n_images)."""
    Edft, Emlip = {}, {}
    for p in glob.glob(os.path.join(d, "*.out")):
        r = parse_out(p)
        if r is None:
            continue
        cell, coord, atm, edft = r
        idx = img_index(os.path.basename(p))
        at = Atoms(symbols=atm, positions=coord, cell=cell, pbc=[True, True, True])
        at.calc = calc
        Emlip[idx] = at.get_potential_energy()
        Edft[idx] = edft
    common = sorted(set(Edft) & set(Emlip))
    if 0 not in common or len(common) < 2:
        return None, None, len(common)
    bm = max(Emlip[i] - Emlip[0] for i in common)
    bd = max(Edft[i] - Edft[0] for i in common)
    return bm, bd, len(common)


def spearman(a, b):
    a, b = np.asarray(a), np.asarray(b)
    ra = a.argsort().argsort().astype(float)
    rb = b.argsort().argsort().astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / d) if d else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="pth path, or 'pretrained'")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gb_root", required=True)
    ap.add_argument("--bulk_root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from mattersim.forcefield import MatterSimCalculator
    calc = (MatterSimCalculator(device=a.device) if a.model == "pretrained"
            else MatterSimCalculator(load_path=a.model, device=a.device))

    dirs = []
    for st in GB_TEST:
        dirs += sorted(glob.glob(os.path.join(a.gb_root, f"dft_nebS_{st}_*")))
    for comp in BULK_TEST:
        dirs += sorted(glob.glob(os.path.join(a.bulk_root, f"dft_nebS_{comp}_*")))

    rows, errs, bm_all, bd_all = [], [], [], []
    for d in dirs:
        tag = os.path.basename(d)[len("dft_nebS_"):]
        bm, bd, nimg = path_barriers(d, calc)
        if bm is None:
            rows.append((tag, None, None, None, nimg)); continue
        e = abs(bm - bd)
        rows.append((tag, bm, bd, e, nimg))
        errs.append(e); bm_all.append(bm); bd_all.append(bd)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        fh.write("tag,MLIP_eV,DFT_eV,abs_err_eV,n_img\n")
        print(f"{'tag':<34}{'MLIP':>9}{'DFT':>9}{'|err|':>9}{'n':>4}")
        for tag, bm, bd, e, nimg in rows:
            s = (f"{tag:<34}{bm:>9.3f}{bd:>9.3f}{e:>9.3f}{nimg:>4}"
                 if bm is not None else f"{tag:<34}{'NA':>9}{'NA':>9}{'NA':>9}{nimg:>4}")
            print(s)
            fh.write(f"{tag},{bm if bm is not None else ''},{bd if bd is not None else ''},"
                     f"{e if e is not None else ''},{nimg}\n")
    if errs:
        errs = np.array(errs)
        print(f"\nMODEL={a.model}")
        print(f"MAE={errs.mean()*1000:.1f} meV  max={errs.max()*1000:.1f}  "
              f"median={np.median(errs)*1000:.1f}  n={len(errs)}")
        no_blow = errs[errs < 2.0]
        print(f"MAE (excl >2eV)={no_blow.mean()*1000:.1f} meV  n={len(no_blow)}")
        print(f"Spearman(MLIP,DFT ordering)={spearman(bm_all, bd_all):.3f}")


if __name__ == "__main__":
    main()

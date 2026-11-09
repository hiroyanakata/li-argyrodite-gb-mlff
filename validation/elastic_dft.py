#!/usr/bin/env python
# =============================================================================
# elastic_dft.py -- generate the finite-strain frames for a DFT (or MLIP)
#                   elastic-constant calculation. Writes ref + 6x(+/-delta)
#                   deformed cells to one extxyz; each frame is tagged with the
#                   Voigt component and strain in info[] so elastic_from_stress.py
#                   can fit C_ij = d(sigma_i)/d(eps_j) from whatever stress source.
#
# This makes the DFT and the MLIP elastic constants use the *identical* strained
# geometries (clamped-ion), so the comparison is exact. Optionally MLIP-relax the
# reference cell first so it sits at ~0 stress.
#
#   python elastic_dft.py --data gb210_dec.extxyz --model best_model.pth \
#          --relax_ref --strain 0.006 --out defo
#   # -> defo.extxyz (13 frames). Then:
#   python make_scf_inputs.py --traj defo.extxyz --out defo_dft --stress \
#          --min_dist 0 --fmax_max 1e9 --pseudo_dir $QM
#   # ... run QE ... then elastic_from_stress.py (DFT and each MLIP model).
# =============================================================================
import argparse
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

from ase.io import read, write
from ase.optimize import BFGS


def voigt_strain_matrix(j, d):
    e = np.zeros((3, 3))
    if j < 3:
        e[j, j] = d
    else:
        a, b = {3: (1, 2), 4: (0, 2), 5: (0, 1)}[j]
        e[a, b] = e[b, a] = d / 2.0
    return e


def cell_filter(atoms):
    mask = [1, 1, 1, 0, 0, 0]
    try:
        from ase.filters import FrechetCellFilter
        return FrechetCellFilter(atoms, mask=mask)
    except Exception:
        from ase.constraints import ExpCellFilter
        return ExpCellFilter(atoms, mask=mask)


def main():
    ap = argparse.ArgumentParser(description="generate finite-strain frames for elastic Cij")
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="pretrained", help="MLIP used only for --relax_ref")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--relax_ref", action="store_true", help="MLIP-relax the reference cell first")
    ap.add_argument("--fmax", type=float, default=0.03)
    ap.add_argument("--relax_steps", type=int, default=300)
    ap.add_argument("--strain", type=float, nargs="+", default=[0.006])
    ap.add_argument("--out", default="defo")
    args = ap.parse_args()

    atoms = read(args.data)
    atoms.set_pbc(True)

    if args.relax_ref:
        from mattersim.forcefield import MatterSimCalculator
        atoms.calc = (MatterSimCalculator(device=args.device) if args.model.lower()
                      in ("", "pretrained", "default", "none")
                      else MatterSimCalculator(load_path=args.model, device=args.device))
        print("MLIP-relaxing reference cell (aniso) ...")
        BFGS(cell_filter(atoms), logfile=None).run(fmax=args.fmax, steps=args.relax_steps)
        atoms.calc = None

    ref_cell = np.array(atoms.get_cell())
    ref_pos = atoms.get_positions().copy()

    frames = []
    r = atoms.copy(); r.info.update(voigt=-1, delta=0.0, tag="ref")
    frames.append(r)
    for j in range(6):
        for d in sorted(set([+x for x in args.strain] + [-x for x in args.strain])):
            F = np.eye(3) + voigt_strain_matrix(j, d)
            a = atoms.copy()
            a.set_cell(ref_cell @ F.T, scale_atoms=False)
            a.set_positions(ref_pos @ F.T)
            a.info.update(voigt=j, delta=float(d), tag="v%d_%+.4f" % (j, d))
            frames.append(a)

    write(args.out + ".extxyz", frames)
    print("wrote %d frames -> %s.extxyz  (ref + 6 Voigt x %d strains)"
          % (len(frames), args.out, 2 * len(set(args.strain))))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# =============================================================================
# neb_dft_compare.py -- DFT single-points on the MLIP NEB path -> MLIP vs DFT
#                       migration barrier on the identical geometry.
#
# neb_barrier.py already wrote gb_neb_barrier.csv (reaction_coord, dE_eV for the
# MLIP) and gb_neb_path.extxyz (the images). Run those images through QE
# (make_scf_inputs -> sub_qe_scf), then:
#
#   python neb_dft_compare.py --barrier gb_neb_barrier.csv --qe_glob 'neb_dft/*.out' \
#          [--model best_model.pth --path gb_neb_path.extxyz] --out neb_cmp
#
# DFT energies are read (ASE espresso-out), referenced to the first image, and
# overlaid on the MLIP curve. Optionally re-evaluate the path with another MLIP
# checkpoint (e.g. the tuned model) to show all three on one plot.
# =============================================================================
import argparse
import glob
import re
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

from ase.io import read
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def main():
    ap = argparse.ArgumentParser(description="MLIP vs DFT NEB barrier on the same path")
    ap.add_argument("--barrier", required=True, help="gb_neb_barrier.csv (MLIP)")
    ap.add_argument("--qe_glob", required=True, help="glob of QE .out for the path images")
    ap.add_argument("--path", default="", help="gb_neb_path.extxyz (needed for --model)")
    ap.add_argument("--model", default="", help="extra MLIP checkpoint to overlay (e.g. tuned)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="neb_cmp")
    args = ap.parse_args()

    d = np.genfromtxt(args.barrier, delimiter=",", names=True)
    rc, dE_mlip = d["reaction_coord"], d["dE_eV"]

    outs = sorted(glob.glob(args.qe_glob), key=natural_key)
    E_dft = np.array([read(o, format="espresso-out").get_potential_energy() for o in outs])
    if len(E_dft) != len(rc):
        print("WARN: %d QE outs vs %d MLIP images; plotting DFT on its own index"
              % (len(E_dft), len(rc)))
        rc_dft = np.linspace(0, 1, len(E_dft))
    else:
        rc_dft = rc
    dE_dft = E_dft - E_dft[0]

    plt.figure(figsize=(6, 4))
    plt.plot(rc, dE_mlip, "o-", label="MLIP (Ea=%.3f eV)" % (dE_mlip.max() - dE_mlip[0]))
    plt.plot(rc_dft, dE_dft, "s--", label="DFT (Ea=%.3f eV)" % (dE_dft.max() - dE_dft[0]))

    if args.model and args.path:
        from mattersim.forcefield import MatterSimCalculator
        calc = (MatterSimCalculator(device=args.device) if args.model.lower()
                in ("pretrained", "default", "none")
                else MatterSimCalculator(load_path=args.model, device=args.device))
        imgs = read(args.path, index=":")
        E2 = []
        for a in imgs:
            b = a.copy(); b.set_pbc(True); b.calc = calc
            E2.append(b.get_potential_energy())
        E2 = np.array(E2) - E2[0]
        plt.plot(rc, E2, "^:", label="MLIP2 (Ea=%.3f eV)" % (E2.max() - E2[0]))
        print("MLIP2 (%s) barrier = %.3f eV (same path)"
              % (args.model.split("/")[-1], E2.max() - E2[0]))

    plt.xlabel("reaction coordinate"); plt.ylabel(r"$\Delta E$ [eV]")
    plt.legend(fontsize=9); plt.tight_layout()
    plt.savefig(args.out + ".png", dpi=150)
    print("MLIP barrier = %.3f eV,  DFT barrier = %.3f eV (same path)"
          % (dE_mlip.max() - dE_mlip[0], dE_dft.max() - dE_dft[0]))
    print("wrote %s.png" % args.out)


if __name__ == "__main__":
    main()

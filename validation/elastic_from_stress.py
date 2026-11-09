#!/usr/bin/env python
# =============================================================================
# elastic_from_stress.py -- fit the elastic tensor C_ij from the stresses of the
# finite-strain frames made by elastic_dft.py. The stress source is either DFT
# (a glob of QE .out) or a MatterSim model -- both on the SAME strained cells,
# so DFT vs generic vs tuned are directly comparable (clamped-ion).
#
#   # DFT:
#   python elastic_from_stress.py --frames defo.extxyz --qe_glob 'defo_dft/*.out' --out C_dft
#   # MLIP (generic / tuned):
#   python elastic_from_stress.py --frames defo.extxyz --model pretrained       --out C_gen
#   python elastic_from_stress.py --frames defo.extxyz --model best_model.pth    --out C_tuned
#
# Writes <out>_Cij.csv, <out>.log.lammps (compliance_python3.py-compatible), and
# prints VRH B/G/E and Born stability.
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

from ase import units
from ase.io import read

EVA3_TO_GPA = 1.0 / units.GPa
CIDX = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2),
        (3, 3), (4, 4), (5, 5), (0, 3), (0, 4), (0, 5),
        (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5),
        (3, 4), (3, 5), (4, 5)]
CNAME = ["C11", "C22", "C33", "C12", "C13", "C23", "C44", "C55", "C66",
         "C14", "C15", "C16", "C24", "C25", "C26", "C34", "C35", "C36",
         "C45", "C46", "C56"]


def natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def stresses_from_qe(frames, qe_glob):
    outs = sorted(glob.glob(qe_glob), key=natural_key)
    if len(outs) != len(frames):
        raise SystemExit("QE outputs (%d) != frames (%d)" % (len(outs), len(frames)))
    S = []
    for o in outs:
        at = read(o, format="espresso-out")
        S.append(at.get_stress(voigt=True) * EVA3_TO_GPA)
    return np.array(S)


def stresses_from_model(frames, model, device):
    from mattersim.forcefield import MatterSimCalculator
    calc = (MatterSimCalculator(device=device) if model.lower()
            in ("", "pretrained", "default", "none")
            else MatterSimCalculator(load_path=model, device=device))
    S = []
    for a in frames:
        b = a.copy(); b.set_pbc(True); b.calc = calc
        S.append(b.get_stress(voigt=True) * EVA3_TO_GPA)
    return np.array(S)


def main():
    ap = argparse.ArgumentParser(description="fit Cij from finite-strain stresses")
    ap.add_argument("--frames", required=True, help="defo.extxyz from elastic_dft.py")
    ap.add_argument("--qe_glob", default="", help="glob of QE .out (DFT stress source)")
    ap.add_argument("--model", default="", help="MatterSim checkpoint (MLIP stress source)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="C")
    args = ap.parse_args()

    frames = read(args.frames, index=":")
    voigt = np.array([f.info.get("voigt", -1) for f in frames])
    delta = np.array([f.info.get("delta", 0.0) for f in frames])

    if args.qe_glob:
        S = stresses_from_qe(frames, args.qe_glob)
    elif args.model:
        S = stresses_from_model(frames, args.model, args.device)
    else:
        raise SystemExit("give --qe_glob or --model")

    s_ref = S[voigt == -1][0]
    C = np.zeros((6, 6))
    for j in range(6):
        m = voigt == j
        eps = np.concatenate([[0.0], delta[m]])
        sig = np.vstack([s_ref, S[m]])
        for i in range(6):
            C[i, j] = np.polyfit(eps, sig[:, i], 1)[0]     # dsigma_i/deps_j [GPa]
    C = 0.5 * (C + C.T)

    np.savetxt(args.out + "_Cij.csv", C, delimiter=",",
               header=",".join("C%d" % (k + 1) for k in range(6)), comments="")
    with open(args.out + ".log.lammps", "w") as fh:
        fh.write("Components of the Elastic Constant Tensor\n")
        for name, (i, j) in zip(CNAME, CIDX):
            fh.write("Elastic Constant %sall = %.6f GPa\n" % (name, C[i, j]))

    BV = (C[0, 0] + C[1, 1] + C[2, 2] + 2 * (C[0, 1] + C[0, 2] + C[1, 2])) / 9.0
    GV = ((C[0, 0] + C[1, 1] + C[2, 2] - C[0, 1] - C[0, 2] - C[1, 2]) / 15.0
          + (C[3, 3] + C[4, 4] + C[5, 5]) / 5.0)
    try:
        Sc = np.linalg.inv(C)
        BR = 1.0 / (Sc[0, 0] + Sc[1, 1] + Sc[2, 2] + 2 * (Sc[0, 1] + Sc[0, 2] + Sc[1, 2]))
        GR = 15.0 / (4 * (Sc[0, 0] + Sc[1, 1] + Sc[2, 2])
                     - 4 * (Sc[0, 1] + Sc[0, 2] + Sc[1, 2])
                     + 3 * (Sc[3, 3] + Sc[4, 4] + Sc[5, 5]))
    except np.linalg.LinAlgError:
        BR = GR = float("nan")
    B, G = 0.5 * (BV + BR), 0.5 * (GV + GR)
    E = 9 * B * G / (3 * B + G)
    nu = (3 * B - 2 * G) / (2 * (3 * B + G))
    eig = np.linalg.eigvalsh(C)
    print("Cij [GPa]:")
    for i in range(6):
        print("  " + "  ".join("%9.2f" % C[i, j] for j in range(6)))
    print("VRH:  B=%.2f  G=%.2f  E=%.2f GPa  nu=%.3f   Born_stable=%s (min eig %.2f)"
          % (B, G, E, nu, bool(np.all(eig > 0)), eig.min()))
    print("wrote %s_Cij.csv, %s.log.lammps" % (args.out, args.out))


if __name__ == "__main__":
    main()

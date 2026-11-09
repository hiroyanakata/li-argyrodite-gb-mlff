#!/usr/bin/env python
# =============================================================================
# elastic_md.py -- elastic constant tensor Cij in pure ASE/Python (MatterSim)
#
# Python/MatterSim replacement for the LAMMPS `in.elastic` + init/potential/
# displace.mod static-deformation recipe (pair_style chgnet won't run under
# MatterSim).  Method is the same finite-strain scheme:
#
#   1. relax the cell to ~0 stress (aniso: normal components only) + ions
#   2. for each of the 6 Voigt strain components, apply +/- delta, relax the
#      ions at fixed (deformed) cell, read the stress tensor
#   3. C[i,j] = d(sigma_i)/d(eps_j)   (least-squares slope over +/-delta and 0)
#
# ASE stress convention is tensile-positive, so sigma_i = C_ij eps_j gives the
# standard stiffness constants directly.  Ion relaxation at each strain yields
# the *relaxed* (not clamped-ion) constants, matching the LAMMPS `minimize`.
#
# Writes:
#   <out>_Cij.csv          full symmetric 6x6 in GPa
#   <out>.log.lammps       21 "Elastic Constant Cxxall = <val> GPa" lines so the
#                          existing compliance_python3.py works unchanged
#   + prints VRH B/G/E/nu and a Born mechanical-stability check
#
#   python elastic_md.py --data output00.data --model ../matersim_model.pth \
#          --symbols "Li P S Br I Cl" --strain 0.005 --fmax 0.02 --out gb210
#   python compliance_python3.py   # (in a dir where the log is named log.lammps)
# =============================================================================
import argparse
import numpy as np

# ASE >= 3.24 moved full_3x3_to_voigt_6_stress; mattersim 1.2.0 imports the old
# location. Re-inject so this runs on newer ASE (no-op on older).
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
from ase.io.lammpsdata import read_lammps_data
from ase.io import read as ase_read
from ase.optimize import BFGS

EVA3_TO_GPA = 1.0 / units.GPa

# order of the 21 independent constants exactly as compliance_python3.py expects
CIDX = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2),
        (3, 3), (4, 4), (5, 5), (0, 3), (0, 4), (0, 5),
        (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5),
        (3, 4), (3, 5), (4, 5)]
CNAME = ["C11", "C22", "C33", "C12", "C13", "C23",
         "C44", "C55", "C66", "C14", "C15", "C16",
         "C24", "C25", "C26", "C34", "C35", "C36",
         "C45", "C46", "C56"]


def readcellInfo(inputdata):
    cellIn = np.zeros(3)
    with open(inputdata) as ip1:
        for line1 in ip1:
            aa = line1.split()
            if len(aa) > 2:
                if aa[2] == "xlo":
                    cellIn[0] = float(aa[1]) - float(aa[0])
                if aa[2] == "ylo":
                    cellIn[1] = float(aa[1]) - float(aa[0])
                if aa[2] == "zlo":
                    cellIn[2] = float(aa[1]) - float(aa[0])
    return cellIn


def lammps_types(path, atom_style):
    """LAMMPS integer atom types in ascending-atom-id order (== read_lammps_data
    order). Needed because read_lammps_data guesses elements from Masses, which
    silently mislabels S(28.085->Si) and Br(78.904->Se) in these data files, and
    for atom_style 'full' the type is NOT stored in atoms.get_tags()."""
    col = {"atomic": 1, "charge": 1, "full": 2, "molecular": 2}.get(atom_style, 1)
    ids, types = [], []
    with open(path) as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        if lines[i].split()[:1] == ["Atoms"]:
            i += 1
            while i < len(lines) and lines[i].strip() == "":
                i += 1
            while i < len(lines) and lines[i].strip() != "":
                p = lines[i].split()
                ids.append(int(p[0]))
                types.append(int(p[col]))
                i += 1
            break
        i += 1
    order = np.argsort(ids)
    return np.asarray(types)[order]


def load_atoms(args):
    if args.data.endswith((".data", ".lmp")) or args.lammps:
        atoms = read_lammps_data(args.data, atom_style=args.atom_style)
        cellIn = readcellInfo(args.data)
        if np.all(cellIn > 0):
            atoms.set_cell([cellIn[0], cellIn[1], cellIn[2]])
        atoms.set_pbc([True, True, True])
        if args.symbols:
            syms = args.symbols.split()
            types = lammps_types(args.data, args.atom_style)
            if len(types) == len(atoms):
                atoms.set_chemical_symbols([syms[t - 1] for t in types])
            else:
                raise ValueError("parsed %d types but %d atoms" % (len(types), len(atoms)))
    else:
        atoms = ase_read(args.data)
        atoms.set_pbc([True, True, True])
    return atoms


def cell_filter(atoms):
    """aniso cell relax: normal components only, tilts frozen (like box/relax aniso)."""
    mask = [1, 1, 1, 0, 0, 0]
    try:
        from ase.filters import FrechetCellFilter
        return FrechetCellFilter(atoms, mask=mask)
    except Exception:
        try:
            from ase.filters import ExpCellFilter
            return ExpCellFilter(atoms, mask=mask)
        except Exception:
            from ase.constraints import ExpCellFilter
            return ExpCellFilter(atoms, mask=mask)


def voigt_strain_matrix(j, d):
    """3x3 strain tensor for a unit perturbation of Voigt component j by d
    (engineering shear: Voigt gamma = 2*eps for j>=3)."""
    e = np.zeros((3, 3))
    if j < 3:
        e[j, j] = d
    else:
        a, b = {3: (1, 2), 4: (0, 2), 5: (0, 1)}[j]
        e[a, b] = e[b, a] = d / 2.0
    return e


def relax_ions(atoms, fmax, steps):
    if steps > 0:
        BFGS(atoms, logfile=None).run(fmax=fmax, steps=steps)


def stress_gpa(atoms):
    return atoms.get_stress(voigt=True) * EVA3_TO_GPA   # [xx,yy,zz,yz,xz,xy]


def main():
    ap = argparse.ArgumentParser(description="ASE/MatterSim elastic constants")
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="best_model.pth")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--atom_style", default="atomic")
    ap.add_argument("--lammps", action="store_true")
    ap.add_argument("--symbols", default="", help="'Li P S Br I Cl' to force type->element")
    ap.add_argument("--strain", type=float, nargs="+", default=[0.005],
                    help="strain magnitude(s); several -> least-squares slope")
    ap.add_argument("--fmax", type=float, default=0.02, help="ion relax force tol [eV/A]")
    ap.add_argument("--relax_steps", type=int, default=200, help="max ion-relax steps per strain")
    ap.add_argument("--init_relax", type=int, default=300, help="cell+ion relax steps (0=skip)")
    ap.add_argument("--out", default="elastic")
    args = ap.parse_args()

    atoms = load_atoms(args)
    from mattersim.forcefield import MatterSimCalculator
    if args.model.lower() in ("", "pretrained", "default", "none"):
        atoms.calc = MatterSimCalculator(device=args.device)     # built-in generic MatterSim
        print("using pretrained (generic) MatterSim force field")
    else:
        atoms.calc = MatterSimCalculator(load_path=args.model, device=args.device)
    print("composition:", atoms.get_chemical_formula(), " cell:", atoms.get_cell().lengths())

    # ---- 1. relax reference cell to ~0 stress (aniso) + ions ----
    if args.init_relax > 0:
        print("relaxing reference cell+ions (aniso) ...")
        BFGS(cell_filter(atoms), logfile=None).run(fmax=args.fmax, steps=args.init_relax)
    p0 = -stress_gpa(atoms)[:3].mean()
    print("reference pressure = %.4f GPa" % p0)
    ref_cell = np.array(atoms.get_cell())
    ref_pos = atoms.get_positions().copy()
    s_ref = stress_gpa(atoms)

    # ---- 2. finite strains ----
    strains = sorted(set([0.0] + [+d for d in args.strain] + [-d for d in args.strain]))
    C = np.zeros((6, 6))
    for j in range(6):
        pts = []   # (eps, sigma_vector[6])
        for d in strains:
            if d == 0.0:
                pts.append((0.0, s_ref))
                continue
            F = np.eye(3) + voigt_strain_matrix(j, d)
            atoms.set_cell(ref_cell @ F.T, scale_atoms=False)
            atoms.set_positions(ref_pos @ F.T)      # affine remap of ions
            relax_ions(atoms, args.fmax, args.relax_steps)
            pts.append((d, stress_gpa(atoms)))
        eps = np.array([p[0] for p in pts])
        sig = np.array([p[1] for p in pts])         # (npts, 6)
        for i in range(6):
            C[i, j] = np.polyfit(eps, sig[:, i], 1)[0]   # dsigma_i/deps_j [GPa]
        print("  Voigt strain %d done" % (j + 1))
        # restore reference for the next column
        atoms.set_cell(ref_cell, scale_atoms=False)
        atoms.set_positions(ref_pos)

    C = 0.5 * (C + C.T)                             # symmetrize
    np.savetxt(args.out + "_Cij.csv", C, delimiter=",",
               header=",".join("C%d" % (k + 1) for k in range(6)), comments="")

    # ---- 3. log.lammps compatible with compliance_python3.py ----
    with open(args.out + ".log.lammps", "w") as fh:
        # leading line so compliance_python3.py's '\nElastic Constant C' search
        # matches the very first constant too
        fh.write("Components of the Elastic Constant Tensor\n")
        for name, (i, j) in zip(CNAME, CIDX):
            fh.write("Elastic Constant %sall = %.6f GPa\n" % (name, C[i, j]))
    print("\nCij [GPa]:")
    for i in range(6):
        print("  " + "  ".join("%9.2f" % C[i, j] for j in range(6)))

    # ---- VRH moduli + Born stability ----
    BV = (C[0, 0] + C[1, 1] + C[2, 2] + 2 * (C[0, 1] + C[0, 2] + C[1, 2])) / 9.0
    GV = ((C[0, 0] + C[1, 1] + C[2, 2] - C[0, 1] - C[0, 2] - C[1, 2]) / 15.0
          + (C[3, 3] + C[4, 4] + C[5, 5]) / 5.0)
    try:
        S = np.linalg.inv(C)
        BR = 1.0 / (S[0, 0] + S[1, 1] + S[2, 2] + 2 * (S[0, 1] + S[0, 2] + S[1, 2]))
        GR = 15.0 / (4 * (S[0, 0] + S[1, 1] + S[2, 2])
                     - 4 * (S[0, 1] + S[0, 2] + S[1, 2])
                     + 3 * (S[3, 3] + S[4, 4] + S[5, 5]))
    except np.linalg.LinAlgError:
        BR = GR = float("nan")
    B, G = 0.5 * (BV + BR), 0.5 * (GV + GR)
    E = 9 * B * G / (3 * B + G)
    nu = (3 * B - 2 * G) / (2 * (3 * B + G))
    eig = np.linalg.eigvalsh(C)
    print("\n== Voigt-Reuss-Hill ==")
    print("  B = %7.2f GPa   G = %7.2f GPa   E = %7.2f GPa   nu = %.3f" % (B, G, E, nu))
    print("  Born stability (all C eigenvalues > 0): %s   (min eig = %.2f GPa)"
          % (bool(np.all(eig > 0)), eig.min()))
    print("wrote %s_Cij.csv and %s.log.lammps" % (args.out, args.out))
    print("  -> run compliance_python3.py on a copy named 'log.lammps' for hardness/VRH")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# =============================================================================
# ps4_probe.py -- PS4(3-) tetrahedron geometry and stiffness probe
#
# The parity breakdown says the fine-tuned FF regresses on the P sublattice
# (the atoms carrying the largest forces). This checks what that does to the
# thing PS4 actually controls: the geometry and the vibrational stiffness of the
# tetrahedron -- both of which have experimental reference values for
# Li6PS5X argyrodite:
#
#   P-S bond length   ~ 2.03 - 2.05 A          (XRD / neutron)
#   S-P-S angle       ~ 109.5 deg (ideal Td)
#   PS4(3-) internal modes (Raman, Li6PS5Cl):
#       nu2 (E,  bend)          ~ 190 - 200 cm^-1
#       nu4 (T2, bend)          ~ 265 - 275 cm^-1
#       nu1 (A1, sym. stretch)  ~ 420 - 427 cm^-1   <- the strong Raman line
#       nu3 (T2, asym. stretch) ~ 565 - 580 cm^-1
#
# For each PS4 unit it builds a PARTIAL Hessian over the 5 atoms (P + its four
# S) by central finite differences, mass-weights it, and assigns the 9 internal
# modes. nu1 is picked by PROJECTING each eigenvector on the symmetric-stretch
# coordinate (all four S moving radially out in phase), not by taking the
# highest mode -- the highest one is nu3. The partial Hessian ignores
# coupling to the Li sublattice, which is exactly the approximation under which
# the "PS4 internal mode" of the Raman assignment is defined.
#
#   python ps4_probe.py --data gb_dec.extxyz --models pretrained best_model.pth \
#          --relax_fmax 0.03 --out ps4
# =============================================================================
import argparse
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

from ase.io import read
from ase.optimize import FIRE
from ase.data import atomic_masses, atomic_numbers

# eV/A^2/amu -> cm^-1
EV_A2_AMU_TO_CM1 = 521.4708


def make_calc(model, device):
    from mattersim.forcefield import MatterSimCalculator
    if model in ("pretrained", "generic", "default"):
        return MatterSimCalculator(device=device)
    return MatterSimCalculator(load_path=model, device=device)


def mic(d, cell):
    inv = np.linalg.inv(cell)
    f = d @ inv
    f -= np.round(f)
    return f @ cell


def ps4_units(atoms, rcut=2.6):
    """each P with its four nearest S -> list of (p_index, [4 s indices])."""
    sym = np.array(atoms.get_chemical_symbols())
    cell = np.array(atoms.get_cell())
    pos = atoms.get_positions()
    P = np.where(sym == "P")[0]
    S = np.where(sym == "S")[0]
    out = []
    for p in P:
        d = np.linalg.norm(mic(pos[S] - pos[p], cell), axis=1)
        order = np.argsort(d)[:4]
        if d[order].max() < rcut:
            out.append((int(p), [int(S[k]) for k in order]))
    return out


AX = {"x": 0, "y": 1, "z": 2}


def unit_dgb(atoms, units, axis, planes):
    """distance from each PS4 centre (its P) to the nearest GB plane [A]."""
    ax = AX[axis]
    L = atoms.cell.lengths()[ax]
    x = atoms.get_positions()[[p for p, _ in units], ax]
    return np.min([np.abs((x - f * L + L / 2) % L - L / 2) for f in planes], axis=0)


def geometry(atoms, units):
    cell = np.array(atoms.get_cell())
    pos = atoms.get_positions()
    bonds, angles = [], []
    for p, ss in units:
        v = np.array([mic(pos[s] - pos[p], cell) for s in ss])
        r = np.linalg.norm(v, axis=1)
        bonds.extend(r.tolist())
        for a in range(4):
            for b in range(a + 1, 4):
                c = np.dot(v[a], v[b]) / (r[a] * r[b])
                angles.append(np.degrees(np.arccos(np.clip(c, -1, 1))))
    return np.array(bonds), np.array(angles)


def partial_hessian(atoms, idx, delta=0.01):
    """central-difference Hessian restricted to `idx` (3n x 3n), eV/A^2."""
    n = len(idx)
    H = np.zeros((3 * n, 3 * n))
    p0 = atoms.get_positions()
    for a, ia in enumerate(idx):
        for c in range(3):
            for sgn in (+1, -1):
                p = p0.copy()
                p[ia, c] += sgn * delta
                atoms.set_positions(p)
                f = atoms.get_forces()[idx].ravel()
                H[3 * a + c] += -sgn * f / (2 * delta)
    atoms.set_positions(p0)
    return 0.5 * (H + H.T)


def frequencies(H, masses):
    """mass-weighted eigenfrequencies [cm^-1] and the mass-weighted modes."""
    m = np.repeat(masses, 3)
    Hm = H / np.sqrt(np.outer(m, m))
    w2, V = np.linalg.eigh(Hm)
    # negative eigenvalue -> imaginary mode, reported as a negative wavenumber
    f = np.sign(w2) * np.sqrt(np.abs(w2)) * EV_A2_AMU_TO_CM1
    V = V / np.sqrt(m)[:, None]            # back to cartesian displacements
    return f, V


def assign_modes(f, V, bond_dirs):
    """assign the 9 internal PS4 modes. idx 0 = P, 1..4 = S.

    nu1 (A1) = the mode with the largest overlap on the symmetric stretch
    (every S moving out along its own P-S bond, in phase); nu3 = the three
    stiffest remaining modes; nu4 / nu2 = the rest, split by frequency."""
    order = np.argsort(f)
    f, V = f[order], V[:, order]
    internal = slice(6, 15)                # drop 6 translation/rotation modes
    fi, Vi = f[internal], V[:, internal]
    q = np.zeros(15)                       # symmetric-stretch coordinate
    for k in range(4):
        q[3 * (k + 1):3 * (k + 2)] = bond_dirs[k]
    q /= np.linalg.norm(q)
    ov = np.abs(q @ (Vi / np.linalg.norm(Vi, axis=0)))
    i1 = int(np.argmax(ov))
    rest = [k for k in range(len(fi)) if k != i1]
    rest.sort(key=lambda k: -fi[k])
    return dict(nu1=fi[i1], nu3=fi[rest[:3]], nu4=fi[rest[3:6]], nu2=fi[rest[6:]],
                all=fi, overlap=ov[i1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--models", nargs="+", default=["pretrained"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--relax_fmax", type=float, default=0.03)
    ap.add_argument("--relax_steps", type=int, default=600)
    ap.add_argument("--nunits", type=int, default=0,
                    help="only probe the first N PS4 units (0 = all)")
    ap.add_argument("--delta", type=float, default=0.01)
    ap.add_argument("--gb_axis", default="", choices=["", "x", "y", "z"],
                    help="also report the PS4 units split by distance to the GB plane")
    ap.add_argument("--gb_pos", default="0.25,0.75")
    ap.add_argument("--gb_bins", default="0,3,99", help="distance-to-GB bin edges [A]")
    ap.add_argument("--out", default="ps4")
    args = ap.parse_args()

    ref = read(args.data)
    ref.set_pbc(True)
    rows = []
    for model in args.models:
        tag = model.split("/")[-1]
        at = ref.copy()
        at.calc = make_calc(model, args.device)
        dyn = FIRE(at, logfile=None)
        dyn.run(fmax=args.relax_fmax, steps=args.relax_steps)
        fm = np.linalg.norm(at.get_forces(), axis=1).max()
        print("== %s ==  relax fmax=%.4f eV/A (%d steps)%s"
              % (tag, fm, dyn.get_number_of_steps(),
                 "" if fm <= args.relax_fmax else "  *** NOT CONVERGED ***"))

        units = ps4_units(at)
        b, ang = geometry(at, units)
        print("  PS4 units found: %d / %d P"
              % (len(units), (np.array(at.get_chemical_symbols()) == "P").sum()))
        print("  P-S bond   = %.3f +/- %.3f A     (exp ~2.03-2.05)" % (b.mean(), b.std()))
        print("  S-P-S angle= %.1f +/- %.1f deg   (ideal Td 109.5)" % (ang.mean(), ang.std()))

        # pick the probed units: with a GB axis, spread them evenly over d_gb so a
        # subsample still covers GB core -> grain interior (needed for the big
        # 69 A cell, where probing all ~106 tetrahedra is too expensive).
        if args.nunits <= 0 or args.nunits >= len(units):
            sel = units
        elif args.gb_axis:
            dg0 = unit_dgb(at, units, args.gb_axis,
                           [float(v) for v in args.gb_pos.split(",")])
            order = np.argsort(dg0)
            pick = np.unique(np.linspace(0, len(units) - 1, args.nunits).astype(int))
            sel = [units[order[k]] for k in pick]
        else:
            sel = units[:args.nunits]
        cell = np.array(at.get_cell())
        acc = {k: [] for k in ("nu1", "nu2", "nu3", "nu4")}
        nuall, ovs, unit_bond = [], [], []
        for p, ss in sel:
            idx = [p] + ss
            H = partial_hessian(at, idx, args.delta)
            m = np.array([atomic_masses[atomic_numbers[at[i].symbol]] for i in idx])
            f, V = frequencies(H, m)
            pos = at.get_positions()
            bd = [mic(pos[s] - pos[p], cell) for s in ss]
            bd = [v / np.linalg.norm(v) for v in bd]
            a = assign_modes(f, V, bd)
            for k in acc:
                acc[k].append(np.atleast_1d(a[k]).mean())
            nuall.append(a["all"]); ovs.append(a["overlap"])
            unit_bond.append(np.mean([np.linalg.norm(mic(pos[s] - pos[p], cell))
                                      for s in ss]))
        nuall = np.array(nuall)
        nimag = int((nuall < -1e-6).sum())
        for k, exp in (("nu2", "190-200"), ("nu4", "265-275"),
                       ("nu1", "420-427"), ("nu3", "565-580")):
            v = np.array(acc[k])
            print("  %s = %4.0f +/- %3.0f cm^-1   (Raman %s)%s"
                  % (k, v.mean(), v.std(), exp,
                     "   <- strong Raman line" if k == "nu1" else ""))
        print("  sym-stretch projection overlap = %.2f (1 = clean A1 assignment)"
              % np.mean(ovs))
        print("  PS4 internal modes: %.0f - %.0f cm^-1, imaginary modes: %d/%d"
              % (nuall.min(), nuall.max(), nimag, nuall.size))
        # GB vs grain-interior split: a distorted GB tetrahedron is SUPPOSED to
        # differ from the crystalline one, so a cell average hides the question
        # of whether the model is wrong where it matters.
        if args.gb_axis:
            planes = [float(v) for v in args.gb_pos.split(",")]
            dg = unit_dgb(at, sel, args.gb_axis, planes)
            edges = [float(v) for v in args.gb_bins.split(",")]
            ub = np.array(unit_bond)
            print("    -- PS4 units by distance to GB plane (%s) --" % args.gb_axis)
            for lo, hi in zip(edges[:-1], edges[1:]):
                m = (dg >= lo) & (dg < hi)
                if not m.sum():
                    continue
                print("    d_gb=[%.1f,%.1f) n=%2d  P-S=%.3f A  nu1=%4.0f  nu3=%4.0f  nu4=%4.0f"
                      % (lo, hi, m.sum(), ub[m].mean(),
                         np.array(acc["nu1"])[m].mean(),
                         np.array(acc["nu3"])[m].mean(),
                         np.array(acc["nu4"])[m].mean()))
        nu1 = np.array(acc["nu1"])
        rows.append((tag, b.mean(), b.std(), ang.mean(), ang.std(),
                     nu1.mean(), nu1.std(), nimag,
                     np.mean(acc["nu2"]), np.mean(acc["nu4"]), np.mean(acc["nu3"])))

    import csv
    with open(args.out + ".csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "PS_bond_A", "PS_bond_sd", "SPS_angle_deg",
                    "SPS_angle_sd", "nu1_cm-1", "nu1_sd", "n_imag",
                    "nu2_cm-1", "nu4_cm-1", "nu3_cm-1"])
        for r in rows:
            w.writerow([r[0]] + ["%.4f" % v for v in r[1:7]] + [r[7]]
                       + ["%.1f" % v for v in r[8:11]])
    print("\nwrote %s.csv" % args.out)
    print("%-16s %11s %12s %8s %8s %8s %8s" %
          ("model", "P-S [A]", "S-P-S [deg]", "nu2", "nu4", "nu1", "nu3"))
    for r in rows:
        print("%-16s %6.3f+-%.3f %6.1f+-%4.1f %8.0f %8.0f %8.0f %8.0f" %
              (r[0], r[1], r[2], r[3], r[4], r[8], r[9], r[5], r[10]))
    print("%-16s %11s %12s %8s %8s %8s %8s" %
          ("experiment", "2.03-2.05", "109.5", "~195", "~270", "~424", "~572"))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# =============================================================================
# parity.py -- energy/force/stress parity of MatterSim vs DFT (QE) on the same
#              geometries. THE core validation for the GB force field: shows
#              where a generic potential fails and the fine-tuned one recovers.
#
# For each QE single-point output (pw.x, run with tprnfor=.true.), read the
# DFT energy/forces/stress via ASE's espresso-out reader (so units and sign
# convention match MatterSim exactly), evaluate one or more MatterSim models on
# the SAME atoms, and report:
#   * force  RMSE / MAE           [eV/A]   (reference-free -> the cleanest metric)
#   * stress RMSE                 [GPa]
#   * relative-energy RMSE        [meV/atom] (mean-subtracted per model set)
# split by an optional per-frame label (e.g. GB / bulk / thermal / rattle).
#
#   python parity.py --qe_glob 'neb_dft/*.out' \
#          --models pretrained /path/tuned.pth --labels frames.extxyz --out parity
#
# --labels reads info['set'] from an extxyz whose frames are in the SAME order
# as the sorted QE outputs (e.g. the frames.extxyz that make_scf_inputs consumed).
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EVA3_TO_GPA = 1.0 / units.GPa


def natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def read_dft(outfile):
    """DFT atoms with energy/forces/stress in ASE units (eV, eV/A, eV/A^3).
    Returns None if the SCF did not finish / has no forces."""
    try:
        at = read(outfile, format="espresso-out")   # last converged image
        e = at.get_potential_energy()
        f = at.get_forces()
        try:
            s = at.get_stress(voigt=True)
        except Exception:
            s = None
        return at, e, f, s
    except Exception as ex:
        print("  skip %s (%s)" % (outfile, ex))
        return None


AX = {"x": 0, "y": 1, "z": 2}


def dist_to_gb(atoms, axis, planes):
    """per-atom minimum-image distance to the nearest GB plane [A]."""
    ax = AX[axis]
    L = atoms.cell.lengths()[ax]
    x = atoms.get_positions()[:, ax]
    return np.min([np.abs((x - f * L + L / 2) % L - L / 2) for f in planes], axis=0)


def make_calc(model, device):
    from mattersim.forcefield import MatterSimCalculator
    if model.lower() in ("", "pretrained", "default", "none"):
        return MatterSimCalculator(device=device)
    return MatterSimCalculator(load_path=model, device=device)


def main():
    ap = argparse.ArgumentParser(description="MatterSim vs DFT(QE) parity")
    ap.add_argument("--qe_glob", required=True, help="glob of QE .out single-points")
    ap.add_argument("--models", nargs="+", default=["pretrained"],
                    help="one or more checkpoints (or 'pretrained'); each is a series")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--gb_axis", default="", choices=["", "x", "y", "z"],
                    help="decompose the force error by distance to the GB plane")
    ap.add_argument("--gb_pos", default="0.25,0.75",
                    help="GB plane positions as fractions of the cell along --gb_axis")
    ap.add_argument("--gb_bins", default="0,2,4,99",
                    help="distance-to-GB bin edges [A]")
    ap.add_argument("--labels", default="", help="extxyz with info['set'] per frame (same order)")
    ap.add_argument("--out", default="parity")
    args = ap.parse_args()

    outs = sorted(glob.glob(args.qe_glob), key=natural_key)
    if not outs:
        raise SystemExit("no QE outputs matched %s" % args.qe_glob)
    print("QE outputs:", len(outs))

    labels = None
    if args.labels:
        frames = read(args.labels, index=":")
        labels = [f.info.get("set", "all") for f in frames]

    # ---- read DFT references once ----
    dft = []
    keep_lab = []
    for k, o in enumerate(outs):
        r = read_dft(o)
        if r is None:
            continue
        dft.append(r)
        keep_lab.append(labels[k] if labels and k < len(labels) else "all")
    print("valid DFT points:", len(dft))
    planes = [float(v) for v in args.gb_pos.split(",")]
    nat = np.array([len(r[0]) for r in dft])

    # ---- per model: evaluate MatterSim on the same atoms ----
    rows = []
    fig_f, ax_f = plt.subplots(figsize=(5, 5))
    fig_s, ax_s = plt.subplots(figsize=(5, 5))
    fig_e, ax_e = plt.subplots(figsize=(5, 5))
    colors = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd"]
    summary = {}
    for mi, model in enumerate(args.models):
        calc = make_calc(model, args.device)
        eD, eM, fD, fM, sD, sM, labs = [], [], [], [], [], [], []
        symb = []
        dgb = []
        for (at, e, f, s), lab in zip(dft, keep_lab):
            symb.append(np.repeat(np.array(at.get_chemical_symbols()), 3))
            if args.gb_axis:
                dgb.append(np.repeat(dist_to_gb(at, args.gb_axis, planes), 3))
            a2 = at.copy(); a2.calc = calc
            eM.append(a2.get_potential_energy()); eD.append(e)
            fM.append(a2.get_forces().ravel()); fD.append(f.ravel())
            if s is not None:
                try:
                    sM.append(a2.get_stress(voigt=True) * EVA3_TO_GPA)
                    sD.append(s * EVA3_TO_GPA)
                except Exception:
                    pass
            labs.append(lab)
        eD = np.array(eD); eM = np.array(eM); labs = np.array(labs)
        # reference-free relative energies (per atom)
        reD = (eD - eD.mean()) / nat * 1000.0     # meV/atom
        reM = (eM - eM.mean()) / nat * 1000.0
        fD_all = np.concatenate(fD); fM_all = np.concatenate(fM)
        f_rmse = float(np.sqrt(np.mean((fD_all - fM_all) ** 2)))
        f_mae = float(np.mean(np.abs(fD_all - fM_all)))
        e_rmse = float(np.sqrt(np.mean((reD - reM) ** 2)))
        s_rmse = float("nan")
        if sD:
            sD_all = np.concatenate(sD); sM_all = np.concatenate(sM)
            s_rmse = float(np.sqrt(np.mean((sD_all - sM_all) ** 2)))
        tag = model.split("/")[-1]
        summary[tag] = dict(E_meV_atom=e_rmse, F_eV_A=f_rmse, F_MAE=f_mae, S_GPa=s_rmse)
        rows.append((tag, e_rmse, f_rmse, f_mae, s_rmse))
        c = colors[mi % len(colors)]
        ax_f.scatter(fD_all, fM_all, s=3, alpha=0.3, color=c,
                     label="%s (RMSE %.3f)" % (tag, f_rmse))
        ax_e.scatter(reD, reM, s=18, alpha=0.7, color=c,
                     label="%s (RMSE %.1f)" % (tag, e_rmse))
        if sD:
            ax_s.scatter(np.concatenate(sD), np.concatenate(sM), s=10, alpha=0.5,
                         color=c, label="%s (RMSE %.2f)" % (tag, s_rmse))
        # per-label breakdown
        for lb in sorted(set(labs)):
            m = labs == lb
            fr = np.concatenate([fD[i] for i in range(len(fD)) if m[i]])
            fmm = np.concatenate([fM[i] for i in range(len(fM)) if m[i]])
            print("  [%s / %-8s] n=%d  F_RMSE=%.3f eV/A" %
                  (tag, lb, m.sum(), np.sqrt(np.mean((fr - fmm) ** 2))))
        # per-element breakdown: which sublattice does the model get wrong?
        sy_all = np.concatenate(symb)
        for el in sorted(set(sy_all.tolist())):
            m = sy_all == el
            print("    <%s / %-2s> n=%6d  F_RMSE=%.3f eV/A  (|F_DFT| mean %.3f)"
                  % (tag, el, m.sum() // 3,
                     np.sqrt(np.mean((fD_all[m] - fM_all[m]) ** 2)),
                     np.abs(fD_all[m]).mean()))
        # force-magnitude bins: equilibrium vs strongly displaced atoms
        fmag = np.linalg.norm(fD_all.reshape(-1, 3), axis=1)
        err = np.linalg.norm((fD_all - fM_all).reshape(-1, 3), axis=1)
        edges = [0.0, 0.3, 0.7, 1.5, 1e9]
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (fmag >= lo) & (fmag < hi)
            if m.sum():
                print("    |F|=[%.1f,%s) n=%6d  err_RMS=%.3f eV/A"
                      % (lo, "inf" if hi > 1e8 else "%.1f" % hi, m.sum(),
                         np.sqrt(np.mean(err[m] ** 2))))
        # GB-vs-crystal decomposition: is the model better WHERE THE GB IS?
        if args.gb_axis:
            dg_all = np.concatenate(dgb)
            edges = [float(v) for v in args.gb_bins.split(",")]
            print("    -- distance to GB plane (%s) --" % args.gb_axis)
            for lo, hi in zip(edges[:-1], edges[1:]):
                m = (dg_all >= lo) & (dg_all < hi)
                if not m.sum():
                    continue
                print("    d_gb=[%.1f,%.1f) n=%6d  F_RMSE=%.3f eV/A  (|F_DFT| mean %.3f)"
                      % (lo, hi, m.sum() // 3,
                         np.sqrt(np.mean((fD_all[m] - fM_all[m]) ** 2)),
                         np.abs(fD_all[m]).mean()))
                for el in ("P", "S", "Li"):
                    me = m & (sy_all == el)
                    if me.sum():
                        print("        %-2s n=%5d  F_RMSE=%.3f" %
                              (el, me.sum() // 3,
                               np.sqrt(np.mean((fD_all[me] - fM_all[me]) ** 2))))
            summary[tag]["by_dgb"] = {
                "%.1f-%.1f" % (lo, hi): float(np.sqrt(np.mean(
                    (fD_all[(dg_all >= lo) & (dg_all < hi)]
                     - fM_all[(dg_all >= lo) & (dg_all < hi)]) ** 2)))
                for lo, hi in zip(edges[:-1], edges[1:])
                if ((dg_all >= lo) & (dg_all < hi)).sum()}

        summary[tag]["per_element"] = {
            el: float(np.sqrt(np.mean((fD_all[sy_all == el] - fM_all[sy_all == el]) ** 2)))
            for el in sorted(set(sy_all.tolist()))}

    for ax, lab, unit in [(ax_f, "force", "eV/A"), (ax_s, "stress", "GPa"),
                          (ax_e, "rel. energy", "meV/atom")]:
        lo, hi = ax.get_xlim()
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.7)
        ax.set_xlabel("DFT %s [%s]" % (lab, unit))
        ax.set_ylabel("MatterSim %s [%s]" % (lab, unit))
        ax.legend(fontsize=8)
    fig_f.tight_layout(); fig_f.savefig(args.out + "_F.png", dpi=150)
    fig_s.tight_layout(); fig_s.savefig(args.out + "_S.png", dpi=150)
    fig_e.tight_layout(); fig_e.savefig(args.out + "_E.png", dpi=150)

    import csv
    with open(args.out + ".csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "E_RMSE_meV_atom", "F_RMSE_eV_A", "F_MAE_eV_A", "S_RMSE_GPa"])
        w.writerows(rows)
    print("=" * 60)
    print("%-24s %12s %12s %10s" % ("model", "E[meV/at]", "F[eV/A]", "S[GPa]"))
    for tag, er, fr, fm, sr in rows:
        print("%-24s %12.2f %12.3f %10.2f" % (tag, er, fr, sr))
    print("wrote %s_{F,S,E}.png and %s.csv" % (args.out, args.out))


if __name__ == "__main__":
    main()

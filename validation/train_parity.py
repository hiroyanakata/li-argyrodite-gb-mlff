#!/usr/bin/env python
# =============================================================================
# train_parity.py -- force/energy accuracy of MatterSim checkpoints against the
#                    DFT-labelled DeepMD training data, broken down by batch,
#                    force regime, and element. Built to TRACK fine-tuning:
#                    the generic-vs-tuned gap should open up on the high-force /
#                    P-S dissociation batches as training converges there.
#
# Reads DeepMD dirs (type.raw + set.*/{coord,box,force,energy}.npy) and, for each
# model, reports:
#   * force RMSE per batch, per |F| bin, and per element (P/S = the reactive bond)
#   * relative-energy RMSE [meV/atom]   (mean-subtracted per batch/model)
#
#   python train_parity.py --data '../data/Li094*.0*' \
#          --models mattersim-1M.pth best_model.pth \
#          --type_map "Li P S Br I Cl" --nsample 30 --out train_parity
#
# Add a new checkpoint to --models and re-run to see whether more epochs closed
# the reactive-regime gap.
# =============================================================================
import argparse
import glob
import os
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

from ase import Atoms

FBINS = [(0, 2), (2, 5), (5, 100)]        # equilibrium / moderate / reactive


def load_batch(d, type_map):
    """Accepts a DeepMD dir (type.raw + set.*/) OR an extxyz file (with forces)."""
    if os.path.isdir(d):
        typ = np.loadtxt(os.path.join(d, "type.raw"), dtype=int)
        sym = [type_map[t] for t in typ]
        C = np.concatenate([np.load(f) for f in sorted(glob.glob(d + "/set.*/coord.npy"))])
        B = np.concatenate([np.load(f) for f in sorted(glob.glob(d + "/set.*/box.npy"))])
        F = np.concatenate([np.load(f) for f in sorted(glob.glob(d + "/set.*/force.npy"))])
        E = np.concatenate([np.load(f) for f in sorted(glob.glob(d + "/set.*/energy.npy"))])
        return sym, C, B, F, E
    from ase.io import read
    frames = read(d, index=":")
    sym = frames[0].get_chemical_symbols()

    def getF(f):
        if "forces" in f.arrays:
            return f.arrays["forces"]
        return f.get_forces()          # stored in the SinglePointCalculator

    def getE(f):
        if "energy" in f.info:
            return f.info["energy"]
        try:
            return f.get_potential_energy()
        except Exception:
            return np.nan
    C = np.array([f.get_positions().ravel() for f in frames])
    B = np.array([np.asarray(f.get_cell()).ravel() for f in frames])
    F = np.array([getF(f).ravel() for f in frames])
    E = np.array([getE(f) for f in frames])
    return sym, C, B, F, E


def make_calc(model, device):
    from mattersim.forcefield import MatterSimCalculator
    if model.lower() in ("", "pretrained", "default", "none"):
        return MatterSimCalculator(device=device)
    return MatterSimCalculator(load_path=model, device=device)


def main():
    ap = argparse.ArgumentParser(description="MatterSim vs DFT parity on the training data")
    ap.add_argument("--data", required=True, nargs="+", help="glob(s) of DeepMD batch dirs")
    ap.add_argument("--models", nargs="+", required=True, help="checkpoints (or 'pretrained')")
    ap.add_argument("--type_map", default="Li P S Br I Cl")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--nsample", type=int, default=30, help="frames sampled per batch")
    ap.add_argument("--elements", default="P S", help="per-element force breakdown")
    ap.add_argument("--out", default="train_parity")
    args = ap.parse_args()

    tmap = args.type_map.split()
    batches = sorted({d for pat in args.data for d in glob.glob(pat)
                      if os.path.isdir(d) or d.endswith((".extxyz", ".xyz"))})
    if not batches:
        raise SystemExit("no batch dirs matched %s" % args.data)
    elems = args.elements.split()
    calcs = {m: make_calc(m, args.device) for m in args.models}
    tags = [os.path.basename(m).replace(".pth", "") for m in args.models]

    import csv
    fh = open(args.out + ".csv", "w", newline="")
    w = csv.writer(fh)
    w.writerow(["batch", "model", "n", "F_RMSE_all"]
               + ["F_RMSE_%d-%d" % (a, b) for a, b in FBINS]
               + ["F_RMSE_%s" % e for e in elems] + ["E_RMSE_meV_atom"])

    print("model tags:", dict(zip(args.models, tags)))
    for d in batches:
        bname = os.path.basename(d)
        sym, C, B, F, E = load_batch(d, tmap)
        sym = np.array(sym)
        nat = len(sym)
        n = len(C)
        idx = np.linspace(0, n - 1, min(args.nsample, n)).astype(int)
        fmax = np.array([np.linalg.norm(F[i].reshape(-1, 3), axis=1).max() for i in idx])
        print("\n== %s  (n=%d, fmax p50=%.2f) ==" % (bname, len(idx), np.median(fmax)))
        for m, tag in zip(args.models, tags):
            calc = calcs[m]
            fe, ee = [], []
            for i in idx:
                a = Atoms(symbols=sym, positions=C[i].reshape(-1, 3),
                          cell=B[i].reshape(3, 3), pbc=True)
                a.calc = calc
                fe.append(a.get_forces()); ee.append(a.get_potential_energy())
            fe = np.array(fe)                       # (nsamp, nat, 3)
            fd = F[idx].reshape(len(idx), nat, 3)
            err = fe - fd
            rmse_all = np.sqrt(np.mean(err ** 2))
            binr = []
            for lo, hi in FBINS:
                m_ = (fmax >= lo) & (fmax < hi)
                binr.append(np.sqrt(np.mean(err[m_] ** 2)) if m_.any() else np.nan)
            elr = [np.sqrt(np.mean(err[:, sym == e, :] ** 2)) if (sym == e).any() else np.nan
                   for e in elems]
            ee = np.array(ee); ed = E[idx]
            re = ((ed - ed.mean()) - (ee - ee.mean())) / nat * 1000.0
            e_rmse = np.sqrt(np.mean(re ** 2))
            print("  %-14s F_all=%.3f  bins[%s]  %s  E=%.1f meV/at"
                  % (tag, rmse_all,
                     " ".join("%.3f" % x for x in binr),
                     " ".join("%s=%.3f" % (e, v) for e, v in zip(elems, elr)), e_rmse))
            w.writerow([bname, tag, len(idx), "%.4f" % rmse_all]
                       + ["%.4f" % x for x in binr] + ["%.4f" % x for x in elr]
                       + ["%.2f" % e_rmse])
    fh.close()
    print("\nwrote %s.csv" % args.out)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# =============================================================================
# eval_v5C_gate.py -- v5C acceptance gate (run in bigGB_v2/)
#
#  (A) cross-energy: dE = E(broken geom) - E(pre-relaxed intact geom) for the
#      same cut, per model, against the DFT single points in dftsp_relax/in/.
#      The fine-tuned models (v5B/phaseB) collapse this to ~0 or negative while
#      DFT/pretrained give +10..+22 eV; v5C must recover the DFT sign+scale.
#  (B) pointwise E/F accuracy on the held-out AL DFT frames (gauge-free:
#      forces directly, energies after removing each model's mean offset).
#
#   python eval_v5C_gate.py --models v5C.pth,v5B.pth,phaseB.pth,pretrained
# =============================================================================
import argparse, glob, os, re, sys

import numpy as np

sys.path.insert(0, ".")
from relax_md_cut import make_calc          # noqa: E402  (ase shims live there)
from ase.io import read                     # noqa: E402

RY2EV = 13.6058
CUTS = ("n8_gb1", "n8_gb2", "uni_gb1")
GEOMS = ("pre", "v5B", "phaseB")


def dft_total(cut, geom):
    """Total energy [eV] of the dftsp_relax single point, or None."""
    tag = "sp" + cut.replace("_", "") + geom           # spn8gb1v5B
    pats = ("in/SP.%s.0.out" % tag, "in/SP.%s*.out" % tag, "in/*%s*.out" % tag)
    for pat in pats:
        for p in sorted(glob.glob(pat)):
            if "failed" in p:
                continue
            txt = open(p, errors="replace").read()
            if "JOB DONE" not in txt or "convergence has been achieved" not in txt:
                continue
            m = re.findall(r"^!\s+total energy\s+=\s+(-?[\d.]+) Ry", txt, re.M)
            if m:
                return float(m[-1]) * RY2EV
    return None


def part_a(models, device):
    calcs = {k: make_calc(v, device) for k, v in models.items()}
    E = {}                                              # (cut, geom) -> {model: E}
    for cut in CUTS:
        for geom in GEOMS:
            f = "dftsp_relax/sp_%s_%s.extxyz" % (cut, geom)
            if not os.path.exists(f):
                continue
            a = read(f)
            row = {}
            for k, c in calcs.items():
                b = a.copy(); b.calc = c
                row[k] = b.get_potential_energy()
            row["DFT"] = dft_total(cut, geom)
            E[(cut, geom)] = row
            print("  read %-22s nat=%d" % (f, len(a)), flush=True)

    cols = ["DFT"] + list(models)
    print("\n=== (A) dE = E(geom) - E(pre geom)  [eV]   (DFT: + means broken is "
          "higher/unstable) ===")
    print("%-9s %-7s" % ("cut", "geom") + "".join("%12s" % c for c in cols))
    out = []
    for cut in CUTS:
        ref = E.get((cut, "pre"))
        if ref is None:
            continue
        for geom in GEOMS:
            row = E.get((cut, geom))
            if row is None:
                continue
            cells = []
            for c in cols:
                if row.get(c) is None or ref.get(c) is None:
                    cells.append("%12s" % "-")
                else:
                    cells.append("%+12.2f" % (row[c] - ref[c]))
            print("%-9s %-7s" % (cut, geom) + "".join(cells))
            out.append((cut, geom, {c: (None if row.get(c) is None or ref.get(c) is None
                                        else row[c] - ref[c]) for c in cols}))
    with open("dftsp_relax/gate_cross_energy.csv", "w") as fo:
        fo.write("cut,geom," + ",".join("dE_" + c for c in cols) + "\n")
        for cut, geom, d in out:
            fo.write("%s,%s," % (cut, geom) +
                     ",".join("" if d[c] is None else "%.4f" % d[c] for c in cols) + "\n")
    print("-> dftsp_relax/gate_cross_energy.csv")
    return calcs


def part_b(calcs, data):
    frames = read(data, ":")
    print("\n=== (B) pointwise vs DFT on %s (%d frames) ===" % (data, len(frames)))
    edft = np.array([a.get_potential_energy() / len(a) for a in frames])
    fdft = [a.get_forces() for a in frames]
    print("%-9s %10s %10s %10s %10s" % ("model", "dE/at MAE", "F MAE", "F RMSE", "F max_err"))
    rows = []
    for k, c in calcs.items():
        em, fm = [], []
        for a in frames:
            b = a.copy(); b.calc = c
            em.append(b.get_potential_energy() / len(b))
            fm.append(b.get_forces())
        em = np.array(em)
        # gauge-free: remove the constant offset between model and DFT
        de = (em - edft) - np.mean(em - edft)
        df = np.concatenate([(x - y).ravel() for x, y in zip(fm, fdft)])
        rows.append((k, np.mean(np.abs(de)) * 1000, np.mean(np.abs(df)),
                     np.sqrt(np.mean(df ** 2)), np.max(np.abs(df))))
        print("%-9s %9.1f* %10.4f %10.4f %10.4f" % rows[-1])
    print("  (*) dE/at MAE in meV/atom after removing the mean offset")
    with open("dftsp_relax/gate_al_test.csv", "w") as fo:
        fo.write("model,dE_per_atom_meV,F_MAE_eVA,F_RMSE_eVA,F_maxerr_eVA\n")
        for r in rows:
            fo.write("%s,%.3f,%.5f,%.5f,%.5f\n" % r)
    print("-> dftsp_relax/gate_al_test.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="v5C.pth,v5B.pth,phaseB.pth,pretrained")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--al_test", default="al/v5C_al_test.extxyz")
    ap.add_argument("--skip_b", action="store_true")
    args = ap.parse_args()
    models = {}
    for m in args.models.split(","):
        models[os.path.basename(m).replace(".pth", "")] = m
    calcs = part_a(models, args.device)
    if not args.skip_b and os.path.exists(args.al_test):
        part_b(calcs, args.al_test)

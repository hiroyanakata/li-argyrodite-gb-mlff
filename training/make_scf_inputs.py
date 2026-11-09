#!/usr/bin/env python
"""Write single-point (scf) pw.x inputs for MD frames selected by screen_candidates.py.

The calculation conditions are copied verbatim from
``newCandidate/Li094P016S078Br009I000Cl009.npt005.40.pwin`` (the relax inputs of the
previous batch): same cutoffs, same smearing, same DFT-D3, same pseudopotentials,
same k-point mesh, same convergence thresholds.  Only the parts that make it a
geometry optimisation are changed:

  calculation = "relax"  ->  "scf"
  nstep / forc_conv_thr / &IONS / &CELL   -> dropped (meaningless for scf)
  tprnfor = .true.                        -> ADDED

``tprnfor`` is not optional: a relax run prints forces because it needs them, a bare
scf run does not print them at all, and a training set without forces is useless.
``tstress`` (stress tensor / virial) is off by default because the DeepMD sets in
``data/`` carry only coord/box/energy/force -- turn it on with --stress if wanted.

Usage
-----
    # 優先1,078件をそのまま入力化
    python make_scf_inputs.py --traj 'mdrun2/runs/*/output/out.traj' \
        --select mdrun2_screen_priority.txt --out newCandidate_scf

    # 予算に合わせて上位300件だけ（--select の順序＝FPS順を保つ）
    python make_scf_inputs.py --traj 'mdrun2/runs/*/output/out.traj' \
        --select mdrun2_screen_priority.txt --limit 300 --out newCandidate_scf

    # 選択リストなしで全フレーム（品質カットは効く）
    python make_scf_inputs.py --traj 'mdrun2/runs/*/output/out.traj' --out newCandidate_scf

Quality cuts (both default ON, see --min_dist / --fmax_max):
  * shortest interatomic distance < 1.75 A  -> skip.  The equilibrium P-S bond is
    ~2.05 A; below ~1.75 A the frames come from collapsing runs and the SCF is
    both hard to converge and not worth converging.
  * |F|max > 15 eV/A (from the MatterSim forces stored in the trajectory) -> skip.
Every skipped frame is recorded in the manifest with the reason.
"""
import argparse
import math
import csv
import glob
import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# calculation conditions -- taken from newCandidate/*.npt005.40.pwin
# ---------------------------------------------------------------------------
CONTROL = [
    ("calculation", '"scf"'),
    # A single-point scf run only has to hand back E/F/sigma; without disk_io
    # QE writes the full charge density and wavefunctions for every frame --
    # measured at 6.5 GB per 206-atom frame, i.e. 24 TB for a 3700-frame batch,
    # which both wastes the filesystem and makes concurrent jobs fail in
    # write_rhog. 'none' changes no number, only what hits the disk.
    ("disk_io", ' "none"'),
    ("max_seconds", " 1.72800e+05"),
    ("tprnfor", ".true."),          # scf does not print forces without this
]
ELECTRONS = [
    ("conv_thr", "  1.00000e-06"),
    ("electron_maxstep", " 200"),
    ("mixing_beta", "  5.00000e-02"),
    ("startingpot", ' "atomic"'),
    ("startingwfc", ' "atomic+random"'),
]
KPOINTS = " 1  1  1  0 0 0"        # the store's historical setting (Gamma only)


def kmesh(cell, klen):
    """Monkhorst-Pack mesh so that mesh_i * |a_i| >= klen along every axis.

    The existing store was all computed at Gamma on ~18-19.6 A cells, i.e. an
    effective 2x2x1 sampling of the 9.86 A argyrodite cube. A NEW 52-atom cube
    at Gamma is NOT the same calculation: measured on one bulk frame, Gamma vs
    2x2x2 moves the forces by 0.235 eV/A RMS -- larger than the model error the
    fine-tune is trying to remove. Tying the mesh to a target length keeps every
    cell, large or small, on the same k-point footing.
    klen <= 0 reproduces the historical Gamma-only behaviour.
    """
    import numpy as _np
    if klen <= 0:
        return (1, 1, 1)
    L = _np.linalg.norm(_np.asarray(cell, dtype=float), axis=1)
    return tuple(max(1, int(math.ceil(klen / max(l, 1e-6)))) for l in L)

# species block, in the canonical order used by type.raw everywhere in this project.
# The masses are the ones the previous inputs carried (they come from the LAMMPS
# data file, not from ASE's table) -- keep them so the two batches are identical.
SPECIES = [
    ("Li", 6.96800, "li_pbe_v1.4.uspp.F.UPF"),
    ("P", 30.97400, "P.pbe-n-rrkjus_psl.1.0.0.UPF"),
    ("S", 32.06500, "s_pbe_v1.4.uspp.F.UPF"),
    ("Br", 79.90400, "br_pbe_v1.4.uspp.F.UPF"),
    ("I", 126.90000, "I.pbe-n-kjpaw_psl.0.2.UPF"),
    ("Cl", 35.45300, "cl_pbe_v1.4.uspp.F.UPF"),
]
SYSTEM_NAME = "Li094P016S078Br009I000Cl009"


def min_pair_distance(pos, cell):
    """Shortest interatomic distance under the minimum image convention [A]."""
    inv = np.linalg.inv(cell)
    d = pos[:, None, :] - pos[None, :, :]
    f = d @ inv
    f -= np.round(f)
    d = np.linalg.norm(f @ cell, axis=-1)
    np.fill_diagonal(d, np.inf)
    return float(d.min())


def run_tag(path):
    """Trajectory tag: the run directory name for */output/out.traj, else the filename.

    mdrun2 writes every run to <run>/output/out.traj, so the file name alone would
    label all twelve runs "out" -- the same trap screen_candidates.py has.
    """
    base = os.path.basename(path)
    if base in ("out.traj", "out.extxyz"):
        return os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(path))))
    return os.path.splitext(base)[0]


def safe_tag(tag):
    """Strip everything but [A-Za-z0-9] so the name survives the .<run>.<step>.pwin split."""
    return "".join(c for c in tag if c.isalnum())


def write_pwin(path, symbols, positions, cell, prefix, pseudo_dir, stress,
               klen=0.0, mixing_beta=None):
    """One pw.x scf input, laid out like the previous batch's files."""
    control = list(CONTROL) + [("prefix", ' "%s"' % prefix),
                               ("pseudo_dir", ' "%s"' % pseudo_dir)]
    if stress:
        control.insert(3, ("tstress", ".true."))
    # mixing_beta only changes the SCF convergence PATH, never the converged
    # E/F/stress, so it is safe to raise it for speed. The store used 0.05; the
    # dense-mesh trainset converges far faster at 0.3 (measured).
    electrons = ([(k, "  %.5e" % mixing_beta) if k == "mixing_beta" else (k, v)
                  for k, v in ELECTRONS] if mixing_beta is not None
                 else list(ELECTRONS))

    def block(name, items):
        w = max(len(k) for k, _ in items)
        out = ["&%s" % name]
        for k, v in items:
            out.append("    %-*s = %s" % (w, k, v))
        out.append("/")
        return out

    L = []
    L += block("CONTROL", control) + [""]
    # nat/ntyp go where they were in the reference file: after ibrav, before tot_charge.
    sysblock = [("constrained_magnetization", ' "none"'), ("degauss", "  1.00000e-02"),
                ("ecutrho", "  4.40536e+02"), ("ecutwfc", "  4.89484e+01"),
                ("ibrav", "    0"), ("tot_charge", "      0"),
                ("nat", "%5d" % len(symbols)), ("ntyp", "%5d" % len(SPECIES)),
                ("occupations", ' "smearing"'), ("smearing", ' "gaussian"'),
                ("vdw_corr", ' "DFT-D3"')]
    L += block("SYSTEM", sysblock) + [""]
    L += block("ELECTRONS", electrons) + [""]
    m = kmesh(cell, klen)
    kline = (KPOINTS if klen <= 0 else
             " %d  %d  %d  0 0 0" % m)
    L += ["K_POINTS {automatic}", kline, "", ""]
    L += ["CELL_PARAMETERS {angstrom}"]
    for row in cell:
        L.append("%15.6f%15.6f%15.6f" % tuple(row))
    L += ["ATOMIC_SPECIES"]
    for s, m, upf in SPECIES:
        L.append("%4s %12.5f  %s" % (s, m, upf))
    L += ["ATOMIC_POSITIONS {angstrom}"]
    for s, p in zip(symbols, positions):
        L.append("%4s %11.7f %11.7f %11.7f" % (s, p[0], p[1], p[2]))
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traj", default="mdrun2/runs/*/output/out.traj",
                    help="trajectory glob (anything ase.io can read)")
    ap.add_argument("--select", default="",
                    help="file listing frames to keep, one 'run:step' per line "
                         "(e.g. mdrun2_screen_priority.txt). Order is preserved.")
    ap.add_argument("--out", default="newCandidate_scf", help="output directory")
    ap.add_argument("--limit", type=int, default=0, help="stop after N inputs (0 = all)")
    ap.add_argument("--min_dist", type=float, default=1.75,
                    help="skip frames whose shortest interatomic distance is below this [A]")
    ap.add_argument("--fmax_max", type=float, default=15.0,
                    help="skip frames whose |F|max exceeds this [eV/A] "
                         "(uses the forces stored in the trajectory; 0 = no cut)")
    ap.add_argument("--pseudo_dir", default="./")
    ap.add_argument("--klen", type=float, default=0.0,
                    help="k-mesh target: mesh_i = ceil(klen / |a_i|) [A]. 0 keeps "
                         "the store's historical Gamma-only setting. Use ~20 to "
                         "put small new cells on the same footing as the existing "
                         "18-19.6 A cells (a 9.86 A cube then gets 2x2x2)")
    ap.add_argument("--stress", action="store_true",
                    help="also compute the stress tensor (tstress = .true.)")
    ap.add_argument("--mixing_beta", type=float, default=None,
                    help="override ELECTRONS mixing_beta (store used 0.05; 0.3 "
                         "converges the dense-mesh trainset ~4x faster with the "
                         "SAME converged E/F/stress). None keeps 0.05.")
    ap.add_argument("--system", default=SYSTEM_NAME, help="file name prefix")
    args = ap.parse_args()

    from ase.io import iread

    wanted = None
    if args.select:
        wanted = [l.strip() for l in open(args.select) if l.strip()]
        order = {name: i for i, name in enumerate(wanted)}
        wanted = set(wanted)
        print("selection: %d frames from %s" % (len(wanted), args.select))

    tfiles = sorted(glob.glob(args.traj))
    if not tfiles:
        sys.exit("no trajectory matched %s" % args.traj)
    os.makedirs(args.out, exist_ok=True)

    rows, kept = [], []
    for tf in tfiles:
        tag = run_tag(tf)
        for k, at in enumerate(iread(tf)):
            name = "%s:%d" % (tag, k)
            if wanted is not None and name not in wanted:
                continue
            cell = np.asarray(at.get_cell())
            pos = at.get_positions()
            sym = at.get_chemical_symbols()
            dmin = min_pair_distance(pos, cell)
            try:
                fmax = float(np.linalg.norm(at.get_forces(), axis=1).max())
            except Exception:
                fmax = float("nan")
            reason = ""
            if dmin < args.min_dist:
                reason = "min_dist %.3f" % dmin
            elif args.fmax_max and fmax == fmax and fmax > args.fmax_max:
                reason = "fmax %.1f" % fmax
            rows.append(dict(name=name, run=tag, step=k, nat=len(sym),
                             V_per_atom=abs(np.linalg.det(cell)) / len(sym),
                             min_pair_dist=dmin, fmax=fmax, skipped=reason))
            if not reason:
                kept.append((name, tag, k, sym, pos, cell, rows[-1]))

    # keep the order of the selection file (= FPS order) so --limit takes the best N
    if wanted is not None:
        kept.sort(key=lambda t: order[t[0]])
    if args.limit:
        kept = kept[:args.limit]
    keep_names = {t[0] for t in kept}
    for r in rows:
        if not r["skipped"] and r["name"] not in keep_names:
            r["skipped"] = "over --limit"

    for name, tag, k, sym, pos, cell, r in kept:
        stem = "%s.%s.%d" % (args.system, safe_tag(tag), k)
        write_pwin(os.path.join(args.out, stem + ".pwin"), sym, pos, cell,
                   stem, args.pseudo_dir, args.stress, args.klen,
                   mixing_beta=args.mixing_beta)
        r["file"] = stem + ".pwin"
        r["kmesh"] = "%dx%dx%d" % kmesh(cell, args.klen)

    man = os.path.join(args.out, "manifest.csv")
    with open(man, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "name", "run", "step", "nat", "V_per_atom",
                    "min_pair_dist", "fmax", "skipped", "kmesh"])
        for r in rows:
            w.writerow([r.get("file", ""), r["name"], r["run"], r["step"], r["nat"],
                        "%.4f" % r["V_per_atom"], "%.3f" % r["min_pair_dist"],
                        "%.2f" % r["fmax"], r["skipped"], r.get("kmesh", "")])

    nskip = sum(1 for r in rows if r["skipped"])
    print("%d inputs written to %s/" % (len(kept), args.out))
    print("  skipped %d (%s)" % (nskip, ", ".join(
        "%s: %d" % (k, v) for k, v in sorted(
            __import__("collections").Counter(
                r["skipped"].split()[0] for r in rows if r["skipped"]).items())) or "-"))
    from collections import Counter as _C
    km = _C(r["kmesh"] for r in rows if r.get("kmesh"))
    print("  k-mesh (klen=%.1f A): %s" % (args.klen,
          ", ".join("%s x%d" % kv for kv in sorted(km.items())) or "-"))
    print("  manifest: %s" % man)
    print("  pseudopotentials must be in %s (%s)" %
          (args.pseudo_dir, ", ".join(u for _, _, u in SPECIES)))


if __name__ == "__main__":
    main()

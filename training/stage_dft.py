#!/usr/bin/env python
# =============================================================================
# stage_dft.py -- turn the selected pool into per-composition QE jobs, and then
#                 into DeePMD systems under matersim_embed/data/.
#
# One DeePMD system = one fixed type.raw = one composition, so the staging is
# per composition all the way through: sel/<formula>.extxyz -> dft/<NAME>/*.pwin
# -> (pjsub) -> scf_to_deepmd.py --name <NAME>.<tag> -> data/<NAME>.<tag>/.
#
# <NAME> follows the store's convention Li094P016S078Br009I000Cl009 so the new
# systems sort next to the existing ones and the composition is readable.
#
#   python stage_dft.py --sel sel --out dft --tag v1 --submit
# =============================================================================
import argparse
import csv
import glob
import os
import subprocess
import sys
from collections import Counter

import numpy as np
from ase.io import read

ATMTYP = ["Li", "P", "S", "Br", "I", "Cl"]


def store_name(atoms):
    c = {s: 0 for s in ATMTYP}
    for s in atoms.get_chemical_symbols():
        c[s] += 1
    return "".join("%s%03d" % (s, c[s]) for s in ATMTYP)


def read_meshes(manifest):
    """Counter of the k-meshes actually assigned, from make_scf_inputs' manifest.
    Frames skipped by make_scf have an empty kmesh cell and are ignored."""
    meshes = Counter()
    if not os.path.exists(manifest):
        return meshes
    with open(manifest) as fh:
        for row in csv.DictReader(fh):
            if not row.get("skipped") and row.get("kmesh"):
                meshes[row["kmesh"]] += 1
    return meshes


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--sel", default="sel", help="dir of selected <formula>.extxyz")
    ap.add_argument("--out", default="dft")
    ap.add_argument("--tag", default="v1", help="suffix for the DeePMD system name")
    ap.add_argument("--make_scf", default=os.path.join(here, "make_scf_inputs.py"))
    ap.add_argument("--pseudo_dir", required=True)
    ap.add_argument("--qe_job", default=os.path.join(here, "genkai", "sub_qe_scf.sh"))
    ap.add_argument("--klen", type=float, default=0.0,
                    help="k-mesh target passed to make_scf_inputs: "
                         "mesh_i = ceil(klen / |a_i|) [A]. 0 = Gamma (historical). "
                         "The old-data recompute used klen=18 (all GB frames -> 1x1x2). "
                         "Compact bulk cells need a larger klen for 3x3x3.")
    ap.add_argument("--klen_bulk", type=float, default=None,
                    help="override --klen for compact (bulk) compositions whose "
                         "median volume/atom < --bulk_vpa_max. Lets GB keep klen=18 "
                         "(1x1x2) while bulk gets 3x3x3 in the SAME run.")
    ap.add_argument("--bulk_vpa_max", type=float, default=20.0,
                    help="a composition with median V/atom below this [A^3/atom] is "
                         "treated as bulk and uses --klen_bulk.")
    ap.add_argument("--mixing_beta", type=float, default=None,
                    help="passed to make_scf_inputs; 0.3 converges the trainset "
                         "~4x faster than the store's 0.05 (same converged labels).")
    ap.add_argument("--submit", action="store_true")
    args = ap.parse_args()

    files = [f for f in sorted(glob.glob(os.path.join(args.sel, "*.extxyz")))
             if not f.endswith("SELECTED.extxyz")]
    os.makedirs(args.out, exist_ok=True)
    rows = []
    splits = []
    for f in files:
        fr = read(f, index=":")
        name = store_name(fr[0])
        d = os.path.join(args.out, name)
        # choose the k-mesh target: compact (bulk) cells need a denser mesh than
        # the large GB supercells, so a single klen would either under-converge
        # bulk or over-cost GB. Classify by median volume/atom.
        vpa = float(np.median([at.get_volume() / len(at) for at in fr]))
        klen = args.klen
        tag = "gb"
        if args.klen_bulk is not None and vpa < args.bulk_vpa_max:
            klen = args.klen_bulk
            tag = "bulk"
        cmd = [sys.executable, args.make_scf, "--traj", f, "--out", d,
               "--min_dist", "0", "--fmax_max", "1e9", "--stress",
               "--pseudo_dir", args.pseudo_dir, "--system", name,
               "--klen", "%g" % klen]
        if args.mixing_beta is not None:
            cmd += ["--mixing_beta", "%g" % args.mixing_beta]
        r = subprocess.run(cmd, capture_output=True, text=True)
        n = len(glob.glob(os.path.join(d, "*.pwin")))
        # verify every frame of this composition landed on ONE k-mesh: mixing
        # meshes puts the k-convergence error straight into the energy labels
        # (same structure, different E) -- the "label inconsistency" the store
        # avoided by pinning the old data to a single 1x1x2 mesh.
        meshes = read_meshes(os.path.join(d, "manifest.csv"))
        if len(meshes) > 1:
            splits.append((name, tag, klen, vpa, meshes))
        rows.append((name, len(fr), n, len(fr[0]), d))
        print("%-34s frames=%3d  pwin=%3d  nat=%3d  V/at=%5.2f  %-4s klen=%g  mesh=%s -> %s" %
              (name, len(fr), n, len(fr[0]), vpa, tag, klen,
               ",".join("%sx%d" % kv for kv in sorted(meshes.items())) or "-", d))
        if r.returncode != 0:
            print("   make_scf_inputs stderr:", r.stderr[-300:])

    if splits:
        print("\n[WARN] %d composition(s) span multiple k-meshes -- adjust --klen/"
              "--klen_bulk so each sits on a single mesh before submitting:" % len(splits))
        for name, tag, klen, vpa, meshes in splits:
            print("   %-34s %-4s klen=%g V/at=%.2f -> %s" %
                  (name, tag, klen, vpa,
                   ", ".join("%sx%d" % kv for kv in sorted(meshes.items()))))

    print("\ntotal %d compositions, %d SCF inputs"
          % (len(rows), sum(r[2] for r in rows)))
    with open(os.path.join(args.out, "systems.txt"), "w") as fh:
        for name, nfr, npw, nat, d in rows:
            fh.write("%s %d %d %d %s.%s\n" % (name, npw, nat, nfr,
                                              name, args.tag))
    print("wrote %s/systems.txt" % args.out)

    # pjsub has no --export, so sub_qe_scf.sh picks INDIR up from indir.txt in
    # its own working directory: give each composition its own run dir.
    runs = []
    for name, nfr, npw, nat, d in rows:
        if npw == 0:
            continue
        rd = os.path.join(args.out, "run_" + name)
        os.makedirs(rd, exist_ok=True)
        with open(os.path.join(rd, "indir.txt"), "w") as fh:
            fh.write(os.path.abspath(d) + "\n")
        runs.append((name, rd))
    if args.submit:
        for name, rd in runs:
            r = subprocess.run(["pjsub", "-N", "qe_" + name[:8],
                                os.path.abspath(args.qe_job)],
                               cwd=rd, capture_output=True, text=True)
            print("  submit %-34s %s" % (name, (r.stdout + r.stderr).strip()))
    else:
        print("\nsubmit with:")
        print("  for r in %s/run_*/ ; do (cd $r && pjsub %s) ; done"
              % (args.out, os.path.abspath(args.qe_job)))


if __name__ == "__main__":
    main()

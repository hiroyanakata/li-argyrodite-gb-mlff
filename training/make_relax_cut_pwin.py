#!/usr/bin/env python
# =============================================================================
# make_relax_cut_pwin.py -- pw.x constrained "relax" input for a GB-core cut
# slab (cut_gb_core.py / relax_md_cut.py output).
#
# Same DFT settings as every other input of the project (make_scf_inputs.py:
# cutoffs, smearing, DFT-D3, pseudopotentials, k-mesh from --klen), turned into
# a geometry optimisation:
#   calculation = "relax", BFGS, forc_conv_thr 1e-3 Ry/bohr (~0.026 eV/A)
#   atoms with arrays['fixed']==1 get if_pos 0 0 0 (outer layers at the vacuum)
#   disk_io "low" + max_seconds so a job that hits the wall can be restarted
#   mixing_mode "local-TF" (slab + vacuum = inhomogeneous charge)
# No stress (vacuum along x makes it meaningless).
#
#   python make_relax_cut_pwin.py --data cut_n8_gb1.extxyz --klen 26 \
#          --pseudo_dir /path/to/pseudo --out dftrx_n8_gb1
# =============================================================================
import argparse
import os
import sys

import numpy as np
from ase.io import read


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--index", default="-1", help="frame index in --data")
    ap.add_argument("--klen", type=float, default=26.0)
    ap.add_argument("--pseudo_dir", required=True)
    ap.add_argument("--scripts", default=os.path.dirname(os.path.abspath(__file__)),
                    help="where make_scf_inputs.py lives")
    ap.add_argument("--max_seconds", type=float, default=82000.0)
    ap.add_argument("--nstep", type=int, default=300)
    ap.add_argument("--out", required=True, help="output directory (= prefix)")
    args = ap.parse_args()

    sys.path.insert(0, args.scripts)
    import make_scf_inputs as M

    at = read(args.data, index=int(args.index))
    fixed = at.arrays.get("fixed", np.zeros(len(at), int))
    os.makedirs(args.out, exist_ok=True)
    stem = os.path.basename(os.path.normpath(args.out))
    path = os.path.join(args.out, stem + ".pwin")
    M.write_pwin(path, at.get_chemical_symbols(), at.get_positions(), np.asarray(at.get_cell()),
                 stem, os.path.expanduser(args.pseudo_dir), False, args.klen, mixing_beta=0.1)

    txt = open(path).read().splitlines()
    out, in_pos, k = [], False, 0
    for line in txt:
        s = line.strip()
        if s.startswith("calculation"):
            line = '    calculation = "relax"'
        elif s.startswith("disk_io"):
            line = '    disk_io     =  "low"'
        elif s.startswith("max_seconds"):
            line = "    max_seconds =  %.5e\n    nstep       =  %d\n    forc_conv_thr = 1.0d-3\n" \
                   "    etot_conv_thr = 1.0d-4" % (args.max_seconds, args.nstep)
        elif s.startswith("mixing_beta"):
            out.append(line)
            line = '    mixing_mode      =  "local-TF"'
        if s.startswith("K_POINTS"):
            out += ["&IONS", '    ion_dynamics = "bfgs"', "/", ""]
        if s.startswith("ATOMIC_POSITIONS"):
            in_pos = True
            out.append(line)
            continue
        if in_pos and s and k < len(at):
            if fixed[k]:
                line = line + "  0 0 0"
            k += 1
        out.append(line)
    assert k == len(at), (k, len(at))
    open(path, "w").write("\n".join(out) + "\n")
    print("wrote %s (nat=%d, fixed=%d, kmesh=%s)" % (path, len(at), int(fixed.sum()),
                                                     "x".join(map(str, M.kmesh(at.get_cell(), args.klen)))))


if __name__ == "__main__":
    main()

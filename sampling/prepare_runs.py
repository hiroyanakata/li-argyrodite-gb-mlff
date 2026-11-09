#!/usr/bin/env python
"""Generate ODPMD run directories over a grid of density and temperature.

The previous batch (lmp.*.npt000-009) used one single protocol: identical
in.test in all ten runs, NVT at a fixed cell, so V/atom stayed inside
18.46-18.75 in every one of them.  That is the same compressed regime the last
four data cycles already sampled, which is why 186 of the 188 candidates landed
in cluster3 (already 34% of the store).  V/atom above 20 has not been touched
since iter0 even though the store reaches 25.2.

This script varies the two knobs that actually move the sampling: the cell
volume and the temperature.  Initial structures are cycled from the existing
runs, so each generated run still starts from a physically equilibrated
configuration.

  python prepare_runs.py --out runs
  python prepare_runs.py --out runs --vatom 18.6 20 22 25 --temp 300 600 900

Then, on the machine that has the model:

  bash runs/stage_model.sh /path/to/best_model.pth
  cd runs/v20.0_T600 && sh ODPMD.para.sh
"""
import argparse, glob, os, re, shutil, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def read_data_cell(path):
    """Return (lines, {axis: (lo, hi, index)}) for a LAMMPS data file."""
    with open(path) as fh:
        lines = fh.readlines()
    box = {}
    for i, ln in enumerate(lines):
        m = re.match(r"\s*(\S+)\s+(\S+)\s+([xyz])lo\s+[xyz]hi", ln)
        if m:
            box[m.group(3)] = (float(m.group(1)), float(m.group(2)), i)
    if len(box) != 3:
        sys.exit("could not parse box bounds from %s" % path)
    return lines, box


def scale_data(src, dst, s):
    """Write src to dst with the cell and all coordinates scaled by s."""
    lines, box = read_data_cell(src)
    for ax in "xyz":
        lo, hi, i = box[ax]
        lines[i] = " %.10f %.10f %slo %shi\n" % (lo * s, hi * s, ax, ax)
    # locate the Atoms section
    start = None
    for i, ln in enumerate(lines):
        if ln.strip().startswith("Atoms"):
            start = i + 1
            break
    if start is None:
        sys.exit("no Atoms section in %s" % src)
    i = start
    n = 0
    while i < len(lines):
        aa = lines[i].split()
        if len(aa) == 0:
            i += 1
            continue
        if not aa[0].isdigit():          # next section header
            break
        if len(aa) < 5:
            break
        x, y, z = (float(aa[2]) * s, float(aa[3]) * s, float(aa[4]) * s)
        lines[i] = " %s %s %.10f %.10f %.10f\n" % (aa[0], aa[1], x, y, z)
        n += 1
        i += 1
    with open(dst, "w") as fh:
        fh.writelines(lines)
    return n


def scale_xyz(src, dst, s):
    """Scale the cell line and the coordinates of an ODPMD-style xyz file."""
    with open(src) as fh:
        lines = fh.readlines()
    nat = int(lines[0].split()[0])
    cell = [float(v) * s for v in lines[1].split()[:3]]
    out = [lines[0], " %.5f  %.5f  %.5f\n" % tuple(cell)]
    for ln in lines[2:2 + nat]:
        aa = ln.split()
        out.append(" %-3s %18.11f %18.11f %18.11f\n"
                   % (aa[0], float(aa[1]) * s, float(aa[2]) * s, float(aa[3]) * s))
    with open(dst, "w") as fh:
        fh.writelines(out)


def colvars_params(text):
    """targetCenters and targetNumStages out of an ODP.colvars file."""
    c = re.search(r"targetCenters\s+([\d.]+)", text)
    g = re.search(r"targetNumStages\s+(\d+)", text)
    s = re.search(r"targetNumSteps\s+(\d+)", text)
    return float(c.group(1)), int(g.group(1)), int(s.group(1))


def write_colvars(src_text, dst, num_steps):
    """Copy the colvars file with targetNumSteps replaced (= change the pull rate)."""
    out = re.sub(r"(targetNumSteps\s+)\d+", r"\g<1>%d" % num_steps, src_text)
    with open(dst, "w") as fh:
        fh.write(out)


def vatom_of(path):
    _, box = read_data_cell(path)
    v = 1.0
    for ax in "xyz":
        lo, hi, _ = box[ax]
        v *= (hi - lo)
    with open(path) as fh:
        for ln in fh:
            m = re.match(r"\s*(\d+)\s+atoms", ln)
            if m:
                return v / int(m.group(1)), int(m.group(1))
    sys.exit("atom count not found in %s" % path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="../lmp.*npt0*",
                    help="glob of existing run dirs used as starting structures")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--vatom", type=float, nargs="+",
                    default=[20.0, 22.0, 25.0],
                    help="target V/atom [A^3] of the density x temperature grid. "
                         "18.6 is deliberately absent: that regime already holds 43,470 frames")
    ap.add_argument("--temp", type=float, nargs="+", default=[300.0, 600.0, 900.0])
    ap.add_argument("--pull", type=float, nargs="+", default=[0.2, 0.1, 0.05],
                    help="pull rates [A/ps] for the probe runs. 0.2 = the rate used so far")
    ap.add_argument("--pull_vatom", type=float, default=18.6,
                    help="V/atom of the pull-rate probe: the well-characterised regime, so the "
                         "rate is the only thing that differs from the previous batch")
    ap.add_argument("--pull_temp", type=float, default=300.0)
    ap.add_argument("--pull_ubias", type=float, default=4.0,
                    help="bias value every probe run must reach; sets its length. The previous "
                         "batch broke at a bias of 2.3-4.3")
    ap.add_argument("--nsteps", type=int, default=30000,
                    help="biased steps of the grid runs (equilibration is added on top)")
    ap.add_argument("--seed0", type=int, default=23456789)
    args = ap.parse_args()

    srcs = sorted(d for d in glob.glob(os.path.join(HERE, args.src)) if os.path.isdir(d))
    srcs = [d for d in srcs if os.path.exists(os.path.join(d, "sample.data"))]
    if not srcs:
        sys.exit("no source run dirs matched %s" % args.src)
    tmpl = open(os.path.join(HERE, "in.test.template")).read()
    outroot = os.path.join(HERE, args.out)
    os.makedirs(outroot, exist_ok=True)

    cv_text = open(os.path.join(HERE, "ODP.colvars.orig")).read()
    centers, nstages, nsteps0 = colvars_params(cv_text)
    dt_ps = float(re.search(r"timestep\s+([\d.]+)", tmpl).group(1))
    equil = int(re.search(r"md_equil\s+(\d+)", tmpl).group(1))
    rate0 = centers / (nstages * nsteps0 * dt_ps)      # A/ps of the unmodified colvars file

    # (V/atom, T, pull rate, biased steps) of every run
    plan = [(va, T, rate0, args.nsteps) for va in args.vatom for T in args.temp]
    for r in args.pull:
        ns = int(round(centers / (r * nstages * dt_ps)))          # targetNumSteps for this rate
        nb = int(round(args.pull_ubias / centers * nstages * ns))  # steps to reach pull_ubias
        plan.append((args.pull_vatom, args.pull_temp, r, nb))

    made = []
    for k, (va, T, rate, nbiased) in enumerate(plan):
        src = srcs[k % len(srcs)]
        ns_stage = int(round(centers / (rate * nstages * dt_ps)))
        name = "v%.1f_T%d" % (va, int(T))
        if abs(rate - rate0) > 1e-9 or va == args.pull_vatom:
            name += "_p%.2f" % rate
        dst = os.path.join(outroot, name)
        os.makedirs(dst, exist_ok=True)
        v0, nat = vatom_of(os.path.join(src, "sample.data"))
        s = (va / v0) ** (1.0 / 3.0)
        scale_data(os.path.join(src, "sample.data"), os.path.join(dst, "sample.data"), s)
        # Reactant.xyz defines the P-P neighbour mapping of the collective
        # variable, so it has to be scaled with the cell.  Product.xyz is
        # read but not used for the CV value (P is set to targetCenters).
        scale_xyz(os.path.join(src, "Reactant.xyz"), os.path.join(dst, "Reactant.xyz"), s)
        shutil.copy(os.path.join(src, "Product.xyz"), dst)
        write_colvars(cv_text, os.path.join(dst, "ODP.colvars"), ns_stage)
        shutil.copy(os.path.join(HERE, "ODPMD.py"), dst)
        shutil.copy(os.path.join(HERE, "ODPMD.para.sh"), dst)
        inp = (tmpl.replace("__TEMP__", "%.1f" % T)
                   .replace("run          30000", "run          %d" % (equil + nbiased))
                   .replace("23456789", str(args.seed0 + k + 1)))
        with open(os.path.join(dst, "in.test"), "w") as fh:
            fh.write(inp)
        made.append((name, os.path.basename(src)[-6:], v0, va, s, T, rate, ns_stage, equil + nbiased))

    with open(os.path.join(outroot, "stage_model.sh"), "w") as fh:
        fh.write("#!/bin/sh\n# usage: bash stage_model.sh /path/to/best_model.pth\n"
                 "for d in %s/*/ ; do cp \"$1\" \"$d/best_model.pth\"; done\n"
                 % os.path.basename(outroot))
    with open(os.path.join(outroot, "run_all.sh"), "w") as fh:
        fh.write("#!/bin/sh\nfor d in */ ; do (cd \"$d\" && pjsub ODPMD.para.sh); done\n")

    print("%-20s %-7s %8s %8s %7s %6s %9s %9s %8s"
          % ("run", "from", "V/at_src", "V/at_new", "scale", "T[K]", "pull[A/ps]", "stgSteps", "run"))
    for m in made:
        print("%-20s %-7s %8.2f %8.2f %7.4f %6.0f %10.3f %9d %8d" % m)
    print("\n%d runs written to %s" % (len(made), outroot))
    print("next: bash %s/stage_model.sh <best_model.pth>, then submit with run_all.sh" % args.out)


if __name__ == "__main__":
    main()

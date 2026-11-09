#!/usr/bin/env python
"""Convert single-point pw.x SCF outputs into a DeePMD raw dataset.

This is the batch counterpart of AutoReaction.v17.py :: QEsolver.GetOPTinfo.
Everything that touches numbers is kept byte-identical to that routine so the
frames produced here are on exactly the same scale as the existing store:

  * bohr2Ang = 0.529177 and ry2ev = 13.6058  (AutoReaction.v17.py L22-23 --
    NOT the CODATA values; a different constant would shift every energy by
    ~0.45 eV/cell relative to the frames already in data/)
  * cell    <- "crystal axes" in alat units, alat = celldm(1) * bohr2Ang
  * coord   <- "Cartesian axes" block, alat units
  * energy  <- "!    total energy" in Ry, raw total (no atmene subtraction)
  * force   <- "Forces acting on atoms" in Ry/au
  * type    <- atmtyp = ["Li","P","S","Br","I","Cl"] (AutoReaction.v17.py L6593)

The type.raw order is the one thing that silently destroys a trained model, so
it is derived from the atom order inside each .out and then cross-checked
against every other frame and against --check_type_against. See the settype()
guard in AutoReaction.v17.py L4296.

Usage:
  python scf_to_deepmd.py newCandidate_scf \\
      --name Li094P016S078Br009I000Cl009.scf01 \\
      --check_type_against data/Li094P016S078Br009I000Cl009.0805/type.raw
  # then:  cd data/<name> && ../raw_to_set.sh 40000
"""
import argparse
import os
import re
import subprocess
import sys

import numpy as np

# --- constants: must match AutoReaction.v17.py exactly ----------------------
BOHR2ANG = 0.529177
RY2EV = 13.6058
ATMTYP = ["Li", "P", "S", "Br", "I", "Cl"]


def parse_out(path, nat_expect=None):
    """Replicate GetOPTinfo for one single-point SCF output.

    Returns (cell(3,3), coord(nat,3), atm[nat], energy, force(nat,3)) or None
    if the run is unusable.
    """
    with open(path, "r", errors="replace") as fh:
        lines = fh.readlines()
    txt = "".join(lines)

    if "JOB DONE." not in txt:
        return None, "no JOB DONE"
    if "convergence has been achieved" not in txt:
        return None, "scf not converged"

    m = re.search(r"number of atoms/cell\s*=\s*(\d+)", txt)
    if not m:
        return None, "no nat"
    nat = int(m.group(1))
    if nat_expect is not None and nat != nat_expect:
        return None, f"nat {nat} != {nat_expect}"

    alat = None
    cell = np.zeros((3, 3))
    coord = np.zeros((nat, 3))
    force = np.zeros((nat, 3))
    atm = [None] * nat
    energy = None
    got_cell = got_coord = got_force = False

    i = 0
    n = len(lines)
    while i < n:
        aa = lines[i].split()
        if len(aa) > 3 and aa[0] == "celldm(1)=" and alat is None:
            alat = float(aa[1]) * BOHR2ANG
        elif len(aa) > 1 and aa[0] == "crystal" and aa[1] == "axes:":
            for k in range(3):
                bb = lines[i + 1 + k].split()
                cell[k] = [float(bb[3]) * alat, float(bb[4]) * alat,
                           float(bb[5]) * alat]
            got_cell = True
            i += 3
        elif len(aa) > 1 and aa[0] == "Cartesian" and aa[1] == "axes":
            for k in range(nat):
                bb = lines[i + 3 + k].split()
                atm[k] = bb[1]
                coord[k] = [float(bb[6]) * alat, float(bb[7]) * alat,
                            float(bb[8]) * alat]
            got_coord = True
            i += 2 + nat
        elif len(aa) > 1 and aa[0] == "!" and aa[1] == "total":
            energy = float(aa[4]) * RY2EV
        elif len(aa) > 1 and aa[0] == "Forces" and aa[1] == "acting":
            for k in range(nat):
                bb = lines[i + 2 + k].split()
                force[k] = [float(bb[6]) * RY2EV / BOHR2ANG,
                            float(bb[7]) * RY2EV / BOHR2ANG,
                            float(bb[8]) * RY2EV / BOHR2ANG]
            got_force = True
            i += 1 + nat
        i += 1

    if not (got_cell and got_coord and got_force) or energy is None:
        missing = [k for k, v in [("cell", got_cell), ("coord", got_coord),
                                  ("force", got_force),
                                  ("energy", energy is not None)] if not v]
        return None, "missing " + ",".join(missing)
    return (cell, coord, atm, energy, force), ""


def stress_from_out(path):
    """QE stress -> ASE-convention stress [eV/A^3] and the DeePMD virial [eV].

    Parsed with ase.io rather than by hand: the sign and unit conversion of the
    pw.x stress block is exactly the kind of thing that silently flips, and
    parity.py already validates the ase reading of these same outputs against
    MatterSim stresses. virial = -stress * V is the DeePMD convention.
    """
    from ase.io import read as _read
    at = _read(path, format="espresso-out")
    sig = at.get_stress(voigt=False)              # eV/A^3, positive = tension
    V = at.get_volume()
    return sig, -sig * V


def types_from_atm(atm):
    t = []
    for a in atm:
        t.append(ATMTYP.index(a) if a in ATMTYP else -1)
    if -1 in t:
        bad = sorted({a for a, x in zip(atm, t) if x < 0})
        raise SystemExit(f"element(s) not in atmtyp {ATMTYP}: {bad}")
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("srcdir", help="directory holding the pw.x .out files")
    ap.add_argument("--name", required=True,
                    help="dataset name -> data/<name>/")
    ap.add_argument("--datadir", default="data")
    ap.add_argument("--check_type_against", default=None,
                    help="existing type.raw that the new one must match")
    ap.add_argument("--nat", type=int, default=None)
    ap.add_argument("--set_size", type=int, default=40000,
                    help="lines per set for raw_to_set.sh")
    ap.add_argument("--run_raw_to_set", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--with_stress", action="store_true",
                    help="also write virial.raw (needs the SCF to have run with "
                         "tstress = .true.; make_scf_inputs.py --stress does that). "
                         "Required if the fine-tune is to use --include_stresses, "
                         "since the existing store carries no virial at all")
    args = ap.parse_args()

    outs = sorted(f for f in os.listdir(args.srcdir) if f.endswith(".out"))
    if not outs:
        sys.exit(f"no .out in {args.srcdir}")

    frames, skipped, ref_type, ref_names = [], [], None, []
    for f in outs:
        res, why = parse_out(os.path.join(args.srcdir, f), args.nat)
        if res is None:
            skipped.append((f, why))
            continue
        cell, coord, atm, energy, force = res
        t = types_from_atm(atm)
        if ref_type is None:
            ref_type = t
        elif t != ref_type:
            skipped.append((f, "ATOM ORDER DIFFERS from first frame"))
            continue
        vir = None
        if args.with_stress:
            try:
                _, vir = stress_from_out(os.path.join(args.srcdir, f))
            except Exception as e:
                print("  SKIP", f, "-- no usable stress:", e)
                continue
        frames.append((cell, coord, energy, force, vir))
        ref_names.append(f[:-4])

    print(f"parsed {len(frames)} frames, skipped {len(skipped)}")
    for f, why in skipped[:20]:
        print("  SKIP", f, "--", why)
    if not frames:
        sys.exit("nothing to write")

    # --- the guard that matters: type.raw must match the existing store ----
    if args.check_type_against:
        ref = open(args.check_type_against).read().split()
        cur = [str(x) for x in ref_type]
        if ref != cur:
            print("\nFATAL: type.raw does not match", args.check_type_against)
            print("  existing:", ref[:20], "...")
            print("  new     :", cur[:20], "...")
            sys.exit("refusing to write -- this is the bug that breaks the MD")
        print(f"type.raw matches {args.check_type_against}  (nat={len(cur)})")
    counts = {ATMTYP[i]: ref_type.count(i) for i in range(len(ATMTYP))}
    print("composition:", counts)

    E = np.array([fr[2] for fr in frames])
    F = np.concatenate([fr[3][None] for fr in frames])
    print(f"E/atom  min {E.min()/len(ref_type):.3f}  p50 "
          f"{np.median(E)/len(ref_type):.3f}  max {E.max()/len(ref_type):.3f} eV")
    fn = np.linalg.norm(F, axis=2)
    print(f"fmax    p50 {np.median(fn.max(axis=1)):.3f}  max "
          f"{fn.max():.3f} eV/A")

    if args.dry_run:
        print("dry run -- nothing written")
        return

    dest = os.path.join(args.datadir, args.name)
    if os.path.exists(dest):
        sys.exit(f"{dest} already exists -- remove it or pick another --name")
    os.makedirs(dest)

    with open(os.path.join(dest, "type.raw"), "w") as fh:
        for t in ref_type:
            fh.write(" %3i" % t)
        fh.write("\n")
    fv = open(os.path.join(dest, "virial.raw"), "w") if args.with_stress else None
    with open(os.path.join(dest, "coord.raw"), "w") as f1, \
         open(os.path.join(dest, "force.raw"), "w") as f2, \
         open(os.path.join(dest, "energy.raw"), "w") as f3, \
         open(os.path.join(dest, "box.raw"), "w") as f4:
        for cell, coord, energy, force, vir in frames:
            f1.write("".join(" %15.8f" % v for v in coord.ravel()) + "\n")
            f2.write("".join(" %15.8f" % v for v in force.ravel()) + "\n")
            f3.write(" %15.8f\n" % energy)
            f4.write("".join(" %15.8f" % v for v in cell.ravel()) + "\n")
            if fv is not None:
                fv.write("".join(" %15.8f" % v for v in vir.ravel()) + "\n")
    if fv is not None:
        fv.close()
        print("wrote virial.raw (DeePMD convention: virial = -stress * V, eV)")
    with open(os.path.join(dest, "frames.txt"), "w") as fh:
        fh.write("\n".join(ref_names) + "\n")
    print(f"wrote {dest}/{{coord,force,energy,box,type}}.raw  "
          f"+ frames.txt ({len(frames)} frames)")

    if args.run_raw_to_set:
        sh = os.path.abspath(os.path.join(args.datadir, "raw_to_set.sh"))
        subprocess.check_call(["bash", sh, str(args.set_size)], cwd=dest)
        print("raw_to_set.sh done ->", dest)
    else:
        print(f"next: cd {dest} && bash ../raw_to_set.sh {args.set_size}")


if __name__ == "__main__":
    main()

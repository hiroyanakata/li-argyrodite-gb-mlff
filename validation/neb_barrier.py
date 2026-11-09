#!/usr/bin/env python
# =============================================================================
# neb_barrier.py -- Li migration barrier by CI-NEB (ASE + MatterSim), with an
#                   identical-path export so DFT can recompute the same barrier.
#
# This is validation tool (a): the DFT-intractable large-GB *diffusion* is
# replaced by a *static* migration-barrier check.  We build a vacancy-mediated
# Li hop, run climbing-image NEB with MatterSim, and export every image so the
# SAME geometry path can be single-pointed with QE (MLIP vs DFT barrier on one
# path -- no AIMD needed).  Compare barriers at a GB site vs a bulk site.
#
# Endpoint construction (vacancy hop i -> site j, both Li in the input):
#   base    = structure with Li j deleted            (a vacancy at site j)
#   initial = base                                    (migrating Li i at its site)
#   final   = base with Li i moved onto site j        (vacancy now at i)
# both endpoints are relaxed at fixed cell, then IDPP-interpolated.
#
# Usage:
#   # list candidate Li-Li hops, tagged GB vs bulk:
#   python neb_barrier.py --data gb210_dec.extxyz --list_hops --gb-axis x --gb-pos 0.25,0.75
#   # run one hop:
#   python neb_barrier.py --data gb210_dec.extxyz --model pretrained \
#          --hop 12 34 --images 7 --climb --out gb_hop12
#   # then, identical DFT path (reuse the existing QE-input maker, keep all images):
#   python ../make_scf_inputs.py --traj gb_hop12_path.extxyz --out gb_hop12_dft \
#          --min_dist 0 --fmax_max 1e9
# =============================================================================
import argparse
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

from ase.io import read, write
from ase.optimize import FIRE
from ase.constraints import FixAtoms
try:
    from ase.mep import NEB
except Exception:                              # older ASE
    from ase.neb import NEB

AX = {"x": 0, "y": 1, "z": 2}


def make_calc(args):
    from mattersim.forcefield import MatterSimCalculator
    if args.model.lower() in ("", "pretrained", "default", "none"):
        return MatterSimCalculator(device=args.device)
    return MatterSimCalculator(load_path=args.model, device=args.device)


def mic(dvec, cell):
    """minimum-image vector."""
    inv = np.linalg.inv(cell)
    f = dvec @ inv
    f -= np.round(f)
    return f @ cell


def dist_to_gb(x, L, planes):
    return np.min([np.abs((x - f * L + L / 2) % L - L / 2) for f in planes], axis=0)


def li_neighbor_pairs(atoms, rmax, ax, planes, gb_width):
    """Li-Li pairs within rmax; tagged GB/bulk by the pair midpoint."""
    sym = np.array(atoms.get_chemical_symbols())
    li = np.where(sym == "Li")[0]
    pos = atoms.get_positions()
    cell = np.array(atoms.get_cell())
    L = atoms.cell.lengths()[ax]
    pairs = []
    for a in range(len(li)):
        for b in range(a + 1, len(li)):
            d = mic(pos[li[b]] - pos[li[a]], cell)
            r = np.linalg.norm(d)
            if r < rmax:
                mid = pos[li[a]] + d / 2
                dg = dist_to_gb(np.array([mid[ax]]), L, planes)[0]
                pairs.append((li[a], li[b], r, dg, "GB" if dg <= gb_width else "bulk"))
    pairs.sort(key=lambda t: t[2])
    return pairs


def build_base(atoms, i, j, neutralize=True, comp_halide="Cl", free_s_only=False):
    """vacancy hop: Li i migrates into site j (Li j becomes the vacancy).
    Returns the *host* cell (Li j removed, charge compensated), the index of the
    migrating Li in it, and the target-site position expressed as the nearest
    image of site j to Li i.

    Removing a Li leaves the cell -1; if neutralize, compensate it by turning the
    S farthest from the migrating Li into a halide (Li6-xPS5-xX1+x), so both
    endpoints are charge-neutral and the compensating defect sits away from the
    hop."""
    cell = np.array(atoms.get_cell())
    posi = atoms.get_positions()[i].copy()
    posj = atoms.get_positions()[j].copy()
    posj_img = posi + mic(posj - posi, cell)   # nearest image of site j to Li i
    base = atoms.copy()
    del base[j]
    i2 = i if i < j else i - 1
    if neutralize:
        sym = np.array(base.get_chemical_symbols())
        sidx = np.where(sym == "S")[0]
        pos = base.get_positions()
        if free_s_only:                        # never touch a PS4 vertex S
            pidx = np.where(sym == "P")[0]
            sidx = np.array([s for s in sidx if min(np.linalg.norm(mic(pos[pidx] - pos[s], cell), axis=1)) > 2.4])
            if len(sidx) == 0:
                raise SystemExit("--comp_free_s: no free (non-PS4) S to compensate with")
        d = np.array([np.linalg.norm(mic(pos[s] - pos[i2], cell)) for s in sidx])
        far = sidx[int(np.argmax(d))]          # S farthest from the migrating Li
        base[far].symbol = comp_halide
        print("  neutralized vacancy: S%d -> %s (%.1f A from the hop)"
              % (far, comp_halide, d.max()))
    return base, i2, posj_img


def freeze_far(atoms, centre, radius):
    """Freeze every atom farther than `radius` from `centre` (MIC).

    Both endpoints then share a bit-identical frozen environment, so the band
    interpolates only the local hop instead of a concerted sublattice shuffle."""
    if radius <= 0:
        return 0
    cell = np.array(atoms.get_cell())
    d = np.linalg.norm(mic(atoms.get_positions() - centre, cell), axis=1)
    idx = np.where(d > radius)[0]
    atoms.set_constraint(FixAtoms(indices=idx))
    return len(idx)


def relax(atoms, calc, fmax, steps, label=""):
    atoms.calc = calc
    dyn = FIRE(atoms, logfile=None)
    dyn.run(fmax=fmax, steps=steps)
    f = np.linalg.norm(atoms.get_forces(), axis=1).max()
    ok = f <= fmax
    print("  relax %-8s fmax=%.4f eV/A after %d steps%s"
          % (label, f, dyn.get_number_of_steps(), "" if ok else "   *** NOT CONVERGED ***"))
    return atoms


def main():
    ap = argparse.ArgumentParser(description="Li migration barrier via CI-NEB (MatterSim)")
    ap.add_argument("--data", required=True, help="decorated structure (extxyz/xyz/cif)")
    ap.add_argument("--model", default="pretrained")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--hop", type=int, nargs=2, default=None,
                    help="two Li atom indices (0-based): i migrates into site j")
    ap.add_argument("--list_hops", action="store_true", help="print candidate hops and exit")
    ap.add_argument("--region", default="GB", choices=["GB", "bulk"],
                    help="auto-pick the shortest hop in this region if --hop is absent")
    ap.add_argument("--rmax", type=float, default=4.0, help="max Li-Li hop distance [A]")
    ap.add_argument("--gb-axis", default="x", choices=list(AX))
    ap.add_argument("--gb-pos", default="0.25,0.75")
    ap.add_argument("--gb-width", type=float, default=6.0)
    ap.add_argument("--images", type=int, default=7, help="total images incl. endpoints")
    ap.add_argument("--climb", action="store_true", help="climbing-image NEB")
    ap.add_argument("--k", type=float, default=0.1, help="spring constant [eV/A^2]")
    ap.add_argument("--fmax", type=float, default=0.05)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--relax_fmax", type=float, default=0.05)
    ap.add_argument("--relax_steps", type=int, default=200)
    ap.add_argument("--relax_radius", type=float, default=6.0,
                    help="relax only atoms within this radius of the hop midpoint "
                         "(0 = relax everything; keeps both endpoints identical outside)")
    ap.add_argument("--charged", action="store_true",
                    help="do NOT charge-compensate the migration vacancy (leaves cell -1)")
    ap.add_argument("--comp_halide", default="Cl", choices=["Cl", "Br"],
                    help="element for the S->X vacancy compensation")
    ap.add_argument("--comp_free_s", action="store_true",
                    help="pick the compensating S only among free S2- (no P within 2.4 A). "
                         "Without it any S may be chosen, incl. a PS4 vertex (-> PS3X; old behaviour)")
    ap.add_argument("--out", default="neb")
    args = ap.parse_args()

    atoms = read(args.data)
    atoms.set_pbc(True)
    ax = AX[args.gb_axis]
    planes = [float(v) for v in args.gb_pos.split(",")]

    if args.list_hops or args.hop is None:
        pairs = li_neighbor_pairs(atoms, args.rmax, ax, planes, args.gb_width)
        if args.list_hops:
            print("# i    j     r[A]   d_gb[A]  region")
            for a, b, r, dg, reg in pairs[:40]:
                print("%5d %5d  %6.3f  %6.3f  %s" % (a, b, r, dg, reg))
            print("(%d pairs < %.1f A total)" % (len(pairs), args.rmax))
            return
        cand = [p for p in pairs if p[4] == args.region]
        if not cand:
            raise SystemExit("no %s hop within rmax=%.1f A" % (args.region, args.rmax))
        i, j = cand[0][0], cand[0][1]
        print("auto-picked %s hop: Li %d -> site %d  (r=%.3f A, d_gb=%.3f A)"
              % (args.region, i, j, cand[0][2], cand[0][3]))
    else:
        i, j = args.hop

    calc = make_calc(args)
    initial, imig, posj_img = build_base(atoms, i, j,
                                         neutralize=not args.charged,
                                         comp_halide=args.comp_halide,
                                         free_s_only=args.comp_free_s)
    Q = {"Li": 1, "P": 5, "S": -2, "Cl": -1, "Br": -1, "I": -1}
    sy = np.array(initial.get_chemical_symbols())
    q = int(sum(Q.get(e, 0) * (sy == e).sum() for e in set(sy)))
    print("endpoint composition: %s  net charge = %+d e" % (initial.get_chemical_formula(), q))

    # The host must be at ITS minimum before the endpoints are made, otherwise
    # each endpoint relaxation drags the whole (very floppy) Li sublattice into
    # a different configuration and the interpolated band is meaningless.
    print("relaxing host (initial endpoint) ...")
    relax(initial, make_calc(args), args.relax_fmax, args.relax_steps, "initial")

    # `final` starts from the RELAXED host: it differs from `initial` by the
    # migrating Li only. Atoms outside --relax_radius of the hop are frozen in
    # both endpoints so the environment stays bit-identical.
    cell0 = np.array(initial.get_cell())
    centre = initial.get_positions()[imig] + mic(posj_img - initial.get_positions()[imig], cell0) / 2
    nfix = freeze_far(initial, centre, args.relax_radius)
    final = initial.copy()
    final.positions[imig] = posj_img
    if nfix:
        print("  froze %d/%d atoms farther than %.1f A from the hop midpoint"
              % (nfix, len(initial), args.relax_radius))
    print("relaxing final endpoint ...")
    relax(final, make_calc(args), args.relax_fmax, args.relax_steps, "final")

    # endpoints must differ locally only -- otherwise the band will crash atoms
    dend = np.linalg.norm(mic(final.get_positions() - initial.get_positions(), cell0), axis=1)
    other = np.delete(dend, imig)
    print("  endpoint displacement: migrating Li %.2f A | others max %.2f A, RMS %.3f A"
          % (dend[imig], other.max(), np.sqrt((other ** 2).mean())))
    if other.max() > 0.7:
        print("  *** WARNING: a non-migrating atom moved %.2f A between the endpoints;"
              " the hop is concerted, not a single jump (shrink --relax_radius)" % other.max())

    nim = args.images
    images = [initial] + [initial.copy() for _ in range(nim - 2)] + [final]
    for im in images:
        im.calc = make_calc(args)
    neb = NEB(images, k=args.k, climb=args.climb)
    try:
        neb.interpolate(method="idpp")
    except TypeError:
        neb.interpolate("idpp")
    except Exception:
        neb.interpolate()                      # linear fallback
    print("running NEB (%d images, climb=%s) ..." % (nim, args.climb))
    dyn = FIRE(neb, logfile=None)
    dyn.run(fmax=args.fmax, steps=args.steps)
    # convergence must be judged on the NEB (spring/tangent-projected) forces --
    # the bare per-image forces keep their along-the-path component by design.
    fband = np.linalg.norm(neb.get_forces().reshape(-1, 3), axis=1).max()
    print("  NEB fmax=%.4f eV/A after %d steps%s"
          % (fband, dyn.get_number_of_steps(), ""
             if fband <= args.fmax else "   *** NOT CONVERGED: barrier is not usable ***"))

    # a barrier of tens of eV is always an interpolation artefact: check contacts
    dmin, kmin = 9e9, -1
    for k, im in enumerate(images):
        P = im.get_positions()
        D = mic((P[:, None, :] - P[None, :, :]).reshape(-1, 3), cell0).reshape(len(P), len(P), 3)
        dd = np.linalg.norm(D, axis=2)
        np.fill_diagonal(dd, 9e9)
        if dd.min() < dmin:
            dmin, kmin = dd.min(), k
    print("  closest contact on the band: %.2f A (image %d)" % (dmin, kmin))
    if dmin < 1.6:
        print("  *** WARNING: atoms overlap on the band -- the barrier is an artefact ***")

    E = np.array([im.get_potential_energy() for im in images])
    # reaction coordinate = cumulative MIC path length, normalised to [0,1]
    cell = np.array(initial.get_cell())
    rc = [0.0]
    for a, b in zip(images[:-1], images[1:]):
        d = mic(b.get_positions() - a.get_positions(), cell)
        rc.append(rc[-1] + np.linalg.norm(d))
    rc = np.array(rc) / rc[-1]
    Ef = E - E[0]
    for im, e in zip(images, Ef):
        im.info["neb_energy_eV"] = float(e)
    write(args.out + "_path.extxyz", images)
    np.savetxt(args.out + "_barrier.csv", np.column_stack([rc, Ef]),
               delimiter=",", header="reaction_coord,dE_eV", comments="")

    Ebar_f = float(E.max() - E[0])
    Ebar_r = float(E.max() - E[-1])
    print("=" * 52)
    print(" forward  barrier Ea = %.3f eV" % Ebar_f)
    print(" reverse  barrier    = %.3f eV" % Ebar_r)
    print(" reaction energy dE  = %.3f eV" % (E[-1] - E[0]))
    print(" wrote %s_path.extxyz (%d images) and %s_barrier.csv"
          % (args.out, nim, args.out))
    print(" DFT on the SAME path:")
    print("   python ../make_scf_inputs.py --traj %s_path.extxyz --out %s_dft"
          " --min_dist 0 --fmax_max 1e9" % (args.out, args.out))
    print("=" * 52)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 4))
        plt.plot(rc, Ef, "o-")
        plt.xlabel("reaction coordinate"); plt.ylabel(r"$\Delta E$ [eV]")
        plt.title("Li migration  Ea=%.3f eV" % Ebar_f)
        plt.tight_layout(); plt.savefig(args.out + "_barrier.png", dpi=150)
        print(" wrote %s_barrier.png" % args.out)
    except Exception:
        pass


if __name__ == "__main__":
    main()

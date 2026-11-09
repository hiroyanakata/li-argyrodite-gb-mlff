#!/usr/bin/env python
# =============================================================================
# lifetime.py -- structural-unit / bond lifetime from an MD trajectory
#
# "Lifetime" for this argyrodite system has two useful, paper-ready meanings:
#
#   1. PS4 (P-S) bond lifetime  -- how long a given P-S bond of the PS4^3-
#      tetrahedron survives.  Measured from the bond autocorrelation
#          C(t) = <h_ij(0) h_ij(t)> / <h_ij(0)>
#      where h_ij = 1 while the P-S pair is within r_cut.  tau = integral C(t) dt
#      (or the 1/e time).  This is the direct dynamic analogue of the P-S
#      dissociation the ODP steered-MD was built to sample.
#
#   2. Li site residence time -- how long a Li stays bonded to the same set of
#      framework S neighbours before hopping (correlation of the Li->nearest-S
#      identity).  Governs the ionic conductivity prefactor.
#
# Reads an ASE-readable trajectory (out.traj / .extxyz / lmp.dump via ase).
#
#   python lifetime.py out.traj --mode ps4 --rcut 2.6 --dt 0.05 --out ps4_life
#   python lifetime.py out.traj --mode li  --rcut 3.0 --dt 0.05 --out li_life
# =============================================================================
import argparse
import numpy as np
from ase.io import read


def min_image_dist(pos_a, pos_b, cell):
    inv = np.linalg.inv(cell)
    d = pos_a[:, None, :] - pos_b[None, :, :]
    f = d @ inv
    f -= np.round(f)
    return np.linalg.norm(f @ cell, axis=-1)          # (na, nb)


def bond_matrix(atoms, sel_a, sel_b, rcut):
    d = min_image_dist(atoms.get_positions()[sel_a],
                       atoms.get_positions()[sel_b], np.array(atoms.get_cell()))
    return (d < rcut)                                  # boolean (na, nb)


def autocorr_from_states(states):
    """states: (nframes, npairs) bool. Returns C(t) = <h(0)h(t)>/<h(0)> averaged
    over time origins (continuous/intermittent definition)."""
    nf = states.shape[0]
    h = states.astype(float)
    h0 = h.mean()
    if h0 == 0:
        return np.zeros(nf)
    C = np.zeros(nf)
    for dt in range(nf):
        C[dt] = np.mean(h[:nf - dt] * h[dt:nf])
    return C / h0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traj")
    ap.add_argument("--mode", default="ps4", choices=["ps4", "li"])
    ap.add_argument("--rcut", type=float, default=2.6, help="bond cutoff [A]")
    ap.add_argument("--dt", type=float, default=0.05, help="time between frames [ps]")
    ap.add_argument("--index", default="::1", help="ASE slice, e.g. '::2'")
    ap.add_argument("--A", default=None, help="central species (default P for ps4, Li for li)")
    ap.add_argument("--B", default="S", help="neighbour species (default S)")
    ap.add_argument("--out", default="lifetime")
    args = ap.parse_args()

    frames = read(args.traj, index=args.index)
    if not isinstance(frames, list):
        frames = [frames]
    print("loaded %d frames" % len(frames))

    a_sym = args.A or ("P" if args.mode == "ps4" else "Li")
    sym = np.asarray(frames[0].get_chemical_symbols())
    selA = np.where(sym == a_sym)[0]
    selB = np.where(sym == args.B)[0]

    # fix the bonded pair set from frame 0 (the PS4 tetrahedron / initial Li shell)
    d0 = min_image_dist(frames[0].get_positions()[selA],
                        frames[0].get_positions()[selB], np.array(frames[0].get_cell()))
    pairs = np.argwhere(d0 < args.rcut)                # (npairs, 2) indices into selA, selB

    states = np.zeros((len(frames), len(pairs)), dtype=bool)
    for fi, at in enumerate(frames):
        d = min_image_dist(at.get_positions()[selA], at.get_positions()[selB],
                           np.array(at.get_cell()))
        states[fi] = d[pairs[:, 0], pairs[:, 1]] < args.rcut

    C = autocorr_from_states(states)
    t = np.arange(len(C)) * args.dt
    trapz = getattr(np, "trapezoid", np.trapz)
    tau_int = trapz(C, t)                              # integral lifetime [ps]
    below = np.where(C < 1.0 / np.e)[0]
    tau_e = t[below[0]] if len(below) else float("nan")

    np.savetxt(args.out + ".csv", np.column_stack([t, C]),
               delimiter=",", header="time_ps,C", comments="")
    print("== %s bond lifetime (%s-%s, rcut=%.2f A) ==" % (args.mode, a_sym, args.B, args.rcut))
    print("  tracked %d bonds over %d frames" % (len(pairs), len(frames)))
    print("  1/e lifetime      = %8.3f ps" % tau_e)
    print("  integral lifetime = %8.3f ps" % tau_int)
    print("  surviving fraction at end = %.3f" % C[-1])
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 4))
        plt.plot(t, C)
        plt.axhline(1 / np.e, ls="--", c="k", lw=0.8)
        plt.xlabel("time [ps]"); plt.ylabel("C(t)")
        plt.title("%s-%s bond autocorrelation" % (a_sym, args.B))
        plt.tight_layout(); plt.savefig(args.out + ".png", dpi=150)
        print("  wrote %s.png / %s.csv" % (args.out, args.out))
    except Exception:
        pass


if __name__ == "__main__":
    main()

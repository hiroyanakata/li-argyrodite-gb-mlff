#!/usr/bin/env python
# =============================================================================
# msd_regions.py -- region-resolved Li diffusion, done properly, from the
#                   unwrapped trajectories saved by diffusion_md.py.
#
# Why this exists
# ---------------
# diffusion_md.py's built-in region MSD labels each ion ONCE, at t0, and
# msd_arrhenius.py fits the SECOND HALF of the curve. For a 7 A-wide grain
# boundary window that combination measures the wrong thing at long times:
# with D from the previous campaign the rms displacement reaches 7 A at
# ~95 ps (600 K), ~48 ps (700 K), ~29 ps (800 K), so by the second half of a
# 100 ps run the "GB" ions are no longer in the GB and every region converges
# to the same D. That is a plausible cause of the earlier "GB vs BULK
# difference flips sign between models" (RESULTS_prodBench.md section D).
#
# What this does instead
# ----------------------
#   1. residence(t): fraction of each region's members still inside its window.
#      -> tells you, from the data, how long the region label is meaningful.
#   2. multi-origin region MSD: average over many time origins and RE-ASSIGN the
#      region at each origin, over a short lag window. This is a local
#      diffusivity, not "where the ion started an hour ago".
#   3. D(x) profile: local D binned along the GB normal, no hand-set regions.
#
# Usage
#   python msd_regions.py msdBIG_phaseB_700K_s*_traj.npz --axis x \
#          --regions "GB1:13.86:20.86 BULK:30:40 GB2:48.58:55.58" \
#          --lag 20 --out reg_phaseB_700K
# Seeds given together are pooled (that is what the error bars come from).
# =============================================================================
import argparse
import json

import numpy as np

AXIS = {"x": 0, "y": 1, "z": 2}


def load(paths):
    """Concatenate several seeds. Each stays a separate trajectory."""
    runs = []
    for p in paths:
        z = np.load(p)
        runs.append({"pos": z["pos"], "t": z["t_ps"], "cell": z["cell"], "file": p})
    return runs


def wrapped_coord(pos, cell, ax):
    """Position folded back into the cell along one axis. Assumes an
    orthorhombic, fixed cell (true for these NVT runs)."""
    L = cell[ax, ax]
    return np.mod(pos[..., ax], L)


def residence(run, regions, ax):
    """Fraction of the ions that were in each window at t0 and are still in it."""
    pos, t, cell = run["pos"], run["t"], run["cell"]
    w0 = wrapped_coord(pos[0], cell, ax)
    out = {}
    for name, lo, hi in regions:
        m0 = (w0 >= lo) & (w0 <= hi)
        if not m0.any():
            continue
        w = wrapped_coord(pos[:, m0, :], cell, ax)
        out[name] = ((w >= lo) & (w <= hi)).mean(axis=1)
    return t, out


def multi_origin_msd(runs, regions, ax, lag_ps, stride):
    """MSD(tau) for tau in [0, lag_ps], averaged over time origins, with the
    region membership decided AT EACH ORIGIN."""
    acc = {}
    for run in runs:
        pos, t, cell = run["pos"], run["t"], run["cell"]
        dt = t[1] - t[0]
        nlag = int(round(lag_ps / dt))
        if nlag < 4:
            raise SystemExit("--lag %g ps is only %d frames; lower --lag or raise "
                             "--traj_every resolution" % (lag_ps, nlag))
        for o in range(0, len(t) - nlag, stride):
            w = wrapped_coord(pos[o], cell, ax)
            d = pos[o:o + nlag + 1] - pos[o]                  # (nlag+1, nLi, 3)
            sq = (d ** 2).sum(axis=2)                          # (nlag+1, nLi)
            for name, lo, hi in [("ALL", -1e9, 1e9)] + list(regions):
                m = (w >= lo) & (w <= hi)
                if not m.any():
                    continue
                acc.setdefault(name, []).append(sq[:, m].mean(axis=1))
    tau = np.arange(nlag + 1) * (runs[0]["t"][1] - runs[0]["t"][0])
    return tau, {k: np.array(v) for k, v in acc.items()}


def fit_D(tau, msd, tmin_frac=0.4):
    """Einstein D from the tail of the lag window (skips the ballistic toe)."""
    m = tau >= tmin_frac * tau.max()
    slope = np.polyfit(tau[m], msd[m], 1)[0]      # A^2/ps
    return slope / 6.0 * 1e-4                     # cm^2/s


def d_profile(runs, ax, lag_ps, nbin, stride):
    """Local D along the GB normal: bin each ion by where it IS at the origin."""
    edges = np.linspace(0, runs[0]["cell"][ax, ax], nbin + 1)
    num = np.zeros(nbin)
    cnt = np.zeros(nbin)
    for run in runs:
        pos, t, cell = run["pos"], run["t"], run["cell"]
        dt = t[1] - t[0]
        nlag = int(round(lag_ps / dt))
        for o in range(0, len(t) - nlag, stride):
            w = wrapped_coord(pos[o], cell, ax)
            sq = ((pos[o + nlag] - pos[o]) ** 2).sum(axis=1)   # (nLi,)
            b = np.clip(np.digitize(w, edges) - 1, 0, nbin - 1)
            np.add.at(num, b, sq)
            np.add.at(cnt, b, 1.0)
    centres = 0.5 * (edges[1:] + edges[:-1])
    with np.errstate(invalid="ignore", divide="ignore"):
        msd = num / cnt
    return centres, msd / (6.0 * lag_ps) * 1e-4, cnt          # D [cm^2/s]


def main():
    ap = argparse.ArgumentParser(description="region-resolved D from saved trajectories")
    ap.add_argument("traj", nargs="+", help="*_traj.npz from diffusion_md.py (seeds pooled)")
    ap.add_argument("--axis", default="x", help="GB normal")
    ap.add_argument("--regions", default="GB1:13.86:20.86 BULK:30:40 GB2:48.58:55.58")
    ap.add_argument("--lag", type=float, default=20.0,
                    help="lag window [ps]. Keep it SHORT enough that ions stay in "
                         "their region -- check the residence table this prints.")
    ap.add_argument("--stride", type=int, default=5, help="time-origin stride [frames]")
    ap.add_argument("--nbin", type=int, default=40, help="bins for the D(x) profile")
    ap.add_argument("--out", default="regions")
    args = ap.parse_args()

    ax = AXIS[args.axis]
    regions = []
    for tok in args.regions.split():
        name, lo, hi = tok.split(":")
        regions.append((name, float(lo), float(hi)))

    runs = load(args.traj)
    print("loaded %d trajectories, %d frames each, %d ions, cell %s"
          % (len(runs), len(runs[0]["t"]), runs[0]["pos"].shape[1],
             np.diag(runs[0]["cell"]).round(2)))

    # 1. residence -- how long does the region label mean anything?
    t, res = residence(runs[0], regions, ax)
    print("\n== residence: fraction of each region's t0 members still inside ==")
    print("%8s %s" % ("t [ps]", " ".join("%8s" % n for n in res)))
    for frac in (0.1, 0.25, 0.5, 1.0):
        i = min(int(frac * (len(t) - 1)), len(t) - 1)
        print("%8.1f %s" % (t[i], " ".join("%8.2f" % res[n][i] for n in res)))
    half = {}
    for n, r in res.items():
        below = np.where(r < 0.5)[0]
        half[n] = float(t[below[0]]) if len(below) else float("inf")
    print("time for residence to fall below 0.5:",
          {k: ("%.1f ps" % v if np.isfinite(v) else ">run") for k, v in half.items()})
    print("-> region-resolved fits should stay well inside these times.")

    # 2. multi-origin region MSD over a short lag
    tau, acc = multi_origin_msd(runs, regions, ax, args.lag, args.stride)
    print("\n== multi-origin region MSD (lag %.1f ps, region re-assigned at every origin) =="
          % args.lag)
    print("%-6s %10s %12s %12s" % ("region", "n_origins", "D [cm^2/s]", "D +- (seed/origin)"))
    summary = {}
    for name, curves in acc.items():
        mean = curves.mean(axis=0)
        D = fit_D(tau, mean)
        # spread over origins, as a crude uncertainty
        Ds = np.array([fit_D(tau, c) for c in curves])
        summary[name] = {"D_cm2_s": D, "D_sem": float(Ds.std() / np.sqrt(len(Ds))),
                         "n_origins": int(len(curves))}
        print("%-6s %10d %12.4e %12.2e" % (name, len(curves), D, summary[name]["D_sem"]))
    if "BULK" in summary:
        for name in summary:
            if name not in ("BULK", "ALL"):
                print("   %s / BULK = %.2f" % (name, summary[name]["D_cm2_s"] / summary["BULK"]["D_cm2_s"]))

    # 3. D(x) profile -- no hand-set windows at all
    centres, Dx, cnt = d_profile(runs, ax, args.lag, args.nbin, args.stride)
    np.savetxt(args.out + "_Dprofile.csv",
               np.column_stack([centres, Dx, cnt]), delimiter=",",
               header="%s_A,D_cm2_s,samples" % args.axis, comments="")
    np.savetxt(args.out + "_msd_tau.csv",
               np.column_stack([tau] + [acc[k].mean(axis=0) for k in acc]),
               delimiter=",", header="tau_ps," + ",".join(acc), comments="")
    json.dump({"regions": summary, "lag_ps": args.lag,
               "residence_half_ps": half}, open(args.out + ".json", "w"), indent=1)
    print("\nwrote %s.json, %s_Dprofile.csv, %s_msd_tau.csv"
          % (args.out, args.out, args.out))


if __name__ == "__main__":
    main()

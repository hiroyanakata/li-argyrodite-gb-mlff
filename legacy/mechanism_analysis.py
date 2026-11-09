#!/usr/bin/env python
# =============================================================================
# mechanism_analysis.py -- why does Li diffusion differ GB vs bulk, and why do
# pretrained and phaseB disagree?  Uses only the Li-atom trajectories saved by
# diffusion_md.py (*_traj.npz: pos[frame,Li,3] unwrapped, t_ps, cell).
#
# Per region (GB = pooled GB1+GB2 windows, BULK = centre window), seeds pooled:
#   1. rho_Li(x)         -- Li number-density along the GB normal (structure)
#   2. cage plateau      -- short-time MSD; a plateau = Li trapped in a cage
#   3. alpha2(t)         -- non-Gaussian parameter; peak = hop-in-cage heterogeneity
#   4. van Hove Gs(r,t*) -- self part at fixed lag: jump-length distribution
#   5. residence/escape  -- survival of region membership, mean escape time
#
# Usage:
#   python mechanism_analysis.py --model pretrained --T 700 --dir results_bigGB_20260903
# writes mech_<model>_<T>K.{json,npz} and returns a summary dict.
# =============================================================================
import argparse, glob, json
import numpy as np

AX = 0  # GB normal = x


def load(model, T, d):
    runs = []
    for p in sorted(glob.glob(f"{d}/msdBIG_{model}_{T}K_s*_traj.npz")):
        z = np.load(p)
        runs.append({"pos": z["pos"], "t": z["t_ps"], "cell": z["cell"]})
    if not runs:
        raise SystemExit(f"no traj for {model} {T}")
    return runs


def wrap_x(pos, L):
    return np.mod(pos[..., AX], L)


def region_mask(wx, Lx, gb_half=3.5, bulk_half=5.0):
    gb1, gb2, bk = 0.25 * Lx, 0.75 * Lx, 0.50 * Lx
    gb = (np.abs(wx - gb1) <= gb_half) | (np.abs(wx - gb2) <= gb_half)
    bulk = np.abs(wx - bk) <= bulk_half
    return gb, bulk


def msd_curve(runs, mask_fn, nlag, stride):
    """Multi-origin MSD(tau), region decided at each origin."""
    acc = []
    for r in runs:
        pos, cell = r["pos"], r["cell"]
        Lx = cell[AX, AX]
        for o in range(0, len(pos) - nlag, stride):
            wx = wrap_x(pos[o], Lx)
            m = mask_fn(wx)
            if not m.any():
                continue
            d = pos[o:o + nlag + 1, m, :] - pos[o, m, :]
            acc.append((d ** 2).sum(-1).mean(1))          # (nlag+1,)
    return np.array(acc)                                    # (norigin, nlag+1)


def alpha2_curve(runs, mask_fn, nlag, stride):
    """Non-Gaussian parameter a2(t) = 3<r^4>/(5<r^2>^2) - 1, per region."""
    r2 = []
    r4 = []
    for r in runs:
        pos, cell = r["pos"], r["cell"]
        Lx = cell[AX, AX]
        for o in range(0, len(pos) - nlag, stride):
            wx = wrap_x(pos[o], Lx)
            m = mask_fn(wx)
            if not m.any():
                continue
            d = pos[o:o + nlag + 1, m, :] - pos[o, m, :]
            sq = (d ** 2).sum(-1)                            # (nlag+1, nsel)
            r2.append(sq.mean(1))
            r4.append((sq ** 2).mean(1))
    r2 = np.array(r2).mean(0)
    r4 = np.array(r4).mean(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        a2 = 3.0 * r4 / (5.0 * r2 ** 2) - 1.0
    a2[0] = 0.0
    return a2


def van_hove_self(runs, mask_fn, nlag, stride, rmax=8.0, nbin=80):
    """Gs(r, t*) at lag = nlag frames: distribution of single-Li displacement."""
    edges = np.linspace(0, rmax, nbin + 1)
    h = np.zeros(nbin)
    for r in runs:
        pos, cell = r["pos"], r["cell"]
        Lx = cell[AX, AX]
        for o in range(0, len(pos) - nlag, stride):
            wx = wrap_x(pos[o], Lx)
            m = mask_fn(wx)
            if not m.any():
                continue
            disp = np.linalg.norm(pos[o + nlag, m, :] - pos[o, m, :], axis=1)
            h += np.histogram(disp, bins=edges)[0]
    cen = 0.5 * (edges[1:] + edges[:-1])
    # 4 pi r^2 normalisation removed: report raw P(r) (prob density of |disp|)
    P = h / max(h.sum(), 1) / (edges[1] - edges[0])
    return cen, P


def residence(runs, mask_fn):
    """Fraction of t0 members still inside, averaged over runs and origins."""
    surv = None
    cnt = 0
    for r in runs:
        pos, cell = r["pos"], r["cell"]
        Lx = cell[AX, AX]
        wx0 = wrap_x(pos[0], Lx)
        m0 = mask_fn(wx0)
        if not m0.any():
            continue
        wx = wrap_x(pos[:, m0, :], Lx)
        inside = mask_fn(wx)                                 # (frames, nsel)
        s = inside.mean(1)
        surv = s if surv is None else surv + s
        cnt += 1
    return surv / cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--T", type=int, required=True)
    ap.add_argument("--dir", default="results_bigGB_20260903")
    ap.add_argument("--lag_ps", type=float, default=15.0)
    ap.add_argument("--vhlag_ps", type=float, default=8.0)
    ap.add_argument("--stride", type=int, default=5)
    args = ap.parse_args()

    runs = load(args.model, args.T, args.dir)
    dt = runs[0]["t"][1] - runs[0]["t"][0]
    Lx = runs[0]["cell"][AX, AX]
    nlag = int(round(args.lag_ps / dt))
    vhlag = int(round(args.vhlag_ps / dt))
    tau = np.arange(nlag + 1) * dt

    gb_fn = lambda wx: region_mask(wx, Lx)[0]
    bk_fn = lambda wx: region_mask(wx, Lx)[1]

    out = {"model": args.model, "T": args.T, "dt_ps": float(dt), "Lx": float(Lx)}
    npz = {}

    # 1. density profile
    edges = np.linspace(0, Lx, 71)
    hist = np.zeros(70)
    for r in runs:
        wx = wrap_x(r["pos"], Lx).ravel()
        hist += np.histogram(wx, bins=edges)[0]
    cen = 0.5 * (edges[1:] + edges[:-1])
    rho = hist / hist.mean()
    npz["rhox_x"] = cen
    npz["rhox"] = rho

    for label, fn in [("GB", gb_fn), ("BULK", bk_fn)]:
        msd = msd_curve(runs, fn, nlag, args.stride).mean(0)
        a2 = alpha2_curve(runs, fn, nlag, args.stride)
        vhr, vhP = van_hove_self(runs, fn, vhlag, args.stride)
        surv = residence(runs, fn)
        # cage metric: MSD at 1 ps (rattling), and D from tail
        i1 = int(round(1.0 / dt))
        cage = float(msd[i1])
        Dtail = np.polyfit(tau[tau > 0.4 * tau.max()],
                           msd[tau > 0.4 * tau.max()], 1)[0] / 6.0 * 1e-4
        a2max = float(np.nanmax(a2))
        a2tmax = float(tau[np.nanargmax(a2)])
        # residence half time
        tfull = np.arange(len(surv)) * dt
        bl = np.where(surv < 0.5)[0]
        thalf = float(tfull[bl[0]]) if len(bl) else float("inf")
        out[label] = {"cage_msd_1ps_A2": cage, "D_cm2_s": float(Dtail),
                      "alpha2_max": a2max, "alpha2_tmax_ps": a2tmax,
                      "residence_half_ps": thalf}
        npz[f"{label}_tau"] = tau
        npz[f"{label}_msd"] = msd
        npz[f"{label}_alpha2"] = a2
        npz[f"{label}_vh_r"] = vhr
        npz[f"{label}_vh_P"] = vhP
        npz[f"{label}_surv"] = surv
        npz[f"{label}_surv_t"] = tfull

    base = f"{args.dir}/mech_{args.model}_{args.T}K"
    np.savez(base + ".npz", **npz)
    json.dump(out, open(base + ".json", "w"), indent=1)
    print(json.dumps(out, indent=1))
    print("wrote", base + ".json /.npz")


if __name__ == "__main__":
    main()

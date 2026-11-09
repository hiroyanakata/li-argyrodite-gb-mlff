#!/usr/bin/env python
# =============================================================================
# mechanism_tier1.py -- Tier-1 Li dynamical mechanism metrics, GB vs bulk, from
# the Li-only diffusion trajectories (diffusion_md.py *_traj.npz).
#
# Region membership is FIXED at t0 (g_GB1/g_GB2/g_BULK index arrays stored by
# diffusion_md.py); PBC-unwrapped, multi-time-origin (stride 1 ps) displacements.
# Per region: MSD(tau), cage MSD at 1 ps, non-Gaussian parameter alpha2 peak,
# median displacement and fraction > 2 A at the lag. Seeds are averaged
# (mean +- s.e.m.); per-seed values are kept.
#
#   python mechanism_tier1.py --dir . --struct uni --temps 500 700 --seeds 0 1 2 \
#          --lag 20 --out mechanism_tier1_3seed_lag20.json
# reads <dir>/msdV2<struct>_<model>_<T>K_s<seed>_traj.npz for model in v5D, pretrained.
# =============================================================================
import argparse, json
import numpy as np


def load(fn):
    d = np.load(fn)
    pos = d["pos"]; t = d["t_ps"]; cell = d["cell"]
    L = np.diag(cell).astype(float)
    dsteps = np.diff(pos, axis=0)
    dsteps -= L * np.round(dsteps / L)                 # minimum-image unwrap
    unw = np.concatenate([pos[:1], pos[:1] + np.cumsum(dsteps, axis=0)], axis=0)
    reg = {k: d[k].astype(int) for k in ("g_GB1", "g_GB2", "g_BULK")}
    return unw, t, np.concatenate([reg["g_GB1"], reg["g_GB2"]]), reg["g_BULK"]


def metrics(unw, t, sel, lag_ps, orig_stride_ps=1.0):
    dt = np.median(np.diff(t))
    lag = int(round(lag_ps/dt)); stride = max(1, int(round(orig_stride_ps/dt)))
    n = unw.shape[0]
    origins = range(0, n-lag, stride)
    msd_t = np.zeros(lag+1); cnt = 0
    r2_at = []; r4_at = []  # for alpha2 at each sub-lag (use full curve)
    # accumulate per-lag <r2>,<r4> for alpha2(t)
    sum_r2 = np.zeros(lag+1); sum_r4 = np.zeros(lag+1)
    disp_final = []  # |dr| at lag
    for o in origins:
        d = unw[o:o+lag+1, sel, :] - unw[o, sel, :][None]
        r2 = (d**2).sum(-1)            # (lag+1, nsel)
        sum_r2 += r2.mean(1); sum_r4 += (r2**2).mean(1)
        cnt += 1
        disp_final.append(np.sqrt(r2[-1]))
    sum_r2 /= cnt; sum_r4 /= cnt
    alpha2 = 3*sum_r4/(5*sum_r2**2) - 1
    disp_final = np.concatenate(disp_final)
    tt = np.arange(lag+1)*dt
    # cage MSD at 1 ps
    i1 = int(round(1.0/dt))
    return dict(
        t=tt.tolist(), msd=sum_r2.tolist(), alpha2=alpha2.tolist(),
        cage_msd_1ps=float(sum_r2[i1]),
        alpha2_peak=float(np.nanmax(alpha2[1:])),
        msd_lag=float(sum_r2[-1]),
        median_disp=float(np.median(disp_final)),
        frac_gt2=float((disp_final > 2.0).mean()),
        n=len(sel),
    )


KEYS = ("cage_msd_1ps", "alpha2_peak", "median_disp", "frac_gt2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".")
    ap.add_argument("--struct", default="uni")
    ap.add_argument("--models", nargs="+", default=["v5D", "pretrained"])
    ap.add_argument("--temps", nargs="+", type=int, default=[500, 700])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--lag", type=float, default=20.0, help="lag [ps]")
    ap.add_argument("--out", default="mechanism_tier1_3seed_lag20.json")
    a = ap.parse_args()
    out = {}
    for T in a.temps:
        for model in a.models:
            acc = {"GB": [], "BULK": []}
            for s in a.seeds:
                unw, t, gb, bulk = load(f"{a.dir}/msdV2{a.struct}_{model}_{T}K_s{s}_traj.npz")
                for rn, sel in (("GB", gb), ("BULK", bulk)):
                    acc[rn].append(metrics(unw, t, sel, lag_ps=a.lag))
            n = len(a.seeds)
            for rn, L in acc.items():
                d = {q: [float(np.mean([m[q] for m in L])),
                         float(np.std([m[q] for m in L], ddof=1) / n ** .5) if n > 1 else float("nan"),
                         [round(m[q], 3) for m in L]] for q in KEYS}
                d["t"] = L[0]["t"]
                d["msd"] = np.mean([m["msd"] for m in L], 0).tolist()
                d["alpha2"] = np.mean([m["alpha2"] for m in L], 0).tolist()
                out[f"{T}K_{model}__{rn}"] = d
                print(T, model, rn, " ".join(f"{q}={d[q][0]:.2f}+-{d[q][1]:.2f}" for q in KEYS), flush=True)
    json.dump(out, open(a.out, "w"))
    print("wrote", a.out)


if __name__ == "__main__":
    main()

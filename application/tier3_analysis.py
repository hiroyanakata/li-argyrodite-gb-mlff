#!/usr/bin/env python
# =============================================================================
# tier3_analysis.py -- Tier-3: framework dynamics vs Li mobility, GB vs bulk.
#
# Consumes the ALL-atom short runs from framework_md.py
# (fw_<model>_<T>K_fwtraj.npz: raw positions, every 0.05 ps, 40 ps, base 1382 cell).
#
# Tier1 (Li-only) showed phaseB freezes Li in the halogen-rich GB while
# pretrained makes it a fast slab.  The remaining question is whether the anion
# FRAMEWORK behaves the same way, i.e. whether the disagreement is a
# paddlewheel / soft-sublattice effect or a Li-only one.
#
# Metrics (multi-time-origin, region re-assigned at every origin -- same
# convention as msd_regions.py so the numbers are comparable to section A):
#   1. PS4 reorientation C1(tau) = <u_PS(t0+tau).u_PS(t0)>   (paddlewheel probe)
#   2. halide (Cl+Br) MSD(tau), GB vs bulk                   (mobile sublattice)
#   3. Li MSD(tau), GB vs bulk                               (internal check)
#   4. region census (how many Li / P / S / Cl / Br sit in each window)
#
#   python tier3_analysis.py --dir results_bigGB_20260903 --out Fig9_tier3
# writes <out>.png and tier3_summary.json
# =============================================================================
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PS_BOND = 2.3                 # P-S bond cutoff [A]
GB_HALF, BULK_HALF = 3.5, 5.0 # same windows as Tier1/Tier2
AX = 0                        # GB normal = x
MODELS = ["pretrained", "phaseB"]
TEMPS = [600, 700, 800]
COL = {"pretrained": "tab:blue", "phaseB": "tab:red"}
NAME = {"pretrained": "generic", "phaseB": "phaseB"}   # legend names as in the paper


def load(path):
    z = np.load(path, allow_pickle=True)
    return (z["pos"].astype(np.float64), np.asarray(z["t_ps"], float),
            np.diag(np.array(z["cell"])), np.array([str(s) for s in z["symbols"]]))


def mic(d, L):
    return d - L * np.round(d / L)


def unwrap(pos, L):
    """min-image unwrap along the trajectory (dump interval small -> step < L/2)."""
    out = np.empty_like(pos)
    out[0] = pos[0]
    out[1:] = pos[0] + np.cumsum(mic(np.diff(pos, axis=0), L), axis=0)
    return out


def region_of(x, Lx):
    wx = np.mod(x, Lx)
    gb = (np.abs(wx - 0.25 * Lx) <= GB_HALF) | (np.abs(wx - 0.75 * Lx) <= GB_HALF)
    bulk = np.abs(wx - 0.50 * Lx) <= BULK_HALF
    return gb, bulk


def msd_multiorigin(pos_uw, pos_raw, L, nlag, stride):
    """MSD(tau) per region, region membership decided at EACH origin."""
    Lx = L[AX]
    acc = {"GB": [], "BULK": []}
    nmem = {"GB": [], "BULK": []}
    for o in range(0, len(pos_uw) - nlag, stride):
        gb, bulk = region_of(pos_raw[o, :, AX], Lx)
        for name, m in (("GB", gb), ("BULK", bulk)):
            if not m.any():
                continue
            d = pos_uw[o:o + nlag + 1, m, :] - pos_uw[o, m, :]
            acc[name].append((d ** 2).sum(-1).mean(1))
            nmem[name].append(int(m.sum()))
    out = {}
    for name in ("GB", "BULK"):
        if not acc[name]:
            out[name] = None
            continue
        c = np.array(acc[name])
        out[name] = {"msd": c.mean(0), "norigin": len(c), "nmean": float(np.mean(nmem[name]))}
    return out


def c1_multiorigin(u, xP, L, nlag, stride):
    """PS4 P->S bond reorientation autocorrelation per region."""
    Lx = L[AX]
    acc = {"GB": [], "BULK": []}
    nb = {"GB": [], "BULK": []}
    for o in range(0, len(u) - nlag, stride):
        gb, bulk = region_of(xP[o], Lx)
        for name, m in (("GB", gb), ("BULK", bulk)):
            if not m.any():
                continue
            c = (u[o:o + nlag + 1, m, :] * u[o, m, :]).sum(-1).mean(1)
            acc[name].append(c)
            nb[name].append(int(m.sum()))
    out = {}
    for name in ("GB", "BULK"):
        if not acc[name]:
            out[name] = None
            continue
        out[name] = {"c1": np.array(acc[name]).mean(0), "norigin": len(acc[name]),
                     "nbond": float(np.mean(nb[name]))}
    return out


def tau_1e(tau, c):
    below = np.where(c <= 1.0 / np.e)[0]
    return float(tau[below[0]]) if len(below) else float("inf")


def dfit(tau, msd, frac=0.5):
    hi = tau > frac * tau.max()
    if hi.sum() < 3:
        return 0.0
    return float(np.polyfit(tau[hi], msd[hi], 1)[0] / 6.0 * 1e-4)   # A^2/ps -> cm^2/s


def analyze(path, lag_ps=10.0, stride_ps=1.0):
    pos, t, L, sym = load(path)
    dt = float(t[1] - t[0])
    nlag, stride = int(round(lag_ps / dt)), max(1, int(round(stride_ps / dt)))
    tau = np.arange(nlag + 1) * dt
    Lx = L[AX]
    r = {"file": os.path.basename(path), "nframe": len(t), "t_end": float(t[-1]),
         "dt_ps": dt, "lag_ps": lag_ps, "tau": tau}

    # --- region census (t=0) -------------------------------------------------
    cens = {}
    for sp in ("Li", "P", "S", "Cl", "Br"):
        idx = np.where(sym == sp)[0]
        gb, bulk = region_of(pos[0, idx, AX], Lx)
        cens[sp] = {"total": int(len(idx)), "GB": int(gb.sum()), "BULK": int(bulk.sum())}
    r["census"] = cens

    # --- 1. PS4 reorientation ------------------------------------------------
    P = np.where(sym == "P")[0]
    S = np.where(sym == "S")[0]
    d0 = np.linalg.norm(mic(pos[0, S][:, None] - pos[0, P][None], L), axis=-1)  # (nS,nP)
    bi, bj = np.where(d0 <= PS_BOND)                    # S index, P index (into S,P)
    pairs = np.stack([P[bj], S[bi]], axis=1)
    pv = mic(pos[:, pairs[:, 1], :] - pos[:, pairs[:, 0], :], L)
    u = pv / np.linalg.norm(pv, axis=-1, keepdims=True)
    r["ps4"] = c1_multiorigin(u, pos[:, pairs[:, 0], AX], L, nlag, stride)
    r["nbond_total"] = int(len(pairs))

    # --- 2. halide (Cl+Br) MSD ----------------------------------------------
    hal = np.concatenate([np.where(sym == "Cl")[0], np.where(sym == "Br")[0]])
    r["halide"] = msd_multiorigin(unwrap(pos[:, hal, :], L), pos[:, hal, :], L, nlag, stride)

    # --- 2b. free-S (non-PS4) MSD -------------------------------------------
    freeS = S[d0.min(1) > PS_BOND]
    r["freeS"] = msd_multiorigin(unwrap(pos[:, freeS, :], L), pos[:, freeS, :], L, nlag, stride)
    r["n_freeS"] = int(len(freeS))

    # --- 3. Li MSD -----------------------------------------------------------
    Li = np.where(sym == "Li")[0]
    r["Li"] = msd_multiorigin(unwrap(pos[:, Li, :], L), pos[:, Li, :], L, nlag, stride)
    return r


def jsonable(r):
    o = {k: v for k, v in r.items() if k not in ("tau", "ps4", "halide", "freeS", "Li")}
    o["ps4"] = {k: (None if v is None else
                    {"tau_1e_ps": tau_1e(r["tau"], v["c1"]), "c1_end": float(v["c1"][-1]),
                     "nbond": v["nbond"], "norigin": v["norigin"]})
                for k, v in r["ps4"].items()}
    for key in ("halide", "freeS", "Li"):
        o[key] = {k: (None if v is None else
                      {"msd_lag_A2": float(v["msd"][-1]), "D_cm2_s": dfit(r["tau"], v["msd"]),
                       "n_mean": v["nmean"], "norigin": v["norigin"]})
                  for k, v in r[key].items()}
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results_bigGB_20260903")
    ap.add_argument("--out", default="results_bigGB_20260903/Fig9_tier3_framework")
    ap.add_argument("--lag", type=float, default=10.0)
    ap.add_argument("--ref_T", type=int, default=700)
    args = ap.parse_args()

    R = {}
    for m in MODELS:
        for T in TEMPS:
            p = f"{args.dir}/fw_{m}_{T}K_fwtraj.npz"
            if not os.path.exists(p):
                print("MISSING", p); continue
            R[(m, T)] = analyze(p, lag_ps=args.lag)
            print("done", m, T, flush=True)

    # ---------------- summary table -----------------
    print("\n%-12s %-4s %-5s | %-12s | %-20s | %-20s" %
          ("model", "T", "reg", "PS4 tau1e", "halide MSD@lag (D)", "Li MSD@lag (D)"))
    print("-" * 92)
    for m in MODELS:
        for T in TEMPS:
            if (m, T) not in R:
                continue
            r = R[(m, T)]
            for reg in ("GB", "BULK"):
                ps, ha, li = r["ps4"][reg], r["halide"][reg], r["Li"][reg]
                ps_s = "n/a" if ps is None else ("%.1f ps (nb %.0f)" % (tau_1e(r["tau"], ps["c1"]), ps["nbond"]))
                ha_s = "n/a" if ha is None else ("%6.2f A2 (%.2e)" % (ha["msd"][-1], dfit(r["tau"], ha["msd"])))
                li_s = "n/a" if li is None else ("%6.2f A2 (%.2e)" % (li["msd"][-1], dfit(r["tau"], li["msd"])))
                print("%-12s %-4d %-5s | %-12s | %-20s | %-20s" % (m, T, reg, ps_s, ha_s, li_s))

    with open(os.path.join(args.dir, "tier3_summary.json"), "w") as fh:
        json.dump({f"{m}_{T}K": jsonable(r) for (m, T), r in R.items()}, fh, indent=1)

    # ---------------- figure -----------------
    T0 = args.ref_T
    fig, ax = plt.subplots(2, 2, figsize=(10.0, 7.4))

    # (a) PS4 reorientation: bulk (rigid in both) vs the few GB-edge tetrahedra
    a = ax[0, 0]
    for m in MODELS:
        if (m, T0) not in R:
            continue
        r = R[(m, T0)]
        for reg, ls, lw in (("BULK", "-", 2.0), ("GB", "--", 1.3)):
            v = r["ps4"][reg]
            if v is None:
                continue
            a.plot(r["tau"], v["c1"], ls, color=COL[m], lw=lw,
                   label="%s %s (n=%.0f)" % (NAME[m], reg, v["nbond"]))
    a.axhline(1 / np.e, color="k", lw=0.8, ls="-.")
    a.text(0.2, 1 / np.e + 0.03, "1/e", fontsize=8)
    a.set_xlabel(r"$\tau$ [ps]"); a.set_ylabel(r"$C_1(\tau)$   PS$_4$ P–S bond")
    a.set_ylim(0, 1.05); a.legend(fontsize=7.5, loc="lower left")
    a.set_title("(a) PS$_4$ reorientation, %d K: bulk framework rigid\n"
                "in both models ($C_1>0.97$ at 10 ps) $\\Rightarrow$ no paddlewheel" % T0,
                fontsize=9)

    # (b) halide MSD at the GB
    b = ax[0, 1]
    for m in MODELS:
        for T, ls in zip(TEMPS, [":", "-", "--"]):
            if (m, T) not in R:
                continue
            r = R[(m, T)]
            v = r["halide"]["GB"]
            if v is None:
                continue
            b.plot(r["tau"], v["msd"], ls, color=COL[m], lw=1.8 if T == T0 else 1.1,
                   label=f"{NAME[m]} {T} K")
    b.set_xlabel(r"$\tau$ [ps]"); b.set_ylabel(r"halide MSD [$\mathrm{\AA}^2$]")
    b.legend(fontsize=7, ncol=2, loc="upper left")
    b.set_title("(b) Cl+Br sublattice inside the GB slab\n(no halide sits in the bulk window)", fontsize=9)

    # (c) Li MSD, GB vs bulk, at T0
    c = ax[1, 0]
    for m in MODELS:
        if (m, T0) not in R:
            continue
        r = R[(m, T0)]
        for reg, ls in (("GB", "-"), ("BULK", "--")):
            v = r["Li"][reg]
            if v is None:
                continue
            c.plot(r["tau"], v["msd"], ls, color=COL[m], lw=1.7, label=f"{NAME[m]} {reg}")
    c.set_xlabel(r"$\tau$ [ps]"); c.set_ylabel(r"Li MSD [$\mathrm{\AA}^2$]")
    c.legend(fontsize=8, loc="upper left")
    c.set_title(f"(c) Li, {T0} K (cf. main-text Fig. 6)", fontsize=9)

    # (d) census: what is actually in the GB window
    d = ax[1, 1]
    sp = ["Li", "P", "S", "Cl", "Br"]
    xg = np.arange(len(sp))
    w = 0.2
    for k, m in enumerate(MODELS):
        if (m, T0) not in R:
            continue
        cen = R[(m, T0)]["census"]
        d.bar(xg + (2 * k - 1.5) * w, [cen[s_]["GB"] for s_ in sp], w, color=COL[m],
              alpha=0.9, label=f"{NAME[m]} GB")
        d.bar(xg + (2 * k - 0.5) * w, [cen[s_]["BULK"] for s_ in sp], w, facecolor="none",
              edgecolor=COL[m], hatch="///", lw=1.0, label=f"{NAME[m]} bulk")
    d.set_xticks(xg); d.set_xticklabels(sp)
    d.set_ylabel("atoms in window")
    d.legend(fontsize=7, ncol=2)
    d.set_title("(d) window census at %d K: the GB core is a halide slab\n"
                "(S nearly absent, no Cl/Br in the bulk window); phaseB expels Li from it" % T0,
                fontsize=9)

    fig.tight_layout()
    fig.savefig(args.out + ".png", dpi=180)
    print("\nwrote", args.out + ".png")
    print("wrote", os.path.join(args.dir, "tier3_summary.json"))


if __name__ == "__main__":
    main()

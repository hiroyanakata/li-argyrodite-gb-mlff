#!/usr/bin/env python
# =============================================================================
# nemd_analysis.py -- turn nemd_ccd.py output into sigma, sigma(x), the Haven
#                     ratio, the linear-response check, Ea, and R_GB.
#
# Everything here reads the *_nemd.npz written by nemd_ccd.py.  Runs given
# together on one command line are pooled (seeds) or swept (fields,
# temperatures), depending on the mode.
#
# Modes
# -----
#   sigma      one number per run: <Jc> -> sigma, with block error bars.
#   profile    local sigma(x) and n_Li(x) along the GB normal.  Use the (A)
#              geometry (field parallel to the GB plane) -- in the (B) geometry
#              continuity makes Jc_k uniform by construction and the profile is
#              a steady-state check, not a local conductivity.
#   linear     sigma vs Fe.  The whole method is a linear-response measurement,
#              so this plot is what licenses the field you ran at.
#   haven      sigma_dilute (from the drift-corrected self-MSD of the same run)
#              vs sigma_NEMD -> H_R = sigma_dilute/sigma.  H_R < 1 means the
#              carriers move cooperatively and Nernst-Einstein underestimates.
#   arrhenius  ln(sigma*T) vs 1/T -> Ea.
#   rgb        (B): combine a perpendicular run (total, blocked) with the bulk
#              plateau of a parallel run to get the specific GB resistance
#                  R_GB = ( L/sigma_perp - L/sigma_bulk ) / n_gb   [ohm cm^2]
#              Cubic argyrodite bulk is isotropic, so the bulk plateau measured
#              in-plane is a valid reference for the cross-plane direction;
#              run the parallel geometry along both y and z to check that.
#
# Examples
#   python nemd_analysis.py sigma    nemd_phaseB_700K_z_F020_s*_nemd.npz
#   python nemd_analysis.py profile  nemd_phaseB_700K_z_F020_s*_nemd.npz \
#          --gb 17.36 52.08 --out prof_phaseB_700K
#   python nemd_analysis.py linear   nemd_phaseB_700K_z_F*_s*_nemd.npz --out lin_phaseB_700K
#   python nemd_analysis.py haven    nemd_phaseB_700K_z_F020_s*_nemd.npz
#   python nemd_analysis.py arrhenius nemd_phaseB_*K_z_F020_s*_nemd.npz --out arr_phaseB
#   python nemd_analysis.py rgb --perp nemd_phaseB_700K_x_F020_s*_nemd.npz \
#          --par nemd_phaseB_700K_z_F020_s*_nemd.npz --bulk 30 40 --ngb 2
# =============================================================================
import argparse
import glob
import json

import numpy as np

SIGMA_CONV = 16.02176634          # sigma[S/cm] = 16.0218 * Jc[1/(A^2 ps)] / Fe[eV/A]
KB_J = 1.380649e-23
KB_EV = 8.617333262e-5
E_C = 1.602176634e-19
BURN = 0.2                        # fraction of blocks discarded as transient


# ---------------------------------------------------------------------------
def load(paths):
    runs = []
    for pat in paths:
        for p in sorted(glob.glob(pat)) or [pat]:
            z = np.load(p, allow_pickle=False)
            runs.append({
                "file": p, "rows": z["rows"],
                "field": float(z["field"]), "temp": float(z["temp"]),
                "nbins": int(z["nbins"]), "dt_ps": float(z["dt_ps"]),
                "field_axis": int(z["field_axis"]), "profile_axis": int(z["profile_axis"]),
                "cell": z["cell"], "volume": float(z["volume"]),
                "traj": z["traj"], "traj_t": z["traj_t"], "nli": len(z["li_idx"]),
            })
    if not runs:
        raise SystemExit("no runs matched")
    return runs


def steady(rows, burn=BURN):
    """Drop the leading transient; what is left is the steady state."""
    s = max(1, int(burn * len(rows)))
    return rows[s:]


def block_stat(x):
    """Mean and standard error from block averages (the blocks are the samples)."""
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return float(x.mean()) if len(x) else np.nan, np.nan
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))


def sigma_of(run):
    """Global sigma [S/cm] and its error, from <Jc>/Fe."""
    r = steady(run["rows"])
    if run["field"] == 0.0:
        return 0.0, 0.0
    m, e = block_stat(r[:, 1])
    f = run["field"]
    return SIGMA_CONV * m / f, SIGMA_CONV * e / abs(f)


def pool(runs, fn):
    """Pool independent seeds: mean of the per-run values, error from their scatter
    if there are several, else the within-run block error."""
    vals, errs = zip(*[fn(r) for r in runs])
    vals = np.asarray(vals, dtype=float)
    if len(vals) > 1:
        return float(vals.mean()), float(vals.std(ddof=1) / np.sqrt(len(vals)))
    return float(vals[0]), float(errs[0])


# ---------------------------------------------------------------------------
def window_mean(x, s, se, lo, hi):
    """Mean sigma over a window, with an error bar that survives a single seed.

    With one seed the per-bin seed-scatter is zero by construction, so fall back
    to the scatter of the bins inside the window."""
    m = (x >= lo) & (x <= hi)
    n = max(int(m.sum()), 1)
    val = float(s[m].mean())
    err = float(se[m].mean() / np.sqrt(n))
    if not err:
        err = float(s[m].std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return val, err


def local_sigma(runs):
    """sigma(x) [S/cm] and n_Li(x) [1/A^3] per slab, pooled over seeds."""
    nb = runs[0]["nbins"]
    pax = runs[0]["profile_axis"]
    L = float(runs[0]["cell"][pax, pax])
    x = (np.arange(nb) + 0.5) * L / nb
    sig, den = [], []
    for r in runs:
        rows = steady(r["rows"])
        jck = rows[:, 6:6 + nb]
        nk = rows[:, 6 + nb:6 + 2 * nb]
        if r["field"] != 0.0:
            sig.append(SIGMA_CONV * jck.mean(axis=0) / r["field"])
        den.append(nk.mean(axis=0))
    sig = np.array(sig) if sig else np.zeros((0, nb))
    den = np.array(den)
    def ms(a):
        return (a.mean(axis=0),
                a.std(axis=0, ddof=1) / np.sqrt(len(a)) if len(a) > 1 else np.zeros(nb))
    smean, serr = ms(sig) if len(sig) else (np.full(nb, np.nan), np.zeros(nb))
    dmean, derr = ms(den)
    return x, smean, serr, dmean, derr


def self_diffusion_1d(run, burn=BURN):
    """D_self along the field axis [cm^2/s], from the drift-corrected self-MSD.

    Under a field each carrier has a mean drift; subtracting the instantaneous
    Li-averaged displacement leaves the diffusive part, so one run gives both
    sigma (from the drift) and sigma_dilute (from the fluctuation around it).
    Multiple time origins, fit over lags up to a quarter of the run.
    """
    pos, t = run["traj"], run["traj_t"]
    if len(pos) < 8:
        return np.nan
    s = max(1, int(burn * len(pos)))
    pos, t = pos[s:], t[s:]
    ax = run["field_axis"]
    u = pos[:, :, ax].astype(np.float64)              # (nframe, nLi)
    u = u - u.mean(axis=1, keepdims=True)             # remove the drift
    nf = len(u)
    maxlag = max(2, nf // 4)
    lags = np.unique(np.linspace(1, maxlag, min(40, maxlag)).astype(int))
    msd = np.array([np.mean((u[lag:] - u[:-lag]) ** 2) for lag in lags])
    dt = t[1] - t[0]
    slope = np.polyfit(lags * dt, msd, 1)[0]          # A^2/ps
    return slope / 2.0 * 1e-4                         # 1D Einstein -> cm^2/s


def sigma_dilute(run):
    """Nernst-Einstein sigma [S/cm] from the same run's self-diffusion."""
    D = self_diffusion_1d(run)
    n = run["nli"] / run["volume"] * 1e24             # cm^-3
    return n * E_C ** 2 * D / (KB_J * run["temp"])


# ---------------------------------------------------------------------------
def cmd_sigma(runs, args):
    out = []
    for r in runs:
        s, e = sigma_of(r)
        rows = steady(r["rows"])
        out.append(dict(file=r["file"], T=r["temp"], field=r["field"],
                        sigma=s, sigma_err=e,
                        T_par=float(rows[:, 4].mean()), T_perp=float(rows[:, 5].mean()),
                        nblocks=len(rows)))
        print("%-46s T=%4.0f F=%+.4f  sigma = %.4e +- %.1e S/cm   "
              "(T_par %.0f K, T_perp %.0f K, %d blocks)"
              % (r["file"].split("/")[-1], r["temp"], r["field"], s, e,
                 out[-1]["T_par"], out[-1]["T_perp"], len(rows)))
    if len(runs) > 1 and len({(r["temp"], r["field"]) for r in runs}) == 1:
        m, e = pool(runs, sigma_of)
        print("pooled over %d seeds: sigma = %.4e +- %.1e S/cm" % (len(runs), m, e))
        out.append(dict(pooled=True, sigma=m, sigma_err=e, nseed=len(runs)))
    return out


def cmd_profile(runs, args):
    x, s, se, d, de = local_sigma(runs)
    rows = np.column_stack([x, s, se, d, de])
    hdr = "x_A,sigma_S_cm,sigma_err,nLi_A-3,nLi_err"
    np.savetxt(args.out + "_profile.csv", rows, delimiter=",", header=hdr, comments="")
    print("wrote %s_profile.csv" % args.out)
    res = dict(x=x.tolist(), sigma=s.tolist(), sigma_err=se.tolist(),
               nLi=d.tolist(), nLi_err=de.tolist())
    if args.bulk:
        lo, hi = args.bulk
        res["sigma_bulk"], res["sigma_bulk_err"] = window_mean(x, s, se, lo, hi)
        print("bulk window [%.1f, %.1f] A: sigma_bulk = %.4e S/cm"
              % (lo, hi, res["sigma_bulk"]))
    if args.gb:
        for i, g in enumerate(args.gb, 1):
            j = int(np.argmin(np.abs(x - g)))
            print("GB%d at x=%.2f A: sigma = %.4e S/cm, n_Li = %.4f A^-3"
                  % (i, x[j], s[j], d[j]))
            res["GB%d" % i] = dict(x=float(x[j]), sigma=float(s[j]), nLi=float(d[j]))
        if args.bulk and "sigma_bulk" in res:
            for i in range(1, len(args.gb) + 1):
                res["GB%d" % i]["ratio_to_bulk"] = res["GB%d" % i]["sigma"] / res["sigma_bulk"]
                print("   GB%d/bulk = %.3f" % (i, res["GB%d" % i]["ratio_to_bulk"]))
    plot_profile(x, s, se, d, de, args)
    return res


def cmd_linear(runs, args):
    by = {}
    for r in runs:
        by.setdefault(round(r["field"], 6), []).append(r)
    fields, sig, err = [], [], []
    for f in sorted(by):
        if f == 0.0:
            continue
        m, e = pool(by[f], sigma_of)
        fields.append(f); sig.append(m); err.append(e)
        print("Fe = %+.4f eV/A : sigma = %.4e +- %.1e S/cm  (%d seeds)"
              % (f, m, e, len(by[f])))
    fields, sig, err = np.array(fields), np.array(sig), np.array(err)
    res = dict(field=fields.tolist(), sigma=sig.tolist(), sigma_err=err.tolist())
    if len(fields) >= 2:
        # linear response <=> sigma independent of Fe. Fit sigma = a + b*|Fe|.
        w = 1.0 / np.maximum(err, 1e-30) ** 2
        b, a = np.polyfit(np.abs(fields), sig, 1, w=np.sqrt(w))
        res["sigma_F0"] = float(a)
        res["slope"] = float(b)
        print("extrapolated sigma(Fe->0) = %.4e S/cm ; d(sigma)/d|Fe| = %+.3e"
              % (a, b))
        for f, s_, e_ in zip(fields, sig, err):
            dev = (s_ - a) / a * 100
            flag = "OK" if abs(s_ - a) < 2 * max(e_, 0.02 * abs(a)) else "NONLINEAR"
            print("   Fe=%+.4f : %+6.1f%% from the zero-field limit   [%s]" % (f, dev, flag))
    plot_linear(fields, sig, err, res, args)
    return res


def cmd_haven(runs, args):
    out = []
    for r in runs:
        s, e = sigma_of(r)
        sd = sigma_dilute(r)
        hr = sd / s if s else np.nan
        out.append(dict(file=r["file"], T=r["temp"], field=r["field"],
                        sigma=s, sigma_err=e, sigma_dilute=sd, H_R=hr))
        print("%-46s sigma = %.3e  sigma_dilute = %.3e  H_R = %.3f"
              % (r["file"].split("/")[-1], s, sd, hr))
    if len(runs) > 1:
        h = np.array([o["H_R"] for o in out], dtype=float)
        print("pooled H_R = %.3f +- %.3f" % (h.mean(), h.std(ddof=1) / np.sqrt(len(h))))
    return out


def cmd_arrhenius(runs, args):
    by = {}
    for r in runs:
        by.setdefault(round(r["temp"], 1), []).append(r)
    T, sig, err = [], [], []
    for t in sorted(by):
        m, e = pool(by[t], sigma_of)
        T.append(t); sig.append(m); err.append(e)
        print("T = %5.0f K : sigma = %.4e +- %.1e S/cm  (%d seeds)" % (t, m, e, len(by[t])))
    T, sig, err = np.array(T), np.array(sig), np.array(err)
    res = dict(T=T.tolist(), sigma=sig.tolist(), sigma_err=err.tolist())
    if len(T) >= 3:
        # sigma*T convention (Arrhenius for ionic conductors)
        b, a = np.polyfit(1.0 / T, np.log(sig * T), 1)
        res["Ea_eV"] = float(-b * KB_EV)
        res["sigma300"] = float(np.exp(a + b / 300.0) / 300.0)
        # plain sigma: this is the one directly comparable to an Ea fitted to D,
        # which is what Table 2 of the paper quotes. The two differ by ~k_B T.
        b2, _ = np.polyfit(1.0 / T, np.log(sig), 1)
        res["Ea_eV_plain"] = float(-b2 * KB_EV)
        print("Ea = %.3f eV  (from sigma*T)   %.3f eV  (from sigma, comparable "
              "to an Ea fitted to D)" % (res["Ea_eV"], res["Ea_eV_plain"]))
        print("extrapolated sigma(300 K) = %.3e S/cm" % res["sigma300"])
    plot_arrhenius(T, sig, err, res, args)
    return res


def cmd_rgb(args):
    perp = load(args.perp)
    par = load(args.par)
    s_perp, e_perp = pool(perp, sigma_of)
    x, s, se, d, de = local_sigma(par)
    lo, hi = args.bulk
    s_bulk, e_bulk = window_mean(x, s, se, lo, hi)

    pax = perp[0]["profile_axis"]
    L_cm = float(perp[0]["cell"][pax, pax]) * 1e-8
    R_tot = L_cm / s_perp
    R_bulk = L_cm / s_bulk
    R_gb = (R_tot - R_bulk) / args.ngb
    # error propagation on 1/sigma
    dR = L_cm * np.hypot(e_perp / s_perp ** 2, e_bulk / s_bulk ** 2) / args.ngb

    # continuity check: in the perpendicular geometry the steady-state local flux
    # must be the same in every slab. If it is not, the run has not reached steady
    # state and R_GB is not yet meaningful.
    nb = perp[0]["nbins"]
    jck = np.concatenate([steady(r["rows"])[:, 6:6 + nb] for r in perp], axis=0).mean(axis=0)
    spread = jck.std() / abs(jck.mean()) if jck.mean() else np.nan

    print("sigma_perp (total, blocked) = %.4e +- %.1e S/cm" % (s_perp, e_perp))
    print("sigma_bulk (in-plane plateau, x in [%.1f, %.1f]) = %.4e +- %.1e S/cm"
          % (lo, hi, s_bulk, e_bulk))
    print("L = %.2f A ; %d GB planes" % (L_cm * 1e8, args.ngb))
    print("R_total = %.4e ohm cm^2 ; R_bulk-equivalent = %.4e ohm cm^2" % (R_tot, R_bulk))
    print("R_GB(specific) = %.4e +- %.1e ohm cm^2" % (R_gb, dR))
    print("rho_GB / rho_bulk (blocking factor) = %.2f" % (R_tot / R_bulk))
    print("continuity check: slab flux spread = %.1f%% of the mean %s"
          % (100 * spread, "(OK)" if spread < 0.25 else "(NOT steady -- run longer)"))
    return dict(sigma_perp=s_perp, sigma_perp_err=e_perp,
                sigma_bulk=s_bulk, sigma_bulk_err=e_bulk,
                L_cm=L_cm, ngb=args.ngb, R_total=R_tot, R_bulk=R_bulk,
                R_GB=R_gb, R_GB_err=dR, blocking=R_tot / R_bulk,
                continuity_spread=float(spread),
                T=perp[0]["temp"], field=perp[0]["field"])


# ---------------------------------------------------------------------------
def _plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception as e:
        print("[plot skipped]", e)
        return None


def plot_profile(x, s, se, d, de, args):
    plt = _plt()
    if plt is None:
        return
    fig, ax = plt.subplots(2, 1, figsize=(6.5, 6), sharex=True)
    ax[0].errorbar(x, s, yerr=se, marker="o", ms=3, lw=1)
    ax[0].set_ylabel(r"$\sigma(x)$  [S/cm]")
    ax[1].errorbar(x, d, yerr=de, marker="o", ms=3, lw=1, color="C1")
    ax[1].set_ylabel(r"$n_{\rm Li}(x)$  [$\AA^{-3}$]")
    ax[1].set_xlabel(r"$x$ (GB normal) [$\AA$]")
    for g in (args.gb or []):
        for a in ax:
            a.axvline(g, ls="--", c="k", lw=0.8)
    if args.bulk:
        for a in ax:
            a.axvspan(args.bulk[0], args.bulk[1], color="C2", alpha=0.10)
    fig.tight_layout(); fig.savefig(args.out + "_profile.png", dpi=150)
    print("wrote %s_profile.png" % args.out)


def plot_linear(f, s, e, res, args):
    plt = _plt()
    if plt is None or not len(f):
        return
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.errorbar(np.abs(f), s, yerr=e, marker="o", lw=1, ls="none")
    if "sigma_F0" in res:
        xx = np.linspace(0, np.abs(f).max() * 1.05, 20)
        ax.plot(xx, res["sigma_F0"] + res["slope"] * xx, "-", lw=1)
        ax.axhline(res["sigma_F0"], ls=":", c="k", lw=0.8)
    ax.set_xlabel(r"$F_e$  [eV/$\AA$]"); ax.set_ylabel(r"$\sigma$  [S/cm]")
    ax.set_title("linear-response check")
    fig.tight_layout(); fig.savefig(args.out + "_linear.png", dpi=150)
    print("wrote %s_linear.png" % args.out)


def plot_arrhenius(T, s, e, res, args):
    plt = _plt()
    if plt is None or len(T) < 2:
        return
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.errorbar(1000.0 / T, s * T, yerr=e * T, marker="o", lw=1, ls="none")
    if "Ea_eV" in res:
        xx = np.linspace((1000.0 / T).min(), (1000.0 / T).max(), 20)
        b = -res["Ea_eV"] / KB_EV
        a = np.log(s[0] * T[0]) - b / T[0]
        ax.plot(xx, np.exp(a + b * xx / 1000.0), "-", lw=1,
                label=r"$E_a$ = %.3f eV" % res["Ea_eV"])
        ax.legend()
    ax.set_yscale("log")
    ax.set_xlabel(r"1000/$T$  [K$^{-1}$]"); ax.set_ylabel(r"$\sigma T$  [S K/cm]")
    fig.tight_layout(); fig.savefig(args.out + "_arrhenius.png", dpi=150)
    print("wrote %s_arrhenius.png" % args.out)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="analyse CCD-NEMD runs")
    ap.add_argument("mode", choices=["sigma", "profile", "linear", "haven",
                                     "arrhenius", "rgb"])
    ap.add_argument("files", nargs="*", help="*_nemd.npz (globs allowed)")
    ap.add_argument("--out", default="nemd_out")
    ap.add_argument("--gb", type=float, nargs="*", default=None,
                    help="GB plane positions along the profile axis [A]")
    ap.add_argument("--bulk", type=float, nargs=2, default=None,
                    help="bulk window lo hi [A]")
    ap.add_argument("--perp", nargs="*", default=[], help="rgb: perpendicular runs")
    ap.add_argument("--par", nargs="*", default=[], help="rgb: parallel runs")
    ap.add_argument("--ngb", type=int, default=2, help="rgb: number of GB planes in the cell")
    args = ap.parse_args()

    if args.mode == "rgb":
        if not (args.perp and args.par and args.bulk):
            raise SystemExit("rgb needs --perp, --par and --bulk lo hi")
        res = cmd_rgb(args)
    else:
        runs = load(args.files)
        res = {"sigma": cmd_sigma, "profile": cmd_profile, "linear": cmd_linear,
               "haven": cmd_haven, "arrhenius": cmd_arrhenius}[args.mode](runs, args)

    with open(args.out + "_%s.json" % args.mode, "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print("wrote %s_%s.json" % (args.out, args.mode))


if __name__ == "__main__":
    main()

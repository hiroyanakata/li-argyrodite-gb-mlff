#!/usr/bin/env python
# =============================================================================
# kconv_analysis.py -- k-point convergence SI.
#
# Parses the QE outputs of the 2026-08-20 k-mesh sweep (raw pw.x .out files,
# recovered from Genkai) and reports, per cell, the energy and force error of
# each mesh against the densest mesh computed for that cell.
#
# Two cells, deliberately different:
#   - a 206-atom GB slab snapshot (disordered)   -> G, 1x1x2, 1x1x4, 2x2x2, 2x2x3
#   - a 51-atom cubic argyrodite bulk (ordered)  -> G, 2x2x2, 3x3x3, 4x4x4
# The point of the figure is that the ordered bulk needs a far denser mesh than
# the disordered slab at the same effective cell length, which is why the
# convention is a single length parameter (klen) and not a fixed mesh.
#
#   python kconv_analysis.py            # table + kconv.png + kconv_summary.json
# =============================================================================
import glob, json, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RY = 13.605698066819       # eV / Ry
BOHR = 0.529177210903      # A / bohr

CELLS = {
    "GB slab (206 atoms)": [
        ("$\\Gamma$", "old_G.out",    1),
        ("1x1x2",     "old_k112.out", 2),
        ("1x1x4",     "old_k114.out", 4),
        ("2x2x2",     "old_k222.out", 8),
        ("2x2x3",     "old_k223.out", 12),
    ],
    "bulk cubic (51 atoms)": [
        ("$\\Gamma$", "kG.bulkkchk.0.out",      1),
        ("2x2x2",     "kG.bulkkchk.0_k222.out", 8),
        ("3x3x3",     "bulk_k333.out",          27),
        ("4x4x4",     "bulk_k444.out",          64),
    ],
}


def parse(path):
    """Return (E_total_eV, forces_eV_per_A (nat,3), nat, wall_s, nk_irr)."""
    txt = open(path, errors="ignore").read()
    e = [float(m) for m in re.findall(r"^!\s+total energy\s+=\s+(-?[\d.]+)\s+Ry",
                                      txt, re.M)]
    nat = int(re.search(r"number of atoms/cell\s+=\s+(\d+)", txt).group(1))
    nkm = re.search(r"number of k points=\s*(\d+)", txt)
    nk = int(nkm.group(1)) if nkm else None
    wm = re.search(r"PWSCF\s+:.*?([\d.]+)s CPU\s+([\d.]+)s WALL", txt, re.S)
    wall = float(wm.group(2)) if wm else float("nan")
    if not wm:                       # "1h 2m" style
        wm2 = re.search(r"PWSCF\s+:.*?WALL", txt, re.S)
    # forces: last "Forces acting on atoms" block
    blocks = re.findall(r"Forces acting on atoms.*?Total force", txt, re.S)
    f = None
    if blocks:
        rows = re.findall(r"atom\s+\d+\s+type\s+\d+\s+force =\s+"
                          r"(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)", blocks[-1])
        if rows:
            f = np.array(rows, float) * RY / BOHR
    return (e[-1] * RY if e else None), f, nat, wall, nk


def wall_seconds(path):
    txt = open(path, errors="ignore").read()
    m = re.search(r"PWSCF\s+:\s+(.*?)\s+CPU\s+(.*?)\s+WALL", txt, re.S)
    if not m:
        return float("nan")
    s = m.group(2).strip()
    tot, num = 0.0, re.findall(r"([\d.]+)\s*([hms])", s)
    for v, u in num:
        tot += float(v) * {"h": 3600, "m": 60, "s": 1}[u]
    return tot


def main():
    out, rows = {}, []
    for cell, entries in CELLS.items():
        data = []
        for label, fn, nk in entries:
            if not os.path.exists(fn):
                print("MISSING", fn); continue
            E, F, nat, _, nkirr = parse(fn)
            data.append({"mesh": label, "file": fn, "nk_full": nk, "nk_irr": nkirr,
                         "E": E, "F": F, "nat": nat, "wall_s": wall_seconds(fn)})
        ref = data[-1]                      # densest mesh = reference
        for d in data:
            d["dE_meV_per_atom"] = abs(d["E"] - ref["E"]) / d["nat"] * 1000
            if d["F"] is not None and ref["F"] is not None:
                df = d["F"] - ref["F"]
                d["dF_rms"] = float(np.sqrt((df ** 2).sum(1).mean()))
                d["dF_max"] = float(np.abs(df).max())
            else:
                d["dF_rms"] = d["dF_max"] = float("nan")
        out[cell] = data
        for d in data:
            rows.append((cell, d["mesh"].replace("$\\Gamma$", "G"), d["nk_full"],
                         d["nk_irr"], d["nat"], d["dE_meV_per_atom"], d["dF_rms"],
                         d["dF_max"], d["wall_s"] / 60))

    print("%-22s %-7s %4s %4s %5s | %10s %10s %10s %8s" %
          ("cell", "mesh", "nk", "irr", "nat", "dE[meV/at]", "dF rms", "dF max", "WALL[m]"))
    print("-" * 96)
    for r in rows:
        print("%-22s %-7s %4d %4s %5d | %10.4f %10.4f %10.4f %8.1f" %
              (r[0], r[1], r[2], str(r[3]), r[4], r[5], r[6], r[7], r[8]))

    with open("kconv_summary.json", "w") as fh:
        json.dump({c: [{k: v for k, v in d.items() if k != "F"} for d in ds]
                   for c, ds in out.items()}, fh, indent=1)

    # ---------------- figure ----------------
    fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.9))
    col = {"GB slab (206 atoms)": "tab:purple", "bulk cubic (51 atoms)": "tab:green"}
    mk = {"GB slab (206 atoms)": "o", "bulk cubic (51 atoms)": "s"}
    for cell, ds in out.items():
        x = [d["nk_full"] for d in ds[:-1]]
        ax[0].plot(x, [d["dF_rms"] for d in ds[:-1]], mk[cell] + "-",
                   color=col[cell], label=cell)
        ax[1].plot(x, [d["dE_meV_per_atom"] for d in ds[:-1]], mk[cell] + "-",
                   color=col[cell], label=cell)
        for d in ds[:-1]:
            ax[0].annotate(d["mesh"].replace("$\\Gamma$", r"$\Gamma$"),
                           (d["nk_full"], d["dF_rms"]), textcoords="offset points",
                           xytext=(5, 4), fontsize=7.5, color=col[cell])
    # klen=26 adopted meshes
    ax[0].axhline(0.01, color="k", lw=0.8, ls="--")
    ax[0].text(1.05, 0.0115, "0.01 eV/$\\mathrm{\\AA}$", fontsize=7.5)
    for a, yl, tt in ((ax[0], "force RMS error [eV/$\\mathrm{\\AA}$]",
                       "(a) force error vs. densest mesh"),
                      (ax[1], "energy error [meV/atom]",
                       "(b) energy error vs. densest mesh")):
        a.set_xscale("log"); a.set_yscale("log")
        a.set_xlabel("$k$-points in the full mesh")
        a.set_ylabel(yl); a.set_title(tt, fontsize=9)
        a.legend(fontsize=7.5); a.grid(alpha=0.3, which="both", lw=0.4)
    # mark the meshes the klen=26 convention actually selects
    for a, key in ((ax[0], "dF_rms"), (ax[1], "dE_meV_per_atom")):
        for cell, mesh in (("bulk cubic (51 atoms)", "3x3x3"),):
            d = [q for q in out[cell] if q["mesh"] == mesh][0]
            a.plot([d["nk_full"]], [d[key]], "*", ms=15, mfc="none",
                   mec="k", mew=1.2, zorder=5,
                   label="adopted ($k_{\\mathrm{len}}=26$)")
        a.legend(fontsize=7.5)
    ax[0].text(0.02, 0.03,
               "reference = densest mesh computed\nGB 2$\\times$2$\\times$3, bulk 4$\\times$4$\\times$4",
               transform=ax[0].transAxes, ha="left", va="bottom", fontsize=7,
               bbox=dict(fc="white", ec="0.7", lw=0.5))
    fig.tight_layout()
    fig.savefig("kconv.png", dpi=180)
    print("\nwrote kconv.png, kconv_summary.json")


if __name__ == "__main__":
    main()

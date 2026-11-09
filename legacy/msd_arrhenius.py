#!/usr/bin/env python
# =============================================================================
# msd_arrhenius.py -- D(T) and activation energy from diffusion_md.py outputs
#
# Feed it the per-temperature MSD csv files written by diffusion_md.py. It fits
# the Einstein slope for each region at each T, then does an Arrhenius fit
#   D(T) = D0 * exp(-Ea / kB T)
# per region and reports Ea [eV] and the room-temperature extrapolation.
#
#   python msd_arrhenius.py msd_400K.csv msd_600K.csv msd_800K.csv \
#          --temps 400 600 800 --tfit 0.5 --out arrhenius
#
# If --temps is omitted it parses the temperature from '<num>K' in the filename.
# =============================================================================
import argparse
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KB = 8.617333262e-5  # eV/K


def slope_D(path, tfit):
    with open(path) as fh:
        names = fh.readline().strip().split(",")
    arr = np.genfromtxt(path, delimiter=",", skip_header=1)
    t = arr[:, 0]
    hi = t > tfit * t.max()
    out = {}
    for j, name in enumerate(names[1:], start=1):
        reg = name.replace("msd_", "").replace("_A2", "")
        slope = np.polyfit(t[hi], arr[hi, j], 1)[0]     # A^2/ps
        out[reg] = slope / 6.0 * 1e-4                   # cm^2/s
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--temps", type=float, nargs="*", default=None)
    ap.add_argument("--tfit", type=float, default=0.5, help="use MSD tail above this fraction")
    ap.add_argument("--out", default="arrhenius")
    args = ap.parse_args()

    temps = args.temps
    if temps is None:
        temps = [float(re.search(r"(\d+)K", p).group(1)) for p in args.csv]
    temps = np.array(temps, dtype=float)

    perT = [slope_D(p, args.tfit) for p in args.csv]
    regions = list(perT[0])

    plt.figure(figsize=(6, 5))
    print("== Arrhenius fit  D = D0 exp(-Ea/kBT) ==")
    for reg in regions:
        D = np.array([d[reg] for d in perT])
        ok = D > 0
        if ok.sum() < 2:
            print("  %-6s: not enough positive D" % reg)
            continue
        x = 1000.0 / temps[ok]                 # 1000/T
        y = np.log(D[ok])
        a, b = np.polyfit(1.0 / temps[ok], y, 1)   # ln D = ln D0 - Ea/kB * (1/T)
        Ea = -a * KB
        D0 = np.exp(b)
        D300 = D0 * np.exp(-Ea / (KB * 300.0))
        print("  %-6s Ea = %6.3f eV   D0 = %9.2e cm^2/s   D(300K) = %9.2e cm^2/s"
              % (reg, Ea, D0, D300))
        pl, = plt.plot(x, np.log10(D[ok]), "o", label="%s (Ea=%.2f eV)" % (reg, Ea))
        xx = np.linspace(x.min(), x.max(), 50)
        plt.plot(xx, np.log10(D0) - (Ea / KB) * (xx / 1000.0) / np.log(10),
                 color=pl.get_color(), lw=1)
    plt.xlabel("1000 / T  [1/K]", fontsize=13)
    plt.ylabel(r"log$_{10}$ D  [cm$^2$/s]", fontsize=13)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(args.out + ".png", dpi=150)
    print("wrote %s.png" % args.out)


if __name__ == "__main__":
    main()

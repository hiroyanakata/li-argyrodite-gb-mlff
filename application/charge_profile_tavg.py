#!/usr/bin/env python
# =============================================================================
# charge_profile_tavg.py -- time-averaged formal-charge density rho_q(x), Li and
# halide densities, and the 1-D space-charge potential phi(x) along the GB
# normal, from the all-atom framework runs (framework_md.py *_fwtraj.npz).
#
# Formal charges Li+1, P+5, S-2, Cl/Br-1. Every saved frame is binned along x
# (NB bins over Lx), the histograms are averaged over time, rho_q is Gaussian-
# smoothed (sigma = 2.5 A) and integrated twice (1-D Poisson, eps_r = 15,
# zero-mean field and potential under PBC).
#
#   python charge_profile_tavg.py --dir fwV2 --tag 700K --out charge_tavg.npz
# reads   <dir>/fwV2_<model>_<cell>_<tag>_fwtraj.npz   for cell in uni, seg and
#         model in v5D, pretrained; writes keys "<cell> / <label>__{xc,Li,hal,rho,phi}"
#         (label: v5D | generic).
#   python charge_profile_tavg.py --static gbV2_uni.extxyz
#         single structure (e.g. the as-built cell): GB net charge (+-3.5 A around
#         the planes at 0.25/0.75 Lx) and the potential step, no averaging.
# =============================================================================
import argparse
import numpy as np

CHG = {"Li": 1, "P": 5, "S": -2, "Cl": -1, "Br": -1, "I": -1}
EPS0 = 0.0055263496          # e / (V A)


def gsmooth(a, xc, sig):
    n = len(a)
    k = np.exp(-0.5 * ((xc - xc[n // 2]) / sig) ** 2); k /= k.sum()
    return np.real(np.fft.ifft(np.fft.fft(a) * np.fft.fft(np.roll(k, -n // 2))))


def poisson(r, dx, eps_r):
    n = len(r)
    g = np.concatenate([[0], np.cumsum(0.5 * (r[:-1] + r[1:]) * dx)])[:n]
    E = -(g - g.mean()) / (EPS0 * eps_r)
    p = np.concatenate([[0], np.cumsum(0.5 * (E[:-1] + E[1:]) * dx)])[:n]
    return p - p.mean()


def load(path, stride):
    if path.endswith(".npz"):
        d = np.load(path)
        return d["pos"][::stride], d["cell"], d["symbols"].astype(str)
    from ase.io import read
    at = read(path)
    return at.get_positions()[None], np.array(at.get_cell()), np.array(at.get_chemical_symbols())


def profile(path, nb, sig, eps_r, stride):
    pos, cell, sym = load(path, stride)
    Lx, Ly, Lz = cell[0, 0], cell[1, 1], cell[2, 2]
    edges = np.linspace(0, Lx, nb + 1); xc = 0.5 * (edges[:-1] + edges[1:]); dx = edges[1] - edges[0]
    vol = dx * Ly * Lz
    q = np.array([CHG.get(s, 0) for s in sym], float)
    li = sym == "Li"; hal = np.isin(sym, ["Cl", "Br", "I"])
    R = np.zeros(nb); L = np.zeros(nb); H = np.zeros(nb)
    for fr in pos:
        x = np.mod(fr[:, 0], Lx)
        R += np.histogram(x, bins=edges, weights=q)[0]
        L += np.histogram(x[li], bins=edges)[0]
        H += np.histogram(x[hal], bins=edges)[0]
    n = len(pos)
    rho = R / n / vol
    gb = np.min(np.abs(xc[:, None] - np.array([0.25, 0.75])[None, :] * Lx), axis=1) < 3.5
    return dict(xc=xc, Li=L / n / vol, hal=H / n / vol, rho=rho,
                phi=poisson(gsmooth(rho, xc, sig), dx, eps_r),
                q_gb=float((rho[gb] * vol).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".")
    ap.add_argument("--tag", default="700K")
    ap.add_argument("--cells", default="uni,seg")
    ap.add_argument("--nb", type=int, default=46)
    ap.add_argument("--sig", type=float, default=2.5, help="Gaussian smoothing of rho_q [A]")
    ap.add_argument("--eps_r", type=float, default=15.0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--out", default="charge_tavg.npz")
    ap.add_argument("--static", default=None, help="single extxyz/npz: print q_GB and the phi step only")
    a = ap.parse_args()
    if a.static:
        r = profile(a.static, a.nb, a.sig, a.eps_r, a.stride)
        print(f"{a.static}: GB net charge {r['q_gb']:+.1f} e, phi step {r['phi'].max() - r['phi'].min():.2f} V")
        return
    out = {}
    for cell in a.cells.split(","):
        for m, lab in (("v5D", "v5D"), ("pretrained", "generic")):
            r = profile(f"{a.dir}/fwV2_{m}_{cell}_{a.tag}_fwtraj.npz", a.nb, a.sig, a.eps_r, a.stride)
            r.pop("q_gb")
            for k, v in r.items():
                out[f"{cell} / {lab}__{k}"] = v
            print(f"{cell:4s} {lab:8s} dphi = {r['phi'].max() - r['phi'].min():.2f} V")
    np.savez_compressed(a.out, **out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()

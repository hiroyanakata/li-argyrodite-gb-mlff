import numpy as np, re

CHG = {"Li": +1, "P": +5, "S": -2, "Cl": -1, "Br": -1, "I": -1}
EPS0 = 0.0055263496  # vacuum permittivity, e/(V*Angstrom)
GB = [17.36, 52.07]; WIN = 3.5

def read_xyz(path):
    with open(path) as f: L = f.read().splitlines()
    n = int(L[0].split()[0])
    cell = np.array(re.search(r'Lattice="([^"]+)"', L[1]).group(1).split(), float).reshape(3, 3)
    sym = []; pos = []
    for l in L[2:2+n]:
        p = l.split(); sym.append(p[0]); pos.append([float(x) for x in p[1:4]])
    return np.array(sym), np.array(pos), cell

def profile(path, nb=46):
    sym, pos, cell = read_xyz(path)
    Lx, Ly, Lz = cell[0, 0], cell[1, 1], cell[2, 2]
    x = np.mod(pos[:, 0], Lx)
    edges = np.linspace(0, Lx, nb + 1); xc = 0.5 * (edges[:-1] + edges[1:]); dx = edges[1] - edges[0]
    vol = dx * Ly * Lz
    dens = {}
    for sp in ["Li", "S", "hal", "P"]:
        if sp == "hal": m = np.isin(sym, ["Cl", "Br", "I"])
        else: m = sym == sp
        dens[sp], _ = np.histogram(x[m], bins=edges)
        dens[sp] = dens[sp] / vol
    q = np.array([CHG.get(s, 0) for s in sym], float)
    rho, _ = np.histogram(x, bins=edges, weights=q); rho = rho / vol  # e/A^3

    def poisson(r, eps_r=1.0):
        g = np.concatenate([[0], np.cumsum(0.5 * (r[:-1] + r[1:]) * dx)])[:nb]
        E = -(g - g.mean()) / (EPS0 * eps_r)
        p = np.concatenate([[0], np.cumsum(0.5 * (E[:-1] + E[1:]) * dx)])[:nb]
        return p - p.mean()

    # Gaussian smoothing along x (periodic) with sigma ~ GB width, to get the MACROSCOPIC potential
    def gsmooth(a, sig=2.5):
        k = np.exp(-0.5 * ((xc - xc[nb // 2]) / sig) ** 2); k /= k.sum()
        return np.real(np.fft.ifft(np.fft.fft(a) * np.fft.fft(np.roll(k, -nb // 2))))
    rho_s = gsmooth(rho)
    phi = poisson(rho)                       # raw (point-charge, artifact-dominated)
    phi_s = poisson(rho_s)                   # macroscopic, vacuum
    phi_s15 = poisson(rho_s, eps_r=15.0)     # macroscopic with dielectric screening eps_r=15

    gbmask = np.min(np.abs(xc[:, None] - np.array(GB)[None, :]), axis=1) < WIN
    q_gb = float((rho[gbmask] * vol).sum())
    # positive charge in GB window (for relative imbalance)
    qpos_gb = float((rho[gbmask][rho[gbmask] > 0] * vol).sum())
    return dict(xc=xc, Li=dens["Li"], S=dens["S"], hal=dens["hal"], P=dens["P"],
                rho=rho, phi=phi_s, qtot=float((rho * vol).sum()), q_gb=q_gb,
                dphi_raw=float(phi.max() - phi.min()),
                dphi_macro=float(phi_s.max() - phi_s.min()),
                dphi_macro15=float(phi_s15.max() - phi_s15.min()),
                rel=abs(q_gb) / qpos_gb * 100)

files = {
    "initial":  "gbBIG_dec.extxyz",
    "phaseB":   "nemd_phaseB_dec_700K_z_F010_s0_ckpt.extxyz",
    "generic":  "nemd_pretrained_dec_700K_z_F010_s0_ckpt.extxyz",
}
out = {}
for k, f in files.items():
    r = profile(f)
    out[k + "__xc"] = r["xc"]
    for key in ["Li", "S", "hal", "P", "rho", "phi"]:
        out[k + "__" + key] = r[key]
    print(f"{k:9s} qtot={r['qtot']:+.2f} | GB net q={r['q_gb']:+6.1f} e ({r['rel']:.1f}% of GB+ charge) | dphi raw={r['dphi_raw']:6.1f}V  macro(vac)={r['dphi_macro']:.2f}V  macro(eps15)={r['dphi_macro15']:.3f}V")
np.savez_compressed("charge_profile.npz", **out)
print("wrote charge_profile.npz")

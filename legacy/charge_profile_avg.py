import numpy as np, re, glob

CHG = {"Li": +1, "P": +5, "S": -2, "Cl": -1, "Br": -1, "I": -1}
EPS0 = 0.0055263496
GB = [17.36, 52.07]; WIN = 3.5; NB = 46

def read_xyz(path):
    with open(path) as f: L = f.read().splitlines()
    n = int(L[0].split()[0])
    cell = np.array(re.search(r'Lattice="([^"]+)"', L[1]).group(1).split(), float).reshape(3, 3)
    sym = []; pos = []
    for l in L[2:2+n]:
        p = l.split(); sym.append(p[0]); pos.append([float(x) for x in p[1:4]])
    return np.array(sym), np.array(pos), cell

def profile(path):
    sym, pos, cell = read_xyz(path)
    Lx, Ly, Lz = cell[0, 0], cell[1, 1], cell[2, 2]
    x = np.mod(pos[:, 0], Lx)
    edges = np.linspace(0, Lx, NB + 1); xc = 0.5*(edges[:-1]+edges[1:]); dx = edges[1]-edges[0]
    vol = dx*Ly*Lz
    dens = {}
    for sp in ["Li", "S", "hal", "P"]:
        m = np.isin(sym, ["Cl","Br","I"]) if sp == "hal" else (sym == sp)
        dens[sp] = np.histogram(x[m], bins=edges)[0] / vol
    q = np.array([CHG.get(s, 0) for s in sym], float)
    rho = np.histogram(x, bins=edges, weights=q)[0] / vol
    def gsmooth(a, sig=2.5):
        k = np.exp(-0.5*((xc - xc[NB//2])/sig)**2); k /= k.sum()
        return np.real(np.fft.ifft(np.fft.fft(a)*np.fft.fft(np.roll(k, -NB//2))))
    def poisson(r, eps_r=1.0):
        g = np.concatenate([[0], np.cumsum(0.5*(r[:-1]+r[1:])*dx)])[:NB]
        E = -(g - g.mean())/(EPS0*eps_r)
        p = np.concatenate([[0], np.cumsum(0.5*(E[:-1]+E[1:])*dx)])[:NB]
        return p - p.mean()
    phi15 = poisson(gsmooth(rho), 15.0)
    gbmask = np.min(np.abs(xc[:,None]-np.array(GB)[None,:]), axis=1) < WIN
    return dict(xc=xc, Li=dens["Li"], S=dens["S"], hal=dens["hal"], P=dens["P"], rho=rho,
                phi15=phi15, q_gb=float((rho[gbmask]*vol).sum()),
                dphi15=float(phi15.max()-phi15.min()))

models = {"phaseB":"nemd_phaseB_dec_700K_z_F010_s{}_ckpt.extxyz",
          "generic":"nemd_pretrained_dec_700K_z_F010_s{}_ckpt.extxyz"}
out = {}
# initial reference
ri = profile("gbBIG_dec.extxyz")
out["initial__xc"]=ri["xc"]
for kk in ["Li","S","hal","P","rho","phi15"]: out["initial__"+kk]=ri[kk]
print(f"initial   GB net q={ri['q_gb']:+6.1f} e   dphi15={ri['dphi15']:.2f} V")
for m, pat in models.items():
    runs=[profile(f) for s in (0,1,2) for f in glob.glob(pat.format(s))]
    def stk(key): return np.array([r[key] for r in runs])
    for kk in ["Li","S","hal","P","rho","phi15"]:
        A=stk(kk); out[f"{m}__{kk}"]=A.mean(0); out[f"{m}__{kk}_sd"]=A.std(0)
    out[m+"__xc"]=runs[0]["xc"]
    qg=stk("q_gb"); dp=stk("dphi15")
    out[m+"__scal"]=np.array([qg.mean(),qg.std(),dp.mean(),dp.std(),len(runs)])
    print(f"{m:9s} n={len(runs)}  GB net q={qg.mean():+6.1f} ± {qg.std():4.1f} e   dphi15={dp.mean():.2f} ± {dp.std():.2f} V")
out["_GB"]=np.array(GB)
np.savez_compressed("charge_profile_avg.npz", **out)
print("wrote charge_profile_avg.npz")

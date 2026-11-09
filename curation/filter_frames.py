#!/usr/bin/env python
"""Physics-based frame filter: drop structures that are unphysical *even allowing
for a reaction barrier of the size being studied*.

Placed between embedding extraction and clustering:

    extract_embeddings.py  ->  filter_frames.py  ->  cluster_embeddings.py --filter
                                                 ->  split_by_cluster.py
                                                 ->  select_from_clusters.py

The whole point is that a transition state is a legitimate high-energy, high-force,
stretched-bond structure. A filter tuned on "large energy" or "short distance" alone
throws those away. So every threshold here is expressed against an explicit energy
budget --barrier_kcal (default 30 kcal/mol): a configuration is rejected only when it
sits *further* from chemistry than the barrier you are studying.

A caution that shaped the design: for a few-hundred-atom periodic cell the thermal
fluctuation of the total energy (sigma of order 1 eV, full span tens of eV) is larger
than a 20-30 kcal/mol barrier (0.87-1.30 eV). Absolute total energy therefore has no
discriminating power at the barrier scale, and the energy test below is deliberately
built so it can only ever fire on gross outliers.

Criteria (all reported; the ones marked [drop] decide the mask):

  [drop] overlap   Per element pair, closest approach vs. a floor obtained by
                   inverting a Morse repulsion at the energy budget:
                       r_floor(i,j) = r_ref(i,j) - ln(1 + sqrt(Ebudget/D)) / a
                   r_ref is auto-calibrated (--calibrate auto) as
                       min(rcov_i + rcov_j, median over frames of the closest i-j)
                   so a pair whose real bond is shorter than the single-bond covalent
                   sum -- aromatic C-C, polar Si-O -- gets the right reference, while
                   never-bonded pairs keep the generous covalent value. Genuine H2
                   (d ~ 0.74 A) survives; a compressed framework bond does not.
  [drop] fmax      Peak atomic force. A 30 kcal/mol barrier over a ~0.5 A reaction
                   coordinate implies forces of a few eV/A; --fmax_max is a generous
                   ceiling well above any real saddle point.
  [drop] energy    Robust outlier *within each listdata block* (median/MAD). Fires
                   only when the frame is both a >--ez_max sigma outlier AND more than
                   the energy budget above the block median, so a barrier crossing can
                   never be removed by this rule alone.
  [opt]  coord     Coordination rule like "Si:O:4,Al:O:4". Under- and over-coordination
                   are counted separately. Off by default: breaking a framework bond
                   may be the chemistry you are studying. Add --coord_drop to act on it.
  [opt]  spike     |E - rolling median of neighbouring frames| > --spike_ev, for
                   single-frame SCF failures. Needs frames in trajectory order; off
                   by default.

The expensive geometry pass is cached in filter_scan.npz, so re-running with different
thresholds is instant. Use --rescan to force a fresh pass.

Outputs:
  frame_filter.npy      bool [N] keep mask, aligned to listdata frame order
  filter_scan.npz       cached per-frame geometry diagnostics
  filter_report.json    thresholds, calibrated floors, per-criterion counts
  filter_verdict.csv    one row per rejected frame with the reasons
  filter_overview.png   diagnostics with the thresholds drawn in

Usage:
  python filter_frames.py fulldata.txt --system system.json
  python filter_frames.py fulldata.txt --system system.json --barrier_kcal 30 \
      --fmax_max 20 --coord_rule "Si:O:4,Al:O:4"
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KCAL_PER_EV = 23.060548
RytoeV = 13.605698066819

# Cordero et al. (2008) covalent radii [A]; extend as needed.
RCOV = {
    "H": 0.31, "Li": 1.28, "Be": 0.96, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66,
    "F": 0.57, "Na": 1.66, "Mg": 1.41, "Al": 1.21, "Si": 1.11, "P": 1.07, "S": 1.05,
    "Cl": 1.02, "K": 2.03, "Ca": 1.76, "Ti": 1.60, "Fe": 1.52, "Co": 1.50, "Ni": 1.24,
    "Cu": 1.32, "Zn": 1.22, "Ga": 1.22, "Ge": 1.20, "As": 1.19, "Se": 1.20, "Br": 1.20,
    "Zr": 1.75, "Mo": 1.54, "Ag": 1.45, "Sn": 1.39, "Sb": 1.39, "Te": 1.38, "I": 1.39,
    "Ba": 2.15, "La": 2.07, "W": 1.62, "Pt": 1.36, "Au": 1.36, "Pb": 1.46,
}

p = argparse.ArgumentParser()
p.add_argument("listdata", help="listdata in the same frame order the embeddings were built in")
p.add_argument("--system", default="system.json", help="JSON with the element order of type.raw")
p.add_argument("--barrier_kcal", type=float, default=30.0,
               help="reaction barrier being studied [kcal/mol]; sets the energy budget")
p.add_argument("--calibrate", choices=["auto", "off"], default="auto",
               help="auto: reference bond lengths from the data (capped by covalent radii)")
p.add_argument("--fmax_max", type=float, default=20.0, help="reject fmax above this [eV/A]")
p.add_argument("--ez_max", type=float, default=10.0, help="robust-z ceiling for the energy test")
p.add_argument("--rep_D", type=float, default=4.0, help="Morse well depth for the overlap floor [eV]")
p.add_argument("--rep_a", type=float, default=2.0, help="Morse width for the overlap floor [1/A]")
p.add_argument("--coord_rule", default="", help='e.g. "Si:O:4,Al:O:4" (centre:ligand:n)')
p.add_argument("--coord_rcut", type=float, default=2.0, help="cutoff for the coordination rule [A]")
p.add_argument("--coord_drop", action="store_true", help="actually drop coordination violations")
p.add_argument("--spike_ev", type=float, default=0.0,
               help="reject |E - rolling median| above this [eV]; 0 disables")
p.add_argument("--batch", type=int, default=96, help="frames per distance batch")
p.add_argument("--cache", default="filter_scan.npz")
p.add_argument("--rescan", action="store_true", help="ignore the cache and redo the geometry pass")
p.add_argument("--out", default="frame_filter.npy")
args = p.parse_args()

Ebudget = args.barrier_kcal / KCAL_PER_EV
print(f"[budget] barrier {args.barrier_kcal:.1f} kcal/mol = {Ebudget:.3f} eV")

with open(args.system) as f:
    sysjson = json.load(f)
elements = np.asarray(sysjson["elements"])
if "atmene_eV" in sysjson:
    atmene = np.asarray(sysjson["atmene_eV"], dtype=float)
elif "atmene_Ry" in sysjson:
    atmene = np.asarray(sysjson["atmene_Ry"], dtype=float) * RytoeV
else:
    raise KeyError(f"{args.system} must define atmene_eV or atmene_Ry")
nel = len(elements)
print(f"[system] elements = {elements.tolist()}")

missing = [e for e in elements if e not in RCOV]
if missing:
    raise SystemExit(f"no covalent radius for {missing}; add it to RCOV in this file")
rcov_sum = np.array([[RCOV[a] + RCOV[b] for b in elements] for a in elements])

pairs = [(i, j) for i in range(nel) for j in range(i, nel)]
pidx = {p_: k for k, p_ in enumerate(pairs)}
npair = len(pairs)

rules = []
for tok in filter(None, (t.strip() for t in args.coord_rule.split(","))):
    c, l, k = tok.split(":")
    rules.append((int(np.where(elements == c)[0][0]), int(np.where(elements == l)[0][0]),
                  int(k), c, l))
if rules:
    print("[coord] rules:", ", ".join(f"{c}-{l} == {k} within {args.coord_rcut} A"
                                      for _, _, k, c, l in rules))

with open(args.listdata) as f:
    lines = [l for l in f if l.strip()]
nat_ref = len(np.loadtxt(lines[0].split()[0] + "type.raw", dtype=int).reshape(-1))

# ---------------- expensive geometry pass (cached) ----------------
cache_ok = False
if os.path.exists(args.cache) and not args.rescan:
    z = np.load(args.cache, allow_pickle=True)
    if (str(z["listdata"]) == args.listdata and int(z["npair"]) == npair
            and list(z["elements"]) == elements.tolist()
            and float(z["coord_rcut"]) == args.coord_rcut
            and str(z["coord_rule"]) == args.coord_rule):
        epa, fmax = z["epa"], z["fmax"]
        block, loc = z["block"], z["loc"]
        dpair, cunder, cover = z["dpair"], z["cunder"], z["cover"]
        nblk = int(z["nblk"])
        cache_ok = True
        print(f"[cache] reusing {args.cache} ({len(epa)} frames) -- use --rescan to redo")

if not cache_ok:
    epa_l, fmax_l, block_l, loc_l, dpair_l, cu_l, co_l = [], [], [], [], [], [], []
    nblk = 0
    for line in lines:
        aa = line.split()
        name = aa[0]
        types = np.loadtxt(name + "type.raw", dtype=int).reshape(-1)
        nat = len(types)
        ezero = atmene[types].sum()
        where = [np.where(types == i)[0] for i in range(nel)]
        for iset in aa[1:]:
            d = name + iset
            ene = np.load(os.path.join(d, "energy.npy")).astype(np.float64).reshape(-1)
            n = len(ene)
            frc = np.load(os.path.join(d, "force.npy")).astype(np.float32).reshape(n, nat, 3)
            crd = np.load(os.path.join(d, "coord.npy")).astype(np.float32).reshape(n, nat, 3)
            box = np.load(os.path.join(d, "box.npy")).astype(np.float32).reshape(n, 3, 3)
            L = np.stack([np.diag(b) for b in box]).astype(np.float32)

            epa_l.append((ene - ezero) / nat)
            fmax_l.append(np.linalg.norm(frc, axis=2).max(axis=1))
            block_l.append(np.full(n, nblk)); loc_l.append(np.arange(n))

            dp = np.full((n, npair), np.inf, np.float32)
            cu = np.zeros(n, np.int32); co = np.zeros(n, np.int32)
            ar = np.arange(nat)
            for s in range(0, n, args.batch):
                e = min(s + args.batch, n)
                x = crd[s:e]
                l = L[s:e][:, None, None, :]
                dv = x[:, :, None, :] - x[:, None, :, :]
                dv -= l * np.round(dv / l)
                r = np.sqrt((dv * dv).sum(-1))
                r[:, ar, ar] = np.float32(1e9)
                for i, j in pairs:
                    ai, bj = where[i], where[j]
                    if ai.size == 0 or bj.size == 0:
                        continue
                    dp[s:e, pidx[(i, j)]] = r[:, ai][:, :, bj].min(axis=(1, 2))
                for ci, li, k, _, _ in rules:
                    nc = (r[:, where[ci]][:, :, where[li]] < args.coord_rcut).sum(axis=2)
                    cu[s:e] += (nc < k).sum(axis=1)
                    co[s:e] += (nc > k).sum(axis=1)
            dpair_l.append(dp); cu_l.append(cu); co_l.append(co)
            print(f"  block {nblk}: {d}  n={n}", flush=True)
            nblk += 1

    epa = np.concatenate(epa_l); fmax = np.concatenate(fmax_l)
    block = np.concatenate(block_l); loc = np.concatenate(loc_l)
    dpair = np.concatenate(dpair_l); cunder = np.concatenate(cu_l); cover = np.concatenate(co_l)
    np.savez(args.cache, epa=epa, fmax=fmax, block=block, loc=loc, dpair=dpair,
             cunder=cunder, cover=cover, nblk=nblk, npair=npair, elements=elements,
             listdata=args.listdata, coord_rcut=args.coord_rcut, coord_rule=args.coord_rule)
    print(f"[cache] wrote {args.cache}")

N = len(epa)
print(f"\nN = {N} frames in {nblk} blocks")

# ---------------- reference bond lengths + overlap floors ----------------
shrink = np.log1p(np.sqrt(Ebudget / args.rep_D)) / args.rep_a
r_ref = rcov_sum.copy()
if args.calibrate == "auto":
    for (i, j), k in pidx.items():
        col = dpair[:, k]
        col = col[np.isfinite(col)]
        if col.size:
            r_ref[i, j] = r_ref[j, i] = min(rcov_sum[i, j], float(np.median(col)))
floor = np.maximum(r_ref - shrink, 0.10)
print(f"[overlap] budget shrinks r_ref by {shrink:.3f} A "
      f"(calibrate={args.calibrate}); r_ref -> floor [A]:")
for i in range(nel):
    row = [f"{elements[i]}-{elements[j]}:{r_ref[i,j]:.3f}->{floor[i,j]:.3f}"
           for j in range(i, nel) if np.isfinite(dpair[:, pidx[(i, j)]]).any()]
    if row:
        print("   " + "  ".join(row))

viol = np.full(N, -np.inf, np.float32)
vpair = np.zeros(N, np.int32)
for (i, j), k in pidx.items():
    v = (floor[i, j] - dpair[:, k]) / floor[i, j]
    upd = v > viol
    viol = np.where(upd, v, viol)
    vpair[upd] = k

# ---------------- criteria ----------------
bad_overlap = viol > 0.0
bad_force = fmax > args.fmax_max

ez = np.zeros(N); dE = np.zeros(N)
for b in range(nblk):
    m = block == b
    v = epa[m]
    med = np.median(v)
    mad = np.median(np.abs(v - med)) * 1.4826
    ez[m] = (v - med) / mad if mad > 0 else 0.0
    dE[m] = v - med
bad_energy = (np.abs(ez) > args.ez_max) & (np.abs(dE) * nat_ref > Ebudget)

bad_coord = (cunder + cover) > 0
bad_spike = np.zeros(N, bool)
if args.spike_ev > 0:
    for b in range(nblk):
        m = np.where(block == b)[0]
        o = m[np.argsort(loc[m])]
        v = epa[o] * nat_ref
        pad = np.pad(v, 2, mode="edge")
        win = np.stack([pad[i:i + len(v)] for i in range(5)])
        bad_spike[o] = np.abs(v - np.median(np.delete(win, 2, axis=0), axis=0)) > args.spike_ev

drop = bad_overlap | bad_force | bad_energy
if args.coord_drop:
    drop |= bad_coord
if args.spike_ev > 0:
    drop |= bad_spike
keep = ~drop
np.save(args.out, keep)

# ---------------- report ----------------
def pct(m):
    return f"{int(m.sum()):6d} ({100 * m.mean():5.2f}%)"

pname = {}
for k in vpair[bad_overlap]:
    i, j = pairs[k]
    key = f"{elements[i]}-{elements[j]}"
    pname[key] = pname.get(key, 0) + 1

print("\n--- criteria ---")
print(f"  [drop] overlap (barrier {args.barrier_kcal:g} kcal/mol) : {pct(bad_overlap)}  {pname}")
print(f"  [drop] fmax > {args.fmax_max:g} eV/A                    : {pct(bad_force)}")
print(f"  [drop] energy outlier (|z|>{args.ez_max:g} & >budget)    : {pct(bad_energy)}")
print(f"  [{'drop' if args.coord_drop else 'info'}] coordination violated "
      f"(under {int((cunder>0).sum())} / over {int((cover>0).sum())}) : {pct(bad_coord)}")
if args.spike_ev > 0:
    print(f"  [drop] energy spike > {args.spike_ev:g} eV              : {pct(bad_spike)}")
print(f"  => dropped {pct(drop)}, kept {int(keep.sum())} / {N}")
if drop.any():
    print(f"     dropped: fmax med={np.median(fmax[drop]):.2f} max={fmax[drop].max():.2f}")
print(f"     kept   : fmax p99={np.percentile(fmax[keep], 99):.2f} max={fmax[keep].max():.2f}")

report = dict(
    listdata=args.listdata, n_frames=int(N), n_blocks=int(nblk),
    barrier_kcal=args.barrier_kcal, barrier_eV=Ebudget, calibrate=args.calibrate,
    r_ref_A={f"{elements[i]}-{elements[j]}": float(r_ref[i, j]) for i, j in pairs},
    overlap_floor_A={f"{elements[i]}-{elements[j]}": float(floor[i, j]) for i, j in pairs},
    observed_min_A={f"{elements[i]}-{elements[j]}": float(np.nanmin(dpair[:, pidx[(i, j)]]))
                    for i, j in pairs if np.isfinite(dpair[:, pidx[(i, j)]]).any()},
    fmax_max=args.fmax_max, ez_max=args.ez_max,
    coord_rule=args.coord_rule, coord_drop=bool(args.coord_drop), spike_ev=args.spike_ev,
    n_overlap=int(bad_overlap.sum()), overlap_pairs=pname,
    n_force=int(bad_force.sum()), n_energy=int(bad_energy.sum()),
    n_coord_under=int((cunder > 0).sum()), n_coord_over=int((cover > 0).sum()),
    n_spike=int(bad_spike.sum()),
    n_dropped=int(drop.sum()), n_kept=int(keep.sum()),
    per_block_dropped={int(b): int(drop[block == b].sum()) for b in range(nblk)},
    fmax_p99_kept=float(np.percentile(fmax[keep], 99)),
    fmax_max_kept=float(fmax[keep].max()),
)
with open("filter_report.json", "w") as f:
    json.dump(report, f, indent=2)

with open("filter_verdict.csv", "w") as f:
    f.write("global_idx,block,local_idx,Epa,fmax,worst_overlap_frac,worst_pair,"
            "coord_under,coord_over,energy_z,overlap,force,energy,coord,spike\n")
    for g in np.where(drop)[0]:
        i, j = pairs[vpair[g]]
        f.write(f"{g},{block[g]},{loc[g]},{epa[g]:.6f},{fmax[g]:.4f},{viol[g]:.4f},"
                f"{elements[i]}-{elements[j]},{cunder[g]},{cover[g]},{ez[g]:.2f},"
                f"{int(bad_overlap[g])},{int(bad_force[g])},{int(bad_energy[g])},"
                f"{int(bad_coord[g])},{int(bad_spike[g])}\n")

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].hist(fmax, bins=120, log=True, color="0.7")
ax[0].axvline(args.fmax_max, color="r", ls="--", label=f"fmax_max={args.fmax_max:g}")
ax[0].set_xlabel("fmax [eV/A]"); ax[0].set_ylabel("frames"); ax[0].legend()
ax[1].hist(np.clip(viol, -1, None), bins=120, log=True, color="0.7")
ax[1].axvline(0.0, color="r", ls="--", label="overlap floor")
ax[1].set_xlabel("relative overlap violation  (floor-d)/floor"); ax[1].legend()
ax[2].scatter(epa[keep], fmax[keep], s=1, alpha=0.2, label="kept")
ax[2].scatter(epa[drop], fmax[drop], s=5, color="r", label="dropped")
ax[2].set_xlabel("E_form / atom [eV]"); ax[2].set_ylabel("fmax [eV/A]"); ax[2].legend()
fig.tight_layout(); fig.savefig("filter_overview.png", dpi=130)

print(f"\nwrote {args.out}, filter_report.json, filter_verdict.csv, filter_overview.png")

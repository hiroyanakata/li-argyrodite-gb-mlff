#!/usr/bin/env python
"""Stratified diversity selection from embedding clusters.

Given the MatterSim-embedding clusters (cluster_labels.npy), build a reduced,
regime-balanced training set:

  1. Test split (leak-aware, representative): the trajectory is cut into --nseg
     contiguous segments; every round(1/test_frac)-th segment is held out as
     test. Contiguous holdout avoids near-duplicate leakage (neighbours of a
     train frame are in the same segment), while spreading test blocks evenly in
     time keeps every volume/energy regime represented -- unlike a global
     last-10% split which lands on one volume block.
  2. Allocation: --nsel structures are split across clusters proportional to
     each cluster's train-pool size (--alloc {prop,sqrt,equal}).
  3. Within each cluster: farthest point sampling in embedding space, seeded with
     any global force/energy/volume extremes that fall in the cluster, so rare
     structures are always kept.

Outputs:
  train_reduce_emb.extxyz    selected training structures
  test_emb.extxyz            held-out test structures
  selection_idx_emb.npy      global frame indices of the selection
  selection_report_emb.json  statistics
"""
import argparse
import json
import os

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

RytoeV = 13.605698066819
atmene = np.asarray([-14.33801816, -13.66813830, -23.81230945,
                     -46.73796118, -378.81243284, -33.16874338]) * RytoeV
atmtyp = np.asarray(["Li", "P", "S", "Br", "I", "Cl"])

p = argparse.ArgumentParser()
p.add_argument("listdata")
p.add_argument("--nsel", type=int, default=4000)
p.add_argument("--emb", default="embeddings.npy")
p.add_argument("--meta", default="emb_meta.npy")
p.add_argument("--labels", default="cluster_labels.npy")
p.add_argument("--pca", type=int, default=50)
p.add_argument("--nseg", type=int, default=100, help="contiguous segments for test split")
p.add_argument("--test_frac", type=float, default=0.1)
p.add_argument("--alloc", choices=["prop", "sqrt", "equal", "cover"], default="prop",
               help="cover = proportional to each cluster's EFFECTIVE diversity "
                    "(frames surviving a min-distance prune), not its raw count; "
                    "prop rewards whichever trajectory was dumped most densely")
p.add_argument("--seg_by_batch", action="store_true",
               help="cut the test segments INSIDE each system of listdata instead of "
                    "across the concatenated index. Global segmentation gives whole "
                    "systems zero test frames when the systems differ in size by 70x")
p.add_argument("--min_dist", type=float, default=0.0,
               help="FPS floor, IN UNITS OF the set's median nearest-neighbour "
                    "spacing: stop selecting from a cluster once the next pick "
                    "would be closer than this, and hand the unspent budget to "
                    "clusters that still have unexplored space (0 = off, 1 = "
                    "refuse picks tighter than the data's own typical spacing)")
p.add_argument("--cover_thr", type=float, default=1.0,
               help="--alloc cover threshold, same units as --min_dist")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out_prefix", default="",
               help="prefix for the output files. Empty keeps the historical "
                    "names train_reduce_emb.extxyz / test_emb.extxyz -- which "
                    "OVERWRITES the current training set, so pass a prefix when "
                    "building a new selection alongside the old one")
p.add_argument("--ntest", type=int, default=0,
               help="subsample the held-out pool to this many frames by farthest "
                    "point sampling (0 = keep every frame in the test segments). "
                    "Needed when the test frames have to be recomputed by DFT too")
args = p.parse_args()
rng = np.random.default_rng(args.seed)

# ---------- embedding space (same recipe as clustering) ----------
E = np.load(args.emb).astype(np.float64)
meta = np.load(args.meta).astype(np.float64)
labels = np.load(args.labels)
vpa, epa, fmax = meta[:, 0], meta[:, 1], meta[:, 2]
N = E.shape[0]
E = E[:, E.std(axis=0) > 1e-8]
Xs = StandardScaler().fit_transform(E)
Z = PCA(n_components=min(args.pca, Xs.shape[1]), whiten=True,
        random_state=args.seed).fit_transform(Xs).astype(np.float32)

# ---------- which system does each frame come from? ----------
def batch_ids(listdata, n_total):
    """frame -> system index, in the same concatenation order load_blocks uses."""
    ids, b = [], 0
    with open(listdata) as f:
        for line in f:
            aa = line.split()
            if not aa:
                continue
            for iset in aa[1:]:
                n = int(np.load(os.path.join(aa[0] + iset, "energy.npy")).shape[0])
                ids.append(np.full(n, b, dtype=np.int32))
            b += 1
    ids = np.concatenate(ids) if ids else np.zeros(n_total, np.int32)
    return ids if len(ids) == n_total else np.zeros(n_total, np.int32)

bid = batch_ids(args.listdata, N)

# ---------- leak-aware test split: evenly spaced contiguous segments ----------
step = max(2, round(1.0 / args.test_frac))
if args.seg_by_batch:
    # segment INSIDE each system, so a 500-frame system still contributes test
    # frames instead of being swallowed whole by a neighbouring segment
    is_test = np.zeros(N, bool)
    for b in np.unique(bid):
        idx = np.where(bid == b)[0]
        nb = len(idx)
        nseg_b = max(step, min(args.nseg, nb))
        sg = (np.arange(nb) * nseg_b // nb)
        tset = set(range(step // 2, nseg_b, step))
        is_test[idx] = np.array([x in tset for x in sg])
else:
    seg = (np.arange(N) * args.nseg // N)           # segment id per frame, in time order
    test_segments = set(range(step // 2, args.nseg, step))
    is_test = np.array([s in test_segments for s in seg])
print("test frames per system:",
      {int(b): int(is_test[bid == b].sum()) for b in np.unique(bid)})
train_pool = np.where(~is_test)[0]
print(f"N={N}  test={is_test.sum()} ({is_test.mean():.1%})  train_pool={len(train_pool)}")

# ---------- per-cluster allocation over the train pool ----------
pool_labels = labels[train_pool]
uniq = np.unique(pool_labels)
sizes = np.array([(pool_labels == c).sum() for c in uniq], dtype=float)

# scale for --alloc cover / --min_dist: the set's own median NN spacing in Z
def median_nn(Zs, k=4000, seed=0):
    r = np.random.default_rng(seed)
    s_ = Zs[r.choice(len(Zs), min(k, len(Zs)), replace=False)]
    d2 = ((s_ ** 2).sum(1)[:, None] + (s_ ** 2).sum(1)[None, :] - 2 * s_ @ s_.T)
    np.fill_diagonal(d2, 9e18)
    return float(np.median(np.sqrt(np.maximum(d2.min(1), 0))))

DTHR = 0.0
MED_NN = 0.0
if args.alloc == "cover" or args.min_dist > 0:
    MED_NN = median_nn(Z[train_pool])
    DTHR = args.min_dist * MED_NN if args.min_dist > 0 else 0.0
    print(f"median NN spacing in PCA-whitened space = {MED_NN:.4f}"
          f"  -> FPS floor {DTHR:.4f}, cover threshold "
          f"{args.cover_thr * MED_NN:.4f}")

def n_cover(idx_global, thr):
    """greedy count of frames that stay >= thr apart -- the cluster's real size."""
    zc = Z[idx_global]
    keep = np.empty((0, zc.shape[1]), np.float32)
    n = 0
    for i in range(len(zc)):
        if n == 0 or np.linalg.norm(keep - zc[i], axis=1).min() >= thr:
            keep = np.vstack([keep, zc[i]]); n += 1
    return n

if args.alloc == "prop":
    w = sizes
elif args.alloc == "sqrt":
    w = np.sqrt(sizes)
elif args.alloc == "cover":
    w = np.array([float(n_cover(train_pool[pool_labels == c],
                                args.cover_thr * MED_NN)) for c in uniq])
    print("cover weights: sum %.0f (raw sizes sum %.0f) -- clusters shrink by "
          "how redundant they are" % (w.sum(), sizes.sum()))
else:
    w = np.ones_like(sizes)
nsel = min(args.nsel, len(train_pool))
alloc = np.floor(w / w.sum() * nsel).astype(int)
alloc = np.minimum(alloc, sizes.astype(int))
# distribute the rounding remainder to the largest clusters
while alloc.sum() < nsel and (alloc < sizes).any():
    room = sizes.astype(int) - alloc
    alloc[np.argmax(np.where(room > 0, w, -1))] += 1

# ---------- global extremes forced as seeds (mapped into their clusters) ----------
tp_set = set(train_pool.tolist())
extremes = set()
extremes.update(np.argsort(fmax)[-20:].tolist())
extremes.update([int(np.argmin(epa)), int(np.argmax(epa)),
                 int(np.argmin(vpa)), int(np.argmax(vpa))])
extremes = {g for g in extremes if g in tp_set}

def fps(cand_global, n, seeds_global):
    """farthest point sampling among cand_global (global idx), returning n picks."""
    zc = Z[cand_global]
    picks = [int(np.where(cand_global == s)[0][0]) for s in seeds_global if s in set(cand_global.tolist())]
    if not picks:
        picks = [int(rng.integers(len(cand_global)))]
    mind = np.full(len(cand_global), np.inf, dtype=np.float32)
    for pi in picks:
        mind = np.minimum(mind, np.linalg.norm(zc - zc[pi], axis=1))
    while len(picks) < n:
        nxt = int(np.argmax(mind))
        if DTHR > 0 and mind[nxt] < DTHR:
            break            # this cluster is exhausted: everything left is a near-duplicate
        picks.append(nxt)
        mind = np.minimum(mind, np.linalg.norm(zc - zc[nxt], axis=1))
    return cand_global[np.array(sorted(set(picks)))]

selected = []
got = np.zeros(len(uniq), int)
for k, (c, nc) in enumerate(zip(uniq, alloc)):
    if nc <= 0:
        continue
    cand = train_pool[pool_labels == c]
    seeds = extremes & set(cand.tolist())
    picked = fps(cand, int(nc), seeds)
    got[k] = len(picked)
    selected.append(picked)
short = int(alloc.sum() - got.sum())
if short > 0:
    # clusters that hit the min-distance floor gave budget back: spend it where
    # there is still unexplored space, largest-cover cluster first
    print(f"min_dist floor left {short} picks unspent -- redistributing")
    order = np.argsort(-w)
    for k in order:
        if short <= 0:
            break
        c = uniq[k]
        cand = train_pool[pool_labels == c]
        have = set(selected[k].tolist()) if k < len(selected) else set()
        extra = fps(cand, int(got[k] + short), extremes & set(cand.tolist()))
        add = np.array([g for g in extra if g not in have], dtype=int)
        if len(add):
            selected.append(add[:short])
            short -= len(add[:short])
sel_global = np.sort(np.unique(np.concatenate(selected)))
np.save(args.out_prefix + "selection_idx_emb.npy", sel_global)

# ---------- load raw geometry to write extxyz ----------
def load_blocks(listdata):
    frames = []
    with open(listdata) as f:
        lines = [l for l in f if l.strip()]
    for line in lines:
        aa = line.split(); name = aa[0]
        types = np.loadtxt(name + "type.raw", dtype=int).astype(int)
        nat = len(types); ezero = atmene[types].sum()
        for iset in aa[1:]:
            d = name + iset
            xyz = np.load(os.path.join(d, "coord.npy")).reshape(-1, nat, 3)
            box = np.load(os.path.join(d, "box.npy")).reshape(-1, 3, 3)
            ene = np.load(os.path.join(d, "energy.npy"))
            frc = np.load(os.path.join(d, "force.npy")).reshape(-1, nat, 3)
            # virial.raw is optional: only the systems recomputed with tstress
            # carry it, and the fine-tune can only use --include_stresses if
            # EVERY system in the run has it.
            vp = os.path.join(d, "virial.npy")
            vir = np.load(vp).reshape(-1, 3, 3) if os.path.exists(vp) else None
            for i in range(ene.shape[0]):
                frames.append((types, nat, ezero, xyz[i], box[i], ene[i], frc[i],
                               None if vir is None else vir[i]))
    return frames

frames = load_blocks(args.listdata)
assert len(frames) == N, (len(frames), N)

def to_atoms(fr):
    types, nat, ezero, xyz, box, ene, frc, vir = fr
    a = Atoms(symbols=atmtyp[types].tolist(), positions=xyz,
              cell=box, pbc=[True, True, True])
    kw = dict(energy=ene - ezero, forces=frc)
    if vir is not None:
        # stored virial = -stress * V (DeePMD convention) -> back to the ASE
        # stress in eV/A^3, which is what finetune_mattersim.py reads via
        # atoms.get_stress(voigt=False) / GPa
        kw["stress"] = -np.asarray(vir) / a.get_volume()
    a.calc = SinglePointCalculator(a, **kw)
    return a

nvir = sum(1 for fr in frames if fr[7] is not None)
print(f"frames carrying a virial: {nvir}/{len(frames)}"
      + ("" if nvir in (0, len(frames)) else
         "  <-- MIXED: --include_stresses needs all-or-nothing"))

pref = args.out_prefix
test_global = np.where(is_test)[0]
if args.ntest and args.ntest < len(test_global):
    # spread the held-out subsample over the whole test pool, and keep it
    # stratified by system so no composition drops out of the test set
    picks = []
    for b in np.unique(bid[test_global]):
        sub = test_global[bid[test_global] == b]
        nb = max(1, int(round(args.ntest * len(sub) / len(test_global))))
        picks.append(fps(sub, min(nb, len(sub)), set()))
    test_global = np.sort(np.unique(np.concatenate(picks)))
    print(f"test pool subsampled: {int(is_test.sum())} -> {len(test_global)}")
write(pref + "train_reduce_emb.extxyz",
      [to_atoms(frames[g]) for g in sel_global], format="extxyz")
write(pref + "test_emb.extxyz",
      [to_atoms(frames[g]) for g in test_global], format="extxyz")
np.save(pref + "test_idx_emb.npy", test_global)
print(f"wrote {pref}train_reduce_emb.extxyz ({len(sel_global)}) and "
      f"{pref}test_emb.extxyz ({len(test_global)})")

# ---------- report ----------
def blockhist(idx):
    return {int(c): int((labels[idx] == c).sum()) for c in uniq}
report = dict(
    n_frames=int(N), n_test=int(is_test.sum()),
    n_train_pool=int(len(train_pool)), nsel=int(len(sel_global)),
    alloc_mode=args.alloc,
    per_cluster_pool=blockhist(train_pool),
    per_cluster_selected=blockhist(sel_global),
    fmax_p99_pool=float(np.percentile(fmax[train_pool], 99)),
    fmax_p99_selected=float(np.percentile(fmax[sel_global], 99)),
    fmax_max_selected=float(fmax[sel_global].max()),
    epa_range_selected=[float(epa[sel_global].min()), float(epa[sel_global].max())],
    vpa_range_selected=[float(vpa[sel_global].min()), float(vpa[sel_global].max())],
)
with open(pref + "selection_report_emb.json", "w") as f:
    json.dump(report, f, indent=2)
print(json.dumps(report, indent=2))
print(f"wrote {len(sel_global)} train + {len(test_global)} test structures "
      f"(report: {pref}selection_report_emb.json)")

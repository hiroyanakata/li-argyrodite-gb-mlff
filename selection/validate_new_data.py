#!/usr/bin/env python
"""Judge whether newly-appended training frames are worth keeping, using the same
MatterSim structure descriptor as extract_embeddings.py.

The new DeepMD dataset is the old trajectory with extra frames appended at the
tail. This splits old (reference) vs new and scores every new frame on three
axes -- the ones that decide the value of data in an active-learning loop:

  1. NOVELTY / coverage gain  -- distance in embedding space to the nearest OLD
     frame. Calibrated against the old set's own nearest-neighbour spread: a new
     frame farther than the old p99 radius sits in a region the old data never
     covered, so it genuinely extends the training distribution.
  2. REDUNDANCY               -- nearest-neighbour distance to any already-kept
     frame (old OR earlier new). Below the near-duplicate radius => adds almost
     no information.
  3. DISTRIBUTION SHIFT       -- is the new batch as a whole displaced from the
     old distribution? Reported both in embedding space (centroid shift, energy
     distance) and on the physical meta channels (E/atom, V/atom).

Per-frame verdict (novelty takes precedence over redundancy):
  novel      far from all old frames  -> valuable, extends coverage
  redundant  near an existing frame   -> little new information
  nominal    inside the old cloud but not a near-duplicate

Runs with only numpy/sklearn/matplotlib. If the new-frame embeddings are not yet
available it still reports the meta-based distribution shift so you get a partial
read before the (GPU) descriptor extraction.

Typical use (after extract_new_embeddings.py):
  python validate_new_data.py alldata.txt \
      --emb_all embeddings_all.npy --meta_all emb_meta_all.npy --n_old 32590
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

ATMTYP = np.asarray(["Li", "P", "S", "Br", "I", "Cl"])

p = argparse.ArgumentParser()
p.add_argument("listdata", help="listdata for the NEW (extended) dataset (for meta)")
p.add_argument("--emb_all", default="embeddings_all.npy",
               help="[N,256] embeddings for old++new (from extract_new_embeddings.py)")
p.add_argument("--meta_all", default="emb_meta_all.npy",
               help="[N,4] meta; if missing it is recomputed from listdata")
p.add_argument("--n_old", type=int, default=0,
               help="number of old frames; 0 = infer from old_emb length")
p.add_argument("--old_emb", default="embeddings.npy",
               help="cached old embeddings, used only to infer n_old when --n_old 0")
p.add_argument("--from-store", dest="from_store", action="store_true",
               help="use the growing embedding cache (emb_store.npy/emb_meta_store.npy) "
                    "and read n_old (start of the newest batch) from emb_store.json -- "
                    "no --n_old / --emb_all bookkeeping needed")
p.add_argument("--pca", type=int, default=50)
p.add_argument("--novel_pct", type=float, default=99.0,
               help="old NN-distance percentile above which a new frame counts as novel")
p.add_argument("--dup_pct", type=float, default=5.0,
               help="old NN-distance percentile below which a frame counts as a near-duplicate")
p.add_argument("--calib_sub", type=int, default=8000, help="old subsample for NN calibration")
p.add_argument("--out_prefix", default="new_data")
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()
rng = np.random.default_rng(args.seed)

# --from-store: point at the growing cache and derive n_old from the manifest,
# so the recursive loop never needs a hand-typed --n_old or an _all file.
if args.from_store:
    import emb_cache as ec
    man = ec.load_manifest()
    if man is None:
        raise SystemExit(f"--from-store needs {ec.MANIFEST}; run update_embeddings.py first")
    args.emb_all = ec.STORE
    args.meta_all = ec.META
    if args.n_old <= 0:
        args.n_old = ec.new_start(man)
    if args.n_old <= 0:
        raise SystemExit("no 'new' batch yet in the store (only the bootstrap iteration); "
                         "run update_embeddings.py after appending new frames")
    print(f"[from-store] emb={args.emb_all} meta={args.meta_all} n_old={args.n_old} "
          f"(newest batch = iteration {len(man['iterations']) - 1})")


# ---------- meta (V/atom, E/atom, fmax) straight from DeepMD raw ----------
def load_meta(listdata):
    meta = []
    with open(listdata) as f:
        lines = [l for l in f if l.strip()]
    gidx = 0
    for line in lines:
        aa = line.split()
        name = aa[0]
        types = np.loadtxt(name + "type.raw", dtype=int).astype(int)
        nat = len(types)
        for iset in aa[1:]:
            d = name + iset
            box = np.load(os.path.join(d, "box.npy")).reshape(-1, 3, 3)
            ene = np.load(os.path.join(d, "energy.npy"))
            frc = np.load(os.path.join(d, "force.npy")).reshape(-1, nat, 3)
            for i in range(ene.shape[0]):
                V = box[i, 0, 0] * box[i, 1, 1] * box[i, 2, 2]
                fmax = np.linalg.norm(frc[i], axis=1).max()
                meta.append((V / nat, ene[i] / nat, fmax, gidx))
                gidx += 1
    return np.asarray(meta, dtype=np.float64)


if os.path.exists(args.meta_all):
    meta = np.load(args.meta_all).astype(np.float64)
else:
    print(f"{args.meta_all} not found; recomputing meta from {args.listdata}")
    meta = load_meta(args.listdata)
vpa, epa, fmax = meta[:, 0], meta[:, 1], meta[:, 2]
N = meta.shape[0]

# ---------- old / new split ----------
n_old = args.n_old
if n_old <= 0:
    if os.path.exists(args.old_emb):
        n_old = int(np.load(args.old_emb, mmap_mode="r").shape[0])
    else:
        raise SystemExit("cannot infer n_old: pass --n_old or provide --old_emb")
n_new = N - n_old
if n_new <= 0:
    raise SystemExit(f"no new frames: N={N} n_old={n_old}")
is_new = np.zeros(N, dtype=bool); is_new[n_old:] = True
old_i = np.arange(n_old); new_i = np.arange(n_old, N)
print(f"N={N}  old={n_old}  new={n_new}")


def rstats(x):
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) * 1.4826
    return dict(min=float(x.min()), p50=med, mean=float(x.mean()),
                max=float(x.max()), mad=mad)


def shift(old_x, new_x, label):
    """robust standardized displacement of the new-batch median vs old."""
    med, mad = np.median(old_x), np.median(np.abs(old_x - np.median(old_x))) * 1.4826
    z = (np.median(new_x) - med) / (mad if mad > 0 else 1.0)
    frac_out = float(np.mean((new_x < old_x.min()) | (new_x > old_x.max())))
    return dict(channel=label, median_shift_robust_z=float(z),
                new_frac_outside_old_range=frac_out,
                old=rstats(old_x), new=rstats(new_x))

report = dict(n_frames=int(N), n_old=int(n_old), n_new=int(n_new),
              meta_shift=[shift(epa[old_i], epa[new_i], "E_per_atom"),
                          shift(vpa[old_i], vpa[new_i], "V_per_atom"),
                          shift(fmax[old_i], fmax[new_i], "fmax")])

# ---------- embedding-space novelty / redundancy ----------
have_emb = os.path.exists(args.emb_all)
verdict = np.array(["meta_only"] * N, dtype=object)
nn_old = np.full(N, np.nan)      # distance to nearest OLD frame
nn_kept = np.full(N, np.nan)     # distance to nearest already-kept frame
if have_emb:
    E = np.load(args.emb_all).astype(np.float64)
    assert E.shape[0] == N, (E.shape, N)
    keep = E[old_i].std(axis=0) > 1e-8          # scale on OLD channels only
    E = E[:, keep]
    scaler = StandardScaler().fit(E[old_i])
    Xs = scaler.transform(E)
    ndim = min(args.pca, Xs.shape[1])
    pca = PCA(n_components=ndim, whiten=True, random_state=args.seed).fit(Xs[old_i])
    Z = pca.transform(Xs).astype(np.float32)
    print(f"embedding space: kept {int(keep.sum())} channels -> PCA {ndim} "
          f"(old cumvar {pca.explained_variance_ratio_.sum():.3f})")

    # distance from every frame to the nearest OLD frame
    nn = NearestNeighbors(n_neighbors=1).fit(Z[old_i])
    nn_old[new_i] = nn.kneighbors(Z[new_i])[0][:, 0]
    # old self-distance (k=2, drop self) -> calibrates the in-distribution radius
    sub = old_i if n_old <= args.calib_sub else rng.choice(old_i, args.calib_sub, replace=False)
    d_old_self = NearestNeighbors(n_neighbors=2).fit(Z[old_i]).kneighbors(Z[sub])[0][:, 1]
    novel_r = float(np.percentile(d_old_self, args.novel_pct))
    dup_r = float(np.percentile(d_old_self, args.dup_pct))

    # redundancy: greedily grow the kept set old->new in time order; a new frame's
    # nn_kept is its distance to the nearest frame kept before it (old + accepted new)
    kept = list(old_i)
    knn = NearestNeighbors(n_neighbors=1).fit(Z[old_i])
    pending = []
    for g in new_i:
        d = knn.kneighbors(Z[g:g + 1])[0][0, 0]
        nn_kept[g] = d
        if d >= dup_r:                      # non-duplicate -> becomes a reference
            pending.append(g)
            if len(pending) >= 256:         # refit in batches for speed
                kept.extend(pending); pending = []
                knn = NearestNeighbors(n_neighbors=1).fit(Z[kept])
    if pending:
        kept.extend(pending)

    v = np.array(["nominal"] * N, dtype=object)
    v[new_i[nn_old[new_i] >= novel_r]] = "novel"
    red = (nn_old[new_i] < novel_r) & (nn_kept[new_i] < dup_r)
    v[new_i[red]] = "redundant"
    verdict = v

    report["embedding"] = dict(
        pca_dims=int(ndim), pca_old_cumvar=float(pca.explained_variance_ratio_.sum()),
        novel_radius=novel_r, dup_radius=dup_r,
        novel_pct=args.novel_pct, dup_pct=args.dup_pct,
        centroid_shift=float(np.linalg.norm(Z[new_i].mean(0) - Z[old_i].mean(0))),
        nn_old_to_new=rstats(nn_old[new_i]),
    )
else:
    print(f"{args.emb_all} not found -> meta-only report "
          f"(run extract_new_embeddings.py on a GPU for novelty/redundancy)")

# ---------- verdict counts + recommendation ----------
vc = {k: int((verdict[new_i] == k).sum()) for k in np.unique(verdict[new_i])}
report["verdict_counts"] = vc
if have_emb:
    nn_new = vc.get("novel", 0)
    frac_novel = nn_new / n_new
    frac_red = vc.get("redundant", 0) / n_new
    if frac_novel >= 0.30:
        rec = (f"KEEP: {frac_novel:.0%} of new frames are novel -- they extend the "
               f"training coverage substantially.")
    elif frac_novel >= 0.05:
        rec = (f"KEEP (selectively): {frac_novel:.0%} novel, {frac_red:.0%} redundant. "
               f"Novel frames add value; run FPS selection to drop the redundant bulk.")
    else:
        rec = (f"LOW VALUE: only {frac_novel:.0%} novel, {frac_red:.0%} redundant -- the "
               f"new batch mostly repeats regions the old data already covers.")
    report["recommendation"] = rec
    report["novel_frac"] = float(frac_novel)
    report["redundant_frac"] = float(frac_red)
    print(f"\nnovel={vc.get('novel',0)}  redundant={vc.get('redundant',0)}  "
          f"nominal={vc.get('nominal',0)}")
    print(rec)

with open(f"{args.out_prefix}_report.json", "w") as f:
    json.dump(report, f, indent=2)

# per-new-frame CSV
import csv
with open(f"{args.out_prefix}_verdict.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["global_idx", "V_per_atom", "E_per_atom", "fmax",
                "nn_dist_to_old", "nn_dist_to_kept", "verdict"])
    for g in new_i:
        w.writerow([int(g), f"{vpa[g]:.5f}", f"{epa[g]:.5f}", f"{fmax[g]:.4f}",
                    "" if np.isnan(nn_old[g]) else f"{nn_old[g]:.4f}",
                    "" if np.isnan(nn_kept[g]) else f"{nn_kept[g]:.4f}",
                    verdict[g]])

# ---------- figure ----------
npan = 4 if have_emb else 3
fig, ax = plt.subplots(1, npan, figsize=(5.2 * npan, 4.6))


def dual_hist(a, ch, name, logy=False):
    lo, hi = np.percentile(meta[:, ch], [0.2, 99.8])
    bins = np.linspace(lo, hi, 60)
    a.hist(meta[old_i, ch], bins=bins, density=True, alpha=0.5, label=f"old ({n_old})")
    a.hist(meta[new_i, ch], bins=bins, density=True, alpha=0.5, label=f"new ({n_new})")
    a.set_title(name); a.set_xlabel(name); a.legend()
    if logy:
        a.set_yscale("log")


dual_hist(ax[0], 1, "E / atom")
dual_hist(ax[1], 0, "V / atom")
dual_hist(ax[2], 2, "fmax", logy=True)
if have_emb:
    Z2 = Z[:, :2]
    ax[3].scatter(Z2[old_i, 0], Z2[old_i, 1], s=3, c="0.7", alpha=0.4, label="old")
    cmap = {"nominal": "tab:blue", "redundant": "tab:orange", "novel": "tab:red"}
    for k, col in cmap.items():
        m = new_i[verdict[new_i] == k]
        if len(m):
            ax[3].scatter(Z2[m, 0], Z2[m, 1], s=6, c=col, alpha=0.7,
                          label=f"{k} ({len(m)})")
    ax[3].set_title("embedding PCA (old vs new)")
    ax[3].set_xlabel("PC1"); ax[3].set_ylabel("PC2"); ax[3].legend()
plt.tight_layout()
plt.savefig(f"{args.out_prefix}_validation.png", dpi=130)

print(f"\nwrote {args.out_prefix}_report.json, {args.out_prefix}_verdict.csv, "
      f"{args.out_prefix}_validation.png")

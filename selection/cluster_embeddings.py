#!/usr/bin/env python
"""Cluster MatterSim structure embeddings into structural regimes and characterise them.

Input:
  embeddings.npy [N,256]  mean|std pooled M3GNet node features (from extract_embeddings.py)
  emb_meta.npy   [N,4]    columns: V_per_atom, E_per_atom, fmax, global_idx

Pipeline:
  StandardScaler -> PCA(whiten) -> KMeans (k chosen by silhouette over a range,
  or fixed via --k). HDBSCAN also reported for reference (density clusters +
  noise/outliers). Each cluster is characterised by its volume / energy / force
  distribution so the structural regimes can be interpreted. A 2D PCA scatter
  (coloured by cluster, by V/atom, by E/atom) is written for a quick look.

Outputs:
  cluster_labels.npy   [N] int   KMeans label per frame (aligned to emb order)
  cluster_report.json  per-cluster statistics + chosen k + silhouette
  cluster_overview.png  2D PCA panels
"""
import argparse
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

p = argparse.ArgumentParser()
p.add_argument("--emb", default="embeddings.npy")
p.add_argument("--meta", default="emb_meta.npy")
p.add_argument("--k", type=int, default=0, help="fixed KMeans k; 0 = auto-pick by silhouette")
p.add_argument("--kmin", type=int, default=4)
p.add_argument("--kmax", type=int, default=16)
p.add_argument("--pca", type=int, default=50, help="PCA dims before clustering")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out_prefix", default="",
               help="prefix for every output file. Empty keeps the historical "
                    "names cluster_labels.npy / cluster_model.npz / ... which "
                    "OVERWRITE the k=8 model the Li094 cycle assigns against, "
                    "so pass a prefix when clustering a different frame set")
args = p.parse_args()
PREF = args.out_prefix

E = np.load(args.emb).astype(np.float64)
meta = np.load(args.meta).astype(np.float64)
vpa, epa, fmax, gidx = meta[:, 0], meta[:, 1], meta[:, 2], meta[:, 3]
N = E.shape[0]
print(f"loaded embeddings {E.shape}, meta {meta.shape}")

# drop dead (zero-variance) feature channels, then standardize + PCA-whiten
keep = E.std(axis=0) > 1e-8
E = E[:, keep]
Xs = StandardScaler().fit_transform(E)
ndim = min(args.pca, Xs.shape[1])
pca = PCA(n_components=ndim, whiten=True, random_state=args.seed)
Z = pca.fit_transform(Xs)
print(f"kept {keep.sum()}/{keep.size} channels; PCA -> {ndim} dims "
      f"(cumvar {pca.explained_variance_ratio_.sum():.3f})")

# ---------- choose k by silhouette (on a subsample for speed) ----------
rng = np.random.default_rng(args.seed)
sub = rng.choice(N, min(5000, N), replace=False)
if args.k > 0:
    best_k = args.k
    best_sil = None
else:
    best_k, best_sil = None, -1.0
    for k in range(args.kmin, args.kmax + 1):
        lab = KMeans(n_clusters=k, n_init=4, random_state=args.seed).fit_predict(Z[sub])
        sil = silhouette_score(Z[sub], lab)
        print(f"  k={k:2d}  silhouette={sil:.4f}")
        if sil > best_sil:
            best_sil, best_k = sil, k
print(f"chosen k={best_k} (silhouette={best_sil})")

km = KMeans(n_clusters=best_k, n_init=10, random_state=args.seed).fit(Z)
labels = km.labels_
np.save(PREF + "cluster_labels.npy", labels.astype(np.int32))

# ---------- persist the fitted pipeline so future frames can be assigned ----------
# Stored as plain numpy so assign_clusters.py can reproduce the transform without
# sklearn/version pinning: E[:,keep] -> (x-mean)/scale -> (x-pca_mean)@comp.T
# -> /sqrt(expl_var) (whiten) -> argmin dist to kmeans centers.
scaler = StandardScaler().fit(E)  # refit to expose params (same fit as above)
per_center_radius = np.array([                      # p99 member->center dist per cluster
    np.percentile(np.linalg.norm(Z[labels == c] - km.cluster_centers_[c], axis=1), 99)
    if (labels == c).any() else 0.0
    for c in range(best_k)], dtype=np.float64)
np.savez(PREF + "cluster_model.npz",
         keep_mask=keep,
         scaler_mean=scaler.mean_, scaler_scale=scaler.scale_,
         pca_mean=pca.mean_, pca_components=pca.components_,
         pca_explained_variance=pca.explained_variance_, pca_whiten=np.array(True),
         kmeans_centers=km.cluster_centers_,
         cluster_radius_p99=per_center_radius,
         k=np.array(best_k), pca_dims=np.array(ndim), seed=np.array(args.seed))
print(f"saved {PREF}cluster_model.npz (k={best_k}, pca={ndim})")

# ---------- characterise clusters ----------
def stats(x):
    return dict(min=float(x.min()), p50=float(np.median(x)), max=float(x.max()),
                mean=float(x.mean()))

clusters = []
order = sorted(range(best_k), key=lambda c: vpa[labels == c].mean())
for rank, c in enumerate(order):
    m = labels == c
    clusters.append(dict(
        cluster=int(c), rank_by_volume=rank, n=int(m.sum()),
        frac=float(m.mean()),
        Vpa=stats(vpa[m]), Epa=stats(epa[m]), fmax=stats(fmax[m]),
        time_span=[int(gidx[m].min()), int(gidx[m].max())],
    ))
report = dict(k=int(best_k), silhouette=(None if best_sil is None else float(best_sil)),
              n_frames=int(N), pca_dims=int(ndim),
              pca_cumvar=float(pca.explained_variance_ratio_.sum()),
              clusters=clusters)
with open(PREF + "cluster_report.json", "w") as f:
    json.dump(report, f, indent=2)
print(json.dumps(report, indent=2))

# ---------- 2D visualisation ----------
Z2 = Z[:, :2] if ndim >= 2 else np.c_[Z[:, 0], np.zeros(N)]
fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))
sc0 = ax[0].scatter(Z2[:, 0], Z2[:, 1], c=labels, cmap="tab20", s=3, alpha=0.5)
ax[0].set_title(f"KMeans k={best_k}"); plt.colorbar(sc0, ax=ax[0], label="cluster")
sc1 = ax[1].scatter(Z2[:, 0], Z2[:, 1], c=vpa, cmap="viridis", s=3, alpha=0.5)
ax[1].set_title("V / atom"); plt.colorbar(sc1, ax=ax[1])
sc2 = ax[2].scatter(Z2[:, 0], Z2[:, 1], c=epa, cmap="magma", s=3, alpha=0.5)
ax[2].set_title("E / atom"); plt.colorbar(sc2, ax=ax[2])
for a in ax:
    a.set_xlabel("PC1"); a.set_ylabel("PC2")
plt.tight_layout()
plt.savefig(PREF + "cluster_overview.png", dpi=130)
print(f"wrote {PREF}cluster_labels.npy, {PREF}cluster_report.json, {PREF}cluster_overview.png")

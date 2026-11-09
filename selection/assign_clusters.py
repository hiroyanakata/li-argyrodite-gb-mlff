#!/usr/bin/env python
"""Classify NEW frames into the existing embedding clusters (or flag them as novel).

Uses the pipeline persisted by cluster_embeddings.py (``cluster_model.npz``:
channel mask + StandardScaler + PCA-whiten + KMeans centers). Each new frame is
embedded with the exact MatterSim descriptor recipe, projected through the frozen
transform, and assigned to its nearest KMeans center. A frame whose distance to
that center exceeds the cluster's p99 member radius is flagged ``novel`` -- it
does not fit any existing regime well and is a candidate for a NEW cluster
(re-run cluster_embeddings.py on the combined set if novel frames pile up).

Two input modes:
  --emb E.npy         use precomputed embeddings [n,256] (no GPU / mattersim needed)
  --listdata L.txt    embed the frames on GPU (needs mattersim + --ckpt)

Outputs:
  assign_report.json   per-cluster assigned counts, novel count, distance stats
  assign_verdict.csv   per frame: global_idx, cluster, dist, radius, novel
With --append (requires --listdata for geometry): each cluster's assigned
non-novel frames are appended as a fresh set.NNN under <out_prefix>.cluster<c>/,
and novel frames are collected into <out_prefix>.clusterNEW/ for review.

Usage:
  # precomputed embeddings
  python assign_clusters.py --emb new_emb.npy --meta new_meta.npy
  # embed + classify + append back into the DeepMD cluster datasets
  python assign_clusters.py --listdata newdata.txt --ckpt matersim_model.pth \
      --append --out_prefix data/Li094
"""
import argparse
import json
import os

import numpy as np

ATMTYP = np.asarray(["Li", "P", "S", "Br", "I", "Cl"])

p = argparse.ArgumentParser()
p.add_argument("--model", default="cluster_model.npz")
p.add_argument("--from-store", dest="from_store", action="store_true",
               help="classify the newest batch straight from the growing cache "
                    "(emb_store.npy rows since emb_store.json's last iteration) -- "
                    "no GPU re-embedding; combine with --append --listdata for geometry")
p.add_argument("--emb", help="precomputed embeddings [n,256]")
p.add_argument("--meta", help="precomputed meta [n,4] (V/atom,E/atom,fmax,gidx)")
p.add_argument("--listdata", help="listdata to embed on GPU (needs mattersim+--ckpt)")
p.add_argument("--ckpt", default="matersim_model.pth")
p.add_argument("--batch_size", type=int, default=32)
p.add_argument("--append", action="store_true",
               help="append assigned frames back into <out_prefix>.cluster<c> DeepMD sets")
p.add_argument("--out_prefix", default="data/Li094")
p.add_argument("--emb_out", default="new_emb_assign.npy",
               help="where to cache freshly computed embeddings (listdata mode)")
args = p.parse_args()

m = np.load(args.model)
keep = m["keep_mask"]
centers = m["kmeans_centers"]
radius = m["cluster_radius_p99"]
K = int(m["k"])


def transform(E_raw):
    """Reproduce StandardScaler -> PCA(whiten) exactly, in numpy."""
    E = E_raw[:, keep].astype(np.float64)
    Xs = (E - m["scaler_mean"]) / m["scaler_scale"]
    Z = (Xs - m["pca_mean"]) @ m["pca_components"].T
    if bool(m["pca_whiten"]):
        Z = Z / np.sqrt(m["pca_explained_variance"])
    return Z


def embed_listdata(listdata):
    """MatterSim (M3GNet) embedding -- recipe identical to extract_embeddings.py."""
    import torch
    from ase import Atoms
    # ASE >=3.24 moved full_3x3_to_voigt_6_stress; mattersim 1.2.0 wants old path.
    import ase.constraints as _c
    if not hasattr(_c, "full_3x3_to_voigt_6_stress"):
        from ase.stress import full_3x3_to_voigt_6_stress as _f3, voigt_6_to_full_3x3_stress as _v6
        _c.full_3x3_to_voigt_6_stress = _f3
        _c.voigt_6_to_full_3x3_stress = _v6
    from mattersim.forcefield.potential import Potential, batch_to_dict
    from mattersim.datasets.utils.build import build_dataloader

    frames, meta = [], []
    with open(listdata) as f:
        lines = [l for l in f if l.strip()]
    gidx = 0
    for line in lines:
        aa = line.split(); name = aa[0]
        types = np.loadtxt(name + "type.raw", dtype=int).astype(int)
        nat = len(types); symbols = ATMTYP[types].tolist()
        for iset in aa[1:]:
            d = name + iset
            xyz = np.load(os.path.join(d, "coord.npy")).reshape(-1, nat, 3)
            box = np.load(os.path.join(d, "box.npy")).reshape(-1, 3, 3)
            ene = np.load(os.path.join(d, "energy.npy"))
            frc = np.load(os.path.join(d, "force.npy")).reshape(-1, nat, 3)
            for i in range(ene.shape[0]):
                frames.append(Atoms(symbols=symbols, positions=xyz[i], cell=box[i], pbc=True))
                V = box[i, 0, 0] * box[i, 1, 1] * box[i, 2, 2]
                meta.append((V / nat, ene[i] / nat,
                             np.linalg.norm(frc[i], axis=1).max(), gidx))
                gidx += 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pot = Potential.from_checkpoint(load_path=args.ckpt, load_training_state=False, device=device)
    model = pot.model; model.eval()
    buf = {}
    model.final.register_forward_pre_hook(lambda mm, inp: buf.__setitem__("x", inp[0].detach()))
    dl = build_dataloader(frames, only_inference=True, shuffle=False,
                          batch_size=args.batch_size, cutoff=5.0, threebody_cutoff=4.0)
    embs = []
    with torch.no_grad():
        for gb in dl:
            gb.to(device); inp = batch_to_dict(gb); model(inp)
            x = buf["x"]; b = inp["batch"].long(); ng = int(inp["num_graphs"]); dfeat = x.shape[1]
            ones = torch.ones(b.shape[0], device=x.device, dtype=x.dtype)
            cnt = torch.zeros(ng, device=x.device, dtype=x.dtype).index_add_(0, b, ones).clamp_min(1.0)
            s = torch.zeros(ng, dfeat, device=x.device, dtype=x.dtype).index_add_(0, b, x)
            ssq = torch.zeros(ng, dfeat, device=x.device, dtype=x.dtype).index_add_(0, b, x * x)
            mean = s / cnt.unsqueeze(1)
            std = (ssq / cnt.unsqueeze(1) - mean * mean).clamp_min(0).sqrt()
            embs.append(torch.cat([mean, std], dim=1).cpu().numpy().astype(np.float32))
    return np.concatenate(embs, axis=0), np.asarray(meta, dtype=np.float32)


# ---------- get embeddings ----------
if args.from_store:
    import emb_cache as ec
    man = ec.load_manifest()
    if man is None:
        raise SystemExit(f"--from-store needs {ec.MANIFEST}; run update_embeddings.py first")
    ns = ec.new_start(man)
    if ns <= 0:
        raise SystemExit("no 'new' batch yet in the store (only the bootstrap iteration)")
    E = np.load(ec.STORE)[ns:].astype(np.float32)
    meta = np.load(ec.META)[ns:].astype(np.float64)
    print(f"[from-store] classifying newest batch: {E.shape[0]} frames (global idx {ns}..)")
elif args.emb:
    E = np.load(args.emb).astype(np.float32)
    meta = np.load(args.meta).astype(np.float64) if args.meta else None
elif args.listdata:
    E, meta = embed_listdata(args.listdata)
    np.save(args.emb_out, E)
    print(f"embedded {E.shape[0]} frames -> {args.emb_out}")
else:
    raise SystemExit("give --from-store, --emb, or --listdata")

# ---------- assign to nearest center + novelty flag ----------
Z = transform(E)
d = np.linalg.norm(Z[:, None, :] - centers[None, :, :], axis=2)   # [n,K]
assign = d.argmin(axis=1)
dmin = d[np.arange(len(Z)), assign]
novel = dmin > radius[assign]
gidx = meta[:, 3].astype(int) if meta is not None else np.arange(len(Z))

# ---------- report ----------
report = {"n_new": int(len(Z)), "k": K, "n_novel": int(novel.sum()),
          "novel_frac": float(novel.mean()),
          "per_cluster_assigned": {int(c): int(((assign == c) & ~novel).sum()) for c in range(K)},
          "dist_to_center": {"min": float(dmin.min()), "p50": float(np.median(dmin)),
                             "max": float(dmin.max())}}
with open("assign_report.json", "w") as f:
    json.dump(report, f, indent=2)
import csv
with open("assign_verdict.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["global_idx", "cluster", "dist", "radius", "novel"])
    for i in range(len(Z)):
        w.writerow([int(gidx[i]), int(assign[i]), f"{dmin[i]:.4f}",
                    f"{radius[assign[i]]:.4f}", int(novel[i])])
print(json.dumps(report, indent=2))
if novel.any():
    print(f"** {int(novel.sum())} frames are NOVEL (beyond every cluster radius) -- "
          "candidates for a new regime; re-cluster the combined set if they accumulate.")

# ---------- optional: append geometry back into the cluster DeepMD datasets ----------
if args.append:
    if not args.listdata:
        raise SystemExit("--append needs --listdata to source the raw geometry")
    # reload raw npy in frame order (native shape, no reshape)
    blk = {k: [] for k in ("coord", "box", "energy", "force")}
    type_txt, nat_seen = None, None
    with open(args.listdata) as f:
        lines = [l for l in f if l.strip()]
    for line in lines:
        aa = line.split(); name = aa[0]
        types = np.loadtxt(name + "type.raw", dtype=int); nat = len(types)
        if nat_seen is None:
            nat_seen = nat; type_txt = open(name + "type.raw").read()
        for iset in aa[1:]:
            dd = name + iset
            for k in blk:
                blk[k].append(np.load(os.path.join(dd, k + ".npy")))
    data = {k: np.concatenate(v, axis=0) for k, v in blk.items()}
    if data["energy"].shape[0] != len(Z):
        raise SystemExit(
            f"--append geometry has {data['energy'].shape[0]} frames but "
            f"{len(Z)} were classified; --listdata must be exactly the batch that "
            f"was embedded (the newest store batch when --from-store).")

    def next_set(dirpath):
        os.makedirs(dirpath, exist_ok=True)
        n = [int(x[4:]) for x in os.listdir(dirpath) if x.startswith("set.") and x[4:].isdigit()]
        return os.path.join(dirpath, f"set.{(max(n) + 1) if n else 0:03d}")

    def write_set(dirpath, idx):
        with open(os.path.join(dirpath, "type.raw"), "w") as tf:
            tf.write(type_txt)
        sd = next_set(dirpath); os.makedirs(sd, exist_ok=True)
        for k in ("coord", "box", "energy", "force"):
            np.save(os.path.join(sd, k + ".npy"), data[k][idx])
        return sd

    for c in range(K):
        idx = np.where((assign == c) & ~novel)[0]
        if idx.size:
            sd = write_set(f"{args.out_prefix}.cluster{c}", idx)
            print(f"appended {idx.size} frames to {sd}")
    nidx = np.where(novel)[0]
    if nidx.size:
        sd = write_set(f"{args.out_prefix}.clusterNEW", nidx)
        print(f"parked {nidx.size} novel frames in {sd} (review before training)")

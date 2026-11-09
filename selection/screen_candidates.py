#!/usr/bin/env python
"""Screen candidate first-principles (Quantum ESPRESSO) inputs BEFORE spending the
DFT budget: is each candidate structure new to the training set, or does the store
already cover it?

The existing loop judges frames *after* the DFT is done (update_embeddings.py ->
validate_new_data.py). The same descriptor answers the cheaper question first: a
``.pwin`` file fully determines the geometry, so it can be embedded with the very
same MatterSim encoder (``matersim_model.pth``) and compared against the cumulative
store (``emb_store.npy``, 43k frames) without running a single SCF step.

Three questions per candidate, in the frozen ``cluster_model.npz`` PCA space so the
numbers are directly comparable with assign_clusters.py:

  1. NOVELTY vs the store   -- distance to the nearest already-computed frame, plus
     the per-cluster p99 member radius. The raw NN distance is a *sampling-density*
     measure as much as a novelty one (the store is consecutive MD frames, the
     candidates are every-N-steps snapshots), so the novel radius is calibrated on a
     store thinned until its own frame spacing matches the candidates' -- otherwise
     any sparse batch scores "novel" regardless of its physics.
  2. REDUNDANCY inside the batch -- candidates coming from the same NPT run at
     10-step intervals are mostly copies of each other. A farthest-point sweep
     *seeded with the whole store* ranks them by marginal information gain and
     cuts the batch where the gain drops below the duplicate radius.
  3. SANITY  -- V/atom against the store range and the shortest interatomic
     distance (a collapsed pair means a wasted, non-converging SCF).

Verdicts (per candidate):
  novel      outside every existing regime -> highest value, compute first
  covered    inside a regime the store already samples -> re-measures known physics
  redundant  a stored frame already sits on top of it -> skip
  duplicate-of-batch  another, better-ranked candidate already covers it -> skip
  suspect    geometry looks broken (too-short bond) -> fix or drop

Usage (GPU for step 1, everything else CPU):
  python screen_candidates.py newCandidate --ckpt matersim_model.pth
  # re-scoring only, no GPU (embeddings cached from the first run):
  python screen_candidates.py newCandidate --emb candidate_emb.npy
"""
import argparse
import csv
import glob
import json
import os
import re

import numpy as np

ATMTYP = ["Li", "P", "S", "Br", "I", "Cl"]
BOHR = 0.529177210903


# ------------------------------------------------------------------ pwin parser
def parse_pwin(path):
    """Geometry out of a pw.x input: (symbols, positions[A], cell[3x3 A])."""
    with open(path) as f:
        lines = f.read().splitlines()

    nat = None
    for l in lines:
        mm = re.match(r"\s*nat\s*=\s*([0-9]+)", l, re.I)
        if mm:
            nat = int(mm.group(1))
            break
    if nat is None:
        raise ValueError(f"{path}: no nat in &SYSTEM")

    def unit_of(line, default):
        mm = re.search(r"[{(]?\s*(angstrom|bohr|alat|crystal|crystal_sg)\s*[)}]?\s*$",
                       line.strip(), re.I)
        return mm.group(1).lower() if mm else default

    cell = None
    pos_u = None
    symbols, xyz = [], []
    i = 0
    while i < len(lines):
        l = lines[i]
        if re.match(r"\s*CELL_PARAMETERS", l, re.I):
            u = unit_of(l, "alat")
            rows = []
            j = i + 1
            while len(rows) < 3:
                s = lines[j].split()
                if s:
                    rows.append([float(x) for x in s[:3]])
                j += 1
            cell = np.asarray(rows, dtype=float)
            if u == "bohr":
                cell *= BOHR
            elif u == "alat":
                raise ValueError(f"{path}: CELL_PARAMETERS in alat is not supported")
            i = j
            continue
        if re.match(r"\s*ATOMIC_POSITIONS", l, re.I):
            pos_u = unit_of(l, "alat")
            j = i + 1
            while len(xyz) < nat and j < len(lines):
                s = lines[j].split()
                if len(s) >= 4 and s[0].isalpha():
                    try:
                        v = [float(x) for x in s[1:4]]
                    except ValueError:
                        j += 1
                        continue
                    symbols.append(s[0])
                    xyz.append(v)
                j += 1
            i = j
            continue
        i += 1

    if cell is None:
        raise ValueError(f"{path}: no CELL_PARAMETERS (ibrav=0 expected)")
    if len(xyz) != nat:
        raise ValueError(f"{path}: read {len(xyz)} positions, expected nat={nat}")
    pos = np.asarray(xyz, dtype=float)
    if pos_u == "bohr":
        pos *= BOHR
    elif pos_u == "crystal":
        pos = pos @ cell
    elif pos_u == "alat":
        raise ValueError(f"{path}: ATOMIC_POSITIONS in alat is not supported")
    return symbols, pos, cell


def min_pair_distance(pos, cell):
    """Shortest interatomic distance under PBC (27-image search, exact for these cells)."""
    inv = np.linalg.inv(cell)
    frac = pos @ inv
    frac -= np.floor(frac)
    cart = frac @ cell
    shifts = np.array([[a, b, c] for a in (-1, 0, 1) for b in (-1, 0, 1)
                       for c in (-1, 0, 1)], dtype=float) @ cell
    best = np.inf
    n = len(cart)
    for s in shifts:
        d = np.linalg.norm(cart[:, None, :] - (cart[None, :, :] + s), axis=2)
        if np.allclose(s, 0):
            d[np.arange(n), np.arange(n)] = np.inf
        best = min(best, float(d.min()))
    return best


# ------------------------------------------------------------------- descriptor
def embed(frames, ckpt, batch_size=16):
    """MatterSim (M3GNet) structure descriptor -- recipe identical to
    extract_embeddings.py / update_embeddings.py (mean|std pooled model.final input)."""
    import torch
    # ASE >=3.24 moved full_3x3_to_voigt_6_stress; mattersim 1.2.0 wants the old path.
    import ase.constraints as _c
    if not hasattr(_c, "full_3x3_to_voigt_6_stress"):
        from ase.stress import full_3x3_to_voigt_6_stress as _f3, voigt_6_to_full_3x3_stress as _v6
        _c.full_3x3_to_voigt_6_stress = _f3
        _c.voigt_6_to_full_3x3_stress = _v6
    from mattersim.forcefield.potential import Potential, batch_to_dict
    from mattersim.datasets.utils.build import build_dataloader

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"encoding {len(frames)} candidates on {device} ...", flush=True)
    pot = Potential.from_checkpoint(load_path=ckpt, load_training_state=False, device=device)
    model = pot.model
    model.eval()
    buf = {}
    model.final.register_forward_pre_hook(lambda mm, inp: buf.__setitem__("x", inp[0].detach()))
    dl = build_dataloader(frames, only_inference=True, shuffle=False,
                          batch_size=batch_size, cutoff=5.0, threebody_cutoff=4.0)
    out = []
    with torch.no_grad():
        for bi, gb in enumerate(dl):
            gb.to(device)
            inp = batch_to_dict(gb)
            model(inp)
            x = buf["x"]
            b = inp["batch"].long()
            ng = int(inp["num_graphs"])
            dfeat = x.shape[1]
            ones = torch.ones(b.shape[0], device=x.device, dtype=x.dtype)
            cnt = torch.zeros(ng, device=x.device, dtype=x.dtype).index_add_(0, b, ones).clamp_min(1.0)
            s = torch.zeros(ng, dfeat, device=x.device, dtype=x.dtype).index_add_(0, b, x)
            ssq = torch.zeros(ng, dfeat, device=x.device, dtype=x.dtype).index_add_(0, b, x * x)
            mean = s / cnt.unsqueeze(1)
            std = (ssq / cnt.unsqueeze(1) - mean * mean).clamp_min(0).sqrt()
            out.append(torch.cat([mean, std], dim=1).cpu().numpy().astype(np.float32))
            if bi % 5 == 0:
                print(f"  batch {bi}: {sum(e.shape[0] for e in out)}/{len(frames)}", flush=True)
    return np.concatenate(out, axis=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("candidates", nargs="?", default="",
                   help="directory of .pwin files (or a glob)")
    p.add_argument("--traj", default="",
                   help="MD trajectory file/glob/dir instead of .pwin (extxyz, .traj, "
                        "anything ase.io can read) -- use with --stride")
    p.add_argument("--stride", type=int, default=1,
                   help="keep every Nth trajectory frame (decorrelate before scoring)")
    p.add_argument("--max_frames", type=int, default=0,
                   help="stop after this many kept frames (0 = no limit)")
    p.add_argument("--ckpt", default="matersim_model.pth")
    p.add_argument("--store", default="emb_store.npy")
    p.add_argument("--store_meta", default="emb_meta_store.npy")
    p.add_argument("--model", default="cluster_model.npz")
    p.add_argument("--emb", help="reuse cached candidate embeddings (skips the GPU pass)")
    p.add_argument("--emb_out", default="",
                   help="where to cache the embeddings (default <out_prefix>_emb.npy)")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--novel_pct", type=float, default=99.0)
    p.add_argument("--dup_pct", type=float, default=5.0)
    p.add_argument("--calib_sub", type=int, default=8000)
    p.add_argument("--min_dist", type=float, default=1.2,
                   help="shortest interatomic distance [A] below which a candidate is 'suspect'")
    p.add_argument("--gain_floor", type=float, default=0.0,
                   help="marginal-gain cut for the recommended list "
                        "(0 = use the density-matched reference spacing)")
    p.add_argument("--nsel", type=int, default=0,
                   help="hard cap on the recommended list (0 = let the duplicate radius decide)")
    p.add_argument("--out_prefix", default="candidate_screen")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    # keep every run's cache next to its own outputs, so a trajectory run never
    # silently overwrites the .pwin run's embeddings
    if not args.emb_out:
        args.emb_out = f"{args.out_prefix}_emb.npy"
    if not args.candidates and not args.traj:
        raise SystemExit("give a .pwin directory/glob, or --traj <trajectory>")

    # ---------------- candidates ----------------
    # Two sources: pw.x inputs (one structure per file) or MD trajectories (any
    # ASE-readable file -- extxyz, .traj, ...). Trajectory frames are consecutive
    # and therefore heavily correlated, so --stride thins them before scoring.
    names, runs, steps = [], [], []
    syms, poss, cells = [], [], []
    if args.traj:
        from ase.io import iread
        tfiles = sorted(glob.glob(args.traj)) if not os.path.isdir(args.traj) else sorted(
            glob.glob(os.path.join(args.traj, "*")))
        if not tfiles:
            raise SystemExit(f"no trajectory matched {args.traj}")
        for tf in tfiles:
            tag = os.path.splitext(os.path.basename(tf))[0]
            kept_here = 0
            for k, at in enumerate(iread(tf)):
                if k % args.stride:
                    continue
                if args.max_frames and len(names) >= args.max_frames:
                    break
                syms.append(at.get_chemical_symbols())
                poss.append(at.get_positions())
                cells.append(np.asarray(at.get_cell()))
                names.append(f"{tag}:{k}")
                runs.append(tag)
                steps.append(k)
                kept_here += 1
            print(f"  {tf}: kept {kept_here} frames (stride {args.stride})")
        files = names
    else:
        if os.path.isdir(args.candidates):
            files = sorted(glob.glob(os.path.join(args.candidates, "*.pwin")))
        else:
            files = sorted(glob.glob(args.candidates))
        if not files:
            raise SystemExit(f"no .pwin found under {args.candidates}")
        for f in files:
            s, x, c = parse_pwin(f)
            syms.append(s); poss.append(x); cells.append(c)
            n = os.path.basename(f)
            names.append(n)
            # trajectory tag: <prefix>.<run>.<step>.pwin  -> run, step
            mm = re.search(r"\.([A-Za-z]+[0-9]+)\.([0-9]+)\.pwin$", n)
            runs.append(mm.group(1) if mm else "?")
            steps.append(int(mm.group(2)) if mm else -1)
    print(f"{len(names)} candidate structures")

    vpa = np.array([abs(np.linalg.det(c)) / len(s) for s, c in zip(syms, cells)])
    dmin_pair = np.array([min_pair_distance(x, c) for x, c in zip(poss, cells)])
    nat = np.array([len(s) for s in syms])
    runs = np.asarray(runs); steps = np.asarray(steps)

    # ---------------- embeddings ----------------
    if args.emb and os.path.exists(args.emb):
        C = np.load(args.emb).astype(np.float32)
        if C.shape[0] != len(names):
            raise SystemExit(f"{args.emb} has {C.shape[0]} rows but {len(names)} candidates")
        print(f"reusing {args.emb} {C.shape}")
    else:
        from ase import Atoms
        frames = [Atoms(symbols=s, positions=x, cell=c, pbc=True)
                  for s, x, c in zip(syms, poss, cells)]
        C = embed(frames, args.ckpt, args.batch_size)
        np.save(args.emb_out, C)
        print(f"wrote {args.emb_out} {C.shape}")

    # ---------------- frozen transform (same space as assign_clusters.py) --------
    m = np.load(args.model)
    keep = m["keep_mask"]
    centers = m["kmeans_centers"]
    radius = m["cluster_radius_p99"]
    K = int(m["k"])

    def transform(E_raw):
        E = E_raw[:, keep].astype(np.float64)
        Xs = (E - m["scaler_mean"]) / m["scaler_scale"]
        Z = (Xs - m["pca_mean"]) @ m["pca_components"].T
        if bool(m["pca_whiten"]):
            Z = Z / np.sqrt(m["pca_explained_variance"])
        return Z

    S = np.load(args.store).astype(np.float32)
    Smeta = np.load(args.store_meta).astype(np.float64)
    Zs = transform(S)
    Zc = transform(C)
    print(f"store {S.shape} -> {Zs.shape[1]}-d frozen PCA space; candidates {Zc.shape}")

    # ---------------- calibrate radii on the store itself ------------------------
    # CAREFUL: a raw nearest-neighbour distance is not a novelty measure on its own,
    # it is a *sampling-density* measure. The store is made of consecutive MD frames
    # (neighbours ~0.5 apart), while candidates are snapshots every N steps, so any
    # sparse batch looks "far" no matter how familiar its physics. So the radii are
    # calibrated on a store thinned until its own frame spacing matches the spacing
    # the candidates have among themselves -- an apples-to-apples reference.
    from sklearn.neighbors import NearestNeighbors
    nn_store = NearestNeighbors(n_neighbors=1).fit(Zs)
    sub = (np.arange(len(Zs)) if len(Zs) <= args.calib_sub
           else rng.choice(len(Zs), args.calib_sub, replace=False))
    d_self = NearestNeighbors(n_neighbors=2).fit(Zs).kneighbors(Zs[sub])[0][:, 1]
    dup_r = float(np.percentile(d_self, args.dup_pct))     # true duplicate: full density

    d_store, i_store = nn_store.kneighbors(Zc)
    d_store = d_store[:, 0]
    i_store = i_store[:, 0]
    d_batch = (NearestNeighbors(n_neighbors=2).fit(Zc).kneighbors(Zc)[0][:, 1]
               if len(Zc) > 1 else np.array([np.inf]))
    target = float(np.median(d_batch))

    thin_curve = {}
    for t in (1, 2, 5, 10, 20, 50, 100):
        idx = np.arange(0, len(Zs), t)
        if len(idx) < 50:
            continue
        ss = (idx if len(idx) <= args.calib_sub
              else rng.choice(idx, args.calib_sub, replace=False))
        dt = NearestNeighbors(n_neighbors=2).fit(Zs[idx]).kneighbors(Zs[ss])[0][:, 1]
        thin_curve[t] = dict(n=int(len(idx)), p50=float(np.median(dt)),
                             p99=float(np.percentile(dt, args.novel_pct)))
    t_match = min(thin_curve, key=lambda t: abs(thin_curve[t]["p50"] - target))
    novel_r = thin_curve[t_match]["p99"]
    print(f"store self-NN (full density): p5={dup_r:.3f}  p50={np.median(d_self):.3f}  "
          f"p99={np.percentile(d_self, args.novel_pct):.3f}")
    print(f"candidates' own spacing p50={target:.3f} -> density-matched reference = "
          f"store thinned 1/{t_match} (self-NN p50={thin_curve[t_match]['p50']:.3f}) "
          f"-> novel radius {novel_r:.3f}")

    # cluster assignment + per-cluster novelty (identical rule to assign_clusters.py)
    dc = np.linalg.norm(Zc[:, None, :] - centers[None, :, :], axis=2)
    assign = dc.argmin(axis=1)
    dcen = dc[np.arange(len(Zc)), assign]
    out_of_cluster = dcen > radius[assign]

    # ---------------- farthest-point sweep, seeded with the store ---------------
    # gain[i] = distance from candidate i to the nearest frame that is already
    # available (store + candidates picked earlier) == marginal information it adds.
    order, gain = [], []
    dmin_cur = d_store.copy()
    remaining = np.ones(len(Zc), dtype=bool)
    for _ in range(len(Zc)):
        idx = np.where(remaining)[0]
        j = idx[np.argmax(dmin_cur[idx])]
        order.append(int(j)); gain.append(float(dmin_cur[j]))
        remaining[j] = False
        dmin_cur = np.minimum(dmin_cur, np.linalg.norm(Zc - Zc[j], axis=1))
    order = np.asarray(order); gain = np.asarray(gain)
    rank = np.empty(len(Zc), dtype=int)
    rank[order] = np.arange(len(Zc))
    gain_at_pick = np.empty(len(Zc))
    gain_at_pick[order] = gain

    # ---------------- verdicts ---------------------------------------------------
    # "covered" = inside an existing regime: the store already has frames of this
    # kind, the candidate would mostly re-measure known physics.
    verdict = np.array(["covered"] * len(Zc), dtype=object)
    verdict[(d_store >= novel_r) | out_of_cluster] = "novel"
    verdict[d_store < dup_r] = "redundant"
    # batch-internal duplicates: the FPS sweep already covered them
    batch_dup = (gain_at_pick < dup_r) & (d_store >= dup_r)
    verdict[batch_dup] = "duplicate-of-batch"
    verdict[dmin_pair < args.min_dist] = "suspect"

    # Recommended list: FPS order, cut where the marginal gain drops below the
    # density-matched reference spacing -- i.e. the next candidate sits closer to
    # what we already have than two independent samples of this density typically do,
    # so its DFT mostly re-measures a point we can already interpolate. "novel"
    # candidates are always kept.
    gain_floor = (args.gain_floor if args.gain_floor > 0
                  else thin_curve[t_match]["p50"])
    keep_mask = ((verdict != "redundant") & (verdict != "duplicate-of-batch")
                 & (verdict != "suspect")
                 & ((gain_at_pick >= gain_floor) | (verdict == "novel")))
    sel = [int(i) for i in order if keep_mask[i]]
    if args.nsel > 0:
        sel = sel[:args.nsel]
    sel_set = set(sel)

    vpa_lo, vpa_hi = float(Smeta[:, 0].min()), float(Smeta[:, 0].max())
    counts = {k: int((verdict == k).sum()) for k in
              ("novel", "covered", "redundant", "duplicate-of-batch", "suspect")}
    # how crowded is the regime the candidates land in?
    Sassign = np.linalg.norm(Zs[:, None, :] - centers[None, :, :], axis=2).argmin(axis=1)
    store_occ = {int(c): int((Sassign == c).sum()) for c in range(K)}
    cand_occ = {int(c): int((assign == c).sum()) for c in range(K)}
    main_c = int(max(cand_occ, key=cand_occ.get))
    crowd = store_occ[main_c] / max(1, len(Zs))
    per_run = {}
    for r in sorted(set(runs.tolist())):
        mrun = runs == r
        per_run[r] = dict(n=int(mrun.sum()),
                          selected=int(sum(1 for i in np.where(mrun)[0] if i in sel_set)),
                          novel=int((verdict[mrun] == "novel").sum()),
                          nn_dist_median=float(np.median(d_store[mrun])))

    report = dict(
        n_candidates=len(names), store_frames=int(S.shape[0]),
        space=dict(dims=int(Zs.shape[1]), model=args.model),
        radii=dict(novel=novel_r, duplicate=dup_r,
                   store_self_nn_median=float(np.median(d_self)),
                   density_matched_thinning=int(t_match),
                   candidate_internal_nn_median=target,
                   store_self_nn_vs_thinning=thin_curve),
        nn_dist_to_store=dict(min=float(d_store.min()), p50=float(np.median(d_store)),
                              max=float(d_store.max())),
        verdict_counts=counts,
        n_recommended=len(sel), gain_floor=float(gain_floor),
        cluster_occupancy=dict(candidates=cand_occ, store=store_occ,
                               main_cluster=main_c,
                               main_cluster_store_share=float(crowd)),
        per_run=per_run,
        V_per_atom=dict(candidates=[float(vpa.min()), float(vpa.max())],
                        store=[vpa_lo, vpa_hi],
                        outside_store_range=int(((vpa < vpa_lo) | (vpa > vpa_hi)).sum())),
        min_pair_distance=dict(min=float(dmin_pair.min()), p50=float(np.median(dmin_pair))),
    )
    if counts["novel"] / len(names) >= 0.3:
        rec = (f"COMPUTE: {counts['novel']}/{len(names)} candidates sit outside every "
               f"existing regime (density-matched novel radius {novel_r:.2f}) -- the batch "
               f"buys genuinely new coverage.")
    elif counts["covered"] / len(names) >= 0.7:
        rec = (f"LOW MARGINAL VALUE: {counts['covered']}/{len(names)} candidates fall inside "
               f"regimes the store already samples; {cand_occ[main_c]} of them land in "
               f"cluster {main_c}, which already holds {store_occ[main_c]} stored frames "
               f"({crowd:.0%} of the store). The DFT would mostly re-measure known physics. "
               f"Compute a diverse subset (FPS order in the CSV) and spend the rest of the "
               f"budget on a regime the store lacks.")
    elif len(sel) < len(names):
        rec = (f"COMPUTE A SUBSET: {len(sel)}/{len(names)} carry independent information; "
               f"the rest repeat the store or each other. Running only the recommended "
               f"list saves {100 * (1 - len(sel) / len(names)):.0f}% of the DFT budget.")
    else:
        rec = "COMPUTE ALL: every candidate adds information beyond the store."
    report["recommendation"] = rec
    report["novel_frac"] = counts["novel"] / len(names)

    with open(f"{args.out_prefix}_report.json", "w") as f:
        json.dump(report, f, indent=2)

    with open(f"{args.out_prefix}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fps_rank", "file", "run", "step", "nat", "V_per_atom",
                    "min_pair_dist", "nn_dist_to_store", "nearest_store_idx",
                    "cluster", "dist_to_center", "cluster_radius",
                    "marginal_gain", "verdict", "recommended"])
        for i in np.argsort(rank):
            w.writerow([int(rank[i]), names[i], runs[i], int(steps[i]), int(nat[i]),
                        f"{vpa[i]:.4f}", f"{dmin_pair[i]:.3f}", f"{d_store[i]:.4f}",
                        int(Smeta[i_store[i], 3]), int(assign[i]), f"{dcen[i]:.4f}",
                        f"{radius[assign[i]]:.4f}", f"{gain_at_pick[i]:.4f}",
                        verdict[i], int(i in sel_set)])

    with open(f"{args.out_prefix}_keep.txt", "w") as f:
        for i in sel:
            f.write(names[i] + "\n")

    # ---------------- figure -----------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.8))
    ax[0].scatter(Zs[:, 0], Zs[:, 1], s=3, c="0.75", alpha=0.35,
                  label=f"store ({S.shape[0]})")
    cmap = {"covered": "tab:blue", "redundant": "tab:orange", "novel": "tab:red",
            "duplicate-of-batch": "tab:green", "suspect": "k"}
    for k, col in cmap.items():
        mk = verdict == k
        if mk.any():
            ax[0].scatter(Zc[mk, 0], Zc[mk, 1], s=22, c=col, alpha=0.85,
                          label=f"{k} ({int(mk.sum())})")
    ax[0].set_xlabel("PC1"); ax[0].set_ylabel("PC2")
    ax[0].set_title("candidates vs store (frozen PCA)"); ax[0].legend(fontsize=8)

    ax[1].hist(d_self, bins=60, density=True, alpha=0.4,
               label="store self-NN (full density)")
    idx_t = np.arange(0, len(Zs), t_match)
    ss_t = (idx_t if len(idx_t) <= args.calib_sub
            else rng.choice(idx_t, args.calib_sub, replace=False))
    d_thin = NearestNeighbors(n_neighbors=2).fit(Zs[idx_t]).kneighbors(Zs[ss_t])[0][:, 1]
    ax[1].hist(d_thin, bins=40, density=True, alpha=0.4,
               label=f"store self-NN (thinned 1/{t_match})")
    ax[1].hist(d_store, bins=30, density=True, alpha=0.6, label="candidate -> store")
    ax[1].axvline(dup_r, color="tab:orange", ls="--", label=f"dup r={dup_r:.2f}")
    ax[1].axvline(novel_r, color="tab:red", ls="--", label=f"novel r={novel_r:.2f}")
    ax[1].set_xlabel("NN distance"); ax[1].set_title("how far from what we already have")
    ax[1].legend(fontsize=8)

    ax[2].plot(np.arange(1, len(gain) + 1), gain, lw=1.4)
    ax[2].axhline(dup_r, color="tab:orange", ls="--", label=f"dup r={dup_r:.2f}")
    ax[2].axhline(gain_floor, color="tab:purple", ls="-.",
                  label=f"gain floor={gain_floor:.2f}")
    if sel:
        ax[2].axvline(len(sel), color="tab:green", ls=":", label=f"cut at {len(sel)}")
    ax[2].set_xlabel("candidates computed (FPS order)")
    ax[2].set_ylabel("marginal gain (min dist to available)")
    ax[2].set_title("diminishing return of the batch"); ax[2].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{args.out_prefix}.png", dpi=130)

    print("\n" + json.dumps({k: report[k] for k in
                             ("verdict_counts", "n_recommended", "nn_dist_to_store",
                              "recommendation")}, indent=2))
    print(f"\nwrote {args.out_prefix}.csv, {args.out_prefix}_keep.txt, "
          f"{args.out_prefix}_report.json, {args.out_prefix}.png")


if __name__ == "__main__":
    main()

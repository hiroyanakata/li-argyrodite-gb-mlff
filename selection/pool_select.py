#!/usr/bin/env python
# =============================================================================
# pool_select.py -- validate a candidate pool and pick the frames that are
#                   genuinely NEW relative to the existing training store.
#
# Novelty is measured in the same embedding space the store already uses:
# `emb_store.npy` (49172 x 256) holds the mean|std-pooled M3GNet node features
# (`m3gnet.final_input.meanstd256`) of every existing training frame, so a new
# frame's distance to its nearest existing neighbour says exactly how much it
# adds. Selection is farthest-point sampling seeded by that novelty, which
# spends the DFT budget on coverage instead of on 800 near-duplicates.
#
# Frames are validated first -- a candidate that DFT cannot converge (atoms on
# top of each other) or that no force field should ever see is dropped here,
# not after the SCF bill.
#
#   python pool_select.py pool/*.extxyz --emb_store ../emb_store.npy \
#          --ckpt ../matersim_model.pth --budget_min 4800 --out sel
# =============================================================================
import argparse
import glob
import os

import numpy as np
import torch

try:
    import ase.constraints as _asec
    if not hasattr(_asec, "full_3x3_to_voigt_6_stress"):
        from ase.stress import (full_3x3_to_voigt_6_stress as _f3,
                                voigt_6_to_full_3x3_stress as _v6)
        _asec.full_3x3_to_voigt_6_stress = _f3
        _asec.voigt_6_to_full_3x3_stress = _v6
except Exception:
    pass

from ase.io import read, write


def mic(d, cell):
    inv = np.linalg.inv(cell)
    f = d @ inv
    f -= np.round(f)
    return f @ cell


def min_pair_dist(at):
    pos = at.get_positions()
    cell = np.array(at.get_cell())
    d = mic((pos[:, None, :] - pos[None, :, :]).reshape(-1, 3), cell)
    dd = np.linalg.norm(d.reshape(len(pos), len(pos), 3), axis=2)
    np.fill_diagonal(dd, 9e9)
    return float(dd.min())


def embed(frames, ckpt, device, batch_size):
    """mean|std pooled node features -- identical recipe to extract_embeddings.py."""
    from mattersim.forcefield.potential import Potential, batch_to_dict
    from mattersim.datasets.utils.build import build_dataloader

    pot = Potential.from_checkpoint(load_path=ckpt, load_training_state=False,
                                    device=device)
    model = pot.model
    model.eval()
    buf = {}
    model.final.register_forward_pre_hook(
        lambda m, inp: buf.__setitem__("x", inp[0].detach()))
    dl = build_dataloader(frames, only_inference=True, shuffle=False,
                          batch_size=batch_size, cutoff=5.0, threebody_cutoff=4.0)
    out = []
    with torch.no_grad():
        for gb in dl:
            gb.to(device)
            inp = batch_to_dict(gb)
            model(inp)
            x = buf["x"]
            b = inp["batch"].long()
            ng = int(inp["num_graphs"])
            df = x.shape[1]
            ones = torch.ones(b.shape[0], device=x.device, dtype=x.dtype)
            cnt = torch.zeros(ng, device=x.device, dtype=x.dtype).index_add_(0, b, ones).clamp_min(1.0)
            s = torch.zeros(ng, df, device=x.device, dtype=x.dtype).index_add_(0, b, x)
            ssq = torch.zeros(ng, df, device=x.device, dtype=x.dtype).index_add_(0, b, x * x)
            mean = s / cnt.unsqueeze(1)
            std = (ssq / cnt.unsqueeze(1) - mean * mean).clamp_min(0).sqrt()
            out.append(torch.cat([mean, std], dim=1).cpu().numpy().astype(np.float32))
    return np.concatenate(out, axis=0)


def nearest_dist(A, B, block=2048):
    """min_j ||A_i - B_j|| without materialising the full matrix."""
    out = np.empty(len(A), dtype=np.float32)
    Bn = (B ** 2).sum(1)
    for i in range(0, len(A), block):
        a = A[i:i + block]
        d2 = (a ** 2).sum(1)[:, None] + Bn[None, :] - 2.0 * a @ B.T
        out[i:i + block] = np.sqrt(np.maximum(d2.min(1), 0))
    return out


def farthest_point(X, n, start_scores):
    """FPS in X, first point = the most novel one."""
    n = min(n, len(X))
    sel = [int(np.argmax(start_scores))]
    d = np.linalg.norm(X - X[sel[0]], axis=1)
    while len(sel) < n:
        k = int(np.argmax(d))
        sel.append(k)
        d = np.minimum(d, np.linalg.norm(X - X[k], axis=1))
    return sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pool", nargs="+", help="pool extxyz files (globs ok)")
    ap.add_argument("--emb_store", required=True, help="existing-data embeddings .npy")
    ap.add_argument("--ckpt", required=True, help="SAME checkpoint the store used")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--min_dist", type=float, default=1.30,
                    help="reject frames with any pair closer than this [A]")
    ap.add_argument("--fmax_max", type=float, default=60.0,
                    help="reject frames whose MLIP fmax exceeds this [eV/A]")
    ap.add_argument("--eval_model", default="",
                    help="model used for the fmax sanity filter (default: --ckpt)")
    ap.add_argument("--min_per_comp", type=int, default=8)
    ap.add_argument("--budget_min", type=float, default=4800.0,
                    help="DFT budget in node-minutes")
    ap.add_argument("--cost_per_atom_min", type=float, default=0.055,
                    help="SCF node-minutes per atom (206 at ~ 13.7 min -> 0.066)")
    ap.add_argument("--out", default="sel")
    args = ap.parse_args()

    files = []
    for p in args.pool:
        files.extend(sorted(glob.glob(p)))
    files = [f for f in files if not f.endswith("ALL.extxyz")]
    frames, src = [], []
    for f in files:
        fr = read(f, index=":")
        frames.extend(fr)
        src.extend([os.path.basename(f)] * len(fr))
    print("pool: %d frames from %d files" % (len(frames), len(files)))

    # ---- validation -------------------------------------------------------
    from mattersim.forcefield import MatterSimCalculator
    ev = args.eval_model or args.ckpt
    calc = (MatterSimCalculator(device=args.device) if ev == "pretrained"
            else MatterSimCalculator(load_path=ev, device=args.device))
    keep, drop = [], {}
    for i, at in enumerate(frames):
        md = min_pair_dist(at)
        if md < args.min_dist:
            drop["min_dist"] = drop.get("min_dist", 0) + 1
            continue
        a2 = at.copy(); a2.calc = calc
        fm = float(np.linalg.norm(a2.get_forces(), axis=1).max())
        if not np.isfinite(fm) or fm > args.fmax_max:
            drop["fmax"] = drop.get("fmax", 0) + 1
            continue
        at.info["min_dist"] = md
        at.info["mlip_fmax"] = fm
        at.info["src"] = src[i]
        keep.append(at)
    print("validated: %d kept, dropped %s" % (len(keep), drop or "none"))
    if not keep:
        return

    # ---- novelty vs the existing store ------------------------------------
    print("embedding %d frames with %s ..." % (len(keep), os.path.basename(args.ckpt)))
    X = embed(keep, args.ckpt, args.device, args.batch_size)
    store = np.load(args.emb_store)
    print("store: %s" % (store.shape,))
    nov = nearest_dist(X, store)
    for at, v in zip(keep, nov):
        at.info["novelty"] = float(v)
    q = np.percentile(nov, [0, 25, 50, 75, 100])
    print("novelty (distance to nearest existing frame): "
          + "  ".join("p%d=%.2f" % (p, v) for p, v in zip([0, 25, 50, 75, 100], q)))
    # a novelty number means nothing without the store's own spacing: how far
    # apart are EXISTING frames from each other in the same metric?
    rs = np.random.default_rng(0).choice(len(store), min(3000, len(store)),
                                         replace=False)
    sub = store[rs]
    d2 = ((sub ** 2).sum(1)[:, None] + (sub ** 2).sum(1)[None, :]
          - 2.0 * sub @ sub.T)
    np.fill_diagonal(d2, 9e18)
    self_nn = np.sqrt(np.maximum(d2.min(1), 0))
    sq = np.percentile(self_nn, [25, 50, 75, 95])
    print("  reference: existing store's own nearest-neighbour spacing "
          + "  ".join("p%d=%.2f" % (p, v) for p, v in zip([25, 50, 75, 95], sq)))
    print("  -> %.0f%% of the pool is further from the store than the store's "
          "own p95 spacing (%.2f) = genuinely new territory"
          % (100.0 * (nov > sq[-1]).mean(), sq[-1]))

    # ---- budgeted selection, farthest-point per composition ---------------
    comps = {}
    for i, at in enumerate(keep):
        comps.setdefault(at.get_chemical_formula(), []).append(i)
    # share the budget in proportion to each composition's novelty mass
    mass = {c: float(np.mean(nov[idx])) * len(idx) for c, idx in comps.items()}
    tot = sum(mass.values())
    print("\n%-24s %7s %7s %9s %8s" % ("composition", "natoms", "pool", "novelty", "picked"))
    selected = []
    for c, idx in sorted(comps.items()):
        nat = len(keep[idx[0]])
        cost = nat * args.cost_per_atom_min
        share = args.budget_min * mass[c] / tot
        n = max(args.min_per_comp, int(share / cost))
        n = min(n, len(idx))
        sub = X[idx]
        pick = farthest_point(sub, n, nov[idx])
        chosen = [idx[k] for k in pick]
        selected.extend(chosen)
        print("%-24s %7d %7d %9.2f %8d" % (c, nat, len(idx), np.mean(nov[idx]), len(chosen)))

    os.makedirs(args.out, exist_ok=True)
    bycomp = {}
    for i in selected:
        bycomp.setdefault(keep[i].get_chemical_formula(), []).append(keep[i])
    total_cost = 0.0
    for c, fr in sorted(bycomp.items()):
        write(os.path.join(args.out, c + ".extxyz"), fr)
        total_cost += len(fr) * len(fr[0]) * args.cost_per_atom_min
    write(os.path.join(args.out, "SELECTED.extxyz"), [keep[i] for i in selected])
    print("\nselected %d frames, %d compositions -> %s/" %
          (len(selected), len(bycomp), args.out))
    print("estimated DFT cost: %.0f node-minutes (%.1f node-hours)"
          % (total_cost, total_cost / 60.0))
    kinds = {}
    for i in selected:
        k = keep[i].info.get("kind", "?")
        kinds[k] = kinds.get(k, 0) + 1
    print("by sampling kind: " + ", ".join("%s:%d" % kv for kv in sorted(kinds.items())))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Append a new-only DeepMD batch to the cumulative embedding cache.

One command per active-learning cycle. Each cycle ships ONLY the frames added
this round (the past data is not re-shipped); this encodes that whole batch and
appends it to the growing store -- no ``--n_old``, no ``_all`` renames,
guarded against appending the same batch twice.

  * reads the manifest for the cumulative count so far (migrating an existing
    embeddings_all.npy / embeddings.npy the first time -- no re-encoding),
  * refuses to re-append a batch whose content signature matches the last one
    (unless --force),
  * encodes every frame of the given (new-only) listdata on the GPU, appends the
    rows to emb_store.npy, offsets their global_idx to continue after the store,
    and records the batch (range / date / source / signature) in the manifest,
  * prints the validate / cluster commands to run next.

Descriptor recipe is identical to extract_embeddings.py (per-atom node feature
fed into model.final, mean|std pooled per structure).

  python update_embeddings.py newdata.txt --ckpt matersim_model.pth
"""
import argparse

import numpy as np
import torch

import emb_cache as ec

# ASE >=3.24 moved full_3x3_to_voigt_6_stress; mattersim 1.2.0 wants the old path.
import ase.constraints as _ase_constraints
if not hasattr(_ase_constraints, "full_3x3_to_voigt_6_stress"):
    from ase.stress import full_3x3_to_voigt_6_stress as _f3, voigt_6_to_full_3x3_stress as _v6
    _ase_constraints.full_3x3_to_voigt_6_stress = _f3
    _ase_constraints.voigt_6_to_full_3x3_stress = _v6

p = argparse.ArgumentParser()
p.add_argument("listdata", help="listdata for the NEW-ONLY batch (this cycle's added frames)")
p.add_argument("--ckpt", default="matersim_model.pth")
p.add_argument("--batch_size", type=int, default=32)
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
p.add_argument("--force", action="store_true",
               help="append even if this batch looks identical to the last appended one")
args = p.parse_args()

# ---------- manifest (bootstrap from any legacy *_all.npy the first time) -----
man = ec.load_manifest()
if man is None:
    man, migrated = ec.bootstrap_manifest(args.ckpt)
    if migrated:
        print(f"bootstrapped {ec.STORE} from legacy embeddings "
              f"({man['n_total']} frames) -> manifest {ec.MANIFEST}")
    else:
        print("no cache found; starting an empty store")
n_cached = man["n_total"]

# ---------- new batch meta (cheap; global_idx continues after the store) ------
meta_new = ec.load_meta(args.listdata)          # gidx 0-based within the batch
n_new = meta_new.shape[0]
if n_new == 0:
    raise SystemExit(f"{args.listdata} has no frames")
meta_new[:, 3] = np.arange(n_cached, n_cached + n_new)   # global idx in the store

# ---------- guard against re-appending the same batch -------------------------
sig = ec.batch_signature(meta_new)
last = man["iterations"][-1] if man["iterations"] else None
if last and last.get("sig") == sig and not args.force:
    raise SystemExit(
        f"this batch matches the last appended one (iter {last['iter']}, "
        f"n={sig[0]}, source={last.get('source')}). Already added? Pass --force to "
        f"append anyway.")

print(f"cached: {n_cached}   new batch: {n_new}   -> total {n_cached + n_new}", flush=True)

# ---------- encode the whole new batch ---------------------------------------
from mattersim.forcefield.potential import Potential, batch_to_dict
from mattersim.datasets.utils.build import build_dataloader

frames = ec.build_atoms(args.listdata)
assert len(frames) == n_new, (len(frames), n_new)

pot = Potential.from_checkpoint(load_path=args.ckpt, load_training_state=False,
                                device=args.device)
model = pot.model
model.eval()
buf = {}
model.final.register_forward_pre_hook(lambda m, inp: buf.__setitem__("x", inp[0].detach()))

dl = build_dataloader(frames, only_inference=True, shuffle=False,
                      batch_size=args.batch_size, cutoff=5.0, threebody_cutoff=4.0)
embs = []
with torch.no_grad():
    for bi, gb in enumerate(dl):
        gb.to(args.device)
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
        embs.append(torch.cat([mean, std], dim=1).cpu().numpy().astype(np.float32))
        if bi % 20 == 0:
            print(f"  batch {bi}: {sum(e.shape[0] for e in embs)}/{n_new} done", flush=True)

tail = np.concatenate(embs, axis=0)
assert tail.shape[0] == n_new, (tail.shape, n_new)

# ---------- append in place + record the batch -------------------------------
if n_cached:
    old_E = np.load(ec.STORE).astype(np.float32)
    old_M = np.load(ec.META).astype(np.float32)
    assert old_E.shape[0] == n_cached == old_M.shape[0], (old_E.shape, n_cached, old_M.shape)
    E = np.concatenate([old_E, tail], axis=0)
    M = np.concatenate([old_M, meta_new.astype(np.float32)], axis=0)
else:
    E, M = tail, meta_new.astype(np.float32)
N = n_cached + n_new
assert E.shape[0] == N == M.shape[0], (E.shape, N, M.shape)
np.save(ec.STORE, E)
np.save(ec.META, M)
ec.add_iteration(man, n_new, source=args.listdata, sig=sig)
man["ckpt"] = args.ckpt
ec.save_manifest(man)

print(f"\nappended {n_new} frames -> {ec.STORE} {E.shape} (total {N})")
print(f"manifest {ec.MANIFEST}: {len(man['iterations'])} iterations, "
      f"newest batch = global idx {ec.new_start(man)}..{N}")
print("\nnext:")
print(f"  python validate_new_data.py {args.listdata} --from-store")
print("  python cluster_embeddings.py --emb emb_store.npy --meta emb_meta_store.npy   # if you re-cluster")

#!/usr/bin/env python
"""Extract MatterSim (M3GNet) structure embeddings for every frame.

The descriptor is the per-atom node feature `atom_attr` produced right before
`model.final` (the energy head), i.e. the output of the last graph-convolution
block. It is a learned, physics-aware fingerprint (128-dim, three-body aware).
Each structure is summarised by mean- and std-pooling its atoms, giving a
256-dim per-structure vector suitable for clustering + FPS selection.

Outputs:
  embeddings.npy   float32 [nframe, 256]  (mean128 | std128) pooled node features
  emb_meta.npy     float32 [nframe, 4]    columns: V_per_atom, E_per_atom, fmax, global_idx

Usage:
  python extract_embeddings.py <listdata> \
      --ckpt ../lmp.example/matersim_model.pth --batch_size 32
"""
import argparse
import os

import numpy as np
import torch
from ase import Atoms

# Compat shim: ASE >=3.24 moved full_3x3_to_voigt_6_stress from ase.constraints
# to ase.stress, but mattersim 1.2.0 imports it from the old location.
import ase.constraints as _ase_constraints
if not hasattr(_ase_constraints, "full_3x3_to_voigt_6_stress"):
    from ase.stress import full_3x3_to_voigt_6_stress as _f3, voigt_6_to_full_3x3_stress as _v6
    _ase_constraints.full_3x3_to_voigt_6_stress = _f3
    _ase_constraints.voigt_6_to_full_3x3_stress = _v6

from mattersim.forcefield.potential import Potential, batch_to_dict
from mattersim.datasets.utils.build import build_dataloader

ATMTYP = np.asarray(["Li", "P", "S", "Br", "I", "Cl"])

p = argparse.ArgumentParser()
p.add_argument("listdata")
p.add_argument("--ckpt", default="../lmp.example/matersim_model.pth",
               help="MatterSim checkpoint used as the encoder (default: pretrained base)")
p.add_argument("--batch_size", type=int, default=32)
p.add_argument("--out", default="embeddings.npy")
p.add_argument("--meta_out", default="emb_meta.npy")
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
p.add_argument("--limit", type=int, default=0, help="if >0, only process first N frames (smoke test)")
args = p.parse_args()

# ---------- load every frame from DeepMD raw ----------
frames, meta = [], []
with open(args.listdata) as f:
    lines = [l for l in f if l.strip()]
gidx = 0
for line in lines:
    aa = line.split()
    name = aa[0]
    types = np.loadtxt(name + "type.raw", dtype=int).astype(int)
    nat = len(types)
    symbols = ATMTYP[types].tolist()
    for iset in aa[1:]:
        d = name + iset
        xyz = np.load(os.path.join(d, "coord.npy")).reshape(-1, nat, 3)
        box = np.load(os.path.join(d, "box.npy")).reshape(-1, 3, 3)
        ene = np.load(os.path.join(d, "energy.npy"))
        frc = np.load(os.path.join(d, "force.npy")).reshape(-1, nat, 3)
        ng = ene.shape[0]
        for i in range(ng):
            a = Atoms(symbols=symbols, positions=xyz[i], cell=box[i], pbc=True)
            frames.append(a)
            V = box[i, 0, 0] * box[i, 1, 1] * box[i, 2, 2]
            fmax = np.linalg.norm(frc[i], axis=1).max()
            meta.append((V / nat, ene[i] / nat, fmax, gidx))
            gidx += 1
            if args.limit and len(frames) >= args.limit:
                break
        print(f"loaded {name}{iset}: {ng} frames, {nat} atoms", flush=True)
        if args.limit and len(frames) >= args.limit:
            break
    if args.limit and len(frames) >= args.limit:
        break
meta = np.asarray(meta, dtype=np.float32)
print(f"total frames: {len(frames)}", flush=True)

# ---------- load encoder ----------
pot = Potential.from_checkpoint(load_path=args.ckpt, load_training_state=False,
                                device=args.device)
model = pot.model
model.eval()

# capture the input to model.final == last graph-conv node features [n_atoms, 128]
buf = {}
model.final.register_forward_pre_hook(lambda m, inp: buf.__setitem__("x", inp[0].detach()))

# ---------- inference, no shuffle to keep frame order ----------
dl = build_dataloader(frames, only_inference=True, shuffle=False,
                      batch_size=args.batch_size, cutoff=5.0, threebody_cutoff=4.0)

embs = []
with torch.no_grad():
    for bi, graph_batch in enumerate(dl):
        graph_batch.to(args.device)
        inp = batch_to_dict(graph_batch)
        model(inp)                        # forward; hook fills buf["x"]
        x = buf["x"]                      # [n_atoms_total, 128]
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
        if bi % 50 == 0:
            print(f"  batch {bi}: {sum(e.shape[0] for e in embs)} frames done", flush=True)

E = np.concatenate(embs, axis=0)
assert E.shape[0] == meta.shape[0], (E.shape, meta.shape)
np.save(args.out, E)
np.save(args.meta_out, meta)
print(f"saved {args.out} {E.shape} and {args.meta_out} {meta.shape}", flush=True)

#!/usr/bin/env python
"""Shared helpers for the *growing* embedding cache used by the recursive
active-learning loop.

Each active-learning cycle hands us a *new-only* DeepMD dataset (just the frames
added this cycle -- the past data is not re-shipped). Instead of juggling
``embeddings.npy`` vs ``embeddings_all.npy`` and a hand-typed ``--n_old`` every
cycle, we keep ONE cumulative store that grows by appending each new batch, plus
a JSON manifest recording how many frames have been encoded and the provenance
(and a content signature) of each batch:

  emb_store.npy       [Ntot,256] float32   one row per frame, appended in place
  emb_meta_store.npy  [Ntot,4]   float32   V/atom, E/atom, fmax, global_idx
  emb_store.json      manifest, e.g.
      {
        "ckpt": "matersim_model.pth",
        "descriptor": "m3gnet.final_input.meanstd256",
        "n_total": 34950,
        "iterations": [
          {"iter": 0, "range": [0, 34950], "n": 34950,
           "date": "...", "source": "bootstrap:embeddings_all.npy"},
          {"iter": 1, "range": [34950, 37310], "n": 2360,
           "date": "...", "source": "newdata.txt", "sig": [...]}
        ]
      }

``iterations[-1]["range"][0]`` is exactly the ``n_old`` that validation and
novelty scoring need -- derived, never typed.
"""
import datetime
import json
import os

import numpy as np

ATMTYP = np.asarray(["Li", "P", "S", "Br", "I", "Cl"])

STORE = "emb_store.npy"
META = "emb_meta_store.npy"
MANIFEST = "emb_store.json"
DESCRIPTOR = "m3gnet.final_input.meanstd256"


# ---------------------------------------------------------------- listdata IO
def _iter_sets(listdata):
    """Yield (dir, type.raw path, symbols, nat) for every set in a listdata."""
    with open(listdata) as f:
        lines = [l for l in f if l.strip()]
    for line in lines:
        aa = line.split()
        name = aa[0]
        types = np.loadtxt(name + "type.raw", dtype=int).astype(int)
        nat = len(types)
        symbols = ATMTYP[types].tolist()
        for iset in aa[1:]:
            yield name + iset, name + "type.raw", symbols, nat


def count_frames(listdata):
    """Total frame count, cheaply (reads only energy.npy)."""
    n = 0
    for d, _, _, _ in _iter_sets(listdata):
        n += int(np.load(os.path.join(d, "energy.npy")).shape[0])
    return n


def load_meta(listdata):
    """[N,4] float64: V/atom, E/atom, fmax, global_idx -- for every frame."""
    meta = []
    gidx = 0
    for d, _, _, nat in _iter_sets(listdata):
        box = np.load(os.path.join(d, "box.npy")).reshape(-1, 3, 3)
        ene = np.load(os.path.join(d, "energy.npy"))
        frc = np.load(os.path.join(d, "force.npy")).reshape(-1, nat, 3)
        for i in range(ene.shape[0]):
            V = box[i, 0, 0] * box[i, 1, 1] * box[i, 2, 2]
            fmax = np.linalg.norm(frc[i], axis=1).max()
            meta.append((V / nat, ene[i] / nat, fmax, gidx))
            gidx += 1
    return np.asarray(meta, dtype=np.float64)


def build_atoms(listdata):
    """ase.Atoms for every frame in listdata (the whole new-only batch)."""
    from ase import Atoms
    frames = []
    for d, _, symbols, nat in _iter_sets(listdata):
        xyz = np.load(os.path.join(d, "coord.npy")).reshape(-1, nat, 3)
        box = np.load(os.path.join(d, "box.npy")).reshape(-1, 3, 3)
        ng = box.shape[0]
        for i in range(ng):
            frames.append(Atoms(symbols=symbols, positions=xyz[i],
                                cell=box[i], pbc=True))
    return frames


def batch_signature(meta):
    """Cheap content fingerprint of a batch (n + rounded E/V sums) so a batch that
    is accidentally re-submitted can be detected and not appended twice."""
    return [int(meta.shape[0]),
            round(float(meta[:, 1].sum()), 4),   # sum E/atom
            round(float(meta[:, 0].sum()), 4)]   # sum V/atom


# ---------------------------------------------------------------- manifest IO
def load_manifest(path=MANIFEST):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def save_manifest(man, path=MANIFEST):
    with open(path, "w") as f:
        json.dump(man, f, indent=2)


def new_start(man):
    """Global index where the newest (last) batch begins == n_old for validation.

    A store with a single bootstrap iteration has nothing "new" yet -> 0.
    """
    if not man or len(man.get("iterations", [])) < 2:
        return 0
    return int(man["iterations"][-1]["range"][0])


def add_iteration(man, n_added, source, sig=None):
    prev = man["n_total"]
    new_total = prev + n_added
    man["iterations"].append(dict(
        iter=len(man["iterations"]), range=[prev, new_total], n=int(n_added),
        date=datetime.datetime.now().isoformat(timespec="seconds"),
        source=source, sig=sig))
    man["n_total"] = new_total
    return man


def bootstrap_manifest(ckpt):
    """Adopt any pre-existing whole-set embeddings/meta as iteration 0 so the
    recursive loop can start without re-encoding. Returns (manifest, migrated?)."""
    for emb_src, meta_src in (("embeddings_all.npy", "emb_meta_all.npy"),
                              ("embeddings.npy", "emb_meta.npy")):
        if os.path.exists(emb_src) and os.path.exists(meta_src):
            E = np.load(emb_src).astype(np.float32)
            M = np.load(meta_src).astype(np.float32)
            assert E.shape[0] == M.shape[0], (E.shape, M.shape)
            np.save(STORE, E)
            np.save(META, M)
            n = int(E.shape[0])
            man = dict(ckpt=ckpt, descriptor=DESCRIPTOR, n_total=n,
                       iterations=[dict(iter=0, range=[0, n], n=n,
                                        date=datetime.datetime.now().isoformat(timespec="seconds"),
                                        source=f"bootstrap:{emb_src}")])
            save_manifest(man)
            return man, True
    # nothing to adopt -> empty store
    man = dict(ckpt=ckpt, descriptor=DESCRIPTOR, n_total=0, iterations=[])
    return man, False

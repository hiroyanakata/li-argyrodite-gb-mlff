#!/usr/bin/env python
# audit_ps4.py -- per-frame PS4-framework audit of extxyz training sets.
# For every frame: P motifs within 2.4 A (S/Cl/Br), PS4-vertex halides, free S
# (no P within 2.4 A) vs the nominal free S = nS - 4 nP. A frame is
#   built_broken : free S > nominal  (S->X was applied to PS4 vertices when the
#                  structure was built; the deficit shows up as missing vertices)
#   ps4_intact   : every P is PS4
#   python audit_ps4.py v3_train_reduce_emb.extxyz ... --out audit_ps4/frames.csv
import argparse, csv, re, collections
import numpy as np

def frames(path):
    f = open(path)
    while True:
        l = f.readline()
        if not l:
            return
        n = int(l); h = f.readline()
        lat = np.array(list(map(float, re.search(r'Lattice="([^"]+)"', h).group(1).split()))).reshape(3, 3)
        props = re.search(r'Properties=(\S+)', h).group(1).split(':')
        col = 0; sc = pc = None
        for i in range(0, len(props), 3):
            if props[i] == "species": sc = col
            if props[i] == "pos": pc = col
            col += int(props[i + 2])
        S, X = [], []
        for _ in range(n):
            p = f.readline().split(); S.append(p[sc]); X.append([float(v) for v in p[pc:pc + 3]])
        yield np.array(S), np.array(X), lat

ap = argparse.ArgumentParser(); ap.add_argument("files", nargs="+"); ap.add_argument("--out", required=True)
a = ap.parse_args()
rows = []
for path in a.files:
    for k, (S, X, lat) in enumerate(frames(path)):
        inv = np.linalg.inv(lat); P = np.where(S == "P")[0]; A = np.where(np.isin(S, ["S", "Cl", "Br"]))[0]
        d = X[A][:, None, :] - X[P][None, :, :]; f = d @ inv; f -= np.round(f)
        r = np.linalg.norm(f @ lat, axis=2)                     # (nA, nP)
        bond = r < 2.4
        isS = S[A] == "S"
        nS_on = (bond & isS[:, None]).sum(0); nX_on = (bond & ~isS[:, None]).sum(0)
        nonps4 = int(((nS_on != 4) | (nX_on != 0)).sum())
        free_s = int((isS & ~bond.any(1)).sum()); nominal = int((S == "S").sum() - 4 * len(P))
        c = collections.Counter(S)
        rows.append(dict(file=path, frame=k, natoms=len(S),
                         formula="".join(f"{e}{c[e]}" for e in sorted(c)),
                         n_P=len(P), n_nonPS4=nonps4, n_vertexX=int(nX_on.sum()),
                         free_S=free_s, nominal_free_S=nominal,
                         built_broken=int(free_s > nominal), ps4_intact=int(nonps4 == 0)))
with open(a.out, "w", newline="") as fo:
    w = csv.DictWriter(fo, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("wrote", a.out, len(rows), "frames")

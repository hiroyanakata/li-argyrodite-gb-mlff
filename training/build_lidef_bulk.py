#!/usr/bin/env python
# build_lidef_bulk.py -- correctly built Li-deficient argyrodite bulk cells
# Li(6-x) P S(5-x) X(1+x): starting from an intact stoichiometric Li6PS5X frame
# (template from the v3 training set), turn n FREE S2- into halide and delete n Li.
# PS4 is never touched (the original v3 cells of these compositions converted
# PS4-vertex S instead -- see audit_ps4/frames.csv, built_broken=1).
# Writes one extxyz per target composition (same formulas as the v3 set).
import re, sys, numpy as np
sys.path.insert(0, "audit_ps4")
rng = np.random.default_rng(0)

def frames(path):
    f = open(path)
    while True:
        l = f.readline()
        if not l: return
        n = int(l); h = f.readline()
        lat = np.array(list(map(float, re.search(r'Lattice="([^"]+)"', h).group(1).split()))).reshape(3, 3)
        props = re.search(r'Properties=(\S+)', h).group(1).split(':'); col = 0
        for i in range(0, len(props), 3):
            if props[i] == "species": sc = col
            if props[i] == "pos": pc = col
            col += int(props[i + 2])
        S, X = [], []
        for _ in range(n):
            p = f.readline().split(); S.append(p[sc]); X.append([float(v) for v in p[pc:pc + 3]])
        yield np.array(S), np.array(X), lat

def bonded(S, X, lat):
    inv = np.linalg.inv(lat); P = np.where(S == "P")[0]
    d = X[:, None, :] - X[P][None, :, :]; f = d @ inv; f -= np.round(f)
    return (np.linalg.norm(f @ lat, axis=2) < 2.4).any(1)

def intact(S, X, lat):
    b = bonded(S, X, lat); P = np.where(S == "P")[0]
    return (S[b] == "S").sum() == 4 * len(P) and not np.isin(S[b], ["Cl", "Br"]).any() and ((S == "S") & ~b).sum() == (S == "S").sum() - 4 * len(P)

def write(path, S, X, lat, info):
    with open(path, "w") as f:
        f.write("%d\n" % len(S))
        f.write('Lattice="%s" Properties=species:S:1:pos:R:3 %s pbc="T T T"\n' % (" ".join("%.8f" % v for v in lat.ravel()), info))
        for s, x in zip(S, X): f.write("%-2s %14.8f %14.8f %14.8f\n" % (s, *x))

# templates: first intact frame of each stoichiometric composition
want = {"Cl4Li24P4S20", "Br4Li24P4S20", "Cl8Li48P8S40", "Br8Li48P8S40"}
tpl = {}
for S, X, lat in frames("v3_train_reduce_emb.extxyz"):
    import collections
    c = collections.Counter(S); key = "".join(f"{e}{c[e]}" for e in sorted(c))
    if key in want and key not in tpl and intact(S, X, lat):
        tpl[key] = (S, X, lat)
    if len(tpl) == len(want): break
print("templates:", list(tpl))
for k, (S, X, lat) in tpl.items():
    write(f"audit_ps4/bulkdef/stoich_{k}.extxyz", S, X, lat, f"src=v3_template_{k}")

# targets = the v3 Li-deficient formulas: (nP, nBr, nCl)
targets = [(4, b, c) for (b, c) in [(0, 8), (1, 7), (2, 6), (7, 1), (8, 0)]] + \
          [(4, b, c) for (b, c) in [(0, 6), (1, 5), (2, 4), (5, 1), (6, 0)]] + \
          [(4, b, c) for (b, c) in [(0, 5), (1, 4), (2, 3), (4, 1), (5, 0)]] + \
          [(8, b, c) for (b, c) in [(2, 10), (6, 6), (10, 2)]] + \
          [(8, b, c) for (b, c) in [(1, 9), (5, 5), (9, 1)]] + \
          [(8, b, c) for (b, c) in [(4, 5), (8, 1), (0, 9)]]
for nP, nBr, nCl in targets:
    nX = nBr + nCl; n = nX - nP                      # extra halides = removed Li = S->X
    base = ("Br" if nBr > nCl else "Cl") + ("4Li24P4S20" if nP == 4 else "8Li48P8S40")
    S, X, lat = (a.copy() for a in tpl[base])
    free = np.where((S == "S") & ~bonded(S, X, lat))[0]
    conv = rng.choice(free, size=n, replace=False)
    S[conv] = "X"
    hal = np.where(np.isin(S, ["Cl", "Br", "X"]))[0]
    lab = np.array(["Br"] * nBr + ["Cl"] * nCl); rng.shuffle(lab); S[hal] = lab
    li = np.where(S == "Li")[0]; drop = set(rng.choice(li, size=n, replace=False).tolist())
    keep = [i for i in range(len(S)) if i not in drop]
    S, X = S[keep], X[keep]
    c = collections.Counter(S); name = "".join(f"{e}{c[e]}" for e in sorted(c))
    q = c["Li"] + 5 * c["P"] - 2 * c["S"] - c["Cl"] - c["Br"]
    assert q == 0 and intact(S, X, lat), (name, q)
    write(f"audit_ps4/bulkdef/lidef_{name}.extxyz", S, X, lat, f"src=lidef_from_{base}")
    print("built", name, "from", base, "| free S", ((S == "S") & ~bonded(S, X, lat)).sum(), "intact")

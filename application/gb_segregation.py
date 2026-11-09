#!/usr/bin/env python3
# =============================================================================
# gb_segregation.py -- halogen (Cl/Br) and free-S segregation to the grain
# boundary, for one MatterSim force field.  PAPER SS6 candidate B.
#
# Quantity: anion-exchange energy.  Take a BULK anion of species A and a GB
# anion of species B (A != B), and for the SAME frozen neighbourhood compute
#       dE = E(swapped A@GB,B@bulk ; local relax) - E(unswapped ; same local relax)
# The matched local reference (null relaxation) cancels residual-strain relief,
# so dE is robust to an imperfectly converged global structure and is a fair
# cross-model comparison.  dE < 0 => A prefers the GB relative to B.
# Averaging dE over many random (bulk-A, GB-B) pairs gives the mean segregation
# tendency of A vs B at this GB, for this force field.
#
# Mobile anion sublattice = {Cl, Br, free S(2-)}, where "free S" is any S not
# bonded to a P (argyrodite S/X site disorder).  Framework S (in PS4) is skipped.
#
# Regions (same x-windows as the diffusion runs, GB planes at 0.25/0.75 Lx):
#   GB   = within GB_HALF of 0.25*Lx or 0.75*Lx
#   BULK = within BULK_HALF of 0.50*Lx
#
#   python gb_segregation.py --data gbBIG_dec.extxyz --model pretrained \
#       --device cuda --nswap 20 --out seg_pretrained
# =============================================================================
import ase.constraints as _asec
if not hasattr(_asec, "full_3x3_to_voigt_6_stress"):
    from ase.stress import full_3x3_to_voigt_6_stress as _f3, voigt_6_to_full_3x3_stress as _v6
    _asec.full_3x3_to_voigt_6_stress = _f3
    _asec.voigt_6_to_full_3x3_stress = _v6

import argparse, json, os, sys, time
import numpy as np
from ase.io import read, write
from ase.optimize import FIRE
from ase.constraints import FixAtoms
from mattersim.forcefield import MatterSimCalculator


def make_calc(model, device):
    if model == "pretrained":
        return MatterSimCalculator(device=device)
    return MatterSimCalculator(load_path=model, device=device)


def relax(atoms, fmax, steps, logfile=None):
    dyn = FIRE(atoms, logfile=logfile, maxstep=0.1)   # gentle step avoids overshoot
    dyn.run(fmax=fmax, steps=steps)
    return atoms.get_potential_energy(), dyn.get_number_of_steps()


def region_labels(atoms, gb_half, bulk_half):
    Lx = atoms.cell[0, 0]
    x = atoms.get_positions()[:, 0] % Lx
    gb1, gb2, mid = 0.25 * Lx, 0.75 * Lx, 0.50 * Lx
    lab = np.full(len(atoms), "NONE", dtype=object)
    dgb = np.minimum(np.abs(x - gb1), np.abs(x - gb2))
    lab[dgb <= gb_half] = "GB"
    lab[np.abs(x - mid) <= bulk_half] = "BULK"
    return lab


def free_S_mask(atoms, ps_cut=2.6):
    """S atoms NOT bonded to any P (within ps_cut) = free anion S2-."""
    sym = np.array(atoms.get_chemical_symbols())
    isS = sym == "S"
    isP = sym == "P"
    if isP.sum() == 0:
        return isS
    pos = atoms.get_positions()
    Pidx = np.where(isP)[0]
    free = np.zeros(len(atoms), bool)
    for i in np.where(isS)[0]:
        d = atoms.get_distances(i, Pidx, mic=True)
        free[i] = d.min() > ps_cut
    return free


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="pretrained")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    ap.add_argument("--nswap", type=int, default=20, help="swaps per ordered species pair")
    ap.add_argument("--gb_half", type=float, default=3.5)
    ap.add_argument("--bulk_half", type=float, default=5.0)
    ap.add_argument("--relax_r", type=float, default=6.0, help="free-atom radius around swapped sites")
    ap.add_argument("--fmax", type=float, default=0.05)
    ap.add_argument("--ref_steps", type=int, default=1500)
    ap.add_argument("--swap_steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ps_cut", type=float, default=2.6, help="P-S bond cutoff; S beyond it = free anion")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    calc = make_calc(args.model, args.device)

    # --- reference relaxation -------------------------------------------------
    ref = read(args.data); ref.set_pbc(True); ref.calc = calc
    t0 = time.time()
    E_ref, nst = relax(ref, args.fmax, args.ref_steps)
    fm = np.linalg.norm(ref.get_forces(), axis=1).max()
    print(f"[ref] relaxed {nst} steps  fmax={fm:.4f}  E_ref={E_ref:.4f} eV  ({time.time()-t0:.0f}s)", flush=True)
    write(args.out + "_ref.extxyz", ref)
    R0 = ref.get_positions().copy()

    lab = region_labels(ref, args.gb_half, args.bulk_half)
    freeS = free_S_mask(ref, args.ps_cut)
    sym = np.array(ref.get_chemical_symbols())

    def pool(species, region):
        if species == "S":
            m = freeS & (lab == region)
        else:
            m = (sym == species) & (lab == region)
        return np.where(m)[0]

    species = ["Cl", "Br", "S"]
    counts = {(sp, rg): len(pool(sp, rg)) for sp in species for rg in ["GB", "BULK"]}
    print("[pools]", {f"{k[0]}@{k[1]}": v for k, v in counts.items()}, flush=True)

    csv = open(args.out + "_swaps.csv", "w")
    csv.write("A_to_GB,B_to_bulk,i_bulk,j_gb,dE_eV,nsteps\n")
    results = {}

    # ordered pairs: bring A (from bulk) to GB, displacing B (from GB) to bulk
    for A in species:
        for B in species:
            if A == B:
                continue
            bulkA = pool(A, "BULK")
            gbB = pool(B, "GB")
            if len(bulkA) == 0 or len(gbB) == 0:
                continue
            n = min(args.nswap, len(bulkA), len(gbB))
            ia = rng.choice(bulkA, size=min(len(bulkA), args.nswap), replace=False)
            jb = rng.choice(gbB, size=min(len(gbB), args.nswap), replace=False)
            dEs = []
            for k in range(min(len(ia), len(jb))):
                i, j = int(ia[k]), int(jb[k])
                # frozen mask: same for null and swap (identity swap keeps positions)
                di = ref.get_distances(i, range(len(ref)), mic=True)
                dj = ref.get_distances(j, range(len(ref)), mic=True)
                free = (di <= args.relax_r) | (dj <= args.relax_r)
                mask = ~free
                try:
                    # matched local reference (no swap), same frozen region -> cancels
                    # residual-strain relief so dE is robust to imperfect global relax
                    a0 = ref.copy(); a0.set_pbc(True); a0.calc = calc
                    a0.set_positions(R0); a0.set_constraint(FixAtoms(mask=mask))
                    E0, ns0 = relax(a0, args.fmax, args.swap_steps)
                    # swapped identities: A(bulk) -> GB site j, B(GB) -> bulk site i
                    a1 = ref.copy(); a1.set_pbc(True); a1.calc = calc
                    a1.set_positions(R0)
                    s = a1.get_chemical_symbols(); s[i], s[j] = B, A
                    a1.set_chemical_symbols(s); a1.set_constraint(FixAtoms(mask=mask))
                    E1, ns1 = relax(a1, args.fmax, args.swap_steps)
                except Exception as e:
                    print(f"  swap {A}->GB/{B}->bulk i={i} j={j} FAILED {e}", flush=True)
                    continue
                dE = E1 - E0
                dEs.append(dE)
                csv.write(f"{A},{B},{i},{j},{dE:.5f},{ns1}\n"); csv.flush()
            if dEs:
                arr = np.array(dEs)
                results[f"{A}_to_GB_vs_{B}"] = dict(
                    mean=float(arr.mean()), sem=float(arr.std(ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else 0.0),
                    n=len(arr), nfree=int(free.sum()))
                print(f"[{A}->GB vs {B}]  dE = {arr.mean():+.4f} +/- "
                      f"{arr.std(ddof=1)/np.sqrt(max(len(arr),1)):.4f} eV  (n={len(arr)})", flush=True)
    csv.close()
    out = dict(model=args.model, data=args.data, E_ref=float(E_ref),
               pools={f"{k[0]}@{k[1]}": v for k, v in counts.items()},
               params=vars(args), results=results)
    json.dump(out, open(args.out + ".json", "w"), indent=1)
    print("[done] wrote", args.out + ".json", flush=True)


if __name__ == "__main__":
    main()

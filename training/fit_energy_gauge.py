#!/usr/bin/env python
"""Fit the per-element energy gauge between a checkpoint and a labelled extxyz.

Why this exists
---------------
`matersim_model.pth` is not the stock MatterSim-1M: it is epoch 39 of an earlier
fine-tune (validation_metrics in the checkpoint: MAE_energy 0.106 eV/atom,
MAE_force 0.172 eV/A, MAE_stress 0) that was trained against **raw QE total
energies**. The v3 extxyz files carry **cohesive** energies
(raw QE total - sum_i atmene[type_i], see V3_EXTXYZ_README.md section 2).

Those two conventions differ by a constant per element, i.e. by ~274 eV/atom for
these compositions. Handing the cohesive labels straight to the checkpoint makes
the energy term of the loss start at 274 eV/atom and stay there: the frozen
normalizer shift cannot absorb it, and with HuberLoss it sits permanently in the
linear branch, injecting a constant-magnitude gradient into the same trunk that
produces the forces. That is a far better explanation of the v2 outcome
("Li improved, P 46 % worse") than the Huber delta alone.

A per-element constant is pure gauge: it changes no force and no stress, and it
cancels in every energy difference at fixed composition. So the fix is to state
the gauge explicitly, put the labels in the checkpoint's gauge for training, and
undo it when reporting. This script measures it:

    c = argmin_c || N c - (E_model - E_label) ||   over the training frames

where N is the composition matrix. After the shift the untouched checkpoint
reproduces the v3 labels to ~0.04 eV/atom, which is what it is actually worth.

    python scripts/fit_energy_gauge.py --model matersim_model.pth \
        --data data/v3_train_reduce_emb.extxyz --stride 5 --out energy_gauge.json
"""
import argparse
import json

# ASE >=3.24 moved full_3x3_to_voigt_6_stress; mattersim 1.2.0 wants the old path.
import ase.constraints as _asec
if not hasattr(_asec, "full_3x3_to_voigt_6_stress"):
    from ase.stress import (full_3x3_to_voigt_6_stress as _f3,
                            voigt_6_to_full_3x3_stress as _v6)
    _asec.full_3x3_to_voigt_6_stress = _f3
    _asec.voigt_6_to_full_3x3_stress = _v6

import numpy as np
from ase.io import read

ELEMENTS = ["Li", "P", "S", "Br", "I", "Cl"]


def composition_matrix(frames, elements=ELEMENTS):
    return np.array([[a.get_chemical_symbols().count(e) for e in elements]
                     for a in frames], float)


def predict_energies(potential, frames, batch_size=8):
    from mattersim.datasets.utils.build import build_dataloader
    labels = [float(a.get_potential_energy()) for a in frames]
    loader = build_dataloader(frames, labels, shuffle=False,
                              only_inference=True, batch_size=batch_size)
    energies, _, _ = potential.predict_properties(loader, include_forces=False)
    return np.array(labels), np.array(energies)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="checkpoint the gauge is measured against")
    ap.add_argument("--data", required=True, help="training extxyz (the gauge is fit on this)")
    ap.add_argument("--validate", nargs="*", default=[],
                    help="further extxyz files to check the gauge transfers to")
    ap.add_argument("--stride", type=int, default=5,
                    help="use every Nth frame of --data for the fit (default 5)")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="energy_gauge.json")
    args = ap.parse_args()

    from mattersim.forcefield.potential import Potential
    potential = Potential.from_checkpoint(load_path=args.model,
                                          load_training_state=False,
                                          device=args.device)

    frames = read(args.data, f"::{args.stride}")
    print(f"{args.data}: fitting on {len(frames)} frames "
          f"({len(set(a.get_chemical_formula() for a in frames))} formulas)")
    label, pred = predict_energies(potential, frames, args.batch_size)
    N = composition_matrix(frames)
    nat = N.sum(1)

    # Elements absent from the data leave their column at zero; lstsq returns 0
    # for them, which is the right answer (an unconstrained gauge direction).
    c, *_ = np.linalg.lstsq(N, pred - label, rcond=None)
    present = [e for e, n in zip(ELEMENTS, N.sum(0)) if n > 0]
    print("\nper-element gauge  c [eV]   (E_ckpt_gauge = E_cohesive + sum_i c_Zi)")
    for e, ci in zip(ELEMENTS, c):
        print("   %-3s %+12.4f%s" % (e, ci, "" if e in present else "   (absent from data)"))

    def stats(tag, fr):
        lab, prd = predict_energies(potential, fr, args.batch_size)
        Nf = composition_matrix(fr)
        n = Nf.sum(1)
        before = (prd - lab) / n
        after = (prd - (lab + Nf @ c)) / n
        print("  %-40s n=%5d  before MAE %10.3f -> after MAE %.4f  RMSE %.4f  max %.4f  eV/atom"
              % (tag, len(fr), np.abs(before).mean(), np.abs(after).mean(),
                 float(np.sqrt((after ** 2).mean())), np.abs(after).max()))
        return dict(n_frames=len(fr),
                    mae_before_per_atom=float(np.abs(before).mean()),
                    mae_after_per_atom=float(np.abs(after).mean()),
                    rmse_after_per_atom=float(np.sqrt((after ** 2).mean())),
                    max_after_per_atom=float(np.abs(after).max()))

    print("\nresidual energy error of the UNMODIFIED checkpoint, before/after the gauge:")
    fit_stats = {args.data: stats(args.data, frames)}
    for path in args.validate:
        fit_stats[path] = stats(path, read(path, "::2"))

    report = dict(elements=ELEMENTS,
                  c_eV=[float(x) for x in c],
                  elements_present=present,
                  source_model=args.model,
                  fit_data=args.data,
                  fit_stride=args.stride,
                  convention="E_model_gauge = E_extxyz_cohesive + sum_i c[element_i]",
                  fit_stats=fit_stats)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

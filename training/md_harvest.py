#!/usr/bin/env python
# =============================================================================
# md_harvest.py -- NVT MD on a DFT-sized cell that dumps FULL frames for
#                  DFT labelling (training-data harvest).
#
# diffusion_md.py keeps only Li trajectories (fine for MSD, useless for DFT).
# This writes every --dump_every step after --equil as extxyz with the MLIP
# energy/forces attached, so frames can be screened (novelty, filter_frames)
# and then single-pointed with make_scf_inputs.py --klen 26 --stress.
#
#   python md_harvest.py --data gbS_d4.extxyz --model pretrained --temp 800 \
#          --nsteps 10000 --equil 2000 --dump_every 500 --out mdH_d4_pre_800K
# =============================================================================
import argparse

import numpy as np

try:
    import ase.constraints as _asec
    if not hasattr(_asec, "full_3x3_to_voigt_6_stress"):
        from ase.stress import (full_3x3_to_voigt_6_stress as _f3,
                                 voigt_6_to_full_3x3_stress as _v6)
        _asec.full_3x3_to_voigt_6_stress = _f3
        _asec.voigt_6_to_full_3x3_stress = _v6
except Exception:
    pass

from ase import units
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="pretrained")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--temp", type=float, default=800.0)
    ap.add_argument("--dt", type=float, default=2.0, help="fs")
    ap.add_argument("--friction", type=float, default=0.01, help="1/fs")
    ap.add_argument("--nsteps", type=int, default=10000)
    ap.add_argument("--equil", type=int, default=2000)
    ap.add_argument("--dump_every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from mattersim.forcefield import MatterSimCalculator
    calc = (MatterSimCalculator(device=args.device)
            if args.model.lower() in ("", "pretrained", "default", "none")
            else MatterSimCalculator(load_path=args.model, device=args.device))

    at = read(args.data)
    at.set_pbc(True)
    at.calc = calc
    MaxwellBoltzmannDistribution(at, temperature_K=args.temp, rng=np.random.default_rng(args.seed))
    Stationary(at)
    dyn = Langevin(at, args.dt * units.fs, temperature_K=args.temp, friction=args.friction / units.fs,
                   rng=np.random.default_rng(args.seed + 1))

    fout = args.out + "_frames.extxyz"
    open(fout, "w").close()
    state = {"n": 0}

    def dump():
        step = dyn.nsteps
        if step <= args.equil or step % args.dump_every:
            return
        fr = at.copy()
        fr.calc = SinglePointCalculator(fr, energy=at.get_potential_energy(), forces=at.get_forces())
        fr.info.update(step=int(step), temp_K=float(at.get_temperature()), md_T=args.temp,
                       model=args.model, src=args.out)
        write(fout, fr, append=True)
        state["n"] += 1
        print("step %6d  T %6.1f K  Epot %.4f eV  dumped %d" %
              (step, at.get_temperature(), at.get_potential_energy(), state["n"]), flush=True)

    dyn.attach(dump, interval=1)
    dyn.run(args.nsteps)
    print("wrote %s (%d frames)" % (fout, state["n"]))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# =============================================================================
# framework_md.py -- Tier-3 mechanism run: SHORT MD that dumps ALL atoms so the
# framework dynamics (PS4 paddlewheel rotation, anion sublattice motion) can be
# correlated with Li hops at the GB vs bulk.
#
# diffusion_md.py only stores the Li sublattice, which is enough for D and the
# Li-only mechanism (density/cage/alpha2/van Hove). To see WHY the phaseB GB
# traps Li we need the anion framework too, so this writes the full-atom
# trajectory (raw positions; unwrap/minimum-image is redone in post-processing).
#
# Deliberately short (few tens of ps): rotational correlation times are ~ps, and
# this is a per-frame full dump, so it stays cheap. Same thermostat / dt / cell
# conventions as diffusion_md.py so it is comparable to the production runs.
#
#   python framework_md.py --data gbBIG_dec.extxyz --model models/phaseB.pth \
#       --temp 700 --dt 2.0 --nsteps 20000 --equil 5000 --dump_every 25 \
#       --out fw_phaseB_700K
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
from ase.io import read as ase_read
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary


class FullDump:
    """Store every atom's raw position every N steps (cheap, short run)."""

    def __init__(self, atoms, dyn, every, dt_ps):
        self.atoms = atoms
        self.dyn = dyn
        self.every = every
        self.dt_ps = dt_ps
        self.frames = []
        self.t = []
        self.off = dyn.get_number_of_steps()

    def __call__(self):
        istep = self.dyn.get_number_of_steps() - self.off
        if istep % self.every != 0:
            return
        self.frames.append(self.atoms.get_positions().astype(np.float32).copy())
        self.t.append(istep * self.dt_ps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="pretrained")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--temp", type=float, default=700.0)
    ap.add_argument("--dt", type=float, default=2.0)
    ap.add_argument("--nsteps", type=int, default=20000)
    ap.add_argument("--equil", type=int, default=5000)
    ap.add_argument("--friction", type=float, default=0.01)
    ap.add_argument("--dump_every", type=int, default=25, help="dump all atoms every N steps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="fw")
    args = ap.parse_args()

    atoms = ase_read(args.data)
    atoms.set_pbc([True, True, True])

    from mattersim.forcefield import MatterSimCalculator
    if args.model.lower() in ("", "pretrained", "default", "none"):
        atoms.calc = MatterSimCalculator(device=args.device)
    else:
        atoms.calc = MatterSimCalculator(load_path=args.model, device=args.device)
    print("composition:", atoms.get_chemical_formula(),
          " cell:", atoms.get_cell().lengths(), " natoms:", len(atoms), flush=True)

    rng = np.random.RandomState(args.seed)
    MaxwellBoltzmannDistribution(atoms, temperature_K=args.temp, force_temp=True, rng=rng)
    Stationary(atoms)

    dt = args.dt * units.fs
    lrng = np.random.RandomState(args.seed + 10007)
    dyn = Langevin(atoms, dt, temperature_K=args.temp,
                   friction=args.friction / units.fs, rng=lrng)

    if args.equil > 0:
        print("equilibrating %d steps..." % args.equil, flush=True)
        dyn.run(args.equil)

    dump = FullDump(atoms, dyn, args.dump_every, args.dt / 1000.0)
    dyn.attach(dump, interval=1)
    print("production %d steps at %.0f K (dump every %d = %.3f ps)..."
          % (args.nsteps, args.temp, args.dump_every, args.dump_every * args.dt / 1000.0),
          flush=True)
    dyn.run(args.nsteps)

    pos = np.array(dump.frames, dtype=np.float32)          # (nframe, natoms, 3) RAW
    np.savez_compressed(args.out + "_fwtraj.npz",
                        pos=pos, t_ps=np.array(dump.t),
                        cell=np.array(atoms.get_cell()),
                        symbols=np.array(atoms.get_chemical_symbols()))
    print("wrote %s_fwtraj.npz (%d frames x %d atoms)"
          % (args.out, len(dump.frames), len(atoms)), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Make a LOCALLY amorphous (disordered, thick) grain boundary by a region-
restricted melt-quench with a MatterSim force field. Li-trapping candidate ③.

The GB slab (|frac_x - gb_pos| < half-width) is heated to T_hot (mobile Li/anion
sublattice melts; PS4 is robust and mostly survives), held, then quenched to
T_cold and briefly relaxed. Bulk atoms outside the slab are frozen (FixAtoms) so
only the interface disorders -> a thicker amorphous GB layer. Half-width sets the
interface thickness.

Audit PS4 integrity (n_vertexX, broken PS4) AFTER quench before trusting it; this
is OUT of v5B's strict training domain, so a few DFT single points are advised.

  python amorphize_gb.py --data gbBIG_uni.extxyz --model v5B.pth --device cuda \
     --gb-pos 0.25,0.75 --half-width 0.10 --t_hot 2000 --t_cold 300 \
     --hot_ps 10 --quench_ps 10 --relax_ps 5 --out gbBIG_amoX.extxyz
"""
import argparse
import numpy as np
import ase.constraints as _asec
if not hasattr(_asec, "full_3x3_to_voigt_6_stress"):
    from ase.stress import full_3x3_to_voigt_6_stress as _f3, voigt_6_to_full_3x3_stress as _v6
    _asec.full_3x3_to_voigt_6_stress = _f3
    _asec.voigt_6_to_full_3x3_stress = _v6
from ase.io import read, write
from ase.constraints import FixAtoms
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units


def gb_mask(atoms, gb_pos, hw, axis=0):
    f = atoms.get_scaled_positions()[:, axis] % 1.0
    m = np.zeros(len(atoms), bool)
    for p in gb_pos:
        d = np.abs(((f - p + 0.5) % 1.0) - 0.5)
        m |= d < hw
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="v5B.pth")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gb-pos", default="0.25,0.75")
    ap.add_argument("--gb-axis", default="x")
    ap.add_argument("--half-width", type=float, default=0.10, help="GB slab half-width (frac of Lx) = thickness knob")
    ap.add_argument("--t_hot", type=float, default=2000.0)
    ap.add_argument("--t_cold", type=float, default=300.0)
    ap.add_argument("--dt", type=float, default=1.0, help="fs (smaller for the hot stage)")
    ap.add_argument("--hot_ps", type=float, default=10.0)
    ap.add_argument("--quench_ps", type=float, default=10.0)
    ap.add_argument("--relax_ps", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    axis = {"x": 0, "y": 1, "z": 2}[a.gb_axis]
    gb_pos = [float(x) for x in a.gb_pos.split(",")]
    atoms = read(a.data)

    from mattersim.forcefield import MatterSimCalculator
    atoms.calc = (MatterSimCalculator(device=a.device) if a.model.lower() in ("", "pretrained")
                  else MatterSimCalculator(load_path=a.model, device=a.device))

    mob = gb_mask(atoms, gb_pos, a.half_width, axis)
    frozen = np.where(~mob)[0]
    atoms.set_constraint(FixAtoms(indices=frozen.tolist()))
    print(f"mobile (GB slab) atoms: {mob.sum()} / {len(atoms)}  (frozen bulk {len(frozen)})")

    rng = np.random.default_rng(a.seed)
    MaxwellBoltzmannDistribution(atoms, temperature_K=a.t_hot, rng=rng)
    dt = a.dt * units.fs

    def run(nps, T):
        steps = int(nps * 1000 / a.dt)
        dyn = Langevin(atoms, dt, temperature_K=T, friction=0.01)
        dyn.run(steps)

    # hot (melt the interface)
    run(a.hot_ps, a.t_hot)
    # quench: step T down linearly in a few stages
    nstage = 10
    for k in range(nstage):
        T = a.t_hot + (a.t_cold - a.t_hot) * (k + 1) / nstage
        run(a.quench_ps / nstage, T)
    # cold relax
    run(a.relax_ps, a.t_cold)

    atoms.set_constraint()  # drop constraints before writing
    write(a.out, atoms, format="extxyz")
    print(f"wrote {a.out}: {len(atoms)} atoms (half_width={a.half_width}, t_hot={a.t_hot})")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# =============================================================================
# diffusion_md.py -- Li self-diffusion MD with grain-boundary vs bulk resolution
#
# Python/MatterSim port of the LAMMPS `in.deepMD.npt` recipe:
#
#     compute msd Li  msd com no                 # global Li MSD
#     region GB01 block 12 18 ... ; region GB02 ... ; region BULK 28 33 ...
#     compute msd2 GB01 ; ... fix ave/time ... file out_Li_msd.GB01.txt
#     fix npt all npt temp 300 300 0.5 aniso 1 1 0.5 ; run 100000
#
# Runs NVT (or NPT) MD with MatterSim, computes region-resolved Li MSD online
# (regions = spatial blocks along an axis, assigned from t0, like LAMMPS), and
# writes MSD(t) and the Einstein diffusion coefficient D for each region.
#
# Run one temperature per invocation; sweep T and feed the msd_*.csv files to
# msd_arrhenius.py for D(T) and the activation energy.
#
#   python diffusion_md.py --data output00.data --model best_model.pth \
#          --temp 600 --nsteps 200000 --dt 1.0 --species Li \
#          --regions "GB01:x:12:18 GB02:x:43:50 BULK:x:28:33" --out msd_600K
# =============================================================================
import argparse
import os
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
from ase.io.lammpsdata import read_lammps_data
from ase.io import read as ase_read
from ase.md.langevin import Langevin
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

AXIS = {"x": 0, "y": 1, "z": 2}


def readcellInfo(inputdata):
    cellIn = np.zeros(3)
    with open(inputdata) as ip1:
        for line1 in ip1:
            aa = line1.split()
            if len(aa) > 2:
                if aa[2] == "xlo":
                    cellIn[0] = float(aa[1]) - float(aa[0])
                if aa[2] == "ylo":
                    cellIn[1] = float(aa[1]) - float(aa[0])
                if aa[2] == "zlo":
                    cellIn[2] = float(aa[1]) - float(aa[0])
    return cellIn


def lammps_types(path, atom_style):
    """LAMMPS integer atom types in ascending-atom-id order (== read_lammps_data
    order). read_lammps_data guesses elements from Masses, which silently
    mislabels S(28.085->Si) and Br(78.904->Se); for atom_style 'full' the type
    is not in atoms.get_tags() either."""
    col = {"atomic": 1, "charge": 1, "full": 2, "molecular": 2}.get(atom_style, 1)
    ids, types = [], []
    with open(path) as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        if lines[i].split()[:1] == ["Atoms"]:
            i += 1
            while i < len(lines) and lines[i].strip() == "":
                i += 1
            while i < len(lines) and lines[i].strip() != "":
                p = lines[i].split()
                ids.append(int(p[0]))
                types.append(int(p[col]))
                i += 1
            break
        i += 1
    order = np.argsort(ids)
    return np.asarray(types)[order]


class MSDMonitor:
    """Online, PBC-unwrapped MSD for a chosen species, optionally split into
    fixed spatial regions assigned from the reference (t0) configuration.
    com=False (like LAMMPS `msd com no`): no centre-of-mass subtraction."""

    def __init__(self, atoms, dyn, species, regions, dt_ps, nsample, out, com=False,
                 traj_every=0):
        self.atoms = atoms
        self.dyn = dyn
        self.dt_ps = dt_ps
        self.nsample = nsample
        self.com = com
        sym = np.asarray(atoms.get_chemical_symbols())
        self.idx = np.where(sym == species)[0]
        if len(self.idx) == 0:
            raise ValueError("no %s atoms found" % species)
        self.ref = atoms.get_positions()[self.idx].copy()
        self.prev = self.ref.copy()
        self.unwrapped = self.ref.copy()
        # region membership fixed at t0
        self.groups = {"ALL": np.arange(len(self.idx))}
        pos0 = atoms.get_positions()[self.idx]
        for name, ax, lo, hi in regions:
            m = (pos0[:, AXIS[ax]] >= lo) & (pos0[:, AXIS[ax]] <= hi)
            if m.any():
                self.groups[name] = np.where(m)[0]
        self.rows = []
        self.out = out
        # unwrapped positions of the tracked species, sampled coarsely. MD is
        # expensive and analysis is cheap: keeping the trajectory lets the region
        # decomposition be redone with other time origins / fit windows later.
        self.traj_every = traj_every
        self.traj, self.traj_t = [], []
        self.cell = np.array(atoms.get_cell())
        self.step0 = 0          # production steps already done in an earlier job
        # dyn has already run the equilibration; measure production time from here
        # (the old code let the equilibration steps offset the MSD time axis, which
        # skewed msd_arrhenius's "second half" tail selection).
        self.istep_off = dyn.get_number_of_steps()
        self.last_istep = -1    # dedupe: ASE calls observers once before step 1
        print("MSD groups:", {k: len(v) for k, v in self.groups.items()})

    def state(self):
        """Everything needed to resume the MSD accumulation in a later job."""
        st = {"ref": self.ref, "prev": self.prev, "unwrapped": self.unwrapped,
              "rows": np.array(self.rows) if self.rows else np.zeros((0, 1 + len(self.groups))),
              "names": np.array(list(self.groups)), "step0": self.step0,
              "cell": self.cell,
              "traj": (np.array(self.traj, dtype=np.float32) if self.traj
                       else np.zeros((0, len(self.idx), 3), dtype=np.float32)),
              "traj_t": np.array(self.traj_t, dtype=np.float64)}
        # region membership is defined at t0 and must NOT be re-derived from the
        # resumed positions (atoms have moved; some have left the window)
        for name, g in self.groups.items():
            st["g_" + name] = g
        return st

    def load_state(self, npz):
        self.ref = npz["ref"]; self.prev = npz["prev"]; self.unwrapped = npz["unwrapped"]
        self.rows = [list(r) for r in npz["rows"]]
        self.step0 = int(npz["step0"])
        self.groups = {str(n): npz["g_" + str(n)] for n in npz["names"]}
        if "traj" in npz.files and len(npz["traj"]):
            self.traj = [a for a in npz["traj"]]
            self.traj_t = [float(t) for t in npz["traj_t"]]
        if self.rows:
            self.last_istep = int(round(self.rows[-1][0] / self.dt_ps))
        print("resumed MSD state: %d production steps, %d samples, groups %s"
              % (self.step0, len(self.rows), {k: len(v) for k, v in self.groups.items()}))

    def _unwrap(self):
        cell = np.array(self.atoms.get_cell())
        pos = self.atoms.get_positions()[self.idx]
        d = pos - self.prev
        frac = d @ np.linalg.inv(cell)
        frac -= np.round(frac)                  # minimum image of the increment
        self.unwrapped += frac @ cell
        self.prev = pos

    def __call__(self):
        self._unwrap()
        istep = self.step0 + self.dyn.get_number_of_steps() - self.istep_off
        if istep % self.nsample != 0 or istep <= self.last_istep:
            return
        self.last_istep = istep
        if self.traj_every and istep % self.traj_every == 0:
            self.traj.append(self.unwrapped.astype(np.float32).copy())
            self.traj_t.append(istep * self.dt_ps)
        disp = self.unwrapped - self.ref
        row = [istep * self.dt_ps]
        for name, g in self.groups.items():
            dg = disp[g]
            if self.com:
                dg = dg - dg.mean(axis=0)
            row.append(float((dg ** 2).sum(axis=1).mean()))  # <|dr|^2> [A^2]
        self.rows.append(row)

    def save(self):
        names = list(self.groups)
        hdr = "time_ps," + ",".join("msd_%s_A2" % n for n in names)
        arr = np.array(self.rows)
        np.savetxt(self.out + ".csv", arr, delimiter=",", header=hdr, comments="")
        if self.traj:
            np.savez_compressed(
                self.out + "_traj.npz",
                pos=np.array(self.traj, dtype=np.float32),   # (nframe, nLi, 3) UNWRAPPED
                t_ps=np.array(self.traj_t), cell=self.cell,
                idx=self.idx, **{"g_" + k: v for k, v in self.groups.items()})
            print("wrote %s_traj.npz (%d frames x %d %s)"
                  % (self.out, len(self.traj), len(self.idx), "atoms"))
        # Einstein D from a robust linear fit over the second half (skip ballistic toe)
        t = arr[:, 0]
        half = t > 0.5 * t.max()
        print("== diffusion coefficients (Einstein, 2nd half of MSD) ==")
        for j, n in enumerate(names, start=1):
            slope = np.polyfit(t[half], arr[half, j], 1)[0]  # A^2/ps
            D = slope / 6.0 * 1e-4                            # A^2/ps -> cm^2/s
            print("  %-6s D = %10.3e cm^2/s   (MSD slope %.4f A^2/ps)" % (n, D, slope))
        print("wrote %s.csv" % self.out)


def main():
    ap = argparse.ArgumentParser(description="region-resolved Li diffusion MD")
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="best_model.pth")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--atom_style", default="atomic")
    ap.add_argument("--symbols", default="", help="'Li P S Br I Cl' to force type->element")
    ap.add_argument("--species", default="Li")
    ap.add_argument("--temp", type=float, default=300.0)
    ap.add_argument("--dt", type=float, default=1.0, help="timestep [fs]")
    ap.add_argument("--nsteps", type=int, default=200000)
    ap.add_argument("--equil", type=int, default=20000, help="equilibration steps before MSD t0")
    ap.add_argument("--thermostat", default="langevin", choices=["langevin", "berendsen"])
    ap.add_argument("--friction", type=float, default=0.01)
    ap.add_argument("--taut", type=float, default=100.0)
    ap.add_argument("--nsample", type=int, default=10, help="MSD sample interval [steps]")
    ap.add_argument("--regions", default="", help="'GB01:x:12:18 BULK:x:28:33' (A, from t0)")
    ap.add_argument("--com", action="store_true", help="subtract group COM (LAMMPS msd com yes)")
    ap.add_argument("--out", default="msd")
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed for the initial velocities and the Langevin noise; "
                         "vary it to get independent trajectories at the same T")
    ap.add_argument("--repeat", default="",
                    help="'1 1 3' to replicate the cell before the run (use the "
                         "directions PARALLEL to the GB so the GB region gains atoms)")
    ap.add_argument("--ckpt_every", type=int, default=0,
                    help="write a restart checkpoint every N production steps (0 = off)")
    ap.add_argument("--traj_every", type=int, default=50,
                    help="store the unwrapped positions of --species every N steps "
                         "(0 = off). Lets the region decomposition be re-analysed "
                         "with other time origins / windows without re-running MD.")
    ap.add_argument("--resume", action="store_true",
                    help="continue from <out>_ckpt.{extxyz,npz} if present")
    args = ap.parse_args()

    if args.data.endswith((".data", ".lmp")):
        atoms = read_lammps_data(args.data, atom_style=args.atom_style)
        cellIn = readcellInfo(args.data)
        if np.all(cellIn > 0):
            atoms.set_cell([cellIn[0], cellIn[1], cellIn[2]])
        atoms.set_pbc([True, True, True])
        if args.symbols:
            syms = args.symbols.split()
            types = lammps_types(args.data, args.atom_style)
            if len(types) == len(atoms):
                atoms.set_chemical_symbols([syms[t - 1] for t in types])
            else:
                raise ValueError("parsed %d types but %d atoms" % (len(types), len(atoms)))
    else:
        atoms = ase_read(args.data)
        atoms.set_pbc([True, True, True])

    if args.repeat:
        rep = tuple(int(x) for x in args.repeat.split())
        n0 = len(atoms)
        atoms = atoms.repeat(rep)
        atoms.set_pbc([True, True, True])
        print("replicated %s: %d -> %d atoms" % (rep, n0, len(atoms)))

    from mattersim.forcefield import MatterSimCalculator
    if args.model.lower() in ("", "pretrained", "default", "none"):
        atoms.calc = MatterSimCalculator(device=args.device)      # built-in generic MatterSim
    else:
        atoms.calc = MatterSimCalculator(load_path=args.model, device=args.device)
    print("composition:", atoms.get_chemical_formula(), " cell:", atoms.get_cell().lengths())

    regions = []
    for tok in args.regions.split():
        name, ax, lo, hi = tok.split(":")
        regions.append((name, ax, float(lo), float(hi)))

    ck_xyz, ck_npz = args.out + "_ckpt.extxyz", args.out + "_ckpt.npz"
    resumed = args.resume and os.path.exists(ck_xyz) and os.path.exists(ck_npz)
    if resumed:
        prev = ase_read(ck_xyz)                 # carries positions AND momenta
        atoms.set_positions(prev.get_positions())
        atoms.set_momenta(prev.get_momenta())
        print("resumed geometry from", ck_xyz)
    else:
        rng = np.random.RandomState(args.seed)
        MaxwellBoltzmannDistribution(atoms, temperature_K=args.temp, force_temp=True, rng=rng)
        Stationary(atoms)

    dt = args.dt * units.fs
    lrng = np.random.RandomState(args.seed + 10007)   # independent of the velocity draw
    if args.thermostat == "langevin":
        dyn = Langevin(atoms, dt, temperature_K=args.temp,
                       friction=args.friction / units.fs, rng=lrng)
    else:
        dyn = NVTBerendsen(atoms, dt, temperature_K=args.temp, taut=args.taut * units.fs)

    if args.equil > 0 and not resumed:
        print("equilibrating %d steps..." % args.equil)
        dyn.run(args.equil)

    mon = MSDMonitor(atoms, dyn, args.species, regions,
                     dt_ps=args.dt / 1000.0, nsample=args.nsample, out=args.out,
                     com=args.com, traj_every=args.traj_every)
    if resumed:
        mon.load_state(np.load(ck_npz, allow_pickle=False))
    dyn.attach(mon, interval=1)                 # unwrap needs every step

    def checkpoint():
        from ase.io import write as ase_write
        st = mon.state()
        st["step0"] = mon.step0 + dyn.get_number_of_steps() - mon.istep_off
        np.savez(ck_npz + ".tmp.npz", **st)
        os.replace(ck_npz + ".tmp.npz", ck_npz)
        ase_write(ck_xyz + ".tmp", atoms, format="extxyz")   # momenta included
        os.replace(ck_xyz + ".tmp", ck_xyz)
        mon.save()                                           # keep the csv current too

    todo = args.nsteps - mon.step0
    if todo <= 0:
        print("already at %d/%d production steps; nothing to do" % (mon.step0, args.nsteps))
        mon.save(); return
    print("production %d steps at %.0f K (seed %d, %d already done)..."
          % (todo, args.temp, args.seed, mon.step0))
    if args.ckpt_every > 0:
        done = 0
        while done < todo:
            chunk = min(args.ckpt_every, todo - done)
            dyn.run(chunk); done += chunk
            checkpoint()
            print("  checkpoint at %d/%d production steps" % (mon.step0 + done, args.nsteps), flush=True)
    else:
        dyn.run(todo)
    mon.save()


if __name__ == "__main__":
    main()

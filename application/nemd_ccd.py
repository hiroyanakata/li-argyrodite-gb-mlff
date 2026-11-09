#!/usr/bin/env python
# =============================================================================
# nemd_ccd.py -- chemical color-diffusion NEMD (CCD-NEMD) for Li6PS5X
#
# Method: Sasaki, Gao, Hitosugi & Tateyama, npj Comput. Mater. 9, 48 (2023),
#         "Nonequilibrium MD for accelerated computation of ion-ion correlated
#          conductivity beyond Nernst-Einstein limitation".
#
# Why this exists
# ---------------
# diffusion_md.py measures D_self.  Turning D_self into a conductivity needs the
# Nernst-Einstein dilute-solution approximation, which throws away the distinct
# (ion-ion correlated) term.  The exact sigma is the *collective* charge MSD
#
#     sigma ~ lim (1/t) < | sum_i z_i dx_i(t) |^2 >
#
# which is ONE random variable no matter how many carriers there are, so its
# relative error falls as 1/sqrt(n_origins), not 1/sqrt(N * n_origins).  For the
# 1836-Li bigGB cell that is a factor ~N = 1800 more sampling for the same
# precision as D_self.  That is why nobody computes it from EMD.
#
# NEMD converts that variance measurement into a *mean* measurement: apply a
# static color field, let a steady flux build up, and read off the mean flux.
# Signal grows like t, noise like sqrt(t), and the signal can be dialled up with
# the field.  What comes out is the correlated sigma, and sigma_dilute/sigma
# gives the Haven ratio.
#
# Color charges ("chemical", i.e. CCD not CD)
# -------------------------------------------
# The older CD-NEMD assigns c_i = (-1)^i over the Li, which pushes half the Li
# against the other half, destroys cooperative motion, and therefore recovers
# only the Nernst-Einstein answer.  CCD-NEMD assigns the real formal valency of
# each chemical unit:
#
#     Li = +1,   PS4 unit = -3 (spread as -0.6 over P and its four S),
#     free S(2-) = -2,   Cl/Br/I = -1
#
# For the argyrodite GB cells this is exactly neutral, e.g. gbBIG_dec
# (Li612 P106 S478 Br93 Cl93): 106 PS4 use 424 S, 54 S are free, and
# 612 - 3*106 - 2*54 - 93 - 93 = 0.  Neutrality sum(c_i) = 0 is required for
# momentum conservation; we check it and, if a decorated/defective cell is not
# neutral, remove the residual as a mass-weighted counter-force so that the net
# force stays exactly zero (the color charges used for the flux are untouched).
#
# Thermostat
# ----------
# The thermostat must NOT act along the field direction, or the effective drive
# becomes c_i*Fe - alpha*p_i and the thermostat eats the drift (paper, Eq. 3).
# We use ASE Langevin with a per-atom-per-direction friction array that is zero
# along the field axis: there the integrator degenerates to plain velocity
# Verlet (c1 = dt/2, c2 = c3 = c4 = c5 = 0), while the two perpendicular
# directions absorb the Joule heat.  T_par and T_perp are both logged so the
# heating can be checked.
#
# Observables (block-averaged, blocks are the error bars)
# -------------------------------------------------------
#   Jc      = (1/V) sum_i c_i v_i,field          global color flux
#   Jc_k    = (1/V_k) sum_{i in slab k} c_i v_i  local color flux, slab assigned
#             from the CURRENT position every step (no t0-label ambiguity --
#             this is the point the paper makes against region-resolved EMD)
#   n_k     = Li number density per slab (steady-state redistribution check)
#   sigma   = e^2 <Jc> / Fe  ->  16.02177 * (Jc [1/(A^2 ps)]) / (Fe [eV/A]) S/cm
#
# Geometry -- (A) and (B) are different runs
# ------------------------------------------
#   (A) field PARALLEL to the GB plane (--field-axis y or z, --profile-axis x):
#       slabs are resistors in PARALLEL, so the local sigma(x) is well defined
#       and directly measurable.  This is what the paper did for the LLZO GB.
#   (B) field PERPENDICULAR to the GB plane (--field-axis x): slabs are in
#       SERIES, continuity forces Jc_k to be uniform, and what you measure is
#       the total blocked conductivity of the cell.  Feed that to
#       nemd_analysis.py --rgb together with an (A) run to get R_GB [ohm cm^2].
#
# Examples
#   # (A) in-plane, local sigma(x) + Haven ratio
#   python nemd_ccd.py --data gbBIG_dec.extxyz --model phaseB.pth --device cuda \
#       --repeat "1 1 3" --temp 700 --field 0.02 --field-axis z \
#       --profile-axis x --nbins 40 --nsteps 50000 --out nemd_phaseB_700K_z_F020_s0
#
#   # (B) cross-plane, blocked conductivity
#   python nemd_ccd.py --data gbBIG_dec.extxyz --model phaseB.pth --device cuda \
#       --repeat "1 1 3" --temp 700 --field 0.02 --field-axis x \
#       --profile-axis x --nbins 40 --nsteps 50000 --out nemd_phaseB_700K_x_F020_s0
#
#   # field sweep for the linear-response check (include 0.0 = EMD baseline)
#   for F in 0.0 0.005 0.01 0.02 0.04; do ... --field $F ... ; done
# =============================================================================
import argparse
import os

import numpy as np

# --- mattersim 1.2.0 / newer ASE shim (same as diffusion_md.py) --------------
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
from ase.calculators.calculator import Calculator, all_changes
from ase.io import read as ase_read
from ase.io.lammpsdata import read_lammps_data
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

AXIS = {"x": 0, "y": 1, "z": 2}

# sigma[S/cm] = SIGMA_CONV * Jc[1/(A^2 ps)] / Fe[eV/A]
#   sigma = e^2 Jc / Fe ; e^2/Fe = (1.602176634e-19)^2 / (Fe * 1.602176634e-9 N)
#   Jc in SI = j * 1e32 m^-2 s^-1  ->  1602.177 * j/f  S/m  =  16.02177 * j/f S/cm
SIGMA_CONV = 16.02176634
KB_J = 1.380649e-23
E_C = 1.602176634e-19

HALIDES = ("F", "Cl", "Br", "I")


# ---------------------------------------------------------------------------
# color charges
# ---------------------------------------------------------------------------
def assign_color_charges(atoms, bond_cut=2.8, search_cut=3.6, verbose=True):
    """Formal valency of each chemical unit, spread over the atoms of that unit.

    Argyrodite Li6PS5X is Li+ / PS4(3-) / free S(2-) / free X(-).  So:

        Li                          -> +1
        PS4 unit                    -> -3, spread as -0.6 over P and its four S
        free S (cage 4a/4d site)    -> -2
        free halide                 -> -1

    Two things this has to get right on these GB cells:

    1. The tetrahedra are a MATCHING, not four independent nearest-neighbour
       lookups.  Neighbouring units otherwise steal each other's corners and
       some P silently end up with two or three, which broke neutrality by
       -218 e on gbBIG_dec.  We walk all P-anion pairs in order of increasing
       distance and take each only if that P still needs a corner and that
       anion is still free.

    2. gbBIG_dec and gbBIG_def have halides sitting ON tetrahedron corners
       (35% and 8% of all halides respectively), not only on the free-anion
       sublattice -- see NEMD.md.  Such a unit is PS(4-n)Xn with formal charge
       -(3 - n), not -3.  We therefore read n off the matched corners per P
       instead of assuming PS4.  With that rule all three cells come out
       exactly neutral, which is the check that the rule is the right one.
    """
    sym = np.asarray(atoms.get_chemical_symbols())
    c = np.zeros(len(atoms))
    c[sym == "Li"] = +1.0

    p_idx = np.where(sym == "P")[0]
    an_idx = np.where(np.isin(sym, ("S",) + HALIDES))[0]
    is_hal = np.isin(sym[an_idx], HALIDES)
    owner = np.full(len(an_idx), -1)          # index into p_idx, -1 = free anion
    corners = [[] for _ in p_idx]

    dmax = 0.0
    if len(p_idx) and len(an_idx):
        from ase.geometry import get_distances
        pos = atoms.get_positions()
        _, d = get_distances(pos[p_idx], pos[an_idx],
                             cell=atoms.get_cell(), pbc=atoms.get_pbc())
        ip, ia = np.where(d < search_cut)
        for k in np.argsort(d[ip, ia]):
            u, v = ip[k], ia[k]
            if len(corners[u]) == 4 or owner[v] >= 0:
                continue
            owner[v] = u
            corners[u].append(v)
            dmax = max(dmax, d[u, v])

        nhal_corner = 0
        for u, cs in enumerate(corners):
            if len(cs) < 4:
                print("  [warn] P %d matched only %d corners within %.1f A"
                      % (p_idx[u], len(cs), search_cut))
            nx = int(is_hal[cs].sum())
            nhal_corner += nx
            q = -(3.0 - nx) / (1 + len(cs))    # PS(4-n)Xn unit, spread over P+corners
            c[p_idx[u]] = q
            c[an_idx[cs]] = q
    else:
        nhal_corner = 0

    free = owner < 0
    c[an_idx[free & ~is_hal]] = -2.0
    c[an_idx[free & is_hal]] = -1.0

    unknown = sorted(set(sym) - {"Li", "P", "S"} - set(HALIDES))
    if unknown:
        print("  [warn] species with no color charge (set to 0):", unknown)

    tot = c.sum()
    if verbose:
        nhal = int(is_hal.sum())
        print("color charges: %d P-centred units (longest P-anion bond %.2f A), "
              "free S %d, free halide %d, Li %d"
              % (len(p_idx), dmax, int((free & ~is_hal).sum()),
                 int((free & is_hal).sum()), int((sym == "Li").sum())))
        if nhal_corner:
            print("               %d of %d halides (%.0f%%) sit ON a tetrahedron "
                  "corner -> those units are PS(4-n)Xn, not PS4"
                  % (nhal_corner, nhal, 100.0 * nhal_corner / max(nhal, 1)))
        print("               sum(c_i) = %+.6f  (must be 0 for momentum "
              "conservation)" % tot)
    return c, tot


# ---------------------------------------------------------------------------
# integrator with the color field
# ---------------------------------------------------------------------------
class ColorFieldCalculator(Calculator):
    """Wraps the real calculator and adds the constant color force c_i*Fe.

    The field MUST live in the calculator, not in the integrator. ASE's
    Langevin.step() evaluates the force twice per position: once passed in for
    the first half-step, and once recomputed inside step() for the second
    half-step at the new positions. Adding the color force only to the argument
    of step() therefore reaches only half of the uses of each force, so the
    drive comes out at HALF the requested field -- a silent factor-2 error in
    sigma. Putting it here makes every atoms.get_forces() carry it.

    The residual (nonzero only if sum(c_i) != 0) is removed mass-weighted, so
    the net force is exactly zero and the total momentum is conserved. A uniform
    shift of all velocities leaves Jc unchanged because sum(c_i) = 0, so
    removing the centre-of-mass motion never touches the observable.
    """

    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(self, base, colors, field_axis, field, masses):
        super().__init__()
        self.base = base
        cf = np.zeros((len(colors), 3))
        cf[:, field_axis] = colors * field                       # eV/A
        cf[:, field_axis] -= masses * (cf[:, field_axis].sum() / masses.sum())
        self.cforce = cf

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        work = self.atoms                 # the copy Calculator.calculate stored
        work.calc = self.base
        e = work.get_potential_energy()
        f = work.get_forces()
        # The field's own potential (-Fe * sum c_i x_i) is not single-valued
        # under PBC, so only the forces carry it; the reported energy is the
        # unperturbed one and is a diagnostic, not a conserved quantity.
        self.results = {"energy": e, "free_energy": e, "forces": f + self.cforce}


class ColorLangevin(Langevin):
    """Langevin whose friction is zero along the field axis, so the thermostat
    cannot eat the drift (there the integrator degenerates to plain velocity
    Verlet). The field itself comes from ColorFieldCalculator."""

    def __init__(self, atoms, timestep, temperature_K, friction, field_axis,
                 rng=None):
        fr = np.zeros((len(atoms), 3))
        for k in range(3):
            if k != field_axis:
                fr[:, k] = friction
        super().__init__(atoms, timestep, temperature_K=temperature_K,
                         friction=fr, fixcm=False, rng=rng)


# ---------------------------------------------------------------------------
# observables
# ---------------------------------------------------------------------------
class ColorFluxMonitor:
    """Accumulates the color flux every step and emits block averages.

    Blocks (default 0.5 ps) are long compared with the velocity correlation time
    and short compared with the run, so the scatter of the block means is an
    honest error bar on <Jc>.
    """

    def __init__(self, atoms, dyn, colors, field_axis, profile_axis, nbins,
                 dt_ps, block, out, traj_every=0):
        self.atoms, self.dyn = atoms, dyn
        self.c = colors
        self.fax, self.pax = field_axis, profile_axis
        self.nbins = nbins
        self.dt_ps, self.block = dt_ps, block
        self.out = out
        sym = np.asarray(atoms.get_chemical_symbols())
        self.li = np.where(sym == "Li")[0]
        self.is_li = (sym == "Li")
        self.m = atoms.get_masses()
        self.V = atoms.get_volume()
        self.L = float(np.array(atoms.get_cell())[self.pax, self.pax])
        self.binw = self.L / nbins
        self.Vbin = self.V / nbins
        # total momentum is conserved along the field axis (no thermostat there
        # and sum(c_i) = 0), so that direction has N-1 DOF; the Langevin noise
        # keeps the COM alive in the two perpendicular directions.
        self.ndof_par = len(atoms) - 1
        self.ndof_perp = 2 * len(atoms)

        self.rows = []          # block averages
        self._reset_block()
        self.nblocks_done = 0
        self.istep_off = dyn.get_number_of_steps()
        self.step0 = 0
        self.last_istep = -1

        # unwrapped Li positions, for the drift-corrected self-MSD -> sigma_dilute
        self.traj_every = traj_every
        self.ref = atoms.get_positions()[self.li].copy()
        self.prev = self.ref.copy()
        self.unwrapped = self.ref.copy()
        self.traj, self.traj_t = [], []
        self.cell = np.array(atoms.get_cell())

    def _reset_block(self):
        n = self.nbins
        self.acc = dict(n=0, jc=0.0, jc_li=0.0, jc_fr=0.0, tpar=0.0, tperp=0.0,
                        jck=np.zeros(n), nk=np.zeros(n))

    def _unwrap(self):
        cell = np.array(self.atoms.get_cell())
        pos = self.atoms.get_positions()[self.li]
        d = pos - self.prev
        frac = d @ np.linalg.inv(cell)
        frac -= np.round(frac)
        self.unwrapped += frac @ cell
        self.prev = pos

    def __call__(self):
        self._unwrap()
        istep = self.step0 + self.dyn.get_number_of_steps() - self.istep_off
        if istep <= self.last_istep:
            return
        self.last_istep = istep

        v = self.atoms.get_velocities() * (units.fs * 1000.0)     # A/ps
        vf = v[:, self.fax]
        cv = self.c * vf

        a = self.acc
        a["n"] += 1
        a["jc"] += cv.sum() / self.V
        a["jc_li"] += cv[self.is_li].sum() / self.V
        a["jc_fr"] += cv[~self.is_li].sum() / self.V

        # kinetic temperatures, field axis vs the two thermostatted axes.
        # eV -> K: 1 eV = 1/units.kB K
        ekin_par = 0.5 * (self.m * v[:, self.fax] ** 2).sum() / (units.fs * 1000.0) ** 2
        ekin_perp = 0.5 * (self.m * (v ** 2).sum(axis=1)).sum() / (units.fs * 1000.0) ** 2 - ekin_par
        a["tpar"] += 2 * ekin_par / (self.ndof_par * units.kB)
        a["tperp"] += 2 * ekin_perp / (self.ndof_perp * units.kB)

        # slab decomposition, membership from the CURRENT position
        x = np.mod(self.atoms.get_positions()[:, self.pax], self.L)
        k = np.minimum((x / self.binw).astype(int), self.nbins - 1)
        a["jck"] += np.bincount(k, weights=cv, minlength=self.nbins) / self.Vbin
        a["nk"] += np.bincount(k[self.is_li], minlength=self.nbins) / self.Vbin

        if self.traj_every and istep % self.traj_every == 0:
            self.traj.append(self.unwrapped.astype(np.float32).copy())
            self.traj_t.append(istep * self.dt_ps)

        if a["n"] >= self.block:
            n = a["n"]
            row = [istep * self.dt_ps, a["jc"] / n, a["jc_li"] / n, a["jc_fr"] / n,
                   a["tpar"] / n, a["tperp"] / n]
            row += list(a["jck"] / n) + list(a["nk"] / n)
            self.rows.append(row)
            self.nblocks_done += 1
            self._reset_block()

    # -- persistence ---------------------------------------------------------
    def header(self):
        h = ["time_ps", "Jc", "Jc_Li", "Jc_frame", "T_par", "T_perp"]
        h += ["Jc_bin%02d" % i for i in range(self.nbins)]
        h += ["nLi_bin%02d" % i for i in range(self.nbins)]
        return h

    def state(self):
        return {"rows": np.array(self.rows) if self.rows else np.zeros((0, 6 + 2 * self.nbins)),
                "step0": self.step0 + self.dyn.get_number_of_steps() - self.istep_off,
                "ref": self.ref, "prev": self.prev, "unwrapped": self.unwrapped,
                "traj": (np.array(self.traj, dtype=np.float32) if self.traj
                         else np.zeros((0, len(self.li), 3), dtype=np.float32)),
                "traj_t": np.array(self.traj_t, dtype=np.float64)}

    def load_state(self, z):
        self.rows = [list(r) for r in z["rows"]]
        self.step0 = int(z["step0"])
        self.ref = z["ref"]; self.prev = z["prev"]; self.unwrapped = z["unwrapped"]
        if "traj" in z.files and len(z["traj"]):
            self.traj = [a for a in z["traj"]]
            self.traj_t = [float(t) for t in z["traj_t"]]
        self.last_istep = self.step0
        print("resumed: %d production steps, %d blocks" % (self.step0, len(self.rows)))

    def save(self, meta):
        arr = np.array(self.rows)
        np.savetxt(self.out + ".csv", arr, delimiter=",",
                   header=",".join(self.header()), comments="")
        np.savez_compressed(
            self.out + "_nemd.npz", rows=arr, names=np.array(self.header()),
            colors=self.c, li_idx=self.li, cell=self.cell, volume=self.V,
            traj=(np.array(self.traj, dtype=np.float32) if self.traj
                  else np.zeros((0, len(self.li), 3), dtype=np.float32)),
            traj_t=np.array(self.traj_t, dtype=np.float64), **meta)
        if len(arr) >= 2:
            self.report(arr, meta)
        print("wrote %s.csv / %s_nemd.npz" % (self.out, self.out))

    def report(self, arr, meta):
        f = float(meta["field"])
        jc = arr[:, 1]
        # discard the first 20% of blocks as transient
        s = max(1, int(0.2 * len(jc)))
        jc = jc[s:]
        mean = jc.mean()
        err = jc.std(ddof=1) / np.sqrt(len(jc))
        print("== steady-state color flux (last %d of %d blocks) ==" % (len(jc), len(arr)))
        print("   <Jc> = %.4e +- %.1e  [1/(A^2 ps)]" % (mean, err))
        if f != 0.0:
            print("   sigma = %.4e +- %.1e S/cm" % (SIGMA_CONV * mean / f,
                                                    SIGMA_CONV * err / abs(f)))
        print("   T_par = %.1f K   T_perp = %.1f K  (target %.0f K)"
              % (arr[s:, 4].mean(), arr[s:, 5].mean(), float(meta["temp"])))


# ---------------------------------------------------------------------------
def build_atoms(args):
    if args.data.endswith((".data", ".lmp")):
        atoms = read_lammps_data(args.data, atom_style=args.atom_style)
        atoms.set_pbc([True, True, True])
        if args.symbols:
            raise SystemExit("pass an extxyz instead; symbol remapping lives in diffusion_md.py")
    else:
        atoms = ase_read(args.data)
        atoms.set_pbc([True, True, True])
    if args.repeat:
        rep = tuple(int(x) for x in args.repeat.split())
        n0 = len(atoms)
        atoms = atoms.repeat(rep)
        atoms.set_pbc([True, True, True])
        print("replicated %s: %d -> %d atoms" % (rep, n0, len(atoms)))
    return atoms


def main():
    ap = argparse.ArgumentParser(description="chemical color-diffusion NEMD")
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="best_model.pth")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--atom_style", default="atomic")
    ap.add_argument("--symbols", default="")
    ap.add_argument("--repeat", default="", help="'1 1 3' to replicate before the run")
    ap.add_argument("--temp", type=float, default=700.0)
    ap.add_argument("--dt", type=float, default=2.0, help="timestep [fs]")
    ap.add_argument("--nsteps", type=int, default=50000)
    ap.add_argument("--equil", type=int, default=5000)
    ap.add_argument("--friction", type=float, default=0.01,
                    help="Langevin friction [1/fs], applied ONLY perpendicular to the field")
    ap.add_argument("--field", type=float, default=0.02,
                    help="color field Fe [eV/A]; 0.0 = EMD baseline for sigma_dilute. "
                         "0.02 is the value validated as linear response in npj "
                         "Comput. Mater. 9, 48 (2023)")
    ap.add_argument("--field-axis", default="z", choices=list(AXIS),
                    help="(A) parallel to the GB plane: y or z. (B) perpendicular: x")
    ap.add_argument("--profile-axis", default="x", choices=list(AXIS),
                    help="axis the slabs are binned along (the GB normal)")
    ap.add_argument("--nbins", type=int, default=40)
    ap.add_argument("--block", type=int, default=250, help="steps per block average")
    ap.add_argument("--traj_every", type=int, default=50,
                    help="store unwrapped Li positions every N steps (0 = off)")
    ap.add_argument("--bond_cut", type=float, default=2.8)
    ap.add_argument("--search_cut", type=float, default=3.6)
    ap.add_argument("--allow-charged", action="store_true",
                    help="proceed even if sum(c_i) != 0 (residual is removed as a "
                         "mass-weighted counter-force, but the drive is then not "
                         "exactly the intended color field)")
    ap.add_argument("--ckpt_every", type=int, default=5000)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="nemd")
    args = ap.parse_args()

    fax, pax = AXIS[args.field_axis], AXIS[args.profile_axis]
    atoms = build_atoms(args)
    print("composition:", atoms.get_chemical_formula(),
          " cell:", np.round(atoms.get_cell().lengths(), 3))

    colors, tot = assign_color_charges(atoms, bond_cut=args.bond_cut,
                                      search_cut=args.search_cut)
    if abs(tot) > 1e-6 and not args.allow_charged:
        raise SystemExit("sum(c_i) = %+.4f != 0.  Fix the composition or pass "
                         "--allow-charged." % tot)

    from mattersim.forcefield import MatterSimCalculator
    if args.model.lower() in ("", "pretrained", "default", "none"):
        base_calc = MatterSimCalculator(device=args.device)
    else:
        base_calc = MatterSimCalculator(load_path=args.model, device=args.device)
    atoms.calc = base_calc

    ck_xyz, ck_npz = args.out + "_ckpt.extxyz", args.out + "_ckpt.npz"
    resumed = args.resume and os.path.exists(ck_xyz) and os.path.exists(ck_npz)
    if resumed:
        prev = ase_read(ck_xyz)
        atoms.set_positions(prev.get_positions())
        atoms.set_momenta(prev.get_momenta())
        print("resumed geometry from", ck_xyz)
    else:
        rng0 = np.random.RandomState(args.seed)
        MaxwellBoltzmannDistribution(atoms, temperature_K=args.temp, force_temp=True, rng=rng0)
        Stationary(atoms)

    dt = args.dt * units.fs
    lrng = np.random.RandomState(args.seed + 10007)

    if args.equil > 0 and not resumed:
        # equilibrate with a normal isotropic Langevin and NO field
        print("equilibrating %d steps (no field, isotropic thermostat)..." % args.equil)
        eq = Langevin(atoms, dt, temperature_K=args.temp,
                      friction=args.friction / units.fs, rng=lrng)
        eq.run(args.equil)

    atoms.calc = ColorFieldCalculator(base_calc, colors, fax, args.field,
                                      atoms.get_masses())
    dyn = ColorLangevin(atoms, dt, temperature_K=args.temp,
                        friction=args.friction / units.fs,
                        field_axis=fax, rng=lrng)
    print("field Fe = %+.4f eV/A along %s ; slabs binned along %s (%d bins) ; "
          "thermostat off along %s" % (args.field, args.field_axis,
                                       args.profile_axis, args.nbins, args.field_axis))

    mon = ColorFluxMonitor(atoms, dyn, colors, fax, pax, args.nbins,
                           dt_ps=args.dt / 1000.0, block=args.block, out=args.out,
                           traj_every=args.traj_every)
    if resumed:
        mon.load_state(np.load(ck_npz, allow_pickle=False))
    dyn.attach(mon, interval=1)

    meta = dict(field=args.field, field_axis=fax, profile_axis=pax,
                temp=args.temp, dt_ps=args.dt / 1000.0, nbins=args.nbins,
                block=args.block, seed=args.seed,
                model=args.model, data=args.data, repeat=args.repeat)

    def checkpoint():
        from ase.io import write as ase_write
        st = mon.state()
        np.savez(ck_npz + ".tmp.npz", **st)
        os.replace(ck_npz + ".tmp.npz", ck_npz)
        ase_write(ck_xyz + ".tmp", atoms, format="extxyz")
        os.replace(ck_xyz + ".tmp", ck_xyz)
        mon.save(meta)

    todo = args.nsteps - mon.step0
    if todo <= 0:
        print("already at %d/%d production steps" % (mon.step0, args.nsteps))
        mon.save(meta); return
    print("production %d steps at %.0f K (seed %d, %d already done)..."
          % (todo, args.temp, args.seed, mon.step0))
    if args.ckpt_every > 0:
        done = 0
        while done < todo:
            chunk = min(args.ckpt_every, todo - done)
            dyn.run(chunk)
            done += chunk
            checkpoint()
            print("  [ckpt] %d / %d production steps done" % (mon.step0 + done, args.nsteps))
    else:
        dyn.run(todo)
    mon.save(meta)


if __name__ == "__main__":
    main()

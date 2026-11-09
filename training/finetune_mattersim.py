# -*- coding: utf-8 -*-
import argparse
import json
import os
import pickle as pkl
import random

import numpy as np
import torch
import torch.distributed

# wandb is optional: every use below is behind --wandb, but a bare "import wandb"
# fails on an offline cluster where it is not installed.
try:
    import wandb
except ImportError:                                   # pragma: no cover
    wandb = None

from ase.data import atomic_numbers as _ASE_Z
from ase.units import GPa

# ASE >=3.24 moved full_3x3_to_voigt_6_stress from ase.constraints to ase.stress,
# but mattersim 1.2.0 still imports the old path (potential.py:18). Same shim as
# extract_embeddings.py / eval_v3.py. Without it the import below dies with
# "cannot import name 'full_3x3_to_voigt_6_stress'".
import ase.constraints as _asec
if not hasattr(_asec, "full_3x3_to_voigt_6_stress"):
    from ase.stress import (full_3x3_to_voigt_6_stress as _f3,
                            voigt_6_to_full_3x3_stress as _v6)
    _asec.full_3x3_to_voigt_6_stress = _f3
    _asec.voigt_6_to_full_3x3_stress = _v6

from mattersim.datasets.utils.build import build_dataloader
from mattersim.forcefield.m3gnet.scaling import AtomScaling
from mattersim.forcefield.potential import Potential
from mattersim.utils.atoms_utils import AtomsAdaptor
from mattersim.utils.logger_utils import get_logger

logger = get_logger()
local_rank = int(os.environ["LOCAL_RANK"])


def load_energy_gauge(path):
    """Per-element constants that map the extxyz energies onto the checkpoint's
    energy zero point. See scripts/fit_energy_gauge.py for how they are measured
    and why they are needed: matersim_model.pth was trained on raw QE total
    energies while the v3 extxyz carries cohesive ones, so the two differ by
    ~274 eV/atom of pure gauge. Training across that offset is what wrecks the
    forces -- the energy term can never be satisfied and never stops pulling."""
    with open(path) as f:
        gauge = json.load(f)
    return dict(zip(gauge["elements"], gauge["c_eV"]))


def restore_best_metric(potential, save_path, metric_name):
    """DDP path only: Potential.__init__ hardcodes best_metric=10000 and
    best_metric_epoch=0, and neither is written into the checkpoint. Resuming
    without restoring them has two consequences, both silent:

      * early stopping compares `epoch > best_metric_epoch + patience`, so a
        resume at epoch 11 with the default patience of 10 stops instantly,
      * best_metric=10000 means the first epoch after the resume overwrites
        best_model.pth even if it is worse than the pre-resume best.

    best_model.pth already carries both numbers, so read them back from it."""
    best_path = os.path.join(save_path, "best_model.pth")
    if not os.path.exists(best_path):
        return None
    ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    key = ["loss", "MAE_energy", "MAE_force", "MAE_stress"][
        ["val_loss", "val_mae_e", "val_mae_f", "val_mae_s"].index(metric_name)]
    potential.best_metric = ckpt["validation_metrics"][key]
    potential.best_metric_epoch = ckpt["last_epoch"]
    return potential.best_metric, potential.best_metric_epoch


def apply_energy_gauge(atoms_list, energies, gauge):
    """energies[i] += sum_i c[element]. Forces and stresses are untouched: a
    per-element constant has zero derivative."""
    shifted = []
    for atoms, energy in zip(atoms_list, energies):
        shifted.append(energy + sum(gauge[s] for s in atoms.get_chemical_symbols()))
    return shifted


def build_force_wtab(spec, device):
    """'Li:3,P:1,S:1' -> a lookup tensor indexed by atomic number Z holding the
    per-element force-loss weight (1.0 for anything unlisted). Returns None when
    spec is empty so the stock (unweighted) path is used unchanged."""
    if not spec:
        return None
    wtab = torch.ones(max(_ASE_Z.values()) + 1, device=device)
    for tok in spec.split(","):
        sym, val = tok.split(":")
        wtab[_ASE_Z[sym.strip()]] = float(val)
    return wtab


def make_patched_loss_calc():
    """Drop-in replacement for Potential.loss_calc adding two anti-forgetting
    levers, both off by default and read off the Potential instance:

      (1) element-/|F|-weighted force loss (self._force_wtab, self._force_mag_alpha):
          the stock loss averages over every atom, so the small-force Li atoms
          (<|F|>~0.4) are drowned by the high-force P/S. A per-atom weighted mean
          lets Li matter again. Weights enter as a *weighted mean*, so uniform
          weights reproduce the plain mean; the reported MAE(f) is weighted the
          same way, so --metric_name val_mae_f early-stops on the balanced error.

      (2) L2-SP anchor (self._l2sp_lambda, self._l2sp_ref): adds
          lambda*||theta - theta_pretrained||^2 to the TRAIN loss (only), so the
          fine-tune cannot drift far from the generic checkpoint and keep the
          regions the generic already gets right (high-|F| tail, Li).

    When neither lever is enabled the force/energy/stress terms are computed
    bit-for-bit like the stock loss_calc."""
    def loss_calc(self, graph_batch, result, loss, include_energy,
                  include_forces, include_stresses, loss_f=1.0, loss_s=0.1):
        e_mae = 0.0
        f_mae = 0.0
        s_mae = 0.0
        loss_ = torch.tensor(0.0, device=self.device, requires_grad=True)
        wtab = getattr(self, "_force_wtab", None)
        alpha = getattr(self, "_force_mag_alpha", 0.0)
        weighted = (wtab is not None) or (alpha > 0.0)

        if include_energy:
            e_gt = graph_batch.energy / graph_batch.num_atoms
            e_pred = result["total_energy"] / graph_batch.num_atoms
            loss_ = loss_ + loss(e_pred, e_gt)
            e_mae = torch.nn.L1Loss()(e_pred, e_gt)
        if include_forces:
            f_gt = graph_batch.forces
            f_pred = result["forces"]
            if not weighted:
                loss_ = loss_ + loss(f_pred, f_gt) * loss_f
                f_mae = torch.nn.L1Loss()(f_pred, f_gt)
            else:
                z = graph_batch.atom_attr.squeeze(1).long()   # [Ntot] atomic numbers
                w = torch.ones_like(z, dtype=f_pred.dtype)
                if wtab is not None:
                    w = wtab[z]
                if alpha > 0.0:
                    w = w * (1.0 + alpha * f_gt.detach().norm(dim=1))
                w3 = w.unsqueeze(1)                            # broadcast over xyz
                red0 = loss.reduction
                loss.reduction = "none"
                base = loss(f_pred, f_gt)                      # [Ntot,3]
                loss.reduction = red0
                fdenom = w3.expand_as(base).sum()
                loss_ = loss_ + (w3 * base).sum() / fdenom * loss_f
                f_mae = (w3 * (f_pred - f_gt).abs()).sum() / fdenom
        if include_stresses:
            s_gt = graph_batch.stress
            s_pred = result["stresses"]
            loss_ = loss_ + loss(s_pred, s_gt) * loss_s
            s_mae = torch.nn.L1Loss()(s_pred, s_gt)

        # L2-SP: penalise drift from the pretrained weights. TRAIN only, so the
        # validation loss/metric stays a clean data term for early stopping.
        lam = getattr(self, "_l2sp_lambda", 0.0)
        ref = getattr(self, "_l2sp_ref", None)
        if lam > 0.0 and ref is not None and self.model.training:
            sp = torch.zeros((), device=self.device)
            for p, p0 in zip(self.model.parameters(), ref):
                if p.requires_grad:
                    sp = sp + ((p - p0) ** 2).sum()
            loss_ = loss_ + lam * sp
        return loss_, e_mae, f_mae, s_mae
    return loss_calc


# Install the patch on the class so every Potential built below uses it. The
# knobs default to off (attributes set in main), so this is a no-op unless the
# corresponding CLI flags are given.
Potential.loss_calc = make_patched_loss_calc()


def main(args):
    if args.device == "cuda":
        torch.distributed.init_process_group(backend="nccl")
    else:
        torch.distributed.init_process_group(backend="gloo")
    args_dict = vars(args)
    if args.wandb and wandb is None:
        raise SystemExit("--wandb was given but wandb is not installed")
    if args.wandb and local_rank == 0:
        wandb_api_key = (
            args.wandb_api_key
            if args.wandb_api_key is not None
            else os.getenv("WANDB_API_KEY")
        )
        wandb.login(key=wandb_api_key)
        wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            config=args,
            # id=args.run_name,
            # resume="allow",
        )

    if args.wandb:
        args_dict["wandb"] = wandb

    torch.distributed.barrier()

    # set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "cuda":
        torch.cuda.set_device(local_rank)

    if args.train_data_path.endswith(".pkl"):
        with open(args.train_data_path, "rb") as f:
            atoms_train = pkl.load(f)
    else:
        atoms_train = AtomsAdaptor.from_file(filename=args.train_data_path)
    energies = []
    forces = [] if args.include_forces else None
    stresses = [] if args.include_stresses else None
    logger.info("Processing training datasets...")
    for atoms in atoms_train:
        energies.append(atoms.get_potential_energy())
        if args.include_forces:
            forces.append(atoms.get_forces())
        if args.include_stresses:
            stresses.append(atoms.get_stress(voigt=False) / GPa)  # convert to GPa

    gauge = load_energy_gauge(args.energy_gauge) if args.energy_gauge else None
    if gauge is not None:
        nat = sum(len(a) for a in atoms_train)
        before = sum(e for e in energies) / nat
        energies = apply_energy_gauge(atoms_train, energies, gauge)
        after = sum(e for e in energies) / nat
        if local_rank == 0:
            logger.info(
                "energy gauge %s applied to train: <E>/atom %.4f -> %.4f eV"
                % (args.energy_gauge, before, after))

    dataloader = build_dataloader(
        atoms_train,
        energies,
        forces,
        stresses,
        shuffle=True,
        pin_memory=(args.device == "cuda"),
        is_distributed=True,
        **args_dict,
    )

    device = args.device
    # build energy normalization module
    if args.re_normalize:
        scale = AtomScaling(
            atoms=atoms_train,
            total_energy=energies,
            forces=forces,
            verbose=True,
            **args_dict,
        ).to(device)

    if args.valid_data_path is not None:
        if args.valid_data_path.endswith(".pkl"):
            with open(args.valid_data_path, "rb") as f:
                atoms_val = pkl.load(f)
        else:
            atoms_val = AtomsAdaptor.from_file(filename=args.valid_data_path)
        energies = []
        forces = [] if args.include_forces else None
        stresses = [] if args.include_stresses else None
        logger.info("Processing validation datasets...")
        for atoms in atoms_val:
            energies.append(atoms.get_potential_energy())
            if args.include_forces:
                forces.append(atoms.get_forces())
            if args.include_stresses:
                stresses.append(atoms.get_stress(voigt=False) / GPa)  # convert to GPa
        if gauge is not None:
            energies = apply_energy_gauge(atoms_val, energies, gauge)
        val_dataloader = build_dataloader(
            atoms_val,
            energies,
            forces,
            stresses,
            pin_memory=(args.device == "cuda"),
            is_distributed=True,
            **args_dict,
        )
    else:
        val_dataloader = None

    # Resume: last_model.pth is written every epoch by train_model, so at most
    # one epoch of work is ever lost. load_training_state=True brings back the
    # optimizer, the scheduler, the EMA and last_epoch, and train_model picks up
    # at range(last_epoch + 1, epochs).
    resume_path = os.path.join(args.save_path, "last_model.pth")
    resuming = args.resume and os.path.exists(resume_path)
    if args.resume and not resuming and local_rank == 0:
        logger.info("--resume given but %s does not exist: starting from %s"
                    % (resume_path, args.load_model_path))
    potential = Potential.from_checkpoint(
        load_path=resume_path if resuming else args.load_model_path,
        load_training_state=resuming,
        **args_dict,
    )
    if resuming:
        best = restore_best_metric(potential, args.save_path, args.metric_name)
        if local_rank == 0:
            logger.info(
                "resumed from %s at epoch %d -> continuing at epoch %d of %d "
                "(lr comes from the restored scheduler, --lr %g is ignored)"
                % (resume_path, potential.last_epoch, potential.last_epoch + 1,
                   args.epochs, args.lr))
            if best is not None:
                logger.info("restored best %s = %.6f from epoch %d"
                            % (args.metric_name, best[0], best[1]))
            else:
                logger.warning(
                    "no best_model.pth in %s: early stopping and best-model "
                    "tracking restart from scratch" % args.save_path)

    if args.re_normalize:
        potential.model.set_normalizer(scale)

    # Anti-forgetting / balanced-loss knobs consumed by the patched loss_calc.
    # The L2-SP reference is snapshotted here, BEFORE the DDP wrap, so it is the
    # untouched pretrained weight of every parameter in model.parameters() order
    # (DDP preserves that order, so the zip in loss_calc stays aligned).
    potential._force_wtab = build_force_wtab(args.force_element_weights, device)
    potential._force_mag_alpha = args.force_mag_weight_alpha
    potential._l2sp_lambda = args.l2sp_lambda
    if args.l2sp_lambda > 0.0:
        # Anchor to a FIXED reference model reloaded from disk, so theta0 is
        # identical on the first launch and on every resume. Snapshotting the
        # live potential.model would instead move the anchor to wherever
        # training had already drifted on resume, silently defeating L2-SP.
        # Defaults to the start model; pass --l2sp_ref_model pretrained (or a
        # path) to hold onto the generic force field instead.
        ref_src = args.l2sp_ref_model or args.load_model_path
        anchor = Potential.from_checkpoint(
            load_path=ref_src, load_training_state=False, **args_dict)
        potential._l2sp_ref = [p.detach().clone().to(device)
                               for p in anchor.model.parameters()]
        del anchor
        if local_rank == 0:
            logger.info("L2-SP anchor loaded from %s (%d param tensors)"
                        % (ref_src, len(potential._l2sp_ref)))
    else:
        potential._l2sp_ref = None
    if local_rank == 0:
        logger.info(
            "anti-forgetting: l2sp_lambda=%g, force_element_weights=%r, "
            "force_mag_weight_alpha=%g"
            % (args.l2sp_lambda, args.force_element_weights or None,
               args.force_mag_weight_alpha))

    if args.device == "cuda":
        potential.model = torch.nn.parallel.DistributedDataParallel(potential.model)
    torch.distributed.barrier()

    # The loss used to be hardcoded as HuberLoss(delta=0.01). With delta that
    # small every force error above 0.01 eV/A sits in the linear branch, so a P
    # atom carrying 5 eV/A contributes exactly the same gradient magnitude as a
    # Li atom at 0.05 eV/A. Measured consequence: the fine-tune improved Li/S
    # and the low-force bins while the P sublattice got 46 % WORSE and the
    # PS4 stretching modes softened by 13 % against Raman. Raise delta (or use
    # MSE) so large errors actually dominate the gradient.
    if args.loss == "huber":
        loss_fn = torch.nn.HuberLoss(delta=args.huber_delta)
    elif args.loss == "mse":
        loss_fn = torch.nn.MSELoss()
    else:
        loss_fn = torch.nn.L1Loss()
    if local_rank == 0:
        logger.info(
            "loss = %s%s, force_loss_ratio = %s, stress_loss_ratio = %s"
            % (args.loss,
               "(delta=%g)" % args.huber_delta if args.loss == "huber" else "",
               args.force_loss_ratio, args.stress_loss_ratio))
        if args.loss == "huber" and args.huber_delta <= 0.02:
            logger.warning(
                "huber_delta=%g is smaller than almost every force error in this "
                "data (P atoms average 1.0 eV/A): the high-force atoms will not "
                "be prioritised. Consider 0.1-0.3 or --loss mse."
                % args.huber_delta)

    # args_dict carries the --loss string ("huber"/"mse"/"l1"), which collides with
    # train_model()'s own `loss` parameter (the nn.Module). Drop it here; loss_fn
    # built above is what train_model must receive.
    train_kwargs = {k: v for k, v in args_dict.items() if k != "loss"}
    potential.train_model(
        dataloader,
        val_dataloader,
        loss=loss_fn,
        is_distributed=True,
        **train_kwargs,
    )

    if local_rank == 0 and args.save_checkpoint and args.wandb:
        wandb.save(os.path.join(args.save_path, "best_model.pth"))


if __name__ == "__main__":
    # Some important arguments
    parser = argparse.ArgumentParser()

    # path parameters
    parser.add_argument(
        "--run_name", type=str, default="example", help="name of the run"
    )
    parser.add_argument(
        "--train_data_path", type=str, default="./sample.xyz", help="train data path"
    )
    parser.add_argument(
        "--valid_data_path", type=str, default=None, help="valid data path"
    )
    parser.add_argument(
        "--load_model_path",
        type=str,
        default="mattersim-v1.0.0-1m",
        help="path to load the model",
    )
    parser.add_argument(
        "--save_path", type=str, default="./results", help="path to save the model"
    )
    parser.add_argument(
        "--save_checkpoint",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--ckpt_interval",
        type=int,
        default=10,
        help="save checkpoint every ckpt_interval epochs",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--energy_gauge", type=str, default=None,
        help="JSON from fit_energy_gauge.py. Adds the per-element constant to "
             "every training/validation energy so the labels sit in the same "
             "energy zero point as --load_model_path. Without it the energy "
             "loss on the v3 extxyz starts at ~274 eV/atom and never comes "
             "down, and the gradient it injects degrades the forces.")

    # model parameters
    parser.add_argument("--cutoff", type=float, default=5.0, help="cutoff radius")
    parser.add_argument(
        "--threebody_cutoff",
        type=float,
        default=4.0,
        help="cutoff radius for three-body term, which should be smaller than cutoff (two-body)",  # noqa: E501
    )

    # training parameters
    parser.add_argument("--epochs", type=int, default=1000, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument(
        "--step_size",
        type=int,
        default=10,
        help="step epoch for learning rate scheduler",
    )
    parser.add_argument(
        "--include_forces",
        type=bool,
        default=True,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--include_stresses",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--loss", type=str, default="huber", choices=["huber", "mse", "l1"],
        help="training loss. huber with a small delta behaves like L1 and gives "
             "every atom the same gradient magnitude regardless of its error")
    parser.add_argument(
        "--huber_delta", type=float, default=0.01,
        help="Huber transition point [eV/A for forces]. 0.01 = the historical "
             "hardcoded value (effectively L1); 0.1-0.3 lets the high-force P "
             "atoms dominate the gradient as they should")
    parser.add_argument("--force_loss_ratio", type=float, default=1.0)
    parser.add_argument("--stress_loss_ratio", type=float, default=0.1)
    parser.add_argument(
        "--l2sp_lambda", type=float, default=0.0,
        help="L2-SP anchor strength. Adds lambda*||theta - theta_pretrained||^2 "
             "to the TRAIN loss so the fine-tune cannot drift far from the "
             "generic checkpoint (anti-forgetting: protects the high-|F| tail "
             "and Li that free fine-tuning regresses). 0 = off. Try 1e-3..1e-1.")
    parser.add_argument(
        "--force_element_weights", type=str, default="",
        help="Per-element force-loss weights, e.g. 'Li:3,P:1,S:1,Br:1,Cl:1'. "
             "Unlisted elements get 1. Counters the small-force Li being drowned "
             "by high-force P/S. Applied as a weighted mean (uniform weights "
             "reproduce the plain mean), and it also reweights MAE(f) so "
             "--metric_name val_mae_f early-stops on the balanced force error.")
    parser.add_argument(
        "--force_mag_weight_alpha", type=float, default=0.0,
        help="Extra per-atom force weight (1 + alpha*|F_gt|); alpha>0 up-weights "
             "high-force atoms to protect the |F|>1.5 tail. 0 = off.")
    parser.add_argument(
        "--l2sp_ref_model", type=str, default="",
        help="Checkpoint the L2-SP penalty anchors to (only used when "
             "--l2sp_lambda>0). Empty = --load_model_path (regularise toward the "
             "start point). Use 'pretrained' to hold onto the generic "
             "mattersim-v1.0.0-1M weights. Reloaded from disk every launch so "
             "resume keeps the same anchor.")
    parser.add_argument("--early_stop_patience", type=int, default=10)
    parser.add_argument(
        "--metric_name", type=str, default="val_loss",
        choices=["val_loss", "val_mae_e", "val_mae_f", "val_mae_s"],
        help="metric used for best_model.pth and for early stopping")
    parser.add_argument(
        "--resume", type=bool, default=False,
        action=argparse.BooleanOptionalAction,
        help="continue from <save_path>/last_model.pth if it exists, restoring "
             "the optimizer/scheduler/EMA/epoch counter. Falls back to "
             "--load_model_path when there is nothing to resume, so the same "
             "command works for the first launch and every restart after it.")
    parser.add_argument("--seed", type=int, default=42)

    # scaling parameters
    parser.add_argument(
        "--re_normalize",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help="re-normalize the energy and forces according to the new data",
    )
    parser.add_argument("--scale_key", type=str, default="per_species_forces_rms")
    parser.add_argument(
        "--shift_key", type=str, default="per_species_energy_mean_linear_reg"
    )
    parser.add_argument("--init_scale", type=float, default=None)
    parser.add_argument("--init_shift", type=float, default=None)
    parser.add_argument(
        "--trainable_scale",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--trainable_shift",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
    )

    # wandb parameters
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_api_key", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="wandb_test")
    args = parser.parse_args()
    main(args)

# Reactive sampling and data curation for a grain-boundary machine-learning force field

Companion code for:

> H. Nakata, S. Kadota, B. Ena, *Building a grain-boundary machine-learning force
> field by reactive sampling and data curation, and its application to
> large-scale grain-boundary transport in Li₆PS₅X (X = Cl, Br)*, submitted to
> *Computational Materials Science*.

This repository contains the ODPMD reactive-path sampler, the data-curation and
descriptor-space selection pipeline, the scripts that build and fine-tune the
**v5D** force field, the DFT-validation scripts, and the large-scale grain-boundary
(GB) transport analysis used in the paper.

* **[`USAGE_en.md`](USAGE_en.md)**: step-by-step manual (English) with the exact
  production commands.
* **[`USAGE_ja.md`](USAGE_ja.md)**: the same manual in Japanese.
* This README: overview, layout, which script does what, and which script backs
  which result in the paper.

---

## 1. Requirements

```bash
pip install -r requirements.txt
```

| package | version | note |
|---|---|---|
| `mattersim` | **1.2.0** (pinned) | must match the checkpoint used for training and for the descriptor space (1.2.5 produces incompatible embeddings) |
| `torch` | ≥ 2.0 | GPU build recommended for embedding extraction and fine-tuning; MD runs fine on CPU |
| `ase` | ≥ 3.24 | `full_3x3_to_voigt_6_stress` moved to `ase.stress`; every script that imports MatterSim carries a small compatibility shim |
| `numpy`, `scipy`, `scikit-learn`, `matplotlib` | — | curation / clustering / analysis |

DFT labelling uses Quantum ESPRESSO 6.7 (`pw.x`), PBE+D3, ultrasoft
pseudopotentials, k-mesh from `k_len = 26`. QE is not a Python dependency.

---

## 2. Repository layout

```
.
├── sampling/     # Step 1  ODPMD reactive-path sampling
├── curation/     # Step 2  physical-consistency filter + PS4-integrity audit
├── selection/    # Step 3  descriptor embedding, clustering, stratified selection
├── training/     # Step 4  DFT labelling, migration-path / active-learning data, v5D fine-tuning
├── validation/   # Step 5  NEB barriers (held-out paths), elastic constants, parity, k-convergence
├── application/  # Step 6  GB models, large-scale GB MD, transport / mechanism / NEMD / space charge
└── legacy/       # superseded phaseB-era scripts (kept for provenance, not used in the paper)
```

Every Python script has a header comment explaining what it does and why, plus a
usage line; `python <script> -h` lists all options.

### Important: halide decoration of GB models

An earlier version of this workflow built charge-neutral Li-deficient structures
by converting S into halide **without distinguishing PS₄-vertex S from free S²⁻**,
which produced PS₃X / PS₂X₂ units that do not exist in Li₆PS₅X. Everything used
in the paper is built so that PS₄ is never touched:

* GB models: `application/gb_build_uniform.py` (halides only on free-anion sites)
  and `application/gb_li2s_defect.py` (removes free S²⁻ + 2 Li).
* Li-deficient bulk cells: `training/build_lidef_bulk.py`.
* NEB on GB cells: `validation/neb_barrier.py --comp_free_s`.
* Training frames whose halides sit on a PS₄ vertex are removed
  (`curation/audit_ps4.py` → `training/assemble_v5_train.py`).

`application/gb_build.py --neutral` still has the old behaviour and must not be
used. The file is kept because `gb_build_uniform.py` imports its reader.

---

## 3. Scripts

### Step 1: ODPMD reactive-path sampling (`sampling/`)

| script | purpose |
|---|---|
| `ODPMD.py` | the sampler: thermostatted MD with a moving harmonic bias along the one-dimensional projected reaction coordinate; writes frames densely while P–S bonds break (`PS4.log`). Reads `in.test` + `ODP.colvars`. |
| `prepare_runs.py` | generates the grid of run directories over V/atom × T × pull rate from `in.test.template` and `ODP.colvars.orig`. |
| `ps4_probe.py` | PS₄ dissociation / novelty diagnostics of a trajectory. |

### Step 2: curation (`curation/`)

| script | purpose |
|---|---|
| `filter_frames.py` | overlap (Morse-inverted floor), `fmax` and robust energy-outlier tests → boolean keep-mask. |
| `audit_ps4.py` | per-frame PS₄ audit of extxyz sets (P motifs, PS₄-vertex halides, free S) → `frames.csv`; frames with `n_vertexX > 0` are dropped from the reactive base (5271 → 4074 frames). |

### Step 3: descriptor-space selection (`selection/`)

| script | purpose |
|---|---|
| `extract_embeddings.py` | 256-d MatterSim descriptor per frame. |
| `cluster_embeddings.py` | StandardScaler → whitening PCA → KMeans; frozen model `cluster_model.npz`. |
| `select_from_clusters.py` | leak-aware test split, per-cluster budget, farthest-point sampling. |
| `assign_clusters.py`, `screen_candidates.py`, `validate_new_data.py`, `update_embeddings.py`, `pool_select.py` | classify / screen / score new data against the frozen descriptor store. |
| `emb_cache.py` | shared helper (keep in the same directory). |

### Step 4: DFT labelling and v5D fine-tuning (`training/`)

| script | purpose |
|---|---|
| `gen_trainset.py` | Subset-B design pool (bulk crystals, Cl:Br and Li-vacancy grids, alternate GBs, perturbations). Its Li-deficient cells were built with the old S→X rule, so its output must go through `audit_ps4.py`. |
| `make_scf_inputs.py`, `stage_dft.py`, `scf_to_deepmd.py` | QE single-point inputs (`--klen 26`), job staging, and QE → DeePMD conversion (Ry→eV = 13.6058). |
| `build_lidef_bulk.py` | correctly built Li-deficient bulk cells Li₆₋ₓPS₅₋ₓX₁₊ₓ (free S²⁻ → X, PS₄ untouched). |
| `md_harvest.py` | NVT MD on DFT-sized GB / bulk cells that dumps full frames for DFT ("reactive MD" frames of Subset C). |
| `assemble_v5_train.py` | `v5_trainB.extxyz` = reactive base minus `n_vertexX>0` frames + clean GB/bulk NEB and MD DFT frames (train split only; NEB ×3). |
| `cut_gb_core.py`, `relax_md_cut.py`, `select_al_frames.py`, `make_relax_cut_pwin.py`, `al_to_extxyz.py` | active learning on GB-core slabs: cut a DFT-sized slab, relax + 700 K MD with candidate models while tracking PS₄ integrity, pick frames, QE relax/single-point inputs, and QE outputs → extxyz on the same cohesive-energy zero. |
| `assemble_v5C.py` | `v5D_train.extxyz` = `v5_trainB` + AL frames × 6. |
| `fit_energy_gauge.py` | per-element cohesive-energy gauge fitted against the warm-start checkpoint → `energy_gauge_v4l0.json` (the file used for the paper is included). |
| `finetune_mattersim.py` | fine-tuning driver (Huber, force:energy = 4, element-specific force weights, optional L2-SP, energy gauge, resumable checkpoints). |
| `run_v5D.sh` | **the exact v5D recipe** (gauge check → assemble → 80 epochs from v4 λ=0, lr 1e-4, Huber δ 0.2, Li:P:S:Br:Cl = 3:1:1:1:1, no stress). |
| `eval_v5C_gate.py` | checkpoint gate: DFT energy penalty of a decomposed vs intact PS₄ in a GB core, and pointwise E/F on held-out AL frames. |

### Step 5: validation against DFT (`validation/`)

| script | purpose |
|---|---|
| `neb_barrier.py` | CI-NEB Li-migration barrier (ASE + MatterSim); `--comp_free_s` compensates the vacancy only with free S²⁻ (required for healthy GB paths). |
| `neb_dft_compare.py` | MLIP vs DFT barrier on identical NEB images. |
| `eval_v5_barriers.py` | pointwise barrier evaluation of any model on the **held-out** DFT paths (GB rnd1/seg3/d2 + bulk Br6Cl6). |
| `elastic_dft.py`, `elastic_from_stress.py`, `elastic_md.py` | finite-strain elastic constants from DFT or model stresses on identical frames. |
| `parity.py`, `train_parity.py` | E/F/stress parity; force accuracy by element and force regime. |
| `kconv_analysis.py` | k-point convergence (SI Section S4). |

### Step 6: GB models and transport (`application/`)

| script | purpose |
|---|---|
| `gb_build_uniform.py` | Σ5(210) GB cell with halides on free-anion sites only: `uni` (`--enrich 2`) and `seg` (`--enrich 4`) → `gbV2_{uni,seg}.extxyz` (1448 atoms). |
| `gb_li2s_defect.py` | Li₂S-deficient GB (free S²⁻ + 2 Li removed per unit). |
| `amorphize_gb.py` | region-restricted melt-quench of the GB slabs → `gbV2_amo.extxyz`. |
| `audit_gb_ps4.py`, `check_md_struct.py` | PS₄ audit of GB structures and of MD end states (P coordination, vertex halides, short S–S / Li–Li / P–P pairs). |
| `diffusion_md.py` | NVT Li diffusion MD on the 4344-atom cell; unwrapped Li trajectories, checkpoint/resume. |
| `msd_regions.py` | region-resolved D with the region re-assigned at every time origin, plus window-free D(x). |
| `refit_arrhenius.py` | seed-averaged GB/bulk ratios and ΔEa(GB−bulk) with seed-bootstrap errors. |
| `mechanism_tier1.py` | Li cage MSD, non-Gaussian α₂, median displacement, GB vs bulk (3 seeds). |
| `coord_tier2.py` | static Li coordination / first-shell halide fraction, GB vs bulk. |
| `framework_md.py`, `tier3_analysis.py`, `tier3_fwV2.py` | 40 ps all-atom runs; PS₄ reorientation C₁(τ), halide MSD, Li MSD (framework analysis). |
| `charge_profile_tavg.py` | time-averaged formal-charge density, Li/halide density and space-charge potential φ(x). |
| `nemd_ccd.py`, `nemd_analysis.py` | colour-charge NEMD and its analysis (σ, profiles, Arrhenius, R_GB). |
| `gb_segregation.py` | single anion-exchange energies (stability of the halogen decoration, SI Section S6). |
| `lifetime.py` | P–S bond and Li-site lifetimes. |

---

## 4. Paper result → script

| result in the paper | script(s) |
|---|---|
| ODPMD sampling analysis (§3.1) | `sampling/ODPMD.py`, `ps4_probe.py` |
| physical filter, novelty and selection (§2.2–2.3, SI S2–S3) | `curation/filter_frames.py`, `selection/*` |
| training-data composition, 4074 PS₄-intact base frames (§3.2) | `curation/audit_ps4.py`, `training/assemble_v5_train.py`, `assemble_v5C.py` |
| v5D fine-tuning, energy gauge (§2.4, §3.2) | `training/run_v5D.sh`, `fit_energy_gauge.py`, `finetune_mattersim.py` |
| L2-SP sweep (SI S5) | `training/finetune_mattersim.py --l2sp_lambda` |
| force parity by element / regime | `validation/train_parity.py`, `parity.py` |
| held-out NEB barriers (25 paths) (§4) | `validation/neb_barrier.py --comp_free_s`, `eval_v5_barriers.py`, `neb_dft_compare.py` |
| elastic constants (SI Table S5) | `validation/elastic_dft.py`, `elastic_from_stress.py` |
| k-point convergence (SI S4) | `validation/kconv_analysis.py` |
| GB models uni / seg / amo (§2.4, §2.5) | `application/gb_build_uniform.py`, `amorphize_gb.py`, `audit_gb_ps4.py` |
| D_GB/D_bulk and ΔEa (§5.1) | `application/diffusion_md.py`, `msd_regions.py`, `refit_arrhenius.py` |
| mechanism indicators (§5.1, SI Fig. S4) | `application/mechanism_tier1.py`, `coord_tier2.py` |
| framework dynamics (§5.2) | `application/framework_md.py`, `tier3_fwV2.py` |
| CCD-NEMD (§5.3) | `application/nemd_ccd.py`, `nemd_analysis.py` |
| space charge (§5.4) | `application/charge_profile_tavg.py` |
| halogen-decoration stability (SI S6) | `application/gb_segregation.py` |

`USAGE_en.md` lists the production commands and parameters for each row.

---

## 5. Data availability

The curated training / validation / external-test sets, the GB template
(`test.GB.xyz`), the GB models (`gbV2_{uni,seg,amo}.extxyz`), and the fine-tuned
`v5D` checkpoint are archived separately (DOI **to be inserted on publication**).
This repository holds only the code, small input templates, and the energy gauge.

---

## 6. License and citation

License: **to be added before publication.** If you use this code, please cite
the paper above.

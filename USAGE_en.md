# User manual (English)

A step-by-step guide to running the code behind *Building a grain-boundary
machine-learning force field by reactive sampling and data curation, and its
application to large-scale grain-boundary transport in Li₆PS₅X*. The commands
below use the settings of the paper. [`README.md`](README.md) gives the overview
and the script ↔ paper-result map; [`USAGE_ja.md`](USAGE_ja.md) is the Japanese
version of this manual.

Conventions used throughout:

* `--model` takes either `pretrained` (stock MatterSim v1.0.0-1M, called
  "universal"/"generic" in the paper) or a path to a fine-tuned `.pth` (`v5D.pth`).
* The GB normal is **x**. In the GB models (Lx = 69.434 Å) the two GB planes sit
  at 0.25·Lx = 17.36 Å and 0.75·Lx = 52.07 Å. GB windows are ±3.5 Å around each
  plane; the bulk window is 30–40 Å.
* Every script prints its options with `python <script> -h`, and its header
  comment explains what it does and why.
* Batch-system wrappers are not included; the commands are what each job ran.

---

## 0. Environment

```bash
pip install -r requirements.txt     # mattersim==1.2.0 (pinned), torch, ase, numpy, scipy, scikit-learn
```

* **mattersim must be 1.2.0.** 1.2.5 changes the embedding space, so the frozen
  clusters and the selection are no longer reproducible.
* With ASE ≥ 3.24, `full_3x3_to_voigt_6_stress` moved to `ase.stress`. Every
  script that imports MatterSim has a compatibility shim, so nothing needs to be
  done.
* DFT: Quantum ESPRESSO 6.7, PBE+D3, ultrasoft pseudopotentials, ecutwfc 48.9 Ry
  / ecutrho 440.5 Ry, conv_thr 1e-6, k-mesh ⌈k_len/|aᵢ|⌉ with k_len = 26. These
  defaults are built into `training/make_scf_inputs.py`.
* Energies are stored as cohesive energies (Ry→eV = 13.6058, bohr→Å = 0.529177).
  The per-element gauge (`training/energy_gauge_v4l0.json`) maps them onto the
  MatterSim zero.

---

## 1. Reactive-path sampling with ODPMD (`sampling/`)

```bash
python sampling/prepare_runs.py --src /path/to/equilibrated_inputs --out runs \
    --vatom 15 17 25 --temp 300 600 900
bash runs/stage_model.sh /path/to/matersim_model.pth     # copies the model into every run
cd runs/v17.0_T600 && python /path/to/sampling/ODPMD.py in.test > output.txt
```

* The paper used 12 runs on the 206-atom GB cell Li₉₄P₁₆S₇₈Br₉Cl₉ over
  V/N ∈ {15, 17, 25} Å³/atom × T ∈ {300, 600, 900} K (36,490 frames).
* `PS4.log` records broken P–S bonds; frames are written densely while the
  framework is breaking.
* `python sampling/ps4_probe.py -h` gives the dissociation / novelty diagnostics.

---

## 2. Curation (`curation/`)

### 2.1 Physical-consistency filter

```bash
python curation/filter_frames.py listdata.txt --system system.json \
    --coord_rule P:S:4 --coord_rcut 2.8 --out frame_filter.npy
```

`listdata.txt` lists DeePMD system directories, one per line. Only `set.000` is
read. Do not glob the directories yourself.

### 2.2 PS₄-integrity audit

```bash
python curation/audit_ps4.py v3_train_reduce_emb.extxyz --out audit_ps4/frames.csv
```

For each frame it records P motifs, PS₄-vertex halides (`n_vertexX`) and free S.
Frames with `n_vertexX > 0` are artefacts of an old construction rule and are
removed in step 4.3 (5271 → 4074 frames).

---

## 3. Descriptor-space selection (`selection/`)

```bash
python selection/extract_embeddings.py listdata.txt --ckpt matersim_model.pth
python selection/cluster_embeddings.py --emb embeddings.npy --meta emb_meta.npy
python selection/select_from_clusters.py listdata.txt --nsel 4000 --alloc prop
```

Later data are judged against the **frozen** model `cluster_model.npz`:

```bash
python selection/screen_candidates.py -h     # rank QE candidates before spending DFT
python selection/assign_clusters.py -h       # assign new frames to clusters or flag them "novel"
python selection/update_embeddings.py -h     # append to the cumulative descriptor store
```

Keep `emb_cache.py` in the same directory as these scripts.

---

## 4. DFT data and the v5D force field (`training/`)

### 4.1 DFT single points

```bash
python training/make_scf_inputs.py --traj frames.extxyz --out dft_frames --klen 26 --stress \
    --pseudo_dir /path/to/pseudo
# run pw.x on dft_frames/*.in, then:
python training/scf_to_deepmd.py -h        # QE -> DeePMD (use --with_stress for virials)
```

Subset A/B frames carry stress. Migration-path and AL frames do not (`--stress`
omitted).

### 4.2 Healthy structures for migration-path data (Subset C)

Li-deficient bulk cells (free S²⁻ → halide, PS₄ untouched). Run this from the
directory that holds `v3_train_reduce_emb.extxyz`:

```bash
mkdir -p audit_ps4/bulkdef
python training/build_lidef_bulk.py          # writes audit_ps4/bulkdef/{stoich,lidef}_*.extxyz
```

DFT-sized GB cells (260 atoms, Σ5(210)) with intact PS₄. These commands
reproduce the paper's structures exactly:

```bash
X=l6ps5cl_210.xyz
python application/gb_build_uniform.py --xyz $X --repeat 1,1,1 --enrich 1 --seed 0 --out gbS_rnd0.extxyz   # random
python application/gb_build_uniform.py --xyz $X --repeat 1,1,1 --enrich 1 --seed 1 --out gbS_rnd1.extxyz
python application/gb_build_uniform.py --xyz $X --repeat 1,1,1 --enrich 4 --lam 3 --seed 2 --out gbS_seg2.extxyz   # GB-biased
python application/gb_build_uniform.py --xyz $X --repeat 1,1,1 --enrich 4 --lam 3 --seed 3 --out gbS_seg3.extxyz
# Li2S-deficient GBs (2 or 4 Li2S units removed per GB plane; the NEB hop Li 0,50,43,63 are protected)
python application/gb_li2s_defect.py --data gbS_seg2.extxyz --n_per_gb 2 --protect 0,50,43,63 --out gbS_d2.extxyz
python application/gb_li2s_defect.py --data gbS_seg2.extxyz --n_per_gb 4 --protect 0,50,43,63 --out gbS_d4.extxyz
```

Reactive-MD frames and NEB images for DFT:

```bash
python training/md_harvest.py --data gbS_rnd0.extxyz --model pretrained --temp 900 \
    --nsteps 10000 --equil 2000 --dump_every 500 --out mdH_rnd0_pre_900K_s0     # 600 and 900 K were used
python validation/neb_barrier.py --data gbS_rnd0.extxyz --list_hops --gb-axis x --gb-pos 0.25,0.75
python validation/neb_barrier.py --data gbS_rnd0.extxyz --model v5D.pth --hop 0 50 \
    --images 7 --climb --k 0.1 --relax_fmax 0.05 --comp_free_s --out nebS_rnd0_0_50
python training/make_scf_inputs.py --traj nebS_rnd0_0_50_path.extxyz --out dft_nebS_rnd0_0_50 \
    --klen 26 --min_dist 0 --fmax_max 1e9
```

**Always pass `--comp_free_s` on GB cells.** Without it, the vacancy-compensating
S may be a PS₄ vertex (the old behaviour).

The train/test split is by structure, never by frame. GB test structures are
rnd1, seg3 and d2. The bulk test composition is Br6Cl6Li44P8S36. Each
composition series also holds out its mixed compositions.

### 4.3 Assemble the clean training set

```bash
python training/assemble_v5_train.py \
    --base v5build/v3_train_reduce_emb.extxyz --frames_csv audit_ps4/frames.csv \
    --gb_root ps4ok --bulk_root bulkdef --neb_mult 3 --md_mult 1 \
    --outA v5build/v5_trainA.extxyz --outB v5build/v5_trainB.extxyz \
    --out_dft v5build/v5_clean_dft.extxyz
```

`v5_trainB` (the reactive base minus `n_vertexX>0` frames, plus clean DFT) is the
set used in the paper. `v5_trainA` keeps every base frame and serves as the
control.

### 4.4 Active learning on GB cores

The fine-tuned predecessors decomposed PS₄ in GB cores at 700 K. The AL loop
gives the model the DFT energy of such configurations:

```bash
# 1) DFT-sized slab around one GB core (y/z periodic, vacuum along x, outer 2.5 A frozen)
python training/cut_gb_core.py --data gbV2_uni.extxyz --gb-x 17.36 --half_width 7 --vacuum 15 --out cut_uni_gb1.extxyz
# 2) constrained relax + 700 K MD per candidate model, logging PS4 integrity every 50 steps
python training/relax_md_cut.py --data cut_uni_gb1.extxyz --model v5B.pth --temp 700 --steps 10000 --out rx_uni_gb1_v5B
# 3) pick frames at decomposition events (broken models) + evenly spaced healthy frames; split by cut
python training/select_al_frames.py --dir . --train_cuts n8_gb1,n8_gb2,uni_gb1 --test_cuts uni_gb2 --out al1
# 4) DFT single points (no stress); optional constrained DFT relax of a cut:
python training/make_scf_inputs.py --traj al1_train.extxyz --out al1train --klen 26 --pseudo_dir /path/to/pseudo
python training/make_relax_cut_pwin.py --data cut_n8_gb1.extxyz --klen 26 --pseudo_dir /path/to/pseudo --out dftrx_n8_gb1
# 5) converged QE outputs -> extxyz on the same cohesive-energy zero
AL_DIR=/path/to/al python training/al_to_extxyz.py      # -> v5C_al_train.extxyz / v5C_al_test.extxyz
```

### 4.5 Fine-tune v5D

```bash
MODEL=models/v4_l0.pth TRAINB=v5build/v5_trainB.extxyz \
AL_TRAIN=al/v5C_al_train.extxyz AL_TEST=al/v5C_al_test.extxyz VALID=v3_test_emb.extxyz \
sh training/run_v5D.sh
```

What the script does:

1. **Energy gauge.** It reuses `energy_gauge_v4l0.json` if present, otherwise it
   refits the gauge. The gauge must be fitted against the checkpoint the
   fine-tune starts from (`v4_l0`). Training stops if any data set still deviates
   by more than the Huber δ = 0.2 eV/atom. A gauge fitted against a different
   checkpoint is off by 1.4–4.5 eV/atom on the Li₂S-ladder AL frames, and those
   frames are then never learned.
2. **Assembly.** It builds `v5D_train.extxyz` = `v5_trainB` + AL × 6.
3. **Training.** 80 epochs warm-started from v4 (λ = 0), lr 1e-4, batch 16, Huber
   δ 0.2, force:energy = 4, Li:P:S:Br:Cl = 3:1:1:1:1, no stress, no early stop,
   a checkpoint every 5 epochs. The model is `results_v5D/best_model.pth`.

Checkpoint gate (decomposed-vs-intact PS₄ energy against DFT single points in
`dftsp_relax/in/`, plus E/F on the held-out AL frames). Run it in the directory
holding `dftsp_relax/` and `al/`:

```bash
python training/eval_v5C_gate.py --models results_v5D/best_model.pth,v5B.pth,pretrained --device cpu
```

The L2-SP sweep (SI S5) used `finetune_mattersim.py --l2sp_lambda <λ>`. The
superseded phaseB recipe is in `legacy/`.

---

## 5. Validation against DFT (`validation/`)

```bash
# held-out Li-hop barriers, pointwise on the DFT geometries (25 physical paths in the paper)
python validation/eval_v5_barriers.py --model v5D.pth --device cuda \
    --gb_root ps4ok --bulk_root bulkdef --out eval_v5/bar_v5D.csv
python validation/eval_v5_barriers.py --model pretrained --device cuda \
    --gb_root ps4ok --bulk_root bulkdef --out eval_v5/bar_pretrained.csv
# MLIP vs DFT on one path's images
python validation/neb_dft_compare.py --barrier nebS_rnd0_0_50_barrier.csv --qe_glob 'dft_nebS_rnd0_0_50/*.out'
# energy / force / stress parity on identical geometries
python validation/parity.py --qe_glob 'exttest/*.out' --models pretrained v5D.pth
python validation/train_parity.py -h
# elastic constants: 13 strained frames (+-0.005), ionic relax 0.02 eV/A, identical for all models and DFT
python validation/elastic_dft.py -h ; python validation/elastic_from_stress.py -h
```

In the paper, three paths with DFT barriers above 2 eV (unconverged endpoints)
are excluded from the barrier MAE.

---

## 6. GB models and transport (`application/`)

### 6.1 GB models

```bash
# uniform and segregated halogen decoration of the Sigma5(210) template (1448 atoms)
python application/gb_build_uniform.py --xyz test.GB.xyz --repeat 1,1,1 --enrich 2 --lam 3 --seed 0 --out gbV2_uni.extxyz
python application/gb_build_uniform.py --xyz test.GB.xyz --repeat 1,1,1 --enrich 4 --lam 3 --seed 0 --out gbV2_seg.extxyz
# amorphous GB: melt only the GB slabs (+-0.10 Lx), bulk frozen; v5D, dt 1 fs, 10 + 10 + 5 ps
python application/amorphize_gb.py --data gbV2_uni.extxyz --model v5D.pth --device cpu \
    --gb-pos 0.25,0.75 --gb-axis x --half-width 0.10 --t_hot 1300 --t_cold 300 --dt 1.0 \
    --hot_ps 10 --quench_ps 10 --relax_ps 5 --seed 0 --out gbV2_amo.extxyz
# PS4 audit (P coordination = 4, no vertex halide, no short S-S / Li-Li / P-P)
python application/audit_gb_ps4.py gbV2_uni.extxyz gbV2_seg.extxyz gbV2_amo.extxyz
```

* The two `gb_build_uniform.py` commands reproduce the paper's
  `gbV2_uni`/`gbV2_seg` exactly (same species on every site).
* Use `--repeat 1,1,1`. The y-halved template (`make_mid`) is not periodic.
* Melting at T_hot = 2000 K broke PS₄ inside the slab. 1300 K is the setting
  used in the paper. Always audit the result.

### 6.2 Li diffusion (4344-atom cell = gbV2 × (1,1,3))

```bash
for M in v5D.pth pretrained; do for S in 0 1 2; do
  python application/diffusion_md.py --data gbV2_uni.extxyz --model $M --device cpu \
      --species Li --temp 700 --dt 2.0 --nsteps 50000 --equil 5000 --nsample 10 \
      --seed $S --repeat "1 1 3" --traj_every 50 --ckpt_every 2000 --resume \
      --regions "GB1:x:13.86:20.86 BULK:x:30:40 GB2:x:48.58:55.58" \
      --out msdV2uni_$(basename $M .pth)_700K_s$S
done; done
```

* Run structures `uni`, `seg` and `amo` at 500, 600, 700 and 800 K.
* At 500 K use `--nsteps 100000` (200 ps).
* One 50k-step run takes about 15 h on 32 CPU threads (≈1.1 s per step).
  `--resume` continues from the checkpoint, so a run can be split over several
  jobs.
* After a run, check that PS₄ survived:
  `python application/check_md_struct.py <out>_ckpt.extxyz`.

Region-resolved diffusion coefficients (per seed) and the seed-averaged
Arrhenius refit:

```bash
mkdir -p reg
for f in msdV2*_traj.npz; do b=${f%_traj.npz}
  python application/msd_regions.py $f --axis x \
      --regions "GB1:13.86:20.86 BULK:30:40 GB2:48.58:55.58" --lag 20 --out reg/reg_V2${b#msdV2}
done
python application/refit_arrhenius.py --reg_dir reg --out refit_summary.csv
```

`msd_regions.py` re-assigns each Li to a region at every time origin and fits
only short lags. A Li ion leaves the 7 Å GB window within 29–48 ps at 700–800 K,
so labelling by the initial position alone is wrong. `refit_arrhenius.py`
reports D_GB/D_bulk per temperature (mean ± s.e.m. over seeds) and
ΔEa(GB − bulk) with a seed-bootstrap error.

### 6.3 Mechanism indicators

```bash
python application/mechanism_tier1.py --dir . --struct uni --temps 500 700 --seeds 0 1 2 --lag 20
python application/coord_tier2.py v5D=nemd_v5D_gbV2_uni_700K_z_F000_s0_ckpt.extxyz \
    pretrained=nemd_pretrained_gbV2_uni_700K_z_F000_s0_ckpt.extxyz --out coord_tier2.json
```

* Tier 1 measures, for GB and bulk: cage MSD at 1 ps, the α₂ peak, the median
  displacement, and the fraction of Li moving more than 2 Å at 20 ps.
* Tier 2 measures the static Li coordination (r < 3.2 Å) and the first-shell
  halide fraction on equilibrated 700 K end states.

### 6.4 Framework dynamics (all-atom, 1448-atom cell)

```bash
python application/framework_md.py --data gbV2_uni.extxyz --model v5D.pth --device cpu \
    --temp 700 --dt 2.0 --nsteps 20000 --equil 5000 --dump_every 25 --seed 0 --out fwV2_v5D_uni_700K
python application/tier3_fwV2.py fwV2_*_fwtraj.npz --out tier3_fwV2_summary.json
```

* Runs used in the paper: `uni` at 500 and 700 K and `seg` at 700 K, for both
  force fields.
* The summary gives PS₄ C₁(10 ps), the halide MSD and the Li MSD for GB vs bulk.

### 6.5 Colour-charge NEMD

```bash
python application/nemd_ccd.py --data gbV2_uni.extxyz --model v5D.pth --device cpu \
    --temp 700 --dt 2.0 --nsteps 50000 --equil 10000 \
    --field 0.02 --field-axis z --profile-axis x --nbins 40 --block 250 \
    --seed 0 --repeat "1 1 3" --ckpt_every 2500 --resume --out nemd_v5D_gbV2_uni_700K_z_F020_s0
python application/nemd_analysis.py profile  "nemd_v5D_gbV2_uni_700K_z_F*_s*_nemd.npz" --gb 17.36 52.08 --bulk 30 40 --out prof_v5D_gbV2_700K
python application/nemd_analysis.py haven    "nemd_v5D_gbV2_uni_700K_z_F*_s*_nemd.npz"
python application/nemd_analysis.py arrhenius "nemd_v5D_gbV2_uni_*K_z_F*_s*_nemd.npz" --out arr_v5D_gbV2
python application/nemd_analysis.py rgb --perp "nemd_v5D_gbV2_uni_700K_x_F*_s*_nemd.npz" \
    --par "nemd_v5D_gbV2_uni_700K_z_F*_s*_nemd.npz" --bulk 30 40 --ngb 2 --out rgb_v5D_gbV2_700K
```

* Linear response was checked at 700 K with F_e = 0, 0.005, 0.01, ±0.02 and
  0.04 eV/Å.
* Production runs used F_e = 0.02 eV/Å, parallel (`z`) and perpendicular (`x`)
  to the GB planes, at 500–800 K with three seeds.
* The F_e = 0 runs (`..._F000_..._ckpt.extxyz`) also serve as equilibrated end
  states for 6.3.

### 6.6 Space charge

```bash
python application/charge_profile_tavg.py --static gbV2_uni.extxyz     # as-built: q_GB = -24 e, phi step 2.34 V
python application/charge_profile_tavg.py --dir . --tag 700K --out charge_tavg.npz
```

This time-averages ρ_q(x) over the 40 ps all-atom runs from 6.4 (formal charges,
46 bins, 2.5 Å Gaussian smoothing, ε_r = 15). Expected Δφ: `uni` 0.25 (v5D) /
0.21 V (generic), `seg` 0.51 / 1.18 V.

### 6.7 Stability of the halogen decoration (SI S6)

```bash
for C in uni seg; do
  python application/gb_segregation.py --data gbV2_$C.extxyz --model v5D.pth   --device cpu --nswap 20 --out segV2_v5D_$C
  python application/gb_segregation.py --data gbV2_$C.extxyz --model pretrained --device cpu --nswap 20 --out segV2_generic_$C
done
```

---

## 7. `legacy/`

These scripts come from the phaseB-era pipeline (the `gbBIG_dec` cell, phaseB
recipe and gauge, older analysis drivers). They are kept for provenance only:
none of the results in the paper depend on them. See `legacy/README.md`.

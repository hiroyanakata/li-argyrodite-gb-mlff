# legacy/

Scripts from the phaseB-era pipeline, kept for provenance only. **None of the
results in the paper depend on them.** They were superseded after an error in
the old halide decoration was found: `gb_build.py --neutral` converted
PS4-vertex S into halide.

| file | superseded by |
|---|---|
| `run_phaseB.sh`, `assemble_phaseB_train.py`, `energy_gauge.json` | `training/run_v5D.sh`, `assemble_v5_train.py` + `assemble_v5C.py`, `energy_gauge_v4l0.json` (the old gauge was fitted against a different checkpoint than the warm start) |
| `msd_arrhenius.py` | `application/refit_arrhenius.py` (seed-averaged, bootstrap errors) |
| `mechanism_analysis.py` | `application/mechanism_tier1.py` |
| `tier2_coordination.py` | `application/coord_tier2.py` |
| `charge_profile.py`, `charge_profile_avg.py`, `plot_charge_avg.py` | `application/charge_profile_tavg.py` (time average over the all-atom runs) |

These scripts read the old `gbBIG_dec.extxyz` / `gbBIG_std.extxyz` cells (4146
atoms), which contain PS4-vertex halides. Do not use them for new work.

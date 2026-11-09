#!/bin/sh
# =============================================================================
# run_v5D.sh -- the exact recipe of the v5D force field used in the paper.
#
#   v5D = v5_trainB (reactive base with PS4-intact frames + clean migration-path
#         DFT, assemble_v5_train.py) + active-learning GB-core frames x AL_MULT
#         (assemble_v5C.py), warm-started from the v4 (lambda = 0) checkpoint.
#
# Steps (each is skipped if its output already exists):
#   1. refit the per-element cohesive-energy gauge against the checkpoint the
#      fine-tune actually starts from (fit_energy_gauge.py -> energy_gauge_v4l0.json;
#      the file shipped in this directory is the one used for the paper), and
#      refuse to train if any validation set is still outside the Huber
#      quadratic branch (delta = 0.2 eV/atom);
#   2. assemble the training set (AL frames duplicated AL_MULT times);
#   3. fine-tune: fixed budget of EPOCHS, early stopping disabled (patience =
#      EPOCHS), checkpoint every 5 epochs. best_model.pth (lowest validation
#      loss) is the v5D model; candidate checkpoints were additionally screened
#      with eval_v5C_gate.py (broken-vs-intact PS4 energy ordering) and
#      validation/eval_v5_barriers.py (held-out Li-hop barriers).
#
# Run from the directory that holds the data; all paths can be overridden:
#   MODEL=models/v4_l0.pth TRAINB=v5build/v5_trainB.extxyz \
#   AL_TRAIN=al/v5C_al_train.extxyz AL_TEST=al/v5C_al_test.extxyz \
#   VALID=v3_test_emb.extxyz sh /path/to/training/run_v5D.sh
#
# Knobs (env): EPOCHS(80) BATCH_SIZE(16) AL_MULT(6) GAUGE(energy_gauge_v4l0.json)
#   SAVE_PATH(results_v5D) DEVICE(cuda) MASTER_PORT(29554)
# =============================================================================
set -e
HERE=$(cd "$(dirname "$0")" && pwd)

EP=${EPOCHS:-80}
BS=${BATCH_SIZE:-16}
MULT=${AL_MULT:-6}
MODEL=${MODEL:-models/v4_l0.pth}
TRAINB=${TRAINB:-v5build/v5_trainB.extxyz}
AL_TRAIN=${AL_TRAIN:-al/v5C_al_train.extxyz}
AL_TEST=${AL_TEST:-al/v5C_al_test.extxyz}
VALID=${VALID:-v3_test_emb.extxyz}
GAUGE=${GAUGE:-energy_gauge_v4l0.json}
SAVE_PATH=${SAVE_PATH:-results_v5D}
DEVICE=${DEVICE:-cuda}
TRAIN=${TRAIN:-v5build/v5D_train.extxyz}

# ---- 1. energy gauge --------------------------------------------------------
if [ -s "$GAUGE" ]; then
  echo "=== reusing existing $GAUGE ==="
else
  echo "=== refitting the per-element gauge against $MODEL ==="
  python "$HERE/fit_energy_gauge.py" \
    --model "$MODEL" --data "$TRAINB" --stride 5 \
    --validate "$VALID" "$AL_TRAIN" "$AL_TEST" \
    --device "$DEVICE" --batch_size 8 --out "$GAUGE"
fi
python - "$GAUGE" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print("gauge source_model =", d["source_model"])
bad = [(k, v["mae_after_per_atom"]) for k, v in d["fit_stats"].items()
       if v["mae_after_per_atom"] > 0.2]
for k, v in d["fit_stats"].items():
    print("  %-40s before %9.3f -> after %.4f eV/atom" % (
        k.split("/")[-1], v["mae_before_per_atom"], v["mae_after_per_atom"]))
if bad:
    sys.exit("FATAL: residual still above the Huber delta: %s" % bad)
PY

# ---- 2. training set ----------------------------------------------------------
if [ ! -s "$TRAIN" ]; then
  python "$HERE/assemble_v5C.py" --trainB "$TRAINB" --al_train "$AL_TRAIN" \
    --al_mult "$MULT" --out "$TRAIN"
fi

# ---- 3. fine-tune -------------------------------------------------------------
export LOCAL_RANK=0 RANK=0 WORLD_SIZE=1
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1} MASTER_PORT=${MASTER_PORT:-29554}
python -c "import torch, mattersim; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'mattersim', mattersim.__version__)"
echo "=== v5D train=$TRAIN epochs=$EP bs=$BS al_mult=$MULT gauge=$GAUGE ==="
python "$HERE/finetune_mattersim.py" \
  --train_data_path "$TRAIN" --valid_data_path "$VALID" \
  --load_model_path "$MODEL" --resume \
  --save_path "$SAVE_PATH" --run_name v5D --save_checkpoint --ckpt_interval 5 \
  --epochs "$EP" --batch_size "$BS" --lr 1e-4 --loss huber --huber_delta 0.2 \
  --include_forces --force_loss_ratio 4 --l2sp_lambda 0 \
  --force_element_weights Li:3,P:1,S:1,Br:1,Cl:1 --force_mag_weight_alpha 0 \
  --no-include_stresses --energy_gauge "$GAUGE" \
  --early_stop_patience "$EP"
ls -lt "$SAVE_PATH" | head -20

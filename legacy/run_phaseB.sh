#!/bin/sh
# Start or resume the Phase B fine-tune on Colab GPU (warm-start from v4_l0, add
# DFT Li-barrier data). Mirrors run_v4_train.sh but with the Phase B recipe:
#   warm = results_v4_l0/best_model.pth   data = phaseB_train.extxyz
#   NO --include_stresses (phaseB frames carry no virial)   l2sp=0   lr=1e-4
#   Li:3 element force weight   huber(0.2)   force_loss_ratio=4   metric=val_mae_f
#
# Disconnect-safe & idempotent (same as v4): every epoch writes
# results_phaseB/last_model.pth on Drive; --resume makes each attempt pick up
# from it; the inner retry loop auto-resumes transient crashes. So after a Colab
# VM dies, reconnect and run the SAME command again.
#
#   sh /path/to/training/run_phaseB.sh                                   # foreground
#   EPOCHS=200 nohup sh /path/to/training/run_phaseB.sh >/dev/null 2>&1 &   # background
#
# Knobs (env, optional): EPOCHS(200) SAVE_PATH(results_phaseB) BATCH_SIZE(16)
#   LR(1e-4) MODEL(results_v4_l0/best_model.pth) TRAIN(phaseB_train.extxyz)
#   VALID(v3_test_emb.extxyz) FWEIGHTS(Li:3,P:1,S:1,Br:1,Cl:1) METRIC(val_mae_f)
set -e
HERE=$(cd "$(dirname "$0")" && pwd)   # this training/ directory; run from your data directory

EPOCHS=${EPOCHS:-200}
SAVE_PATH=${SAVE_PATH:-results_phaseB}
BATCH_SIZE=${BATCH_SIZE:-16}
LR=${LR:-1e-4}
GAUGE=${GAUGE:-$HERE/energy_gauge.json}
MODEL=${MODEL:-results_v4_l0/best_model.pth}
TRAIN=${TRAIN:-phaseB_train.extxyz}
VALID=${VALID:-v3_test_emb.extxyz}
FWEIGHTS=${FWEIGHTS:-Li:3,P:1,S:1,Br:1,Cl:1}
METRIC=${METRIC:-val_mae_f}
MAX_RETRIES=${MAX_RETRIES:-20}
LOG="$SAVE_PATH/train.log"

mkdir -p "$SAVE_PATH"

if [ ! -f "$GAUGE" ]; then
    echo "FATAL: energy gauge $GAUGE missing -- Phase B must reuse the SAME gauge" | tee -a "$LOG"
    echo "as v3/v4 (do NOT refit; a different gauge shifts the energy target)." | tee -a "$LOG"
    exit 2
fi

PIDFILE="$SAVE_PATH/train.pid"
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "a run against $SAVE_PATH is already in flight (pid $(cat "$PIDFILE")):"
    ps -o pid,etime,cmd -p "$(cat "$PIDFILE")" 2>/dev/null || true
    echo "stop it first, or use a different SAVE_PATH."
    exit 1
fi
echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT INT TERM

if [ -f "$SAVE_PATH/last_model.pth" ]; then
    echo "resuming from $SAVE_PATH/last_model.pth (target $EPOCHS epochs)" | tee -a "$LOG"
else
    echo "starting from $MODEL (target $EPOCHS epochs)" | tee -a "$LOG"
fi

export LOCAL_RANK=0 RANK=0 WORLD_SIZE=1
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1} MASTER_PORT=${MASTER_PORT:-29541}

n=0
while : ; do
    if python "$HERE/finetune_mattersim.py" \
        --train_data_path "$TRAIN" \
        --valid_data_path "$VALID" \
        --load_model_path "$MODEL" \
        --energy_gauge "$GAUGE" \
        --resume \
        --save_path "$SAVE_PATH" --run_name phaseB --save_checkpoint --ckpt_interval 5 \
        --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" --lr "$LR" \
        --loss huber --huber_delta 0.2 \
        --include_forces --force_loss_ratio 4 \
        --l2sp_lambda 0 \
        --force_element_weights "$FWEIGHTS" --force_mag_weight_alpha 0 \
        --metric_name "$METRIC" \
        >> "$LOG" 2>&1
    then
        echo "training finished cleanly." | tee -a "$LOG"
        break
    fi
    rc=$?
    n=$((n + 1))
    echo "run exited rc=$rc (attempt $n/$MAX_RETRIES); resuming in 15s" | tee -a "$LOG"
    if [ "$n" -ge "$MAX_RETRIES" ]; then
        echo "giving up after $n attempts" | tee -a "$LOG"
        exit 1
    fi
    sleep 15
done

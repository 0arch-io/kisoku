#!/bin/bash
# Long-context memory smoke test (synthetic data, no checkpoints).
# Usage: kisoku-smoke.sh NAME LEN GA VOCAB_TILING [extra overrides...]
set -u
NAME=$1; LEN=$2; GA=$3; VT=$4; shift 4
rm -f /tmp/libtpu_lockfile
cd "$HOME/maxtext" || exit 1
export LIBTPU_INIT_ARGS="--xla_enable_async_all_gather=true TPU_MEGACORE=MEGACORE_DENSE"
export GOOGLE_APPLICATION_CREDENTIALS="$HOME/gcs-key.json"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
LOG="$HOME/logs/smoke-$NAME.log"
echo "=== SMOKE START $(date -u) $NAME len=$LEN ga=$GA vt=$VT $* on $(hostname) ===" >> "$LOG"
exec .venv/bin/python -u -m maxtext.trainers.pre_train.train "$HOME/kisoku-v2-1b.yml" \
  run_name="smoke-$NAME" \
  base_output_directory=gs://kisoku-v2-training/smoke/ \
  dataset_type=synthetic \
  ici_fsdp_parallelism=16 \
  per_device_batch_size=1 \
  gradient_accumulation_steps="$GA" \
  max_target_length="$LEN" \
  remat_policy=full \
  num_vocab_tiling="$VT" \
  attention=flash \
  steps=20 \
  enable_checkpointing=False \
  gcs_metrics=False enable_tensorboard=False \
  eval_interval=-1 \
  log_period=5 \
  "$@" >> "$LOG" 2>&1

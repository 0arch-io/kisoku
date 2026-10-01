#!/bin/bash
# Kisoku v2 chat fine-tune (SFT) entrypoint, run on all 4 workers at once via systemd-run.
# usage: kisoku-sft-run.sh RUN_NAME STEPS [extra MaxText overrides...]
# No auto-restart on purpose: a failed SFT run should stop and be looked at, not loop.
set -u
RUN="$1"; STEPS="$2"; shift 2
export PATH="$HOME/.local/bin:$PATH"
rm -f /tmp/libtpu_lockfile
cd "$HOME/maxtext" || exit 1
export LIBTPU_INIT_ARGS="--xla_enable_async_all_gather=true TPU_MEGACORE=MEGACORE_DENSE"
export GOOGLE_APPLICATION_CREDENTIALS="$HOME/gcs-key.json"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

echo "=== START $(date -u) run=$RUN steps=$STEPS extra=$* on $(hostname) ===" >> "$HOME/logs/$RUN.log"
exec .venv/bin/python -u -m maxtext.trainers.post_train.sft.train_sft_native "$HOME/kisoku-v2-1b-sft.yml" \
  run_name="$RUN" \
  steps="$STEPS" \
  learning_rate_schedule_steps="$STEPS" \
  "$@" \
  >> "$HOME/logs/$RUN.log" 2>&1

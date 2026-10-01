#!/bin/bash
# Auto hand-off Phase A -> Phase B (2026-10-01, TRC grant already ended so the TPU may vanish any time: never idle).
# Runs on every worker (systemd-run). Waits for Phase A to finish, worker 0 copies the final checkpoint to the
# permanent path kisoku-run.sh expects for stage B, every worker then switches ~/kisoku-stage.env and starts training.
set -u
export PATH="$HOME/.local/bin:/snap/bin:$PATH"
LOG="$HOME/logs/next-stage.log"
RUN_A=kisoku-v2-1b-longctx-a; STEP_A=2899
FINAL_A="gs://kisoku-v2-training/runs/kisoku-v2-1b-longctx-a-final"
FINAL_A_FUSE="$HOME/gcsfuse/runs/kisoku-v2-1b-longctx-a-final"
W=$(hostname | grep -o 'w-[0-9]*$' | cut -d- -f2)
echo "=== $(date -u) next-stage watcher started on worker $W ===" >> "$LOG"

# 1. wait for the Phase A final checkpoint and a stopped trainer
until [ -f "$HOME/gcsfuse/runs/$RUN_A/checkpoints/$STEP_A/commit_success.txt" ] && ! systemctl --user is-active --quiet kisoku-train; do sleep 60; done
echo "$(date -u) phase A done (ckpt $STEP_A present, trainer inactive)" >> "$LOG"

# 2. worker 0 copies the checkpoint out (max_num_checkpoints_to_keep=5 would delete it during phase B)
if [ "$W" = "0" ]; then
  export CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE="$HOME/gcs-key.json"
  if ! gcloud storage ls "$FINAL_A/commit_success.txt" >/dev/null 2>&1; then
    gcloud storage cp -r "gs://kisoku-v2-training/runs/$RUN_A/checkpoints/$STEP_A/*" "$FINAL_A/" >> "$LOG" 2>&1
    echo "$(date -u) copied $STEP_A -> $FINAL_A" >> "$LOG"
  fi
fi
# 3. everyone waits until the copy is visible (gcsfuse metadata cache can lag; poll the bucket path)
until [ -f "$FINAL_A_FUSE/commit_success.txt" ] || ls "$FINAL_A_FUSE/items" >/dev/null 2>&1; do sleep 30; done
sleep 60   # let worker 0's copy fully land before anyone restores from it

# 4. switch to stage B and start
printf 'STAGE=B\nRUN=kisoku-v2-1b-longctx-b\nSTEPS=2860\n' > "$HOME/kisoku-stage.env"
rm -f "$HOME/.kisoku-guard"
systemctl --user reset-failed kisoku-train 2>/dev/null
systemctl --user start kisoku-train
echo "$(date -u) stage B started: $(systemctl --user is-active kisoku-train)" >> "$LOG"

#!/bin/bash
# Auto hand-off Phase B -> Phase C (2026-10-01). Same pattern as kisoku-next-stage.sh (A -> B).
# To cancel before it fires: systemctl --user stop kisoku-next-stage-c   (on every worker)
set -u
export PATH="$HOME/.local/bin:/snap/bin:$PATH"
LOG="$HOME/logs/next-stage.log"
RUN_B=kisoku-v2-1b-longctx-b; STEP_B=2859
FINAL_B="gs://kisoku-v2-training/runs/kisoku-v2-1b-longctx-b-final"
FINAL_B_FUSE="$HOME/gcsfuse/runs/kisoku-v2-1b-longctx-b-final"
W=$(hostname | grep -o 'w-[0-9]*$' | cut -d- -f2)
echo "=== $(date -u) next-stage-c watcher started on worker $W ===" >> "$LOG"
until [ -f "$HOME/gcsfuse/runs/$RUN_B/checkpoints/$STEP_B/commit_success.txt" ] && ! systemctl --user is-active --quiet kisoku-train; do sleep 60; done
echo "$(date -u) phase B done (ckpt $STEP_B present, trainer inactive)" >> "$LOG"
if [ "$W" = "0" ]; then
  export CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE="$HOME/gcs-key.json"
  if ! gcloud storage ls "$FINAL_B/commit_success.txt" >/dev/null 2>&1; then
    gcloud storage cp -r "gs://kisoku-v2-training/runs/$RUN_B/checkpoints/$STEP_B/*" "$FINAL_B/" >> "$LOG" 2>&1
    echo "$(date -u) copied $STEP_B -> $FINAL_B" >> "$LOG"
  fi
fi
until [ -f "$FINAL_B_FUSE/commit_success.txt" ] || ls "$FINAL_B_FUSE/items" >/dev/null 2>&1; do sleep 30; done
sleep 60
printf 'STAGE=C\nRUN=kisoku-v2-1b-longctx-c\nSTEPS=1300\n' > "$HOME/kisoku-stage.env"
rm -f "$HOME/.kisoku-guard"
systemctl --user reset-failed kisoku-train 2>/dev/null
systemctl --user start kisoku-train
echo "$(date -u) stage C started: $(systemctl --user is-active kisoku-train)" >> "$LOG"

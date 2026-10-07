#!/bin/bash
# For each STEP: wait for the DPO checkpoint, rewrite it as items (CPU), export HF + F16 GGUF to the bucket.
: "${KISOKU_BUCKET:?set KISOKU_BUCKET to your GCS bucket name (no gs:// prefix)}"
RUN=$1; shift; cd ~/maxtext
for STEP in "$@"; do
  until gcloud storage ls "gs://${KISOKU_BUCKET}/runs/$RUN/checkpoints/$STEP/commit_success.txt" >/dev/null 2>&1; do sleep 60; done
  GOOGLE_APPLICATION_CREDENTIALS=$HOME/gcs-key.json JAX_PLATFORMS=cpu .venv/bin/python ~/bin/dpo_to_items.py gs://${KISOKU_BUCKET}/runs/$RUN/checkpoints/$STEP >> ~/logs/dpo-export.log 2>&1 || { echo "ITEMS FAILED $STEP" >> ~/logs/dpo-export.log; continue; }
  bash ~/bin/export-chat-shm.sh $RUN $STEP kisoku-1.6b-chat-dpo$STEP >> ~/logs/dpo-export.log 2>&1
  rm -rf /dev/shm/kisoku-hf/kisoku-1.6b-chat-dpo$STEP /dev/shm/kisoku-hf/kisoku-1.6b-chat-dpo$STEP-F16.gguf
done

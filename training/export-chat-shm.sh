#!/bin/bash
# Export a finished chat SFT run without touching worker 0's nearly full root disk: everything is written to /dev/shm (RAM).
# Waits for the final checkpoint, converts to HF bf16 (convert-chat.sh with its paths pointed at /dev/shm), builds the F16 GGUF,
# uploads both. usage: export-chat-shm.sh RUN STEP NAME
set -u
RUN="$1"; STEP="$2"; NAME="$3"
export CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE="$HOME/gcs-key.json" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p /dev/shm/kisoku-hf/fp32
sed "s#\$HOME/kisoku-hf#/dev/shm/kisoku-hf#g" "$HOME/bin/convert-chat.sh" > /dev/shm/convert-chat-shm.sh
bash /dev/shm/convert-chat-shm.sh "$RUN" "$STEP" "$NAME" || exit 1
cd "$HOME/llama.cpp" && PYTHONPATH="$HOME/llama.cpp/gguf-py" "$HOME/hf-venv/bin/python" convert_hf_to_gguf.py "/dev/shm/kisoku-hf/$NAME" --outtype f16 \
  --outfile "/dev/shm/kisoku-hf/$NAME-F16.gguf" >> "$HOME/logs/convert-chat.log" 2>&1 || exit 1
gcloud storage cp "/dev/shm/kisoku-hf/$NAME-F16.gguf" "gs://kisoku-v2-training/hf/gguf-$NAME/" >> "$HOME/logs/convert-chat.log" 2>&1 && echo "GGUF UPLOADED $NAME $(date -u)" >> "$HOME/logs/convert-chat.log"
rm -rf "/dev/shm/kisoku-hf/fp32/$NAME"

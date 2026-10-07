#!/bin/bash
# GGUF (F16, Q8_0, Q4_K_M) for a chat export in ~/kisoku-hf/<name>, with the llama.cpp already built on worker 0; uploads to the bucket.
: "${KISOKU_BUCKET:?set KISOKU_BUCKET to your GCS bucket name (no gs:// prefix)}"
set -eu
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE="$HOME/gcs-key.json"
name="${1:-kisoku-1.6b-chat-sft001}"; H=$HOME/kisoku-hf; O=$H/gguf-$name; mkdir -p $O
cd ~/llama.cpp && git log -1 --format="llama.cpp %h %cd"
PYTHONPATH=~/llama.cpp/gguf-py ~/hf-venv/bin/python convert_hf_to_gguf.py $H/$name --outtype f16 --outfile $O/$name-F16.gguf
for q in Q8_0 Q4_K_M; do ./build/bin/llama-quantize $O/$name-F16.gguf $O/$name-$q.gguf $q > /dev/null; done
ls -la $O
gcloud storage cp $O/*.gguf gs://${KISOKU_BUCKET}/hf/gguf-$name/ 2>&1 | tail -1
echo GGUF-OK

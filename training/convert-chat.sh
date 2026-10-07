#!/bin/bash
# Wait for the final chat SFT checkpoint, then convert it to HF bf16 on worker 0's CPU and upload to gs://${KISOKU_BUCKET}/hf/.
# Chat export: finalize_hf.py keeps <|eot_id|> as EOS and the chat template. Two patches after it: max_position_embeddings 65536
# (the converter hardcodes 4096) and no repetition_penalty in generation_config.json (1.1 cost ~7 RULER points on the base model;
# set a penalty in the Ollama Modelfile instead if chat needs one).
: "${KISOKU_BUCKET:?set KISOKU_BUCKET to your GCS bucket name (no gs:// prefix)}"
set -u
export PATH="$HOME/.local/bin:$PATH"
export JAX_PLATFORMS=cpu
export GOOGLE_APPLICATION_CREDENTIALS="$HOME/gcs-key.json"
export CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE="$HOME/gcs-key.json"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
RUN="${1:-kisoku-v2-1b-chat-sft-001}"; STEP="${2:-3299}"; name="${3:-kisoku-1.6b-chat-sft001}"
cd "$HOME/maxtext" || exit 1
LOG="$HOME/logs/convert-chat.log"
echo "=== START $(date -u) waiting for $RUN/$STEP ===" >> "$LOG"
until gcloud storage ls "gs://${KISOKU_BUCKET}/runs/$RUN/checkpoints/$STEP/commit_success.txt" >/dev/null 2>&1; do sleep 60; done
sleep 30
out="$HOME/kisoku-hf/fp32/$name"; bf="$HOME/kisoku-hf/$name"
echo "--- $(date -u) convert -> $out" >> "$LOG"
.venv/bin/python -u "$HOME/bin/convert_kisoku_hf.py" "$HOME/kisoku-v2-1b.yml" model_name=qwen3-1.7b override_model_config=True \
  base_num_kv_heads=4 base_mlp_dim=8192 base_num_decoder_layers=22 vocab_size=128256 rope_max_timescale=5000000 \
  load_parameters_path="gs://${KISOKU_BUCKET}/runs/$RUN/checkpoints/$STEP/items" base_output_directory="$out" scan_layers=True weight_dtype=float32 \
  ici_fsdp_parallelism=1 per_device_batch_size=1 skip_jax_distributed_system=True --hf_model_path=unsloth/Llama-3.2-1B >> "$LOG" 2>&1 || { echo "CONVERT FAILED $name" >> "$LOG"; exit 1; }
"$HOME/hf-venv/bin/python" "$HOME/bin/finalize_hf.py" "$out" "$bf" >> "$LOG" 2>&1 || { echo "FINALIZE FAILED $name" >> "$LOG"; exit 1; }
"$HOME/hf-venv/bin/python" - "$bf" <<'PY' >> "$LOG" 2>&1
import json, sys, os
d = sys.argv[1]
p = os.path.join(d, "config.json"); j = json.load(open(p)); j["max_position_embeddings"] = 65536; json.dump(j, open(p, "w"), indent=2)
p = os.path.join(d, "generation_config.json"); j = json.load(open(p)); j.pop("repetition_penalty", None); json.dump(j, open(p, "w"), indent=2)
print("patched", d, j)
PY
ls -la "$bf" >> "$LOG"
gcloud storage cp -r "$bf" "gs://${KISOKU_BUCKET}/hf/" >> "$LOG" 2>&1 && echo "UPLOADED $name $(date -u)" >> "$LOG"
echo "=== DONE $(date -u) ===" >> "$LOG"

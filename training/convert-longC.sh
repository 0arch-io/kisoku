#!/bin/bash
# Convert the Phase C final checkpoint (64K, step 1299) to HF bf16 on worker 0 CPU and upload to gs://kisoku-v2-training/hf/.
set -u
export PATH="$HOME/.local/bin:$PATH"
export JAX_PLATFORMS=cpu
export GOOGLE_APPLICATION_CREDENTIALS="$HOME/gcs-key.json"
export CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE="$HOME/gcs-key.json"  # the compute service account cannot write to the bucket
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$HOME/maxtext" || exit 1
LOG="$HOME/logs/convert-longC.log"
echo "=== START $(date -u) ===" >> "$LOG"
for pair in "kisoku-v2-1b-longctx-c-final:kisoku-1.6b-base-longC-1299"; do
  src="${pair%%:*}"; name="${pair##*:}"
  out="$HOME/kisoku-hf/fp32/$name"; bf="$HOME/kisoku-hf/$name"
  echo "--- $(date -u) convert $src -> $out" >> "$LOG"
  .venv/bin/python -u "$HOME/bin/convert_kisoku_hf.py" "$HOME/kisoku-v2-1b.yml" model_name=qwen3-1.7b override_model_config=True \
    base_num_kv_heads=4 base_mlp_dim=8192 base_num_decoder_layers=22 vocab_size=128256 rope_max_timescale=5000000 \
    load_parameters_path="gs://kisoku-v2-training/runs/$src/items" base_output_directory="$out" scan_layers=True weight_dtype=float32 \
    ici_fsdp_parallelism=1 per_device_batch_size=1 skip_jax_distributed_system=True --hf_model_path=unsloth/Llama-3.2-1B >> "$LOG" 2>&1 || { echo "CONVERT FAILED $name" >> "$LOG"; continue; }
  echo "--- $(date -u) finalize -> $bf" >> "$LOG"
  "$HOME/hf-venv/bin/python" "$HOME/bin/finalize_hf.py" "$out" "$bf" >> "$LOG" 2>&1 || { echo "FINALIZE FAILED $name" >> "$LOG"; continue; }
  # Base model: end-of-text is the real EOS (finalize sets the chat <|eot_id|>, which a base model never emits).
  "$HOME/hf-venv/bin/python" - "$bf" <<'EOF' >> "$LOG" 2>&1
import json, sys, os
d = sys.argv[1]
for f, k in (("config.json", "eos_token_id"), ("generation_config.json", "eos_token_id")):
    p = os.path.join(d, f)
    if os.path.exists(p):
        j = json.load(open(p)); j[k] = 128001; json.dump(j, open(p, "w"), indent=2)
# Base exports must not carry chat sampling defaults: finalize_hf.py writes do_sample/temperature/top_p/repetition_penalty 1.1,
# and lm_eval keeps the repetition penalty on even with greedy decoding (it skewed every Kisoku generation eval, found 2026-10-01).
j = json.load(open(os.path.join(d, "config.json"))); j["max_position_embeddings"] = 65536; json.dump(j, open(os.path.join(d, "config.json"), "w"), indent=2)  # trained to 64K (the converter hardcodes 4096)
json.dump({"bos_token_id": 128000, "eos_token_id": 128001, "pad_token_id": 128001}, open(os.path.join(d, "generation_config.json"), "w"), indent=2)
p = os.path.join(d, "tokenizer_config.json")
j = json.load(open(p)); j["eos_token"] = "<|end_of_text|>"; json.dump(j, open(p, "w"), indent=2)
print("eos set to <|end_of_text|> (128001) in", d)
EOF
  ls -la "$bf" >> "$LOG"
  echo "--- $(date -u) upload $name" >> "$LOG"
  gcloud storage cp -r "$bf" "gs://kisoku-v2-training/hf/" >> "$LOG" 2>&1 && echo "UPLOADED $name" >> "$LOG"
done
echo "=== DONE $(date -u) ===" >> "$LOG"

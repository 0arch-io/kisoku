#!/bin/bash
# Batch 13b (2026-10-05, HumanEval only; the other v1 tests ran on the CTI box in batch 14): Kisoku v1 (0arch-io/kisoku-3b-base, 3B, ~60B training tokens, early 2026) through the same short-eval
# harness as the scorecard, to show v1 -> v2 growth. Waits for batch 12 (tmux kisoku-eval12) so only one model is on the GPU.
# v1 runs in float32: bfloat16 rounding breaks its tiny logits (generation becomes "!!!!", HumanEval 0.0; found 2026-10-05).
# v1 quirk: it was trained with vocab_size 128000, so the 256 special-token rows are untrained zeros and can win greedy decoding.
# Its own generation_config suppresses them; that list is kept (v1 cannot generate without it), while the sampling defaults and
# repetition_penalty 1.1 are removed so generation settings match every other model in the table.
export PATH=/usr/lib/wsl/lib:$HOME/ai/bin:$PATH
export HF_ALLOW_CODE_EVAL=1 TOKENIZERS_PARALLELISM=false
R=~/kisoku-eval/results; M=~/kisoku-eval/models; LOG=~/logs/kisoku-eval.log; V1=$M/kisoku-v1-3b-base

[ -d $R/kisoku-v1-humaneval ] && grep -q bfloat16 $R/kisoku-v1-humaneval/*/results_*.json 2>/dev/null && mv $R/kisoku-v1-humaneval $R/_bf16-kisoku-v1-humaneval
echo "=== BATCH 13b (Kisoku v1 short evals) START $(date -u)" >> $LOG
if [ ! -f $V1/model.safetensors ]; then
  ~/ai/bin/python - <<PY >> $LOG 2>&1
from huggingface_hub import snapshot_download
snapshot_download("0arch-io/kisoku-3b-base", local_dir="$V1")
PY
fi
~/ai/bin/python - $V1 <<'PY' >> $LOG 2>&1
import json, os, sys
p = os.path.join(sys.argv[1], "generation_config.json"); g = json.load(open(p))
if "repetition_penalty" in g or g.get("do_sample"):
    json.dump(g, open(p + ".orig", "w"), indent=1)
    json.dump({"bos_token_id": 128000, "eos_token_id": 128001, "pad_token_id": 128001, "suppress_tokens": g["suppress_tokens"]}, open(p, "w"), indent=1)
    print("v1 generation_config cleaned, suppress_tokens kept:", len(g["suppress_tokens"]))
PY
run() { local name=$1 pre=$2 tasks=$3 fs=$4 bs=$5; shift 5
  if ls $R/$name/*/results_*.json >/dev/null 2>&1; then echo "--- skip $name (done)" >> $LOG; return; fi
  echo "--- $(date -u) $name tasks=$tasks fewshot=$fs bs=$bs" >> $LOG
  local fsarg=(); [ "$fs" != "default" ] && fsarg=(--num_fewshot "$fs")
  lm_eval --model hf --model_args "pretrained=$pre,dtype=float32" --include_path ~/kisoku-eval/tasks --tasks "$tasks" "${fsarg[@]}" \
    --batch_size "$bs" --output_path "$R/$name" --log_samples "$@" >> $LOG 2>&1
  ls $R/$name/*/results_*.json >/dev/null 2>&1 && echo "--- $(date -u) $name done" >> $LOG || echo "--- $(date -u) $name FAILED (no results file)" >> $LOG
}
n=kisoku-v1
run "$n-humaneval" $V1 "humaneval" 0 16 --confirm_run_unsafe_code
echo "=== BATCH 13b DONE $(date -u)" >> $LOG

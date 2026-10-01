#!/bin/bash
# Batch 6 (2026-10-01): RULER 4K-32K preview. Kisoku Phase-A step 2500 (32K) vs the sub-2B long-context field.
# 100 samples per task per length (full RULER is 500; too slow on one 4090 at 32K). 64K/128K after Phase B.
export PATH=/usr/lib/wsl/lib:$HOME/ai/bin:$PATH
export TOKENIZERS_PARALLELISM=false
R=~/kisoku-eval/results; M=~/kisoku-eval/models; LOG=~/logs/kisoku-eval.log
[ -f $M/kisoku-1.6b-base-longA-2500/model.safetensors ] || cp -r /mnt/x/WSL/kisoku-hf/kisoku-1.6b-base-longA-2500 $M/
while tmux has-session -t =kisoku-eval3 2>/dev/null || tmux has-session -t =kisoku-eval4 2>/dev/null || tmux has-session -t =kisoku-eval5 2>/dev/null; do sleep 60; done
echo "=== BATCH 6 (RULER 32K) START $(date -u)" >> $LOG
run() { local name=$1 pre=$2 maxlen=$3
  if ls $R/$name/*/results_*.json >/dev/null 2>&1; then echo "--- skip $name (done)" >> $LOG; return; fi
  echo "--- $(date -u) $name ruler maxlen=$maxlen" >> $LOG
  lm_eval --model hf --model_args "pretrained=$pre,dtype=bfloat16,max_length=$maxlen" --tasks ruler \
    --metadata "{\"max_seq_lengths\":[4096,8192,16384,32768]}" --limit 100 --batch_size 1 --output_path "$R/$name" >> $LOG 2>&1
  local rc=$?
  ls $R/$name/*/results_*.json >/dev/null 2>&1 && echo "--- $(date -u) $name done rc=$rc" >> $LOG || echo "--- $(date -u) $name FAILED rc=$rc (no results file)" >> $LOG
}
run "kisoku-longA-ruler32k" "$M/kisoku-1.6b-base-longA-2500" 32768
run "llama-3.2-1b-ruler32k" "unsloth/Llama-3.2-1B" 32768
run "qwen3-1.7b-ruler32k" "Qwen/Qwen3-1.7B-Base" 32768
run "qwen3-0.6b-ruler32k" "Qwen/Qwen3-0.6B-Base" 32768
run "gemma-3-1b-ruler32k" "google/gemma-3-1b-pt" 32768
echo "=== BATCH 6 DONE $(date -u)" >> $LOG

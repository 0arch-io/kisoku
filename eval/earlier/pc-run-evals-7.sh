#!/bin/bash
# Batch 7 (2026-10-01): RULER 4K-32K preview rerun. Batch 6 failed in seconds on every model: wonderwords + nltk were missing (now installed).
export PATH=/usr/lib/wsl/lib:$HOME/ai/bin:$PATH
export TOKENIZERS_PARALLELISM=false
R=~/kisoku-eval/results; M=~/kisoku-eval/models; LOG=~/logs/kisoku-eval.log
echo "=== BATCH 7 (RULER 32K rerun) START $(date -u)" >> $LOG
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
run "lfm2.5-1.2b-ruler32k" "LiquidAI/LFM2.5-1.2B-Base" 32768
run "qwen3.5-0.8b-ruler32k" "Qwen/Qwen3.5-0.8B-Base" 32768
run "qwen3.5-2b-ruler32k" "Qwen/Qwen3.5-2B-Base" 32768
echo "=== BATCH 7 DONE $(date -u)" >> $LOG

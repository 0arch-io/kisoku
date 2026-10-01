#!/bin/bash
# Batch 4 (2026-09-30): Gemma-3-1B (gated; HF token installed at ~/.cache/huggingface/token), all sets, after batch 3.
export PATH=/usr/lib/wsl/lib:$HOME/ai/bin:$PATH
export HF_ALLOW_CODE_EVAL=1 TOKENIZERS_PARALLELISM=false
R=~/kisoku-eval/results; LOG=~/logs/kisoku-eval.log
while tmux has-session -t =kisoku-eval3 2>/dev/null; do sleep 60; done
echo "=== BATCH 4 START $(date -u)" >> $LOG
run() { local name=$1 pre=$2 tasks=$3 fs=$4 bs=$5; shift 5
  if ls $R/$name/*/results_*.json >/dev/null 2>&1; then echo "--- skip $name (done)" >> $LOG; return; fi
  echo "--- $(date -u) $name tasks=$tasks fewshot=$fs bs=$bs" >> $LOG
  local fsarg=(); [ "$fs" != "default" ] && fsarg=(--num_fewshot "$fs")
  lm_eval --model hf --model_args "pretrained=$pre,dtype=bfloat16" --tasks "$tasks" "${fsarg[@]}" \
    --batch_size "$bs" --output_path "$R/$name" "$@" >> $LOG 2>&1
  local rc=$?
  ls $R/$name/*/results_*.json >/dev/null 2>&1 && echo "--- $(date -u) $name done rc=$rc" >> $LOG || echo "--- $(date -u) $name FAILED rc=$rc (no results file)" >> $LOG
}
name=gemma-3-1b; pre=google/gemma-3-1b-pt
run "$name-core" "$pre" "hellaswag,arc_easy,arc_challenge,piqa,winogrande" 0 auto
run "$name-gsm8k" "$pre" "gsm8k" 5 16
run "$name-humaneval" "$pre" "humaneval" 0 16 --confirm_run_unsafe_code
run "$name-mmlu" "$pre" "mmlu" 5 auto
run "$name-triviaqa" "$pre" "triviaqa" 5 16
run "$name-bbh" "$pre" "bbh_fewshot" default 16
run "$name-drop" "$pre" "drop" 3 16
echo "=== BATCH 4 DONE $(date -u)" >> $LOG

#!/bin/bash
# Batch 5 (2026-09-30): stage-1 MMLU with a fixed batch (auto crashed twice), and BBH with the whitespace fix (bbh_ws) for all models.
export PATH=/usr/lib/wsl/lib:$HOME/ai/bin:$PATH
export HF_ALLOW_CODE_EVAL=1 TOKENIZERS_PARALLELISM=false
R=~/kisoku-eval/results; M=~/kisoku-eval/models; LOG=~/logs/kisoku-eval.log
while tmux has-session -t =kisoku-eval3 2>/dev/null || tmux has-session -t =kisoku-eval4 2>/dev/null; do sleep 60; done
echo "=== BATCH 5 START $(date -u)" >> $LOG
run() { local name=$1 pre=$2 tasks=$3 fs=$4 bs=$5; shift 5
  if ls $R/$name/*/results_*.json >/dev/null 2>&1; then echo "--- skip $name (done)" >> $LOG; return; fi
  echo "--- $(date -u) $name tasks=$tasks fewshot=$fs bs=$bs" >> $LOG
  local fsarg=(); [ "$fs" != "default" ] && fsarg=(--num_fewshot "$fs")
  lm_eval --model hf --model_args "pretrained=$pre,dtype=bfloat16" --include_path ~/kisoku-eval/tasks --tasks "$tasks" "${fsarg[@]}" \
    --batch_size "$bs" --output_path "$R/$name" "$@" >> $LOG 2>&1
  local rc=$?
  ls $R/$name/*/results_*.json >/dev/null 2>&1 && echo "--- $(date -u) $name done rc=$rc" >> $LOG || echo "--- $(date -u) $name FAILED rc=$rc (no results file)" >> $LOG
}
run "kisoku-s1-mmlu" "$M/kisoku-1.6b-base-s1-198999" "mmlu" 5 16
for pair in "kisoku-s3:$M/kisoku-1.6b-base-s3-242999" "kisoku-s1:$M/kisoku-1.6b-base-s1-198999" "llama-3.2-1b:unsloth/Llama-3.2-1B" "smollm2-1.7b:HuggingFaceTB/SmolLM2-1.7B" "qwen2.5-1.5b:Qwen/Qwen2.5-1.5B" "gemma-3-1b:google/gemma-3-1b-pt"; do
  name=${pair%%:*}; pre=${pair#*:}
  run "$name-bbhws" "$pre" "bbh_ws" default 16
done
echo "=== BATCH 5 DONE $(date -u)" >> $LOG

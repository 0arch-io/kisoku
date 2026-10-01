#!/bin/bash
# Kisoku base-model eval sweep v2 (2026-09-30). Skips any set that already has a results file.
# Generation tasks (gsm8k, humaneval) use a FIXED batch size: --batch_size auto probes max-length batches
# for generate_until, which overflowed the 4090 under WSL ("CUDA error: device not ready") and wasted 16-50 min.
export PATH=/usr/lib/wsl/lib:$HOME/ai/bin:$PATH
export HF_ALLOW_CODE_EVAL=1 TOKENIZERS_PARALLELISM=false
R=~/kisoku-eval/results; M=~/kisoku-eval/models; LOG=~/logs/kisoku-eval.log
echo "=== V2 START $(date -u)" >> $LOG
run() { # name, pretrained, tasks, fewshot, batch, extra...
  local name=$1 pre=$2 tasks=$3 fs=$4 bs=$5; shift 5
  if ls $R/$name/*/results_*.json >/dev/null 2>&1; then echo "--- skip $name (done)" >> $LOG; return; fi
  echo "--- $(date -u) $name tasks=$tasks fewshot=$fs bs=$bs" >> $LOG
  lm_eval --model hf --model_args "pretrained=$pre,dtype=bfloat16" --tasks "$tasks" --num_fewshot "$fs" \
    --batch_size "$bs" --output_path "$R/$name" --log_samples "$@" >> $LOG 2>&1
  local rc=$?
  ls $R/$name/*/results_*.json >/dev/null 2>&1 && echo "--- $(date -u) $name done rc=$rc" >> $LOG || echo "--- $(date -u) $name FAILED rc=$rc (no results file)" >> $LOG
}
for pair in "kisoku-s3:$M/kisoku-1.6b-base-s3-242999" "kisoku-s1:$M/kisoku-1.6b-base-s1-198999" "llama-3.2-1b:unsloth/Llama-3.2-1B" "smollm2-1.7b:HuggingFaceTB/SmolLM2-1.7B" "gemma-3-1b:google/gemma-3-1b-pt" "qwen2.5-1.5b:Qwen/Qwen2.5-1.5B"; do
  name=${pair%%:*}; pre=${pair#*:}
  run "$name-core" "$pre" "hellaswag,arc_easy,arc_challenge,piqa,winogrande" 0 auto
  run "$name-gsm8k" "$pre" "gsm8k" 5 32
  run "$name-humaneval" "$pre" "humaneval" 0 32 --confirm_run_unsafe_code
  run "$name-mmlu" "$pre" "mmlu" 5 auto
done
echo "=== V2 DONE $(date -u)" >> $LOG

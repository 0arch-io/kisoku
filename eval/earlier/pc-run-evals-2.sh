#!/bin/bash
# Second reference batch (Qwen2.5-1.5B, Gemma-3-1B). Waits for the first sweep to finish, same sets.
export PATH=/usr/lib/wsl/lib:$HOME/ai/bin:$PATH
export HF_ALLOW_CODE_EVAL=1 TOKENIZERS_PARALLELISM=false
R=~/kisoku-eval/results; LOG=~/logs/kisoku-eval.log
while tmux has-session -t kisoku-eval 2>/dev/null; do sleep 60; done
echo "=== BATCH 2 START $(date -u)" >> $LOG
run() { local name=$1 pre=$2 tasks=$3 fs=$4; shift 4
  echo "--- $(date -u) $name tasks=$tasks fewshot=$fs" >> $LOG
  lm_eval --model hf --model_args "pretrained=$pre,dtype=bfloat16" --tasks "$tasks" --num_fewshot "$fs" \
    --batch_size auto --output_path "$R/$name" --log_samples "$@" >> $LOG 2>&1
  echo "--- $(date -u) $name done rc=$?" >> $LOG; }
for pair in "qwen2.5-1.5b:Qwen/Qwen2.5-1.5B" "gemma-3-1b:google/gemma-3-1b-pt"; do
  name=${pair%%:*}; pre=${pair#*:}
  run "$name-core" "$pre" "hellaswag,arc_easy,arc_challenge,piqa,winogrande" 0
  run "$name-mmlu" "$pre" "mmlu" 5
  run "$name-gsm8k" "$pre" "gsm8k" 5
  run "$name-humaneval" "$pre" "humaneval" 0 --confirm_run_unsafe_code
done
echo "=== BATCH 2 DONE $(date -u)" >> $LOG

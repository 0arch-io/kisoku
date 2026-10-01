#!/bin/bash
# Kisoku base-model eval on the 4090: same lm-eval set for each model, results in ~/kisoku-eval/results/<name>/.
# Models are copied from /mnt/x (slow NTFS mount) to the ext4 home first so loading isn't the bottleneck.
export PATH=/usr/lib/wsl/lib:$HOME/ai/bin:$PATH
export HF_ALLOW_CODE_EVAL=1 TOKENIZERS_PARALLELISM=false
R=~/kisoku-eval/results; M=~/kisoku-eval/models; mkdir -p $R $M ~/logs
LOG=~/logs/kisoku-eval.log
echo "=== START $(date -u)" >> $LOG
for n in kisoku-1.6b-base-s3-242999 kisoku-1.6b-base-s1-198999; do
  [ -f $M/$n/model.safetensors ] || cp -r /mnt/x/WSL/kisoku-hf/$n $M/
done
run() { # name, pretrained, tasks, fewshot, extra
  local name=$1 pre=$2 tasks=$3 fs=$4; shift 4
  echo "--- $(date -u) $name tasks=$tasks fewshot=$fs" >> $LOG
  lm_eval --model hf --model_args "pretrained=$pre,dtype=bfloat16" --tasks "$tasks" --num_fewshot "$fs" \
    --batch_size auto --output_path "$R/$name" --log_samples "$@" >> $LOG 2>&1
  echo "--- $(date -u) $name done rc=$?" >> $LOG
}
for pair in "kisoku-s3:$M/kisoku-1.6b-base-s3-242999" "kisoku-s1:$M/kisoku-1.6b-base-s1-198999" "llama-3.2-1b:unsloth/Llama-3.2-1B" "smollm2-1.7b:HuggingFaceTB/SmolLM2-1.7B"; do
  name=${pair%%:*}; pre=${pair#*:}
  run "$name-core" "$pre" "hellaswag,arc_easy,arc_challenge,piqa,winogrande" 0
  run "$name-mmlu" "$pre" "mmlu" 5
  run "$name-gsm8k" "$pre" "gsm8k" 5
  run "$name-humaneval" "$pre" "humaneval" 0 --confirm_run_unsafe_code
done
echo "=== DONE $(date -u)" >> $LOG

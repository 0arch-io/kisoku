#!/bin/bash
# Batch 16 (2026-10-06): the released chat model (pass 9, kisoku-1.6b-chat-sft009) through the short-eval suite on the CTI box, so the
# report's chat benchmark table matches the model that ships (the table was pass 5). HumanEval stays on the 4090 (executes code).
# usage: cti-run-evals-16.sh STREAM ACCESS_TOKEN   (H9 on card 2: core, gsm8k, bbh; I9 on card 1: mmlu, triviaqa)
cd /models/kisoku-eval; export PATH=/models/kisoku-eval/venv/bin:$PATH HF_HOME=/models/kisoku-eval/hf TOKENIZERS_PARALLELISM=false
R=results; M=/models/kisoku-eval/models; LOG=logs/stream-$1.log; C9=$M/kisoku-1.6b-chat-sft009
if [ ! -f $C9/model.safetensors ]; then mkdir -p $C9; for f in config.json generation_config.json tokenizer.json tokenizer_config.json special_tokens_map.json chat_template.jinja model.safetensors; do
  curl -sf -H "Authorization: Bearer $2" -o $C9/$f "https://storage.googleapis.com/kisoku-v2-training/hf/kisoku-1.6b-chat-sft009/$f" || { echo "download failed: $f" >> $LOG; exit 1; }; done; fi
run() { local name=$1 pre=$2 tasks=$3 fs=$4 bs=$5; shift 5
  ls $R/$name/*/results_*.json >/dev/null 2>&1 && { echo "--- skip $name" >> $LOG; return; }
  echo "--- $(date -u) $name tasks=$tasks fewshot=$fs bs=$bs" >> $LOG
  local fsarg=(); [ "$fs" != "default" ] && fsarg=(--num_fewshot "$fs")
  lm_eval --model hf --model_args "pretrained=$pre,dtype=bfloat16" --include_path /models/kisoku-eval/tasks --tasks "$tasks" "${fsarg[@]}" \
    --batch_size "$bs" --output_path "$R/$name" --log_samples "$@" >> $LOG 2>&1
  ls $R/$name/*/results_*.json >/dev/null 2>&1 && echo "--- $(date -u) $name done" >> $LOG || echo "--- $(date -u) $name FAILED" >> $LOG
}
echo "=== STREAM $1 START $(date -u)" >> $LOG
case $1 in
  H9) export CUDA_VISIBLE_DEVICES=2; run kisoku-chat9-core $C9 "hellaswag,arc_easy,arc_challenge,piqa,winogrande" 0 16; run kisoku-chat9-gsm8k $C9 gsm8k 5 16; run kisoku-chat9-bbhws $C9 bbh_ws default 16 ;;
  I9) export CUDA_VISIBLE_DEVICES=1; run kisoku-chat9-mmlu $C9 mmlu 5 8; run kisoku-chat9-triviaqa $C9 triviaqa 5 16 ;;
esac
echo "=== STREAM $1 DONE $(date -u)" >> $LOG

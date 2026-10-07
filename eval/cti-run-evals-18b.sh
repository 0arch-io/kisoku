#!/bin/bash
# Batch 18 on the CTI box (2026-10-07, the 4090 went offline overnight): the final long-context checkpoint (Phase C 1299, plain config)
# on the short-eval suite (no HumanEval here: it executes generated code, that one stays on the 4090), plain-config RULER at 128K,
# and the Qwen3 1.7B YaRN x2 RULER 64K job that died with the PC. Both streams on card 2: the host ai-guard timer freezes us
# (SIGSTOP) when card 1 hits 86 C or Jarvis gets slow, and card 1 is where 3 of 4 past freezes came from. usage: cti-run-evals-18.sh STREAM
cd /models/kisoku-eval; export PATH=/models/kisoku-eval/venv/bin:$PATH HF_HOME=/models/kisoku-eval/hf TOKENIZERS_PARALLELISM=false
R=results; M=/models/kisoku-eval/models; LOG=logs/stream-$1.log; P=$M/kisoku-1.6b-base-longC-1299
run() { local name=$1 pre=$2 tasks=$3 fs=$4 bs=$5; shift 5
  ls $R/$name/*/results_*.json >/dev/null 2>&1 && { echo "--- skip $name" >> $LOG; return; }
  echo "--- $(date -u) $name tasks=$tasks fewshot=$fs bs=$bs" >> $LOG
  local fsarg=(); [ "$fs" != "default" ] && fsarg=(--num_fewshot "$fs")
  lm_eval --model hf --model_args "pretrained=$pre,dtype=bfloat16" --include_path /models/kisoku-eval/tasks --tasks "$tasks" "${fsarg[@]}" \
    --batch_size "$bs" --output_path "$R/$name" --log_samples "$@" >> $LOG 2>&1
  ls $R/$name/*/results_*.json >/dev/null 2>&1 && echo "--- $(date -u) $name done" >> $LOG || echo "--- $(date -u) $name FAILED" >> $LOG
}
ruler() { local name=$1 pre=$2 len=$3 lim=$4 out="$R/$1-ruler-$3"
  ls $out/*/results_*.json >/dev/null 2>&1 && { echo "--- skip $name $len" >> $LOG; return; }
  echo "--- $(date -u) $name ruler len=$len limit=$lim" >> $LOG
  lm_eval --model hf --model_args "pretrained=$pre,dtype=bfloat16,max_length=$len" --tasks ruler --metadata "{\"max_seq_lengths\":[$len]}" \
    --limit $lim --batch_size 1 --gen_kwargs max_gen_toks=128,max_new_tokens=128 --output_path "$out" >> $LOG 2>&1
  ls $out/*/results_*.json >/dev/null 2>&1 && echo "--- $(date -u) $name $len done" >> $LOG || echo "--- $(date -u) $name $len FAILED" >> $LOG
}
echo "=== STREAM $1 START $(date -u)" >> $LOG
case $1 in
  S) export CUDA_VISIBLE_DEVICES=2; run kisoku-longC-core $P "hellaswag,arc_easy,arc_challenge,piqa,winogrande" 0 16; run kisoku-longC-gsm8k $P gsm8k 5 16
     run kisoku-longC-bbhws $P bbh_ws default 16; run kisoku-longC-mmlu $P mmlu 5 8; run kisoku-longC-triviaqa $P triviaqa 5 16 ;;
  T) export CUDA_VISIBLE_DEVICES=2; ruler qwen3-1.7b-yarn2 $M/qwen3-1.7b-yarn2 65536 50 ;;
  U) export CUDA_VISIBLE_DEVICES=2; ruler kisoku-longC-ctiplain $P 131072 50 ;;
esac
echo "=== STREAM $1 DONE $(date -u)" >> $LOG

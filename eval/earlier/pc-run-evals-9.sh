#!/bin/bash
# Batch 9 (2026-10-01): rerun Kisoku's GENERATION evals with a clean generation config.
# Why: our HF exports shipped chat defaults in generation_config.json (do_sample, temperature 0.7, top_p 0.9, repetition_penalty 1.1).
# lm_eval forces greedy but the repetition penalty stayed on, and no baseline has one. It penalises every token already in the
# prompt, i.e. copying from the context, which is what RULER / QA / math need. Loglikelihood tasks (mmlu, arc, ...) are unaffected.
# Old tainted results are kept: <name> (gsm8k etc.) and _reppen-kisoku-longA-ruler-<len>. New ones: <name>-clean.
export PATH=/usr/lib/wsl/lib:$HOME/ai/bin:$PATH
export HF_ALLOW_CODE_EVAL=1 TOKENIZERS_PARALLELISM=false
R=~/kisoku-eval/results; M=~/kisoku-eval/models; LOG=~/logs/kisoku-eval.log
for d in $M/kisoku-1.6b-base-*; do
  [ -f $d/generation_config.chat.json ] || cp $d/generation_config.json $d/generation_config.chat.json
  echo '{"bos_token_id": 128000, "eos_token_id": 128001, "pad_token_id": 128001}' > $d/generation_config.json
done
for L in 4096 8192; do [ -d $R/kisoku-longA-ruler-$L ] && mv $R/kisoku-longA-ruler-$L $R/_reppen-kisoku-longA-ruler-$L; done
echo "=== BATCH 9 (clean gen config reruns) START $(date -u)" >> $LOG
run() { local name=$1 pre=$2 tasks=$3 fs=$4 bs=$5; shift 5
  if ls $R/$name/*/results_*.json >/dev/null 2>&1; then echo "--- skip $name (done)" >> $LOG; return; fi
  echo "--- $(date -u) $name tasks=$tasks fewshot=$fs bs=$bs" >> $LOG
  local fsarg=(); [ "$fs" != "default" ] && fsarg=(--num_fewshot "$fs")
  lm_eval --model hf --model_args "pretrained=$pre,dtype=bfloat16" --include_path ~/kisoku-eval/tasks --tasks "$tasks" "${fsarg[@]}" \
    --batch_size "$bs" --output_path "$R/$name" --log_samples "$@" >> $LOG 2>&1
  ls $R/$name/*/results_*.json >/dev/null 2>&1 && echo "--- $(date -u) $name done" >> $LOG || echo "--- $(date -u) $name FAILED (no results file)" >> $LOG
}
ruler() { local name=$1 pre=$2 len=$3 lim=$4 out="$R/$1-ruler-$3"
  ls $out/*/results_*.json >/dev/null 2>&1 && return
  for try in 1 2 3; do
    echo "--- $(date -u) $name ruler len=$len limit=$lim try=$try (batch 9)" >> $LOG
    lm_eval --model hf --model_args "pretrained=$pre,dtype=bfloat16,max_length=32768" --tasks ruler --metadata "{\"max_seq_lengths\":[$len]}" --limit $lim --batch_size 1 --gen_kwargs max_gen_toks=128 --output_path "$out" >> $LOG 2>&1
    ls $out/*/results_*.json >/dev/null 2>&1 && { echo "--- $(date -u) $name $len done" >> $LOG; return; }; sleep 60
  done
}
for pair in "kisoku-s3:$M/kisoku-1.6b-base-s3-242999" "kisoku-s1:$M/kisoku-1.6b-base-s1-198999" "qwen2.5-1.5b:Qwen/Qwen2.5-1.5B"; do
  name=${pair%%:*}; pre=${pair#*:}
  run "$name-gsm8k-clean" "$pre" "gsm8k" 5 32
  run "$name-humaneval-clean" "$pre" "humaneval" 0 32 --confirm_run_unsafe_code
  run "$name-bbhws-clean" "$pre" "bbh_ws" default 16
  run "$name-triviaqa-clean" "$pre" "triviaqa" 5 16
done
ruler kisoku-longA "$M/kisoku-1.6b-base-longA-2500" 4096 100
ruler kisoku-longA "$M/kisoku-1.6b-base-longA-2500" 8192 100
echo "=== BATCH 9 DONE $(date -u)" >> $LOG

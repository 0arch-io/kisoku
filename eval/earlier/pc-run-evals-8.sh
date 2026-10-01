#!/bin/bash
# Batch 8 (2026-10-01): RULER preview, ONE LENGTH PER RUN. Why: lm_eval orders ruler docs by length, so --limit N with a list of
# lengths only ever scores the shortest one. Base models also never stop generating (2048 new tokens, ~30 s/sample), so every model
# gets the same cap max_gen_toks=128 (RULER's own caps are 30-128). 100 samples per task at 4K/8K, 50 at 16K/32K.
export PATH=/usr/lib/wsl/lib:$HOME/ai/bin:$PATH
export TOKENIZERS_PARALLELISM=false
R=~/kisoku-eval/results; M=~/kisoku-eval/models; LOG=~/logs/kisoku-eval.log
echo "=== BATCH 8 (RULER per length) START $(date -u)" >> $LOG
run() { local name=$1 pre=$2 len=$3 lim=$4 out="$R/$1-ruler-$3"
  if ls $out/*/results_*.json >/dev/null 2>&1; then echo "--- skip $name $len (done)" >> $LOG; return; fi
  for try in 1 2 3; do
    echo "--- $(date -u) $name ruler len=$len limit=$lim try=$try" >> $LOG
    lm_eval --model hf --model_args "pretrained=$pre,dtype=bfloat16,max_length=32768" --tasks ruler \
      --metadata "{\"max_seq_lengths\":[$len]}" --limit $lim --batch_size 1 --gen_kwargs max_gen_toks=128 --output_path "$out" >> $LOG 2>&1
    ls $out/*/results_*.json >/dev/null 2>&1 && { echo "--- $(date -u) $name $len done" >> $LOG; return; }
    sleep 60
  done
  echo "--- $(date -u) $name $len FAILED (no results file)" >> $LOG
}
for spec in "4096 100" "8192 100" "16384 50" "32768 50"; do set -- $spec
  run kisoku-longA "$M/kisoku-1.6b-base-longA-2500" $1 $2
  run llama-3.2-1b "unsloth/Llama-3.2-1B" $1 $2
  run qwen3-1.7b "Qwen/Qwen3-1.7B-Base" $1 $2
  run qwen3-0.6b "Qwen/Qwen3-0.6B-Base" $1 $2
  run gemma-3-1b "google/gemma-3-1b-pt" $1 $2
  run lfm2.5-1.2b "LiquidAI/LFM2.5-1.2B-Base" $1 $2
  run qwen3.5-0.8b "Qwen/Qwen3.5-0.8B-Base" $1 $2
  run qwen3.5-2b "Qwen/Qwen3.5-2B-Base" $1 $2
done
echo "=== BATCH 8 DONE $(date -u)" >> $LOG

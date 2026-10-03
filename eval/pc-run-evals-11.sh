#!/bin/bash
# Batch 11 (2026-10-03): RULER at 64K and 128K for the 9 baselines, ONE LENGTH PER RUN, one model at a time, nothing else on the GPU.
# Same settings as batch 8 (bs 1, 128 new tokens, 50 samples per task); max_length follows the tested length.
# Order: models with a native window >= 64K first (Granite is the bar), then the 32K-native ones, which are run past their
# trained range on purpose so the table shows a measured number instead of an assumption. The whole 64K round runs before 128K.
# An out-of-memory run fails fast and is logged as FAILED; 2 tries each.
export PATH=/usr/lib/wsl/lib:$HOME/ai/bin:$PATH
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=~/kisoku-eval/results; M=~/kisoku-eval/models; LOG=~/logs/kisoku-eval.log
echo "=== BATCH 11 (RULER 64K/128K baselines) START $(date -u)" >> $LOG
run() { local name=$1 pre=$2 len=$3 lim=$4 out="$R/$1-ruler-$3"
  if ls $out/*/results_*.json >/dev/null 2>&1; then echo "--- skip $name $len (done)" >> $LOG; return; fi
  for try in 1 2; do
    echo "--- $(date -u) $name ruler len=$len limit=$lim try=$try" >> $LOG
    lm_eval --model hf --model_args "pretrained=$pre,dtype=bfloat16,max_length=$len" --tasks ruler \
      --metadata "{\"max_seq_lengths\":[$len]}" --limit $lim --batch_size 1 --gen_kwargs max_gen_toks=128,max_new_tokens=128 --output_path "$out" >> $LOG 2>&1
    ls $out/*/results_*.json >/dev/null 2>&1 && { echo "--- $(date -u) $name $len done" >> $LOG; return; }
    sleep 60
  done
  echo "--- $(date -u) $name $len FAILED (no results file)" >> $LOG
}
for spec in "65536 50" "131072 50"; do set -- $spec
  run granite-4.0-1b "ibm-granite/granite-4.0-1b-base" $1 $2
  run qwen3.5-2b "Qwen/Qwen3.5-2B-Base" $1 $2
  run qwen3.5-0.8b "Qwen/Qwen3.5-0.8B-Base" $1 $2
  run llama-3.2-1b "unsloth/Llama-3.2-1B" $1 $2
  run qwen3-1.7b "Qwen/Qwen3-1.7B-Base" $1 $2
  run qwen3-0.6b "Qwen/Qwen3-0.6B-Base" $1 $2
  run lfm2.5-1.2b "LiquidAI/LFM2.5-1.2B-Base" $1 $2
  run gemma-3-1b "google/gemma-3-1b-pt" $1 $2
done
echo "=== BATCH 11 DONE $(date -u)" >> $LOG

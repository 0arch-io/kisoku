#!/bin/bash
# Batch 10 (2026-10-02): Granite 4.0 1B base and Falcon-H1 1.5B base, same settings as batch 8.
# Batch 8 notes (2026-10-01): RULER preview, ONE LENGTH PER RUN. Why: lm_eval orders ruler docs by length, so --limit N with a list of
# lengths only ever scores the shortest one. Base models also never stop generating (2048 new tokens, ~30 s/sample), so every model
# gets the same cap max_gen_toks=128 (RULER's own caps are 30-128). 100 samples per task at 4K/8K, 50 at 16K/32K.
# max_new_tokens=128 is passed too: Qwen3 generation_config.json sets max_new_tokens=2048, which beats lm_eval max_length (17 s/sample).
export PATH=/usr/lib/wsl/lib:$HOME/ai/bin:$PATH
export TOKENIZERS_PARALLELISM=false
R=~/kisoku-eval/results; M=~/kisoku-eval/models; LOG=~/logs/kisoku-eval.log
echo "=== BATCH 10 (RULER, two same-size 128K baselines) START $(date -u)" >> $LOG
run() { local name=$1 pre=$2 len=$3 lim=$4 out="$R/$1-ruler-$3"
  if ls $out/*/results_*.json >/dev/null 2>&1; then echo "--- skip $name $len (done)" >> $LOG; return; fi
  for try in 1 2 3; do
    echo "--- $(date -u) $name ruler len=$len limit=$lim try=$try" >> $LOG
    lm_eval --model hf --model_args "pretrained=$pre,dtype=bfloat16,max_length=32768" --tasks ruler \
      --metadata "{\"max_seq_lengths\":[$len]}" --limit $lim --batch_size 1 --gen_kwargs max_gen_toks=128,max_new_tokens=128 --output_path "$out" >> $LOG 2>&1
    ls $out/*/results_*.json >/dev/null 2>&1 && { echo "--- $(date -u) $name $len done" >> $LOG; return; }
    sleep 60
  done
  echo "--- $(date -u) $name $len FAILED (no results file)" >> $LOG
}
for spec in "4096 100" "8192 100" "16384 50" "32768 50"; do set -- $spec
  run granite-4.0-1b "ibm-granite/granite-4.0-1b-base" $1 $2
  run falcon-h1-1.5b "tiiuae/Falcon-H1-1.5B-Base" $1 $2
done
echo "=== BATCH 10 DONE $(date -u)" >> $LOG

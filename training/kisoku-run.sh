#!/bin/bash
# Kisoku v2 training entrypoint. Stage/run/steps come from ~/kisoku-stage.env so
# moving to the next stage is an edit of that file plus a service restart.
: "${KISOKU_BUCKET:?set KISOKU_BUCKET to your GCS bucket name (no gs:// prefix)}"
set -u
export PATH="$HOME/.local/bin:$PATH"
source "$HOME/kisoku-stage.env"

D="$HOME/gcsfuse/datasets"
NEM="$D/nemotron-cc-v2.1-hqs/*.arrayrecord"
UFW="$D/ultra-fineweb-en-text/*.arrayrecord"
SC="$D/starcoderdata-text/*.arrayrecord"
FM="$D/finemath-4plus/*.arrayrecord"
OWM="$D/open-web-math/*.arrayrecord"
MWP="$D/megamath-web-pro/*.arrayrecord"
OT="$D/openthoughts3-text/*.arrayrecord"
NCM="$D/nemotron-cc-math-4plus/*.arrayrecord"
# Long-context sources (2026-09-23): avg tokens/record code 65,460 / science 38,889 / pg19 102,114.
LCODE="$D/longctx-code-repos/*.arrayrecord"
LSCI="$D/longctx-science-pdfs/*.arrayrecord"
LPG="$D/longctx-pg19/*.arrayrecord"
# Synthetic long-context tasks (2026-10-01): 22,386 docs, avg 25,709 tokens, all <= 60K tokens (questions never split from their document).
LSYN="$D/longctx-synth-tasks/*.arrayrecord"

# Long-context phases (A=32K, B=64K), 2026-09-30. Mix weights pick DOCUMENTS, so the
# 60% long / 40% short TOKEN split is converted with avg tokens per record: short stage-3
# mix ~1,000 tok/doc (OT ~12K, others ~800), code 65K, science 39K, pg19 102K.
# Token shares: short 0.40, code 0.20, science 0.22, pg19 0.18 -> doc weights below.
# Short part = stage-3 weights scaled by 0.9263 (sum 0.9745); long appended LAST.
LONGMIX="$NEM,0.250;$UFW,0.259;$SC,0.204;$FM,0.046;$OWM,0.028;$MWP,0.046;$OT,0.0111;$NCM,0.130;$LCODE,0.00744;$LSCI,0.01378;$LPG,0.00429"
STAGE3_FINAL="gs://${KISOKU_BUCKET}/runs/kisoku-v2-1b-stage3-final/242999/items"
PHASE_A_FINAL="gs://${KISOKU_BUCKET}/runs/kisoku-v2-1b-longctx-a-final/items"
PHASE_B_FINAL="gs://${KISOKU_BUCKET}/runs/kisoku-v2-1b-longctx-b-final/items"
# Phase C mix. Token shares: short 0.40, code 0.15, science 0.15, pg19 0.12, synthetic tasks 0.18 -> document weights
# (share / avg tokens per doc, normalised). Short part = stage-3 weights x 0.9177. Synthetic source appended LAST.
CMIX="$NEM,0.2478;$UFW,0.2570;$SC,0.2019;$FM,0.0459;$OWM,0.0275;$MWP,0.0459;$OT,0.01101;$NCM,0.1285;$LCODE,0.005531;$LSCI,0.009309;$LPG,0.002836;$LSYN,0.016898"

EXTRA=""
PDBS=2; GA=16; LR_ARGS=""
case "$STAGE" in
  1) MIX="$NEM,0.35;$UFW,0.35;$SC,0.16;$FM,0.06;$OWM,0.04;$MWP,0.04" ;;
  2) MIX="$NEM,0.30;$UFW,0.30;$SC,0.22;$FM,0.07;$OWM,0.05;$MWP,0.06" ;;
  # Stage 3 (2026-09-27): OT weight is by documents and OT docs are ~10x longer, so 0.012 ~= 15% of tokens.
  # NCM (Nemotron-CC-Math 4plus) is appended LAST: grain restores mix sources by position.
  # use_truncation=False keeps long docs (chunks them) instead of cutting at 4096; the 223999
  # iter state was converted to include the FlatMap layer this adds (backup: runs/kisoku-v2-1b-stage2-iter-backup).
  3) MIX="$NEM,0.27;$UFW,0.28;$SC,0.22;$FM,0.05;$OWM,0.03;$MWP,0.05;$OT,0.012;$NCM,0.14"; EXTRA="use_truncation=False" ;;
  # Phase A (32K): fresh run + fresh optimizer from the stage-3 final params. 16 devices x 1 x 32768 x GA 4
  # = 2.1M tok/step, 2,900 steps ~= 6B tokens, ~38.5 s/step (smoke 09-27: 13.0 GB/device).
  # LR: warmup 2% to 1e-4 (= stage-3 end LR), cosine to 1e-5. Eval off: eval batch 8 x 32K would OOM.
  A) MIX="$LONGMIX"; PDBS=1; GA=4
     EXTRA="use_truncation=False max_target_length=32768 remat_policy=full num_vocab_tiling=8 attention=flash eval_interval=-1 load_parameters_path=$STAGE3_FINAL"
     LR_ARGS="lr_schedule_type=cosine learning_rate=1.0e-4 learning_rate_final_fraction=0.1 warmup_steps_fraction=0.02" ;;
  # Phase B (64K): same as A from the Phase A final params, GA 2, vocab tiling 16 (smoke: 22.5 GB/device, 65.8 s/step).
  # Fallback if OOM: ici_fsdp_parallelism=8 ici_context_parallelism=2 per_device_batch_size=0.5 context_parallel_strategy=all_gather.
  B) MIX="$LONGMIX"; PDBS=1; GA=2
     EXTRA="use_truncation=False max_target_length=65536 remat_policy=full num_vocab_tiling=16 attention=flash eval_interval=-1 load_parameters_path=$PHASE_A_FINAL"
     LR_ARGS="lr_schedule_type=cosine learning_rate=1.0e-4 learning_rate_final_fraction=0.1 warmup_steps_fraction=0.01" ;;
  # Phase C (64K + synthetic tasks, 2026-10-01): same memory settings as B, from the Phase B final params, fresh optimizer.
  # 1,300 steps ~= 2.7B tokens ~= 24 h. LR warmup 2% to 5e-5, cosine to 5e-6.
  C) MIX="$CMIX"; PDBS=1; GA=2
     EXTRA="use_truncation=False max_target_length=65536 remat_policy=full num_vocab_tiling=16 attention=flash eval_interval=-1 load_parameters_path=$PHASE_B_FINAL"
     LR_ARGS="lr_schedule_type=cosine learning_rate=5.0e-5 learning_rate_final_fraction=0.1 warmup_steps_fraction=0.02" ;;
  *) echo "bad stage $STAGE"; exit 1 ;;
esac

EV="$HOME/gcsfuse/datasets-eval"
EVMIX="$EV/nemotron-cc-v2.1-hqs/*.arrayrecord,0.35;$EV/ultra-fineweb-en-text/*.arrayrecord,0.35;$EV/starcoderdata-text/*.arrayrecord,0.16;$EV/finemath-4plus/*.arrayrecord,0.05;$EV/open-web-math/*.arrayrecord,0.04;$EV/megamath-web-pro/*.arrayrecord,0.05"

# Checkpoint cadence: 1000 steps for the 4K stages (~3.7 h), 250 for the long phases (~2.7 h at 32K, ~4.6 h at 64K).
CKPT_PERIOD=1000
case "$STAGE" in A|B|C) CKPT_PERIOD=250 ;; esac

# --- Restart guard (added 2026-09-04) ---
# If this unit keeps starting without a new checkpoint landing, the run is stuck
# replaying the same steps (seen 09-02..09-04: gcsfuse served cached zeros for one
# data file, MaxText exited "cleanly", systemd restarted, repeat for 30h). On the
# 3rd start with no progress: remount gcsfuse (drops the bad cache) and come back.
CK_DIR="$HOME/gcsfuse/runs/$RUN/checkpoints"
GUARD="$HOME/.kisoku-guard"
latest=$(ls "$CK_DIR" 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -n1)
prev=""; count=0
[ -f "$GUARD" ] && read -r prev count < "$GUARD"
if [ -n "$latest" ] && [ "$latest" = "$prev" ]; then count=$((count+1)); else count=1; fi
echo "$latest $count" > "$GUARD"
if [ "$count" -ge 3 ]; then
  echo "=== GUARD $(date -u): start #$count with no new checkpoint (still $latest); remounting gcsfuse, train restarts in 60s ===" >> "$HOME/logs/$RUN.log"
  echo "$latest 0" > "$GUARD"
  # Restarting gcsfuse stops this unit too (Requires=), so schedule both from outside this process.
  systemd-run --user --on-active=5  --unit="kisoku-guard-remount-$(date +%s)" systemctl --user restart kisoku-gcsfuse
  systemd-run --user --on-active=60 --unit="kisoku-guard-train-$(date +%s)"   systemctl --user start kisoku-train
  exit 0
fi

rm -f /tmp/libtpu_lockfile
cd "$HOME/maxtext" || exit 1
export LIBTPU_INIT_ARGS="--xla_enable_async_all_gather=true TPU_MEGACORE=MEGACORE_DENSE"
export GOOGLE_APPLICATION_CREDENTIALS="$HOME/gcs-key.json"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

echo "=== START $(date -u) stage=$STAGE run=$RUN steps=$STEPS on $(hostname) ===" >> "$HOME/logs/$RUN.log"
exec .venv/bin/python -u -m maxtext.trainers.pre_train.train "$HOME/kisoku-v2-1b.yml" \
  run_name="$RUN" \
  base_output_directory=gs://${KISOKU_BUCKET}/runs/ \
  ici_fsdp_parallelism=16 \
  per_device_batch_size=$PDBS \
  gradient_accumulation_steps=$GA \
  grain_train_files="$MIX" \
  grain_eval_files="$EVMIX" \
  gcs_metrics=False enable_tensorboard=False \
  steps="$STEPS" \
  learning_rate_schedule_steps="$STEPS" \
  checkpoint_period=$CKPT_PERIOD \
  enable_checkpointing=True \
  async_checkpointing=False \
  num_epoch=100 \
  grain_worker_count=8 \
  grain_worker_count_eval=2 \
  $LR_ARGS \
  $EXTRA \
  >> "$HOME/logs/$RUN.log" 2>&1

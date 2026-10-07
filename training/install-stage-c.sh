#!/bin/bash
# Adds stage C (64K + synthetic long-context tasks) to kisoku-run.sh and installs the B -> C hand-off watcher.
# Safe while Phase B trains: kisoku-run.sh exec's python, and the file is replaced atomically after bash -n.
: "${KISOKU_BUCKET:?set KISOKU_BUCKET to your GCS bucket name (no gs:// prefix)}"
set -u
W=$(hostname | grep -o 'w-[0-9]*$' | cut -d- -f2)
cd ~/bin || exit 1
if grep -q LSYN kisoku-run.sh; then echo "w$W: stage C already present"; else
cp kisoku-run.sh kisoku-run.sh.phaseB-bak
python3 - <<'PY'
s=open('kisoku-run.sh').read()
def ins(anchor, new, after=True):
    global s
    assert s.count(anchor)==1, anchor
    s=s.replace(anchor, anchor+new if after else new+anchor)
ins('LPG="$D/longctx-pg19/*.arrayrecord"\n', '# Synthetic long-context tasks (2026-10-01): 22,386 docs, avg 25,709 tokens, all <= 60K tokens (questions never split from their document).\nLSYN="$D/longctx-synth-tasks/*.arrayrecord"\n')
ins('PHASE_A_FINAL="gs://${KISOKU_BUCKET}/runs/kisoku-v2-1b-longctx-a-final/items"\n', '''PHASE_B_FINAL="gs://${KISOKU_BUCKET}/runs/kisoku-v2-1b-longctx-b-final/items"
# Phase C mix. Token shares: short 0.40, code 0.15, science 0.15, pg19 0.12, synthetic tasks 0.18 -> document weights
# (share / avg tokens per doc, normalised). Short part = stage-3 weights x 0.9177. Synthetic source appended LAST.
CMIX="$NEM,0.2478;$UFW,0.2570;$SC,0.2019;$FM,0.0459;$OWM,0.0275;$MWP,0.0459;$OT,0.01101;$NCM,0.1285;$LCODE,0.005531;$LSCI,0.009309;$LPG,0.002836;$LSYN,0.016898"
''')
ins('  *) echo "bad stage $STAGE"; exit 1 ;;\n', '''  # Phase C (64K + synthetic tasks, 2026-10-01): same memory settings as B, from the Phase B final params, fresh optimizer.
  # 1,300 steps ~= 2.7B tokens ~= 24 h. LR warmup 2% to 5e-5, cosine to 5e-6.
  C) MIX="$CMIX"; PDBS=1; GA=2
     EXTRA="use_truncation=False max_target_length=65536 remat_policy=full num_vocab_tiling=16 attention=flash eval_interval=-1 load_parameters_path=$PHASE_B_FINAL"
     LR_ARGS="lr_schedule_type=cosine learning_rate=5.0e-5 learning_rate_final_fraction=0.1 warmup_steps_fraction=0.02" ;;
''', after=False)
ins('case "$STAGE" in A|B) CKPT_PERIOD=250 ;; esac', '', after=True)
s=s.replace('case "$STAGE" in A|B) CKPT_PERIOD=250 ;; esac','case "$STAGE" in A|B|C) CKPT_PERIOD=250 ;; esac')
open('kisoku-run.sh.new','w').write(s)
PY
bash -n kisoku-run.sh.new && chmod +x kisoku-run.sh.new && mv kisoku-run.sh.new kisoku-run.sh && echo "w$W: kisoku-run.sh patched" || { echo "w$W: PATCH FAILED, original untouched"; exit 1; }
fi
cat > kisoku-next-stage-c.sh <<'SH'
#!/bin/bash
# Auto hand-off Phase B -> Phase C (2026-10-01). Same pattern as kisoku-next-stage.sh (A -> B).
# To cancel before it fires: systemctl --user stop kisoku-next-stage-c   (on every worker)
set -u
export PATH="$HOME/.local/bin:/snap/bin:$PATH"
LOG="$HOME/logs/next-stage.log"
RUN_B=kisoku-v2-1b-longctx-b; STEP_B=2859
FINAL_B="gs://${KISOKU_BUCKET}/runs/kisoku-v2-1b-longctx-b-final"
FINAL_B_FUSE="$HOME/gcsfuse/runs/kisoku-v2-1b-longctx-b-final"
W=$(hostname | grep -o 'w-[0-9]*$' | cut -d- -f2)
echo "=== $(date -u) next-stage-c watcher started on worker $W ===" >> "$LOG"
until [ -f "$HOME/gcsfuse/runs/$RUN_B/checkpoints/$STEP_B/commit_success.txt" ] && ! systemctl --user is-active --quiet kisoku-train; do sleep 60; done
echo "$(date -u) phase B done (ckpt $STEP_B present, trainer inactive)" >> "$LOG"
if [ "$W" = "0" ]; then
  export CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE="$HOME/gcs-key.json"
  if ! gcloud storage ls "$FINAL_B/commit_success.txt" >/dev/null 2>&1; then
    gcloud storage cp -r "gs://${KISOKU_BUCKET}/runs/$RUN_B/checkpoints/$STEP_B/*" "$FINAL_B/" >> "$LOG" 2>&1
    echo "$(date -u) copied $STEP_B -> $FINAL_B" >> "$LOG"
  fi
fi
until [ -f "$FINAL_B_FUSE/commit_success.txt" ] || ls "$FINAL_B_FUSE/items" >/dev/null 2>&1; do sleep 30; done
sleep 60
printf 'STAGE=C\nRUN=kisoku-v2-1b-longctx-c\nSTEPS=1300\n' > "$HOME/kisoku-stage.env"
rm -f "$HOME/.kisoku-guard"
systemctl --user reset-failed kisoku-train 2>/dev/null
systemctl --user start kisoku-train
echo "$(date -u) stage C started: $(systemctl --user is-active kisoku-train)" >> "$LOG"
SH
chmod +x kisoku-next-stage-c.sh; bash -n kisoku-next-stage-c.sh || exit 1
systemctl --user is-active --quiet kisoku-next-stage-c || systemd-run --user --unit=kisoku-next-stage-c bash "$HOME/bin/kisoku-next-stage-c.sh" 2>&1 | tail -1
sleep 2
echo "w$W: watcher=$(systemctl --user is-active kisoku-next-stage-c) train=$(systemctl --user is-active kisoku-train) stage=$(grep STAGE ~/kisoku-stage.env) synth_shards=$(ls ~/gcsfuse/datasets/longctx-synth-tasks/*.arrayrecord 2>/dev/null | wc -l) caseC=$(grep -c '  C) MIX' kisoku-run.sh) step=$(grep -a 'completed step' ~/logs/kisoku-v2-1b-longctx-b.log | tail -1 | grep -o 'step: [0-9]*')"

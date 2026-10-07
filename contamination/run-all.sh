#!/bin/bash
# Full scan, then summary + upload + power off (the VM bills while running).
: "${KISOKU_BUCKET:?set KISOKU_BUCKET to your GCS bucket name (no gs:// prefix)}"
cd ~/contam; PY=~/.venv/bin/python; D=gs://${KISOKU_BUCKET}/eval/contamination-20261001
PROCS=8 $PY scan.py > scan.log 2>&1
PROCS=4 $PY scan.py >> scan.log 2>&1   # second pass retries anything that failed (finished shards are skipped)
$PY summarize.py > summary.txt 2>&1
tar czf out.tgz out
gcloud storage cp --quiet summary.txt contamination-summary.json patterns.json scan.log out.tgz common.py build_patterns.py scan.py summarize.py $D/ >> upload.log 2>&1
echo "ALL DONE $(date -u)" >> scan.log
sudo shutdown -h +2

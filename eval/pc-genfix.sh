#!/bin/bash
tmux kill-session -t =kisoku-eval8 2>/dev/null; sleep 3; pkill -f "lm_eval --model hf"; sleep 8
python3 - <<'PY'
import json,glob,os
for f in glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--*/snapshots/*/generation_config.json')):
    c=json.load(open(f)); m=f.split('/hub/')[1].split('/')[0]
    if 'max_new_tokens' in c or 'max_length' in c:
        old={k:c.pop(k) for k in ('max_new_tokens','max_length') if k in c}
        os.remove(f) if os.path.islink(f) else None   # snapshot entry is a symlink to a blob: replace the link, keep the blob
        json.dump(c,open(f,'w'),indent=2); print('patched',m,old)
    else: print('clean  ',m)
PY
echo "=== BATCH 8 restarted $(date -u): removed max_new_tokens from cached generation_config.json (it overrode lm_eval max_length)" >> ~/logs/kisoku-eval.log
tmux new -d -s kisoku-eval8 bash /mnt/x/WSL/pc-run-evals-8.sh; sleep 270; tmux ls | head -1; tr '\r' '\n' < ~/logs/kisoku-eval.log | tail -1 | cut -c1-130

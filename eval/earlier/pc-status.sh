#!/bin/bash
export PATH=/usr/lib/wsl/lib:$PATH
grep -E "^---|=== |Error|Traceback|OutOfMemory" ~/logs/kisoku-eval.log | tail -5
tail -c 600 ~/logs/kisoku-eval.log | tr '\r' '\n' | grep -v '^$' | tail -2 | cut -c1-160
nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv,noheader
tmux ls 2>/dev/null

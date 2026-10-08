#!/bin/bash
cd /models/kisoku-eval; tmux ls 2>&1 | cut -c1-40; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
for s in H I J K L N O P; do echo "[$s] $(tr '\r' '\n' < logs/stream-$s.log 2>/dev/null | grep -a -E '^(---|===)' | tail -1 | cut -c1-110) | $(tr '\r' '\n' < logs/stream-$s.log 2>/dev/null | grep -a -E 'it/s|s/it|Error|error' | tail -1 | cut -c1-110)"; done

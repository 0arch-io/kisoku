#!/bin/bash
# Start (or switch) the local Kisoku chat. usage: eval/chat.sh [sft002|sft001]   then open http://localhost:8912
cd "$(dirname "$0")/.." || exit 1
m="models/kisoku-1.6b-chat-${1:-sft002}-Q8_0.gguf"; [ -f "$m" ] || { echo "no such model: $m"; ls models; exit 1; }
pkill -f "llama-server -m" 2>/dev/null; pkill -f "eval/chat_ui.py" 2>/dev/null; sleep 1
nohup llama-server -m "$m" --port 8911 -c 8192 --jinja -ngl 99 > /tmp/kisoku-llama-server.log 2>&1 &
nohup python3 eval/chat_ui.py > /tmp/kisoku-chat-ui.log 2>&1 &
for i in $(seq 1 40); do curl -s localhost:8911/health | grep -q ok && break; sleep 1; done
echo "Kisoku ${1:-sft002} is ready: http://localhost:8912"

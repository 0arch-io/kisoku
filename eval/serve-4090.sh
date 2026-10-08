#!/bin/bash
# usage: serve-4090.sh GGUF_NAME PORT [extra llama-server args]   Serves a model from the 4090 PC on localhost:PORT and keeps
# reconnecting (the Mac's Tailscale gets switched to another tailnet for a minute now and then). Only the server on PORT is replaced.
G=$1; P=$2; shift 2
while true; do
  ssh -o IdentityAgent=none -o IdentitiesOnly=yes -o ConnectTimeout=8 -o BatchMode=yes -o ServerAliveInterval=20 -o ServerAliveCountMax=2 -o ExitOnForwardFailure=yes \
    -L $P:127.0.0.1:$P "${KISOKU_PC_HOST:?set KISOKU_PC_HOST=user@host (the Windows PC with the GPU)}" "Get-NetTCPConnection -LocalPort $P -State Listen -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id \$_.OwningProcess -Force }; X:\\WSL\\llama\\llama-server.exe -m X:\\WSL\\llama\\$G --host 127.0.0.1 --port $P --jinja -ngl 99 $*" > /tmp/claude-501/kisoku-pc-$P.log 2>&1
  sleep 15
done

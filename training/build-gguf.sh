#!/bin/bash
# bf16 HF copy + llama.cpp build + GGUF (f16, Q8_0, Q4_K_M) for Kisoku 1.6B Preview.
set -eu
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
H=$HOME/kisoku-hf
~/hf-venv/bin/python ~/bin/finalize_hf.py $H/fp32 $H/kisoku-1.6b-preview
if [ ! -d ~/llama.cpp ]; then git clone --depth 1 https://github.com/ggml-org/llama.cpp ~/llama.cpp; fi
cd ~/llama.cpp && git log -1 --format="llama.cpp %h %cd"
cmake -B build -DLLAMA_CURL=OFF -DGGML_NATIVE=ON > /dev/null && cmake --build build -j 64 --target llama-quantize llama-cli llama-simple > /dev/null
~/.local/bin/uv pip install --python ~/hf-venv/bin/python -r requirements/requirements-convert_hf_to_gguf.txt > /dev/null 2>&1 || true
mkdir -p $H/gguf
PYTHONPATH=~/llama.cpp/gguf-py ~/hf-venv/bin/python convert_hf_to_gguf.py $H/kisoku-1.6b-preview --outtype f16 --outfile $H/gguf/kisoku-1.6b-preview-F16.gguf
for q in Q8_0 Q4_K_M; do ./build/bin/llama-quantize $H/gguf/kisoku-1.6b-preview-F16.gguf $H/gguf/kisoku-1.6b-preview-$q.gguf $q > /dev/null; done
ls -la $H/gguf
echo GGUF-OK

---
license: apache-2.0
language:
- en
pipeline_tag: text-generation
library_name: gguf
base_model:
- 0arch-io/kisoku-1.6b
- 0arch-io/kisoku-1.6b-chat
tags:
- gguf
- llama.cpp
- ollama
- kisoku
---

# Kisoku 1.6B GGUF

GGUF builds of [Kisoku 1.6B (base)](https://huggingface.co/0arch-io/kisoku-1.6b) and [Kisoku 1.6B Chat (preview)](https://huggingface.co/0arch-io/kisoku-1.6b-chat) for llama.cpp and Ollama. Benchmarks and known failures are on those cards. The chat model is a preview.

## Files

| File | Model | Quantization | Approx size |
|---|---|---|---|
| `kisoku-1.6b-base-longC-1299-F16.gguf` | base (final, Phase C step 1299) | F16 | 3.21 GB |
| `kisoku-1.6b-base-longC-1299-Q8_0.gguf` | base | Q8_0 | 1.71 GB |
| `kisoku-1.6b-base-longC-1299-Q4_K_M.gguf` | base | Q4_K_M | 1.02 GB |
| `kisoku-1.6b-chat-sft009-F16.gguf` | chat (pass 9) | F16 | 3.21 GB |
| `kisoku-1.6b-chat-sft009-Q8_0.gguf` | chat (pass 9) | Q8_0 | 1.71 GB |
| `kisoku-1.6b-chat-sft009-Q4_K_M.gguf` | chat (pass 9) | Q4_K_M | 1.02 GB |

F16 files were converted with llama.cpp's `convert_hf_to_gguf.py`; the quantized files with `llama-quantize`. The Q8_0 files were smoke-tested with `llama-server` (the base completes a factual prompt, the chat model follows its template).

Quantization costs some quality. I have not measured how much for Kisoku. I did not measure it; if Q4_K_M quality matters to you, run the harness on the quantized file.

## llama.cpp

```bash
# chat model, interactive
llama-cli -m kisoku-1.6b-chat-sft009-Q8_0.gguf -cnv --temp 0.7 --top-p 0.9 --repeat-penalty 1.0

# server with an OpenAI-compatible API
llama-server -m kisoku-1.6b-chat-sft009-Q8_0.gguf -c 8192 --port 8080

# base model, plain completion
llama-cli -m kisoku-1.6b-base-Q8_0.gguf -p "The three main causes of the French Revolution were" -n 100 --temp 0 --repeat-penalty 1.0
```

Keep the repetition penalty at 1.0 (off). It hurt every generation score in my tests. The chat template is read from the GGUF metadata (Llama 3 style, `<|eot_id|>` ends a turn).

Context: the GGUF stores the plain RoPE config (theta 5,000,000, trained to 64K). For YaRN x2 beyond 64K, add llama.cpp's YaRN flags, for example `-c 131072 --rope-scaling yarn --rope-scale 2 --yarn-orig-ctx 65536`. I have not verified that these flags reproduce the RULER numbers, which were measured with the Hugging Face YaRN config. Plain config is the safer default up to 32K. YaRN trades about 1.3 points at 32K for a large gain at 64K.

## Ollama

Save as `Modelfile` next to the GGUF:

```
FROM ./kisoku-1.6b-chat-sft009-Q8_0.gguf

TEMPLATE """{{- range .Messages }}<|start_header_id|>{{ .Role }}<|end_header_id|>

{{ .Content }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>

"""

SYSTEM """You are Kisoku, a helpful AI assistant created by 0ARCH."""

PARAMETER stop "<|eot_id|>"
PARAMETER stop "<|end_of_text|>"
PARAMETER stop "<|start_header_id|>"
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 8192
```

```bash
ollama create kisoku-chat -f Modelfile
ollama run kisoku-chat
```

This Modelfile is the earlier preview model's with two changes: no `repeat_penalty` (a repetition penalty hurt every Kisoku generation score in my tests) and `num_ctx` raised from 4096 to 8192. The template prints every message including the system one; the leading BOS token comes from the GGUF. Tested 2026-10-07 with a two-turn chat in Ollama 0.x on the Q8_0 file: the template and stop tokens work (the answer it gave about Canberra's population was wrong, which is the confident-wrong-answer failure on the chat card).

For the base model, a plain completion Modelfile works: `FROM ./kisoku-1.6b-base-Q8_0.gguf`, `TEMPLATE "{{ .Prompt }}"`, `PARAMETER temperature 0.7`.

## Limitations and licence

Same as the model cards. Licence: Apache 2.0 (see the base card's note about the Llama 3 tokenizer vocabulary).

---
license: PLACEHOLDER_LICENCE
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
| `kisoku-1.6b-base-F16.gguf` | base (final, Phase C) | F16 | 3.2 GB |
| `kisoku-1.6b-base-Q8_0.gguf` | base | Q8_0 | about 1.7 GB |
| `kisoku-1.6b-base-Q4_K_M.gguf` | base | Q4_K_M | about 1 GB |
| `kisoku-1.6b-chat-sft009-F16.gguf` | chat (pass 9) | F16 | 3.2 GB (3,209,526,912 bytes) |
| `kisoku-1.6b-chat-sft009-Q8_0.gguf` | chat (pass 9) | Q8_0 | about 1.7 GB |
| `kisoku-1.6b-chat-sft009-Q4_K_M.gguf` | chat (pass 9) | Q4_K_M | about 1 GB |

[CHECK] File names and sizes of the Q8_0, Q4_K_M and base files are expected, not confirmed: only `kisoku-1.6b-chat-sft009-F16.gguf` exists in the bucket today. The build script (`training/build-gguf-chat.sh`) produces F16, Q8_0 and Q4_K_M with the same naming. Quantized sizes are estimates. Rename or re-convert before upload as decided in MANIFEST.md.

Quantization costs some quality. I have not measured how much for Kisoku. [CHECK] If Q4_K_M scores matter, run a quick check before claiming anything.

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

Context: the GGUF stores the plain RoPE config (theta 5,000,000, trained to 64K). For YaRN x2 beyond 64K, add llama.cpp's YaRN flags, for example `-c 131072 --rope-scaling yarn --rope-scale 2 --yarn-orig-ctx 65536`. [CHECK] I have not tested these flags against the RULER numbers, which were measured with the HF YaRN config. Plain config is the safer default up to 32K. YaRN trades about 1.3 points at 32K for a large gain at 64K.

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

[CHECK] This Modelfile is the one from the earlier preview model with one deliberate change: no `repeat_penalty 1.1` (the old file had it, and the report shows a repetition penalty hurts), and `num_ctx` raised from 4096 to 8192. The template prints every message including the system one, and the model was trained with a leading BOS token that llama.cpp adds from the GGUF. Test a multi-turn chat in Ollama before publishing.

For the base model, a plain completion Modelfile works: `FROM ./kisoku-1.6b-base-Q8_0.gguf`, `TEMPLATE "{{ .Prompt }}"`, `PARAMETER temperature 0.7`.

## Limitations and licence

Same as the model cards. Licence: PLACEHOLDER_LICENCE (see the base card's note about the Llama 3 tokenizer vocabulary).

---
license: PLACEHOLDER_LICENCE
language:
- en
pipeline_tag: text-generation
library_name: transformers
base_model: none
tags:
- text-generation
- pretrained
- from-scratch
- qwen3
- long-context
- tpu
- kisoku
# Dataset ids below are best guesses from the report's data section. [CHECK] each one resolves on the Hub before publishing.
datasets:
- nvidia/Nemotron-CC-v2.1
- nvidia/Nemotron-CC-Math-v1
- openbmb/Ultra-FineWeb
- bigcode/starcoderdata
- HuggingFaceTB/finemath
- open-web-math/open-web-math
- LLM360/MegaMath
- open-thoughts/OpenThoughts3-1.2M
- deepmind/pg19
- princeton-nlp/prolong-data-64K
---

# Kisoku 1.6B (base)

Kisoku 1.6B is a base (not chat) language model pretrained from random initialization by one person, Joseph Rodriguez at 0ARCH, on a TPU v4-32 from Google's TPU Research Cloud. It has 1,600,749,056 parameters, saw about 0.5 trillion tokens of pretraining text (524.4B including the long-context phases), and was extended to 64K tokens of context. All numbers below come from the technical report draft (draft 3, 2026-10-05) and were measured by me on my own hardware.

This repo holds the final base model: the checkpoint after long-context Phase C (step 1299). Earlier checkpoints are on branches, see "Other checkpoints" below. A chat fine-tune is in [0arch-io/kisoku-1.6b-chat](https://huggingface.co/0arch-io/kisoku-1.6b-chat) (preview) and GGUF builds are in [0arch-io/kisoku-1.6b-gguf](https://huggingface.co/0arch-io/kisoku-1.6b-gguf).

## What to know first

- It matches Llama 3.2 1B on a 10-benchmark suite (five wins each) with about 18 times less training data. That is a statement about token efficiency on one suite, not a ranking. SmolLM2 1.7B and Qwen2.5 1.5B beat it on most tests.
- Kisoku has 1.6B parameters against Llama 3.2 1B's 1.24B. Llama 3.2 1B was distilled from larger models. Kisoku used no distillation loss, but roughly a third or more of the documents it read were written or rewritten by larger models (see "Training data").
- Long context is better than Llama 3.2 1B at most lengths but trails Granite 4.0 1B and the Qwen3.5 models at every length.
- It is a base model. It continues text and does not follow instructions reliably.

## Benchmarks (short context)

lm-evaluation-harness 0.4.13, bfloat16, one RTX 4090, same harness for every model. Scores are percent correct. The first Kisoku column is the **stage 3 checkpoint, before long-context training**, which the report uses throughout; the second is the released checkpoint after long-context training, rerun on 2026-10-07 on the same harness. They agree within about a point on every test.

| Benchmark | Kisoku 1.6B (stage 3) | Kisoku 1.6B (released) | Llama 3.2 1B | SmolLM2 1.7B |
|---|---|---|---|---|
| GSM8K (5-shot) | 15.3 | 15.0 | 5.8 | 30.0 |
| MMLU (5-shot) | 33.0 | 34.2 | 31.3 | 50.0 |
| ARC-Easy (0-shot) | 65.4 | 64.4 | 61.8 | 73.5 |
| ARC-Challenge (0-shot) | 39.8 | 40.1 | 36.9 | 46.9 |
| BBH (3-shot) | 29.2 | 28.6 | 28.3 | 31.4 |
| PIQA (0-shot) | 73.9 | 73.0 | 74.9 | 77.9 |
| WinoGrande (0-shot) | 57.3 | 56.8 | 60.5 | 66.3 |
| HellaSwag (0-shot) | 57.7 | 57.8 | 64.2 | 71.3 |
| HumanEval (pass@1) | 13.4 | 14.6 | 18.9 | not reported |
| TriviaQA (5-shot) | 22.8 | 23.1 | 40.7 | 49.7 |

Against Llama 3.2 1B, Kisoku wins GSM8K, MMLU, ARC-Easy, ARC-Challenge and BBH. Llama wins PIQA, WinoGrande, HellaSwag, HumanEval and TriviaQA. The BBH gap (0.9) and MMLU gap (1.7) are inside the 95 percent intervals in the report, so do not read them as established. HumanEval has 164 problems, so one problem is 0.6 points.

Settings: HellaSwag, ARC and PIQA and WinoGrande 0-shot with length-normalized accuracy. BBH uses a custom task copy that strips whitespace before exact match (the stock task scores 0.0 for every model). Generation tasks use greedy decoding with no repetition penalty. The baselines were run through the same setup, so their numbers will not match their own model cards. DROP is excluded because its targets load as a CSV header in this harness.

Against my first model (Kisoku v1, 3B, about 60B tokens), this model is better on all ten tests, for example GSM8K 0.0 to 15.3 and HellaSwag 29.0 to 57.7.

## Long context (RULER)

Final checkpoint, harness-built-in RULER (13 tasks, average), 100 samples per task at 4K and 8K and 50 per task above, greedy, no repetition penalty, 128 new tokens max, bfloat16. Runs were split across an RTX 4090 and an RTX PRO 6000. Single runs, no confidence intervals.

| Model | 4K | 8K | 16K | 32K | 64K | 128K |
|---|---|---|---|---|---|---|
| **Kisoku 1.6B, YaRN x2 (shipped default)** | **76.1** | **66.2** | **64.2** | **58.8** | **56.7** | **43.2** |
| Kisoku 1.6B, plain (no YaRN) | 76.2 | 66.7 | 64.6 | 60.1 | 44.7 | 4.7 |
| Llama 3.2 1B | 73.5 | 67.4 | 61.9 | 56.7 | 49.2 | 43.1 |
| Granite 4.0 1B (128K native) | 85.3 | 77.5 | 71.6 | 60.7 | 63.4 | 46.7 |
| Qwen3.5 0.8B | 86.7 | 83.0 | 79.2 | 74.2 | 68.5 | 64.2 |
| Qwen3.5 2B | 91.6 | 89.7 | 86.8 | 83.6 | 78.7 | 70.7 |

With YaRN, Kisoku is ahead of Llama 3.2 1B at 4K, 16K, 32K and 64K, level at 128K (43.2 against 43.1) and behind at 8K. It is behind Granite 4.0 1B and both Qwen3.5 models at every length. I make no claim of leading at long context. Other models in the full table (Qwen3 1.7B and 0.6B, LFM2.5 1.2B, Gemma 3 1B) are in the report.

Caveats that matter:
- The Phase C training mix has 18% synthetic documents (needle lookup, definition chains, word counts, text position) that resemble parts of RULER. The RULER scores are partly format familiarity. A held-out long-context suite has not been run.
- YaRN helps at 64K (44.7 plain to 56.7) and costs about 1.3 points at 32K. On one machine, 64K went 45.7 plain, 53.4 at factor 1.5, 56.7 at factor 2, 53.1 at factor 4. Neither half of YaRN (rescaled frequencies, attention factor) gives the gain alone. My untested reading is that the 64K training phases were short (about 4,160 steps), so the top octave is under-trained and YaRN repairs it.
- Without YaRN the model collapses at 128K (4.7 on RULER), so keep the shipped rope_scaling if you run past 64K.

## Model

| | |
|---|---|
| Architecture | Qwen3-style decoder (`Qwen3ForCausalLM`), implemented in MaxText |
| Parameters | 1,600,749,056 (the "1B" in some run names is a leftover) |
| Layers / width / MLP | 22 / 2048 / 8192 |
| Attention | 16 query heads, 4 key/value heads (GQA), head dim 128 |
| Embeddings | tied input and output, vocab 128,256 |
| RoPE theta | 5,000,000 |
| Context | pretrained at 4096, extended to 32K then 64K, YaRN x2 for about 128K |
| Tokenizer | Llama 3 tokenizer vocabulary (copied from unsloth/Llama-3.2-1B) |

The HF export was checked against the training framework in float32: argmax agreement 100%, maximum logit difference 0.04. The end-of-sequence token is set to the base end-of-text token (128001).

## Training

Optimizer Muon (momentum 0.95, weight decay 0.1) on 2D attention and MLP matrices, AdamW for embeddings, norms and biases, peak learning rate 3e-4 for both, gradient clipping 1.0. Warmup 1,215 steps, warmup-stable-decay schedule written for 243,000 steps. About 2.1M tokens per step.

| Phase | Steps | Tokens |
|---|---|---|
| Stages 1 to 3 (4K context) | 243,000 | 509.6B |
| Long-context Phase A (32K) | 2,900 | 6.08B |
| Long-context Phase B (64K) | 2,860 | 6.00B |
| Long-context Phase C (64K, plus 18% synthetic tasks) | 1,300 | 2.73B |
| All phases | | 524.4B |

Stage 1 ended at the full 3e-4 learning rate and was not annealed (a launcher was lost during a laptop migration, so the stage 2 and 3 mixes started late). Learning rate decayed during stage 3. Everything was a single run with no seeds and no ablations.

### Training data

Stage 1 mix weights (by document): Nemotron-CC 0.35, Ultra-FineWeb 0.35, StarCoder 0.16, FineMath 0.06, OpenWebMath 0.04, MegaMath-Web-Pro 0.04. Stage 2 shifted toward code and math. Stage 3 added Nemotron-CC-Math-v1 (4plus subset) and OpenThoughts3 text. Long-context phases added repository-level code (The Stack v1 via ProLong), Dolma 3 Longmino science PDFs and PG19 books. About 13.1B long tokens were available. ProLong's "book" subset was excluded on purpose.

Model-written text in the data: Nemotron-CC v2.1 synthetic subset (rephrased by Qwen3-30B-A3B), MegaMath-Web-Pro (rewritten by Llama-3.3-70B-Instruct), Nemotron-CC-Math-v1 (cleaned by Phi-4), OpenThoughts3 (traces by QwQ-32B). Together roughly a third or more of the documents read. This is weaker than logit distillation but it is real knowledge transfer, so read the token-efficiency comparison with it in mind. [CHECK] the report's own TBD: confirm from data-prep logs that the Nemotron-CC subset used was the synthetic one.

## How to run

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

repo = "0arch-io/kisoku-1.6b"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.bfloat16, device_map="auto")

inputs = tok("The three main causes of the French Revolution were", return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=100, do_sample=False)
print(tok.decode(out[0], skip_special_tokens=True))
```

Use greedy decoding or light sampling. Do not set a repetition penalty: it hurt every Kisoku generation score in my tests (RULER 63.5 to 70.7 at 4K on an earlier checkpoint when removed). The shipped `generation_config.json` only sets token ids.

### YaRN rope config

`config.json` in this repo enables YaRN with factor 2 over the 64K training length, which is the configuration the long-context numbers above used as the default row. [CHECK] The checkpoint currently in the bucket has `"rope_scaling": null`; patch the config before upload (see MANIFEST.md). Intended settings:

```json
"max_position_embeddings": 131072,
"rope_parameters": {"rope_type": "yarn", "factor": 2.0, "original_max_position_embeddings": 65536, "rope_theta": 5000000.0},
"rope_scaling":    {"rope_type": "yarn", "factor": 2.0, "original_max_position_embeddings": 65536}
```

YaRN in transformers is static: it applies to all lengths, which is why 32K costs about 1.3 points. If you only work up to about 32K, switch it off:

```python
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained(repo)
cfg.rope_scaling = None
cfg.rope_parameters = {"rope_type": "default", "rope_theta": 5000000.0}
cfg.max_position_embeddings = 65536
model = AutoModelForCausalLM.from_pretrained(repo, config=cfg, torch_dtype=torch.bfloat16, device_map="auto")
```

(Written for transformers 4.57.x, which the export used. [CHECK] the override on the transformers version you target.) A 116K-token prompt peaked at about 15 GB in bfloat16 in my rehearsal, so 128K fits on a 24 GB card.

## Other checkpoints

Branches in this repo (see MANIFEST.md for which exist): `stage1-step198999` (not annealed), `stage3-step242999` (the suite numbers above), `long-phaseA-step2500`, and the Phase B checkpoint. [CHECK] branch names and which are published.

## Limitations

- Fact recall is weak: TriviaQA 22.8 against 40.7 for Llama 3.2 1B, HellaSwag 57.7 against 64.2. It saw about 0.5T tokens against 2T to 18T for its peers.
- Qwen2.5 1.5B and SmolLM2 1.7B beat it by large margins on most tests.
- No confidence intervals, one training run per configuration, one harness and machine, bfloat16.
- Long context trails Granite 4.0 1B and Qwen3.5. The long-context set-up has format familiarity with RULER and no held-out suite was run.
- Long-context training did not change the short-context scores (second Kisoku column above).
- Not evaluated: safety, bias, multilingual ability. It is English-focused.
- It can reproduce web text. It is a raw base model with no safety tuning.

## Contamination audit

For each test item I probed the first and last 80 characters (61,366 probes over GSM8K, MMLU, ARC, HellaSwag, PIQA, WinoGrande, HumanEval, TriviaQA) against all 784 shards (about 1.66 TB, 11 datasets) with an Aho-Corasick scan. Verbatim overlap exists, mostly from homework-help and quiz pages in Common Crawl derived sets, for example any-probe hit rates of 27.6% on GSM8K, 28.7% on MMLU, 20.9% on ARC-Easy, 26.3% on ARC-Challenge, 25.6% on HumanEval prompts, 6.9% on HellaSwag, 3.3% on TriviaQA, 2.6% on PIQA, 0.5% on WinoGrande. Accuracy on overlapping items is not higher than on clean items for GSM8K, MMLU or HellaSwag (GSM8K 13.7 overlap against 15.9 clean). The audit only catches exact matches of 80-character probes, not paraphrases or answer-only leakage, so a clean result means no evidence of verbatim leakage driving the scores, not no contamination. Scripts and per-item results are in the code release.

## Licence note

Licence: PLACEHOLDER_LICENCE. [CHECK] Final decision pending. The tokenizer is the Llama 3 vocabulary (128,000 base tokens plus 256 special tokens; 100,256 of the base tokens are identical to tiktoken `cl100k_base`). The Llama 3.2 Community License does not mention tokenizers, and other from-scratch models ship the same vocabulary under Apache-2.0 without a Llama name, but I am not a lawyer. Training data licences vary: Nemotron-CC and Nemotron-CC-Math-v1 allow publishing models trained on them but not redistributing the text, so the data manifests list sources and counts only.

## Citation

```bibtex
@misc{rodriguez2026kisoku,
  title        = {Kisoku 1.6B: A Solo, From-Scratch Pretrain on a Free TPU Grant},
  author       = {Rodriguez, Joseph},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/0arch-io/kisoku-1.6b}},
  note         = {Technical report: [CHECK] link}
}
```

Built with MaxText on Google's TPU Research Cloud. Thanks to the authors of lm-evaluation-harness and the open datasets and recipes listed in the report (Nemotron-CC, Ultra-FineWeb, StarCoder, FineMath, OpenWebMath, MegaMath, OpenThoughts, Dolma 3, ProLong, PG-19, SmolLM3, Olmo 3).

---
license: apache-2.0
language:
- en
pipeline_tag: text-generation
library_name: transformers
base_model: 0arch-io/kisoku-1.6b
tags:
- text-generation
- conversational
- chat
- preview
- kisoku
# Dataset ids are best guesses from the report's chat data section. [CHECK] each one resolves before publishing.
datasets:
- HuggingFaceTB/smoltalk2
- NousResearch/Hermes-3-Dataset
- teknium/OpenHermes-2.5
- HuggingFaceH4/no_robots
- open-thoughts/OpenThoughts3-1.2M
---

# Kisoku 1.6B Chat (preview)

**This is a preview, not a finished assistant.** It is the base model [0arch-io/kisoku-1.6b](https://huggingface.co/0arch-io/kisoku-1.6b) after nine supervised fine-tuning passes. This repo is pass 9 (the pass labelled sft009 in my run names). It loses the thread in longer conversations, is behind Llama 3.2 1B Instruct on a multi-turn test, and states wrong answers confidently. Read "Known failures" before using it.

All numbers come from the technical report draft (draft 3, section 7B and 7) and were measured by me.

## How it was made

Chat fine-tuning ran from 2026-10-04 to 2026-10-06 on the TPU v4-32, nine supervised passes of about 70 to 95 minutes each (about 131K tokens per step). Every pass started from the final long-context base checkpoint (Phase C, step 1299) with a fresh optimizer. No pass continued from an earlier chat model. One DPO experiment was run and not shipped.

Training data (every example containing a benchmark test item was dropped by verbatim matching, and a rescan of the final shards found zero hits):
- Public chat sets: SmolTalk2 subsets, Hermes 3, OpenHermes, no_robots, tool traces (xlam, Hermes function calling).
- A first teacher-written set of 34,879 examples from DeepSeek-V4.1-Flash.
- A second teacher-written set of about 47,500 conversations written as Kisoku (everyday chat, revise-my-work, casual build requests, explanations, short Q&A, tool conversations, identity and capabilities, small talk, verified math and logic).
- Targets were written only by the DeepSeek teacher or taken from public datasets. The teacher output is text from another language model, so check the DeepSeek terms for your use. [CHECK] licence implications of the teacher-written data.
- Corrections of the model's own bad turns in conversation, "I don't know" examples built from the model's own wrong answers, a set for defensive self-talk, and keyboard typo noise on short user messages.

Thinking mode was trained in pass 3 and removed. "Hold your ground" examples were removed (the model could not tell its right answers from wrong ones).

## Benchmarks (7B suite)

Base-style tests, no chat template, same harness as the base card (lm-evaluation-harness 0.4.13, bfloat16, RTX 4090). The base column is the **stage 3** checkpoint, before long-context training, so part of each difference may come from that training and not from chat tuning. [CHECK] the short-context rerun on the final long-context checkpoint (batch 18) was queued on 2026-10-07; if it finished, replace the base column.

| Test | Base (stage 3) | Chat, pass 9 |
|---|---|---|
| GSM8K | 15.3 | 20.6 |
| HumanEval | 13.4 | 14.0 |
| ARC-Easy | 65.4 | 66.0 |
| ARC-Challenge | 39.8 | 42.2 |
| WinoGrande | 57.3 | 57.2 |
| PIQA | 73.9 | 73.0 |
| HellaSwag | 57.7 | 57.2 |
| TriviaQA | 22.8 | 20.8 |
| MMLU | 33.0 | 29.8 |
| BBH | 29.2 | 26.3 |

Math and code went up, MMLU and BBH went down by about 3 points, and the rest moved by about a point or less. No confidence intervals.

## Long context (RULER, YaRN x2)

| Model | 4K | 32K | 64K |
|---|---|---|---|
| Kisoku chat, pass 9 | 76.5 | 60.5 | 54.3 |
| Kisoku base, final | 76.1 | 58.8 | 56.7 |
| Llama 3.2 1B (base) | 73.5 | 56.7 | 49.2 |

Single runs of 650 samples, no confidence interval. The chat model keeps most of the base long-context ability. The Phase C mix included synthetic tasks resembling RULER, so see the base card's caveat. Chat-model RULER at other lengths was not reported. [CHECK]

## Held-out conversation test

After pass 5 failed a real nine-turn conversation, I built a test that is not a replay. A teacher model wrote 2,396 conversation scripts of up to nine user turns that make sense whatever the assistant answers (small talk, a casual build request, "make it shorter", "that's not what I asked", a change of topic and a return). The chat model answers every turn itself. The teacher then reads the transcript once and marks each assistant turn acceptable or not. One script in eight is held out by a hash of its id (297 scripts, about 2,530 turns) and never trained on.

| Model on the same 297 held-out scripts | Turns acceptable | Bad first turns |
|---|---|---|
| Llama 3.1 8B Instruct | 59.5% | 10% |
| Llama 3.2 1B Instruct | 56.7% | 11% |
| Kisoku chat, pass 6 (best of mine) | 50.3% | 8% |
| **Kisoku chat, pass 9 (this model)** | **48.9%** | **10%** |

The Llama models were each given a system prompt saying they are Kisoku (with Kisoku's facts and style) so the judge would not penalize their own names. They still lost some turns by calling themselves "a large language model", so their true figures are a little higher.

**Kisoku chat is 6 to 8 points behind Llama 3.2 1B Instruct on multi-turn conversation.** Llama 3.2 1B was distilled from the 8B and 70B models and post-trained by a large team. This model is nine passes by one person over three days on a base trained on about a sixth of the data. The judge is strict for everyone (the 8B model clears only 60%), and almost no conversation gets through all nine turns with every turn acceptable (0 to 2 of 297 for every model). I have not measured how much the judge's marks vary between two readings of the same transcript. Passes 6 to 9 (50.3, 49.5, 49.5, 48.9) are a tie.

What moved the number: training on corrections of the model's own bad turns took the held-out figure from 37.6% (pass 5) to 50.3% (pass 6). A second round gave nothing. DPO gave nothing.

## Known failures

From "What remains in the released chat model" and the manual-chat findings:

- About half of its turns in a long conversation are unacceptable to a strict judge, most often because it loses track of what it said earlier or builds on its own previous mistake.
- A typo in a longer request can lose the request. Typo-tolerance gain was small: a typo in the first message gives an on-topic reply 79% of the time, after a short chat 70%, against 99.6% when spelled correctly. A misspelled website request after a short chat was on-topic 14 of 36 times.
- It refuses some requests it could simply take. "Can u be my tutor for math" is still refused 15 of 24 times because older data says it is not a language tutor.
- Defensive self-talk after questions about itself. Replies judged bad by a pattern check on openings that should be accepted: 20 of 120 on covered situations, 29 of 216 on situations the data never described.
- It states wrong answers to hard math as confidently as right ones. On one competition-style algebra question it gave the correct value in 9 of 24 samples and six different wrong values in the others.
- It declines only some of the questions it gets wrong: on held-out long-tail questions it declined 38% of previously-wrong ones and wrongly declined 15% of previously-right ones. Abstention and coverage trade off at this size.
- It still makes unneeded tool calls on some probes (an earlier pass called a weather tool for "What is 12 times 12?", 4 of 12 no-tool probes). [CHECK] pass 9 rate was not reported.
- Safety, bias and multilingual behavior were not evaluated. Instruction following and tool use were only probed by hand and on small held-out sets.

## Chat template

The template is Llama 3 style and ships in `chat_template.jinja` and `tokenizer_config.json` (source: `training/finalize_hf.py` and the training template in `sft/build_sft_v2.py`). Training used one `<|begin_of_text|>` at the start of the conversation only, then each message as `<|start_header_id|>ROLE<|end_header_id|>\n\n` + content + `<|eot_id|>`, and the assistant turn is generated after `<|start_header_id|>assistant<|end_header_id|>\n\n`. The end-of-turn token is `<|eot_id|>` (128009).

Roles: `system`, `user`, `assistant`, and `tool` for tool results. The template has no default system prompt. In training some examples had a system prompt like "You are Kisoku, a helpful AI assistant created by 0ARCH." and many had none. [CHECK] whether a default system prompt should be recommended.

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

repo = "0arch-io/kisoku-1.6b-chat"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.bfloat16, device_map="auto")

messages = [{"role": "user", "content": "hey, can you help me write a short bio?"}]
ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
out = model.generate(ids, max_new_tokens=300, do_sample=True, temperature=0.7, top_p=0.9)
print(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True))
```

`generation_config.json` sets sampling on with temperature 0.7 and top-p 0.9 and stops on 128009 and 128001. It sets no repetition penalty, and you should not add one.

Tool use (trained, lightly tested): put the tool list as JSON inside `<tools></tools>` in the system message, with the instruction that calls are a JSON object inside `<tool_call></tool_call>` and results come back inside `<tool_response></tool_response>` in a `tool` message. The exact system prompt text used in training is `TOOL_SYSTEM` in `sft/build_sft_v2.py` in the code release.

## YaRN and context

The config follows the base model: trained to 64K, YaRN factor 2 for about 128K. See the base card for the exact config and how to turn it off. [CHECK] the chat export's config.json in the bucket also has `rope_scaling: null`; patch like the base before upload.

## GGUF

Builds for llama.cpp and Ollama are in [0arch-io/kisoku-1.6b-gguf](https://huggingface.co/0arch-io/kisoku-1.6b-gguf). [CHECK] only the F16 file for this pass exists in the bucket so far; Q8_0 and Q4_K_M still need to be built.

## Limitations

This card's failure list above is the main one. Also: English-focused, one evaluation setup (one harness version, bfloat16, my prompt settings), the judge in the conversation test is a language model that I did not calibrate, and the model has had no safety or red-team evaluation. Do not use it for anything where a wrong answer matters.

## Licence note and citation

Licence: Apache 2.0. The tokenizer is the Llama 3 vocabulary; see the base card's licence note. Teacher-written data came from DeepSeek-V4.1-Flash.

```bibtex
@misc{rodriguez2026kisoku,
  title        = {Kisoku 1.6B: A Solo, From-Scratch Pretrain on a Free TPU Grant},
  author       = {Rodriguez, Joseph},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/0arch-io/kisoku-1.6b-chat}}
}
```

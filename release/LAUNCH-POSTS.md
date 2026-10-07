# Launch posts (drafts for Friday 2026-10-10)

Post after the repos are public. Links: report https://github.com/0arch-io/kisoku/blob/main/report/kisoku-report-draft.md, weights https://huggingface.co/0arch-io/kisoku-1.6b, site https://0arch.io/kisoku. Chart: `release/fig-efficiency.png`.

## Hacker News

Title (under 80 characters):

> Show HN: Kisoku 1.6B, a from-scratch LLM trained solo on a free TPU grant

URL: the GitHub repo.

First comment (post it yourself right after submitting):

> I trained this alone over the last few months on a TPU v4-32 from Google's TPU Research Cloud, with an RTX 4090 at home for evals. 1.6B parameters, about 0.5T tokens, Qwen3-style architecture, Llama 3 tokenizer, MaxText on JAX.
>
> The result I think is interesting: on a 10-benchmark suite run through lm-evaluation-harness on the same machine for every model, it matches Llama 3.2 1B (five wins each) with roughly 18 times less training data. The wins are reasoning and math (GSM8K 15.0 vs 5.8, MMLU, ARC, BBH); Llama wins knowledge and commonsense (TriviaQA 40.7 vs 23.1, HellaSwag). Qwen2.5 1.5B and SmolLM2 1.7B are ahead of both on most tests, so this is a token-efficiency result, not "best small model".
>
> Things I tried to be careful about: a verbatim contamination audit of the whole pretraining corpus against the test sets (overlap exists; accuracy on the clean subset is the same or higher), bootstrap confidence intervals, and a section on what went wrong, including an eval bug that understated my own generation scores for weeks. A good chunk of the pretraining text was rewritten by other LLMs (Nemotron-CC and friends), which I count as real knowledge transfer and say so.
>
> It was also extended to 64K context (about 128K with YaRN). On RULER it is ahead of Llama 3.2 1B at most lengths but behind Granite 4.0 1B and Qwen3.5 at every length. The chat fine-tune is a preview and loses the thread in long conversations; the report has the numbers.
>
> Out of pocket was about $320 (storage and API calls). Weights, intermediate checkpoints, GGUFs, code, eval outputs and the report are all Apache 2.0. Happy to answer anything about the recipe or the mistakes.

## X thread

1/ I trained a 1.6B language model from scratch, alone, on a free TPU grant. It matches Llama 3.2 1B on 10 benchmarks with ~18x less training data. Weights, code and a full report are out today, Apache 2.0. [chart]

2/ Same harness, same machine, every model. Five wins each against Llama 3.2 1B: Kisoku takes GSM8K, MMLU, ARC and BBH; Llama takes PIQA, WinoGrande, HellaSwag, HumanEval, TriviaQA. Qwen2.5 1.5B and SmolLM2 1.7B beat both. It is a token-efficiency result, nothing more.

3/ About 0.5T tokens on a TPU v4-32 from Google's TPU Research Cloud. MaxText on JAX, Qwen3-style decoder, Llama 3 tokenizer. Three pretraining stages, then 32K and 64K long-context phases. Out of pocket: about $320.

4/ I audited the whole corpus for verbatim test-set overlap. It exists. Accuracy on the clean subset is the same or higher, so it does not explain the scores. Code and results are in the repo.

5/ What went wrong is its own section: a wrong repetition-penalty default that understated my generation scores, a lost stage 2 checkpoint, a long-context goal I did not reach. The chat fine-tune is a preview that loses the thread in long conversations.

6/ Everything: base weights + intermediate checkpoints, chat preview, GGUFs for Ollama, training and eval code, raw per-sample eval outputs, the report. [links]

## LinkedIn

Today I am releasing Kisoku 1.6B, a language model I pretrained from scratch, by myself, on a TPU grant from Google's TPU Research Cloud.

On a ten-benchmark suite run through the same harness on the same machine for every model, it matches Meta's Llama 3.2 1B (five wins each) while using about 18 times less training data. It wins on reasoning and math, Llama wins on knowledge, and larger-data models like Qwen2.5 1.5B are ahead of both. I say that plainly in the report because the point of the project was to do the whole thing honestly: a contamination audit of the training data, confidence intervals on every table, and a section on what went wrong.

What is released, all Apache 2.0: the weights and intermediate checkpoints, a chat preview, GGUF builds that run offline on a laptop through Ollama, the training and evaluation code, the raw evaluation outputs, and a technical report.

Out-of-pocket cost was about $320. The rest was the grant, one RTX 4090, and time.

Report and weights: [links]

## Hugging Face

The model card is the post. Optionally add the three repos to a collection named "Kisoku 1.6B" with the report link in the description.

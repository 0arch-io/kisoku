"""Build the Kisoku v2 SFT mix: stream sources, clean, filter refusals and foreign identities,
sample, length-filter, and write parquet shards (2 pre-shuffled passes) plus an eval split."""
import itertools, json, os, random, re, sys, time
from collections import Counter, defaultdict

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from transformers import AutoTokenizer

QUICK = bool(os.environ.get("QUICK"))  # test mode: first 3000 rows per source
OUT = os.path.expanduser("~/sft-build/kisoku-sft-quick" if QUICK else "~/sft-build/kisoku-sft-v1")
IDENTITY = os.path.expanduser("~/sft-build/identity.jsonl")
MAX_TOKENS = 4096
N_EVAL = 200 if QUICK else 2000
N_TRAIN_SHARDS_PER_PASS = 8
N_EVAL_SHARDS = 4
SEED = 1234

# smoltalk2 SFT split -> target examples (None = take everything that survives filtering)
SMOLTALK2 = {
    "smoltalk_smollm3_smol_magpie_ultra_no_think": 120000,
    "OpenHermes_2.5_no_think": 40000,
    "smoltalk_smollm3_everyday_conversations_no_think": None,
    "smoltalk_smollm3_systemchats_30k_no_think": None,
    "smoltalk_smollm3_explore_instruct_rewriting_no_think": 15000,
    "smoltalk_smollm3_smol_rewrite_no_think": 20000,
    "smoltalk_smollm3_smol_summarize_no_think": 25000,
    "tulu_3_sft_personas_instruction_following_no_think": None,
    "OpenThoughts3_1.2M_no_think_no_think": 30000,
    "Mixture_of_Thoughts_science_no_think": 15000,
}
HERMES_TARGET = 100000

KISOKU_SYSTEMS = [
    "You are Kisoku, a helpful AI assistant created by 0ARCH.",
    "You are Kisoku, an AI assistant made by 0ARCH. Answer directly and helpfully.",
]
ADD_SYSTEM_PROB = 0.25

# Refusal / AI-disclaimer / moralizing phrases, matched case-insensitively in assistant turns only.
# Hand-picked from the WizardLM-uncensored list, keeping only phrases that signal a refusal or
# disclaimer (its topic words like "ethics" or "racial" would drop ordinary history and science).
REFUSAL = [
    "as an ai language model", "as a language model", "as a large language model", "as an ai,",
    "as an ai assistant", "as an artificial intelligence", "i'm an ai", "i am an ai",
    "an ai language model", "text-based ai", "i cannot fulfill", "i can't fulfill",
    "i cannot assist", "i can't assist", "i cannot help with", "i can't help with",
    "i'm unable to help", "i am unable to help", "i'm sorry, but i can", "i'm sorry, but i cannot",
    "i'm sorry, i can't", "i'm sorry, i cannot", "i apologize, but i can", "i apologize, but i cannot",
    "i cannot provide", "i can't provide", "i won't provide", "i will not provide",
    "not able to provide", "unable to provide", "against my programming", "my programming",
    "my guidelines", "ethical guidelines", "content policy", "usage policies", "i must decline",
    "i have to decline", "i'm not comfortable", "i am not comfortable", "i cannot engage",
    "i can't engage", "responsible ai", "prioritize safety", "prioritize user safety",
    "promote safety", "potentially harmful", "harmful or illegal", "illegal or harmful",
    "it is not appropriate", "it's not appropriate", "not appropriate for me", "i cannot condone",
    "i can't condone", "i do not condone", "i don't condone", "cannot support or promote",
    "my knowledge cutoff", "my knowledge cut off", "september 2021",
    "this chat conversation is shared from", "this conversation is shared from",
]
FOREIGN_ID = re.compile(
    r"\b(openai|chatgpt|gpt-?3\.5|gpt-?4o?|anthropic|smollm\d?|nous research|hermes[ -]?\d|qwen|"
    r"tongyi|alibaba cloud|mistral ai|deepseek)\b"
    r"|\b(i am|i'm|my name is) (claude|gemini|bard|llama|hermes|chatgpt)\b",
    re.I,
)
THINK_EMPTY = re.compile(r"<think>\s*</think>\s*")
SLASH_THINK = re.compile(r"\s*/(no_)?think\s*$")

# Training template: BOS only on the message flagged bos (MaxText formats each round separately).
TRAIN_TEMPLATE = (
    "{% for message in messages %}{% if message['bos'] %}{{ '<|begin_of_text|>' }}{% endif %}"
    "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' }}"
    "{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
)
# Inference template: BOS at the start of the conversation.
INFER_TEMPLATE = (
    "{% for message in messages %}{% if loop.index0 == 0 %}{{ '<|begin_of_text|>' }}{% endif %}"
    "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' }}"
    "{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
)

stats = defaultdict(Counter)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def normalize(msgs, source, system=None):
    """Return cleaned [{role, content}] or None; count the drop reason in stats[source]."""
    out = []
    if system and system.strip():
        out.append({"role": "system", "content": system.strip()})
    for m in msgs:
        role = m.get("role") or m.get("from")
        content = m.get("content") if "content" in m else m.get("value")
        role = {"human": "user", "gpt": "assistant"}.get(role, role)
        if role not in ("system", "user", "assistant"):
            stats[source]["drop_role"] += 1
            return None
        if content is None:
            stats[source]["drop_empty"] += 1
            return None
        if role == "assistant":
            content = THINK_EMPTY.sub("", content)
        if role in ("system", "user"):
            content = SLASH_THINK.sub("", content)
        content = content.strip()
        if not content:
            if role == "system":
                continue
            stats[source]["drop_empty"] += 1
            return None
        out.append({"role": role, "content": content})
    body = out[1:] if out and out[0]["role"] == "system" else out
    if any(m["role"] == "system" for m in body):
        stats[source]["drop_structure"] += 1
        return None
    if len(body) < 2 or len(body) % 2:
        stats[source]["drop_structure"] += 1
        return None
    for j, m in enumerate(body):
        if m["role"] != ("user" if j % 2 == 0 else "assistant"):
            stats[source]["drop_structure"] += 1
            return None
    for m in out:
        c = m["content"]
        if "<think>" in c or "<tool_call>" in c or "<tools>" in c:
            stats[source]["drop_tools_think"] += 1
            return None
    for m in out:
        if m["role"] == "assistant":
            low = m["content"].lower()
            if any(p in low for p in REFUSAL):
                stats[source]["drop_refusal"] += 1
                return None
        if m["role"] in ("assistant", "system") and FOREIGN_ID.search(m["content"]):
            stats[source]["drop_foreign_identity"] += 1
            return None
    return out


def reservoir(it, k, rng):
    """Uniform sample of up to k items from an iterator (k=None keeps all)."""
    if k is None:
        return list(it)
    res = []
    for i, x in enumerate(it):
        if i < k:
            res.append(x)
        else:
            j = rng.randint(0, i)
            if j < k:
                res[j] = x
    return res


def stream_smoltalk(split):
    ds = load_dataset("HuggingFaceTB/smoltalk2", "SFT", split=split, streaming=True)
    for row in ds:
        stats[split]["seen"] += 1
        kw = row.get("chat_template_kwargs") or {}
        sysmsg = kw.get("custom_instructions") if isinstance(kw, dict) else None
        msgs = row["messages"]
        if msgs and msgs[0].get("role") == "system":
            sysmsg = None  # already has one
        n = normalize(msgs, split, sysmsg)
        if n:
            stats[split]["passed_filters"] += 1
            yield {"messages": n, "source": split}


def stream_hermes():
    src = "hermes3"
    ds = load_dataset("NousResearch/Hermes-3-Dataset", split="train", streaming=True)
    for row in ds:
        stats[src]["seen"] += 1
        n = normalize(row["conversations"], src)
        if n:
            stats[src]["passed_filters"] += 1
            yield {"messages": n, "source": src}


def stream_no_robots():
    src = "no_robots"
    ds = load_dataset("HuggingFaceH4/no_robots", split="train", streaming=True)
    for row in ds:
        stats[src]["seen"] += 1
        n = normalize(row["messages"], src)
        if n:
            stats[src]["passed_filters"] += 1
            yield {"messages": n, "source": src}


PARTS = os.path.expanduser("~/sft-build/parts-quick" if QUICK else "~/sft-build/parts")


def get_tok():
    tok = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B")
    tok.chat_template = INFER_TEMPLATE
    return tok


def length_ok(tok, ex):
    text = tok.apply_chat_template(ex["messages"], tokenize=False)
    n = len(tok(text, add_special_tokens=False)["input_ids"])
    ex["n_tokens"] = n
    return n <= MAX_TOKENS


def plan():
    p = {s: ((lambda s=s: stream_smoltalk(s)), k) for s, k in SMOLTALK2.items()}
    p["hermes3"] = (stream_hermes, HERMES_TARGET)
    p["no_robots"] = (stream_no_robots, None)
    return p


def run_source(name):
    """Stage 1 (one process per source, run in parallel): stream, filter, sample, length-check."""
    os.makedirs(PARTS, exist_ok=True)
    fn, k = plan()[name]
    rng = random.Random(f"{SEED}-{name}")
    tok = get_tok()
    t = time.time()
    over = None if k is None else int(k * 1.15)
    it = itertools.islice(fn(), 3000) if QUICK else fn()
    if QUICK and over:
        over = min(over, 1000)
    picked = reservoir(it, over, rng)
    kept = []
    for ex in picked:
        if length_ok(tok, ex):
            kept.append(ex)
        else:
            stats[name]["drop_too_long"] += 1
    if k is not None:
        kept = kept[: (min(k, 1000) if QUICK else k)]
    stats[name]["kept"] = len(kept)
    with open(f"{PARTS}/{name}.jsonl.tmp", "w") as f:
        for ex in kept:
            f.write(json.dumps(ex) + "\n")
    with open(f"{PARTS}/{name}.stats.json", "w") as f:
        json.dump(dict(stats[name]), f)
    os.rename(f"{PARTS}/{name}.jsonl.tmp", f"{PARTS}/{name}.jsonl")
    log(f"{name}: kept {len(kept)} ({dict(stats[name])}) in {time.time()-t:.0f}s")


def merge():
    """Stage 2: identity + system prompts + BOS flags, shuffle, eval split, write 2 passes of shards."""
    rng = random.Random(SEED)
    os.makedirs(OUT, exist_ok=True)
    tok = get_tok()
    examples = []
    for name in plan():
        with open(f"{PARTS}/{name}.jsonl") as f:
            examples += [json.loads(l) for l in f]
        with open(f"{PARTS}/{name}.stats.json") as f:
            stats[name].update(json.load(f))

    with open(IDENTITY) as f:
        ident = [json.loads(l) for l in f]
    for ex in ident:
        assert length_ok(tok, ex)
    stats["kisoku_identity"]["kept"] = len(ident)
    examples += ident
    log(f"identity: {len(ident)}")

    for ex in examples:
        if ex["messages"][0]["role"] != "system" and rng.random() < ADD_SYSTEM_PROB:
            ex["messages"].insert(0, {"role": "system", "content": rng.choice(KISOKU_SYSTEMS)})
            ex["n_tokens"] += 20
        for i, m in enumerate(ex["messages"]):
            m["bos"] = i == 0

    rng.shuffle(examples)
    eval_set, train = examples[:N_EVAL], examples[N_EVAL:]

    schema = pa.schema([
        ("messages", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string()), ("bos", pa.bool_())]))),
        ("source", pa.string()),
        ("n_tokens", pa.int32()),
    ])

    def write(rows, path):
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), path, compression="zstd")

    def shards(rows, n):
        return [rows[i::n] for i in range(n)]

    idx = 0
    for p in range(2):
        order = train[:]
        random.Random(SEED + 100 + p).shuffle(order)
        for part in shards(order, N_TRAIN_SHARDS_PER_PASS):
            write(part, f"{OUT}/train-{idx:05d}-of-{2*N_TRAIN_SHARDS_PER_PASS:05d}.parquet")
            idx += 1
    for i, part in enumerate(shards(eval_set, N_EVAL_SHARDS)):
        write(part, f"{OUT}/eval-{i:05d}-of-{N_EVAL_SHARDS:05d}.parquet")

    tokens = sum(e["n_tokens"] for e in train)
    by_src = Counter(e["source"] for e in train)
    summary = {
        "train_examples_per_pass": len(train),
        "eval_examples": len(eval_set),
        "train_tokens_per_pass": tokens,
        "mean_tokens": tokens / max(1, len(train)),
        "by_source": dict(by_src),
        "filter_stats": {k: dict(v) for k, v in stats.items()},
        "train_template": TRAIN_TEMPLATE,
        "infer_template": INFER_TEMPLATE,
    }
    with open(f"{OUT}/summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    log("DONE", json.dumps({k: summary[k] for k in ("train_examples_per_pass", "eval_examples", "train_tokens_per_pass", "mean_tokens")}))



if __name__ == "__main__":
    if sys.argv[1] == "--source":
        run_source(sys.argv[2])
    elif sys.argv[1] == "--merge":
        merge()
    elif sys.argv[1] == "--list":
        print(" ".join(plan()))
    # datasets' streaming leaves background threads that can block interpreter exit forever
    # (hung 45 min on 09-22 after the work was done); all output is written and closed by now.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)

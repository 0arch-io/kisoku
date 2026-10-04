"""Build the Kisoku v2 FINAL chat SFT mix (v2 of the builder; build_sft.py made the preview set).

What changed from the preview build:
  - adds the DeepSeek-written set (gen_sft.py output: code, general, honest, math, multiturn, norobots),
    skipping unreadable lines and the 4 math rows that overlap MMLU items (contamination/scan_sft.py);
  - adds tool use: SmolTalk2 xlam_traces + hermes_function_calling_v1, rewritten to one fixed format
    (tools in the system prompt inside <tools>, calls as JSON inside <tool_call>, results in a `tool` turn);
  - smaller caps on the public chat sources so the DeepSeek set is ~10% of the mix instead of ~7%;
  - public rows whose first user message is also a DeepSeek prompt are dropped (no prompt with two answers);
  - MAX_TOKENS 8192 (the model is long-context now; the SFT yml uses max_target_length 8192).

MaxText only knows the roles system / user / assistant and silently drops anything else, so a tool result
is stored as role "user" with tool=True and the chat template prints the header `tool` for it. It is then
masked like any other prompt turn. The per-message `bos` flag works the same way as in the preview build.

Runs anywhere with datasets + pyarrow + transformers. SFT_DIR (default ~/sft-build) must hold identity.jsonl
and deepseek/<source>.jsonl. Usage: --list | --source NAME | --merge   (QUICK=1 for a small test build)
"""
import ast, itertools, json, os, random, re, sys, time
from collections import Counter

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

import build_sft as b
from build_sft import FOREIGN_ID, REFUSAL, log, normalize, reservoir, stats

QUICK = b.QUICK
ROOT = os.path.expanduser(os.environ.get("SFT_DIR", "~/sft-build"))
OUT = f"{ROOT}/kisoku-sft-v2" + ("-quick" if QUICK else "")
PARTS = f"{ROOT}/parts-v2" + ("-quick" if QUICK else "")
IDENTITY = f"{ROOT}/identity.jsonl"
DEEPSEEK_DIR = f"{ROOT}/deepseek"
MAX_TOKENS = 8192
N_EVAL = 200 if QUICK else 2000
N_TRAIN_SHARDS_PER_PASS = 8
N_EVAL_SHARDS = 4
SEED = 4321

SMOLTALK2 = {
    "smoltalk_smollm3_smol_magpie_ultra_no_think": 60000,
    "OpenHermes_2.5_no_think": 25000,
    "smoltalk_smollm3_everyday_conversations_no_think": None,
    "smoltalk_smollm3_systemchats_30k_no_think": None,
    "smoltalk_smollm3_explore_instruct_rewriting_no_think": 15000,
    "smoltalk_smollm3_smol_rewrite_no_think": 20000,
    "smoltalk_smollm3_smol_summarize_no_think": 25000,
    "tulu_3_sft_personas_instruction_following_no_think": None,
    "OpenThoughts3_1.2M_no_think_no_think": 30000,
    "Mixture_of_Thoughts_science_no_think": 15000,
}
HERMES_TARGET = 50000
TOOLS = {"xlam_traces_no_think": 10000, "hermes_function_calling_v1_no_think": None}
DEEPSEEK = ["code", "general", "honest", "math", "multiturn", "norobots"]
# 0-indexed lines of math.jsonl that contain MMLU test items (contamination/sft-overlap result)
DEEPSEEK_DROP = {"math": {1143, 1449, 1576, 2569}}

TOOL_SYSTEM = (
    "You are Kisoku, a helpful AI assistant created by 0ARCH.\n\n"
    "You can call tools. The tools you may use are listed as JSON inside <tools></tools>. "
    "To call a tool, reply with a JSON object holding its name and arguments inside <tool_call></tool_call>, like this:\n"
    '<tool_call>\n{"name": "tool_name", "arguments": {"argument": "value"}}\n</tool_call>\n'
    "Tool results come back inside <tool_response></tool_response>. If no tool fits, answer directly.\n\n"
    "<tools>\n%s\n</tools>"
)
HDR = "('tool' if message['tool'] else message['role'])"
TRAIN_TEMPLATE = (
    "{% for message in messages %}{% if message['bos'] %}{{ '<|begin_of_text|>' }}{% endif %}"
    "{{ '<|start_header_id|>' + " + HDR + " + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' }}"
    "{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
)
COUNT_TEMPLATE = TRAIN_TEMPLATE.replace("message['bos']", "loop.index0 == 0")
TOOLS_BLOCK = re.compile(r"<tools>\s*(.*?)\s*</tools>", re.S)
CALL_BLOCK = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)


def loads_loose(s):
    """JSON first, then a Python literal (the Hermes split stores dict reprs with single quotes)."""
    try:
        return json.loads(s)
    except Exception:
        return ast.literal_eval(s)


def tool_example(row, src):
    """Rewrite one SmolTalk2 tool row to the Kisoku tool format, or return None (reason counted)."""
    kw = row.get("chat_template_kwargs") or {}
    xml = kw.get("xml_tools") or []
    # the Hermes prompts also mention an empty "<tools></tools>" in prose: take the last non-empty block
    found = [x for x in TOOLS_BLOCK.finditer(xml[0]) if x.group(1).strip()] if xml else []
    m = found[-1] if found else None
    if not m:
        stats[src]["drop_no_tools"] += 1
        return None
    try:
        tools = loads_loose(m.group(1))
        assert isinstance(tools, list) and tools
    except Exception:
        stats[src]["drop_bad_tools_json"] += 1
        return None
    msgs = [x for x in row["messages"] if x.get("role") != "system"]
    while msgs and msgs[-1]["role"] != "assistant":
        msgs.pop()  # a conversation must end on an assistant turn
    out = [{"role": "system", "content": TOOL_SYSTEM % json.dumps(tools, ensure_ascii=False), "tool": False}]
    for j, x in enumerate(msgs):
        want = "assistant" if j % 2 else ("user" if j == 0 else None)
        role, content = x["role"], (x.get("content") or "").strip()
        if not content or (want and role != want) or (want is None and role not in ("user", "tool")):
            stats[src]["drop_structure"] += 1
            return None
        if role == "assistant":
            try:
                def fix(mm):
                    call = loads_loose(mm.group(1))
                    return "<tool_call>\n" + json.dumps(
                        {"name": call["name"], "arguments": call.get("arguments", {})}, ensure_ascii=False) + "\n</tool_call>"
                content = CALL_BLOCK.sub(fix, content)
            except Exception:
                stats[src]["drop_bad_call_json"] += 1
                return None
            low = content.lower()
            if any(p in low for p in REFUSAL):
                stats[src]["drop_refusal"] += 1
                return None
            if FOREIGN_ID.search(content):
                stats[src]["drop_foreign_identity"] += 1
                return None
        out.append({"role": "user" if role == "tool" else role, "content": content, "tool": role == "tool"})
    if len(out) < 3 or "<think>" in "".join(x["content"] for x in out):
        stats[src]["drop_structure"] += 1
        return None
    if not any("<tool_call>" in x["content"] for x in out if x["role"] == "assistant"):
        stats[src]["no_call_examples"] += 1  # kept: tools offered, answered directly
    return out


def stream_tools(split):
    for row in load_dataset("HuggingFaceTB/smoltalk2", "SFT", split=split, streaming=True):
        stats[split]["seen"] += 1
        n = tool_example(row, split)
        if n:
            stats[split]["passed_filters"] += 1
            yield {"messages": n, "source": split}


def stream_deepseek(name):
    """The DeepSeek set was written to order, so the refusal filter is off (the `honest` source says
    things like 'I can't browse the web' on purpose); the foreign-identity check stays on."""
    src = f"deepseek_{name}"
    drop = DEEPSEEK_DROP.get(name, set())
    saved, b.REFUSAL[:] = b.REFUSAL[:], []
    try:
        with open(f"{DEEPSEEK_DIR}/{name}.jsonl", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                stats[src]["seen"] += 1
                if i in drop:
                    stats[src]["drop_eval_overlap"] += 1
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    stats[src]["drop_bad_json"] += 1
                    continue
                n = normalize(row["messages"], src)
                if n:
                    stats[src]["passed_filters"] += 1
                    yield {"messages": n, "source": src}
    finally:
        b.REFUSAL[:] = saved


def plan():
    p = {s: ((lambda s=s: b.stream_smoltalk(s)), k) for s, k in SMOLTALK2.items()}
    p["hermes3"] = (b.stream_hermes, HERMES_TARGET)
    p["no_robots"] = (b.stream_no_robots, None)
    for s, k in TOOLS.items():
        p[s] = ((lambda s=s: stream_tools(s)), k)
    for s in DEEPSEEK:
        p[f"deepseek_{s}"] = ((lambda s=s: stream_deepseek(s)), None)
    return p


def get_tok():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B")
    tok.chat_template = COUNT_TEMPLATE
    return tok


def length_ok(tok, ex):
    for m in ex["messages"]:
        m.setdefault("tool", False)
    text = tok.apply_chat_template(ex["messages"], tokenize=False)
    ex["n_tokens"] = len(tok(text, add_special_tokens=False)["input_ids"])
    return ex["n_tokens"] <= MAX_TOKENS


def run_source(name):
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


def first_user(ex):
    for m in ex["messages"]:
        if m["role"] == "user":
            return " ".join(m["content"].split()).lower()
    return ""


def merge():
    rng = random.Random(SEED)
    os.makedirs(OUT, exist_ok=True)
    tok = get_tok()
    by = {}
    for name in plan():
        with open(f"{PARTS}/{name}.jsonl") as f:
            by[name] = [json.loads(l) for l in f]
        with open(f"{PARTS}/{name}.stats.json") as f:
            stats[name].update(json.load(f))

    ds_prompts = {first_user(ex) for n, rows in by.items() if n.startswith("deepseek_") for ex in rows}
    examples = []
    for name, rows in by.items():
        if not name.startswith("deepseek_") and name not in TOOLS:
            keep = [ex for ex in rows if first_user(ex) not in ds_prompts]
            stats[name]["drop_same_prompt_as_deepseek"] = len(rows) - len(keep)
            rows = keep
        examples += rows

    with open(IDENTITY) as f:
        ident = [json.loads(l) for l in f]
    for ex in ident:
        assert length_ok(tok, ex)
    stats["kisoku_identity"]["kept"] = len(ident)
    examples += ident
    log(f"identity: {len(ident)}")

    for ex in examples:
        if ex["messages"][0]["role"] != "system" and rng.random() < b.ADD_SYSTEM_PROB:
            ex["messages"].insert(0, {"role": "system", "content": rng.choice(b.KISOKU_SYSTEMS)})
            ex["n_tokens"] += 20
        for i, m in enumerate(ex["messages"]):
            m["bos"] = i == 0
            m.setdefault("tool", False)
    examples = [ex for ex in examples if ex["n_tokens"] <= MAX_TOKENS]

    rng.shuffle(examples)
    eval_set, train = examples[:N_EVAL], examples[N_EVAL:]
    schema = pa.schema([
        ("messages", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string()),
                                         ("bos", pa.bool_()), ("tool", pa.bool_())]))),
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
    tok_src = Counter()
    for e in train:
        tok_src[e["source"]] += e["n_tokens"]
    summary = {
        "train_examples_per_pass": len(train),
        "eval_examples": len(eval_set),
        "train_tokens_per_pass": tokens,
        "mean_tokens": tokens / max(1, len(train)),
        "max_tokens": MAX_TOKENS,
        "by_source": dict(Counter(e["source"] for e in train)),
        "tokens_by_source": dict(tok_src),
        "filter_stats": {k: dict(v) for k, v in stats.items()},
        "train_template": TRAIN_TEMPLATE,
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
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)  # datasets streaming threads can block exit forever

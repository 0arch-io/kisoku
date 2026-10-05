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
OUT = f"{ROOT}/{os.environ.get('SFT_SET', 'kisoku-sft-v2')}" + ("-quick" if QUICK else "")
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


# ---- Kisoku-specific conversations written by the teacher (gen_kisoku.py -> data/gen2/<category>.jsonl) ----
GEN2_DIR = os.environ.get("GEN2_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "gen2"))
GEN2 = ["persona", "build", "revise", "chat", "explain", "tools", "shortqa", "think_math", "think_logic"]
THINK_SYSTEMS = ["/think", "You are Kisoku, a helpful AI assistant created by 0ARCH. /think", "Think step by step before answering. /think"]
# Fewer public examples than the first two passes, so the Kisoku-specific data is a third of the mix instead of a tenth.
MERGE_CAPS = {"smoltalk_smollm3_smol_magpie_ultra_no_think": 30000, "hermes3": 25000, "OpenHermes_2.5_no_think": 15000,
              "OpenThoughts3_1.2M_no_think_no_think": 15000, "tulu_3_sft_personas_instruction_following_no_think": 15000,
              "smoltalk_smollm3_systemchats_30k_no_think": 15000, "smoltalk_smollm3_smol_summarize_no_think": 12000,
              "smoltalk_smollm3_smol_rewrite_no_think": 10000, "smoltalk_smollm3_explore_instruct_rewriting_no_think": 8000,
              "Mixture_of_Thoughts_science_no_think": 8000, "kisoku2_shortqa": 18000}


def stream_gen2(cat):
    """Turn generator records into training rows. Thinking traces become '<think>...</think>' ONLY under a system prompt that
    ends with /think (plain-text tags, off by default). Tool conversations get the fixed tool system prompt; a tool result is
    stored as a flagged user turn (see the module docstring). Short Q/A pairs become single turns, short multi-question chats,
    and no-tool-needed examples (a tool list in the system prompt, a direct answer)."""
    path = f"{GEN2_DIR}/{cat}.jsonl"
    if not os.path.exists(path):
        return
    rng = random.Random(f"gen2-{cat}")
    pool = None
    for line in open(path):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        src = f"kisoku2_{cat}"
        stats[src]["seen"] += 1
        if cat == "shortqa":
            pairs = r["pairs"]
            if pool is None:
                pool = [json.loads(l)["messages"][0]["content"] for l in open(f"{PARTS}/xlam_traces_no_think.jsonl")][:4000]
            k = rng.random()
            if k < 0.25 and len(pairs) >= 3:  # one short multi-question chat
                m = []
                for q in pairs[: rng.randint(3, 5)]:
                    m += [{"role": "user", "content": q["q"]}, {"role": "assistant", "content": q["a"]}]
                yield {"messages": m, "source": src}
            else:
                for q in pairs:
                    m = [{"role": "user", "content": q["q"]}, {"role": "assistant", "content": q["a"]}]
                    if k >= 0.65:  # tools are available, the question does not need one
                        yield {"messages": [{"role": "system", "content": rng.choice(pool)}] + m, "source": src + "_tools"}
                    else:
                        yield {"messages": m, "source": src}
        elif cat == "tools":
            m = [{"role": "system", "content": TOOL_SYSTEM % json.dumps(r["tools"], ensure_ascii=False), "tool": False}]
            m += [{"role": "user" if x["role"] == "tool" else x["role"], "content": x["content"], "tool": x["role"] == "tool"} for x in r["messages"]]
            yield {"messages": m, "source": src}
        elif cat.startswith("think"):
            # The teacher's own reasoning traces turned out telegraphic and full of notes about our formatting instructions
            # ("We need answer simple ... Need brief working final short"), so they are NOT used as thinking data. The answers
            # were checked (math against the dataset reference, logic by an independent re-solve), so they are kept as plain
            # worked answers. Thinking-mode data comes from SmolTalk2's think splits instead (stream_think).
            yield {"messages": r["messages"], "source": src.replace("think_", "verified_")}
        else:
            yield {"messages": r["messages"], "source": src}


THINK_SPLITS = {"smoltalk_everyday_convs_reasoning_Qwen3_32B_think": 6000, "smoltalk_systemchats_Qwen3_32B_think": 6000,
                "multi_turn_reasoning_if_think": 3000, "table_gpt_Qwen3_32B_think": 2000, "s1k_1.1_think": None}
EMOJI = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF]")
THINK_OK = re.compile(r"^\s*<think>.*?</think>\s*\S", re.S)


def stream_think(split):
    """Thinking mode: SmolTalk2 conversations whose assistant turns open with a <think>...</think> block (traces by Qwen3-32B and
    others). They are only ever shown under a system prompt ending in /think, so the model thinks when asked to and not otherwise."""
    rng = random.Random(f"think-{split}")
    for row in load_dataset("HuggingFaceTB/smoltalk2", "SFT", split=split, streaming=True):
        stats[split]["seen"] += 1
        kw = row.get("chat_template_kwargs") or {}
        msgs = [m for m in row["messages"] if m.get("role") != "system"]
        ok = len(msgs) >= 2 and len(msgs) % 2 == 0
        out = []
        for j, m in enumerate(msgs):
            c = (m.get("content") or "").strip()
            if m.get("role") != ("assistant" if j % 2 else "user") or not c:
                ok = False; break
            if j % 2:
                if not THINK_OK.match(c) or c.count("<think>") != 1:
                    ok = False; break
                final = c.split("</think>", 1)[1].lower()
                if EMOJI.search(final):  # Kisoku's answers carry no emoji in any mode
                    ok = False; break
                if any(p in final for p in REFUSAL) or FOREIGN_ID.search(c):
                    ok = False; break
            else:
                c = b.SLASH_THINK.sub("", c).strip() or c
            out.append({"role": m["role"], "content": c})
        if not ok:
            stats[split]["drop_filtered"] += 1
            continue
        ci = (kw.get("custom_instructions") or "").strip()
        sysmsg = (ci + " /think") if ci and not FOREIGN_ID.search(ci) else rng.choice(THINK_SYSTEMS)
        stats[split]["passed_filters"] += 1
        yield {"messages": [{"role": "system", "content": sysmsg}] + out, "source": split}


ADDITIONS = ["kisoku_idk", "kisoku_known", "kisoku_unknown_term", "kisoku_correction", "kisoku_hold", "tools_not_needed"]


def stream_additions(name):
    """Second-pass additions written by build_additions.py (already in final form, no filtering). Skipped when absent."""
    path = f"{ROOT}/additions/{name}.jsonl"
    if os.path.exists(path):
        for line in open(path):
            stats[name]["seen"] += 1
            yield json.loads(line)


def plan():
    p = {s: ((lambda s=s: b.stream_smoltalk(s)), k) for s, k in SMOLTALK2.items()}
    p["hermes3"] = (b.stream_hermes, HERMES_TARGET)
    p["no_robots"] = (b.stream_no_robots, None)
    for s, k in TOOLS.items():
        p[s] = ((lambda s=s: stream_tools(s)), k)
    for s in DEEPSEEK:
        p[f"deepseek_{s}"] = ((lambda s=s: stream_deepseek(s)), None)
    if os.environ.get("THINK") == "1":
        for sp, k in THINK_SPLITS.items():
            p[sp] = ((lambda sp=sp: stream_think(sp)), k)
    for c in GEN2:
        if os.path.exists(f"{GEN2_DIR}/{c}.jsonl"):
            p[f"kisoku2_{c}"] = ((lambda c=c: stream_gen2(c)), None)
    if os.path.isdir(f"{ROOT}/additions"):
        for s in ADDITIONS:
            p[s] = ((lambda s=s: stream_additions(s)), None)
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


def drop_eval_overlap(examples):
    """Drop every example that contains a verbatim benchmark probe (same patterns.json and matching as
    contamination/scan_sft.py). Needs PATTERNS=/path/patterns.json and pyahocorasick; the build refuses to
    run without it so an unscanned set cannot be produced by accident (set PATTERNS=skip for a test build)."""
    path = os.environ.get("PATTERNS", "")
    if path == "skip" or (QUICK and not path):
        return examples
    import ahocorasick
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "contamination"))
    from common import norm
    pats = [p for p in json.load(open(path))["patterns"] if p["bench"] != "humaneval_solution"]
    A = ahocorasick.Automaton()
    for p in pats:
        A.add_word(p["pat"], p["bench"])
    A.make_automaton()
    keep = []
    for ex in examples:
        hit = next(A.iter(norm("\n".join(m["content"] for m in ex["messages"]))), None)
        if hit:
            stats[ex["source"]]["drop_eval_overlap"] += 1
            stats["_eval_overlap_by_bench"][hit[1]] += 1
        else:
            keep.append(ex)
    log(f"eval-overlap filter: dropped {len(examples) - len(keep)}", dict(stats["_eval_overlap_by_bench"]))
    return keep


def merge():
    rng = random.Random(SEED)
    os.makedirs(OUT, exist_ok=True)
    tok = get_tok()
    by = {}
    capped = os.environ.get("MERGE_CAPS") == "1"
    for name in plan():
        with open(f"{PARTS}/{name}.jsonl") as f:
            by[name] = [json.loads(l) for l in f]
        if capped and name in MERGE_CAPS and len(by[name]) > MERGE_CAPS[name]:
            by[name] = random.Random(f"cap-{name}").sample(by[name], MERGE_CAPS[name])
        with open(f"{PARTS}/{name}.stats.json") as f:
            stats[name].update(json.load(f))

    ds_prompts = {first_user(ex) for n, rows in by.items() if n.startswith("deepseek_") for ex in rows}
    examples = []
    for name, rows in by.items():
        if not name.startswith(("deepseek_", "kisoku2_")) and name not in TOOLS and name not in ADDITIONS and name not in THINK_SPLITS:
            keep = [ex for ex in rows if first_user(ex) not in ds_prompts]
            stats[name]["drop_same_prompt_as_deepseek"] = len(rows) - len(keep)
            rows = keep
        examples += rows

    examples = drop_eval_overlap(examples)

    # identity.jsonl (the September hand-made set) is skipped when IDENTITY=none: the teacher-written persona conversations replace it
    ident = []
    if os.environ.get("IDENTITY") != "none":
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

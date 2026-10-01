#!/usr/bin/env python3
"""Kisoku SFT data generator: real user prompts from public datasets, answers written by a frontier open model on
Ollama cloud (DeepSeek-V4.1-Flash by default). Resumable, budget-capped, 3 workers (Ollama Pro concurrency).

usage: gen_sft.py prep                 # sample prompts into prompts.jsonl
       gen_sft.py run [--limit N] [--budget USD] [--model M]
       gen_sft.py status
Output: out/<source>.jsonl lines {"id","source","messages":[...]} in the chat format build_sft.py expects."""
import argparse, json, os, random, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path.home() / "kisoku-sft-gen"
OUT = ROOT / "out"; OUT.mkdir(parents=True, exist_ok=True)
PROMPTS = ROOT / "prompts.jsonl"
KEY = (Path.home() / ".ollama-key").read_text().strip()
API = "https://ollama.com/api/chat"
RATES = {"deepseek-v4.1-flash": (0.30, 1.20), "glm-5.3-flash": (0.15, 0.50), "gpt-oss:120b": (0.15, 0.60)}  # $/M in, out

# Mix for ~50K examples. Counts are targets; prep samples that many prompts per source.
MIX = {
    "math": 15000, "code": 10000, "general": 12000, "norobots": 3000, "multiturn": 5000, "honest": 3000,
}

SYSTEM = (
    "You are a helpful assistant writing model answers for training a small language model. Write like a knowledgeable "
    "person talking plainly: direct, specific, no filler, no headings unless the answer is long, no emoji. "
    "Never mention your own name, company, or that you are an AI unless the user asks about you. "
    "Math: show the key steps briefly, then end with a line 'Answer: ...'. "
    "Code: give working code in one fenced block with the language tag, then a two or three sentence explanation. "
    "If a question rests on a false premise or asks for something unknowable, say so plainly and give what can be said. "
    "Keep answers under about 500 words unless the task needs more. Never use em dashes; use commas, colons or parentheses. "
    "No horizontal rules, no bold step headings; for math keep the working compact, plain arithmetic where possible, and use LaTeX only when notation truly needs it."
)

HONEST_SEEDS = [
    "a question that assumes a false historical fact", "a question about a private person the assistant cannot know",
    "a question about an event after the model's training data", "a math question with missing information",
    "a request to predict a specific future stock price", "a question with a made-up scientific term",
    "a question asking for a source the assistant cannot verify", "a question that has no single correct answer",
    "a question with a subtle false premise about geography", "a question that assumes a product feature that does not exist",
    "a medical question that needs a doctor", "a question about the assistant's own training data",
]


def boxed(sol):
    """Last \\boxed{...} in a solution, or None."""
    i = sol.rfind("\\boxed{")
    if i < 0: return None
    j, depth = i + 7, 1
    while j < len(sol) and depth:
        depth += {"{": 1, "}": -1}.get(sol[j], 0); j += 1
    return sol[i + 7: j - 1].strip() if depth == 0 else None


def norm(a):
    a = a.strip().rstrip(".").replace("$", "").replace("\\boxed", "").replace("\\text", "").replace("{", "").replace("}", "")
    a = a.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac").replace(" ", "").replace(",", "").lower()
    try: return f"{float(a):.6g}"
    except ValueError: return a


def load(name, split, config=None):
    from datasets import load_dataset
    return load_dataset(name, config, split=split) if config else load_dataset(name, split=split)


def prep(seed=0):
    random.seed(seed)
    rows = []

    def add(source, items):
        random.shuffle(items)
        for i, m in enumerate(items[: MIX[source]]):
            rows.append({"id": f"{source}-{i:06d}", "source": source, "messages": m})
        print(source, min(len(items), MIX[source]), flush=True)

    ds = load("AI-MO/NuminaMath-CoT", "train")
    # School-level sources only (a 1.6B student gains nothing from olympiad problems); keep the reference answer for checking.
    ds = ds.filter(lambda r: r["source"] in ("gsm8k", "orca_math", "synthetic_math", "math", "cn_k12"))
    items = []
    for r in ds.shuffle(seed=seed).select(range(min(len(ds), MIX["math"] * 3))):
        ref = boxed(r["solution"])
        if ref and 30 < len(r["problem"]) < 1500:
            items.append([{"role": "user", "content": r["problem"]}, {"role": "ref", "content": ref}])
    add("math", items)
    ds = load("ise-uiuc/Magicoder-OSS-Instruct-75K", "train")
    add("code", [[{"role": "user", "content": r["problem"]}] for r in ds if 50 < len(r["problem"]) < 2500])
    ds = load("HuggingFaceTB/smoltalk2", "smoltalk_smollm3_smol_magpie_ultra_no_think", config="SFT")
    ds = ds.shuffle(seed=seed).select(range(min(len(ds), MIX["general"] * 3)))
    add("general", [[{"role": "user", "content": r["messages"][0]["content"]}] for r in ds if r["messages"][0]["role"] == "user" and 20 < len(r["messages"][0]["content"]) < 3000])
    ds = load("HuggingFaceH4/no_robots", "train")
    add("norobots", [[m for m in r["messages"] if m["role"] != "assistant"][:1] for r in ds if r["messages"][0]["role"] == "user"])
    ds = load("HuggingFaceTB/smoltalk2", "smoltalk_smollm3_everyday_conversations_no_think", config="SFT")
    ds = ds.shuffle(seed=seed).select(range(min(len(ds), MIX["multiturn"] * 2)))
    add("multiturn", [[m for m in r["messages"] if m["role"] == "user"][:4] for r in ds if sum(m["role"] == "user" for m in r["messages"]) >= 2])
    add("honest", [[{"role": "meta", "content": random.choice(HONEST_SEEDS)}] for _ in range(MIX["honest"])])
    with PROMPTS.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("total prompts", len(rows))


class Spend:
    def __init__(self, model, cap):
        self.model, self.cap, self.lock = model, cap, threading.Lock()
        self.inp = self.out = 0; self.n = 0
        f = ROOT / "spend.json"
        if f.exists():
            d = json.loads(f.read_text()); self.inp, self.out, self.n = d["in"], d["out"], d["n"]

    def usd(self):
        ri, ro = RATES.get(self.model, (1.0, 4.0))
        return self.inp / 1e6 * ri + self.out / 1e6 * ro

    def add(self, i, o):
        with self.lock:
            self.inp += i; self.out += o; self.n += 1
            (ROOT / "spend.json").write_text(json.dumps({"in": self.inp, "out": self.out, "n": self.n, "usd": round(self.usd(), 2)}))
            return self.usd() < self.cap


def chat(model, messages, max_tokens=1200):
    for attempt in range(5):
        try:
            r = requests.post(API, headers={"Authorization": f"Bearer {KEY}"}, timeout=300,
                              json={"model": model, "messages": messages, "stream": False, "think": False,
                                    "options": {"temperature": 0.7, "num_predict": max_tokens}})
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(5 * (attempt + 1)); continue
            r.raise_for_status()
            j = r.json()
            if j.get("done_reason") == "length":  # cut off mid-answer: count the spend, drop the example
                return None, j.get("prompt_eval_count", 0), j.get("eval_count", 0)
            return j["message"]["content"].strip(), j.get("prompt_eval_count", 0), j.get("eval_count", 0)
        except (requests.RequestException, KeyError, ValueError) as e:
            time.sleep(5 * (attempt + 1))
    return None, 0, 0


def make(model, row, spend):
    src, msgs = row["source"], row["messages"]
    if src == "honest":
        ask = (f"Write one realistic user question that is {msgs[0]['content']}, then the ideal honest reply. "
               "Return JSON with keys \"question\" and \"answer\" and nothing else.")
        txt, i, o = chat(model, [{"role": "system", "content": SYSTEM}, {"role": "user", "content": ask}], 900)
        if txt is None: return None, i, o
        try:
            t = txt[txt.index("{"): txt.rindex("}") + 1]; d = json.loads(t)
            return [{"role": "user", "content": d["question"].strip()}, {"role": "assistant", "content": d["answer"].strip()}], i, o
        except (ValueError, KeyError):
            return None, i, o
    ref = next((m["content"] for m in msgs if m["role"] == "ref"), None)
    msgs = [m for m in msgs if m["role"] != "ref"]
    conv, ti, to = [], 0, 0
    for turn in msgs:  # single-turn sources have one user turn; multiturn re-answers each user turn in context
        conv.append(turn)
        txt, i, o = chat(model, [{"role": "system", "content": SYSTEM}] + conv, 2500 if src == "math" else 1200)
        ti += i; to += o
        if txt is None: return None, ti, to
        conv.append({"role": "assistant", "content": txt})
    if ref is not None:  # math: keep only answers whose final line agrees with the dataset's reference answer
        m = [l for l in conv[-1]["content"].splitlines() if l.strip().lower().startswith("answer:")]
        if not m or norm(m[-1].split(":", 1)[1]) != norm(ref):
            return None, ti, to
    return conv, ti, to


def run(model, limit, budget):
    done = set()
    for f in OUT.glob("*.jsonl"):
        for line in f.open():
            try: done.add(json.loads(line)["id"])
            except ValueError: pass
    rows = [json.loads(l) for l in PROMPTS.open()]
    rows = [r for r in rows if r["id"] not in done]
    random.Random(1).shuffle(rows)  # interleave sources so a partial run is still a balanced mix
    if limit: rows = rows[:limit]
    spend = Spend(model, budget)
    print(f"{len(done)} done, {len(rows)} to go, spent so far ${spend.usd():.2f}, cap ${budget}", flush=True)
    lock = threading.Lock(); stop = threading.Event(); count = [0]

    def work(row):
        if stop.is_set(): return
        conv, i, o = make(model, row, spend)
        if not spend.add(i, o): stop.set(); print("BUDGET CAP REACHED", flush=True)
        if conv is None: return
        with lock:
            with (OUT / f"{row['source']}.jsonl").open("a") as f:
                f.write(json.dumps({"id": row["id"], "source": row["source"], "teacher": model, "messages": conv}, ensure_ascii=False) + "\n")
            count[0] += 1
            if count[0] % 100 == 0: print(f"{count[0]} written, ${spend.usd():.2f}", flush=True)

    with ThreadPoolExecutor(3) as ex:
        list(ex.map(work, rows))
    print(f"finished: {count[0]} written this run, total spend ${spend.usd():.2f}", flush=True)


def status():
    c = {f.stem: sum(1 for _ in f.open()) for f in OUT.glob("*.jsonl")}
    print(json.dumps({"written": c, "total": sum(c.values()), "spend": json.loads((ROOT / "spend.json").read_text()) if (ROOT / "spend.json").exists() else None}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--budget", type=float, default=55.0); ap.add_argument("--model", default="deepseek-v4.1-flash")
    a = ap.parse_args()
    {"prep": lambda: prep(), "run": lambda: run(a.model, a.limit, a.budget), "status": status}[a.cmd]()

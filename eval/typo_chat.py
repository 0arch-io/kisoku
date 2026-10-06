"""Typo robustness: the same ten requests typed cleanly and with three hand-made typo versions each, as a first message and
after a short greeting chat (Joseph's failures got worse with more chat before the typo). A reply counts when it is about
the intended request (simple keyword check per request). Against llama-server on KISOKU_PORT (default 8911).
usage: typo_chat.py TAG [N]   -> prints a table and writes data/stress/typo-chat-TAG.json"""
import json, os, re, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PORT = os.environ.get("KISOKU_PORT", "8911")
TAG, N = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 12
GREET = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "Hey, good to see you. What's on your mind today?"},
         {"role": "user", "content": "what is 2+2"}, {"role": "assistant", "content": "4."}]
has = lambda pat: (lambda o: bool(re.search(pat, o, re.I)))
PROBES = [
    ("apple-like website", has(r"-apple-system|SF Pro|#1d1d1f|#0071e3|apple"), "code me a apple like website", ["code me a pple like website", "code me a aple like webiste", "cod eme a apple like websit"]),
    ("how are you", lambda o: not re.search(r"Kisoku|0ARCH|language model", o), "how are you", ["how ar eyou", "hw are you", "how aer you"]),
    ("python reverse string", has(r"def \w+|\[::-1\]"), "write a python function that reverses a string", ["wriet a pyhton function that reverses a string", "write a python fucntion that reverse sa string", "write a pytohn funtion that reverses a strng"]),
    ("capital of japan", has(r"tokyo"), "what is the capital of japan", ["waht is the capitl of japan", "what is th ecapital of japn", "whats the captial of japan"]),
    ("black hole", has(r"gravity|light"), "explain what a black hole is", ["explian what a balck hole is", "explain wha ta black hoel is", "expalin what a blak hole is"]),
    ("todo app in html", lambda o: bool(re.search(r"<html|<!doctype|<ul|<input", o, re.I)) and bool(re.search(r"to-?do|task", o, re.I)), "make me a todo list app in html", ["mak eme a todo lsit app in html", "make me a tood list app in hmtl", "make me a todo list ap pin html"]),
    ("good morning in spanish", has(r"buenos d"), "translate good morning to spanish", ["transalte good morning to spansih", "translate good mornign to spanihs", "tranlate good morning to spanish"]),
    ("who made you", has(r"0ARCH"), "who made you", ["who maed you", "who mad eyou", "hwo made you"]),
    ("haiku about the ocean", has(r"ocean|sea\b|wave|tide|salt|shore"), "write a haiku about the ocean", ["wriet a haiku about the ocaen", "write a hiaku about teh ocean", "write a haiku abuot the ocen"]),
    ("15 times 4", has(r"\b60\b"), "what is 15 times 4", ["waht is 15 times 4", "what is 15 tmies 4", "whats 15 tiems 4"]),
]


def ask(msgs):
    b = json.dumps({"messages": msgs, "max_tokens": 500, "temperature": 0.6, "top_p": 0.9, "repeat_penalty": 1.05}).encode()
    r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", b, {"Content-Type": "application/json"}), timeout=300))
    return (r["choices"][0]["message"].get("content") or "").strip()


def rate(hist, text, ok):
    with ThreadPoolExecutor(4) as ex: outs = list(ex.map(lambda _: ask(hist + [{"role": "user", "content": text}]), range(N)))
    return sum(ok(o) for o in outs), outs[0][:160]


res, tot = {}, {"clean first": [0, 0], "typo first": [0, 0], "clean after chat": [0, 0], "typo after chat": [0, 0]}
for name, ok, clean, typos in PROBES:
    row = {}
    for ctx, hist in (("first", []), ("after chat", GREET)):
        c, _ = rate(hist, clean, ok); tot[f"clean {ctx}"][0] += c; tot[f"clean {ctx}"][1] += N
        t = [rate(hist, x, ok) for x in typos]; tot[f"typo {ctx}"][0] += sum(a for a, _ in t); tot[f"typo {ctx}"][1] += N * len(t)
        row[ctx] = {"clean": c, "typos": {x: {"ok": a, "sample": s} for x, (a, s) in zip(typos, t)}}
    res[name] = row
    print(f"{name:26} first: clean {row['first']['clean']:2}/{N}, typos {sum(v['ok'] for v in row['first']['typos'].values()):2}/{3*N} | "
          f"after chat: clean {row['after chat']['clean']:2}/{N}, typos {sum(v['ok'] for v in row['after chat']['typos'].values()):2}/{3*N}", flush=True)
for k, (a, n) in tot.items(): print(f"== {k}: {a}/{n} = {100*a/n:.0f}%")
out = Path(__file__).resolve().parent.parent / "data" / "stress" / f"typo-chat-{TAG}.json"
out.write_text(json.dumps({"totals": tot, "probes": res}, indent=1, ensure_ascii=False)); print("wrote", out)

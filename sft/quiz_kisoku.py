"""Quiz a Kisoku chat model on long-tail questions with known answers (R-tuning step 1).
Sources: PopQA (entity-attribute questions), definitional 'Who is / What is X?' questions built from PopQA's
low-popularity subjects (graded by whether the description mentions the known attribute), and NQ-open train.
None of these is one of our eval benchmarks. Output: quiz.jsonl with the model's answer and a grade
(right / wrong / abstain). Needs `llama-server -m <gguf> --port 8911 -np 8 -c 8192 --jinja` running.
usage: quiz_kisoku.py OUT.jsonl [LIMIT]"""
import json, os, random, re, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
from datasets import load_dataset
PORT = os.environ.get("KISOKU_PORT", "8911")   # llama-server port of the model under test

OUT = sys.argv[1]; LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else None
ABSTAIN = re.compile(r"\b(i don'?t (know|have)|i do not (know|have)|i'?m not (sure|familiar|aware)|i am not (sure|familiar|aware)|"
                     r"not familiar with|no (reliable )?(information|data|record)|can'?t (confirm|verify|find)|cannot (confirm|verify|find)|"
                     r"couldn'?t find|could you (clarify|provide)|not enough information)\b", re.I)
norm = lambda s: " " + re.sub(r"[^a-z0-9]+", " ", s.lower()).strip() + " "


def grade(ans, golds):
    a = norm(ans)
    if any(len(g.strip()) >= 2 and norm(g) in a for g in golds):
        return "right"
    return "abstain" if ABSTAIN.search(ans) else "wrong"


def questions():
    rng = random.Random(7)
    pop = load_dataset("akariasai/PopQA", split="test")
    qs = []
    for r in pop:
        golds = json.loads(r["possible_answers"])
        qs.append({"kind": "attr", "q": r["question"], "gold": golds, "subj": r["subj"], "prop": r["prop"], "obj": r["obj"], "s_pop": r["s_pop"]})
        if r["s_pop"] < 1500 and r["prop"] in ("occupation", "sport", "genre", "author", "director", "country"):
            q = ("Who is %s?" if r["prop"] in ("occupation", "sport") else "What is %s?") % r["subj"]
            qs.append({"kind": "def", "q": q, "gold": golds, "subj": r["subj"], "prop": r["prop"], "obj": r["obj"], "s_pop": r["s_pop"]})
    nq = load_dataset("google-research-datasets/nq_open", split="train")
    for i in rng.sample(range(len(nq)), 8000):
        q = nq[i]["question"].strip()
        qs.append({"kind": "nq", "q": q[0].upper() + q[1:] + ("" if q.endswith("?") else "?"), "gold": nq[i]["answer"]})
    rng.shuffle(qs)
    return qs[:LIMIT] if LIMIT else qs


def ask(item):
    b = json.dumps({"messages": [{"role": "user", "content": item["q"]}], "max_tokens": 90, "temperature": 0}).encode()
    for _ in range(3):
        try:
            r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://localhost:{PORT}/v1/chat/completions", b, {"Content-Type": "application/json"}), timeout=120))
            c = r["choices"][0]
            item["answer"] = (c["message"].get("content") or "").strip(); item["finish"] = c["finish_reason"]
            item["grade"] = grade(item["answer"], item["gold"])
            return item
        except Exception as e:
            err = repr(e)
    item["answer"] = ""; item["grade"] = "error"; item["finish"] = err
    return item


if __name__ == "__main__":
    done = set()
    if os.path.exists(OUT):
        done = {json.loads(l)["q"] for l in open(OUT)}
    todo = [q for q in questions() if q["q"] not in done]
    print("to ask:", len(todo), "already done:", len(done), flush=True)
    with open(OUT, "a") as f, ThreadPoolExecutor(8) as ex:
        for n, item in enumerate(ex.map(ask, todo), 1):
            f.write(json.dumps(item) + "\n")
            if n % 1000 == 0:
                f.flush(); print(n, flush=True)
    sys.stdout.flush(); os._exit(0)

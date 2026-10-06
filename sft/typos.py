"""Typing noise for user messages (build_sft_v2.py merge, TYPOS=<share of examples>, e.g. TYPOS=0.25).

Joseph's own chats kept tripping the chat model on ordinary typos ("how ar eyou", "hw are you", "code me a pple like website"):
the teacher wrote tidy user messages, so the model rarely saw a misspelled request with the reply to the intended one. This adds
the kind of slips people make on a keyboard to a share of short user messages and leaves the assistant's reply untouched.
Left alone: long or pasted messages, code, quotes, numbers, capitalised words (names in factual questions), and any request
where the exact spelling is the point (spelling, grammar, translation, letter counting)."""
import random, re

EXACT = re.compile(r"spell|letter|typo|gramm|proofread|correct|fix|edit|translat|rewrit|rephras|anagram|palindrom|rhym|count|regex|password|synonym|acronym|abbrev|pronounc|"
                   r"what does .* mean|meaning of|define|definition|word", re.I)
SKIP_SOURCES = ("kisoku_idk", "kisoku_known", "kisoku_unknown_term", "kisoku_correction", "kisoku_hold", "kisoku_onpolicy", "deepseek_honest", "kisoku2_shortqa", "kisoku2_tools",
                "kisoku2_verified", "xlam", "hermes_function_calling", "tools_not_needed")
NEAR = {"a": "sq", "s": "ad", "d": "sf", "e": "wr", "r": "et", "t": "ry", "i": "uo", "o": "ip", "n": "bm", "m": "n", "l": "k", "c": "xv", "u": "yi", "h": "gj", "g": "fh"}
WORD = re.compile(r"(?<![\w'/.@#-])[a-z]{3,}(?![\w'/.@-])")


def slip(word, nxt, rng):
    """One keyboard slip on a lowercase word; nxt is True when another word follows (allows a moved space)."""
    i = rng.randrange(1, len(word))
    kind = rng.choices(["swap", "drop", "double", "split", "near", "shift_space"], [3, 3, 1, 2, 1, 2 if nxt else 0])[0]
    if kind == "swap" and i < len(word) - 0: return word[:i - 1] + word[i] + word[i - 1] + word[i + 1:], False
    if kind == "drop": return word[:i] + word[i + 1:], False
    if kind == "double": return word[:i] + word[i - 1] + word[i:], False
    if kind == "split": return word[:i] + " " + word[i:], False
    if kind == "near" and word[i] in NEAR: return word[:i] + rng.choice(NEAR[word[i]]) + word[i + 1:], False
    if kind == "shift_space": return word[:-1] + " " + word[-1], True     # "are you" -> "ar eyou": the caller removes the following space
    return word[:i] + word[i + 1:], False


def noisy(text, rng):
    """Return text with one or two slips, or None when the message should be left alone."""
    if len(text) > 160 or len(text) < 6 or any(ch in text for ch in "`\"“”<>{}\\=") or "\n" in text or EXACT.search(text):
        return None
    spans = [m for m in WORD.finditer(text)]
    if not spans: return None
    out, n = text, 0
    for m in sorted(rng.sample(spans, min(len(spans), 1 if rng.random() < 0.7 else 2)), key=lambda m: -m.start()):
        follows = out[m.end():m.end() + 2]
        nxt = len(follows) == 2 and follows[0] == " " and follows[1].islower()
        new, eat = slip(m.group(), nxt, rng)
        out = out[:m.start()] + new + out[m.end() + (1 if eat else 0):]; n += 1
    return out if out != text else None


def add_typos(ex, rng, share):
    """Maybe return a copy of the example with slips in one user message (never a system, tool or assistant message)."""
    if ex["source"].startswith(SKIP_SOURCES) or rng.random() >= share:
        return ex, False
    idx = [i for i, m in enumerate(ex["messages"]) if m["role"] == "user" and not m.get("tool")]
    rng.shuffle(idx)
    for i in idx[:3]:
        new = noisy(ex["messages"][i]["content"], rng)
        if new:
            msgs = [dict(m) for m in ex["messages"]]; msgs[i]["content"] = new
            return {**ex, "messages": msgs}, True
    return ex, False


if __name__ == "__main__":
    r = random.Random(0)
    for t in ["how are you", "code me a apple like website", "can you make it shorter", "what is the capital of France?", "whats 12 times 12", "how do you spell necessary",
              "yo can u make a todo app pls", "hello", "explain recursion like im five", "who made you"]:
        print(repr(t), "->", [noisy(t, r) for _ in range(5)])

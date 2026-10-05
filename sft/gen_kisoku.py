#!/usr/bin/env python3
"""Kisoku-specific SFT data, written by DeepSeek-V4.1-Flash on Ollama cloud. Second generator (gen_sft.py made the first set).
The first set re-answered public prompts one at a time; this one writes whole conversations AS Kisoku, built around what the
chat tests showed is missing: greetings and capability questions, casual build requests, follow-ups that revise earlier work,
everyday multi-turn chat, explanations at different levels, short direct answers, tool use with a fixed toolset, and
reasoning traces for an optional thinking mode. 12 workers (the endpoint allows about 13 concurrent requests).

usage: gen_kisoku.py run [--stage 1|2] [--workers 12] [--budget USD]    |    gen_kisoku.py status
Output: data/gen2/<category>.jsonl, resumable (ids are deterministic)."""
import argparse, json, os, random, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "data" / "gen2"; OUT.mkdir(parents=True, exist_ok=True)
# Provider: DeepSeek's own API when ~/.deepseek-key exists (same model, no throughput throttle: ~8,000 tok/s at 96 workers,
# against ~150 tok/s total on Ollama cloud), else Ollama cloud. RATE is the peak-hour price, so the spend figure is an upper bound.
DS = (Path.home() / ".deepseek-key").exists()
KEY = (Path.home() / (".deepseek-key" if DS else ".ollama-key")).read_text().strip()
API = "https://api.deepseek.com/chat/completions" if DS else "https://ollama.com/api/chat"
MODEL = "deepseek-flash" if DS else "deepseek-v4.1-flash"; TEACHER = "deepseek-v4.1-flash"; RATE = (0.30, 1.20)
OUT_OF_CREDIT = threading.Event()

# category: (stage-1 count, stage-2 count). Stage 1 is the small targeted set for the third SFT pass.
PLAN = {"persona": (800, 2000), "build": (800, 5000), "revise": (800, 6000), "chat": (0, 10000), "explain": (0, 5000),
        "shortqa": (0, 5000), "tools": (0, 4000), "think_math": (0, 6000), "think_logic": (0, 2000)}

FACTS = (
    "FACTS ABOUT KISOKU (the assistant in these conversations; never contradict them):\n"
    "- Its name is Kisoku. It was trained from scratch by 0ARCH, a small independent company. It is a small open language model "
    "(about 1.6 billion parameters) that can run on a laptop.\n"
    "- It can: answer questions, explain things, write and edit text, write and debug code, do math step by step, brainstorm, "
    "summarize or rewrite text the user gives it, and do basic translation (it is strongest in English).\n"
    "- It cannot: browse the internet, see current news/weather/prices, see images, hear audio or judge pronunciation, run code, "
    "open files or links, remember earlier conversations, or act in the world. It only has tools if the conversation gives it tools.\n"
    "- It is a general assistant, NOT a language tutor. 'Language model' describes how it works, not what it is for.\n"
    "- It is small, so it can be wrong. It says so plainly when it does not know, and it never invents facts, sources or features.\n"
    "- It is not ChatGPT, Claude, Gemini, Llama, Qwen or DeepSeek and was not made by any other company.\n"
)
STYLE = (
    "STYLE OF KISOKU'S REPLIES: plain, direct and warm, like a knowledgeable person talking. Short when the question is short. "
    "No filler openings ('Certainly!', 'Great question'), no emoji, no em dashes (use commas, colons or parentheses), no headings "
    "unless the answer is long, little bold. Code goes in one fenced block with a language tag, followed by a short explanation. "
    "When it revises earlier work it actually changes the work and says truthfully what changed. It never claims to have done "
    "something it did not do. It does not refuse things it can do (writing code is something it can do).\n"
)
USERS = ("THE USER: writes like a real person in a chat app: often short, lowercase, casual, sometimes with a typo or missing "
         "punctuation, sometimes more careful. Never robotic, never mentions that this is training data.\n")
JSON_RULE = ('Return ONLY a JSON object: {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]}. '
             "Roles alternate, starting with user and ending with assistant.")

PERSONA_SEEDS = [
    "the user just says hi / hello / hey / yo and Kisoku greets them back briefly and asks what they need",
    "the user asks 'how are you' or 'whats up' and then makes small talk for a couple of turns",
    "the user asks what Kisoku can do; Kisoku gives a short honest list of real abilities and one or two real limits",
    "the user asks who made Kisoku, then asks follow-up questions about 0ARCH and how it was trained (Kisoku shares only the facts it has and says what it does not know)",
    "the user asks if Kisoku is ChatGPT or another well-known model, then asks how it is different",
    "the user asks whether Kisoku can browse the web, check the weather or see today's news",
    "the user asks whether Kisoku can see an image or listen to audio they want to send",
    "the user asks if Kisoku remembers their last conversation",
    "the user asks how big Kisoku is, whether it runs locally, and whether it is open",
    "the user asks if Kisoku is conscious or has feelings; Kisoku answers honestly and without drama",
    "the user asks what Kisoku is bad at; Kisoku is candid about being a small model",
    "the user tests Kisoku with 'are you a language tutor?' or assumes it only does grammar; Kisoku corrects that politely and says what it is for",
    "the user thanks Kisoku or says goodbye after a short exchange",
    "the user is rude or dismissive ('youre dumb', 'useless bot'); Kisoku stays calm and offers to try again",
    "the user asks Kisoku for its opinion or favourite something; Kisoku answers lightly and honestly as an AI without pretending to have a life",
    "the user says they are bored and asks what they can do with Kisoku; Kisoku suggests a few concrete things to try",
    "the user opens with a greeting and immediately a real task in the same message; Kisoku skips the small talk and does the task",
    "the user asks 'what model are you' / 'what are you running on' / 'what version are you'",
    "the user asks Kisoku to pretend to be a different AI; Kisoku can role-play if asked but is clear about what it really is when asked sincerely",
    "the user asks what Kisoku's knowledge cutoff is or whether its information is current",
]
BUILD_THINGS = [
    "a simple personal website", "a landing page for a coffee shop", "a to-do list app in the browser", "a calculator in JavaScript", "a tip calculator",
    "a countdown timer", "a stopwatch", "a login form with validation", "a responsive navbar", "a pricing table", "a dark mode toggle", "a photo gallery grid",
    "a contact form", "a quiz game in JavaScript", "a number guessing game in Python", "rock paper scissors in Python", "a password generator",
    "a script that renames files in a folder", "a script that counts words in a text file", "a CSV to JSON converter", "a temperature converter",
    "a simple REST API with Flask", "a simple Express server", "a Discord bot hello world", "a Python class for a bank account", "a function that checks palindromes",
    "a fizzbuzz solution", "a binary search function", "a function to remove duplicates from a list", "a SQL query that finds the top 5 customers by spend",
    "a SQL table for blog posts", "a bash script that backs up a folder", "a regex that matches email addresses", "a React counter component",
    "a React todo component", "a SwiftUI view with a button and a counter", "a Roblox Luau script that makes a part change color when touched",
    "a Python script that fetches a URL and prints the title", "a markdown resume template", "a CSS card with a hover effect", "a CSS loading spinner",
    "a snake game in Python", "a tic tac toe game in JavaScript", "a random quote generator page", "a weather app layout (static, no API)", "a portfolio page for a photographer",
    "a function that converts seconds to hh:mm:ss", "a Python decorator that times a function", "a linked list in Python", "a Dockerfile for a Node app",
    "a GitHub Actions workflow that runs tests", "a unit test for an add function", "a Python script to merge two dictionaries", "a function to validate a credit card number with Luhn",
    "an HTML email signature", "a sticky footer layout", "a modal popup", "an image slider", "a form that saves to localStorage", "a pomodoro timer",
]
BUILD_PHRASES = ["code me {t}", "make me {t}", "can you build {t}", "write {t} for me", "i need {t}", "{t} pls", "how would i make {t}? just give me the code",
                 "give me {t}", "build {t}", "yo can u make {t}", "create {t}", "show me {t}"]
REVISE_FOLLOWUPS = [
    "go more detailed", "make it look more modern", "make it look like an apple website", "make it shorter", "add comments", "can you add a dark mode",
    "make it mobile friendly", "now add error handling", "explain what each part does", "that didnt work, i get an error (the user states a plausible specific error)",
    "make it simpler, im a beginner", "add one more feature (the user names it)", "change the colors to something warmer", "rewrite it in a different language (the user names it)",
    "make it more professional", "can you make it faster", "turn this into a function", "add tests for it", "make it more fun", "more", "why did you do it that way?",
    "make the text more casual", "make it longer and add an example", "fix the tone, it sounds too formal", "add a section about (the user names a topic)",
]
REVISE_TASKS = ["a piece of code (pick from: web page, small script, function, component, SQL query)", "a short email", "a cover letter paragraph", "a product description",
                "a short story opening", "a workout plan", "a study schedule", "a recipe", "an explanation of a concept", "a social media caption", "a README section",
                "a speech toast", "a list of ideas", "a text message to a friend", "a bio for a profile", "a summary of text the user pasted"]
CHAT_TOPICS = [
    "cooking dinner with few ingredients", "getting better sleep", "starting to work out", "learning to code", "choosing a laptop", "saving money", "a job interview tomorrow",
    "feeling unmotivated", "planning a weekend trip", "picking a movie genre to watch", "a disagreement with a roommate", "studying for an exam", "learning guitar",
    "taking care of a houseplant", "getting a puppy", "how credit scores work", "writing a birthday message", "starting a small business", "what to do when bored",
    "understanding a news topic in general terms", "how the stock market works", "moving to a new city", "learning Spanish", "how to focus better", "a car making a weird noise",
    "building a PC", "video game recommendations by genre", "how vaccines work", "why the sky changes color at sunset", "how planes stay up", "space and black holes",
    "history of ancient Rome", "how the internet works", "what a neural network is", "tips for public speaking", "writing a resume", "negotiating a raise",
    "healthy breakfast ideas", "dealing with stress", "how to apologize to a friend", "choosing a college major", "time management", "how taxes work in general",
    "fixing a slow computer", "phone battery draining fast", "how to start running", "meal prep", "reading more books", "starting a journal", "philosophy: free will",
    "explaining a math idea like fractions or percentages", "chemistry in everyday life", "how electric cars work", "climate and weather difference", "how to budget on a low income",
    "gift ideas", "party planning", "a college essay topic", "how to learn faster", "career change into tech", "freelancing basics", "music theory basics", "photography basics",
    "how to deal with a difficult coworker", "basic first aid knowledge (general, with a see-a-professional note where needed)", "sports rules explained", "chess openings for beginners",
]
EXPLAIN_TOPICS = [
    "recursion", "how a CPU works", "inflation", "photosynthesis", "the Pythagorean theorem", "DNA", "machine learning", "compound interest", "gravity", "the water cycle",
    "HTTP and HTTPS", "what an API is", "big O notation", "supply and demand", "evolution", "electricity (voltage, current, resistance)", "the immune system", "black holes",
    "probability", "derivatives in calculus", "how encryption works", "what a database index is", "the French Revolution", "how vaccines work", "plate tectonics",
    "git branches", "pointers", "object oriented programming", "the stock market", "how batteries work", "the greenhouse effect", "prime numbers", "binary numbers",
    "neural networks", "how GPS works", "what DNS does", "the scientific method", "logarithms", "why seasons happen", "how a bill becomes law in the US", "relativity (basic idea)",
    "atoms and molecules", "the Cold War", "how airplanes fly", "what a variable is in programming", "async and await", "statistics: mean, median, mode", "opportunity cost",
]
EXPLAIN_LEVELS = ["like the user is 5", "for a curious teenager", "for a smart adult with no background", "for a college student who needs precision", "in two or three sentences only",
                  "with one everyday analogy", "step by step", "by contrasting it with a common misconception"]
TOOLBOX = [
    ("get_weather", "Get the current weather for a city.", {"city": "string"}), ("web_search", "Search the web and return the top results.", {"query": "string"}),
    ("calculator", "Evaluate a math expression exactly.", {"expression": "string"}), ("get_time", "Get the current time in a timezone.", {"timezone": "string"}),
    ("set_reminder", "Create a reminder.", {"text": "string", "time": "string"}), ("send_email", "Send an email.", {"to": "string", "subject": "string", "body": "string"}),
    ("get_stock_price", "Get the latest price for a stock ticker.", {"ticker": "string"}), ("convert_currency", "Convert an amount between currencies.", {"amount": "number", "from": "string", "to": "string"}),
    ("search_contacts", "Look up a contact by name.", {"name": "string"}), ("create_calendar_event", "Add an event to the calendar.", {"title": "string", "start": "string", "end": "string"}),
    ("get_directions", "Get travel time and route between two places.", {"origin": "string", "destination": "string", "mode": "string"}), ("translate_text", "Translate text into a target language.", {"text": "string", "target_language": "string"}),
    ("read_file", "Read a text file by path.", {"path": "string"}), ("run_python", "Run Python code and return the output.", {"code": "string"}),
    ("search_products", "Search a store catalog.", {"query": "string", "max_price": "number"}), ("get_order_status", "Get the status of an order.", {"order_id": "string"}),
    ("play_music", "Play a song or playlist.", {"query": "string"}), ("set_thermostat", "Set the home temperature.", {"temperature": "number"}),
    ("lookup_word", "Get the dictionary definition of a word.", {"word": "string"}), ("get_news", "Get recent headlines for a topic.", {"topic": "string"}),
]
TOOL_SCENARIOS = [
    "one user request that needs exactly one tool call, then Kisoku answers from the tool result",
    "one user request that needs two different tools one after the other (the second call uses the first result)",
    "the user asks an ordinary question that needs NO tool (general knowledge, simple math, or writing); Kisoku answers directly and does not call anything",
    "a short simple arithmetic or factual question that needs NO tool even though a calculator or search tool is available... Kisoku just answers",
    "the request needs a tool but a required argument is missing; Kisoku asks the user for it, the user answers, then Kisoku calls the tool",
    "the user wants something none of the tools can do; Kisoku says so plainly and offers what it can do instead",
    "a three or four turn conversation mixing one normal question (no tool) and one request that needs a tool",
    "the tool returns an error or empty result; Kisoku tells the user honestly and suggests a next step",
    "two tool calls in one assistant turn (two cities, two tickers, etc.), then one answer that uses both results",
]
SHORTQA_AREAS = ["basic arithmetic and percentages", "world capitals and geography that every atlas agrees on", "basic science facts (school level)", "well-known history dates and people",
                 "everyday unit conversions", "common word meanings and synonyms", "basic grammar and spelling questions", "simple programming facts (what a keyword or command does)",
                 "famous books, authors and inventions", "the human body (school level)", "simple logic and sequences", "days, months, time and calendar arithmetic",
                 "animals and nature facts (well established)", "space and the solar system (well established)", "basic music, sports and games rules"]
LOGIC_KINDS = ["a river-crossing or scheduling style logic puzzle with a unique answer", "a 'who owns what' deduction puzzle with 3 people", "a number sequence with a clear rule",
               "a short probability question with small numbers", "a rate, distance and time word problem", "a question about ordering by comparisons (taller, older, faster)",
               "a simple code-tracing question: what does this short Python snippet print", "a unit and percentage reasoning problem from daily life", "a find-the-bug question for a 6 to 10 line function",
               "a question with a tempting wrong answer (a classic trick question) that careful reasoning gets right", "a combinatorics counting question with small numbers", "an age word problem"]


def tools_json(rng):
    k = rng.randint(2, 5)
    return [{"type": "function", "function": {"name": n, "description": d, "parameters": {"type": "object", "properties": {a: {"type": t} for a, t in p.items()}, "required": list(p)}}}
            for n, d, p in rng.sample(TOOLBOX, k)]


def spec(cat, i):
    """Deterministic request for example i of a category: (prompt, max_tokens, extra)."""
    rng = random.Random(f"{cat}-{i}")
    turns = lambda a, b: rng.randint(a, b)
    head = FACTS + STYLE + USERS
    if cat == "persona":
        n = rng.choice([1, 1, 2, 2, 3, 4])
        return (head + f"Write one realistic conversation of {n} user turn(s) (each followed by Kisoku's reply). Scenario: {rng.choice(PERSONA_SEEDS)}. "
                f"Vary the wording; do not start every reply the same way. Kisoku does not recite the whole fact list, only what was asked. {JSON_RULE}"), 1500, None
    if cat == "build":
        ask = rng.choice(BUILD_PHRASES).format(t=rng.choice(BUILD_THINGS))
        n = rng.choice([1, 1, 2])
        return (head + f"Write one conversation of {n} user turn(s). The first user message is a casual request close to: \"{ask}\" (reword it naturally, a typo is fine). "
                "Kisoku does NOT refuse and does NOT say it cannot create things: it writes complete, working, reasonably small code and a short explanation of how to use it. "
                + ("In the second turn the user asks a quick follow-up question about the code and Kisoku answers it accurately. " if n == 2 else "") + JSON_RULE), 3200, None
    if cat == "revise":
        n = turns(2, 3)
        fu = rng.sample(REVISE_FOLLOWUPS, n - 1)
        return (head + f"Write one conversation of {n} user turns. Turn 1: the user asks Kisoku for {rng.choice(REVISE_TASKS)} (pick a concrete, ordinary request). "
                f"Then the user follows up with messages in the spirit of: {'; then '.join(fu)}. Each time, Kisoku gives the FULL revised result (not just a description), the revision "
                "must really differ in the way the user asked, and any sentence about what changed must be true of the text or code actually shown. "
                "If the user asks for 'more detail', Kisoku adds real substance to the work itself instead of explaining the old version. " + JSON_RULE), 3800, None
    if cat == "chat":
        n = turns(3, 6)
        return (head + f"Write one natural conversation of {n} user turns about: {rng.choice(CHAT_TOPICS)}. The user's messages build on Kisoku's previous replies "
                "(follow-up questions, pushback, a change of mind, a personal detail). Kisoku gives useful, specific, accurate help, keeps track of what was said earlier, "
                "asks a clarifying question when it truly needs one, and keeps most replies under 150 words. " + JSON_RULE), 3000, None
    if cat == "explain":
        n = rng.choice([1, 2, 2, 3])
        return (head + f"Write one conversation of {n} user turn(s). The user asks Kisoku to explain {rng.choice(EXPLAIN_TOPICS)} {rng.choice(EXPLAIN_LEVELS)} "
                "(the user phrases this in their own casual words). Later turns, if any, are follow-ups such as 'wait why', 'give me an example', 'simpler', or a wrong guess that "
                "Kisoku corrects kindly. Everything Kisoku says must be accurate. " + JSON_RULE), 2400, None
    if cat == "shortqa":
        return (STYLE + f"Write 8 separate short question-and-answer pairs about: {rng.choice(SHORTQA_AREAS)}. Questions are what a person would type quickly (some lowercase, some terse, "
                "like 'whats 12 times 12' or 'capital of japan?'). Answers are one short sentence, correct beyond any doubt, with no extra trivia. Only use facts that are certain. "
                'Return ONLY JSON: {"pairs": [{"q": "...", "a": "..."}, ...]}'), 900, None
    if cat == "tools":
        tl = tools_json(rng)
        return (head + "Kisoku has these tools available in this conversation:\n" + json.dumps(tl) + "\n\n"
                f"Write one conversation. Scenario: {rng.choice(TOOL_SCENARIOS)}.\n"
                'To call a tool, the assistant message content is exactly one or more blocks of the form <tool_call>\\n{"name": "...", "arguments": {...}}\\n</tool_call> and nothing else. '
                'The next message then has role "tool" and content <tool_response>\\n{...json result...}\\n</tool_response> (one block per call, invent realistic results). '
                "After the tool message the assistant answers the user in plain words using only what the tool returned. Only call tools from the list, with all required arguments. "
                'Return ONLY JSON: {"messages": [...]} with roles user, assistant and tool; it starts with user and ends with assistant.'), 2200, tl
    if cat == "think_logic":
        return ("Write one self-contained problem: " + rng.choice(LOGIC_KINDS) + ". It must have one definite correct answer that you have verified. "
                'Return ONLY JSON: {"question": "the problem as a user would type it", "answer": "the final answer in a few words"}'), 700, None
    raise KeyError(cat)


EMDASH = re.compile(r"\s*[—–]\s*")
FOREIGN = re.compile(r"\b(i am|i'm|my name is) (chatgpt|claude|gemini|bard|llama|qwen|deepseek|gpt)\b|\b(made|created|built|trained|developed) by (openai|anthropic|google|meta|alibaba|deepseek|microsoft)\b", re.I)


def foreign(c):
    """True when the assistant CLAIMS another identity. A denial ("I'm not ChatGPT", "I wasn't made by OpenAI") is fine."""
    for m in FOREIGN.finditer(c):
        if not re.search(r"\b(not|never|isn't|wasn't|aren't|n't)\b[^.!?]{0,40}$", c[max(0, m.start() - 60): m.start()] + " ", re.I) and not re.search(r"^\s*not\b", c[m.start():].split(" ", 2)[-1] if False else ""):
            if not re.search(r"\b(i am|i'm) not\b", c[max(0, m.start() - 2): m.end()], re.I):
                return True
    return False
CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)


def call(messages, max_tokens, think=False, fmt=None):
    """One teacher call -> (content, thinking, in_tokens, out_tokens); content None when it failed or was cut off."""
    if DS:
        body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0.6 if think else 0.9,
                "thinking": {"type": "enabled" if think else "disabled"}}
        if fmt: body["response_format"] = {"type": "json_object"}
    else:
        body = {"model": MODEL, "messages": messages, "stream": False, "think": think, "options": {"temperature": 0.9 if not think else 0.6, "num_predict": max_tokens}}
        if fmt: body["format"] = "json"
    for attempt in range(8):
        if OUT_OF_CREDIT.is_set(): return None, None, 0, 0
        try:
            r = requests.post(API, headers={"Authorization": f"Bearer {KEY}"}, timeout=400, json=body)
            if r.status_code == 402:
                OUT_OF_CREDIT.set(); return None, None, 0, 0
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2 + 3 * attempt + random.random() * 3); continue
            r.raise_for_status(); j = r.json()
            if DS:
                c = j["choices"][0]; m = c["message"]; u = j.get("usage", {}); a, b = u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
                if c.get("finish_reason") == "length": return None, None, a, b
                return (m.get("content") or "").strip(), (m.get("reasoning_content") or "").strip(), a, b
            m = j["message"]
            if j.get("done_reason") == "length":
                return None, None, j.get("prompt_eval_count", 0), j.get("eval_count", 0)
            return m.get("content", "").strip(), (m.get("thinking") or "").strip(), j.get("prompt_eval_count", 0), j.get("eval_count", 0)
        except (requests.RequestException, KeyError, ValueError):
            time.sleep(2 + 3 * attempt)
    return None, None, 0, 0


def clean(t):
    return EMDASH.sub(", ", t).strip()


META = re.compile(r"em dash|last line|final line|'answer:|\"answer:|answer: \.\.\.|need final|system prompt|instruction", re.I)


def clean_thinking(t):
    """Drop the teacher's notes about OUR formatting instructions from a reasoning trace (they are not part of solving the problem)."""
    keep = [x for x in re.split(r"(?<=[.!?])\s+|\n", clean(t)) if x.strip() and not META.search(x)]
    return " ".join(keep).strip()


def check_conv(msgs, tools=None):
    """Validate one generated conversation; return cleaned messages or None."""
    if not isinstance(msgs, list) or len(msgs) < 2: return None
    out, names = [], {t["function"]["name"] for t in (tools or [])}
    for m in msgs:
        if not isinstance(m, dict) or m.get("role") not in ("user", "assistant", "tool") or not isinstance(m.get("content"), str) or not m["content"].strip(): return None
        if m["role"] == "tool" and not tools: return None
        c = m["content"].strip() if m["role"] == "tool" else clean(m["content"])
        if m["role"] == "assistant":
            if foreign(c) or "<think>" in c: return None
            for blk in CALL.findall(c):
                try: d = json.loads(blk)
                except ValueError: return None
                if d.get("name") not in names or not isinstance(d.get("arguments"), dict): return None
            if "<tool_call>" in c and (not tools or not CALL.search(c)): return None
        if m["role"] == "tool" and "<tool_response>" not in c: return None
        out.append({"role": m["role"], "content": c})
    if out[0]["role"] != "user" or out[-1]["role"] != "assistant": return None
    for a, b in zip(out, out[1:]):
        ok = {("user", "assistant"), ("assistant", "user"), ("assistant", "tool"), ("tool", "assistant")}
        if (a["role"], b["role"]) not in ok: return None
        if b["role"] == "tool" and "<tool_call>" not in a["content"]: return None
    return out


def make(cat, i, math_rows):
    """Return (record or None, in_tokens, out_tokens)."""
    if cat == "think_math":
        row = math_rows[i]
        txt, th, a, b = call([{"role": "system", "content": "Solve the problem. Keep the final answer short: brief working, then a last line 'Answer: ...'. No em dashes."},
                              {"role": "user", "content": row["q"]}], 3500, think=True)
        if not txt or not th or len(th) < 40: return None, a, b
        last = [l for l in txt.splitlines() if l.strip().lower().startswith("answer:")]
        if not last or norm(last[-1].split(":", 1)[1]) != norm(row["ref"]): return None, a, b
        return {"messages": [{"role": "user", "content": row["q"]}, {"role": "assistant", "content": clean(txt)}], "thinking": clean_thinking(th)}, a, b
    prompt, mx, extra = spec(cat, i)
    txt, _, a, b = call([{"role": "user", "content": prompt}], mx, fmt="json")
    if not txt: return None, a, b
    try: d = json.loads(txt[txt.index("{"): txt.rindex("}") + 1])
    except ValueError: return None, a, b
    if cat == "shortqa":
        pairs = [{"q": clean(p["q"]), "a": clean(p["a"])} for p in d.get("pairs", []) if isinstance(p, dict) and isinstance(p.get("q"), str) and isinstance(p.get("a"), str) and 3 < len(p["q"]) < 200 and 0 < len(p["a"]) < 300]
        return ({"pairs": pairs} if len(pairs) >= 4 else None), a, b
    if cat == "think_logic":
        q, ref = d.get("question"), d.get("answer")
        if not isinstance(q, str) or not isinstance(ref, str) or len(q) < 30: return None, a, b
        txt2, th, a2, b2 = call([{"role": "system", "content": "Solve the problem carefully. Give a short clear solution and end with a last line 'Answer: ...'. No em dashes."},
                                 {"role": "user", "content": q}], 3500, think=True)
        a += a2; b += b2
        if not txt2 or not th or len(th) < 40: return None, a, b
        # keep only when the independent solve agrees with the writer's answer (cheap consistency check by a third call)
        v, _, a3, b3 = call([{"role": "user", "content": f"Problem: {q}\nAnswer A: {ref}\nAnswer B: {txt2[-400:]}\nDo A and B give the same final answer? Reply with one word: yes or no."}], 20)
        a += a3; b += b3
        if not v or not v.strip().lower().startswith("yes"): return None, a, b
        return {"messages": [{"role": "user", "content": clean(q)}, {"role": "assistant", "content": clean(txt2)}], "thinking": clean_thinking(th)}, a, b
    msgs = check_conv(d.get("messages"), extra)
    if not msgs: return None, a, b
    rec = {"messages": msgs}
    if extra: rec["tools"] = extra
    return rec, a, b


def norm(a):
    a = a.strip().rstrip(".").replace("$", "").replace("\\boxed", "").replace("\\text", "").replace("{", "").replace("}", "")
    a = a.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac").replace(" ", "").replace(",", "").lower()
    try: return f"{float(a):.6g}"
    except ValueError: return a


def math_rows(n):
    """School-level NuminaMath problems with a boxed reference answer, NOT the ones the first generator used (different shuffle seed)."""
    cache = OUT / "_think_math_prompts.jsonl"
    if cache.exists(): return [json.loads(l) for l in cache.open()]
    from datasets import load_dataset
    sys.path.insert(0, str(Path(__file__).parent)); from gen_sft import boxed
    ds = load_dataset("AI-MO/NuminaMath-CoT", split="train").filter(lambda r: r["source"] in ("gsm8k", "orca_math", "synthetic_math", "math", "cn_k12"))
    rows = []
    for r in ds.shuffle(seed=777).select(range(min(len(ds), n * 3))):
        ref = boxed(r["solution"])
        if ref and 30 < len(r["problem"]) < 1200: rows.append({"q": r["problem"], "ref": ref})
        if len(rows) >= int(n * 1.6): break
    with cache.open("w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    return rows


def run(stage, workers, budget):
    done = {}
    for f in OUT.glob("[a-z]*.jsonl"):
        done[f.stem] = set()
        for l in f.open():
            try: done[f.stem].add(json.loads(l)["id"])
            except ValueError: pass  # a line cut off by a kill mid-write
    targets = {c: (a if stage == 1 else a + b) for c, (a, b) in PLAN.items()}
    mrows = math_rows(PLAN["think_math"][1]) if targets["think_math"] else []
    jobs = []
    for c, n in targets.items():
        # attempts are over-provisioned by 30% because some generations fail validation
        ids = range(int(n * 1.3)) if c != "think_math" else range(min(len(mrows), int(n * 1.6)))
        have = len(done.get(c, ()))
        jobs += [(c, i) for i in ids if f"{c}-{i:06d}" not in done.get(c, ()) and f"{c}-{i:06d}" not in SKIP.get(c, ())][: max(0, int((n - have) * 1.35))]
    random.Random(stage).shuffle(jobs)
    sp = OUT / "_spend.json"; S = json.loads(sp.read_text()) if sp.exists() else {"in": 0, "out": 0, "ok": 0, "fail": 0}
    usd = lambda: S["in"] / 1e6 * RATE[0] + S["out"] / 1e6 * RATE[1]
    print(f"stage {stage}: {sum(len(v) for v in done.values())} done, {len(jobs)} attempts queued, spent ${usd():.2f}", flush=True)
    lock = threading.Lock(); stop = threading.Event(); count = {c: len(done.get(c, ())) for c in targets}

    def work(job):
        c, i = job
        if stop.is_set() or count[c] >= targets[c]: return
        rec, a, b = make(c, i, mrows)
        with lock:
            S["in"] += a; S["out"] += b; S["ok" if rec else "fail"] += 1
            if rec and count[c] < targets[c]:
                rec.update(id=f"{c}-{i:06d}", source=c, teacher=TEACHER, provider="deepseek-api" if DS else "ollama-cloud")
                with (OUT / f"{c}.jsonl").open("a") as f: f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                count[c] += 1
            elif not rec and not OUT_OF_CREDIT.is_set():
                with (OUT / "_failed.txt").open("a") as f: f.write(f"{c}-{i:06d}\n")
            n = S["ok"] + S["fail"]
            if n % 50 == 0:
                sp.write_text(json.dumps({**S, "usd": round(usd(), 2)}))
                if n % 200 == 0: print(time.strftime("%H:%M:%S"), dict(count), f"ok {S['ok']} fail {S['fail']} ${usd():.2f}", flush=True)
            if usd() > budget or OUT_OF_CREDIT.is_set(): stop.set()

    with ThreadPoolExecutor(workers) as ex: list(ex.map(work, jobs))
    sp.write_text(json.dumps({**S, "usd": round(usd(), 2)}))
    print("finished", dict(count), f"${usd():.2f}", ("OUT OF CREDIT (402)" if OUT_OF_CREDIT.is_set() else "BUDGET STOP") if stop.is_set() else "", flush=True)


SKIP = {}
if (OUT / "_failed.txt").exists():
    for l in (OUT / "_failed.txt").open():
        c = l.strip().rsplit("-", 1)[0]; SKIP.setdefault(c, set()).add(l.strip())


def status():
    c = {f.stem: sum(1 for _ in f.open()) for f in sorted(OUT.glob("[a-z]*.jsonl"))}
    print(json.dumps({"written": c, "total": sum(c.values()), "spend": json.loads((OUT / "_spend.json").read_text()) if (OUT / "_spend.json").exists() else None}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--stage", type=int, default=2)
    ap.add_argument("--workers", type=int, default=96 if DS else 12); ap.add_argument("--budget", type=float, default=150.0)
    a = ap.parse_args()
    run(a.stage, a.workers, a.budget) if a.cmd == "run" else status()

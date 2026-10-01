"""Build verbatim-overlap probes from the eval sets we report.
Each item gives up to 2 probes: first 80 and last 80 chars of the normalized text
(one probe if the text is 50-159 chars; items under 50 chars are skipped as too generic)."""
import json, os, sys, collections
from datasets import load_dataset
from common import norm
MINLEN, W = 50, 80
out, stats = [], collections.OrderedDict()
def add(bench, iid, text):
    n = norm(text)
    s = stats.setdefault(bench, {'items': 0, 'probed': 0})
    s['items'] += 1
    if len(n) < MINLEN: return
    s['probed'] += 1
    pats = [n] if len(n) < 2 * W else [n[:W], n[-W:]]
    for k, p in enumerate(pats):
        out.append({'bench': bench, 'id': f'{iid}', 'k': k, 'pat': p.strip()})
def safe(name, fn):
    try: fn()
    except Exception as e: print('FAILED', name, repr(e)[:300], flush=True)
def gsm():
    for i, r in enumerate(load_dataset('openai/gsm8k', 'main', split='test')): add('gsm8k', i, r['question'])
def mmlu():
    for i, r in enumerate(load_dataset('cais/mmlu', 'all', split='test')): add('mmlu', i, r['question'])
def arc():
    for cfg, b in (('ARC-Easy', 'arc_easy'), ('ARC-Challenge', 'arc_challenge')):
        for r in load_dataset('allenai/ai2_arc', cfg, split='test'): add(b, r['id'], r['question'])
def hs():
    for i, r in enumerate(load_dataset('Rowan/hellaswag', split='validation')):
        add('hellaswag', i, r['ctx'] + ' ' + r['endings'][int(r['label'])])
def piqa():
    for i, r in enumerate(load_dataset('baber/piqa', split='validation')):
        add('piqa', i, r['goal'] + ' ' + (r['sol1'], r['sol2'])[int(r['label'])])
def wino():
    for i, r in enumerate(load_dataset('allenai/winogrande', 'winogrande_xl', split='validation')): add('winogrande', i, r['sentence'])
def he():
    for r in load_dataset('openai/openai_humaneval', split='test'):
        add('humaneval_prompt', r['task_id'], r['prompt'])
        add('humaneval_solution', r['task_id'], r['canonical_solution'])
def tqa():
    for r in load_dataset('mandarjoshi/trivia_qa', 'rc.nocontext', split='validation'): add('triviaqa', r['question_id'], r['question'])
for name, fn in (('gsm8k', gsm), ('mmlu', mmlu), ('arc', arc), ('hellaswag', hs), ('piqa', piqa), ('winogrande', wino), ('humaneval', he), ('triviaqa', tqa)):
    safe(name, fn)
json.dump({'patterns': out, 'stats': stats, 'minlen': MINLEN, 'window': W}, open('patterns.json', 'w'))
print(json.dumps(stats, indent=1)); print('patterns', len(out), flush=True)
os._exit(0)

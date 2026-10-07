"""Phase C synthetic long-context tasks (Kisoku v2).
Takes our own long documents (science PDFs, PG19 books), optionally plants notes / definition chains in them, and appends
question-answer pairs whose answers are computed exactly from the final text. Pretraining-style records (one `text` feature).
Templates are our own (not RULER's). Every document stays under MAX_TOK tokens so MaxText's 64K chunker never separates
the questions from the document."""
import os, sys, re, json, random, glob, subprocess, collections, time
sys.path.insert(0, os.path.expanduser('~/maxtext-tools/src'))
from maxtext.input_pipeline.protos import example_pb2, feature_pb2
from array_record.python.array_record_module import ArrayRecordReader, ArrayRecordWriter

BUCKET = 'gs://' + os.environ['KISOKU_BUCKET'] + '/datasets'
WORK = os.path.expanduser('~/synth'); SRC = f'{WORK}/src'; OUT = f'{WORK}/out'
MAX_TOK = 60000
LENGTHS = [(8000, 0.15), (16000, 0.20), (32000, 0.30), (56000, 0.35)]
STOP = set('''that this with from have were been their which would there could other these those about after before where when what while into
than then them they your some such only also more most over under between through because during each both being within without against
using used based shown figure table data results study studies however therefore thus given since among upon very much many said will shall
must might should like just even still well made make does done here have having were'''.split())
ADJ = 'amber brisk cobalt dusty eager faded gilded hollow ivory jagged keen lunar mellow narrow opal pale quiet rustic silver tidal umber velvet woven young zesty'.split()
NOUN = 'anchor beacon canyon dynamo ember falcon glacier harbor island jungle kettle lantern meadow nebula orchard pylon quarry reactor summit tunnel utensil valley windmill yacht zeppelin archive bridge compass depot engine'.split()
ATTR = ['registration number', 'access code', 'batch number', 'serial number', 'locker number', 'catalog number', 'dispatch code', 'permit number']
NAMES = 'alder birch cedar dahlia elm fennel ginkgo hazel iris juniper kelp laurel maple nettle olive poplar quince rowan sorrel thistle umbra violet willow yarrow zinnia aspen briar clover'.split()
NOTE_T = ['(Margin note: the {attr} of the {key} is {val}.)', '[Reference: {key}, {attr} {val}.]', 'For the record, the {attr} assigned to the {key} is {val}.',
          'Editor\'s note: {val} is the {attr} of the {key}.', '<<{key}: {attr} = {val}>>', 'Keep in mind that the {key} carries the {attr} {val}.']
DEF_T = ['Let {a} be {v}.', 'Define {a} as {v}.', 'Set {a} to {v}.', '{a} is fixed at {v}.']
REF_T = ['Let {a} be equal to {b}.', 'Define {a} as the same value as {b}.', 'Set {a} to whatever {b} is.', '{a} takes the value of {b}.']
HEADERS = ['', '', 'Read the document below. Questions about it follow at the end.\n\n', 'The following text is followed by several questions and their answers.\n\n',
           'Document:\n\n', 'Study the text carefully, then answer the questions that come after it.\n\n']
WORD = re.compile(r'[a-z]{4,}')
SENT = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')

def read_records(path):
    r = ArrayRecordReader(path); n = r.num_records(); ex = example_pb2.Example()
    for s in range(0, n, 256):
        for rec in r.read(s, min(n, s + 256)):
            ex.ParseFromString(rec); yield ex.features.feature['text'].bytes_list.value[0].decode('utf-8', 'ignore')
    r.close()
def serialize(text):
    feat = feature_pb2.Feature(bytes_list=feature_pb2.BytesList(value=[text.encode('utf-8')]))
    return example_pb2.Example(features=feature_pb2.Features(feature={'text': feat})).SerializeToString()
def pick_len(rng):
    x = rng.random(); c = 0
    for l, w in LENGTHS:
        c += w
        if x <= c: return l
    return LENGTHS[-1][0]
def paragraphs(text):
    ps = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    if len(ps) < 20: ps = [p.strip() for p in text.split('\n') if p.strip()]
    return ps

def build(text, tok, rng, kind):
    """Returns (rendered_text, meta) or None."""
    ps = paragraphs(text)
    if len(ps) < 12: return None
    target = pick_len(rng)
    lens = [len(x) for x in tok(ps, add_special_tokens=False)['input_ids']]
    start = rng.randrange(0, max(1, len(ps) // 2)) if kind == 'pg19' else 0
    body, tot = [], 0
    for p, l in zip(ps[start:], lens[start:]):
        if l > 3000: continue                      # skip giant unbroken blocks (tables, OCR noise)
        if tot + l + 2 > target - 1800: break      # leave room for planted lines and the Q/A block
        body.append(p); tot += l + 2
    if tot < 3000 or len(body) < 10: return None
    qa = []; used_tasks = collections.Counter()
    def slot(): return rng.randrange(1, len(body))
    plan = []  # (position, line)
    # --- planted notes: single, many keys, repeated keys ---
    keys = rng.sample([f'{a} {n}' for a in ADJ for n in NOUN], k=rng.choice([0, 3, 8, 16, 30]))
    notes = collections.OrderedDict()
    for k in keys:
        attr = rng.choice(ATTR); nvals = rng.choices([1, 2, 3, 4], [0.7, 0.15, 0.1, 0.05])[0]
        for _ in range(nvals):
            val = str(rng.randrange(10000, 9999999)) if rng.random() < 0.7 else rng.choice(NAMES) + '-' + str(rng.randrange(10, 999))
            notes.setdefault((k, attr), []).append(val)
            plan.append((slot(), rng.choice(NOTE_T).format(attr=attr, key=k, val=val)))
    # --- definition chains ---
    chains = []
    for _ in range(rng.choice([0, 0, 1, 2, 3])):
        v = str(rng.randrange(10000, 99999)); hops = rng.randrange(2, 7)
        names = [rng.choice(NAMES).upper() + str(rng.randrange(10, 99)) for _ in range(hops)]
        if len(set(names)) < hops: continue
        if len(body) <= hops + 1: continue
        pos = sorted(rng.sample(range(1, len(body)), hops))   # distinct slots so the hops stay in order
        plan.append((pos[0], rng.choice(DEF_T).format(a=names[0], v=v)))
        for i in range(1, hops): plan.append((pos[i], rng.choice(REF_T).format(a=names[i], b=names[i - 1])))
        chains.append((v, names))
    # stable insert: lines planted at the same slot keep their creation order, chains keep hop order because pos is sorted
    for pos, line in sorted(plan, key=lambda x: x[0], reverse=True): body.insert(pos, line)
    # re-inserting shifts indices, so verify chain order on the final text instead of trusting positions
    doc = '\n\n'.join(body); low = doc.lower()
    # --- Q/A from planted notes ---
    items = list(notes.items()); rng.shuffle(items)
    for (k, attr), vals in items[:rng.randrange(2, 9)]:
        if len(vals) == 1:
            qa.append((f'What is the {attr} of the {k}?', f'The {attr} of the {k} is', vals[0])); used_tasks['note_single'] += 1
        else:
            qa.append((f'List every {attr} given for the {k}.', f'All values of the {attr} given for the {k} are', ', '.join(vals))); used_tasks['note_multi'] += 1
    if len(items) >= 4 and rng.random() < 0.6:
        pick = rng.sample([it for it in items if len(it[1]) == 1], k=min(3, len([it for it in items if len(it[1]) == 1])))
        if len(pick) >= 2:
            qa.append(('Give the ' + '; the '.join(f'{a} of the {k}' for (k, a), _ in pick) + ', in that order.', 'In order, the requested values are', ', '.join(v[0] for _, v in pick))); used_tasks['note_multiquery'] += 1
    for v, names in chains:
        idx = [doc.find(n) for n in names]
        if min(idx) < 0 or idx != sorted(idx): continue
        qa.append((f'Which names have the value {v}? List all of them.', f'The names whose value is {v} are', ', '.join(names))); used_tasks['chain'] += 1
        j = rng.randrange(len(names))
        qa.append((f'What is the value of {names[j]}?', f'The value of {names[j]} is', v)); used_tasks['chain_value'] += 1
    # --- word counting on the final document text ---
    cnt = collections.Counter(w for w in WORD.findall(low) if w not in STOP)
    cands = [w for w, c in cnt.items() if 3 <= c <= 60]
    rng.shuffle(cands)
    hi = [w for w in cands if cnt[w] >= 8][:rng.randrange(1, 4)]   # do not let every count be 3 or 4
    for w in hi + [w for w in cands if w not in hi][:rng.randrange(2, 5)]:
        qa.append((f'How many times does the word "{w}" appear in the text above?', f'The number of times the word "{w}" appears in the text above is', str(cnt[w]))); used_tasks['count'] += 1
    for _ in range(rng.randrange(1, 4)):
        if len(cands) < 8: break
        opts = rng.sample(cands, 4); cs = sorted((cnt[o] for o in opts), reverse=True)
        if cs[0] - cs[1] < 2: continue
        best = max(opts, key=lambda o: cnt[o])
        qa.append((f'Which of these words appears most often in the text: {", ".join(opts)}?', f'Among {", ".join(opts)}, the word that appears most often is', best)); used_tasks['most_frequent'] += 1
    top = cnt.most_common(6)
    if len(top) == 6 and len({c for _, c in top}) == 6:   # strict order, and no tie between 5th and 6th
        qa.append(('What are the five most frequent words of four or more letters in the text, ignoring common function words? List them from most to least frequent.',
                   'From most to least frequent, the five most common such words are', ', '.join(w for w, _ in top[:5]))); used_tasks['top5'] += 1
    # --- sentence that follows / which comes first ---
    sents = []
    for p in body:
        ss = SENT.split(p)
        for a, b in zip(ss, ss[1:]):
            if 60 <= len(a) <= 200 and 40 <= len(b) <= 250 and doc.count(a) == 1: sents.append((a, b))
    rng.shuffle(sents)
    for a, b in sents[:rng.randrange(1, 4)]:
        qa.append((f'What sentence comes directly after this one? "{a}"', f'The sentence that comes directly after "{a}" is', b)); used_tasks['next_sentence'] += 1
    once = [w for w, c in cnt.items() if c == 1 and len(w) >= 7]
    for _ in range(rng.randrange(0, 3)):
        if len(once) < 6: break
        opts = rng.sample(once, 3); first = min(opts, key=lambda o: low.find(o))
        qa.append((f'Which of these words appears first in the text: {", ".join(opts)}?', f'Of {", ".join(opts)}, the one that appears first in the text is', first)); used_tasks['first'] += 1
    if len(qa) < 4: return None
    rng.shuffle(qa); qa = qa[:rng.randrange(8, 25)]
    style = rng.choice(['QA', 'QA', 'qa', 'lead', 'mixed'])
    blocks = []
    for q, lead, a in qa:
        s = rng.choice(['QA', 'qa', 'lead']) if style == 'mixed' else style
        blocks.append(f'Question: {q}\nAnswer: {a}' if s == 'QA' else f'Q: {q}\nA: {a}' if s == 'qa' else f'{lead}: {a}')
    out = rng.choice(HEADERS) + doc + '\n\n' + '\n\n'.join(blocks)
    ntok = len(tok(out, add_special_tokens=False)['input_ids'])
    if ntok > MAX_TOK: return None
    return out, {'tokens': ntok, 'target': target, 'nqa': len(qa), 'tasks': dict(used_tasks), 'kind': kind, 'qa_text': '\n\n'.join(blocks)}

def run(job):
    kind, obj, take, seed = job
    from transformers import AutoTokenizer
    tokp = os.path.expanduser('~/hf_token'); token = open(tokp).read().strip() if os.path.exists(tokp) else None
    tok = AutoTokenizer.from_pretrained('unsloth/Llama-3.2-1B', token=token)
    rng = random.Random(seed); name = obj.split('/')[-1]; local = f'{SRC}/{name}'
    if not os.path.exists(local): subprocess.run(['gcloud', 'storage', 'cp', '--quiet', obj, local], capture_output=True)
    outp = f'{OUT}/part-{name}'
    w = ArrayRecordWriter(outp + '.tmp', 'group_size:1'); stats = []; samples = []; n = 0
    for text in read_records(local):
        if n >= take: break
        if rng.random() > 0.35: continue          # spread the picks across the shard
        try: res = build(text, tok, rng, kind)
        except Exception as e: res = None; stats.append({'error': repr(e)[:200]})
        if not res: continue
        out, meta = res; w.write(serialize(out)); n += 1
        if len(samples) < 3: samples.append({'tokens': meta['tokens'], 'head': out[:400], 'qa': meta['qa_text'][:2500]})
        meta.pop('qa_text'); stats.append(meta)
    w.close(); os.rename(outp + '.tmp', outp); os.remove(local)
    json.dump({'stats': stats, 'samples': samples}, open(outp + '.json', 'w'))
    return name, n, sum(s.get('tokens', 0) for s in stats)

if __name__ == '__main__':
    from multiprocessing import Pool
    smoke = len(sys.argv) > 1 and sys.argv[1] == 'smoke'
    os.makedirs(SRC, exist_ok=True); os.makedirs(OUT, exist_ok=True)
    jobs = []
    for kind, ds, take in (('science', 'longctx-science-pdfs', 800), ('pg19', 'longctx-pg19', 600)):
        ls = sorted(x for x in subprocess.run(['gcloud', 'storage', 'ls', f'{BUCKET}/{ds}/'], capture_output=True, text=True).stdout.split() if x.endswith('.arrayrecord'))
        for i, o in enumerate(ls): jobs.append((kind, o, 12 if smoke else take, 1000 + len(jobs)))
    if smoke: jobs = [jobs[0], jobs[-1]]
    t0 = time.time()
    with Pool(8) as pool:
        for name, n, toks in pool.imap_unordered(run, jobs): print(time.strftime('%H:%M:%S'), name, 'docs', n, 'tokens', toks, flush=True)
    print('GEN DONE', round(time.time() - t0), 's', flush=True)

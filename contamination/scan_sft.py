"""Check the generated SFT examples against the same verbatim benchmark probes used for the pretraining audit.
Usage: python scan_sft.py patterns.json out_dir   (out_dir holds <source>.jsonl files of {"messages": [...]})"""
import sys, json, glob, os, collections
import ahocorasick
from common import norm
P = json.load(open(sys.argv[1]))['patterns']
A = ahocorasick.Automaton(); seen = {}
for i, p in enumerate(P): seen.setdefault(p['pat'], []).append(i)
for pat, idxs in seen.items(): A.add_word(pat, idxs)
A.make_automaton()
hit_items = collections.defaultdict(set); by_src = collections.defaultdict(lambda: collections.Counter()); ex = []
tot = collections.Counter(); bad = collections.Counter()
for f in sorted(glob.glob(os.path.join(sys.argv[2], '*.jsonl'))):
    src = os.path.basename(f)[:-6]
    for n, line in enumerate(open(f)):
        try: msgs = json.loads(line)['messages']
        except Exception:
            bad[src] += 1; continue   # a line cut off by a crash or reboot mid-write
        tot[src] += 1
        text = norm('\n'.join(m['content'] for m in msgs))
        for end, idxs in A.iter(text):
            for i in idxs:
                p = P[i]
                if p['bench'] == 'humaneval_solution': continue   # generic code, not meaningful (see the pretraining audit)
                hit_items[p['bench']].add(p['id']); by_src[p['bench']][src] += 1
                if len(ex) < 12: ex.append((p['bench'], src, n, text[max(0, end - 120):end + 40]))
print('examples scanned:', dict(tot), 'total', sum(tot.values()), '| unreadable lines:', dict(bad))
st = json.load(open(sys.argv[1]))['stats']
for b in st:
    if b == 'humaneval_solution': continue
    print('%-18s probed %6d  items found in SFT data %5d (%.2f%%)  %s' % (b, st[b]['probed'], len(hit_items[b]), 100 * len(hit_items[b]) / max(1, st[b]['probed']), dict(by_src[b])))
for e in ex: print('EX', e[0], e[1], e[2], '|', e[3])

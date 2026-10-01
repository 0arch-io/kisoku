import json, glob, os, collections
P = json.load(open('patterns.json')); pats = P['patterns']
per_ds = collections.defaultdict(lambda: {'shards': 0, 'records': 0, 'text_gb': 0.0})
item_hits = collections.defaultdict(lambda: collections.defaultdict(set))  # bench -> item -> set(k)
item_ds = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))  # bench -> item -> ds counts
examples = collections.defaultdict(list)
for f in sorted(glob.glob(os.path.expanduser('~/contam/out/*/*.json'))):
    d = json.load(open(f)); s = per_ds[d['ds']]
    s['shards'] += 1; s['records'] += d['records']; s['text_gb'] += d['text_bytes'] / 1e9
    for h in d['hits']:
        for i in h['p']:
            p = pats[i]; item_hits[p['bench']][p['id']].add(p['k']); item_ds[p['bench']][p['id']][d['ds']] += 1
            if h['ctx'] and len(examples[p['bench']]) < 8: examples[p['bench']].append({'id': p['id'], 'ds': d['ds'], 'shard': d['shard'], 'rec': h['rec'], 'ctx': h['ctx']})
npat = collections.defaultdict(lambda: collections.Counter())
for p in pats: npat[p['bench']][p['id']] += 1
res = {'corpus': per_ds, 'bench': {}, 'examples': examples, 'method': {'minlen': P['minlen'], 'window': P['window']}}
for b, st in P['stats'].items():
    any_hit = len(item_hits[b]); full = sum(1 for i, ks in item_hits[b].items() if len(ks) == npat[b][i])
    by_ds = collections.Counter()
    for i, c in item_ds[b].items():
        for ds in c: by_ds[ds] += 1
    res['bench'][b] = {'items': st['items'], 'probed': st['probed'], 'any_probe_hit': any_hit, 'all_probes_hit': full,
                       'pct_any': round(100 * any_hit / max(1, st['probed']), 2), 'pct_all': round(100 * full / max(1, st['probed']), 2),
                       'items_hit_by_dataset': dict(by_ds), 'hit_ids': sorted(item_hits[b])[:5000]}
json.dump(res, open('contamination-summary.json', 'w'), indent=1, default=dict)
print('corpus:'); [print(f"  {k:28s} shards {v['shards']:4d} records {v['records']:>12,} text {v['text_gb']:8.1f} GB") for k, v in per_ds.items()]
print(f"{'bench':20s} {'items':>6s} {'probed':>6s} {'any':>6s} {'%any':>6s} {'all':>6s} {'%all':>6s}  by dataset")
for b, r in res['bench'].items(): print(f"{b:20s} {r['items']:6d} {r['probed']:6d} {r['any_probe_hit']:6d} {r['pct_any']:6.2f} {r['all_probes_hit']:6d} {r['pct_all']:6.2f}  {r['items_hit_by_dataset']}")

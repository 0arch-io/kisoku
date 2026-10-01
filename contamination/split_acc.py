import json,glob,os,sys,collections
sys.path.insert(0,'/mnt/x/WSL/contam'); from common import norm
H=json.load(open('/mnt/x/WSL/contam/hits.json')); R=os.path.expanduser('~/kisoku-eval/results')
mp=set(H['mmlu_probed'])
hit={b:set(map(str,H[b])) for b in H}
def key(b,s):
    if b=='mmlu':
        n=norm(s['doc']['question']); return n if len(n)<160 else n[:80]
    if b.startswith('arc'): return str(s['doc']['id'])
    return str(s['doc_id'])
models=['kisoku-s3','kisoku-s1','llama-3.2-1b','gemma-3-1b','smollm2-1.7b','qwen2.5-1.5b']
sets={'gsm8k':('gsm8k','gsm8k','exact_match'),'mmlu':('mmlu','mmlu_*','acc'),'arc_easy':('core','arc_easy','acc'),'arc_challenge':('core','arc_challenge','acc'),'hellaswag':('core','hellaswag','acc'),'piqa':('core','piqa','acc')}
print('%-14s %-13s %6s %7s %6s %7s %7s'%('bench','model','n_hit','acc_hit','n_cln','acc_cln','gap'))
for b,(d,pat,met) in sets.items():
    for m in models:
        a=collections.defaultdict(list)
        for f in glob.glob(f'{R}/{m}-{d}/*/samples_{pat}_*.jsonl'):
            for line in open(f):
                s=json.loads(line); k=key(b,s)
                if b=='mmlu' and k not in mp: continue   # unprobed (short) questions left out of both groups
                a[k in hit[b]].append(float(s[met]))
        if not a: print('%-14s %-13s no samples'%(b,m)); continue
        h,c=a[True],a[False]; ah=100*sum(h)/max(1,len(h)); ac=100*sum(c)/max(1,len(c))
        print('%-14s %-13s %6d %7.1f %6d %7.1f %+7.1f'%(b,m,len(h),ah,len(c),ac,ah-ac))

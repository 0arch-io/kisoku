"""Scan every pretraining shard for verbatim benchmark probes. Resumable: one JSON per shard."""
import os, sys, json, subprocess, time, collections
from multiprocessing import Pool
sys.path.insert(0, os.path.expanduser('~/maxtext-tools/src'))
from common import norm
BUCKET = 'gs://' + os.environ['KISOKU_BUCKET'] + '/datasets'
DATASETS = ['finemath-4plus', 'open-web-math', 'megamath-web-pro', 'nemotron-cc-math-4plus', 'openthoughts3-text',
            'longctx-code-repos', 'longctx-pg19', 'longctx-science-pdfs',
            'starcoderdata-text', 'nemotron-cc-v2.1-hqs', 'ultra-fineweb-en-text']
OUT = os.path.expanduser('~/contam/out'); TMP = os.path.expanduser('~/contam/tmp')
A = None; PATS = None
def init():
    global A, PATS
    import ahocorasick
    PATS = json.load(open('patterns.json'))['patterns']
    A = ahocorasick.Automaton()
    seen = {}
    for i, p in enumerate(PATS): seen.setdefault(p['pat'], []).append(i)
    for pat, idxs in seen.items(): A.add_word(pat, idxs)
    A.make_automaton()
def scan_shard(obj):
    from maxtext.input_pipeline.protos import example_pb2
    from array_record.python.array_record_module import ArrayRecordReader
    ds, name = obj.split('/')[-2], obj.split('/')[-1]
    outp = f'{OUT}/{ds}/{name}.json'
    if os.path.exists(outp): return ds, name, 'skip'
    os.makedirs(os.path.dirname(outp), exist_ok=True); os.makedirs(f'{TMP}/{ds}', exist_ok=True)
    local = f'{TMP}/{ds}/{name}'; t0 = time.time()
    for attempt in range(3):
        if subprocess.run(['gcloud', 'storage', 'cp', '--quiet', obj, local], capture_output=True).returncode == 0: break
        time.sleep(10)
    else: return ds, name, 'DOWNLOAD_FAILED'
    try:
        r = ArrayRecordReader(local); n = r.num_records()
        hits = []; nbytes = 0; ex = example_pb2.Example(); B = 512
        for s in range(0, n, B):
            for j, rec in enumerate(r.read(s, min(n, s + B))):
                ex.ParseFromString(rec)
                t = ex.features.feature['text'].bytes_list.value[0]
                nbytes += len(t)
                nt = norm(t)
                for end, idxs in A.iter(nt):
                    exc = nt[max(0, end - 200):end + 120] if len(hits) < 400 else ''
                    hits.append({'p': idxs, 'rec': s + j, 'ctx': exc})
        r.close()
        json.dump({'ds': ds, 'shard': name, 'records': n, 'text_bytes': nbytes, 'secs': round(time.time() - t0, 1), 'hits': hits}, open(outp + '.tmp', 'w'))
        os.rename(outp + '.tmp', outp)
        return ds, name, f'ok rec={n} MB={nbytes>>20} hits={len(hits)} {time.time()-t0:.0f}s'
    except Exception as e:
        return ds, name, 'ERROR ' + repr(e)[:300]
    finally:
        try: os.remove(local)
        except OSError: pass
if __name__ == '__main__':
    only = sys.argv[1:]  # optional: explicit object list (smoke test)
    objs = only
    if not objs:
        for d in DATASETS:
            ls = subprocess.run(['gcloud', 'storage', 'ls', f'{BUCKET}/{d}/'], capture_output=True, text=True).stdout.split()
            objs += sorted(x for x in ls if x.endswith('.arrayrecord'))
    print('shards', len(objs), flush=True)
    init()
    # self-test: a benchmark probe buried in junk must be found
    probe = PATS[0]['pat']; junk = ('Lorem IPSUM, dolor!! ' * 5 + probe.upper().replace(' ', ' ,\n ') + ' tail').encode()
    assert any(0 in idxs for _, idxs in A.iter(norm(junk))), 'self-test failed'
    print('self-test ok', flush=True)
    with Pool(int(os.environ.get('PROCS', '8')), initializer=init) as pool:
        for k, (ds, name, msg) in enumerate(pool.imap_unordered(scan_shard, objs)):
            print(time.strftime('%H:%M:%S'), k + 1, len(objs), ds, name, msg, flush=True)
    print('SCAN DONE', flush=True)

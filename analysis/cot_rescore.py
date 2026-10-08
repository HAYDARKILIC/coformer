"""Post hoc sensitivity analysis of the chain-of-thought baseline (follow-up study, family D).
Usage: python cot_rescore.py <path-to-CoFormer-folder> <path-to-notebook.ipynb>
Rebuilds the held-out split of each seed with the notebook's own functions, looks up the cached
greedy generations (checkpoints/llm_generations.json, keyed by model, token limit and prompt),
re-parses them, and reports accuracy (i) as in the study (unparsable -> Unknown), (ii) strict
(unparsable -> wrong) and (iii) on parsable generations only. Writes cot_rescore.json next to
this script. Step (i) must reproduce the reported accuracies exactly.
"""
import json,re,hashlib,numpy as np,ast
import sys,os
ROOT=sys.argv[1]; NB=sys.argv[2]
os.chdir(ROOT)
nb=json.load(open(NB))
src='\n'.join(''.join(c['source']) for c in nb['cells'])
def getfunc(name):
    i=src.index('def '+name+'(')
    # take until next line starting with non-space (top-level)
    lines=src[i:].split('\n'); out=[lines[0]]
    for l in lines[1:]:
        if l and not l[0].isspace(): break
        out.append(l)
    return '\n'.join(out)
g={}
exec('import re, numpy as np\n'+getfunc('resolve_option_answer')+'\n'+getfunc('normalize_label'),g)
i=src.index('_ANSWER_RE = '); exec('import re as _re_d\n'+src[i:src.index('\n',i)],g)
MODEL=re.search(r'"(Qwen/Qwen2\.5-7B-Instruct)"',src).group(1)
cache=json.load(open('checkpoints/llm_generations.json'))
misses=[0]
def llm_generate(prompt,max_new_tokens=512,temperature=0.0):
    k=hashlib.md5(f"{MODEL}|{max_new_tokens}|{prompt}".encode()).hexdigest()
    if k not in cache: misses[0]+=1; return None
    return cache[k]
g['llm_generate']=llm_generate
exec(getfunc('baseline_cot')+'\n'+getfunc('baseline_direct'),g)
real=json.load(open('data_cache/real_benchmarks_v4.json'))
def pools(seed):
    pool=[]
    for raw in real['proofwriter']:
        e=dict(raw); e['dataset']='ProofWriter'; pool.append((e,g['normalize_label'](e.get('label'),e.get('options'))))
    for raw in real['folio']:
        if raw.get('label') is None: continue
        fol=raw.get('premises-FOL',raw.get('fol',''))
        if isinstance(fol,str): fol=[fol] if fol else []
        if any(isinstance(f,str) and f.count('(')!=f.count(')') for f in fol): continue
        e=dict(raw); e['dataset']='FOLIO'; pool.append((e,g['normalize_label'](e.get('label'))))
    rng=np.random.default_rng(seed); rng.shuffle(pool)
    n=len(pool); nh=min(max(40,int(0.2*n)),max(1,int(0.75*n)))
    return pool[:nh]
res={}
for seed in range(4):
    H=pools(seed); rows=[]
    for e,lab in H:
        r=g['baseline_cot'](e); d=g['baseline_direct'](e)
        rows.append((lab,r['pred'],r['used_fallback'],d['pred'],e['dataset']))
    n=len(rows)
    acc=np.mean([p==l for l,p,f,_,_ in rows]); fb=np.mean([f for _,_,f,_,_ in rows])
    strict=np.mean([(p==l) and not f for l,p,f,_,_ in rows])
    parsed=[(l,p) for l,p,f,_,_ in rows if not f]; accp=np.mean([p==l for l,p in parsed])
    dacc=np.mean([dp==l for l,_,_,dp,_ in rows])
    unk=np.mean([l=='Unknown' for l,_,_,_,_ in rows])
    fb_unk=np.mean([f and l=='Unknown' for l,p,f,_,_ in rows])
    res[seed]=dict(n=n,acc=acc,fallback=fb,acc_strict=strict,acc_parsed_only=accp,n_parsed=len(parsed),direct_acc=dacc,unknown_share=unk,fallback_and_gold_unknown=fb_unk)
    print(seed,{k:(round(v,4) if isinstance(v,float) else v) for k,v in res[seed].items()})
print('misses',misses[0])
json.dump(res,open(os.path.join(os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '.','cot_rescore.json'),'w'),indent=1,default=float)

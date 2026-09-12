"""Summarize actual ratings only, with paired scenario bootstrap."""
from pathlib import Path
from collections import defaultdict
import numpy as np
from .data import read_csv,dump


def summarize(root):
    key={(r['case_id'],r['answer_id']):r['method'] for r in read_csv(root/'artifacts/selector/review/test-key.csv')}
    files=[root/f'deliverables/selector/review/test-reviewer_{i}.csv' for i in (1,2)]
    records=[];seen=set();missing=0
    scales=('behavior','voice','relevance')
    flags=('fabricated_history','factual_error','invalid')
    for i,p in enumerate(files,1):
        for r in read_csv(p):
            if r['reviewer'] != f'reviewer_{i}': raise ValueError('Reviewer identity does not match rating file')
            identity=(r['reviewer'],r['case_id'],r['answer_id'])
            if identity in seen: raise ValueError('Duplicate rating')
            seen.add(identity)
            if (r['case_id'],r['answer_id']) not in key: raise ValueError('Unknown blinded case/answer')
            if any(r[k]=='' for k in scales+flags): missing+=1;continue
            for k in scales+flags:
                r[k]=float(r[k])
                valid=range(1,6) if k in scales else (0,1)
                if r[k] not in valid: raise ValueError('Rating outside rubric')
            r['method']=key[(r['case_id'],r['answer_id'])]; records.append(r)
    missing=max(missing,800-len(records))
    report={'status':'pending' if missing else 'complete','completed_ratings':len(records),'missing_ratings':missing,
            'unit':'scenario, averaging two reviewers; not independent answer rows','comparisons':{}}
    if missing:
        dump(root/'reports/selector/human_evaluation.json',report);return report
    if len(records)!=800: raise ValueError('Expected two raters x100cases x4answers')
    groups=defaultdict(list)
    for r in records: groups[(r['case_id'],r['method'])].append(r)
    if any(len(v)!=2 for v in groups.values()): raise ValueError('Two distinct reviewer ratings required')
    rng=np.random.default_rng(310)
    for stratum,lo,hi in [('primary',0,60),('secondary',60,80),('controls',80,100)]:
        report['comparisons'][stratum]={}
        for left,right in [('C','B'),('D','C'),('B','A')]:
            metric='relevance' if stratum=='controls' else 'behavior'
            by_metric={}
            for measure in scales+flags:
                scores=[]
                for c in sorted({r['case_id'] for r in records}):
                    if not lo<=int(c.rsplit('-',1)[1])<hi: continue
                    a=np.mean([r[measure] for r in groups[(c,left)]])
                    b=np.mean([r[measure] for r in groups[(c,right)]])
                    scores.append(a-b)
                samples=np.mean(rng.choice(scores,size=(2000,len(scores)),replace=True),axis=1)
                by_metric[measure]={'n_scenarios':len(scores),'mean_difference':float(np.mean(scores)),
                         'ci95':np.quantile(samples,[.025,.975]).tolist()}
            report['comparisons'][stratum][left+'-'+right]={'metric':metric,**by_metric[metric],
                'all_metrics':by_metric,'interpretation':'Exploratory; a wide interval is inconclusive. Lower error flags are better.'}
    agreement={}
    for metric in scales+flags:
        pairs=[(v[0][metric],v[1][metric]) for v in groups.values()]
        agreement[metric]={'exact_agreement':float(np.mean([a==b for a,b in pairs])),
                           'mean_absolute_difference':float(np.mean([abs(a-b) for a,b in pairs]))}
    report['agreement']=agreement
    report['by_method']={m:{k:float(np.mean([r[k] for r in records if r['method']==m])) for k in scales+flags} for m in ('A','B','C','D')}
    import json
    domains={}
    for p in (root/'artifacts/selector/responses/test').glob('*.json'):
        case=json.loads(p.read_text(encoding='utf-8'))['case']
        domains[case['case_id']]=case['domain']
    if set(domains)!={r['case_id'] for r in records}: raise ValueError('Missing frozen scenario-domain metadata')
    report['domain_breakdown']={domain:{'n_scenarios':len({r['case_id'] for r in records if domains[r['case_id']]==domain}),
        'methods':{m:{k:float(np.mean([r[k] for r in records if domains[r['case_id']]==domain and r['method']==m]))
                   for k in scales+flags} for m in ('A','B','C','D')}} for domain in sorted(set(domains.values()))}
    report['domain_rule']='Predeclared context keywords; mixed/other retained, not human-validated domains'
    dump(root/'reports/selector/human_evaluation.json',report)
    return report

"""Frozen diagnostic probes; no tuning or training from their results."""
import re
import numpy as np
from .data import LABELS,dump,digest
from .features import pooled
from .experiment import Selector,verify
from .encoder import Encoder


def macro_f1(gold,predicted):
    scores=[]
    for c in LABELS:
        tp=sum(a==c and b==c for a,b in zip(gold,predicted))
        fp=sum(a!=c and b==c for a,b in zip(gold,predicted))
        fn=sum(a==c and b!=c for a,b in zip(gold,predicted))
        scores.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)
    return float(np.mean(scores))


def run(root):
    verify(root);selector=Selector(root,'test');encoder=Encoder(root/'data/cache/selector')
    rows=[];seen=set()
    for label in LABELS:
        pool=sorted([r for r in selector.rows if r['split']=='test' and r['label']==label],key=lambda r:digest(r['row_id']))
        n=0
        for r in pool:
            if r['group_id'] in seen:continue
            seen.add(r['group_id']);rows.append(r);n+=1
            if n==20:break
        if n!=20:raise ValueError('Insufficient stress-test groups')
    def text(r,original=False):
        response=r['response' if original else 'model_response']
        context=r['context' if original else 'model_context']
        return response if selector.scorer.source=='response' else context+'\nTARGET: '+response
    anon=[text(r) for r in rows]
    raw=[text(r,True) for r in rows]
    pattern=r"\b(?:I am|I'm|I tend to be)\s+(?:(?:very|quite|rather|highly)\s+)?(?:open-minded|conscientious|extroverted|extraverted|introverted|agreeable|neurotic|outgoing|organized|anxious)\b"
    masked=[re.sub(pattern,'[SELF-DESCRIPTION]',t,flags=re.IGNORECASE) for t in anon]
    report={'n_groups':len(rows),'masked_changed_rows':sum(a!=b for a,b in zip(anon,masked)),
            'scope':'Frozen logistic-only diagnostics on200 test groups; no model selection', 'variants':{}}
    gold=[r['label'] for r in rows]
    for name,texts in [('anonymized',anon),('original_names',raw),('masked_direct_self_description',masked)]:
        p=selector.scorer.predict(pooled(encoder,texts,'stress-'+name))
        predicted=[selector.scorer.classes[i] for i in p.argmax(axis=1)]
        report['variants'][name]={'macro_f1':macro_f1(gold,predicted),'accuracy':float(np.mean(np.array(gold)==predicted))}
    dump(root/'reports/selector/stress.json',report)
    return report

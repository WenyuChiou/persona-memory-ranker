"""Application adapter: caller supplies only already-permitted memories."""
import json
import numpy as np
from .features import Scorer,pooled
from .ranking import rank_candidates,pack,memory_block
from persona_memory_ranker.retrieval import Encoder


def retrieve(root,request):
    allowed={'query','trait','persona_id','memories','method','budget'}
    if set(request)-allowed:
        raise ValueError('Unknown request fields; answer/evidence labels are forbidden')
    if not isinstance(request.get('query'),str) or not request['query'].strip():
        raise ValueError('Nonempty query required')
    budget=request.get('budget',2000)
    if type(budget) is not int or not 0<=budget<=2000:
        raise ValueError('budget must be an integer within 0..2000')
    model=Scorer(root/'models/selector/logistic.json')
    if request.get('trait') not in model.classes: raise ValueError('Unknown trait')
    memories=request.get('memories',[])
    if not isinstance(memories,list): raise ValueError('memories must be a list')
    permitted={'memory_id','context','response','source_ref','memory_type','owner_id'}
    for m in memories:
        if set(m)-permitted: raise ValueError('Unknown memory fields')
        if not all(isinstance(m.get(k),str) and m[k].strip() for k in ('memory_id','context','response','source_ref','memory_type')):
            raise ValueError('Memory metadata and original context/response required')
        if m['memory_type']=='owned_episode' and (not request.get('persona_id') or m.get('owner_id')!=request['persona_id']):
            raise ValueError('Owned episode belongs to another persona')
    if not memories: return {'selected':[],'memory_block':'','token_upper_bound':0}
    encoder=Encoder(root/'data/cache/selector')
    x=pooled(encoder,[m['response'] if model.source=='response' else m['context']+'\nTARGET: '+m['response'] for m in memories],'new-memory')
    probs=model.predict(x)
    situation=pooled(encoder,[m['context'] for m in memories],'new-situation')
    q=pooled(encoder,[request['query']],'new-query')[0]
    centers=np.asarray(json.loads((root/'artifacts/selector/clusters.json').read_text(encoding='utf-8'))['centers'])
    centers/=np.maximum(np.linalg.norm(centers,axis=1,keepdims=True),1e-12)
    candidates=[]
    for i,m in enumerate(memories):
        candidates.append(dict(memory_id=m['memory_id'],text=m['context']+'\nTARGET: '+m['response'],
             source_ref=m['source_ref'],memory_type=m['memory_type'],owner_id=m.get('owner_id'),
             similarity=float(situation[i]@q),trait_score=float(probs[i,model.classes.index(request['trait'])]),
             class_scores=dict(zip(model.classes,map(float,probs[i]))),cluster_id=int(np.argmax(situation[i]@centers.T))))
    # Validate the entire request before limiting candidates, including duplicates.
    from .ranking import validate
    validate(candidates)
    candidates=sorted(candidates,key=lambda c:(-c['similarity'],c['memory_id']))[:50]
    weights=json.loads((root/'artifacts/selector/weights.json').read_text(encoding='utf-8'))['weights']
    method=request.get('method','C')
    selected=pack(rank_candidates(candidates,method,weights.get(method,.5)),lambda t:len(t.encode()),budget)
    block=memory_block(selected)
    return {'selected':selected,'memory_block':block,'token_upper_bound':len(block.encode()),
            'budget_unit':'UTF-8 byte upper bound; conservative for local byte-token models',
            'model_schema':model.document['schema_version'],'method':method}

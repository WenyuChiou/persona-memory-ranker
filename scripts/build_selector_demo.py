"""Publish validation examples only; blinded test cases stay out of the demo."""
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
target=root/'web/selector/data.js'
target.parent.mkdir(parents=True,exist_ok=True)
def optional(path,default):
    p=root/path
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else default
cases=[json.loads(p.read_text(encoding='utf-8')) for p in sorted((root/'artifacts/selector/responses/val').glob('*.json'))]
if cases:
    import numpy as np
    from persona_memory_selector.features import Scorer
    scorer=Scorer(root/'models/selector/logistic.json')
    base=root/f'artifacts/selector/features_{scorer.source}'
    metadata=json.loads(base.with_suffix('.json').read_text(encoding='utf-8'))
    positions={identifier:i for i,identifier in enumerate(metadata['row_ids'])}
    matrix=np.load(base.with_suffix('.npy'),mmap_mode='r',allow_pickle=False)
    memories={m['memory_id']:m for case in cases for response in case['responses'].values() for m in response['selected']}
    identifiers=sorted(memories)
    scores=scorer.predict(matrix[[positions[identifier] for identifier in identifiers]]) if identifiers else []
    by_id={identifier:dict(zip(scorer.classes,map(float,p))) for identifier,p in zip(identifiers,scores)}
    for case in cases:
        for response in case['responses'].values():
            for memory in response['selected']:
                memory['class_scores']=by_id[memory['memory_id']]
                memory['predicted_label']=max(memory['class_scores'],key=memory['class_scores'].get)
                memory['predicted_score']=memory['class_scores'][memory['predicted_label']]
human=optional('reports/selector/human_evaluation.json',{})
data={'cleaning':optional('reports/selector/cleaning.json',{}),
      'classification':optional('artifacts/selector/selection.json',{}),
      'human_status':('人工盲評已完成；配對比較與不確定性請見 repo 的 human_evaluation.json。' if human.get('status')=='complete' else '待兩位評分者完成盲評，尚無人格效果結論'),
      'cases':cases,'source':'https://huggingface.co/datasets/wenkai-li/big5_chat',
      'revision':'adf1cd37997b498ff7b220827eaacdeb8aa6d905'}
target.write_text('window.SELECTOR_DATA = '+json.dumps(data,ensure_ascii=False).replace('</','<\\/')+';\n',encoding='utf-8')
print(json.dumps({'validation_cases':len(cases),'output':str(target)}))

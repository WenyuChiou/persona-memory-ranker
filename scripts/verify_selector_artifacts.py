"""Acceptance checks against real source rows and exported R scores."""
import json
from pathlib import Path
import numpy as np
from persona_memory_selector.data import read_csv,check_source_record,dump
from persona_memory_selector.features import Scorer

root=Path(__file__).resolve().parents[1]
raw=read_csv(root/'data/raw/selector/big5_chat_dataset.csv')
rows=read_csv(root/'data/processed/selector/rows.csv')
ids=[r['row_id'] for r in rows]
assert len(ids)==len(set(ids)), 'Duplicate prepared row IDs'
group_splits={};group_roles={}
for r in rows:
    i=int(r['row_id'].split('-')[1]);source=raw[i]
    check_source_record(r,source,i)
    group_splits.setdefault(r['group_id'],set()).add(r['split'])
    group_roles.setdefault(r['group_id'],set()).add(r['role'])
assert all(len(s)==1 for s in group_splits.values()), 'Group crosses split'
assert all(len(s)==1 for s in group_roles.values()), 'Group crosses memory/query role'
report={'source_rows_checked':len(rows),'original_text_and_speaker_metadata_match':True,
        'feature_allowlist_exact':True,'isolated_retained_groups':len(group_splits),'features':{}}
for variant in ('context','response','situation'):
    base=root/f'artifacts/selector/features_{variant}'
    meta=json.loads(base.with_suffix('.json').read_text(encoding='utf-8'))
    x=np.load(base.with_suffix('.npy'),mmap_mode='r',allow_pickle=False)
    assert meta['row_ids']==ids, 'Feature rows reordered'
    assert x.shape==(len(rows),384) and np.isfinite(x).all(), 'Invalid feature matrix'
    report['features'][variant]={'shape':list(x.shape),'finite':True}
fixture=root/'artifacts/selector/parity_fixture.csv'
if fixture.exists():
    scorer=Scorer(root/'models/selector/logistic.json')
    source=read_csv(fixture)
    expected=read_csv(root/'artifacts/selector/parity_expected_probabilities.csv')
    assert [r['row_id'] for r in source]==[r['row_id'] for r in expected], 'Parity fixture order mismatch'
    x=np.array([[float(r[c]) for c in scorer.document['input_feature_order']] for r in source])
    y=np.array([[float(r[c]) for c in scorer.classes] for r in expected])
    error=float(np.max(np.abs(scorer.predict(x)-y)))
    assert error<1e-8, f'R/Python mismatch: {error}'
    report['r_python_parity']={'rows':len(source),'max_absolute_error':error,'tolerance':1e-8}
else:
    report['r_python_parity']={'status':'pending training export'}
dump(root/'reports/selector/acceptance.json',report)
print(json.dumps(report,indent=2))

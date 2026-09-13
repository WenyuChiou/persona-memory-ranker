"""Report measured classifier/completion evidence without inventing human ratings."""
import csv,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
def load(path,default=None):
    p=root/path
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else default
clean=load('reports/selector/cleaning.json',{})
selection=load('artifacts/selector/selection.json',{})
human=load('reports/selector/human_evaluation.json',{'status':'pending','completed_ratings':0})
human_complete=human['status']=='complete'
lines=['# Persona Memory Selector results','',
 ('Classification results concern synthetic generation targets. The completed blinded human analysis is available in human_evaluation.json.' if human_complete else
  'These results concern synthetic generation-target classification. Human persona-quality effects remain pending until actual blinded ratings are complete.'),'',
 '## Data','',f"Raw: {clean.get('raw_rows')}; retained: {clean.get('kept_rows')}; excluded: {clean.get('excluded_rows')}.",'',
 '## Validation classification','', '| Model | Macro-F1 |','|---|---:|']
for name,value in selection.get('candidates',{}).items(): lines.append(f'| {name} | {value:.4f} |')
lines+=['',f"Selected offline model: {selection.get('chosen_model','pending')}. Portable model: {selection.get('deployed_model','pending')}.",'',
 '## Frozen test classification','']
p=root/'artifacts/selector/test_metrics.csv'
if p.exists():
    with p.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    lines+=['| Model | Test Macro-F1 | Test rows |','|---|---:|---:|']
    for row in rows:
        lines.append(f"| {row['candidate']} | {float(row['macro_f1']):.4f} | {row['n']} |")
    lines+=['','Response-only logistic and TF-IDF logistic are close on this split; no significance claim is made. Adding context to the fixed classification representation reduced performance. Context is still used for retrieval.']
else:lines+=['Pending. No final test result is inferred from validation.']
completion={}
for split in ('val','test','test-diagnostics'):
    paths=list((root/f'artifacts/selector/responses/{split}').glob('*.json'))
    cases=[load(p.relative_to(root)) for p in paths]
    responses=[r for c in cases for r in c['responses'].values()]
    completion[split]={'groups':len(cases),'answers':len(responses),
        'valid_completions':sum(r.get('valid_completion') is True for r in responses),
        'invalid_completions':sum(r.get('valid_completion') is False for r in responses)}
lines+=['','## Local generation','', '```json',json.dumps(completion,indent=2),'```','',
 '## Human evaluation','',f"Status: {human['status']}; completed rating rows: {human.get('completed_ratings',0)}. Two reviewers x100 situations x4 answers =800 expected rows.",'',
 'No ranking method is declared better at persona behavior from classifier scores alone. Graph advantage is a separate D-minus-C hypothesis.','',
 '## Reproducibility','',
 'See [the artifact manifest](artifact_manifest.json), [the frozen protocol copy](frozen.json), source_manifest.json, features.json and the local generation cache. Measured data, model and result copies retain their original bytes and hashes; frozen.json records the documented package-only encoder move. Packs use a conservative 2,000-byte upper bound and at most 5 examples. Local AI completion counts are not human quality scores.','',
 'The 200-group name-shift diagnostic is reported in stress.json. Direct-self-description masking changed zero selected rows, so that diagnostic provides no evidence about removal of explicit trait statements.',
]
out=root/'reports/selector/RESULTS.md';out.parent.mkdir(parents=True,exist_ok=True);out.write_text('\n'.join(lines)+'\n',encoding='utf-8')
(root/'reports/selector/generation_status.json').write_text(json.dumps(completion,indent=2),encoding='utf-8')
print(out)

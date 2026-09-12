"""Pinned input, auditable cleaning, and transitive scenario isolation."""
from __future__ import annotations

import csv
import hashlib
import json
import re
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

REVISION = 'adf1cd37997b498ff7b220827eaacdeb8aa6d905'
SHA256 = '1a3a81cac3d12f864c6f2926254fcfb379eb6270f6d45652cf5291dadddb206b'
URL = f'https://huggingface.co/datasets/wenkai-li/big5_chat/resolve/{REVISION}/big5_chat_dataset.csv'
TRAITS = ('openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism')
LABELS = sorted(f'{t}_{v}' for t in TRAITS for v in ('high', 'low'))


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0]) if rows else [])
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def normalize(text):
    return re.sub(r'\s+', ' ', text).strip()


def anonymize(text, x, y):
    names = {x: 'PERSON_X', y: 'PERSON_Y'}
    valid = [n for n in names if n.strip()]
    if not valid:
        return normalize(text)
    pattern = r'(?<!\w)(' + '|'.join(re.escape(n) for n in sorted(valid, key=len, reverse=True)) + r')(?!\w)'
    return normalize(re.sub(pattern, lambda m: names[m.group()], text))


def check_source_record(row, source, index):
    """Verify each feature-bearing field and its target against original bytes."""
    mc=anonymize(source['train_input'],source['personx'],source['persony'])
    mr=anonymize(source['train_output'],source['personx'],source['persony'])
    expected=dict(context=source['train_input'],response=source['train_output'],
                  personx=source['personx'],persony=source['persony'],original_index=source['original_index'],
                  model_context=mc,model_response=mr,model_text=mc+'\nTARGET: '+mr,
                  label=source['trait']+'_'+source['level'],
                  source_ref=f'hf:wenkai-li/big5_chat@{REVISION}/big5_chat_dataset.csv#row={index+2}')
    for field,value in expected.items():
        if row.get(field)!=value: raise ValueError(f'Source mismatch in {field}')
    if 'train_instruction' in row: raise ValueError('Generation instruction included')


def clean_rows(raw):
    parent = list(range(len(raw)))
    def find(a):
        while a != parent[a]:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)
    # Include incomplete rows when constructing groups: missing responses cannot
    # disconnect two otherwise related scenarios and cause leakage.
    keys = {}
    for i, row in enumerate(raw):
        for kind, value in [('id', row.get('original_index', '')),
                            ('head', normalize(row.get('head', '')).casefold()),
                            ('input', anonymize(row.get('train_input', ''), row.get('personx',''), row.get('persony','')).casefold())]:
            if value:
                key = (kind, value)
                if key in keys:
                    union(i, keys[key])
                else:
                    keys[key] = i
    components = defaultdict(list)
    for i in range(len(raw)):
        components[find(i)].append(i)
    assignments = {}
    group_sizes = {}
    for members in components.values():
        gid = digest('|'.join(sorted({str(raw[i].get('original_index', i)) for i in members})))[:20]
        group_sizes[gid] = len(members)
        # Deterministic, label-independent allocation. Never balance by splitting a group.
        u = int(digest('split310:' + gid)[:12], 16) / 16**12
        split = 'train' if u < .7 else ('val' if u < .85 else 'test')
        role = 'train' if split == 'train' else ('memory' if int(digest('role310:' + gid)[:8],16) % 2 == 0 else 'query')
        for i in members:
            assignments[i] = gid, split, role
    rows, excluded = [], []
    for i, row in enumerate(raw):
        reasons = [f'missing_{f}' for f in ('original_index','train_input','train_output','head','personx','persony') if not row.get(f, '').strip()]
        label = row.get('trait', '') + '_' + row.get('level', '')
        if label not in LABELS:
            reasons.append('invalid_label')
        if reasons:
            excluded.append({'raw_row':i+2, 'reason':';'.join(reasons)})
            continue
        gid, split, role = assignments[i]
        x, y = row['personx'], row['persony']
        context, response = row['train_input'], row['train_output']
        mc, mr = anonymize(context, x, y), anonymize(response, x, y)
        rows.append(dict(row_id=f'big5-{i:06d}', group_id=gid, split=split, role=role,
                         label=label, original_index=row['original_index'], context=context, response=response,
                         model_context=mc, model_response=mr, model_text=mc+'\nTARGET: '+mr,
                         scenario=normalize(row['head']), personx=x, persony=y,
                         source_ref=f'hf:wenkai-li/big5_chat@{REVISION}/big5_chat_dataset.csv#row={i+2}',
                         memory_type='behavioral_exemplar'))
    report = dict(raw_rows=len(raw), kept_rows=len(rows), excluded_rows=len(excluded),
                  missing_input=sum(not r.get('train_input','').strip() for r in raw),
                  missing_response=sum(not r.get('train_output','').strip() for r in raw),
                  groups=len(components), largest_groups=sorted(group_sizes.values(),reverse=True)[:20],
                  split_counts=dict(Counter(r['split'] for r in rows)),
                  label_counts=dict(Counter(r['label'] for r in rows)),
                  role_counts=dict(Counter(r['split']+':'+r['role'] for r in rows)),
                  rule='Transitive original_index, normalized head, anonymized nonempty input; seed310 hash allocation.')
    return rows, excluded, report


def prepare(root):
    rawpath = root/'data/raw/selector/big5_chat_dataset.csv'
    rawpath.parent.mkdir(parents=True, exist_ok=True)
    if not rawpath.exists():
        data = urllib.request.urlopen(URL, timeout=120).read()
        if hashlib.sha256(data).hexdigest() != SHA256:
            raise ValueError('Official data hash mismatch')
        rawpath.write_bytes(data)
    if hashlib.sha256(rawpath.read_bytes()).hexdigest() != SHA256:
        raise ValueError('Existing raw data hash mismatch')
    rows, excluded, report = clean_rows(read_csv(rawpath))
    write_csv(root/'data/processed/selector/rows.csv', rows)
    write_csv(root/'reports/selector/exclusions.csv', excluded, ['raw_row','reason'])
    dump(root/'reports/selector/cleaning.json', report)
    dump(root/'reports/selector/source_manifest.json', dict(url=URL, revision=REVISION, sha256=SHA256,
         license='Apache-2.0 (official dataset card)', bytes=rawpath.stat().st_size))
    return report

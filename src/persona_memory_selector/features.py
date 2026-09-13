"""Fixed windowed MiniLM features and portable R softmax inference."""
import json
import hashlib
from pathlib import Path
import numpy as np
from .encoder import Encoder
from .config import MODEL, MODEL_REVISION, WINDOW_TOKENS, WINDOW_OVERLAP
from .data import read_csv, write_csv, dump

FEATURE_SPEC={'encoder':MODEL,'revision':MODEL_REVISION,'window_tokens':WINDOW_TOKENS,
              'window_overlap':WINDOW_OVERLAP,'pooling':'window-mean-then-l2-v1','dtype':'float32'}


def feature_signature(texts):
    return hashlib.sha256(json.dumps({'texts':texts,'spec':FEATURE_SPEC},sort_keys=True).encode()).hexdigest()


def pooled(encoder, texts, namespace):
    unique = list(dict.fromkeys(texts))
    chunks, owners = [], []
    for i,t in enumerate(unique):
        windows=encoder.windows(t)
        chunks.extend(windows); owners.extend([i]*len(windows))
    vectors=encoder.encode(chunks,namespace)
    result=np.zeros((len(unique),384),dtype=np.float32)
    np.add.at(result,owners,vectors)
    result /= np.maximum(np.linalg.norm(result,axis=1,keepdims=True),1e-12)
    lookup={s:i for i,s in enumerate(unique)}
    return result[[lookup[t] for t in texts]]


def build(root):
    rows=read_csv(root/'data/processed/selector/rows.csv')
    encoder=Encoder(root/'data/cache/selector')
    import torch
    if torch.cuda.is_available():
        encoder.model.to('cuda')
        original_encode=encoder.model.encode
        encoder.model.encode=lambda texts,**kw:original_encode(texts,**(kw|{'batch_size':16,'show_progress_bar':True}))
    manifest={}
    for kind,column in [('context','model_text'),('response','model_response'),('situation','model_context')]:
        texts=[r[column] for r in rows]
        signature=feature_signature(texts)
        path=root/f'artifacts/selector/features_{kind}.npy'
        meta=path.with_suffix('.json')
        cache_valid=path.exists() and meta.exists() and json.loads(meta.read_text(encoding='utf-8'))['input_sha256']==signature
        if cache_valid:
            values=np.load(path,allow_pickle=False)
        else:
            values=pooled(encoder,texts,'selector-'+kind)
            path.parent.mkdir(parents=True,exist_ok=True)
            np.save(path,values,allow_pickle=False)
            dump(meta,{'input_sha256':signature,'feature_spec':FEATURE_SPEC,'shape':list(values.shape),'row_ids':[r['row_id'] for r in rows]})
        if values.shape!=(len(rows),384) or not np.isfinite(values).all():
            raise ValueError('Invalid feature matrix')
        csvpath=path.with_suffix('.csv')
        if not csvpath.exists() or not cache_valid:
            import csv
            with csvpath.open('w',encoding='utf-8',newline='') as f:
                w=csv.writer(f); w.writerow(['row_id']+[f'e{i:03d}' for i in range(384)])
                for r,v in zip(rows,values):
                    w.writerow([r['row_id']]+[format(float(x),'.9g') for x in v])
        manifest[kind]={'rows':len(rows),'sha256':hashlib.sha256(csvpath.read_bytes()).hexdigest()}
        print(f'{kind}: {len(rows)} rows encoded',flush=True)
    dump(root/'reports/selector/features.json',manifest)


class Scorer:
    def __init__(self,path):
        self.document=json.loads(Path(path).read_text(encoding='utf-8'))
        d=self.document
        if d['schema_version']!='selector-logistic-v1':
            raise ValueError('Unknown selector model schema')
        self.classes=d['class_order']
        self.source=d['feature_source']
    def predict(self,x):
        d=self.document; p=d['preprocessing']; m=d['model']
        x=np.asarray(x,dtype=float)
        if x.ndim!=2 or x.shape[1]!=len(d['input_feature_order']) or not np.isfinite(x).all():
            raise ValueError('Invalid feature shape or values')
        scale=np.asarray(p['scale'])
        if not np.isfinite(scale).all() or np.any(scale<=0):
            raise ValueError('Invalid exported scales')
        z=((x-np.asarray(p['center']))/scale)[:,p['selected_indices']]
        logits=z@np.asarray(m['coefficients']).T+np.asarray(m['intercept'])
        if not np.isfinite(logits).all():
            raise ValueError('Nonfinite logits')
        e=np.exp(logits-logits.max(axis=1,keepdims=True))
        return e/e.sum(axis=1,keepdims=True)

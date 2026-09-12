import argparse
import json
from pathlib import Path
from . import data,features,experiment


def main(argv=None):
    p=argparse.ArgumentParser(description='Persona Memory Selector: local classified behavioral exemplar retrieval')
    p.add_argument('--root',type=Path,default=Path.cwd())
    p.add_argument('command',choices=['prepare','features','tune','freeze','generate','blind-export','verify','retrieve','review-summary','stress'])
    p.add_argument('--split',choices=['val','test'],default='test')
    p.add_argument('--limit',type=int)
    p.add_argument('--input',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--diagnostics',action='store_true')
    a=p.parse_args(argv);root=a.root.resolve()
    if a.limit is not None and a.limit < 1: p.error('--limit must be positive')
    if a.command in ('prepare','features') and (root/'artifacts/selector/frozen.json').exists():
        raise ValueError('Study frozen; use a separate workspace for new development')
    if a.command=='prepare': result=data.prepare(root)
    elif a.command=='features': result=features.build(root)
    elif a.command=='tune': result=experiment.tune(root)
    elif a.command=='freeze': result=experiment.freeze(root)
    elif a.command=='verify': result=experiment.verify(root)
    elif a.command=='generate': result=experiment.run(root,a.split,a.limit,a.diagnostics)
    elif a.command=='blind-export': result=experiment.blind_export(root,a.split)
    elif a.command=='review-summary':
        from .review import summarize
        result=summarize(root)
    elif a.command=='stress':
        from .stress import run
        result=run(root)
    else:
        from .interface import retrieve
        if not a.input: p.error('retrieve requires --input')
        result=retrieve(root,json.loads(a.input.read_text(encoding='utf-8')))
    if a.output: data.dump(a.output,result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__': main()

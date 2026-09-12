"""Self-contained offline rating UI without method keys or retrieval metadata."""
import json
import hashlib
from pathlib import Path
root=Path(__file__).resolve().parents[1]
template='''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Persona 盲評</title>
<style>body{font:17px/1.7 system-ui;max-width:1050px;margin:30px auto;padding:20px;background:#f7f8f4;color:#183237}select,button,input,textarea{font:inherit;padding:8px;max-width:100%}article{background:white;padding:20px;margin:20px 0;border-top:3px solid #477b68}pre{white-space:pre-wrap;font:inherit}label{display:inline-block;margin:6px}textarea{display:block;width:100%}.notice{background:#efe8d8;padding:15px}h1{font-size:28px}</style>
<h1>人格回答盲評</h1><p class="notice">請獨立評分，不查看方法對照表。分開評估立場、語氣與切題程度。未填寫的欄位保持空白。這裡不會自動判定哪個方法較好。</p>
<label>評分者 <select id="reviewer"><option>reviewer_1</option><option>reviewer_2</option></select></label>
<label>題目 <select id="case"></select></label><button id="save">下載目前評分 CSV</button><p id="progress"></p>
<h2 id="trait"></h2><blockquote id="query"></blockquote><div id="answers"></div>
<p>behavior／voice／relevance：1 明顯不符或無用；3 中性／部分；5 清楚符合且自然。fabricated_history／factual_error／invalid：0 沒有，1 有。事實控制題的 behavior 與 voice 可填3，主要看正確與切題程度。</p>
<script>const DATA=__DATA__;const SPLIT=__SPLIT__;const STUDY=__STUDY__;
const $=x=>document.getElementById(x);const metrics=['behavior','voice','relevance','fabricated_history','factual_error','invalid'];
let values={};try{values=JSON.parse(localStorage.getItem('selector-ratings-'+SPLIT+'-'+STUDY)||'{}')}catch(e){alert('本機暫存無法讀取，請先保留既有 CSV。')}
function put(tag,text,parent){const n=document.createElement(tag);n.textContent=text;parent.append(n);return n;}
DATA.forEach((c,i)=>{const o=put('option',c.case_id,$('case'));o.value=i});
function key(c,a){return [$('reviewer').value,c.case_id,a.answer_id].join('|')}
function store(){try{localStorage.setItem('selector-ratings-'+SPLIT+'-'+STUDY,JSON.stringify(values))}catch(e){$('progress').textContent='無法暫存，請立即下載 CSV';}}
function render(){const c=DATA[Number($('case').value)||0];if(!c)return;$('trait').textContent=c.trait;$('query').textContent=c.query;$('answers').replaceChildren();
for(const a of c.answers){const article=put('article','',$('answers'));put('h3',a.answer_id,article);put('pre',a.text,article);const k=key(c,a);values[k]??={};for(const m of metrics){const l=put('label',m+' ',article);const s=put('select','',l);for(const v of ['',...(metrics.indexOf(m)<3?[1,2,3,4,5]:[0,1])]){const o=put('option',v===''?'待評':String(v),s);o.value=v}s.value=values[k][m]??'';s.onchange=()=>{values[k][m]=s.value;store()}}const notes=put('textarea','',article);notes.placeholder='評分依據／分歧備註';notes.value=values[k].notes||'';notes.oninput=()=>{values[k].notes=notes.value;store()}}
$('progress').textContent=`共 ${DATA.length} 組題目，每組四個匿名答案。`;}
$('case').onchange=render;$('reviewer').onchange=render;
$('save').onclick=()=>{const reviewer=$('reviewer').value;const rows=[['reviewer','case_id','answer_id',...metrics,'notes']];for(const c of DATA)for(const a of c.answers){const r=values[[reviewer,c.case_id,a.answer_id].join('|')]||{};rows.push([reviewer,c.case_id,a.answer_id,...metrics.map(m=>r[m]??''),r.notes||''])}const csv=rows.map(row=>row.map(x=>'"'+String(x).replaceAll('"','""')+'"').join(',')).join('\\r\\n');const url=URL.createObjectURL(new Blob(['\\uFEFF'+csv],{type:'text/csv;charset=utf-8'}));const a=document.createElement('a');a.href=url;a.download=SPLIT+'-'+reviewer+'.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};render();</script></html>'''
for split in ('val','test'):
    p=root/f'deliverables/selector/review/{split}-blinded.json'
    if not p.exists(): continue
    data=json.loads(p.read_text(encoding='utf-8'))
    study=hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()
    html=template.replace('__SPLIT__',json.dumps(split)).replace('__STUDY__',json.dumps(study)).replace('__DATA__',json.dumps(data,ensure_ascii=False).replace('</','<\\/'))
    p.with_suffix('.html').write_text(html,encoding='utf-8')
    print(split,len(data))

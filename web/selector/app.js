/* All dataset/model text is inserted as textContent, never executable HTML. */
const data=window.SELECTOR_DATA;
const names={A:'A 固定人格設定',B:'B 加入相似範例',C:'C 加入人格分類',D:'D 加入分類與圖譜'};
const el=id=>document.getElementById(id);
function node(tag,text,parent,cls){const n=document.createElement(tag);n.textContent=text;if(cls)n.className=cls;parent.append(n);return n;}
el('status').textContent=data.human_status;
el('metrics').textContent=`${(data.cleaning.kept_rows||0).toLocaleString()} 筆清理後資料 · ${data.cases.length} 個實際驗證案例`;
if(data.classification.candidates){const p=node('p','驗證集分類 Macro-F1（生成標籤辨識）',el('metrics').parentElement);for(const [name,value] of Object.entries(data.classification.candidates))node('div',`${name}: ${(100*value).toFixed(1)}%`,p,'small');}
data.cases.forEach((x,i)=>{const o=node('option',`${i+1}. ${x.case.trait} — ${x.case.query.slice(0,65)}`,el('case'));o.value=i;});
let active='D';
function svgNode(tag,attrs,text){const n=document.createElementNS('http://www.w3.org/2000/svg',tag);Object.entries(attrs).forEach(([k,v])=>n.setAttribute(k,v));if(text)n.textContent=text;el('graph').append(n);return n;}
function draw(rows,trait){el('graph').replaceChildren();const mid=190;
 svgNode('text',{x:25,y:mid,fill:'#194c44','font-size':18},'目前情境');
 svgNode('text',{x:650,y:mid,fill:'#194c44','font-size':16},trait);
 rows.forEach((r,i)=>{const y=50+i*65;svgNode('line',{x1:115,y1:mid,x2:280,y2:y,stroke:'#9baa9d'});svgNode('line',{x1:470,y1:y,x2:635,y2:mid,stroke:'#c19161'});svgNode('text',{x:290,y:y,fill:'#183237','font-size':16},r.memory_id);svgNode('text',{x:290,y:y+21,fill:'#647770','font-size':12},`情境群 ${r.cluster_id} · 特質分數 ${r.trait_score.toFixed(3)}`);});
 if(!rows.length)svgNode('text',{x:310,y:190,fill:'#647770','font-size':18},'此方法沒有加入記憶');
}
function render(){const d=data.cases[Number(el('case').value)||0];if(!d){el('query').textContent='本機回答仍在計算，完成後更新此頁。';return;}
 el('query').textContent=d.case.query;el('trait').textContent=`指定特質表現：${d.case.trait}`;el('answers').replaceChildren();
 for(const m of ['A','B','C','D']){const a=node('article','',el('answers'),`answer ${m===active?'active':''}`);const b=node('button',names[m],a);b.onclick=()=>{active=m;render();};node('p',d.responses[m].text,a);if(d.responses[m].valid_completion!==true)node('p','此回答未正常完成：'+(d.responses[m].invalid_reason||'完成狀態未知'),a,'notice');node('p',`${d.responses[m].selected.length} 個範例 · ${d.responses[m].memory_budget_upper_bound} bytes 的保守 token 上限`,a,'small');}
 const selected=d.responses[active].selected;draw(selected,d.case.trait);el('memories').replaceChildren();
 selected.forEach(r=>{const p=node('article','',el('memories'),'memory');node('h3',r.memory_id,p);if(r.predicted_label){node('p',`模型預測：${r.predicted_label} · 分數 ${r.predicted_score.toFixed(3)}`,p);const details=node('details','',p);node('summary','查看十類分類分數',details);Object.entries(r.class_scores).sort((a,b)=>b[1]-a[1]).forEach(([label,score])=>node('div',`${label}: ${score.toFixed(3)}`,details,'small'));}node('p',`相似度 ${r.similarity.toFixed(3)} · 目標特質分數 ${r.trait_score.toFixed(3)} · 排序分數 ${r.rank_score.toFixed(4)}`,p,'small');node('pre',r.text,p);node('code',r.source_ref,p);});}
el('case').onchange=render;render();

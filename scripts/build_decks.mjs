// Uses the Codex bundled @oai/artifact-tool runtime. The PPTX remains editable.
import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {createRequire} from 'node:module';

const root = path.resolve(process.cwd());
const milestone = process.argv[2];
if (!['M1','M2'].includes(milestone)) throw new Error('Expected M1 or M2');
const skill = process.env.PMR_PRESENTATIONS_SKILL;
const modules = process.env.PMR_NODE_MODULES;
const python = process.env.PMR_RUNTIME_PYTHON;
if (![skill,modules,python].every(p=>p && path.isAbsolute(p))) throw new Error('Set PMR_PRESENTATIONS_SKILL, PMR_NODE_MODULES and PMR_RUNTIME_PYTHON');
process.env.RUNTIME_NODE_MODULES = modules;
process.env.RUNTIME_PYTHON = python;
process.env.RUNTIME_NODE = process.execPath;
const require = createRequire(path.join(modules,'package.json'));
const {Presentation,PresentationFile} = await import(pathToFileURL(require.resolve('@oai/artifact-tool')).href);
const {resolvePresentationFont,applyPresentationChartFont,finalizePresentation} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const input = JSON.parse(await fs.readFile(path.join(root,`artifacts/presentations/${milestone}-content.json`),'utf8'));
const font = resolvePresentationFont();
const out = path.join(root,'artifacts/presentations',milestone);
await fs.mkdir(out,{recursive:true});
const deck = Presentation.create({slideSize:{width:1280,height:720}});
function text(slide,value,x,y,w,h,size=28,bold=false,color='#142735') {
  const box=slide.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
  box.text=value;box.text.style={typeface:font,fontSize:size,bold,color,autoFit:'none'};
  return box;
}
for (let i=0;i<input.slides.length;i++) {
  const info=input.slides[i],s=deck.slides.add();s.background.fill='#FAFCFD';
  text(s,info.title,70,44,1140,i===0?145:100,i===0?62:43,true);
  if (info.chart) {
    text(s,info.body.join('\n'),74,132,1130,115,25);
    const chart=s.charts.add('bar',{position:{left:90,top:270,width:1090,height:380},categories:info.chart.categories,
      series:info.chart.series.map(x=>({...x,fill:'#207D83'})),barOptions:{direction:'column',grouping:'clustered'},hasLegend:false,
      dataLabels:{showValue:true,position:'outEnd'}});
    applyPresentationChartFont(chart,{fontFamily:font});
  } else {
    text(s,info.body.join('\n\n'),74,i===0?250:198,1120,400,i===0?31:31);
  }
  text(s,`${milestone} · ${i+1}/${input.slides.length}`,74,672,1100,30,16,false,'#526876');
  s.speakerNotes.textFrame.setText(info.notes+'\n\nSource: '+info.source+'\nProject reports generated from pinned dataset revision.');
  const preview=await deck.export({slide:s,format:'png',scale:1});
  await fs.writeFile(path.join(out,`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await preview.arrayBuffer()));
}
const candidate=path.join(out,'candidate.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidate);
const finalPath=path.join(root,`deliverables/${milestone}-presentation.pptx`);
const chartSlides=input.slides.flatMap((s,i)=>s.chart?[i+1]:[]);
await finalizePresentation({workspaceDir:root,candidatePath:candidate,finalPath,pythonExecutable:python,
  integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
  explicitTotalSlideCount:input.slides.length,requiredNativeChartOwnerSlides:chartSlides,
  materializeLiteralChartWorkbooks:true,
  fontPolicy:{basis:'design',families:[font]},verifyArtifactToolImport:true,
  receiptPath:path.join(out,'validation.json')});
console.log(JSON.stringify({milestone,slides:input.slides.length,finalPath,font}));

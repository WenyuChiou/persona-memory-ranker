// Inspect rendered slides from the finalized PPTX, rather than only its draft.
import fs from 'node:fs/promises';
import path from 'node:path';
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';
const milestone=process.argv[2];
if (!['M1','M2'].includes(milestone)) throw new Error('Expected M1 or M2');
const revision=process.argv[3] || '';
if (revision && !/^v[1-9][0-9]*$/.test(revision)) throw new Error('Revision must be v1, v2, ...');
const modules=process.env.PMR_NODE_MODULES;
if (!modules) throw new Error('Set PMR_NODE_MODULES');
const require=createRequire(path.join(modules,'package.json'));
const {PresentationFile,FileBlob}=await import(pathToFileURL(require.resolve('@oai/artifact-tool')).href);
const root=process.cwd();
const deck=await PresentationFile.importPptx(await FileBlob.load(path.join(root,`deliverables/${milestone}-presentation${revision ? '-'+revision : ''}.pptx`)));
const snapshot=await deck.inspect({kind:'slide',maxChars:20000});
const records=snapshot.ndjson.trim().split('\n').map(JSON.parse).filter(r=>r.kind==='slide');
for(const record of records) {
  const slide=deck.resolve(record.id);
  const image=await deck.export({slide,format:'png',scale:1});
  await fs.writeFile(path.join(root,`artifacts/presentations/${milestone}/slide-${String(record.slide).padStart(2,'0')}.png`),new Uint8Array(await image.arrayBuffer()));
}
console.log(JSON.stringify({milestone,final_slides_rendered:records.length}));

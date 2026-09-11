# Persona Memory Ranker static explorer

This directory is a self-contained interface for **saved experiment evidence**.
It performs no server inference and requires no CDN, API keys, font downloads, or
live query service. `demo-data.js` is generated from completed local artifacts.
Until that bundle is built, the page explicitly shows that evidence is absent.

From the installed project environment:

```powershell
python scripts/build_demo.py --split benchmark --limit 24
```

For development evidence, use `--split val`. The page prominently displays
**DEVELOPMENT / VALIDATION**, retains the subset marker from the evaluation report,
and never presents these rows as held-out benchmark results. `--root PATH` selects
another project artifact root; the bundle is written to that root's `web/` folder.
The command refuses mismatched evaluation/query keys and detects artifact changes
during bundling. It does not call training, feature generation, or evaluation.

Open `web/index.html` directly in a browser, or serve `web/` with any static server.
The directory can be published on GitHub Pages after a real bundle is built.
Creating these files does not deploy a website.

The builder selects at most 24 questions by default using a seeded SHA-256 ordering
of query IDs after simple topic/query-text exclusions. Selection never examines
performance. All candidates for each selected query are retained; source memory
units are shared across cases to limit bundle size. This is a showcase sample,
not the dataset-level evaluation. Metrics at the bottom come directly from the
saved report at the experiment's fixed context budget and do not change with the
interactive slider.

`ranking.js` reproduces Python's descending-score / ascending-ID tie rule and
greedy whole-unit budget packing. Unlabeled candidates are not called irrelevant.
Missing annotated memories remain in the recall denominator. The source inspector
renders original role/content messages with `textContent`, preserving full quotes.
Source links point to the pinned public PersonaMem-v2 revision. JSON is escaped
for `<`, `>`, `&`, U+2028, and U+2029; it is never interpolated into HTML.

Checks:

```powershell
node --check web/app.js
node --check web/ranking.js
node web/tests.js
python -m py_compile scripts/build_demo.py
python web/test_build_demo.py
```

Source attribution is displayed in the interface: PersonaMem-v2 by bowen-upenn,
CC BY 4.0. The data are synthetic and annotations are incomplete. Derived memory
units and ranking scores are project outputs; original source quotes are preserved.

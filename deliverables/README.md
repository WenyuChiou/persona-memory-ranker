# Course delivery files

- **M0:** `M0-proposal.md` contains the approved title, abstract and dataset attribution. The Chrome form was filled and saved as a draft; it was not submitted.
- **M1:** `../R/eda.Rmd` is the executable R notebook; `M1-eda.html` is the generated M1 HTML; `../reports/eda.html` also includes the final benchmark analysis. `M1-presentation.pptx` is editable, and `M1-speaker-notes.md` contains the narration draft.
- **M2:** `M2-presentation-v2.pptx` (16 slides) and `M2-speaker-notes.md` accompany the complete source, results and static evidence explorer. This revision includes paired uncertainty and subgroup failures.

Videos are generated locally under `deliverables/video/` and excluded from Git:

| File | Target duration | Narration |
|---|---:|---|
| M1-presentation.mp4 | 10 minutes | Microsoft David Desktop, synthetic English |
| M2-presentation.mp4 | 15 minutes | Microsoft David Desktop, synthetic English |

The videos and notes are preparation aids. Review them, add your own explanation and follow the instructor's requirements before course submission. A generated voice does not represent a recording of the student.

## Rebuilding presentation artifacts

Run `python scripts/presentation_content.py --milestone M1` (or `M2`) after the relevant reports exist. M2 verifies the frozen benchmark provenance. The native PowerPoint builder uses the Codex bundled `@oai/artifact-tool` runtime and presentation utilities, located through the Codex workspace dependency tool. Set `PMR_NODE_MODULES`, `PMR_RUNTIME_PYTHON` and `PMR_PRESENTATIONS_SKILL` to those installed paths, then run:

```powershell
node scripts/build_decks.mjs M1
node scripts/render_deck.mjs M1
powershell -File scripts/narrate.ps1 -Milestone M1
python scripts/build_video.py --milestone M1
```

Replace `M1` with `M2` for the final deck, and pass `v2` as the third argument to both Node commands to build/render the delivered revision. Use a fresh revision such as `v3` when rebuilding an existing deck: finalized PowerPoints and validation receipts are protected from overwrite. Narration requires Windows System.Speech and the named voice; video assembly requires FFmpeg and ffprobe on PATH. You can also edit the delivered PowerPoint directly and record your own narration using your presentation software.

The numerical experiment itself uses ordinary Python/R dependencies and does not require the presentation runtime.

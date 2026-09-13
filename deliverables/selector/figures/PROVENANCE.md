# Workflow figure provenance

Both figures were created with the built-in image generation tool following `technical-visual-explainer`, then visually inspected. The model's internal version was not exposed by the tool. Relationship semantics are specified in `RELATIONSHIP_MAP.md`.

- `data-workflow.png`: source generation `exec-8c6bf2bf-2d16-48b3-ac90-1eb1bf0391b3.png`. Required relationships: BIG5-CHAT → cleaning → group split; training → R fit → validation selection → freeze; held-out test + frozen method → final evaluation. Verified 100,000 raw and 99,656 retained labels, isolated test path, and readable arrows.
- `answer-workflow.png`: first generation `exec-2cfaa258-89a7-4d26-a76f-f5da421b66dd.png`; targeted edit `exec-d28dcbf0-4617-4790-ab78-74ab0a103ada.png`. The edit restored the missing C-to-shared-memory-pack connection. Verified that B/C/D start from the same candidate set, all feed the shared pack, persona prompt enters prompt assembly, and human ratings are marked pending.
- Final answer-diagram edit `exec-613ff5b4-a747-405d-b558-d3f671b471db.png` adds the current question as a separate input to prompt assembly. This is the same question used by similarity search. Parent visually verified the new input, unchanged B/C/D connections, memory-pack budget, fixed persona prompt, and pending human evaluation.

Figures use only project-generated content and public data labels. They explain the implemented research design and do not depict measured human outcomes. The figures are raster images embedded in PowerPoint; surrounding text, chart data and speaker notes remain editable.

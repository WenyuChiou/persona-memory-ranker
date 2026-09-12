# Figure relationship specification

Tool: built-in imagegen. Style skill: technical-visual-explainer. Audience: DSCI310 students. White background, navy labels; teal data, blue operations, purple model/method, green checks, gold human review. English labels match the slides. Generated workflow images are not empirical charts or fully editable vector diagrams.

## Data workflow
Raw BIG5-CHAT -> Clean records -> Group split.
Group split -> Training data -> Fit classifiers (R).
Group split -> Validation data -> Select model.
Fit classifiers -> Select model -> Freeze method.
Group split -> Test data -> Final evaluation.
Freeze method -> Final evaluation.
Training-fitted vocabulary/transforms cannot receive validation or test input during fitting. Show no arrow from test data back to selection. No claim final metrics already available.

## Answer workflow
Question -> Similarity search. Permitted example bank -> Similarity search -> Same 50 candidates.
Same candidates -> B Vector; -> C Classification + flat; -> D Classification + graph (parallel alternatives, never a chain).
Target trait -> C and D. Each method -> Memory pack (max5, budget cap2000 tokens).
Memory pack -> Prompt assembly. Fixed persona prompt -> Prompt assembly -> Local AI -> Answer.
Current question (the same query used in similarity search) -> Prompt assembly.
A prompt-only uses the same fixed prompt with empty memory. Examples are not autobiographical episodes.
Answer -> Human ratings (pending); no evaluation arrow back to runtime.

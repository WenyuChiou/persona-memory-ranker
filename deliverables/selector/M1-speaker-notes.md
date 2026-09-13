# M1 spoken script

Timing is a planning estimate, not a measured rehearsal. Human ratings remain pending.

## 1. Persona Memory Selector (40 seconds planned)

Imagine an AI playing a careful, dependable colleague. A client suddenly asks for an earlier deadline. A personality instruction tells the AI what this colleague is like, but it does not show how that colleague might respond under pressure. Our project asks whether selecting suitable behavioral examples helps. We compare four ways to give the same AI information. The course contribution is data cleaning, classification, and a controlled comparison. A separate human evaluation will determine whether the answers actually fit the role. Classification accuracy alone cannot answer that question.

Source: https://huggingface.co/datasets/wenkai-li/big5_chat

## 2. The deadline example (40 seconds planned)

The key decision is how to respond to the request. The character could accept immediately, ask which requirements can change, or explain why the original commitment matters. Several responses may be reasonable. We want to know whether the selected examples help the AI make a choice that fits the target character and the current situation. Simply retrieving the word deadline is not enough. Equally, adding a very polite example does not necessarily improve the decision. This example explains why we score behavioral choices and speaking style separately.

Source: https://huggingface.co/datasets/wenkai-li/big5_chat

## 3. Ten labels from five traits (40 seconds planned)

BIG5-CHAT provides existing labels, so we do not need to invent a large annotation project before we can train a classifier. Each record has a conversation input and a target response generated under one specified personality setting. There are ten target labels in total. A label tells us what the generation process requested. It does not guarantee that every response successfully expresses the trait. It also does not measure the other four traits. Our model predicts these dataset targets, and we keep that interpretation separate from claims about real human personality.

Source: https://huggingface.co/datasets/wenkai-li/big5_chat

## 4. One record, two different speakers (50 seconds planned)

Here is a real training record. One speaker proposes an outing for the children. The target speaker prefers the familiar routine and worries about a mess or bad weather. Its generation label is low openness. Notice that the worries could also suggest another trait. This is why we call these generation targets, not verified psychological truths. The two speakers have different names in the source. For the model, we replace those names with role markers, while keeping the original text and record identifier. That prevents a name from becoming an easy shortcut and keeps the prediction auditable.

Source: https://huggingface.co/datasets/wenkai-li/big5_chat

## 5. Cleaning results (50 seconds planned)

The raw file contains one hundred thousand records. Our cleaning procedure retains ninety-nine thousand six hundred and fifty-six. We exclude three hundred and forty-four records that lack required information. The original audit found two hundred and ninety-four missing inputs and sixty-five missing responses. These counts overlap, so adding them would give the wrong exclusion total. We keep a row-level exclusion report and never ask a language model to invent the missing text. The original file is checked against a fixed hash, which makes the source bytes reproducible.

Source: reports/selector/cleaning.json

## 6. Keeping related scenarios together (60 seconds planned)

A random split by row would be misleading. The same original scenario appears under different personality settings. If one version went into training and another into testing, the model could recognize the scenario rather than generalize. We connect records sharing an original identifier, event description, or equivalent input. Whole connected groups move together. Training data fit the model. Validation data choose between approaches. Test data remain reserved until the method is fixed. Within validation and test, we also separate questions from the examples available to retrieval, so the system cannot retrieve the answer’s own scenario.

Source: reports/selector/cleaning.json

## 7. How text becomes model input (50 seconds planned)

A statistical classifier needs numbers rather than raw paragraphs. TF-IDF creates one representation based on how distinctive each word is in the training collection. A fixed sentence encoder gives us another representation based on learned similarities between texts. We do not train that language model in this course project. We compare whether the response alone is enough or whether including the preceding context helps classification. Long text is encoded in windows instead of silently cutting off the end. Vocabulary and fitted transformations are learned only from the training portion of each comparison.

Source: https://huggingface.co/datasets/wenkai-li/big5_chat

## 8. Classifying generation labels (50 seconds planned)

These are measured validation results, not human personality ratings. Logistic regression using the response alone reaches 58.3 percent Macro-F1. Adding the context to the fixed embedding reaches 39.7 percent. So, with this representation and these data, more text did not improve classification. That does not make context useless: our retrieval step still uses it to find similar situations. The word-based TF-IDF baseline reaches 58.0 percent, very close to the best score. We select the highest validation score, but do not claim that this small margin proves statistical superiority. Random forest is a course comparator with a limited parameter grid. These results show why a simple baseline and a response-only comparison are necessary before choosing a more complex design.

Source: artifacts/selector/selection.json

## 9. Four controlled comparisons (60 seconds planned)

Every group uses the same target persona prompt and the same question. Group A receives no examples. Group B retrieves examples based on similarity. Group C also considers the classifier’s score for the requested trait. Group D uses a graph connecting examples with situation clusters and trait scores. The three retrieval methods start from the same fifty candidates and share the same evidence budget. Comparing C with B estimates the value of classification. Comparing D with C tests the graph’s additional value. Comparing B with A asks the more basic question of whether adding examples helps at all.

Source: https://huggingface.co/datasets/wenkai-li/big5_chat

## 10. The graph and the prompt (50 seconds planned)

The graph is a structure for finding examples. It is not a map of psychological causes. A situation group connects related conversation contexts, and a trait connection stores a classifier score. The selected source text becomes an evidence block next to the fixed persona instruction. The prompt explicitly says these are examples, not events the AI personally experienced. That rule keeps the experiment honest: public dialogue examples can guide a response, but they cannot silently become the character’s biography.

Source: https://huggingface.co/datasets/wenkai-li/big5_chat

## 11. Measuring the answers (60 seconds planned)

The final evaluation asks two people to rate answers without seeing which method produced them. There are sixty primary-model personality situations, twenty situations for a second local model, and twenty factual or calculation controls. Each group contains four answers in random order. We score decisions and voice separately and record whether an answer invents personal history. The controls check whether personality examples interfere with ordinary useful answers. Both reviewers first discuss separate practice questions. Their real ratings remain blank until they complete the task. We cannot replace those ratings with the classifier’s own opinion.

Source: https://huggingface.co/datasets/wenkai-li/big5_chat

## 12. What this project can establish (50 seconds planned)

The useful outcome is a component that can be inspected and reused, together with evidence about when it helps. It may turn out that classification helps but the graph adds little. In that case the simpler flat selector is the supported result. A wide uncertainty interval means we need more evidence, not that the graph has been disproved. This study uses synthetic dialogues and two local models. It cannot establish human psychological validity or universal transfer to all personalities. The current delivery includes the software and evaluation materials; any claim about better persona behavior must wait for the human ratings.

Source: https://huggingface.co/datasets/wenkai-li/big5_chat
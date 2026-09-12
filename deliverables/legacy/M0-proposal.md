# Milestone 0

**Name:** Wenyu Chiou

**Project Title:** PersonaGraph: Classifying and Retrieving Behavioral Memories for Adaptive AI Personas

**Project Abstract**

Personalized AI agents often retrieve facts about a user but fail to retrieve behavioral evidence that should shape how they respond. This project investigates whether classical machine learning can identify, classify, and retrieve evidence-backed persona memories under a fixed prompt budget. Using the public PersonaMem-v2 text dataset, we will clean and segment long conversation histories, create a human-reviewed taxonomy of communication, decision, planning, motivation, risk, emotional-regulation, and social-interaction memories, and build a provenance graph linking each behavioral-memory node to its original conversation turns. Logistic regression and random forest models in R will first distinguish persona-relevant memories from ordinary information and then classify their behavioral dimensions. A class-aware ranker will select graph nodes for a dynamic persona prompt. Evaluation will use persona-separated cross-validation, macro-F1, Recall@5, MRR@10, fixed-budget recall, and blinded comparisons against no-persona, fact-only, semantic-retrieval, and shuffled-persona prompts. The deliverables will include reproducible R analyses, an inspectable persona-memory graph demo, and a reusable prompt-compilation interface. The study evaluates evidence-backed persona adaptation in synthetic conversations; it does not diagnose personality or establish psychological causality.

**Data Source:** [PersonaMem-v2](https://huggingface.co/datasets/bowen-upenn/PersonaMem-v2), text benchmark and 32k conversation histories. CC BY 4.0. Revision `ed956dea41521fc4499acbc63f966e0fd3c053ba`. The source evidence annotations support the retrieval pilot; the new persona-memory classification labels will be documented as a separate human-reviewed derived dataset.

The Chrome form was filled and saved as a draft. Submission has not been performed.

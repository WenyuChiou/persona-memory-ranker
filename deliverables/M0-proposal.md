# Milestone 0

**Name:** Wenyu Chiou

**Project Title:** Persona Memory Selector: Classifying Behavioral Exemplars for Consistent AI Role-Playing

**Project Abstract**

This project tests whether classified behavioral examples help an AI maintain a specified persona when answering new situations. Using the labeled BIG5-CHAT synthetic dialogue dataset, we will clean source records, isolate related scenarios across data splits, and train multinomial logistic regression and random forest models in R. We will compare a fixed persona prompt, vector retrieval, classification-aware flat retrieval, and graph retrieval under a shared context budget. Two locally hosted language models and blinded human ratings will evaluate behavioral consistency, voice, relevance, and fabricated personal history. Deliverables include reproducible R analyses, a source-linked interactive demo, and a reusable memory-selection interface. The study evaluates a component for persona applications rather than diagnosing human personality.

**Data Source:** [BIG5-CHAT](https://huggingface.co/datasets/wenkai-li/big5_chat), Apache-2.0 according to the official dataset card. Revision `adf1cd37997b498ff7b220827eaacdeb8aa6d905`.

**白話摘要：** 同一個 AI 套用同一份人格設定後，我們讓它取用不同方法挑選的行為範例，比較它是否更能維持角色的立場與語氣。課堂重點是資料清理、分類模型與公平的實驗比較。公開對話是參考範例，不能冒充角色親身經歷。

**Submission status:** This file is updated. The earlier Chrome form draft may contain the previous proposal; this implementation has not submitted a form.

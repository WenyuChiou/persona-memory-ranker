# 課程交付與閱讀順序

這個專案測試：AI 套用同一份人格設定後，能否透過分類過的行為範例，做出更符合角色的回答。

1. 先讀[白話資料處理步驟](DATA_WALKTHROUGH.md)與[研究設計](../../docs/SELECTOR_PROTOCOL.md)，確認研究問題、資料來源與限制。
2. 查看[清理報告](../../reports/selector/cleaning.json)與 [R EDA](../../reports/selector/eda.html)，了解原始資料如何變成可訓練的資料。
3. 查看[實驗結果](../../reports/selector/RESULTS.md)，分清楚「辨識生成標籤」與「回答更符合角色」是不同成果。
4. 開啟[互動 Demo](https://wenyuchiou.github.io/persona-memory-ranker/selector/)，比較實際驗證案例與取回的範例。
5. 正式評分先讀[評分手冊](REVIEW_GUIDE.md)，再使用離線評分網頁。答案對照表保留在本機 artifacts，不公開。

## 簡報與流程圖

簡報對象是修 DSCI310 的同學。使用 **Presentations** 製作可編輯的文字、圖表與講者備註，使用 **research-talk-coach** 安排例子與說明順序。流程圖使用你常用的 **technical-visual-explainer**，以 imagegen 產生；圖像本身是 PNG，圖上的節點不是 PowerPoint 原生形狀。

- [M1：清理、EDA 與分類，10 分鐘](M1-presentation-v4.pptx)；[完整講稿](M1-speaker-notes.md)。
- [M2：完整實驗設計與自動結果，15 分鐘](M2-presentation-v4.pptx)；[完整講稿](M2-speaker-notes.md)。人工效果評估仍待完成。
- [資料處理流程圖](figures/data-workflow.png)與[回答／檢索流程圖](figures/answer-workflow.png)；[製圖來源與檢查紀錄](figures/PROVENANCE.md)。

簡報與英文講稿供課堂報告；本頁、資料步驟與評分手冊提供中文說明。合成語音影片保存在本機 `deliverables/video/selector/`，供排練參考，不代表學生已錄製或提交作業。

## 人工盲評

本機開啟 `deliverables/selector/review/val-blinded.html` 做10組練習，再使用 `test-blinded.html` 做100組正式評分。評分網頁與 CSV 保留本機，不推送 GitHub；需要重建時執行 `python scripts/build_blind_review.py`（先完成回答生成與 blind-export）。

兩位評分者各填一份 CSV。同一題的四個答案順序已隨機排列。

| 欄位 | 評分方式 |
|---|---|
| behavior | 1：立場與目標表現相反；3：無明顯表現；5：情境中的選擇清楚符合角色 |
| voice | 1：語氣明顯不符；3：中性；5：自然符合角色，無須自稱人格標籤 |
| relevance | 1：沒有回答問題；3：部分回應；5：切合情境且有用 |
| fabricated_history | 1：把參考範例說成自己的往事；0：沒有 |
| factual_error | 1：有可確認的事實／計算錯誤；0：沒有 |
| invalid | 1：空白、截斷、無法理解或未有效作答；0：有效 |

先用10組練習題討論規則，正式100組先各自評分，再討論分歧。評分不要只看「親切」或「保守」：目標可能是低宜人性或高開放性。無效回答不能因為沒有冒犯就得到高分。

人工欄位目前預留空白。執行 `pms review-summary` 只會分析真的已填寫資料。

## 課程時間

M0：9/12 題目與摘要。M1：10/10 清理、EDA、初步分類及10分鐘說明。M2：11/28 完整比較、Demo、簡報與15分鐘說明。

簡報與講稿會清楚註明哪些結果已完成、哪些需要人工評分。自動產生影片不代表學生已完成錄影或老師已收到作業。

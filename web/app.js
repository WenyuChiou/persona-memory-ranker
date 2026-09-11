(function () {
  "use strict";
  const $ = id => document.getElementById(id);
  const methods = {
    bm25: ["BM25", "Lexical retrieval: matches question terms to memory text."],
    semantic: ["Semantic similarity", "Embedding similarity: compares meaning beyond exact word overlap."],
    rrf: ["RRF", "Reciprocal rank fusion: combines lexical and semantic retrieval ranks."],
    logistic: ["Logistic reranker", "A small R-trained model combining lexical, semantic, overlap, length, and history-position features."],
    logistic_no_position: ["Logistic · no position", "The logistic ablation removes history position to inspect dependence on this order proxy."],
    random_forest: ["Random forest", "An R-trained probability forest using the same five features as the full logistic model."],
  };
  const controls = ["query-search", "query-select", "method-select", "budget"];
  const data = window.PMR_DATA;
  let query = null;
  let activeMemory = null;
  let state = null;
  const integer = value => value.toLocaleString("en-US");
  const percent = value => (value * 100).toFixed(1) + "%";
  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }
  function badge(text, className = "") { return element("span", className, text); }
  function sourceHref(memory) {
    const expected = `https://huggingface.co/datasets/bowen-upenn/PersonaMem-v2/blob/${data.dataset_revision}/`;
    if (!/^[a-f0-9]{40}$/.test(data.dataset_revision) || !memory.source_url.startsWith(expected)) {
      throw new Error("Source URL does not match the pinned public dataset revision");
    }
    return memory.source_url;
  }
  function showError(error) {
    $("load-status").textContent = "Evidence could not be loaded: " + error.message;
    $("load-status").classList.add("error");
    $("load-status").setAttribute("role", "alert");
    controls.forEach(id => { $(id).disabled = true; });
  }
  function guarded(fn) {
    return event => { try { fn(event); } catch (error) { showError(error); } };
  }
  function filterCases() {
    const term = $("query-search").value.trim().toLocaleLowerCase();
    const matching = data.cases.filter(q => `${q.query} ${q.topic_query} ${q.query_id}`.toLocaleLowerCase().includes(term));
    $("query-select").replaceChildren();
    matching.forEach(q => {
      const option = element("option", "", q.query);
      option.value = q.query_id;
      $("query-select").append(option);
    });
    $("search-count").textContent = `${matching.length} of ${data.cases.length} saved questions`;
    $("query-select").disabled = !matching.length;
    if (!matching.length) {
      $("search-count").textContent += ". No matches; the previously selected question remains visible.";
      return;
    }
    const retained = matching.find(q => q.query_id === query?.query_id);
    $("query-select").value = (retained || matching[0]).query_id;
    if (!retained) selectQuery();
  }
  function selectQuery() {
    query = data.cases.find(q => q.query_id === $("query-select").value);
    if (!query) throw new Error("Unknown saved question");
    activeMemory = null;
    $("query-text").textContent = query.query;
    $("query-meta").replaceChildren(badge(query.topic_query || "No topic label"), badge(query.query_id),
      badge(query.updated ? "Update annotation: yes" : "Update annotation: no"));
    renderCase();
  }
  function renderCase() {
    const method = $("method-select").value;
    const budget = Number($("budget").value);
    $("budget-value").textContent = `${integer(budget)} tokens`;
    $("method-description").textContent = methods[method][1];
    state = window.PMRRanking.evaluateCase(query, method, budget);
    if (!activeMemory || !state.ordered.some(r => r.memory_id === activeMemory)) {
      activeMemory = (state.selected[0] || state.ordered[0])?.memory_id || null;
    }
    $("selected-count").textContent = integer(state.selected.length);
    $("tokens-count").textContent = integer(state.tokenCount);
    $("recall-count").textContent = percent(state.annotatedRecall);
    $("recall-count").title = `${state.annotatedSelected} of ${state.annotatedTotal} annotated memory units selected`;
    $("candidate-count").textContent = integer(state.ordered.length);
    $("pool-coverage").textContent = `${percent(state.candidateRecall)} annotation coverage in pool`;
    renderCandidates(method);
    renderGraph();
    renderSource();
    document.querySelectorAll("#results-rows tr").forEach(row => row.classList.toggle("current-method", row.dataset.method === method));
  }
  function activateMemory(memoryId) {
    activeMemory = memoryId;
    document.querySelectorAll("#candidate-rows tr").forEach(row => {
      row.classList.toggle("is-active", row.dataset.memoryId === memoryId);
      row.querySelector("button").setAttribute("aria-pressed", String(row.dataset.memoryId === memoryId));
    });
    renderGraph();
    renderSource();
  }
  function renderCandidates(method) {
    const gold = new Set(query.gold_memory_ids);
    const packed = new Set(state.selected.map(r => r.memory_id));
    const fragment = document.createDocumentFragment();
    state.ordered.forEach((row, index) => {
      const memory = data.memories[row.memory_id];
      if (!memory) throw new Error("A saved candidate is missing its original messages");
      const tr = element("tr", activeMemory === row.memory_id ? "is-active" : "");
      tr.dataset.memoryId = row.memory_id;
      const cell = element("td");
      const button = element("button");
      button.type = "button";
      button.setAttribute("aria-label", `Open rank ${index + 1}, ${row.memory_id}, ${gold.has(row.memory_id) ? "annotated support" : "unlabeled"}`);
      button.setAttribute("aria-pressed", String(activeMemory === row.memory_id));
      button.addEventListener("click", guarded(() => activateMemory(row.memory_id)));
      const heading = element("span", "memory-heading");
      heading.append(badge(String(index + 1).padStart(2, "0"), "rank-number"), element("strong", "", row.memory_id));
      const preview = memory.text.length > 140 ? memory.text.slice(0, 140) + "…" : memory.text;
      button.append(heading, element("span", "memory-preview", preview),
        badge(gold.has(row.memory_id) ? "Annotated support" : "Unlabeled", "support-label" + (gold.has(row.memory_id) ? "" : " unlabeled")));
      cell.append(button);
      const score = row[method];
      const scoreCell = element("td", "", score === 0 ? "0.0000" : Math.abs(score) < 0.0001 ? score.toExponential(2) : score.toFixed(4));
      scoreCell.title = String(score);
      const contextCell = element("td");
      contextCell.append(badge(packed.has(row.memory_id) ? "Selected" : "—", packed.has(row.memory_id) ? "packed" : "not-packed"),
        badge(`${integer(row.token_count)} tokens`, "token-mini"));
      tr.append(cell, scoreCell, contextCell);
      fragment.append(tr);
    });
    $("candidate-rows").replaceChildren(fragment);
  }
  const svgNamespace = "http://www.w3.org/2000/svg";
  function svgElement(tag, attributes, text) {
    const node = document.createElementNS(svgNamespace, tag);
    Object.entries(attributes || {}).forEach(([key, value]) => node.setAttribute(key, String(value)));
    if (text !== undefined) node.textContent = text;
    return node;
  }
  function graphNode(x, y, width, label, subtitle, className, activate) {
    const group = svgElement("g", {class: "graph-node " + className, transform: `translate(${x},${y})`});
    group.append(svgElement("rect", {width, height: 38, rx: 4}),
      svgElement("text", {x: 12, y: 16}, label),
      svgElement("text", {x: 12, y: 29, class: "graph-subtitle"}, subtitle));
    if (activate) {
      group.classList.add("interactive");
      group.setAttribute("tabindex", "0");
      group.setAttribute("role", "button");
      group.setAttribute("aria-label", `${label}: ${subtitle}`);
      group.addEventListener("click", guarded(activate));
      group.addEventListener("keydown", guarded(event => {
        if (event.key === "Enter" || event.key === " ") { event.preventDefault(); activate(); }
      }));
    }
    return group;
  }
  function renderGraph() {
    const svg = $("provenance-graph");
    svg.replaceChildren();
    const visible = state.selected.slice(0, 4);
    const gold = new Set(query.gold_memory_ids);
    const sourceY = 85;
    visible.forEach((row, index) => {
      const y = 14 + index * 47 + (4 - visible.length) * 23.5;
      svg.append(svgElement("path", {d: `M164,104 C230,104 230,${y + 19} 290,${y + 19}`, class: "graph-edge"}),
        svgElement("path", {d: `M474,${y + 19} C532,${y + 19} 532,104 590,104`, class: "graph-edge"}));
    });
    svg.append(graphNode(5, sourceY, 159, "Saved question", query.query_id, "query"));
    visible.forEach((row, index) => {
      const y = 14 + index * 47 + (4 - visible.length) * 23.5;
      const className = (gold.has(row.memory_id) ? "annotated " : "") + (activeMemory === row.memory_id ? "active" : "");
      svg.append(graphNode(290, y, 184, row.memory_id, `${integer(row.token_count)} tokens · ${gold.has(row.memory_id) ? "annotated" : "unlabeled"}`,
        className, () => activateMemory(row.memory_id)));
    });
    svg.append(graphNode(590, sourceY, 164, "Original history", "Pinned dataset source ↗", "", () => {
      $("source-link").focus();
      $("source-link").scrollIntoView({block: "nearest"});
    }));
    $("graph-note").textContent = visible.length
      ? `Showing ${visible.length} of ${state.selected.length} selected units. ${state.annotatedSelected}/${state.annotatedTotal} annotated units fit this budget. All candidates remain available below.`
      : "No complete memory unit fits this budget. Increase the budget or inspect any candidate below.";
  }
  function renderSource() {
    if (!activeMemory) return;
    const memory = data.memories[activeMemory];
    const row = state.ordered.find(r => r.memory_id === activeMemory);
    const annotated = query.gold_memory_ids.includes(activeMemory);
    const selected = state.selected.some(r => r.memory_id === activeMemory);
    $("source-title").textContent = memory.memory_id;
    $("source-meta").replaceChildren(badge(`History unit ${memory.turn_index + 1}`), badge(`${integer(row.token_count)} tokens`),
      badge(annotated ? "Annotated support" : "Unlabeled", "support-label" + (annotated ? "" : " unlabeled")));
    $("source-state").textContent = selected ? "Selected into the current context. Full original messages follow." : "Outside the current packed context. Full original messages follow.";
    const messages = document.createDocumentFragment();
    memory.messages.forEach(message => {
      const article = element("article", "source-message");
      article.append(element("p", "role-label", message.role.toUpperCase()), element("pre", "", message.content));
      messages.append(article);
    });
    $("source-messages").replaceChildren(messages);
    $("source-link").href = sourceHref(memory);
    $("source-link").hidden = false;
    $("source-link").title = memory.source_ref;
  }
  function renderResults() {
    const subset = data.persona_limit_per_split === null ? "complete evaluation partition" : `${data.persona_limit_per_split}-persona subset per split`;
    $("results-scope").textContent = `${data.split_label}. ${integer(data.evaluation_queries)} evaluated questions; ${subset}. Exact alignment retained ${integer(data.alignment.aligned_queries)} of ${integer(data.alignment.cleaned_queries)} cleaned questions and excluded ${integer(data.alignment.excluded_queries)}. The ${data.cases.length} showcase cases are a separate, illustrative selection.`;
    $("results-caption").textContent = `Saved ${data.split === "val" ? "development / validation" : "held-out benchmark"} report · ${integer(data.reported_budget)}-token context budget · 95% persona-bootstrap intervals`;
    const fragment = document.createDocumentFragment();
    data.methods.forEach(method => {
      const row = element("tr");
      row.dataset.method = method;
      row.append(element("td", "", methods[method][0]));
      ["budget_recall", "recall_at_5", "mrr_at_10"].forEach(metric => {
        const summary = data.summary.find(r => r.method === method && r.metric === metric);
        if (!summary || ![summary.mean, summary.ci_low, summary.ci_high].every(Number.isFinite)) {
          throw new Error("Saved evaluation summary is missing a finite metric");
        }
        const format = metric === "mrr_at_10" ? value => value.toFixed(3) : percent;
        const cell = element("td", "", format(summary.mean));
        cell.append(element("span", "interval", `${format(summary.ci_low)}–${format(summary.ci_high)}`));
        row.append(cell);
      });
      fragment.append(row);
    });
    $("results-rows").replaceChildren(fragment);
  }
  function initialize() {
    if (!data) {
      $("load-status").textContent = "The static interface is ready. Build web/demo-data.js from completed experiment artifacts to explore saved cases; no example scores are fabricated.";
      return;
    }
    if (data.schema_version !== 1 || !data.cases?.length || !["val", "benchmark"].includes(data.split)) {
      throw new Error("Unsupported or empty evidence bundle");
    }
    $("split-marker").textContent = data.split_label;
    $("sample-marker").textContent = `${data.cases.length} saved cases · full candidate pools · no live inference`;
    $("load-status").textContent = "";
    controls.forEach(id => { $(id).disabled = false; });
    renderResults();
    const topics = new Set();
    data.cases.forEach(q => {
      if (topics.size >= 4 || topics.has(q.topic_query) || !q.topic_query) return;
      topics.add(q.topic_query);
      const button = element("button", "", q.topic_query);
      button.type = "button";
      button.addEventListener("click", guarded(() => {
        $("query-search").value = "";
        filterCases();
        $("query-select").value = q.query_id;
        selectQuery();
      }));
      $("presets").append(button);
    });
    const attribution = $("source-attribution");
    attribution.replaceChildren(document.createTextNode("Source: "));
    const datasetLink = element("a", "", "PersonaMem-v2 by bowen-upenn");
    datasetLink.href = `https://huggingface.co/datasets/bowen-upenn/PersonaMem-v2/tree/${data.dataset_revision}`;
    datasetLink.target = "_blank";
    datasetLink.rel = "noopener noreferrer";
    const license = element("a", "", "CC BY 4.0");
    license.href = "https://creativecommons.org/licenses/by/4.0/";
    license.target = "_blank";
    license.rel = "noopener noreferrer";
    attribution.append(datasetLink, document.createTextNode(" · "), license,
      document.createTextNode(`. Revision ${data.dataset_revision.slice(0, 12)}. Memory units and ranking scores are derived by this project; original quoted messages are preserved.`));
    $("query-search").addEventListener("input", guarded(filterCases));
    $("query-select").addEventListener("change", guarded(selectQuery));
    $("method-select").addEventListener("change", guarded(renderCase));
    $("budget").addEventListener("input", guarded(renderCase));
    filterCases();
  }
  try { initialize(); } catch (error) { showError(error); }
})();

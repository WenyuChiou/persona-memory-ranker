/* Pure counterparts of the Python metric ordering and whole-unit packer. */
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.PMRRanking = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";
  const methods = ["bm25", "semantic", "rrf", "logistic", "logistic_no_position", "random_forest"];
  function ranked(rows, method) {
    if (!methods.includes(method)) throw new Error("Unknown ranking method");
    const ids = new Set();
    rows.forEach(row => {
      if (ids.has(row.memory_id)) throw new Error("Duplicate candidate memory ID");
      ids.add(row.memory_id);
      if (!Number.isFinite(row[method])) throw new Error("Nonfinite ranking score");
    });
    // Python's tie rule is lexical code-point order. IDs here are ASCII; do not
    // use localeCompare, whose punctuation ordering depends on the browser locale.
    return [...rows].sort((a, b) => b[method] - a[method] ||
      (a.memory_id < b.memory_id ? -1 : a.memory_id > b.memory_id ? 1 : 0));
  }
  function pack(rows, budget) {
    if (!Number.isInteger(budget) || budget < 0) throw new Error("Budget must be a nonnegative integer");
    let used = 0;
    const seen = new Set();
    const selected = [];
    rows.forEach(row => {
      if (seen.has(row.memory_id)) throw new Error("Duplicate memory in context input");
      seen.add(row.memory_id);
      if (!Number.isInteger(row.token_count) || row.token_count < 1) throw new Error("Invalid token count");
      if (used + row.token_count <= budget) {
        selected.push(row);
        used += row.token_count;
      }
    });
    return selected;
  }
  function evaluateCase(query, method, budget) {
    const ordered = ranked(query.candidates, method);
    const selected = pack(ordered, budget);
    const gold = new Set(query.gold_memory_ids);
    if (!gold.size) throw new Error("Annotated support is required to calculate recall");
    return {ordered, selected, tokenCount: selected.reduce((n, r) => n + r.token_count, 0),
      annotatedSelected: selected.filter(r => gold.has(r.memory_id)).length,
      annotatedTotal: gold.size,
      annotatedRecall: selected.filter(r => gold.has(r.memory_id)).length / gold.size,
      candidateRecall: ordered.filter(r => gold.has(r.memory_id)).length / gold.size};
  }
  return {methods, ranked, pack, evaluateCase};
});

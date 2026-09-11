"""Export verified portable models and feature definitions without raw histories."""
from pathlib import Path

from persona_memory_ranker.config import (DATASET, DATASET_REVISION, FEATURE_NAMES, FEATURE_VERSION,
                                        MODEL, MODEL_REVISION, RRF_K, TOP_N, WINDOW_TOKENS, WINDOW_OVERLAP)
from persona_memory_ranker.io import digest, read_json, stable_hash, write_json
from persona_memory_ranker.pipeline import verify_evaluation, verify_frozen
from persona_memory_ranker.retrieval import LogisticScorer

root = Path(__file__).resolve().parents[1]
frozen = verify_frozen(root)
verify_evaluation(root, "val")
verify_evaluation(root, "benchmark")
selection = read_json(root/"artifacts/models/selection.json")
expected_selection = {"schema_version": 1, "feature_version": FEATURE_VERSION,
                      "selected_method": frozen["selected_portable_method"],
                      "selection_metric": frozen["selection_metric"], "protocol_sha256": stable_hash(frozen)}
if selection != expected_selection:
    raise ValueError("Portable selection does not match the frozen experiment")
destination = root/"models"
destination.mkdir(exist_ok=True)
files = ("logistic.json", "logistic_no_position.json", "selection.json")
for name in files:
    source = root/"artifacts/models"/name
    if name != "selection.json":
        LogisticScorer(read_json(source))
    target = destination/name
    target.write_bytes(source.read_bytes())
    if digest(target) != digest(source):
        raise ValueError(f"Export differs from the verified model: {name}")
definitions = {
    "bm25": "BM25 against the complete eligible history: lowercase Unicode word tokens, k1=1.5, b=0.75.",
    "semantic": "Maximum cosine similarity over all query-window and memory-window pairs; normalized frozen embeddings.",
    "overlap": "Fraction of distinct lowercase query word tokens present in the memory.",
    "log_length": "Natural log of one plus the memory's WordPiece count, including its source header.",
    "position": "Nonnegative turn index divided by the maximum index in the eligible history; denominator at least one."
}
write_json(destination/"feature_spec.json", {
    "schema_version": 1, "feature_version": FEATURE_VERSION,
    "features": [{"name": name, "type": "finite float", "definition": definitions[name]} for name in FEATURE_NAMES],
    "encoder": {"model": MODEL, "revision": MODEL_REVISION, "window_tokens": WINDOW_TOKENS, "overlap_tokens": WINDOW_OVERLAP},
    "candidate_pool": {"rule": "union of BM25 and semantic top-N", "N": TOP_N},
    "rrf_k": RRF_K, "rrf_is_learned_feature": False,
    "memory_unit": "User turn plus following assistant messages; source text and roles preserved.",
    "source_identity": ["source_ref", "turn_index"],
    "rendered_memory_template": "[{memory_id} | {source_ref}]\n{text}",
    "word_tokenizer": {"regex": "\\w+", "lowercase": True, "Unicode": True},
    "preprocessing": "Apply the model JSON's training means and scales in its declared feature order.",
    "scope": "Only rank eligible evidence. The host enforces permissions, validity, mandatory evidence and its own tokenizer budget."
})
write_json(destination/"provenance.json", frozen)
write_json(destination/"manifest.json", {
    "dataset": DATASET, "dataset_revision": DATASET_REVISION, "data_license": "CC BY 4.0",
    "selected_method": frozen["selected_portable_method"],
    "files": {name: digest(destination/name) for name in (*files, "feature_spec.json", "provenance.json")}
})
print(f"Exported verified portable models and feature specification to {destination}")

import json
import math
from pathlib import Path

import numpy as np
import pytest

from persona_memory_ranker.config import FEATURE_NAMES
from persona_memory_ranker.data import align_gold, checked_history_path, make_memories, structured
from persona_memory_ranker.io import digest, read_csv, write_csv, write_json, write_jsonl
from persona_memory_ranker.metrics import aggregate, pack, query_metrics, ranked
from persona_memory_ranker.retrieval import LogisticScorer, MemoryIndex, validate_memories, words


class TestEncoder:
    """Small deterministic test double, never used for reported retrieval results."""
    __test__ = False

    def count(self, text):
        return max(1, len(words(text)))

    def windows(self, text):
        return [text]

    def encode(self, texts, namespace):
        result = []
        for text in texts:
            row = np.array([words(text).count(w) for w in ("books", "quiet", "music", "coffee")], dtype=float) + .01
            result.append(row/np.linalg.norm(row))
        return np.array(result).reshape((-1, 4))


def memories():
    return [dict(memory_id="a", source_ref="same-history", turn_index=0, text="quiet books and coffee"),
            dict(memory_id="b", source_ref="same-history", turn_index=1, text="loud music")]


def exported(names=None):
    names = names or FEATURE_NAMES
    return dict(schema_version=1, feature_version="pmr-v1", model_type="logistic", feature_names=names,
                means=[0]*len(names), scales=[1]*len(names), coefficients=[1]*len(names), intercept=0)


def test_python_literals_parse_without_execution():
    assert structured("{'role': 'user', 'content': 'hello'}")["content"] == "hello"
    with pytest.raises((ValueError, SyntaxError)):
        structured("__import__('os').system('echo unsafe')")


@pytest.mark.parametrize("path", ["../secret", "/tmp/x", "C:/file.json", "data/chat_history_128k/a.json", "data\\chat_history_32k\\a.json"])
def test_dataset_path_rejects_escape_and_wrong_variant(path):
    with pytest.raises(ValueError):
        checked_history_path(path)


def test_unicode_csv_roundtrip(tmp_path):
    rows = [{"id": "001", "query": '安靜的圖書館\n“books, please” ☕', "gold": json.dumps([{"role": "user", "content": "原文"}], ensure_ascii=False)}]
    path = tmp_path / "unicode.csv"
    write_csv(path, rows)
    assert read_csv(path) == rows


def test_turn_segmentation_and_alignment_preserve_original_text():
    messages = [{"role": "user", "content": "  café\nbooks  "}, {"role": "assistant", "content": "OK"}, {"role": "user", "content": "new preference"}]
    units = make_memories(messages, "01", "raw.json")
    assert len(units) == 2
    assert units[0]["messages"][0]["content"] == messages[0]["content"]
    assert align_gold(messages, messages[:2], units) == (["p01:t00000"], "aligned")
    assert align_gold(messages, [{"role": "user", "content": "café\nbooks"}], units)[1] == "missing_exact_sequence"


def test_ambiguous_gold_is_excluded_not_arbitrarily_assigned():
    message = {"role": "user", "content": "repeat"}
    units = make_memories([message, message], "1", "raw.json")
    assert align_gold([message, message], [message], units)[1] == "ambiguous_duplicate_sequence"


def test_embedded_system_persona_is_never_a_memory():
    messages = [{"role": "system", "content": "SECRET GOLD PERSONA"}, {"role": "user", "content": "books"}]
    units = make_memories(messages, "1", "raw.json")
    assert len(units) == 1 and units[0]["memory_id"] == "p1:t00001"
    assert "SECRET" not in units[0]["text"]
    assert align_gold(messages, messages[1:], units) == (["p1:t00001"], "aligned")


def test_role_only_assistant_exclusion_keeps_source_indices():
    messages = [{"role": "user", "content": "first"}, {"role": "assistant"}, {"role": "user", "content": "second"}]
    units = make_memories(messages, "1", "raw.json")
    assert [m["memory_id"] for m in units] == ["p1:t00000", "p1:t00002"]
    assert align_gold(messages, messages[2:], units) == (["p1:t00002"], "aligned")


def test_candidate_features_cannot_read_gold_or_persona_profile():
    original = memories()
    augmented = [m | {"correct_answer": "music", "persona": "secret", "gold_memory_ids": ["b"]} for m in original]
    a = MemoryIndex(original, TestEncoder()).candidates("quiet books")
    b = MemoryIndex(augmented, TestEncoder()).candidates("quiet books")
    assert a == b
    assert all(set(row) == {"memory_id", *FEATURE_NAMES, "rrf", "token_count"} for row in a)
    assert ranked(a, "semantic")[0]["memory_id"] == "a"


def test_empty_candidates_and_invalid_query():
    assert MemoryIndex([], TestEncoder()).candidates("books") == []
    with pytest.raises(ValueError):
        MemoryIndex([], TestEncoder()).candidates(" ")
    assert pack([]) == []
    assert LogisticScorer(exported()).score([]) == []


def test_duplicate_id_rejected_but_same_source_distinct_turns_kept():
    validate_memories(memories())
    with pytest.raises(ValueError, match="Duplicate"):
        validate_memories([memories()[0], memories()[0]])
    with pytest.raises(ValueError, match="Duplicate source turn"):
        validate_memories([memories()[0], memories()[0] | {"memory_id": "different-id"}])


def test_budget_counts_full_units_skips_oversize_and_does_not_truncate():
    rows = [dict(memory_id="large", token_count=2001), dict(memory_id="a", token_count=1500), dict(memory_id="b", token_count=501), dict(memory_id="c", token_count=500)]
    assert [r["memory_id"] for r in pack(rows)] == ["a", "c"]
    assert pack(rows, 0) == []
    with pytest.raises(ValueError):
        pack([rows[0], rows[0]])
    with pytest.raises(ValueError):
        pack([dict(memory_id="invalid", token_count=float("nan"))])


def test_missing_retrievals_remain_in_denominator():
    rows = [dict(memory_id="found", token_count=100, score=.5)]
    result = query_metrics(rows, ["found", "missing"], "score")
    assert result["recall_at_5"] == .5
    assert result["candidate_recall"] == .5
    assert result["budget_recall"] == .5
    assert result["mrr_at_10"] == 1
    assert query_metrics([], ["missing"], "score")["budget_recall"] == 0


def test_ties_are_stable_and_nonfinite_scores_fail():
    assert [r["memory_id"] for r in ranked([dict(memory_id="b", score=1), dict(memory_id="a", score=1)], "score")] == ["a", "b"]
    with pytest.raises(ValueError):
        ranked([dict(memory_id="a", score=float("nan"))], "score")


def test_scorer_allowlist_and_extreme_logits():
    scorer = LogisticScorer(exported())
    assert scorer.score([{k: 0 for k in FEATURE_NAMES}]) == [.5]
    assert scorer.score([{k: 1000 for k in FEATURE_NAMES}]) == [1]
    assert scorer.score([{k: -1000 for k in FEATURE_NAMES}]) == [0]
    with pytest.raises(ValueError, match="allowlist"):
        scorer.score([{k: 0 for k in FEATURE_NAMES} | {"correct_answer": 1}])
    with pytest.raises(ValueError):
        scorer.score([{k: float("nan") for k in FEATURE_NAMES}])
    with pytest.raises(ValueError):
        LogisticScorer(exported() | {"feature_version": "future"})


def test_cluster_bootstrap_reproducible():
    rows = [dict(persona_id=p, method="m", recall_at_5=v, mrr_at_10=v, budget_recall=v, candidate_recall=1)
            for p, v in [("a", 1), ("a", 0), ("b", .5)]]
    a = aggregate(rows, 50)
    assert a == aggregate(rows, 50)
    assert a[0]["queries"] == 3 and a[0]["personas"] == 2
    assert a[0]["mean"] == .5


@pytest.mark.parametrize("command", [["download"], ["prepare"], ["train"], ["features", "--split", "train"]])
def test_frozen_experiment_rejects_development_mutation(tmp_path, command):
    from persona_memory_ranker.cli import main
    (tmp_path/"artifacts").mkdir()
    (tmp_path/"artifacts/frozen_protocol.json").write_text("{}")
    with pytest.raises(ValueError, match="frozen"):
        main(["--root", str(tmp_path), *command])


def test_stale_evaluation_cannot_select_a_deployment_method(tmp_path, monkeypatch):
    from persona_memory_ranker import pipeline
    old = {"files": {"features.csv": "old"}, "source": {}}
    current = {"files": {"features.csv": "changed"}, "source": {}}
    monkeypatch.setattr(pipeline, "evaluation_inputs", lambda *_: current)
    with pytest.raises(ValueError, match="Stale"):
        pipeline.verify_evaluation(tmp_path, "val", {"input_sha256": old})
    paths = ["reports/val_query_metrics.csv", "reports/val_summary.csv"]
    for path in paths:
        write_csv(tmp_path/path, [{"example": 1}])
    result = {"input_sha256": current, "output_sha256": {p: digest(tmp_path/p) for p in paths}}
    assert pipeline.verify_evaluation(tmp_path, "val", result)["input_sha256"] == current
    (tmp_path/paths[0]).write_text("modified metrics\n")
    with pytest.raises(ValueError, match="metric tables"):
        pipeline.verify_evaluation(tmp_path, "val", result)


@pytest.mark.parametrize("field", ["alignment_correct", "other_supporting_memory_ids", "notes", "reviewer", "reviewed_at"])
def test_audit_regeneration_preserves_any_human_input(tmp_path, field):
    from persona_memory_ranker.pipeline import audit_queue
    path = tmp_path/"reports/private/human_audit_100.csv"
    write_csv(path, [{"query_id": "q1", field: "human entry"}])
    before = path.read_bytes()
    with pytest.raises(ValueError, match="reviewer input"):
        audit_queue(tmp_path)
    assert path.read_bytes() == before


def score_fixture(root, split="val"):
    input_split = "train" if split == "cv" else split
    rows = [{"query_id": f"q{p}", "persona_id": p, "memory_id": f"m{p}",
             **{name: 0 for name in FEATURE_NAMES}, "rrf": .02, "label": 1, "token_count": 10} for p in ("a", "b")]
    write_csv(root/f"data/processed/features_{input_split}.csv", rows)
    write_jsonl(root/f"data/processed/{input_split}_queries.jsonl",
                [{"query_id": r["query_id"], "persona_id": r["persona_id"], "gold_memory_ids": [r["memory_id"], "missing"]} for r in rows])
    for name, features in [("logistic", FEATURE_NAMES), ("logistic_no_position", FEATURE_NAMES[:-1])]:
        write_json(root/f"artifacts/models/{name}.json", exported(features))
    predictions = [{"query_id": r["query_id"], "memory_id": r["memory_id"], "persona_id": r["persona_id"],
                    "fold": i+1, "logistic": .5, "logistic_no_position": .5, "random_forest": .4} for i, r in enumerate(rows)]
    write_csv(root/f"artifacts/models/{split}_predictions.csv", predictions)
    return rows, predictions


@pytest.mark.parametrize("method", ["logistic", "logistic_no_position", "random_forest"])
@pytest.mark.parametrize("bad", ["NaN", "Inf", "-1", "2"])
def test_invalid_imported_r_predictions_fail_before_parity(tmp_path, method, bad):
    from persona_memory_ranker.pipeline import load_scored
    _, predictions = score_fixture(tmp_path)
    predictions[0][method] = bad
    write_csv(tmp_path/"artifacts/models/val_predictions.csv", predictions)
    with pytest.raises(ValueError, match="Invalid imported R prediction"):
        load_scored(tmp_path, "val")


def test_oof_uses_heldout_scores_and_full_gold_denominator(tmp_path):
    from persona_memory_ranker.pipeline import load_scored
    _, predictions = score_fixture(tmp_path, "cv")
    predictions[0]["logistic"] = .2
    write_csv(tmp_path/"artifacts/models/cv_predictions.csv", predictions)
    write_csv(tmp_path/"artifacts/models/cv_assignments.csv", [{"persona_id": "a", "fold": 1}, {"persona_id": "b", "fold": 2}])
    write_json(tmp_path/"artifacts/models/training_report.json", {"folds": [
        {"fold": 1, "train_personas": ["b"], "heldout_personas": ["a"]},
        {"fold": 2, "train_personas": ["a"], "heldout_personas": ["b"]}]})
    rows, queries, parity = load_scored(tmp_path, "cv")
    assert parity is None and rows[0]["logistic"] == .2
    assert query_metrics([rows[0]], queries[0]["gold_memory_ids"], "logistic")["budget_recall"] == .5
    predictions[0]["fold"] = 2
    write_csv(tmp_path/"artifacts/models/cv_predictions.csv", predictions)
    with pytest.raises(ValueError, match="wrong held-out"):
        load_scored(tmp_path, "cv")

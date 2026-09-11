"""Offline bundle checks with synthetic temporary artifacts; no published data."""
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("build_demo", Path(__file__).resolve().parents[1] / "scripts/build_demo.py")
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


class DemoBundleTests(unittest.TestCase):
    def test_serialization_and_source_boundary(self):
        text = '</script><img src=x onerror=alert(1)>&\u2028\u2029'
        script = builder.js_bundle({"quote": text})
        assignment = script.split("window.PMR_DATA=", 1)[1].removesuffix(";\n")
        self.assertNotIn("<", assignment)
        self.assertNotIn(">", assignment)
        self.assertNotIn("&", assignment)
        self.assertNotIn("\u2028", assignment)
        self.assertNotIn("\u2029", assignment)
        self.assertEqual(json.loads(assignment)["quote"], text)
        with self.assertRaises(ValueError):
            builder.source_url("https://private.example/persona.json")
        with self.assertRaises(ValueError):
            builder.source_url("data/chat_history_32k/../../private.json")

    def test_complete_candidates_and_development_marker(self):
        with tempfile.TemporaryDirectory(prefix="pmr-demo-test-") as temporary:
            root = Path(temporary)
            def write(relative, value):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(value, encoding="utf-8")
            query = {"query_id": "q001", "persona_id": "p1", "query": "Which books do I enjoy?",
                     "topic_query": "Hobbies", "updated": False, "gold_memory_ids": ["p1:t003", "p1:absent"]}
            rows = [{"query_id": "q001", "memory_id": f"p1:t{i:03}", "token_count": 200,
                     **{method: 1 / i for method in builder.METHODS}} for i in range(1, 105)]
            write("data/processed/val_queries.jsonl", json.dumps(query) + "\n")
            write("reports/features_val.json", json.dumps({"source_query_sha256": builder.digest(root / "data/processed/val_queries.jsonl")}))
            write("data/processed/features_val.csv", "fixture features hashed but score-loading is mocked\n")
            write("artifacts/models/val_predictions.csv", "fixture predictions hashed but score-loading is mocked\n")
            write("artifacts/models/logistic.json", "{}")
            write("artifacts/models/logistic_no_position.json", "{}")
            write("reports/val_evaluation.json", json.dumps({"split": "val", "queries": 1, "persona_limit_per_split": 4, "summary": []}))
            self.enterContext(patch.object(builder, "verify_evaluation", side_effect=lambda *_: json.loads((root / "reports/val_evaluation.json").read_text())))
            write("reports/val_query_metrics.csv", "query_id,method\n" + "".join(f"q001,{m}\n" for m in builder.METHODS))
            write("reports/alignment.csv", "query_id,split,status\nq001,val,aligned\nq002,val,missing_exact_sequence\n")
            memories = [{"memory_id": row["memory_id"], "text": "user: Source, café.\nSecond line.",
                         "turn_index": i, "source_ref": "data/chat_history_32k/fixture.json",
                         "messages": [{"role": "user", "content": "Source, café.\nSecond line."}]}
                        for i, row in enumerate(rows)]
            write("data/processed/memories.jsonl", "".join(json.dumps(m) + "\n" for m in memories))
            with patch.object(builder, "load_scored", return_value=(rows, [query], 0.0)):
                result = builder.build(root, "val", 24)
            script = (root / "web/demo-data.js").read_text(encoding="utf-8")
            bundle = json.loads(script.split("window.PMR_DATA=", 1)[1].removesuffix(";\n"))
            self.assertEqual(len(bundle["cases"][0]["candidates"]), 104)
            self.assertEqual(len(bundle["memories"]), 104)
            self.assertEqual(bundle["alignment"], {"cleaned_queries": 2, "aligned_queries": 1, "excluded_queries": 1})
            self.assertIn("DEVELOPMENT", result["marker"])
            self.assertIn("SUBSET", result["marker"])
            self.assertNotIn("HELD-OUT", result["marker"])
            self.assertEqual(bundle["cases"][0]["gold_memory_ids"], query["gold_memory_ids"])
            self.assertEqual(bundle["memories"]["p1:t001"]["messages"][0]["content"], "Source, café.\nSecond line.")
            self.assertEqual(builder.selection_key(query), builder.selection_key(dict(query, score=100)))
            write("reports/val_evaluation.json", json.dumps({"split": "val", "queries": 2, "persona_limit_per_split": None, "summary": []}))
            with patch.object(builder, "load_scored", return_value=(rows, [query], 0.0)):
                with self.assertRaisesRegex(ValueError, "query count"):
                    builder.build(root, "val", 24)


if __name__ == "__main__":
    unittest.main()

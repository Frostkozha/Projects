import copy
import json
import unittest

from brain.budget import check_budget
from brain.parse import CompletionFailure, parse_completion
from contracts.draft import DraftAnswer
from contracts.json_codec import canonical_json, validate_wire


def fixture():
    return {
        "schema_version": "brain-draft-0.2", "status": "draft",
        "sentences": [{"sentence_id": "s1", "text": "The synthetic sample has one layer.",
            "kind_hint": "factual", "visibility": "student", "cites": ["p1"], "depends_on": []}],
        "used_passage_ids": ["p1"],
    }


class BrainPrimitiveTests(unittest.TestCase):
    def test_exact_context_boundary(self):
        check_budget(2816)
        with self.assertRaises(ValueError): check_budget(2817)

    def test_forward_dependency_rejected(self):
        value = fixture()
        value["sentences"][0]["depends_on"] = ["s2"]
        with self.assertRaises(ValueError):
            validate_wire(DraftAnswer, canonical_json(value), max_bytes=262144)

    def test_duplicate_key_rejected(self):
        with self.assertRaises(ValueError):
            validate_wire(DraftAnswer, b'{"status":"draft","status":"no_evidence"}', max_bytes=262144)

    def test_unshown_citation_rejected(self):
        value = fixture()
        value["sentences"][0]["cites"] = ["p2"]
        value["used_passage_ids"] = ["p2"]
        self.assert_rejected_completion(value)

    def test_length_completion_rejected(self):
        self.assert_rejected_completion(fixture(), finish="length")

    def assert_rejected_completion(self, value, finish="stop"):
        envelope = {"model": "synthetic", "choices": [{"finish_reason": finish,
            "message": {"role": "assistant", "content": json.dumps(value)}}]}
        with self.assertRaises(CompletionFailure):
            parse_completion(canonical_json(envelope), expected_alias="synthetic",
                supplied_ids=("p1",), sentence_policy=lambda draft: None)
        # The no-op sentence policy above is an isolated structural fixture only.


if __name__ == "__main__":
    unittest.main()

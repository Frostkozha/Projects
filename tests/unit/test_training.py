"""Training tests T52-T54 plus calibration reproduction (spec sections 15-16)."""

from __future__ import annotations

import numpy as np
import pytest

from gate_classifier.encoder import FixtureEncoder
from training.calibrate import calibrate_heads
from training.make_synthetic_fixture import generate
from training.split_dataset import merged_groups, split
from training.train import TrainingError, encode_records, targets, train_heads
from training.validate_dataset import validate_dataset

ENABLED = ("lib1", "lib2", "lib5", "lib6")


@pytest.fixture(scope="module")
def records():
    return generate(7)


def test_synthetic_fixture_validates_only_with_flag(records):
    assert validate_dataset(records, ENABLED, allow_fixture=True) == []
    assert validate_dataset(records[:3], ENABLED, allow_fixture=False)  # fixtures refused for real training


def test_T52_no_positive_labels_fails_training(records):
    stripped = [dict(r, risks={**r["risks"], "self_harm_crisis": False}) for r in records]
    X = encode_records(FixtureEncoder(), stripped, "query: ")
    with pytest.raises(TrainingError, match="risk.self_harm_crisis"):
        train_heads(stripped, X, ENABLED, seed=1)


def test_T53_null_library_labels_excluded_from_denominator(records):
    blocked = [r for r in records if any(r["risks"].values())]
    assert blocked and all(all(v is None for v in r["libraries"].values()) for r in blocked)
    tgt = targets(records, ENABLED)
    _, mask, _ = tgt["library.lib1"]
    by_id = {r["id"]: i for i, r in enumerate(records)}
    assert not any(mask[by_id[r["id"]]] for r in blocked)
    assert mask.sum() == sum(1 for r in records if r["libraries"]["lib1"] is not None)
    assert "library.lib3" not in tgt and "library.lib4" not in tgt


def test_T54_group_members_share_one_partition(records):
    parts = split(records, seed=20261009)
    where = {rid: name for name, ids in parts.items() for rid in ids}
    groups = merged_groups(records)
    by_group = {}
    for r in records:
        by_group.setdefault(groups[r["id"]], set()).add(where[r["id"]])
    assert all(len(p) == 1 for p in by_group.values())
    assert sum(len(v) for v in parts.values()) == len(records)


def test_identical_text_in_different_groups_is_merged():
    base = generate(7)[:2]
    a, b = dict(base[0], group_id="g1", text="Explain cartilage."), dict(base[1], group_id="g2", text="explain  CARTILAGE!")
    g = merged_groups([a, b])
    assert g[a["id"]] == g[b["id"]]


def test_calibrated_export_reproduces_sklearn(records):
    parts = split(records, seed=20261009)
    by_id = {r["id"]: r for r in records}
    train = [by_id[i] for i in parts["train"]]
    cal = [by_id[i] for i in parts["calibration"]]
    enc = FixtureEncoder()
    models, _ = train_heads(train, encode_records(enc, train, "query: "), ENABLED, seed=1)
    heads, meta = calibrate_heads(models, cal, encode_records(enc, cal, "query: "), ENABLED)
    for name in ("mode", "topic_scope", "risk.real_person_advice"):
        assert meta[name]["repro_max_abs_err"] <= 1e-6
    p = heads["topic_scope"].predict_proba(encode_records(enc, cal[:5], "query: "))
    assert np.allclose(p.sum(axis=1), 1.0)

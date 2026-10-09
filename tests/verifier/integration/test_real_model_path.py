"""Real code path with a tiny locally generated three-label BERT (no download, no network).

The tiny model is NOT the pinned nli-deberta-v3-xsmall and says nothing about accuracy; it exercises the
real tokenizer, truncation-free limits, raw logits, label mapping, hashes and the supervised subprocess.
Its classifier bias is set so one class dominates, making outcomes deterministic.
"""

from __future__ import annotations

import hashlib
import time

import pytest
import yaml

from contracts.models import ContentReason, OperationalError
from tests.verifier.conftest import PROFILE, ROOT, TEXTS, FakeRegistry, Kit, S, draft, passage, request
from verifier.evidence import VerifierFault
from verifier.nli import NLIUnavailable, TransformersNLI
from verifier.service import InMemoryVerifierAudit, Verifier

LABELS = {"0": "contradiction", "1": "entailment", "2": "neutral"}
REV = "c" * 40


def _build(root, name, bias, id2label=None, num_labels=3):
    transformers = pytest.importorskip("transformers")
    torch = pytest.importorskip("torch")
    d = root / name
    words = ("[PAD] [UNK] [CLS] [SEP] [MASK] simple squamous epithelium has one layer of flattened cells lines "
             "the alveoli lung stratified protects oral cavity goblet secrete mucus").split()
    d.mkdir()
    (d / "vocab.txt").write_text("\n".join(words))
    tok = transformers.BertTokenizerFast(vocab_file=str(d / "vocab.txt"))
    cfg = transformers.BertConfig(vocab_size=len(words), hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                                  intermediate_size=32, max_position_embeddings=512, num_labels=num_labels,
                                  id2label=id2label or {int(k): v for k, v in LABELS.items()},
                                  label2id={v: int(k) for k, v in (id2label or LABELS).items()})
    model = transformers.BertForSequenceClassification(cfg)
    with torch.no_grad():
        model.classifier.weight.zero_()
        model.classifier.bias.copy_(torch.tensor(bias, dtype=torch.float32))
    tok.save_pretrained(d)
    model.save_pretrained(d, safe_serialization=True)
    return d


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    root = tmp_path_factory.mktemp("nli")
    return {
        "entail": _build(root, "entail", [-6.0, 6.0, -6.0]),
        "contra": _build(root, "contra", [6.0, -6.0, -6.0]),
        "swapped": _build(root, "swapped", [0.0, 0.0, 0.0], id2label={0: "entailment", 1: "contradiction",
                                                                          2: "neutral"}),
        "two": _build(root, "two", [0.0, 0.0], num_labels=2, id2label={0: "contradiction", 1: "entailment"}),
    }


def _sha(d):
    return hashlib.sha256((d / "model.safetensors").read_bytes()).hexdigest()


def _profile(tmp_path, model_dir, worker="in_process", **nli):
    data = yaml.safe_load(PROFILE.read_text())
    data.update(operating_mode="real_model_development", thresholds_path=str(ROOT / data["thresholds_path"]),
                rules_dir=str(ROOT / data["rules_dir"]))
    data["nli"].update(local_path=str(model_dir), revision=REV, weights_sha256=_sha(model_dir), **nli)
    data["runtime"]["worker"] = worker
    path = tmp_path / f"profile-{model_dir.name}-{worker}.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def _kit(verifier) -> Kit:
    kit = Kit.__new__(Kit)
    kit.verifier, kit.audit, kit.registry = verifier, verifier.audit, FakeRegistry()
    kit.passages = [passage(k, v) for k, v in TEXTS.items()]
    kit.conflicts = ()
    return kit


def test_real_path_supported_and_contradicted(tmp_path, models):
    v = Verifier.from_profile(_profile(tmp_path, models["entail"]), audit=InMemoryVerifierAudit())
    r = v.readiness()
    assert r["ready"] and not r["student_release_ready"] and r["verifier_version"].startswith("verifier-0.2+")
    kit = _kit(v)
    res, _, _ = kit.run(draft(S("s1", TEXTS["p1"], ["p1"])))
    assert res.result.status == "approved" and res.result.sentence_results[0].nli_scores["p1"].entailment > 0.99
    bad = Verifier.from_profile(_profile(tmp_path, models["contra"]), audit=InMemoryVerifierAudit())
    res2, _, _ = _kit(bad).run(draft(S("s1", TEXTS["p1"], ["p1"])))
    assert ContentReason.CONTRADICTED in res2.result.sentence_results[0].reason_codes


def test_T32_real_label_mapping_and_architecture_checks(models):
    for name in ("swapped", "two"):
        with pytest.raises(NLIUnavailable):
            TransformersNLI(str(models[name]), "tiny", REV, LABELS, 512)
    with pytest.raises(NLIUnavailable):
        TransformersNLI(str(models["entail"]), "tiny", REV, LABELS, 512, weights_sha256="0" * 64)
    with pytest.raises(NLIUnavailable):
        TransformersNLI(str(models["entail"]), "tiny", "short", LABELS, 512)
    with pytest.raises(NLIUnavailable):
        TransformersNLI(str(models["entail"]), "tiny", REV, LABELS, 1024)  # pair ceiling beyond positions


def test_T39_T40_real_tokenizer_limits_without_truncation(tmp_path, models):
    v = Verifier.from_profile(_profile(tmp_path, models["entail"]), audit=InMemoryVerifierAudit())
    backend = v.worker.backend
    calls = {"n": 0}
    original = backend.logits

    def spy(pairs):
        calls["n"] += 1
        return original(pairs)

    backend.logits = spy
    long_premise = " ".join(["Simple squamous epithelium lines the alveoli."] * 70)  # > 384 verifier tokens
    assert backend.count(long_premise) > 384
    kit = _kit(v)
    kit.passages = [passage("p1", long_premise)]
    res, _, _ = kit.run(draft(S("s1", "Simple squamous epithelium lines the alveoli.", ["p1"])))
    assert res.result.error_code == OperationalError.EVIDENCE_LIMIT and calls["n"] == 0
    kit.passages = [passage(k, t) for k, t in TEXTS.items()]
    long_h = "Simple squamous epithelium " + "a " * 150 + "lines the alveoli."  # < 600 chars, > 96 tokens
    res2, _, _ = kit.run(draft(S("s1", long_h, ["p1"])))
    assert ContentReason.INVALID_DRAFT in res2.answer_reasons and calls["n"] == 0


def test_T54_T55_subprocess_worker_timeout_restart_and_late_discard(tmp_path, models):
    v = Verifier.from_profile(_profile(tmp_path, models["entail"], worker="subprocess"), audit=InMemoryVerifierAudit())
    try:
        worker = v.worker
        assert v.readiness()["ready"] and worker.healthy
        out = worker.infer([(TEXTS["p1"], TEXTS["p1"])], time.monotonic() + 30)
        assert out.shape == (1, 3)
        with pytest.raises(VerifierFault) as exc:
            worker.infer([(TEXTS["p1"], TEXTS["p1"])] * 8, time.monotonic() + 0.0005)
        assert exc.value.code == OperationalError.DEADLINE_EXCEEDED and worker.restarts == 1
        out2 = worker.infer([(TEXTS["p2"], TEXTS["p2"])], time.monotonic() + 30)  # never the stale attempt
        assert out2.shape == (1, 3)
        kit = _kit(v)
        res, _, _ = kit.run(draft(S("s1", TEXTS["p1"], ["p1"])))
        assert res.result.status == "approved"
    finally:
        v.worker.close()


def test_request_helper_is_canonical():
    assert request(draft(S("s1", TEXTS["p1"], ["p1"]))).schema_version == "verify-request-0.2"

"""Readiness and bundle-loading tests T22, T50, T55, T60 (spec sections 13, 18)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from gate_classifier.config import load_config, load_registry
from gate_classifier.encoder import FixtureEncoder, FixtureTokenizer, Preprocessing
from gate_classifier.heads import BundleError, LinearHead, head_specs, load_bundle
from gate_classifier.schema import ErrorCode, GateRequest, Route, new_request_id
from gate_classifier.service import GateService
from tests.conftest import ROOT
from training.bundle import write_bundle

REV = "a" * 40


class FakeRealEncoder:
    """Stands in for a pinned local encoder (not a fixture) so readiness logic can be exercised offline."""

    is_fixture = False

    def __init__(self, dim=384):
        self.preprocessing = Preprocessing("intfloat/e5-small-v2", REV, REV, "mean", "query: ", "l2", dim)
        self.tokenizer = FixtureTokenizer()
        self._inner = FixtureEncoder(dim)

    def encode(self, texts):
        return self._inner.encode(texts)


def random_heads(dim=384, seed=0):
    rng = np.random.default_rng(seed)
    heads = {}
    for name, (kind, classes) in head_specs().items():
        if name in ("library.lib3", "library.lib4"):
            continue
        k = len(classes) if kind == "multiclass" else 1
        heads[name] = LinearHead(name, kind, classes, rng.normal(size=(k, dim)) * 0.01, np.zeros(k),
                                 -np.ones(k), np.zeros(k))
    return heads


def dev_config(**updates):
    cfg = load_config(ROOT / "config/development.yaml")
    enc = cfg.encoder.model_copy(update={"revision": REV, "tokenizer_revision": REV})
    return cfg.model_copy(update={"operating_mode": "development", "encoder": enc, **updates})


def build(tmp_path, cfg, encoder, mode="development", **kw):
    return write_bundle(tmp_path / "bundle", random_heads(), cfg=cfg, encoder=encoder, operating_mode=mode,
                        dataset={"dataset_sha256": "b" * 64, "split_sha256": "c" * 64,
                                 "calibration_data_version": "cal-v1"}, bundle_version="dev-bundle-1", **kw)


def load(path, cfg, encoder, mode=None):
    return load_bundle(path, operating_mode=mode or cfg.operating_mode, config_sha256=cfg.config_sha256(),
                       preprocessing_fingerprint=encoder.preprocessing.fingerprint(), dimension=cfg.encoder.dimension)


def service(cfg, encoder, bundle):
    return GateService(cfg, load_registry(ROOT / "config/library_registry.yaml"), tokenizer=encoder.tokenizer,
                       encoder=encoder, bundle=bundle)


def test_valid_development_bundle_is_ready_and_routes(tmp_path):
    cfg, enc = dev_config(), FakeRealEncoder()
    svc = service(cfg, enc, load(build(tmp_path, cfg, enc), cfg, enc))
    assert svc.readiness().ready and svc.readiness().status == "ready"
    from tests.conftest import Harness
    ctx = Harness().ctx()
    d = svc.evaluate(GateRequest(text="Explain simple squamous epithelium."), ctx, new_request_id()).decision
    assert d.route != Route.unavailable and d.versions.bundle == "dev-bundle-1"


def test_T22_missing_bundle_calibrator_or_checksum(tmp_path):
    cfg, enc = dev_config(), FakeRealEncoder()
    svc = service(cfg, enc, None)
    assert not svc.readiness().ready and "bundle_missing" in svc.readiness().reasons
    from tests.conftest import Harness
    d = svc.evaluate(GateRequest(text="Explain epithelium."), Harness().ctx(), new_request_id()).decision
    assert d.route == Route.unavailable

    with pytest.raises(BundleError, match="manifest_missing"):
        load(tmp_path / "nothing", cfg, enc)

    path = build(tmp_path, cfg, enc)
    manifest = json.loads((path / "manifest.json").read_text())
    manifest["heads"]["risk.self_harm_crisis"]["calibration"] = None
    (path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(BundleError, match="calibration_missing"):
        load(path, cfg, enc)

    path = build(tmp_path, cfg, enc)
    with open(path / "mode.npz", "ab") as fh:
        fh.write(b"tamper")
    with pytest.raises(BundleError, match="checksum_mismatch"):
        load(path, cfg, enc)

    path = build(tmp_path, cfg, enc)
    manifest = json.loads((path / "manifest.json").read_text())
    del manifest["heads"]["risk.imminent_emergency"]
    (path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(BundleError, match="mandatory_head_missing"):
        load(path, cfg, enc)

    with pytest.raises(BundleError, match="dimension_mismatch"):
        load_bundle(build(tmp_path, cfg, enc), operating_mode="development", config_sha256=cfg.config_sha256(),
                    preprocessing_fingerprint=enc.preprocessing.fingerprint(), dimension=768)


def test_T22_readiness_endpoint_and_gate_503(tmp_path):
    from fastapi.testclient import TestClient

    from gate_classifier.api import create_app

    cfg, enc = dev_config(), FakeRealEncoder()
    svc = service(cfg, enc, None)
    app = create_app(svc, lambda *a: None, ["tok"])
    c = TestClient(app)
    assert c.get("/health/ready").status_code == 503
    assert c.get("/health/live").status_code == 200


def test_T50_fixture_bundle_refused_outside_fixture_mode(tmp_path):
    cfg = dev_config()
    fixture_enc = FixtureEncoder()
    path = build(tmp_path, cfg, fixture_enc, mode="fixture")
    with pytest.raises(BundleError, match="fixture_bundle_not_allowed"):
        load(path, cfg, fixture_enc)
    svc = service(cfg, fixture_enc, None)
    assert "real_encoder_missing" in svc.readiness().reasons
    # fixture scorer in development mode is also refused
    svc2 = GateService(cfg, load_registry(ROOT / "config/library_registry.yaml"), tokenizer=FixtureTokenizer(),
                       fixture_scorer=lambda t: {})
    assert "fixture_scorer_not_allowed" in svc2.readiness().reasons


def test_T50_example_only_values_refused(tmp_path):
    cfg, enc = dev_config(), FakeRealEncoder()
    path = build(tmp_path, cfg, enc)
    manifest = json.loads((path / "manifest.json").read_text())
    manifest["bundle_version"] = "example-only"
    (path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(BundleError, match="example_value_in_real_bundle"):
        load(path, cfg, enc)


def test_T55_config_change_without_bundle_update(tmp_path):
    cfg, enc = dev_config(), FakeRealEncoder()
    path = build(tmp_path, cfg, enc)
    th = cfg.thresholds.model_copy(update={"topic": 0.65})
    changed = cfg.model_copy(update={"thresholds": th})
    assert changed.config_sha256() != cfg.config_sha256()
    with pytest.raises(BundleError, match="config_checksum_mismatch"):
        load(path, changed, enc)
    svc = service(changed, enc, load(path, cfg, enc))
    assert not svc.readiness().ready and "config_checksum_mismatch" in svc.readiness().reasons


def test_T60_placeholder_contacts_block_production(tmp_path):
    cfg = dev_config(operating_mode="production")
    assert "placeholder_contacts" in cfg.production_problems()
    enc = FakeRealEncoder()
    svc = service(cfg, enc, None)
    reasons = svc.readiness().reasons
    assert "placeholder_contacts" in reasons and not svc.readiness().ready
    good = cfg.model_copy(update={"contacts": {"approved_emergency_contact": "Campus Security +7 7172 00 00 00",
                                               "approved_student_support_contact": "Student Wellbeing Office"}})
    assert "placeholder_contacts" not in good.production_problems()


def test_production_requires_accepted_bundle(tmp_path):
    cfg, enc = dev_config(operating_mode="production"), FakeRealEncoder()
    path = build(tmp_path, cfg, enc, mode="production")
    with pytest.raises(BundleError, match="production_acceptance_not_passed"):
        load(path, cfg, enc)


def test_api_error_codes_for_unavailable(tmp_path):
    cfg, enc = dev_config(), FakeRealEncoder()
    svc = service(cfg, enc, None)
    from fastapi.testclient import TestClient

    from gate_classifier.api import create_app
    from tests.conftest import Harness

    h = Harness()
    app = create_app(svc, lambda o, t, c, s: h.ctx(), ["tok"])
    r = TestClient(app).post("/v1/gate", json={"text": "Explain epithelium"},
                             headers={"Authorization": "Bearer tok", "x-gate-owner": "owner-a",
                                      "x-gate-tenant": "dev-tenant", "x-gate-course": "histology-dev"})
    assert r.status_code == 503 and r.json()["error_code"] == ErrorCode.SERVICE_UNAVAILABLE.value
    assert set(r.json()) == {"error_code", "request_id"}

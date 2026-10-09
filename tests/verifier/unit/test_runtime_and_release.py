"""T54-T57, T60: deadlines, queue bounds, readiness, audit outage and evaluation/release integrity."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest
import yaml

from contracts.models import OperationalError, ResponseCode
from evaluation.metrics import group_bootstrap, wilson_interval, zero_miss_upper_bound
from evaluation.splits import grouped_split, leaks
from tests.verifier.conftest import PROFILE, ROOT, SUPPORT, TEXTS, Kit, S, draft
from verifier.config import ThresholdProfile, load_thresholds
from verifier.evidence import VerifierFault
from verifier.nli import FixtureNLI, NLIUnavailable
from verifier.service import InMemoryVerifierAudit, Verifier
from verifier.worker import InProcessWorker

PAIR = [("premise.", "hypothesis.")]


class Blocking:
    """Backend that blocks until released (simulates in-flight CPU inference)."""

    is_fixture = True
    version = "blocking"
    label_index = {"contradiction": 0, "entailment": 1, "neutral": 2}

    def __init__(self, delay: float = 0.0):
        self.release = threading.Event()
        self.started = threading.Event()
        self.delay = delay

    def logits(self, pairs):
        self.started.set()
        if self.delay:
            time.sleep(self.delay)
        else:
            self.release.wait(5)
        return np.zeros((len(pairs), 3))


def test_T54_deadline_exceeded_discards_late_result():
    worker = InProcessWorker(Blocking(delay=0.3), capacity=16)
    with pytest.raises(VerifierFault) as exc:
        worker.infer(PAIR, time.monotonic() + 0.1)
    assert exc.value.code == OperationalError.DEADLINE_EXCEEDED
    assert worker.infer(PAIR, time.monotonic() + 2.0).shape == (1, 3)  # worker still bounded and serving


def test_T54_verifier_timeout_is_error_not_A5(kit):
    slow = {"n": 0}

    def score(p, h):
        slow["n"] += 1
        time.sleep(0.3)
        return SUPPORT

    kit.backend.score_fn = score
    v, _, _ = kit.run(draft(S("s1", TEXTS["p1"], ["p1"])), deadline=time.monotonic() + 0.1)
    r = v.result
    assert (r.status, r.error_code, r.response_code) == ("error", OperationalError.DEADLINE_EXCEEDED, None)
    assert r.verified_text == "" and r.reply_key == "service_unavailable"


def test_T55_full_queue_rejected():
    backend = Blocking()
    worker = InProcessWorker(backend, capacity=1)
    errors, results = [], []

    def call():
        try:
            results.append(worker.infer(PAIR, time.monotonic() + 5))
        except VerifierFault as exc:
            errors.append(exc.code)

    active = threading.Thread(target=call)
    active.start()
    backend.started.wait(2)
    waiting = threading.Thread(target=call)
    waiting.start()
    time.sleep(0.05)
    with pytest.raises(VerifierFault) as exc:
        worker.infer(PAIR, time.monotonic() + 5)
    assert exc.value.code == OperationalError.QUEUE_FULL
    backend.release.set()
    active.join(5)
    waiting.join(5)
    assert len(results) == 2 and errors == []


def _profile(tmp_path, **updates):
    data = yaml.safe_load(PROFILE.read_text())
    data["thresholds_path"] = str(ROOT / data["thresholds_path"])
    data["rules_dir"] = str(ROOT / data["rules_dir"])
    for k, v in updates.items():
        if isinstance(v, dict):
            data[k] = {**data.get(k, {}), **v}
        else:
            data[k] = v
    path = tmp_path / "profile.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def test_T56_missing_artifacts_fail_startup_offline(tmp_path):
    real = _profile(tmp_path, operating_mode="real_model_development",
                    nli={"local_path": str(tmp_path / "absent"), "revision": "a" * 40})
    with pytest.raises(NLIUnavailable):
        Verifier.from_profile(real, audit=InMemoryVerifierAudit())
    unpinned = _profile(tmp_path, operating_mode="real_model_development", nli={"revision": None})
    with pytest.raises(NLIUnavailable):
        Verifier.from_profile(unpinned, audit=InMemoryVerifierAudit())
    with pytest.raises(NLIUnavailable):  # a fixture can never stand in for the real model
        Verifier.from_profile(real, audit=InMemoryVerifierAudit(), fixture_backend=FixtureNLI(lambda p, h: SUPPORT))
    with pytest.raises(NLIUnavailable):
        Verifier.from_profile(PROFILE, audit=InMemoryVerifierAudit())  # fixture mode without injected scores


def test_T56_fixture_never_production_ready():
    v = Verifier.from_profile(PROFILE, audit=InMemoryVerifierAudit(), fixture_backend=FixtureNLI(lambda p, h: SUPPORT))
    v.profile = v.profile.model_copy(update={"operating_mode": "production"})
    r = v.readiness()
    assert r["ready"] is False and r["student_release_ready"] is False
    assert r["checks"]["backend_matches_mode"] is False and r["checks"]["thresholds_release_ready"] is False
    missing_audit = Verifier.from_profile(PROFILE, audit=None, fixture_backend=FixtureNLI(lambda p, h: SUPPORT))
    assert missing_audit.readiness()["checks"]["audit_adapter"] is False


def test_T57_audit_outage_blocks_delivery_but_keeps_A7(make_kit):
    kit = make_kit(audit=InMemoryVerifierAudit(fail=True))
    alarms = []
    kit.verifier.alarm = lambda code, rid: alarms.append(code)
    v, _, _ = kit.run(draft(S("s1", TEXTS["p1"], ["p1"])))
    assert v.result.error_code == OperationalError.AUDIT_UNAVAILABLE and v.result.verified_text == ""
    v2, _, _ = kit.run(draft(S("s1", TEXTS["p1"], ["p1"])),
                       redacted_request="Someone just collapsed and is not breathing!")
    assert v2.result.response_code == ResponseCode.A7 and "verifier_audit_failed" in alarms


def test_audit_records_hold_no_text(kit):
    secret = "ZETA-UNIQUE-REQUEST"
    kit.run(draft(S("s1", TEXTS["p1"], ["p1"])), redacted_request=f"Explain epithelium {secret}.")
    blob = str(kit.audit.records)
    assert secret not in blob and TEXTS["p1"] not in blob and "versions" in kit.audit.records[-1]


# ----------------------------------------------------------------------------- T60


def test_T60_fixtures_cannot_set_release_ready():
    with pytest.raises(ValueError):
        ThresholdProfile(profile_id="x", Tc=0.3, T_lo=0.5, T_hi=0.9, release_ready=True)
    assert load_thresholds(ROOT / "config/verifier/thresholds_test.yaml").release_ready is False
    for bad in ({"Tc": 0.0}, {"T_lo": 0.9, "T_hi": 0.5}, {"Tc": float("nan")}):
        with pytest.raises(ValueError):
            ThresholdProfile(**{"profile_id": "x", "Tc": 0.3, "T_lo": 0.5, "T_hi": 0.9, **bad})


def test_T60_grouped_split_never_leaks():
    cases = []
    for q in range(40):
        for m in range(3):  # mutations and paraphrases stay with their parent
            cases.append({"case_id": f"q{q}-m{m}", "question_family": f"q{q}", "source_family": f"sec{q % 13}",
                          "text": f"Question family {q} variant {m} about section {q % 13}."})
    dev, final = grouped_split(cases, final_fraction=0.3)
    assert dev and final and len(dev) + len(final) == len(cases)
    assert {c["question_family"] for c in dev}.isdisjoint({c["question_family"] for c in final})
    assert {c["source_family"] for c in dev}.isdisjoint({c["source_family"] for c in final})
    assert [x for x in leaks(dev, final) if x[0] == "group"] == []
    planted = final + [{**dev[0], "case_id": "dup"}]
    assert any(k == "exact" for k, *_ in leaks(dev, planted))


def test_metrics_match_spec_worked_examples():
    lo, hi = wilson_interval(190, 200)
    assert round(lo, 3) == 0.910 and round(hi, 3) == 0.973
    assert round(zero_miss_upper_bound(100), 4) == 0.0295
    lo2, hi2 = group_bootstrap({f"g{i}": [1, 1, 0] if i % 4 == 0 else [1, 1, 1] for i in range(40)})
    assert 0.8 < lo2 <= hi2 <= 1.0


def test_kit_runs_fixture_mode_only():
    assert Kit().verifier.profile.operating_mode == "fixture"

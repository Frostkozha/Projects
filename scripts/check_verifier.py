"""Real-model check for the verifier's three-class NLI model (offline, pinned local files).

Loads cross-encoder/nli-deberta-v3-xsmall exactly as the verifier does (supervised subprocess worker),
checks files, label mapping and readiness, prints the weight hash to pin, scores three polarity pairs and
runs one complete synthetic verification. Scores are model outputs, not probabilities of correctness.
Usage: python scripts/check_verifier.py [--profile config/verifier_real_dev.yaml]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

NEEDED = ("config.json", "model.safetensors", "tokenizer_config.json")
PREMISE = "Simple squamous epithelium has one layer of flattened cells and lines the alveoli of the lung."
PAIRS = [
    ("entailment", "Simple squamous epithelium lines the alveoli of the lung."),
    ("contradiction", "Simple squamous epithelium has many layers of cells."),
    ("neutral", "Goblet cells secrete mucus in the trachea."),
]


def main(argv=None) -> int:
    from verifier.config import load_profile  # noqa: PLC0415
    from verifier.nli import NLIUnavailable, to_scores  # noqa: PLC0415
    from verifier.service import InMemoryVerifierAudit, Verifier  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=str(ROOT / "config/verifier_real_dev.yaml"))
    args = ap.parse_args(argv)
    profile = load_profile(args.profile)
    nli = profile.nli
    path = profile.path(nli.local_path or "")
    print(f"model_id:  {nli.model_id}")
    print(f"revision:  {nli.revision}")
    print(f"folder:    {path}")
    ok = True
    for name in NEEDED:
        if not (path / name).is_file():
            print(f"MISSING    {name}")
            ok = False
    if not any((path / n).is_file() for n in ("tokenizer.json", "spm.model")):
        print("MISSING    tokenizer.json (or spm.model)")
        ok = False
    if not ok:
        return 1
    digest = hashlib.sha256((path / "model.safetensors").read_bytes()).hexdigest()
    print(f"weights:   sha256 {digest}")
    if not nli.weights_sha256:
        print("NOTE       set nli.weights_sha256 to the value above to pin these exact weights")
    if not nli.revision or len(nli.revision) != 40:
        print("FAIL       set nli.revision to the exact 40-character commit you downloaded (never guess it)")
        return 1
    try:
        verifier = Verifier.from_profile(args.profile, audit=InMemoryVerifierAudit())
    except NLIUnavailable as exc:
        print(f"FAIL       model did not load: {exc}")
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL       verifier did not start: {type(exc).__name__}: {exc}")
        return 1
    try:
        ready = verifier.readiness()
        for name, passed in ready["checks"].items():
            print(f"{'ok' if passed else 'FAILED':<7}readiness: {name}")
        start = time.monotonic()
        logits = verifier.worker.infer([(PREMISE, h) for _, h in PAIRS], time.monotonic() + 60)
        took = time.monotonic() - start
        scores = to_scores(logits, verifier.label_index, len(PAIRS))
        checks = {}
        for (expected, hyp), sc in zip(PAIRS, scores):
            top = max(sc, key=sc.get)
            print(f"C {sc['contradiction']:.3f}  E {sc['entailment']:.3f}  N {sc['neutral']:.3f}  "
                  f"expected {expected:<13} | {hyp}")
            if expected == "neutral":
                # small NLI models often call an unrelated topic "contradiction"; both reject the sentence.
                # What matters is that an unsupported sentence is never entailed.
                checks["unrelated pair not entailed"] = top != "entailment" and sc["entailment"] < 0.5
            else:
                checks[f"{expected} pair ranked {expected}"] = top == expected
        print(f"batch of {len(PAIRS)} pairs: {took * 1000:.0f} ms (warm process, CPU)")
        result = _sample_verification(verifier)
        print(f"sample verification: status={result.status} code={getattr(result.response_code, 'value', None)} "
              f"error={getattr(result.error_code, 'value', None)}")
        checks["readiness"] = ready["ready"]
        checks["sample sentence approved as A2"] = result.status == "approved" and result.response_code.value == "A2"
        for name, passed in checks.items():
            print(f"{'ok' if passed else 'FAILED':<7}{name}")
        if all(checks.values()):
            print("PASS       verifier NLI loads offline and behaves as expected (development profile, unevaluated)")
            return 0
        print("FAIL       one or more checks failed")
        return 1
    finally:
        close = getattr(verifier.worker, "close", None)
        if close:
            close()


def _sample_verification(verifier):
    from contracts.models import DraftAnswer, VerifyRequest  # noqa: PLC0415
    from retriever.schema import EvidencePassage, Reason, RetrievalResult, Status  # noqa: PLC0415
    from verifier.evidence import EvidenceBundle  # noqa: PLC0415
    from verifier.service import VerifyContext  # noqa: PLC0415

    class AllLive:
        def eligible(self, passages, kb_version):
            return {p.passage_id: True for p in passages}

        def epoch(self):
            return 0

    rid = str(uuid.uuid4())
    text = PREMISE
    p = EvidencePassage(passage_id="check-p1", source_id="check-src", source_version="v1", title="Synthetic check",
                        library_id="lib1", locator={"kind": "section", "start": None, "end": None,
                                                    "label": "Check", "anchor": "check"},
                        section_path=("Check",), edition=None, publication_year=None, text=text,
                        text_sha256=hashlib.sha256(text.encode()).hexdigest(), review_status="live",
                        rights_reference="synthetic", relevance_score=None, score_type=None,
                        evidence_uri="/v1/evidence/check-p1")
    r = RetrievalResult(request_id=rid, status=Status.ok, reason=Reason.QUALIFYING_PASSAGES, error_code=None,
                        passages=(p,), coverage="unknown", conflicts=(), libraries_searched=("lib1",),
                        kb_version="check-kb", index_version="check", profile_version="check", revocation_epoch=0)
    d = DraftAnswer.model_validate({"schema_version": "brain-draft-0.2", "status": "draft", "sentences": [
        {"sentence_id": "s1", "text": PAIRS[0][1], "kind_hint": "factual", "visibility": "student",
         "cites": ["check-p1"], "depends_on": []}], "used_passage_ids": ["check-p1"]})
    req = VerifyRequest(schema_version="verify-request-0.2", request_id=rid, draft=d, prompt_version="check")
    ctx = VerifyContext(request_id=rid, route="retrieve", tenant_id="t", course_id="c",
                        redacted_request="Where is simple squamous epithelium found?", mode="answer",
                        evidence=EvidenceBundle(rid, "t", "c", r, ("check-p1",)), registry=AllLive(),
                        deadline=time.monotonic() + 30)
    return verifier.verify(req, ctx)


if __name__ == "__main__":  # required on Windows: the model worker is a spawn subprocess
    sys.exit(main())

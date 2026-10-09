# Verifier v0.2 (claim verification)

Implements `Verifier_Technical_Plan_English_v0_2`. The verifier checks every sentence of a Brain draft
against the exact passages the Brain was shown, before anything reaches a student.

**Status: development.** The NLI model has no medical-domain evaluation here, the thresholds are a
test-only profile (`release_ready: false`), and the cue, alias and allowlist files are unreviewed
candidates. Passing the software tests is not an accuracy result.

## Pipeline

`Verifier.verify(VerifyRequest, VerifyContext) -> VerificationResult` (`verifier/service.py`):

| Stage | Module | On failure |
|---|---|---|
| readiness, trusted context, pilot mode | `service.py` | error (`MODEL_UNAVAILABLE`, `PROFILE_MISMATCH`, `CONTEXT_MISMATCH`) |
| 0a format and segmentation (spaCy) | `format.py` | whole draft A5 `INVALID_DRAFT` |
| 0b output policy (request + draft) | `policy.py` | A7 (current request only), A6, A5 `OUTPUT_POLICY_UNCERTAIN`, error `POLICY_UNAVAILABLE` |
| 1 evidence and citations | `evidence.py` | error on hash, scope or registry fault; sentence rejected for missing, unknown or ineligible cites |
| NLI token limits (no truncation) | `service.py` | `EVIDENCE_LIMIT` error (premise), A5 `INVALID_DRAFT` (sentence) |
| 2 hard facts | `hard_facts.py` | sentence `HARD_FACT_MISMATCH` / `HARD_FACT_UNRESOLVED` |
| 3 three-class NLI (bounded worker) | `nli.py`, `worker.py` | sentence `CONTRADICTED`, `NOT_ENTAILED`, `UNCERTAIN_SUPPORT` |
| 5 answer level | `decision.py` | A5 for 2+ failures, unsafe trim, failed hidden answer, no visible support, incomplete conflict |
| deadline, audit commit | `service.py` | `DEADLINE_EXCEEDED`, `AUDIT_UNAVAILABLE` (a fixed A7 stays deliverable) |

Pair rule (premise = passage, hypothesis = sentence): contradiction >= Tc wins; else entailment >= T_hi
supports; else entailment < T_lo is unsupported; the band between (including exactly T_lo) is uncertain
and rejected. Scores are never averaged across passages; one cited passage must support the whole
sentence on its own.

Answer outcomes: A1 only with a trusted full-coverage signal (approved item evidence sets); A2 for
unknown coverage, one safely trimmed final sentence, or a fully supported known conflict; A4 only with
gate authorization **and** the `a4_educational` feature. Notices are code-owned (`config/verifier/notices.json`).

Delivery (`verifier/formatter.py`): the formatter rebuilds the student text from the original sentence
objects, renders citations from passage metadata, escapes HTML, and appends notices. `DeliveryAuthorizer`
rechecks binding digests and live eligibility under one lock and records the payload digest; the UI
accepts only that payload.

## Provision the NLI model (Windows, one time, needs internet)

From the project folder in PowerShell (Python 3.11 venv active):

```powershell
py -3.11 -m pip install huggingface_hub
py -3.11 -c "from huggingface_hub import HfApi, snapshot_download; rid='cross-encoder/nli-deberta-v3-xsmall'; sha=HfApi().model_info(rid).sha; print('COMMIT:', sha); snapshot_download(rid, revision=sha, local_dir='artifacts/verifier/nli-deberta-v3-xsmall', allow_patterns=['config.json','model.safetensors','tokenizer.json','tokenizer_config.json','spm.model','special_tokens_map.json','added_tokens.json'])"
```

Then edit `config/verifier_real_dev.yaml`:

1. `nli.revision`: the exact 40-character `COMMIT` printed above (never shorten or guess it).
2. Run `python scripts/check_verifier.py`. It prints `weights: sha256 ...`; copy that into
   `nli.weights_sha256` and run the check again. It should end with `PASS`.

If `model.safetensors` is missing from that revision, the check stops with `MISSING model.safetensors`:
the verifier loads safetensors only. If `tokenizer.json` is missing, the DeBERTa tokenizer is built
from `spm.model` and needs `pip install sentencepiece`.

The runtime never downloads anything. `artifacts/` is gitignored.

## Migration note (Brain draft contract)

The free-text `draft_text` draft is gone. The Brain now returns `contracts.models.DraftAnswer`
(`brain-draft-0.2`): up to 16 sentence objects, each with one sentence, its own cited passage IDs,
visibility (`student` or hidden `internal`) and earlier-sentence dependencies. `used_passage_ids`
must equal the sorted union of cites.

The orchestrator (`gate_classifier/orchestrator.py`) builds the trusted `EvidenceBundle` from the passages
actually placed in the prompt, calls the verifier, maps rejected results to fixed replies (A5/A6, A7 with
the alert dispatcher), and delivers only the authorized compiled payload. The old
`final_code_from_verification`, prompt display numbers and response-wide citation check are removed.
Supported free-text search answers are now A2 (coverage unknown); approved quiz item evidence gives A1.

## Tests

- `tests/verifier/unit/`: T01-T60 with injected fixture scores (never production-ready).
- `tests/verifier/integration/`: delivery races and tampering, the internal HTTP wrapper, and the real
  model code path using a tiny locally built three-label BERT (tokenizer limits without truncation,
  label/architecture/hash checks, supervised subprocess timeout and restart).
- `evaluation/`: grouped split with leak detection, Wilson and zero-miss bounds, group bootstrap.

## Manifest of rule files

`config/verifier/`: `allowlist.json`, `units.json`, `antonyms.json`, `cues.json`, `notices.json`
(each carries `review.status`; all are `development`), and `thresholds_test.yaml`. Their SHA-256
digests enter `verifier_version` together with the model, tokenizer, parser, hard-fact rules,
thresholds and policy version.

## Known gaps

- Real `nli-deberta-v3-xsmall` is pinned at a150876415327c80daeff35ca6f68f5ed8cf5c24 and passed
  `scripts/check_verifier.py` on Windows (CPU, about 60 ms for 3 pairs). It labels an unrelated topic as
  contradiction rather than neutral; both reject, so this is safe but shows the scores are not calibrated.
- Hard facts are deterministic clause patterns, not a dependency-parse relation extractor; uncovered
  prose goes to NLI with the limitation recorded.
- Segmentation uses the rule-based spaCy sentencizer in development; production needs a pinned
  `en_core_web_sm` path (`segmenter.kind: spacy_model`).
- No coverage adapter beyond approved item evidence sets, so free-text answers are never A1.
- No faculty-labeled data, approved thresholds, reviewed cue/alias lists or release record:
  `student_release_ready` stays false.

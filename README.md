# Medical AI Education Project, Module A: Gate Classifier and Retriever v0.2 (English-only)

Two packages for the histology study tutor:

- `gate_classifier/`: the text gate (`Gate Classifier English-Only Implementation Specification v0.2`).
- `retriever/`: permitted-evidence retrieval (`Retriever English-Only Implementation Specification v0.2`).
  See [Retriever](#retriever-v02) below.

**Status: development package.** No trained weights, reviewed dataset, measured accuracy/latency or
production approval are included. Completing this code is not approval to collect student data or
deploy a tutor.

## Setup (Python 3.11)

```bash
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r requirements.lock && pip install -e . --no-deps
pytest
```

## Operating modes

| Mode | Scores come from | Readiness |
|---|---|---|
| `fixture` | deterministic test scorer or fixture bundle | `fixture_only` (never production) |
| `development` | pinned local E5 encoder + calibrated bundle | `ready` only when every artifact validates |
| `production` | as development + real contacts, retention, alert adapter, accepted bundle | `ready` only when all checks pass |

Missing/corrupt weights, calibrators, checksum or config mismatch → readiness false and every
request returns `SERVICE_UNAVAILABLE`. There is no random-score or "search everything" fallback.

## Layout

- `gate_classifier/` – schema, config, normalize, privacy, policy_rules, encoder, heads, session,
  decide (pure routing), service, audit, api (FastAPI internal), adapters, orchestrator (harness + `Fixture*` stubs)
- `training/` – validate_dataset, split_dataset, train, calibrate, tune_thresholds, evaluate,
  make_synthetic_fixture (smoke tests only)
- `config/` – development config and histology library registry (lib3/lib4 disabled)
- `tests/` – contract/integration acceptance tests T01–T60, API contract and training tests (fixture mode)

## Bootstrap (offline runtime)

Model download is a separate, explicit step; runtime uses `local_files_only`. Pin exact revisions in
`config/*.yaml` (`encoder.revision`, `tokenizer_revision`) and install a pinned `en_core_web_sm`
package. Never run `spacy download` inside the service.

## Training pipeline

```bash
python -m training.validate_dataset data.jsonl
python -m training.split_dataset data.jsonl --seed 20261009 --out split.json
python -m training.train data.jsonl --split split.json --config config/development.yaml \
  --registry config/library_registry.yaml --course histology-dev --encoder e5 --out artifacts/bundles/candidate
python -m training.tune_thresholds ...   # tuning partition only
python -m training.evaluate ... --partition holdout --confirm-frozen
```

## Known gaps

- Real E5 encoder verified offline on Windows (Python 3.11, `scripts/check_real_model.py` PASS, pinned revision ffb93f3). The spaCy name detector is still untested.
- Inference runs in a thread worker; production should use a killable process.
- Institutional items: contacts, welfare procedure, retention, SSO, approved source registry, reviewed dataset.

## Retriever v0.2

Finds approved, permitted course evidence for a gate-approved request: filtered exact dense search
(E5-small-v2) + SQLite FTS5 BM25, reciprocal rank fusion, cross-encoder raw-logit reranking
(ms-marco-MiniLM-L6-v2), an explicit threshold, overlap removal and at most five unchanged passages with
stable IDs and citations. It never generates answers. Status: development package; no approved corpus,
no evaluated threshold, no measured retrieval quality.

The canonical retrieval contract lives in `retriever/schema.py`; the gate's orchestrator imports it
(no duplicate dictionaries). Every call returns exactly one `RetrievalResult` with status `ok`,
`no_evidence` (orchestrator returns A5, no Brain call) or `error` (service unavailable, no Brain call).

| Module | Role |
|---|---|
| `schema.py` | Strict requests/results, enums, error codes, `EmbeddingRef` |
| `config.py` | Profile validation, token budgets, fingerprints, readiness problems |
| `register.py` | Source records, rights/approval/review-date eligibility, revocation registry |
| `parse.py` | Markdown/JSON/text-PDF adapters, safe import paths, quarantine |
| `chunk.py` | Sentence-aware chunking, locators, stable passage IDs |
| `build.py` / `snapshot.py` | Staged validated snapshots, atomic activation, pinned handles |
| `dense.py` / `lexical.py` / `fuse.py` / `rerank.py` / `select.py` | Search, RRF, raw logits, selection, conflicts |
| `fetch.py` | Approved Quiz-item evidence sets |
| `service.py` | Validation, scope rules, deadlines, bounded worker, typed errors, audit |
| `api.py` | Optional authenticated internal HTTP wrapper |
| `tools/` | `build_index`, `validate_sources`, `evaluate`, `tune_profile`, `benchmark`, `export_schemas`, `make_fixture_corpus` |

Profiles: `config/retriever_development.yaml` (`fixture` by default). Docs: `docs/retriever_snapshot_and_evaluation.md`
(manifest, evaluation report template, benchmark procedure) and `docs/schemas/`.

### Quick fixture run (synthetic corpus, fake approvals)

```bash
python -m tools.make_fixture_corpus --out artifacts/retriever/fixture
python -m tools.validate_sources --profile config/retriever_development.yaml \
    --register artifacts/retriever/fixture/register.json --import-root artifacts/retriever/fixture/sources
python -m tools.build_index --profile config/retriever_development.yaml \
    --register artifacts/retriever/fixture/register.json --import-root artifacts/retriever/fixture/sources \
    --tenant dev-tenant --course histology-dev --kb-version fixture-kb-002 --index-version idx-1 \
    --synthetic-fixture --activate
python -m tools.benchmark --profile config/retriever_development.yaml --course histology-dev \
    --queries artifacts/retriever/fixture/queries.txt --clients 1,4 --repeats 5
```

`validate_sources` exits 1 on the fixture because it deliberately contains quarantined sources
(missing rights, under review, a blank/scanned PDF page).

### Real-model development (offline)

1. Provision the reranker on a machine with internet access, then copy it to
   `artifacts/rerankers/ms-marco-MiniLM-L6-v2/` (the E5 encoder is already at
   `artifacts/encoders/e5-small-v2/`). Needed files: `config.json`, `model.safetensors`,
   `tokenizer.json`, `tokenizer_config.json` (plus `vocab.txt`/`special_tokens_map.json` if present).
2. Set `reranker.revision` in a copy of the profile to the exact 40-character commit you downloaded,
   `operating_mode: real_model_development`, and an explicitly **provisional** `threshold`
   (`status: provisional`). Results are then labelled `UNEVALUATED PROFILE`.
3. Build from the reviewed register with `tools.build_index` (no `--synthetic-fixture`), record the
   reviewer of the stratified sample with `--approve-review NAME`, then `--activate`.

The runtime never downloads models, sources or datasets.

### Preparing real course sources

See `docs/preparing_course_sources.md`: write notes with `templates/lecture_notes_template.md`, register
each file with `python -m tools.register_source` (computes the hash; missing rights/approval means the
source is quarantined), then validate, build and search with `config/retriever_real_dev.yaml`
(real models, provisional threshold, development only) and `python -m tools.search`.

### Retriever known gaps

- Reranker pinned at revision 233902d25c440f23af6f7d6e94d2946bac0bee0a (provisioned on Windows); check it with
  `python scripts/check_reranker.py`. Its raw-logit path is also tested here against a tiny local BERT
  cross-encoder. The real E5 encoder was verified on Windows.
- No approved histology corpus, rights records, item mappings, evaluation labels or evaluated threshold.
- PDF adapter is conservative: no OCR, no table-cell inference; pages that look like tables/columns
  or are blank/scanned are quarantined for reviewed transcription.
- The bounded worker is a thread; a stuck model call is abandoned (bounded restarts, then not ready).
  Production should move inference into a killable worker process.

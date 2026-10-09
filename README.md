# Gate Classifier v0.2 (English-only)

Text gate for Module A (histology study tutor) of the Medical AI Education Project, implementing
`Gate Classifier English-Only Implementation Specification v0.2`.

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

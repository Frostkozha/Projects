# Coordinator and local chat (System Integration plan v0.3)

`tutor_app/` connects the gate, retriever, Brain, verifier and a minimal student chat page into one local
service. Wire schemas stay at 0.2. **Status: development prototype. `release_ready` is false.** It is not
approved for real students.

## Start the chat

Close any other `llama-server` on port 8080 first, such as the `start-llama-menu.bat` chat launcher, because
the tutor starts and supervises its own Brain there. Then double-click `scripts\start-tutor.bat`, or run:

```powershell
.venv\Scripts\python.exe -m tutor_app.cli start --config config\local-dev.yaml
```

Open <http://127.0.0.1:8000/>. Sign in with any development name, accept the notice, then:

| Mode | What to type | What happens |
|---|---|---|
| Answer | Any question about the indexed notes | Retrieval, then Brain draft, then NLI verification. Returns cited A1/A2 or a fixed A5 |
| Quiz | `stem cells` (or `Quiz me on stem cells`) | Shows an approved multiple-choice item. Answer with one letter |
| Tutor | `stem cells` (or `Tutor me on stem cells`) | Shows an approved open question. Type `show explanation` or `continue` |

Startup takes about 45 seconds, mostly loading the models and checking the Brain.

## Profiles

| Profile | Gate | Retriever | Brain | Verifier | Readiness label |
|---|---|---|---|---|---|
| `config/fixture.yaml` | rules + fixed fixture scores | fixture corpus | deterministic fixture | fixture NLI | `fixture_only` |
| `config/local-dev.yaml` | rules + fixed fixture scores (no trained bundle yet) | real index of your notes | real Qwen3.5-9B | real DeBERTa NLI | `development_mixed` |
| `config/local-real.yaml` | trained bundle required | real | real | real | not ready until a gate bundle exists |
| student_release | as real_model + institutional JWT identity, release records | | | | refused today |

There is no automatic downgrade. Fixture adapters can never satisfy `real_model` or `student_release` readiness.

## Request path

```
POST /v1/courses/{course}/interactions  (Idempotency-Key, 32 KiB, strict JSON, dev cookie or signed JWT)
  -> notice check -> idempotency record -> owned session (revision CAS)
  -> gate (rules, privacy, risk, route)
       clarify A3 / reply A5,A6 / escalate A7 (+ local welfare outbox) / unavailable 503  -- no teaching calls
       retrieve:
         approved Quiz/Tutor item (exact alias, exact item evidence fetch, key compared in code)
         or generated answer: one GPU slot (8 waiting, 8 s)
            retriever -> live registry -> frozen evidence -> Brain -> verifier -> formatter
  -> final BEGIN IMMEDIATE transaction: request state, deployment state, source eligibility, session CAS,
     learning event, audit row, payload digest -> commit -> reply
```

| Module | Role |
|---|---|
| `tutor_app/api.py` | Routes, bounded body reading, error envelope, security headers, static page |
| `tutor_app/orchestrator.py` | The state machine, idempotency, learning paths, generated path, breakers |
| `tutor_app/delivery.py` | Final authorization transaction (linearization point) |
| `tutor_app/store.py`, `migrations/` | SQLite (WAL, one dedicated DB thread), migrations, live registry, audit |
| `tutor_app/components.py` | Wiring of the existing gate/retriever/Brain/verifier packages, live-registry adapter |
| `tutor_app/auth.py` | Dev cookie identity; `TrustedProxyJwtIdentity` (PyJWT, pinned algorithm/issuer/audience/roles) |
| `tutor_app/items.py`, `review.py` | Approved item bank, exact topic aliases, review schedule, practice indicator |
| `tutor_app/capacity.py` | Generated slot, pending caps, deadlines, circuit breakers |
| `tutor_app/formatter.py` | Exact verified sentences + citation markers + authoritative citations |
| `tutor_app/readiness.py`, `bundle.py` | Profile readiness, release bundle activation/rollback |
| `tutor_app/static/` | Accessible chat page (textContent only, generation counter, cancel, retry) |

## Operator commands

```bash
python -m tutor_app.cli validate --config config/local-dev.yaml [--start-brain]
python -m tutor_app.cli status --config config/local-dev.yaml
python -m tutor_app.cli suspend --config config/local-dev.yaml --expected-revision 0 --actor me [--exam]
python -m tutor_app.cli resume --config config/local-dev.yaml --expected-revision 1 --actor me
python -m tutor_app.cli revoke-source --config config/local-dev.yaml --source-id stem-cells-notes
python -m tutor_app.cli inspect-outbox --config config/local-dev.yaml
python -m tutor_app.cli ack-outbox --config config/local-dev.yaml --event-id ID --actor me
python -m tutor_app.cli activate-bundle --config config/local-dev.yaml --bundle path/to/release.json
python -m tutor_app.cli evaluate --suite integration --out reports
python -m tutor_app.cli export-schemas --out docs/schemas
```

## Measured results (this PC, 10 Oct 2026)

| Record | Result |
|---|---|
| Engineering tests | `pytest`: 606 passed, 8 skipped (the Brain real-model suite, run separately: 8 passed) |
| Integration I01-I40 | 63 tests in `tests/tutor` and `tests/test_integration_primitives.py`, all passing |
| Real local pipeline (`local-dev`) | Grounded questions answered A2 with citations in 4.6-9.7 s; off-topic A5 in 0.2 s; quiz path 0.0 s |
| Real load, 3 concurrent dev users x 2 questions | p50 8.0 s, max 14.6 s (nearest rank); 2 of 6 refused `CAPACITY_EXCEEDED` by the 8 s queue ceiling; lowest free VRAM 1,787 MiB; `reports/tutor/real-dev-load.json` |
| Fixture load (4 users x 5) | `reports/tutor/fixture-load.json` |

These are engineering and real-model smoke records only. No faculty evaluation exists.

## Findings

- **Verifier strictness.** With `thresholds_test.yaml` and DeBERTa-v3-xsmall, correct paraphrases are often
  rejected. For example, "Pluripotent stem cells can become any cell type in the body" was marked CONTRADICTED
  against a source sentence that continues "...but cannot form the placenta". Answers then fall back to A5.
  The coordinator enforces the verifier's decision exactly; calibration is verifier work (see next steps).
- **One GPU slot.** Concurrent users queue for at most 8 s, then receive 429. This is the plan's bounded
  admission, not a fault.
- **Brain cancellation** restarts the owned `llama-server` (about 16 s) because this llama.cpp build does
  not free the slot within 1 s. See `docs/brain.md`.

## Acceptance scenarios I01-I40

All are implemented in `tests/tutor/` (synthetic fixtures). Partial items are marked.

| ID | Status | Notes |
|---|---|---|
| I01-I06 | pass | 422/401/403/413/NOTICE_REQUIRED/A3/A5/A6 with zero teaching calls |
| I07 | pass | Hard crisis with gate inference failure: A7 and a pending local outbox event |
| I08 | pass (generated case) | A generated fictional crisis creates no welfare event; quoted-educational crisis wording relies on the gate rule engine and is untested |
| I09-I10 | pass | A7 displays under audit/outbox outage (health degraded); other responses refuse without audit |
| I11-I20 | pass | Scope, evidence binding, Brain/verifier routes, A1/A2 coverage, formatter, hidden keys |
| I21-I23 | pass | Revocation before Brain / before authorization / after authorization; suspension and exam switch |
| I24-I28 | pass | CAS with one learning event, idempotency in progress/replay/conflict/isolation, restart recovery, buffer expiry |
| I29-I33 | pass | Exact item fetch, invalid options, version invalidation, fixed Tutor path, schedule and indicator |
| I34-I36 | pass | Bounded queue, cancel with no late delivery, breakers that count faults only. Native Brain restart is covered by B26 |
| I37 | pass | Fixture adapters and missing manifests fail real_model readiness |
| I38 | pass | JWT key/issuer/audience/expiry/nbf/role/tenant errors and forged headers. Campus issuer still unknown |
| I39 | pass | Bundle pointer with previous kept; incompatible migrations refused; revocations untouched. Timed one-hour rehearsal not done |
| I40 | partial | Fixture and real dev load runs recorded with p50/p95/p99, VRAM and RAM. No offline packet capture or chaos run |

## Dependency report

| Item | State |
|---|---|
| Trained gate bundle | **Missing.** `local-dev` uses the gate rule engine with fixed fixture scores; `local-real` is not ready |
| Verifier thresholds | Test thresholds only; needs calibration on a development set (larger NLI model worth evaluating) |
| Course sources | One source indexed (student stem-cells notes, self-approved). Faculty-approved sources and rights needed |
| Practice item bank | `config/items/histology_dev_bank.json`: 5 quiz and 3 tutor items written from the notes, **unreviewed** |
| Institutional identity | JWT adapter built and tested; campus SSO issuer, keys and role mapping unknown (NU IT) |
| Pseudonym service, retention, welfare procedure, approved contacts | Missing (placeholders in `config/development.yaml`) |
| Release manifest and faculty evaluation records | Missing; `student_release` readiness reports them |
| Lock | `requirements.lock` + PyJWT, cryptography, cffi, pycparser (installed and tested on Windows) |

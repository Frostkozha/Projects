# Brain (local drafting service)

Implements *Brain AI Technical and Coding Plan v0.3* (wire schemas 0.2). The Brain receives approved
evidence, writes strict cited sentence objects (`brain-draft-0.2`) and returns them to the coordinator.
It never sends text to a student, never returns a response code and cannot bypass the independent verifier.

**Status: development package, `release_ready: false`.** Hardware fit and runtime behaviour are measured on
one machine (below). There is no faculty evaluation, holdout set or clinical validation.

## How it works

```
gate (retrieve) -> retriever (1-5 permitted passages) -> coordinator freezes the shown passages
   -> Brain: validate request + frozen evidence digest -> render prompt with the runtime's own template
      -> exact GGUF token count (<= 2,816) -> constrained JSON generation (one slot, 30 s incl. queue)
      -> strict parse + sentence policy -> BrainResult (ok / no_evidence / error)
   -> verifier (support, citations, harm) -> authorized formatter -> student
```

| Module | Role |
|---|---|
| `contracts/json_codec.py` | Strict JSON: duplicate keys, NaN, bad UTF-8, trailing data rejected; canonical JSON/digests |
| `contracts/draft.py` | Re-exports the single `DraftAnswer`/`DraftSentence` (`contracts/models.py`) plus `MachineID`, `StrictModel` |
| `contracts/evidence.py` | `FrozenEvidence`: hash-bound canonical bundle bytes; `bundle_value` from a `RetrievalResult` |
| `brain/request_schema.py`, `brain/schema.py` | `BrainRequest`, `BrainResult`, `BrainMetrics`, `BrainContext`, `PresentationHint`, error codes |
| `brain/config.py` | Strict YAML profile (unknown keys, duplicate keys, multiple slots, budget mismatch, paths outside approved roots all rejected) |
| `brain/manifest.py` | Candidate manifest, full-revision pins, SHA-256 verification of model and executable |
| `brain/provision.py` | Explicit online step: resolve HF revision, verify local GGUF bytes against the repo digest, resolve runtime tag commit, measure placement, probe |
| `brain/supervisor.py` | One owned `llama-server` (argument array, `shell=False`, own process group, bounded start/stop, never kills a foreign process); flag check against `--help`; placement preflight |
| `brain/capabilities.py` | 16 readiness probes (identity, template hash, slot context, auth, no web UI, exact token count, special tokens, reasoning-off, unsupported grammar rejected, hostile-format schema enforcement, slot idle) |
| `brain/prompt.py`, `prompts/*.txt` | Fixed versioned system prompts; one JSON user data record (question, passages, presentation hint) |
| `brain/grammar.py` | Inline flat JSON schema from the draft contract, cites restricted to the current passage IDs |
| `brain/tokenizer.py`, `brain/budget.py` | Runtime `/apply-template` + `/tokenize` count (proven equal to the runtime's `usage.prompt_tokens`); 2,816 + 1,024 + 256 = 4,096 |
| `brain/transport.py` | Loopback-only HTTPX client: no redirects/proxies/compression, 256 KiB buffer, absolute deadline |
| `brain/parse.py` | Whole-object parser (no brace search or repair) and the verifier-compatible sentence policy |
| `brain/service.py` | Queue (8 / 8 s), 30 s deadline, cancellation checks, recovery that holds the slot until the native slot is proven idle or the owned server is restarted and re-probed |
| `brain/adapter.py` | Plugs the service into the orchestrator's `Brain` protocol (`INVALID_OUTPUT` -> fixed A5; infrastructure errors -> 503) |
| `brain/evaluate.py`, `brain/cli.py` | Benchmark runner/metrics and the command interface |

## Setup (Windows, PowerShell)

Runtime and model live outside the repository:

- `D:\llama-cuda\llama-b11435-bin-win-cuda-12.4-x64\llama-server.exe` (official prebuilt release b11435)
- `D:\llama-cuda\cudart-llama-bin-win-cuda-12.4-x64\` (CUDA 12.4 runtime DLLs)
- `D:\models\Qwen3.5-9B-Q4_K_M.gguf`

Paths are set in `config/brain-local-9b.yaml` (`artifacts`, `approved_roots`). Create the local API key once
(never committed; `secrets/` is git-ignored):

```powershell
New-Item -ItemType Directory -Force secrets | Out-Null
.venv\Scripts\python.exe -c "import secrets;print(secrets.token_urlsafe(32))" | Out-File -Encoding ascii secrets\brain.key
```

Bash equivalent: `mkdir -p secrets && python -c "import secrets;print(secrets.token_urlsafe(32))" > secrets/brain.key`.

## Commands

```bash
python -m brain.cli provision --candidate qwen35-9b-q4      # online, explicit; --download only if the file is absent
python -m brain.cli validate --config config/brain-local-9b.yaml
python -m brain.cli start --config config/brain-local-9b.yaml   # or scripts\start-brain.bat
python -m brain.cli smoke --config config/brain-local-9b.yaml
python -m brain.cli benchmark --dataset tests/fixtures/brain/synthetic.jsonl --config config/brain-local-9b.yaml --out reports/brain --repeats 3
python -m pytest tests/brain tests/test_brain_primitives.py
python -m pytest tests/brain/real_model --run-local-model
```

The Brain uses port 8080 and refuses to start if anything else holds it (for example the interactive chat
launcher `D:\llama.cpp\start-llama-menu.bat`). Run one or the other.

## Runtime findings (pinned build b11435-43fe9c642, verified 10 Oct 2026)

- `--fit` defaults to **on** and may silently lower GPU layers: the profile sets `--fit off` and placement is
  read from the loader log (`offloaded 33/33 layers to GPU`) in a request-free preflight.
- A cross-request prompt cache (`--cache-ram`, 8 GiB default) is enabled unless `--cache-ram 0`; the profile disables it.
- With `--reasoning off` the Qwen3.5 template pre-fills an empty `<think>\n\n</think>\n\n` scaffold in the
  prompt; it never appears in `content`. The probe requires exactly this suffix and rejects any think text.
- Special-token strings in user content (e.g. `<|im_end|>`) are parsed as real control tokens by the server.
  Requests/evidence whose tokenization differs with and without special parsing are rejected (B09).
- Unsupported schema features (remote `$ref`) return HTTP 500 rather than being ignored; image input is refused.
- Rendered-prompt token count via `/apply-template` + `/tokenize` equals the runtime's `usage.prompt_tokens` exactly.
- **Cancellation:** after a client cancels a non-streaming generation, the slot was *not* idle within the 1 s
  bound. The service therefore restarts the owned server and re-probes it (measured ~16 s of unavailability).
  This is the plan's fail-closed path, not a defect, but it means cancellations are expensive on this build.

## Measured results (RTX 3070 Ti 8 GiB, driver 610.88, Ryzen 7 5800X, 32 GiB RAM)

| Measure | Value |
|---|---|
| Placement | 33/33 layers on GPU; CUDA model 4,861 MiB, KV 128 MiB, recurrent 50 MiB, compute 27 MiB |
| Free VRAM low point (benchmark) | 1,854 MiB (criterion >= 512 MiB) |
| Latency (42 synthetic runs) | p50 1.9 s, p95 4.8 s |
| Prefill / decode | ~1,818 / ~83 tokens/s |
| Prompt size | mean 434, max 475 tokens |
| Structure | 0 invalid outputs of 39 generations; 42/42 runs matched expected status |
| Repeat consistency | identical status and citation set across 3 repeats for every case |

Report: `reports/brain/benchmark-20261010-181513.json` (synthetic only). Gaps: CPU services (E5, MiniLM,
DeBERTa) were **not** loaded concurrently during this benchmark; no faculty labels; one-slot only.

## Acceptance scenarios B01-B36

`unit/contract` = fake runtime in `tests/brain`; `real` = `tests/brain/real_model` on the pinned GGUF.

| ID | Status | Evidence |
|---|---|---|
| B01 | pass | `test_parse_contracts.py::test_B01_*` |
| B02 | pass | `test_service.py::test_B02_*` |
| B03 | pass | `test_service.py::test_B03_*`, source-change abort |
| B04 | pass | `test_service.py::test_B04_B08_B11_payload_contents` |
| B05 | pass (unit + real) | probe `known_token_count`, `rendered_count_matches_runtime`; real B05 test |
| B06 | pass | `test_B06_exact_ceiling_passes_one_over_fails`, primitives |
| B07 | pass (unit + real) | `test_B07_*`, real oversize |
| B08 | pass | system role is the fixed prompt file only |
| B09 | pass (unit + real) | static screen + runtime tokenizer comparison |
| B10 | pass (unit + real probe) | reasoning-off probe, parser rejects reasoning/think |
| B11 | pass | payload key set, loopback-only transport, `--no-mmproj`, image refused by runtime |
| B12 | pass (unit + real probe) | inline schema tests; hostile-format probe on real runtime |
| B13 | pass (unit + real probe) | `unsupported_grammar_rejected` |
| B14-B19 | pass | `test_parse_contracts.py` |
| B20-B22 | pass | finish reasons, null metrics, transport faults |
| B23 | pass | single completion after invalid content |
| B24 | pass | queue full / bounded wait, no deferred generation |
| B25 | pass (unit + real) | cancel before/during; late content discarded |
| B26 | pass (unit + real) | real build required restart; admission blocked until re-probed |
| B27 | partial | placement/OOM parsing unit-tested and measured; no forced-OOM stress run |
| B28 | pass | manifest pins/bytes; startup rejects candidate/prompt/schema drift |
| B29 | pass (unit + real) | audit and server log sentinel checks |
| B30 | partial | `--offline`, `HF_HUB_OFFLINE`, no network code at serve time; no firewall deny-all capture run |
| B31 | pass | counterfactuals marked synthetic, separate dataset, never indexed |
| B32 | partial | mutations share parent `group_id`; no holdout partition exists yet |
| B33 | pass | 3-repeat consistency report (5-repeat run is a CLI flag: `--repeats 5`) |
| B34 | partial | one-slot benchmark with VRAM/RAM; CPU services not concurrently loaded |
| B35 | pass (config) | separate profiles/manifest paths; only 9B provisioned |
| B36 | pass (unit + real) | orchestrator -> verifier -> formatter; INVALID_OUTPUT -> A5 |

## Dependency report

| Item | State |
|---|---|
| Qwen3.5-9B Q4_K_M | provisioned, bytes match `unsloth/Qwen3.5-9B-GGUF@3885219b…` digest |
| llama.cpp runtime | official prebuilt b11435 (`43fe9c64…`), executable hash recorded. The plan prefers a local source build; the prebuilt release was used and is recorded as such |
| Qwen3.5-4B / Qwen3-4B comparison models | configs only; not downloaded (`provision --candidate … --download`) |
| System Integration plan v0.3 (coordinator) | not supplied; integration uses the existing orchestrator via `brain/adapter.py` |
| tutor_clue | disabled; needs a faculty-reviewed prompt and leakage evaluation (`prompts/tutor_clue.reviewed`) |
| Faculty 200-question set, holdout, labels | missing; blocks any release-validation claim |
| Pinned spaCy `en_core_web_sm` | missing; sentence policy uses the rule-based sentencizer (same as verifier dev profile) |

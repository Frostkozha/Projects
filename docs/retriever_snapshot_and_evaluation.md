# Retriever: snapshot manifest, evaluation report template and benchmark procedure

Companion to Retriever English-Only Specification v0.2. JSON Schemas for the source register, profile,
item mappings, conflict records and the retrieval contract are in `docs/schemas/`
(regenerate with `python -m tools.export_schemas --out docs/schemas`).

## 1. Snapshot layout and manifest

```
<snapshots_root>/<course_id>/
  ACTIVE.json                 {"index_version", "kb_version"}  (atomic os.replace under a lock)
  <index_version>/
    snapshot.sqlite           metadata, FTS5 (unicode61 remove_diacritics 0), items, conflicts, quarantine
    vectors.npy               float32 [passages x 384], row i <-> passages.vector_row i
    manifest.json
```

`manifest.json` fields:

| Field | Meaning |
|---|---|
| schema_version | `retriever-snapshot-0.2` |
| index_version, kb_version, tenant_id, course_id | Identity; one snapshot per tenant/course |
| created_at | UTC build time |
| operating_mode, synthetic_fixture | `fixture` snapshots are refused outside fixture mode |
| profile.{profile_version, representation_fingerprint, chunking_fingerprint, query_builder_version, retrieval_policy_version} | A changed encoder/chunking/normalization requires a rebuild |
| encoder.{encoder_id, encoder_revision, tokenizer_revision, pooling, dimension, dtype, normalization, preprocessing_fingerprint} | Vector compatibility |
| reranker.{model_id, revision} | Reranker used for build-time checks |
| parser_version, dimension, passage_count | Integrity checks at load |
| sources[] | source_id, source_version, file_sha256 of every indexed source |
| quarantine_count | Details in the `quarantine` table |
| review.{sample_passage_ids, approved, reviewer} | >=50 (or all) stratified passages; must be approved outside fixture mode |
| evaluation_ref | Required in production |
| artifacts | SHA-256 of snapshot.sqlite and vectors.npy |

Load-time validation rejects: checksum mismatch, vector/row mapping mismatch, FTS count mismatch,
non-unit or non-finite vectors, wrong dimension, passage text/hash mismatch, recomputed passage-ID
mismatch, passages without a source record, and fingerprint changes. Revocations live outside snapshots
(`revocations_path`) and are never rolled back with an index.

## 2. Evaluation dataset record

JSONL, one question per line (validated by `tools/evaluate.py`):

```json
{"id": "q-0001", "group_id": "epi-squamous-01", "partition": "development",
 "query_text": "Where is simple squamous epithelium found?", "course_id": "histology-pilot",
 "kb_version": "kb-2026-10-01", "topic": "epithelium", "answerability": "answerable",
 "relevant_passage_ids": ["psg-..."], "sufficient_evidence_sets": [["psg-..."]],
 "expected_conflict": null, "annotator_ids": ["rater-a", "rater-b"], "approval_status": "approved"}
```

Unanswerable records have empty `relevant_passage_ids`/`sufficient_evidence_sets` and a reviewed
`missing_scope`. A `group_id` may not appear in both partitions. Tuning (`tools/tune_profile.py`) refuses
final-partition records; the final partition needs `--confirm-frozen` and an evaluated profile.

## 3. Evaluation report template

Fill from `tools/evaluate.py` output. Every rate carries numerator/denominator and a Wilson 95% CI.

| Item | Value |
|---|---|
| Corpus snapshot (kb_version / index_version) | |
| Profile version / threshold status (`UNEVALUATED PROFILE` unless evaluated) | |
| Partition (development / final) and frozen-targets reference | |
| Reviewers, double-review agreement on >=50 cases | |
| Hit@5 (target >=90%, lower Wilson >=0.85) | |
| Candidate Hit@30 (target >=95%) | |
| Passage Recall@5 (report only) | |
| Sufficient-set coverage@5, multi-evidence subset separately (target >=90%) | |
| Context precision@5 on answerable set (target >=80%) | |
| Correct no_evidence (target >=90%, lower Wilson >=0.85) | |
| False no_evidence (target <=5%, upper Wilson <=0.10) | |
| Operational errors (counted as failures, never as correct no_evidence) | |
| Access/citation integrity violations (target 0) | |
| Ablations on development: dense-only, lexical-only, RRF-only, hybrid+reranker | |
| Gate-only / retriever-only / end-to-end losses | |
| Known failures and follow-up plan | |

Targets are the specification's proposed values; they require Evaluation Lead agreement before the
final test and must not be relaxed after seeing final results.

## 4. Benchmark procedure

1. Provision pinned models locally; build and activate the reviewed corpus snapshot.
2. Prepare a query file of representative course questions (no student data).
3. Run with the Brain loaded concurrently if it shares the machine:
   `python -m tools.benchmark --profile config/retriever_development.yaml --course <course> --queries queries.txt --clients 1,4,8 --repeats 20`
4. Record model load time, p50/p95 per client count, error counts (DEADLINE_EXCEEDED / CAPACITY_EXCEEDED),
   passage count, CPU model and thread settings. Peak RSS is reported on Linux/macOS; on Windows read it
   from Task Manager or Performance Monitor.
5. Report measurements as measured on that machine. The p95 <= 2 s aim at one client is an engineering
   aim, not a promise.

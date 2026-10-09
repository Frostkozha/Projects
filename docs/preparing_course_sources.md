# Preparing real course sources for the Retriever

This is the workflow for turning approved histology material into a searchable index on your PC.
Nothing here is shown to students: `config/retriever_real_dev.yaml` is a real-model *development* profile
with a provisional, unevaluated threshold.

## 1. What may be indexed

Only material for which you hold, in writing:

- permission to process it locally and to show short excerpts to enrolled students (rights), and
- a faculty member's approval of the content (approval).

Do not index textbooks, websites, StatPearls, guidelines, classmates' notes, preprints or AI-written text
just because they are available. Missing rights or approval means the tool marks the source
`pending_review` and the build quarantines it.

## 2. Formats

| Format | Use for | Notes |
|---|---|---|
| Markdown `.md` | Lecture notes (best choice) | Use `templates/lecture_notes_template.md`. Headings become citations. |
| JSON `.json` | Slides with slide numbers, practice questions | `{"blocks": [{"block_id", "section_path", "kind", "text", "slide"}]}` |
| Text PDF `.pdf` | Approved PDFs with selectable text | Scanned pages, tables and two-column layouts are quarantined. Convert those to Markdown. |

Put files under `artifacts\retriever\sources\` (this folder is never committed to Git).

## 3. Register each file

```powershell
.\.venv\Scripts\python.exe -m tools.register_source --register artifacts\retriever\real\register.json `
  --file epithelium_lecture.md --source-id epithelium-lecture --version 1 `
  --title "Epithelium lecture notes" --owner "Department of Histology" --library lib1 --topics epithelium `
  --rights-ref rights-epithelium-2026 --rights-reviewer "Your name" --rights-document "Email from Dr X, 2026-10-09" `
  --approver "Dr X" --review-due 2027-06-30 --check "exact sentence with units from the notes"
```

| Field | Meaning |
|---|---|
| `--library` | `lib1` lecture notes/slides, `lib2` approved textbooks, `lib5` practice questions, `lib6` atlas captions/transcripts |
| `--source-id` | Short stable name (letters, digits, `-`, `_`, `.`) |
| `--version` | Increase it whenever the file changes, then rebuild with a new `--kb-version` |
| `--review-due` | After this date the source is automatically excluded until re-reviewed |
| `--check` | Phrases that must survive extraction exactly; the build fails if one is lost |

Re-running the command for the same id and version replaces that entry (and recomputes the file hash).

## 4. Validate, build and activate

```powershell
.\.venv\Scripts\python.exe -m tools.validate_sources --profile config\retriever_real_dev.yaml `
  --register artifacts\retriever\real\register.json --import-root artifacts\retriever\sources

.\.venv\Scripts\python.exe -m tools.build_index --profile config\retriever_real_dev.yaml `
  --register artifacts\retriever\real\register.json --import-root artifacts\retriever\sources `
  --tenant dev-tenant --course histology-dev --kb-version kb-2026-10-09 --index-version idx-2026-10-09 `
  --approve-review "Your name" --activate
```

Before using `--approve-review`, open `artifacts\retriever\real\snapshots\histology-dev\<index>\manifest.json`,
read the passages listed under `review.sample_passage_ids` (inspect them with any SQLite viewer on
`snapshot.sqlite`, table `passages`) and confirm the text was extracted correctly. Build first without
`--approve-review --activate` if you want to review before approving.

Each new build needs a new `--index-version`; a changed corpus also needs a new `--kb-version`.

## 5. Try a search

```powershell
.\.venv\Scripts\python.exe -m tools.search --profile config\retriever_real_dev.yaml --course histology-dev `
  --query "Where is simple squamous epithelium found?"
```

Results are labelled UNEVALUATED. The threshold becomes trustworthy only after the faculty-reviewed
evaluation set is built and `tools.tune_profile` / `tools.evaluate` are run.

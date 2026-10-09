"""SQLite FTS5 BM25 search with a literal query builder (Retriever spec v0.2, section 10.1).

The builder emits only quoted word/number tokens joined by OR and binds everything as SQL
parameters. User MATCH operators, column filters, NEAR/AND/NOT and SQL are never executed.
"""

from __future__ import annotations

import json
import re
import sqlite3
from typing import Optional, Sequence

from .schema import ErrorCode, RetrievalError

FTS_TOKENIZER = "unicode61 remove_diacritics 0"
_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)
MAX_QUERY_TERMS = 64


def build_match_query(text: str) -> Optional[str]:
    """Quoted literal tokens joined by OR, or None when the query has no word/number tokens."""
    seen, terms = set(), []
    for tok in _TOKEN.findall(text.casefold()):
        if tok not in seen:
            seen.add(tok)
            terms.append(f'"{tok}"')  # tokens contain no quotes by construction
        if len(terms) >= MAX_QUERY_TERMS:
            break
    return " OR ".join(terms) if terms else None


def lexical_search(conn: sqlite3.Connection, query_text: str, eligible_rowids: Sequence[int],
                   k: int) -> list[tuple[str, float]]:
    """BM25 (lower is better). Eligibility filter is inside the SQL selection, before LIMIT."""
    match = build_match_query(query_text)
    if match is None or not eligible_rowids:
        return []
    sql = (
        "SELECT p.passage_id, bm25(passages_fts) AS score FROM passages_fts "
        "JOIN passages p ON p.rowid = passages_fts.rowid "
        "WHERE passages_fts MATCH ? AND passages_fts.rowid IN (SELECT value FROM json_each(?)) "
        "ORDER BY score ASC, p.passage_id ASC LIMIT ?"
    )
    try:
        rows = conn.execute(sql, (match, json.dumps([int(r) for r in eligible_rowids]), int(k))).fetchall()
    except sqlite3.Error:
        raise RetrievalError(ErrorCode.SEARCH_FAILED) from None
    return [(pid, float(score)) for pid, score in rows]

"""Real-model check for the retriever's cross-encoder reranker (offline, pinned local files).

Loads the reranker exactly as the retriever does, verifies its configuration against the profile and
scores a few synthetic pairs. Raw logits are scores, not probabilities. Does not need a corpus.
Usage: python scripts/check_reranker.py [--profile config/retriever_development.yaml]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from retriever.config import load_profile  # noqa: E402
from retriever.rerank import CrossEncoderReranker  # noqa: E402
from retriever.schema import RetrievalError  # noqa: E402

NEEDED = ("config.json", "tokenizer_config.json")
PAIRS = [
    ("Where is simple squamous epithelium found?",
     "Simple squamous epithelium has one layer of flattened cells and lines the alveoli of the lung."),
    ("Where is simple squamous epithelium found?",
     "Compact bone is organised into osteons around central Haversian canals."),
    ("How is hyaline cartilage nourished?",
     "Hyaline cartilage has no blood vessels and is nourished by diffusion from the perichondrium."),
    ("How is hyaline cartilage nourished?",
     "Stratified squamous epithelium protects against abrasion in the oesophagus."),
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=str(ROOT / "config/retriever_development.yaml"))
    args = ap.parse_args(argv)
    profile = load_profile(args.profile)
    rr = profile.reranker
    path = Path(rr.local_path or "")
    if not path.is_absolute():
        path = ROOT / path
    print(f"model_id:  {rr.model_id}")
    print(f"revision:  {rr.revision}")
    print(f"folder:    {path}")
    ok = True
    for name in NEEDED:
        if not (path / name).is_file():
            print(f"MISSING    {name}")
            ok = False
    if not any((path / n).is_file() for n in ("tokenizer.json", "vocab.txt")):
        print("MISSING    tokenizer.json (or vocab.txt)")
        ok = False
    if not (path / "model.safetensors").is_file():
        print("MISSING    model.safetensors")
        ok = False
    if not ok:
        return 1
    try:
        reranker = CrossEncoderReranker(str(path), rr.model_id, rr.revision or "", rr.max_pair_tokens, rr.torch_threads)
    except RetrievalError as exc:
        print(f"FAIL       reranker did not load ({exc.code.value}): needs a pinned 40-char revision, one output "
              f"logit and >= {rr.max_pair_tokens} positions")
        return 1
    specials = reranker.tokenizer.pair_special_tokens()
    lengths = [reranker.tokenizer.count_pair(q, d) for q, d in PAIRS]
    logits = np.asarray(reranker.score(PAIRS), dtype=float).reshape(-1)
    print(f"pair special tokens: {specials} (profile reserve {profile.limits.pair_reserve_tokens})")
    print(f"pair lengths: {lengths} (ceiling {rr.max_pair_tokens})")
    for (q, d), s in zip(PAIRS, logits):
        print(f"logit {s:+8.3f}  | {q[:40]:<40} | {d[:50]}")
    checks = {
        "one finite logit per pair": logits.shape == (len(PAIRS),) and bool(np.all(np.isfinite(logits))),
        "special tokens fit reserve": specials <= profile.limits.pair_reserve_tokens,
        "pairs within ceiling": max(lengths) <= rr.max_pair_tokens,
        "relevant pair ranks above unrelated (both questions)": logits[0] > logits[1] and logits[2] > logits[3],
    }
    for name, passed in checks.items():
        print(f"{'ok' if passed else 'FAILED':<7}{name}")
    if all(checks.values()):
        print("PASS       reranker loads offline and behaves as expected (raw logits, not probabilities)")
        return 0
    print("FAIL       one or more checks failed")
    return 1


if __name__ == "__main__":
    sys.exit(main())

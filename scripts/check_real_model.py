"""Real-model check: load the pinned local E5 encoder exactly as the service does and sanity-check it.

Offline only (local_files_only). Does not need trained heads; it checks the encoder and tokenizer.
Usage: python scripts/check_real_model.py [--config config/development.yaml]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gate_classifier.config import load_config  # noqa: E402
from gate_classifier.encoder import E5Encoder, EncoderError, build_canonical_text  # noqa: E402

NEEDED = ("config.json", "tokenizer_config.json")
TOKENIZER_FILES = ("tokenizer.json", "vocab.txt")  # at least one


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "config/development.yaml"))
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    enc_cfg = cfg.encoder
    path = Path(enc_cfg.local_path or "")
    if not path.is_absolute():
        path = ROOT / path
    print(f"model_id:  {enc_cfg.model_id}")
    print(f"revision:  {enc_cfg.revision}")
    print(f"folder:    {path}")

    ok = True
    for name in NEEDED:
        if not (path / name).is_file():
            print(f"MISSING    {name}")
            ok = False
    if not any((path / n).is_file() for n in TOKENIZER_FILES):
        print("MISSING    tokenizer.json (or vocab.txt)")
        ok = False
    weights = [w for w in ("model.safetensors", "pytorch_model.bin") if (path / w).is_file()]
    if not weights:
        print("MISSING    model.safetensors (or pytorch_model.bin)")
        ok = False
    elif weights[0] != "model.safetensors":
        print("WARNING    only pytorch_model.bin found; prefer model.safetensors (no pickle loading)")
    if not ok:
        return 1

    try:
        encoder = E5Encoder(str(path), enc_cfg.model_id, enc_cfg.revision or "", enc_cfg.tokenizer_revision or "",
                            enc_cfg.prefix, enc_cfg.dimension, enc_cfg.max_tokens, enc_cfg.torch_threads)
    except EncoderError as exc:
        print(f"FAIL       encoder did not load: {exc}")
        return 1

    texts = [
        "Explain the structure of simple squamous epithelium.",
        "Which epithelium lines the alveoli of the lung?",
        "Recommend a good laptop for gaming.",
    ]
    model_texts = [enc_cfg.prefix + build_canonical_text(t) for t in texts]
    vecs = encoder.encode(model_texts)
    norms = np.linalg.norm(vecs, axis=1)
    sims = vecs @ vecs.T
    tokens = encoder.tokenizer.count(model_texts[0], True)
    print(f"shape:     {vecs.shape} (expected (3, {enc_cfg.dimension}))")
    print(f"norms:     {np.round(norms, 4).tolist()} (expected 1.0)")
    print(f"tokens:    {tokens} for the first request incl. special tokens")
    print(f"similarity related={sims[0, 1]:.3f} unrelated={sims[0, 2]:.3f}")
    checks = [
        vecs.shape == (3, enc_cfg.dimension),
        bool(np.all(np.isfinite(vecs))),
        bool(np.allclose(norms, 1.0, atol=1e-4)),
        sims[0, 1] > sims[0, 2],
        0 < tokens <= enc_cfg.max_tokens,
    ]
    if all(checks):
        print("PASS       encoder and tokenizer load offline and behave as expected")
        return 0
    print("FAIL       one or more checks failed")
    return 1


if __name__ == "__main__":
    sys.exit(main())

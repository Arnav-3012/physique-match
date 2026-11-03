"""Load the committed index and refuse to run if it does not match its manifest (ADR-048)."""

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

KB_DIR = Path(os.environ.get("KB_DIR", Path(__file__).parents[4] / "kb"))


@dataclass(frozen=True)
class Corpus:
    chunks: list[dict]
    embeddings: np.ndarray
    manifest: dict


def load(kb_dir: Path = KB_DIR) -> Corpus:
    text = (kb_dir / "chunks.jsonl").read_text()
    manifest = json.loads((kb_dir / "manifest.json").read_text())
    expected = hashlib.sha256((text + manifest["model_sha256"] + manifest["tokenizer_sha256"]).encode()).hexdigest()
    if expected != manifest["corpus_hash"]:
        raise ValueError("kb/chunks.jsonl does not match manifest.corpus_hash; rebuild with `python -m kb.build_index`")
    chunks = [json.loads(line) for line in text.splitlines()]
    embeddings = np.load(kb_dir / "embeddings.npy")
    if embeddings.shape != (len(chunks), manifest["dim"]) or len(chunks) != manifest["chunks"]:
        raise ValueError("kb/embeddings.npy or chunk count does not match the manifest")
    return Corpus(chunks, embeddings, manifest)

"""Build the committed index: `kb/chunks.jsonl`, `kb/embeddings.npy`, `kb/manifest.json` (ADR-048).

Chunks: one per exercise (`ingest_exercises`) and one per guideline-doc heading with a body.
Doc chunk id is `doc:<doc id>#<heading slug>` (see evals/README.md).
Offline only; run from the repo root:

    uv run python -m kb.build_index
"""

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

from kb import ingest_exercises

KB = Path(__file__).parent
MODEL_DIR = KB.parent / "ml" / "data" / "models" / "bge-small"
sys.path.insert(0, str(KB.parent / "services" / "api"))
from app.rag.embed import DIM, MODEL_FILE, TOKENIZER_FILE, Embedder  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def slug(heading: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-")


def frontmatter(text: str) -> tuple[dict, str]:
    _, head, body = text.split("---\n", 2)
    meta = {}
    for line in head.splitlines():
        key, _, value = line.partition(": ")
        meta[key] = (
            [v.strip() for v in value[1:-1].split(",") if v.strip()]
            if value.startswith("[")
            else [value]
        )
    return meta, body


def doc_chunks(path: Path) -> list[dict]:
    meta, body = frontmatter(path.read_text())
    sections, heading, lines = [], None, []
    for line in body.splitlines():
        match = re.match(r"#{1,2} (.+)", line)
        if match:
            sections.append((heading, lines))
            heading, lines = match.group(1).strip(), []
        else:
            lines.append(line)
    sections.append((heading, lines))
    chunks = []
    for heading, lines in sections:
        text = " ".join(" ".join(lines).split())
        if heading and text:
            chunks.append(
                {
                    "id": f"doc:{meta['id'][0]}#{slug(heading)}",
                    "kind": "guideline",
                    "name": heading,
                    "text": f"{heading}. {text}",
                    "muscle": meta["muscle"],
                    "level": meta["level"][0],
                    "goal": meta["goal"],
                    "source_ids": meta["source_ids"],
                    "plannable": False,
                    "source": "kb/docs",
                }
            )
    return chunks


def main():
    chunks = ingest_exercises.load()
    for path in sorted((KB / "docs").glob("*.md")):
        chunks += doc_chunks(path)
    ids = [c["id"] for c in chunks]
    assert len(ids) == len(set(ids)), "duplicate chunk ids"
    embeddings = Embedder(MODEL_DIR).embed_passages([c["text"] for c in chunks])
    assert embeddings.shape == (len(chunks), DIM)
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4), "embeddings are not unit norm"

    chunks_text = "".join(json.dumps(c, sort_keys=True) + "\n" for c in chunks)
    (KB / "chunks.jsonl").write_text(chunks_text)
    np.save(KB / "embeddings.npy", embeddings)
    model_sha, tokenizer_sha = sha256(MODEL_DIR / MODEL_FILE), sha256(MODEL_DIR / TOKENIZER_FILE)
    corpus_hash = hashlib.sha256((chunks_text + model_sha + tokenizer_sha).encode()).hexdigest()
    manifest = {
        "corpus_hash": corpus_hash,
        "chunks": len(chunks),
        "exercise_chunks": sum(c["kind"] == "exercise" for c in chunks),
        "guideline_chunks": sum(c["kind"] == "guideline" for c in chunks),
        "model_sha256": model_sha,
        "tokenizer_sha256": tokenizer_sha,
        "pooling": "cls",
        "normalised": True,
        "dim": DIM,
    }
    (KB / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

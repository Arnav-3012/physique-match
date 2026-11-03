import sys
from pathlib import Path

import numpy as np
import pytest

from kb.build_index import MODEL_DIR, doc_chunks, slug

sys.path.insert(0, str(Path(__file__).parents[1] / "services" / "api"))
from app.rag.embed import DIM, Embedder  # noqa: E402

needs_model = pytest.mark.skipif(not (MODEL_DIR / "model_quantized.onnx").exists(), reason="bge model not downloaded")
KB_DOCS = Path(__file__).parents[1] / "kb" / "docs"


def test_slug_matches_the_eval_readme_rule():
    assert slug('What "volume" means here') == "what-volume-means-here"
    assert slug("Training frequency (sessions per muscle group per week)") == (
        "training-frequency-sessions-per-muscle-group-per-week"
    )


def test_doc_chunk_ids_cover_every_guideline_gold_id():
    import json

    ids = {c["id"] for p in KB_DOCS.glob("*.md") for c in doc_chunks(p)}
    gold = {
        g
        for line in (Path(__file__).parents[1] / "evals" / "rag.jsonl").read_text().splitlines()
        for g in json.loads(line)["gold"]
        if g.startswith("doc:")
    }
    assert gold <= ids, gold - ids


@needs_model
def test_embeddings_are_unit_norm_cls_vectors_and_rank_related_text_higher():
    embedder = Embedder(MODEL_DIR)
    passages = embedder.embed_passages(["Barbell squat. Targets quadriceps.", "How much protein to eat per day."])
    query = embedder.embed_queries(["leg exercise with a barbell"])
    assert passages.shape == (2, DIM)
    assert np.allclose(np.linalg.norm(passages, axis=1), 1.0, atol=1e-4)
    assert (query @ passages.T)[0, 0] > (query @ passages.T)[0, 1]
    again = embedder.embed_passages(["Barbell squat. Targets quadriceps."])
    # one text at a time, so no batch padding effect
    assert np.allclose(passages[0], again[0], atol=1e-6)

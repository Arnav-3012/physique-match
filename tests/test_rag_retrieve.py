import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "services" / "api"))
from app.rag import corpus as corpus_mod  # noqa: E402
from app.rag.corpus import Corpus  # noqa: E402
from app.rag.query import dataset_equipment, decompose, levels_up_to, normalise  # noqa: E402
from app.rag.retrieve import Filters, Retriever  # noqa: E402


class FakeEmbedder:
    """Fixed 3-d vectors keyed on one word, so retrieval logic is tested without the ONNX model."""

    def embed_queries(self, texts):
        return np.array([[1.0, 0.0, 0.0] if "squat" in t else [0.0, 1.0, 0.0] for t in texts])


def toy() -> Retriever:
    chunks = [
        {
            "id": "ex:a",
            "kind": "exercise",
            "text": "Barbell squat quadriceps",
            "muscle": ["quadriceps"],
            "equipment": "barbell",
            "level": "beginner",
            "plannable": True,
        },
        {
            "id": "ex:b",
            "kind": "exercise",
            "text": "Leg press quadriceps machine",
            "muscle": ["quadriceps"],
            "equipment": "machine",
            "level": "intermediate",
            "plannable": True,
        },
        {
            "id": "ex:c",
            "kind": "exercise",
            "text": "Dumbbell curl biceps",
            "muscle": ["biceps"],
            "equipment": "dumbbell",
            "level": "beginner",
            "plannable": True,
        },
        {
            "id": "doc:g#x",
            "kind": "guideline",
            "text": "Weekly sets for growth",
            "muscle": ["all"],
            "level": "all",
        },
    ]
    emb = np.array([[1, 0, 0], [0.8, 0.6, 0], [0, 1, 0], [0, 0.6, 0.8]], dtype=float)
    return Retriever(Corpus(chunks, emb, {}), FakeEmbedder())


def test_filters_keep_only_matching_exercises():
    r = toy()
    assert r.search("leg press", "bm25", Filters(muscle=("quadriceps",), equipment=("machine",))) == ["ex:b"]
    assert r.search("leg press", "bm25", Filters(muscle=("biceps",))) == []
    assert r.search("sets", "dense", Filters(level=("beginner",)), k=5) == [
        "ex:c",
        "ex:a",
    ]  # guideline excluded


def test_filtered_search_uses_the_plannable_pool_unless_asked():
    chunks = [
        {
            "id": "ex:bike",
            "kind": "exercise",
            "text": "Stationary bike quadriceps",
            "muscle": ["quadriceps"],
            "equipment": "machine",
            "level": "beginner",
            "plannable": False,
        },
        {
            "id": "ex:press",
            "kind": "exercise",
            "text": "Leg press quadriceps",
            "muscle": ["quadriceps"],
            "equipment": "machine",
            "level": "beginner",
            "plannable": True,
        },
        {
            "id": "ex:curl",
            "kind": "exercise",
            "text": "Dumbbell curl biceps",
            "muscle": ["biceps"],
            "equipment": "dumbbell",
            "level": "beginner",
            "plannable": True,
        },
    ]  # three chunks and terms found in one chunk each, so BM25's idf is positive
    r = Retriever(Corpus(chunks, np.eye(3), {}), FakeEmbedder())
    quads = Filters(muscle=("quadriceps",))
    assert r.search("bike press", "bm25", quads) == ["ex:press"]
    assert set(r.search("bike press", "bm25", Filters(muscle=("quadriceps",), all_categories=True))) == {
        "ex:bike",
        "ex:press",
    }
    # unfiltered: whole index
    assert set(r.search("bike press", "bm25")) == {"ex:bike", "ex:press"}


def test_bm25_needs_a_term_match_and_dense_does_not():
    r = toy()
    assert r.search("zzz", "bm25") == []
    assert len(r.search("zzz", "dense", k=5)) == 4


def test_hybrid_is_rrf_of_both_lists_and_deterministic():
    r = toy()
    first = r.search("barbell squat", "hybrid", k=3)
    assert first[0] == "ex:a" and first == r.search("barbell squat", "hybrid", k=3)


def test_unknown_mode_fails():
    with pytest.raises(ValueError):
        toy().search("x", "magic")


def test_query_manager_closed_vocabulary():
    assert normalise("  Leg   PRESS \n") == "leg press"
    assert dataset_equipment(["Dumbbells", "my secret gadget", "Bodyweight"]) == (
        "body only",
        "dumbbell",
    )
    assert levels_up_to("intermediate") == ("beginner", "intermediate")


def test_decompose_uses_region_map_groups():
    subs = decompose("thigh", ["machines"], "beginner")
    assert [s.group for s in subs] == ["quadriceps", "hamstrings", "adductors"]
    assert all(s.filters.equipment == ("machine",) and s.filters.level == ("beginner",) for s in subs)
    with pytest.raises(KeyError):
        decompose("elbow", [], "beginner")


def test_committed_index_matches_its_manifest():
    corpus = corpus_mod.load()
    assert corpus.manifest["chunks"] == len(corpus.chunks) == 893

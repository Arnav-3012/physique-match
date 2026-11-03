import json
import sys
from pathlib import Path

import pytest
from tokenizers import Tokenizer

from kb.build_index import KB, MODEL_DIR, over_limit

sys.path.insert(0, str(Path(__file__).parents[1] / "services" / "api"))
from app.rag import corpus as corpus_mod  # noqa: E402

needs_tokenizer = pytest.mark.skipif(not (MODEL_DIR / "tokenizer.json").exists(), reason="bge tokenizer not downloaded")


@needs_tokenizer
def test_only_the_two_known_chunks_exceed_the_model_input():
    # Known limit, ADR-048 Amendment 4: their tail is not embedded; chunking left as is by decision.
    chunks = [json.loads(line) for line in (KB / "chunks.jsonl").read_text().splitlines()]
    assert set(over_limit(chunks)) == {"ex:Power_Clean", "ex:Clean_and_Jerk"}


@needs_tokenizer
def test_stored_text_of_a_long_chunk_is_not_cut():
    # Truncation happens only in the embedder's input; the corpus keeps the full text.
    text = {c["id"]: c["text"] for c in corpus_mod.load().chunks}["ex:Power_Clean"]
    assert len(Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json")).encode(text).ids) > 512

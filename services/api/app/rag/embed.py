"""bge-small-en-v1.5 int8 embeddings with tokenizers + onnxruntime (no torch).

Pooling is CLS and the vector is L2-normalised: BAAI's `1_Pooling/config.json` sets
`pooling_mode_cls_token: true` (the Xenova README example uses mean pooling, which contradicts it).
Queries get the BAAI retrieval instruction, passages never do (BAAI README, model table footnote 1).
Corpus and queries must be embedded by the same model file (ADR-048 same-model rule).
"""

from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
MODEL_FILE, TOKENIZER_FILE = "model_quantized.onnx", "tokenizer.json"
MAX_LEN = 512
DIM = 384


class Embedder:
    def __init__(self, model_dir: Path):
        self.tokenizer = Tokenizer.from_file(str(model_dir / TOKENIZER_FILE))
        self.tokenizer.enable_truncation(MAX_LEN)
        self.tokenizer.enable_padding()
        self.session = ort.InferenceSession(str(model_dir / MODEL_FILE), providers=["CPUExecutionProvider"])
        self.input_names = {i.name for i in self.session.get_inputs()}

    # batch_size 1 on purpose: the int8 model's vectors shift by about 1e-3 per element with the
    # padding of a batch (observed 2026-11-02), so corpus and query vectors take the same path.
    def _embed(self, texts: list[str], batch_size: int = 1) -> np.ndarray:
        out = []
        for start in range(0, len(texts), batch_size):
            enc = self.tokenizer.encode_batch(texts[start : start + batch_size])
            feeds = {
                "input_ids": np.array([e.ids for e in enc], dtype=np.int64),
                "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64),
                "token_type_ids": np.array([e.type_ids for e in enc], dtype=np.int64),
            }
            hidden = self.session.run(None, {k: v for k, v in feeds.items() if k in self.input_names})[0]
            out.append(hidden[:, 0])  # CLS token
        vectors = np.concatenate(out).astype(np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._embed([QUERY_PREFIX + t for t in texts])

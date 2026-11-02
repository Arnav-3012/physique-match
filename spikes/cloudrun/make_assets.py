"""Write the spike's test assets here: model/dummy.onnx (~50 MB) and test.png (~100 KB).

Run from the repo root:
    uv run --with onnx --with numpy --with pillow python spikes/cloudrun/make_assets.py

The ONNX is one Gemm with a random weight matrix: it exercises load time, memory and serving,
not ResNet-18 speed. The < 150 ms Siamese gate is measured in P3 on the real model.
"""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from PIL import Image

HERE = Path(__file__).parent
SEED = 0
IN_DIM, OUT_DIM = 4096, 3072  # 4096 * 3072 * 4 B = 50.3 MB
PNG_SIDE = (256, 400)  # 8-bit noise barely compresses: ~100 KB


def write_onnx(rng: np.random.Generator) -> Path:
    w = rng.standard_normal((IN_DIM, OUT_DIM), dtype=np.float32) * 0.01
    graph = helper.make_graph(
        [helper.make_node("Gemm", ["x", "w"], ["y"])],
        "dummy",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, IN_DIM])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, OUT_DIM])],
        [numpy_helper.from_array(w, "w")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9  # old enough for any recent onnxruntime
    onnx.checker.check_model(model)
    path = HERE / "model" / "dummy.onnx"
    path.parent.mkdir(exist_ok=True)
    onnx.save(model, path)
    return path


def write_png(rng: np.random.Generator) -> Path:
    path = HERE / "test.png"
    Image.fromarray(rng.integers(0, 256, PNG_SIDE, dtype=np.uint8), "L").save(path)
    return path


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    for p in (write_onnx(rng), write_png(rng)):
        print(f"{p.relative_to(HERE)}: {p.stat().st_size / 1e6:.2f} MB")

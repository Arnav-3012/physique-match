import numpy as np
import pytest

torch = pytest.importorskip("torch")  # train group only
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from ml.src.data.regions import LABELLED_REGIONS  # noqa: E402
from ml.src.export_onnx import Proportions, export, session  # noqa: E402
from ml.src.model import SiameseSubtract  # noqa: E402
from ml.src.preprocess import MODEL_H, MODEL_W  # noqa: E402

N_REGIONS = len(LABELLED_REGIONS)


def test_onnx_matches_torch_for_any_batch_size(tmp_path):
    torch.manual_seed(0)
    model = SiameseSubtract(hidden=16, pretrained=False)
    model.fit_labels(torch.rand(20, N_REGIONS) + 0.1)  # non-trivial label scale inside the graph
    model.eval()
    export(model, tmp_path / "m.onnx")
    sess = session(tmp_path / "m.onnx", threads=1)
    for n in (1, 3):
        x = torch.rand(n, 3, MODEL_H, MODEL_W)
        with torch.no_grad():
            expected = Proportions(model)(x).numpy()
        got = sess.run(None, {"x": x.numpy()})[0]
        assert got.shape == (n, N_REGIONS)
        assert np.abs(got - expected).max() < 2e-4

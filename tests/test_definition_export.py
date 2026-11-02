"""The pieces of the hybrid export that need no checkpoint (ADR-053 Amendment 6). The real parity runs in
`python -m ml.src.definition.export_onnx` over the val identities and prints PASS / FAIL against fixed tolerances."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ml.src.definition.data import MEAN, STD  # noqa: E402
from ml.src.definition.export_onnx import (  # noqa: E402
    INT8_AGREEMENT_MIN,
    SingleHead,
    agreement,
    classes,
    unit_tensor,
)
from ml.src.definition.model8 import DefinitionNet8  # noqa: E402
from ml.src.definition.views import HEADS8  # noqa: E402


def test_single_head_equals_the_full_model_head():
    torch.manual_seed(0)
    model = DefinitionNet8(pretrained=False).eval()
    x = torch.rand(3, 3, 64, 48)
    normalised = (x - torch.from_numpy(MEAN).view(1, 3, 1, 1)) / torch.from_numpy(STD).view(1, 3, 1, 1)
    with torch.no_grad():
        full = model(normalised)
        for head in ("definition", "adiposity"):
            assert torch.allclose(SingleHead(model, head).eval()(x), full[:, HEADS8.index(head)], atol=1e-6)


def test_single_head_outputs_exactly_four_logits():
    out = SingleHead(DefinitionNet8(pretrained=False).eval(), "definition").eval()(torch.rand(2, 3, 64, 48))
    assert out.shape == (2, 4)  # no regional output can leave the exported graph


def test_agreement():
    a = np.array([[1.0, 1.0, -1.0, -1.0], [1.0, -1.0, -1.0, -1.0], [1.0, 1.0, 1.0, 1.0]])
    assert classes(a).tolist() == [2, 1, 4]
    same, step = agreement(a, a)
    assert (same, step) == (1.0, 0)
    b = a.copy()
    b[2] = -1.0  # class 4 -> 0: one sample off by 4
    same, step = agreement(a, b)
    assert step == 4 and same == pytest.approx(2 / 3) and same < INT8_AGREEMENT_MIN


def test_front_heads_output_only_definition_and_adiposity():
    from ml.src.definition.export_fallback import FrontHeads

    torch.manual_seed(0)
    model = DefinitionNet8(pretrained=False).eval()
    x = torch.rand(3, 3, 64, 48)
    normalised = (x - torch.from_numpy(MEAN).view(1, 3, 1, 1)) / torch.from_numpy(STD).view(1, 3, 1, 1)
    with torch.no_grad():
        full = model(normalised)
        out = FrontHeads(model).eval()(x)
    assert len(out) == 2  # no regional output can leave the exported graph
    assert torch.allclose(out[0], full[:, HEADS8.index("definition")], atol=1e-6)
    assert torch.allclose(out[1], full[:, HEADS8.index("adiposity")], atol=1e-6)


def test_serving_map():
    from ml.src.definition.export_onnx import SPECS

    served = {name: (run, head) for name, run, _, head in SPECS}
    assert served == {
        "definition": ("definition8_finetune", "definition"),
        "adiposity": ("definition_finetune", "adiposity"),
    }


def _conv_graph():
    from onnx import helper

    def conv(name, weight, bias, group=1):
        return helper.make_node("Conv", ["x", weight, bias], [f"{name}_out"], name=name, group=group)

    nodes = [conv("first", "w0", "b0"), conv("dw", "w1", "onnx::Conv_412", group=16),
           conv("se", "w2", "onnx::Conv_415"),
             conv("last", "w3", "b3")]  # fmt: skip
    return helper.make_model(helper.make_graph(nodes, "g", [], []))


def test_variant_exclusions(monkeypatch):
    pytest.importorskip("onnx")
    from ml.src.definition import export_onnx as ex

    model = _conv_graph()
    assert ex.excluded_nodes(model, "v1") == []
    assert ex.excluded_nodes(model, "v3") == [
        "dw",
        "first",
        "last",
    ]  # depthwise + first + last, not the 1x1 se conv
    monkeypatch.setattr(ex, "OVERFLOW_BIASES", ("onnx::Conv_412", "onnx::Conv_415"))
    assert ex.excluded_nodes(model, "v2") == ["dw", "se"]
    monkeypatch.setattr(ex, "OVERFLOW_BIASES", ("onnx::Conv_412", "onnx::Conv_999"))
    with pytest.raises(SystemExit):
        ex.excluded_nodes(model, "v2")  # a named layer that is not in the graph is an error, not a silent skip


def test_calibration_items_cover_every_sex_and_rig():
    from types import SimpleNamespace

    from ml.src.definition.export_onnx import calibration_items

    train = [
        SimpleNamespace(sex=s, rig=r, id=f"{s}{r}{k}") for s in "mf" for r in ("left", "right") for k in range(200)
    ]
    picked = calibration_items(train, 384)
    assert len(picked) == 384 and len({p.id for p in picked}) == 384
    assert {(p.sex, p.rig) for p in picked} == {(s, r) for s in "mf" for r in ("left", "right")}


def test_drift_rule():
    from ml.src.definition.export_onnx import within_drift

    ref = {"spearman": 0.90, "mae": 0.30}
    assert within_drift({"spearman": 0.895, "mae": 0.305}, ref)
    assert not within_drift({"spearman": 0.88, "mae": 0.30}, ref)
    assert not within_drift({"spearman": 0.90, "mae": 0.32}, ref)


def test_unit_tensor_is_chw_in_unit_range():
    from PIL import Image

    img = Image.new("RGB", (48, 64), (255, 0, 51))
    x = unit_tensor(img)
    assert x.shape == (3, 64, 48) and x.dtype == np.float32
    assert x[0].max() == 1.0 and x[1].max() == 0.0 and x[2].max() == pytest.approx(0.2)

import json
import logging
import time

import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from test_analyzer import png_b64, scan
from test_features import side_figure
from test_rows import figure

from app import security
from app.main import app
from ml.src.preprocess import H, W, from_bodym
from physique_schemas.models import AnalyzeRequest, UserProfile

client = TestClient(app)
PROFILE = dict(height_cm=175, weight_kg=75, age=30, sex="male", activity="moderate",
               level="beginner", goal="cut", days_per_week=4)  # fmt: skip


@pytest.fixture(autouse=True)
def fresh_limiter(monkeypatch):
    monkeypatch.setattr(security, "limiter", security.SlidingWindowLimiter(1000))


@pytest.fixture
def body() -> dict:
    s = scan(from_bodym(figure()), from_bodym(side_figure()))
    return json.loads(
        AnalyzeRequest(goal=s, current=s, profile=UserProfile(**PROFILE)).model_dump_json()
    )


def post(payload: dict | bytes):
    data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    return client.post("/analyze", content=data, headers={"content-type": "application/json"})


def test_valid_request_returns_gap_and_body_comp(body):
    r = post(body)
    assert r.status_code == 200
    out = r.json()
    assert len(out["gap"]["regions"]) == 10 and out["gap"]["analyzer"] == "geometric"
    assert out["body_comp"]["body_fat"]["low"] < out["body_comp"]["body_fat"]["high"]
    assert "silhouette" in out["body_comp"]["body_fat"]["method"]


def test_rgb_upload_is_rejected_and_not_echoed(body):
    rgb = png_b64(np.full((H, W, 3), 120, np.uint8))
    body["current"]["views"]["front"]["mask_png_b64"] = rgb
    r = post(body)
    assert r.status_code == 422
    assert rgb[:60] not in r.text and "1-channel" in r.text


def test_grey_texture_and_wrong_size_are_rejected(body):
    grey = png_b64(np.full((H, W), 128, np.uint8))
    body["goal"]["views"]["left"]["mask_png_b64"] = grey
    assert post(body).status_code == 422
    body["goal"]["views"]["left"]["mask_png_b64"] = png_b64(np.zeros((64, 64), np.uint8))
    assert post(body).status_code == 422


def test_extra_fields_and_missing_views_are_rejected(body):
    assert post(body | {"rgb_image": "AAAA"}).status_code == 422
    del body["current"]["views"]["left"]
    assert post(body).status_code == 422


def test_oversized_payload_is_413_before_parsing():
    assert post(b'{"x": "' + b"A" * (security.MAX_BODY_BYTES + 10) + b'"}').status_code == 413


def test_wrong_content_type_and_bad_json():
    r = client.post("/analyze", content=b"{}", headers={"content-type": "text/plain"})
    assert r.status_code == 415
    assert post(b"{not json").status_code == 422


def test_gap_does_not_depend_on_weight(body):
    light = post(body).json()
    body["profile"]["weight_kg"] = 120
    heavy = post(body).json()
    assert light["gap"] == heavy["gap"]
    assert light["body_comp"]["rmr_kcal"] != heavy["body_comp"]["rmr_kcal"]  # weight matters here


def test_tape_waist_overrides_the_silhouette(body):
    body["profile"]["tape_waist_cm"] = 88
    out = post(body).json()["body_comp"]["body_fat"]
    assert "tape" in out["method"]


def test_rate_limit_returns_429_with_retry_after(monkeypatch, body):
    monkeypatch.setattr(security, "limiter", security.SlidingWindowLimiter(2, 60))
    assert [post(body).status_code for _ in range(3)] == [200, 200, 429]
    assert int(post(body).headers["retry-after"]) >= 1


def test_logs_contain_deltas_but_never_the_mask(body, caplog):
    caplog.set_level(logging.INFO, logger="analyze")
    post(body)
    assert "deltas=" in caplog.text
    assert body["current"]["views"]["front"]["mask_png_b64"][:40] not in caplog.text


def test_limiter_forgets_old_hits():
    lim = security.SlidingWindowLimiter(1, window_s=0.05)
    lim.check("a")
    with pytest.raises(HTTPException) as e:
        lim.check("a")
    assert e.value.status_code == 429
    time.sleep(0.06)
    lim.check("a")  # window passed

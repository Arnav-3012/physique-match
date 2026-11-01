from fastapi.testclient import TestClient

from app.main import app
from app.routes import health


def test_health_without_keys_reports_not_configured(monkeypatch):
    for key in ("GROQ_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    health._cache.clear()

    body = TestClient(app).get("/health").json()

    assert body["status"] == "ok"
    assert body["providers"] == {
        "groq": {"status": "not_configured"},
        "openrouter": {"status": "not_configured"},
    }

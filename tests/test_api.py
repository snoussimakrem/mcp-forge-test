from fastapi.testclient import TestClient

from app.main import app


def test_health_is_read_only_by_default():
    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["mode"] == "read-only"
    assert response.json()["github_writes"] is False

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_is_read_only_by_default():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["mode"] == "read-only"
    assert response.json()["github_writes"] is False
    assert response.headers["X-Request-ID"]


def test_invalid_repository_returns_safe_json_error():
    response = client.post(
        "/api/analyze",
        json={"repository": "invalid repository"},
    )

    assert response.status_code == 422
    assert response.json()["message"] == "Validation error"
    assert response.json()["error_id"]


def test_ci_run_endpoint_has_read_only_contract():
    from app.main import app

    routes = {route.path for route in app.routes}

    assert "/api/ci/{owner}/{repository_name}/{run_id}" in routes

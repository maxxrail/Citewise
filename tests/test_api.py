from __future__ import annotations

from fastapi.testclient import TestClient

from citewise.api.main import app
from citewise.core.db import get_db


def test_health_ok(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "db": "ok"}


def test_health_reports_db_down() -> None:
    class DeadSession:
        def execute(self, *_a, **_k):
            raise RuntimeError("connection refused")

        def close(self) -> None: ...

    app.dependency_overrides[get_db] = lambda: DeadSession()
    try:
        resp = TestClient(app).get("/health")
    finally:
        app.dependency_overrides.clear()
    assert resp.status_code == 503
    assert resp.json() == {"status": "degraded", "db": "error"}


def test_create_and_list_brands(client: TestClient) -> None:
    payload = {
        "name": "Acme",
        "domain": "https://www.Acme.com/",
        "category": "widgets",
        "aliases": ["ACME"],
        "competitors": ["Globex", "Initech"],
    }
    created = client.post("/brands", json=payload)
    assert created.status_code == 201
    body = created.json()
    assert body["id"] > 0
    assert body["domain"] == "acme.com"

    listed = client.get("/brands").json()
    assert [b["name"] for b in listed] == ["Acme"]
    assert listed[0]["competitors"] == ["Globex", "Initech"]


def test_duplicate_brand_conflicts(client: TestClient) -> None:
    payload = {"name": "Acme", "domain": "acme.com", "category": "widgets"}
    assert client.post("/brands", json=payload).status_code == 201
    assert client.post("/brands", json=payload).status_code == 409


def test_invalid_brand_rejected(client: TestClient) -> None:
    resp = client.post("/brands", json={"name": "  ", "domain": "x.com", "category": "c"})
    assert resp.status_code == 422


def test_concurrent_duplicate_maps_integrity_error_to_409(client: TestClient, monkeypatch) -> None:
    from sqlalchemy.exc import IntegrityError

    from citewise.services import brands as svc

    def racing_create(db, payload):
        raise IntegrityError("INSERT", {}, Exception("UNIQUE constraint failed: brands.name"))

    monkeypatch.setattr(svc, "create_brand", racing_create)
    resp = client.post("/brands", json={"name": "Acme", "domain": "acme.com", "category": "c"})
    assert resp.status_code == 409

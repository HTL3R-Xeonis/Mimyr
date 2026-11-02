"""The generator API's /health: whether its config is loaded (needs pytest, fastapi with httpx, pydantic, pyyaml)."""

import importlib

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def api(monkeypatch, tmp_path):
    monkeypatch.setenv("CONFIG_PATH", str(tmp_path / "missing.yml"))
    module = importlib.import_module("topology_api")
    return importlib.reload(module)


def test_health_is_503_while_the_config_cannot_be_loaded(api):
    response = TestClient(api.app).get("/health")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["servers"] == 0
    assert "config not loaded" in body["problems"][0]


def test_health_is_ok_with_a_loaded_config_and_does_not_ask_the_llm_servers(api):
    api.CONFIG = {"ollama_servers": [{"name": "a"}, {"name": "b"}], "max_retries": 3}
    response = TestClient(api.app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "servers": 2, "problems": []}

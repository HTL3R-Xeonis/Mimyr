"""The generator API's /health and /api/suggest-fix (needs pytest, fastapi with httpx, pydantic, pyyaml)."""

import importlib
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import topology_core


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


FIX_BODY = {
    "device_role": "ROUTER",
    "device_image": "Cisco IOSv",
    "context": "ctx",
    "goal": "goal",
}


def test_suggest_fix_500s_without_a_loaded_config(api):
    response = TestClient(api.app).post("/api/suggest-fix", json=FIX_BODY)
    assert response.status_code == 500


def test_suggest_fix_503s_without_a_configured_fix_template(api):
    api.CONFIG = {"ollama_servers": [{"name": "a"}], "fix_prompt_template": None}
    response = TestClient(api.app).post("/api/suggest-fix", json=FIX_BODY)
    assert response.status_code == 503


def test_suggest_fix_400s_for_an_empty_device_role(api):
    api.CONFIG = {"ollama_servers": [{"name": "a"}], "fix_prompt_template": "x"}
    response = TestClient(api.app).post(
        "/api/suggest-fix", json={**FIX_BODY, "device_role": "  "}
    )
    assert response.status_code == 400


def test_suggest_fix_returns_the_parsed_reply(api):
    api.CONFIG = {"ollama_servers": [{"name": "a"}], "fix_prompt_template": "x"}
    reply = {
        "valid": True,
        "warnings": [],
        "commands": ["conf t"],
        "explanation": "why",
        "ollama_server_used": "a",
    }
    with patch.object(api, "suggest_fix", return_value=reply) as fake:
        response = TestClient(api.app).post("/api/suggest-fix", json=FIX_BODY)
    assert response.status_code == 200
    assert response.json() == reply
    fake.assert_called_once_with("ROUTER", "Cisco IOSv", "ctx", "goal", api.CONFIG)


def test_suggest_fix_rejects_model_commands_when_the_status_is_a_string(api):
    api.CONFIG = {"ollama_servers": [{"name": "a"}], "fix_prompt_template": "x"}
    raw = '{"valid":"false","commands":["show version"],"explanation":"why"}'
    with patch.object(
        topology_core, "call_ollama_with_failover", return_value=(raw, "a")
    ):
        response = TestClient(api.app).post("/api/suggest-fix", json=FIX_BODY)
    assert response.status_code == 200
    reply = response.json()
    assert reply["valid"] is False
    assert reply["commands"] == []
    assert reply["explanation"] == ""
    assert "JSON boolean" in reply["warnings"][0]


def test_suggest_fix_maps_a_value_error_to_503(api):
    api.CONFIG = {"ollama_servers": [{"name": "a"}], "fix_prompt_template": "x"}
    with patch.object(api, "suggest_fix", side_effect=ValueError("no template")):
        response = TestClient(api.app).post("/api/suggest-fix", json=FIX_BODY)
    assert response.status_code == 503


def test_suggest_fix_maps_an_unexpected_error_to_502(api):
    api.CONFIG = {"ollama_servers": [{"name": "a"}], "fix_prompt_template": "x"}
    with patch.object(api, "suggest_fix", side_effect=RuntimeError("all servers down")):
        response = TestClient(api.app).post("/api/suggest-fix", json=FIX_BODY)
    assert response.status_code == 502

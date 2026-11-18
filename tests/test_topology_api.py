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


SUMMARY_BODY = {"text": "some log text", "goal": "goal"}


def test_summarize_text_500s_without_a_loaded_config(api):
    response = TestClient(api.app).post("/api/summarize-text", json=SUMMARY_BODY)
    assert response.status_code == 500


def test_summarize_text_503s_without_a_configured_summary_template(api):
    api.CONFIG = {"ollama_servers": [{"name": "a"}], "summary_prompt_template": None}
    response = TestClient(api.app).post("/api/summarize-text", json=SUMMARY_BODY)
    assert response.status_code == 503


def test_summarize_text_400s_for_empty_text(api):
    api.CONFIG = {"ollama_servers": [{"name": "a"}], "summary_prompt_template": "x"}
    response = TestClient(api.app).post(
        "/api/summarize-text", json={**SUMMARY_BODY, "text": "  "}
    )
    assert response.status_code == 400


def test_summarize_text_returns_the_summary(api):
    api.CONFIG = {"ollama_servers": [{"name": "a"}], "summary_prompt_template": "x"}
    reply = {"summary": "a short summary", "ollama_server_used": "a"}
    with patch.object(api, "summarize_text", return_value=reply) as fake:
        response = TestClient(api.app).post("/api/summarize-text", json=SUMMARY_BODY)
    assert response.status_code == 200
    assert response.json() == reply
    fake.assert_called_once_with("some log text", "goal", api.CONFIG)


def test_summarize_text_maps_a_value_error_to_503(api):
    api.CONFIG = {"ollama_servers": [{"name": "a"}], "summary_prompt_template": "x"}
    with patch.object(api, "summarize_text", side_effect=ValueError("no template")):
        response = TestClient(api.app).post("/api/summarize-text", json=SUMMARY_BODY)
    assert response.status_code == 503


def test_summarize_text_maps_an_unexpected_error_to_502(api):
    api.CONFIG = {"ollama_servers": [{"name": "a"}], "summary_prompt_template": "x"}
    with patch.object(
        api, "summarize_text", side_effect=RuntimeError("all servers down")
    ):
        response = TestClient(api.app).post("/api/summarize-text", json=SUMMARY_BODY)
    assert response.status_code == 502


def test_config_snippet_endpoint_requires_service_token(api, monkeypatch):
    monkeypatch.setenv("XEONIS_AI_SNIPPETS_TOKEN", "t" * 32)
    client = TestClient(api.app)
    url = "/api/config-snippets/generate"
    body = {"topic": "vlan", "prompt": "VLAN 30 Lab"}
    api.CONFIG = {"ollama_servers": []}
    with patch.object(
        topology_core,
        "call_ollama_with_failover",
        return_value=('{"vlan_id":30,"name":"Lab"}', "test"),
    ) as model:
        assert client.post(url, json=body).status_code == 403
        model.assert_not_called()
        response = client.post(
            url, json=body, headers={"Authorization": "Bearer " + "t" * 32}
        )
    assert response.status_code == 200
    assert response.json() == {"vlan_id": 30, "name": "Lab"}
    assert "VLAN 30 Lab" in model.call_args.args[0]


def test_config_snippet_rejects_a_model_command_list(api, monkeypatch):
    monkeypatch.setenv("XEONIS_AI_SNIPPETS_TOKEN", "t" * 32)
    api.CONFIG = {"ollama_servers": []}
    with patch.object(
        topology_core,
        "call_ollama_with_failover",
        return_value=('{"commands":["show version"]}', "test"),
    ):
        response = TestClient(api.app).post(
            "/api/config-snippets/generate",
            json={"topic": "vlan", "prompt": "VLAN 30"},
            headers={"Authorization": "Bearer " + "t" * 32},
        )
    assert response.status_code == 422


def test_classroom_mode_disables_the_older_fix_route(api, monkeypatch):
    monkeypatch.setenv("XEONIS_CLASSROOM_MODE", "1")
    assert (
        TestClient(api.app).post("/api/suggest-fix", json=FIX_BODY).status_code == 403
    )


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        (
            '[{"kind":"description","interface":"GigabitEthernet0/1","description":"To R1"}]',
            200,
        ),
        ('["configure terminal", "reload"]', 422),
        ('[{"kind":"ipv4","interface":"GigabitEthernet0/1","address":true}]', 422),
        ("[]", 422),
    ],
)
def test_config_draft_uses_token_and_passes_device_context(
    api, monkeypatch, reply, expected
):
    token = "d" * 32
    monkeypatch.setenv("XEONIS_AI_SNIPPETS_TOKEN", token)
    api.CONFIG = {"ollama_servers": [{"name": "a"}]}
    client = TestClient(api.app)
    body = {
        "prompt": "Describe the port to R1",
        "context": {
            "device": "SW1",
            "role": "SWITCH",
            "image": "Cisco IOSvL2",
            "platform": "iosvl2",
            "ports": ["GigabitEthernet0/1"],
            "links": [
                {
                    "port": "GigabitEthernet0/1",
                    "neighbor": "R1",
                    "neighbor_port": "Gi0/0",
                }
            ],
        },
    }
    with patch.object(
        topology_core, "call_ollama_with_failover", return_value=(reply, "a")
    ) as model:
        assert client.post("/api/config-draft", json=body).status_code == 403
        model.assert_not_called()
        response = client.post(
            "/api/config-draft", json=body, headers={"Authorization": f"Bearer {token}"}
        )
    assert response.status_code == expected
    prompt = model.call_args.args[0]
    assert "Describe the port to R1" in prompt
    assert '"ports": ["GigabitEthernet0/1"]' in prompt
    assert '"neighbor": "R1"' in prompt
    if expected == 200:
        assert response.json()[0]["description"] == "To R1"


@pytest.mark.parametrize(
    ("platform", "reply", "expected"),
    [
        (
            "linux",
            '[{"kind":"ipv4","interface":"ens160","address":"10.0.0.5/24"}]',
            200,
        ),
        ("linux", '[{"kind":"hostname","name":"web-1"}]', 200),
        (
            "windows",
            '[{"kind":"ipv4","interface":"Ethernet0","address":"10.0.0.6/24"}]',
            200,
        ),
        ("routeros", '[{"kind":"hostname","name":"edge1"}]', 200),
        (
            "vpcs",
            '[{"kind":"ipv4","interface":"Ethernet0","address":"10.0.0.8/24"}]',
            200,
        ),
        ("linux", '[{"kind":"vlan","vlan_id":30,"name":"LAB"}]', 422),
        (
            "linux",
            '[{"kind":"description","interface":"ens160","description":"x"}]',
            422,
        ),
        ("iosv", '[{"kind":"hostname","name":"r1"}]', 422),
        ("iosvl2", '[{"kind":"vlan","vlan_id":30,"name":"LAB"}]', 200),
    ],
)
def test_config_draft_accepts_the_kinds_that_fit_the_platform(
    api, monkeypatch, platform, reply, expected
):
    token = "d" * 32
    monkeypatch.setenv("XEONIS_AI_SNIPPETS_TOKEN", token)
    api.CONFIG = {"ollama_servers": [{"name": "a"}]}
    body = {
        "prompt": "configure it",
        "context": {
            "device": "D1",
            "role": "VM",
            "image": "ubuntu",
            "platform": platform,
            "ports": ["ens160", "Ethernet0"],
            "links": [],
        },
    }
    with patch.object(
        topology_core, "call_ollama_with_failover", return_value=(reply, "a")
    ):
        response = TestClient(api.app).post(
            "/api/config-draft", json=body, headers={"Authorization": f"Bearer {token}"}
        )
    assert response.status_code == expected, response.text


def test_config_draft_still_refuses_an_unknown_platform(api, monkeypatch):
    token = "d" * 32
    monkeypatch.setenv("XEONIS_AI_SNIPPETS_TOKEN", token)
    api.CONFIG = {"ollama_servers": [{"name": "a"}]}
    body = {
        "prompt": "x",
        "context": {
            "device": "D1",
            "role": "VM",
            "image": "x",
            "platform": "plan9",
            "ports": [],
            "links": [],
        },
    }
    response = TestClient(api.app).post(
        "/api/config-draft", json=body, headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 422

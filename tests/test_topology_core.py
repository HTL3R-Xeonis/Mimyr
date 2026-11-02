"""
Tests for topology_core's validation and automatic repair. Run with
`python -m pytest tests` from the repository root (pytest is not part of
requirements.txt - the Docker image only ships topology_core.py and
topology_api.py).
"""

import os
import sys
from unittest.mock import patch

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import topology_core as tc

DEVICES = [{"name": "VPCS"}, {"name": "Ethernet switch"}]

REUSED_INTERFACES = """nodes:
  - {image: VPCS, role: PC, names: [PC1, PC2, PC3]}
  - {image: Ethernet switch, role: SWITCH, names: [SW-A, SW-B]}
edges:
  - [PC1, eth0, SW-A, gi0/1]
  - [SW-A, gi0/1, SW-B, gi0/1]
  - [PC2, eth0, SW-A, gi0/2]
  - [PC3, eth0, SW-B, gi0/1]
"""


def _validate(text):
    fixes = []
    parsed, warnings, corrected = tc.validate_topology(text, DEVICES, [], fixes=fixes)
    return parsed, warnings, corrected, fixes


def _interfaces_per_device(parsed):
    used = {}
    for device_a, interface_a, device_b, interface_b in parsed["edges"]:
        used.setdefault(device_a, []).append(interface_a)
        used.setdefault(device_b, []).append(interface_b)
    return used


def test_reused_interfaces_are_renumbered_instead_of_rejected():
    parsed, warnings, corrected, fixes = _validate(REUSED_INTERFACES)

    assert warnings == []
    assert corrected is True
    assert len(fixes) == 2
    for device, interfaces in _interfaces_per_device(parsed).items():
        assert len(interfaces) == len(set(interfaces)), device


def test_first_use_keeps_its_name_and_unique_names_are_untouched():
    parsed, _, _, _ = _validate(REUSED_INTERFACES)

    assert parsed["edges"][0] == ["PC1", "eth0", "SW-A", "gi0/1"]
    assert parsed["edges"][2] == ["PC2", "eth0", "SW-A", "gi0/2"]


def test_the_replacement_skips_names_a_later_edge_already_uses():
    parsed, _, _, _ = _validate(REUSED_INTERFACES)

    # SW-A's reused gi0/1 must not become gi0/2: a later edge owns that one.
    assert parsed["edges"][1][1] == "gi0/3"


def test_a_name_without_a_trailing_number_is_still_reported():
    text = (
        "nodes:\n  - {image: Ethernet switch, role: SWITCH, names: [S1, S2]}\n"
        "edges:\n  - [S1, uplink, S2, e1]\n  - [S1, uplink, S2, e2]\n"
    )
    _, warnings, corrected, fixes = _validate(text)

    assert any("used twice" in w for w in warnings)
    assert corrected is False and fixes == []


def test_two_different_devices_may_use_the_same_interface_name():
    text = (
        "nodes:\n  - {image: VPCS, role: PC, names: [A, B]}\n"
        "edges:\n  - [A, eth0, B, eth0]\n"
    )
    _, warnings, corrected, fixes = _validate(text)

    assert warnings == [] and corrected is False and fixes == []


def test_everything_topologybuilder_rejects_is_flagged_here_too():
    text = """nodes:
  - {image: VPCS, role: SERVER, names: [A, B]}
  - {image: VPCS, role: PC, names: [B]}
  - {image: VPCS, role: PC}
  - {image: VPCS, role: PC, names: [1, 2]}
edges:
  - [A, 0, B, 1]
  - [B, eth1, B, eth2]
"""
    _, warnings, _, _ = _validate(text)
    joined = " | ".join(warnings)

    for expected in (
        "Role 'SERVER'",
        "'B' is used more than once",
        "missing the 'names' key",
        "list of text values",
        "only text values",
        "to itself",
    ):
        assert expected in joined, expected


def test_a_retry_prompt_carries_the_previous_answer_and_the_problems():
    prompt = tc.build_retry_prompt("BASE", "```yaml\nnodes: []\n```", ["problem one"])

    assert prompt.startswith("BASE")
    assert "nodes: []" in prompt and "```" not in prompt
    assert "- problem one" in prompt


def _config():
    return {
        "gns3_api_url": "http://x",
        "template_api_url": None,
        "ollama_servers": [{"name": "fake", "url": "u", "model": "m"}],
        "max_retries": 3,
        "request_timeout": 1,
        "api_timeout": 1,
        "prompt_template": "{example}{gns3_device_list}{vm_device_list}{user_request}",
        "example_topology": "EX",
    }


def test_a_repairable_answer_is_valid_after_one_llm_call():
    prompts = []

    def fake_call(prompt, servers, timeout, skip=None):
        prompts.append(prompt)
        return REUSED_INTERFACES, "fake"

    with (
        patch.object(tc, "fetch_available_devices", return_value=DEVICES),
        patch.object(tc, "call_ollama_with_failover", side_effect=fake_call),
    ):
        result = tc.generate_topology("x", _config())

    assert result["valid"] is True
    assert len(prompts) == 1
    assert len(result["auto_fixes"]) == 2
    edges = yaml.safe_load(result["yaml"])["edges"]
    assert len({(e[0], e[1]) for e in edges} | {(e[2], e[3]) for e in edges}) == 8


def test_an_unrepairable_answer_is_retried_with_feedback():
    broken = "nodes:\n  - {image: VPCS, role: SERVER, names: [A]}\nedges: []\n"
    good = "nodes:\n  - {image: VPCS, role: PC, names: [A]}\nedges: []\n"
    prompts = []

    def fake_call(prompt, servers, timeout, skip=None):
        prompts.append(prompt)
        return (broken if len(prompts) == 1 else good), "fake"

    with (
        patch.object(tc, "fetch_available_devices", return_value=DEVICES),
        patch.object(tc, "call_ollama_with_failover", side_effect=fake_call),
    ):
        result = tc.generate_topology("x", _config())

    assert result["valid"] is True
    assert len(prompts) == 2
    assert "PREVIOUS ANSWER" not in prompts[0]
    assert "PREVIOUS ANSWER" in prompts[1] and "Role 'SERVER'" in prompts[1]


# ---------- OpenAI-compatible servers (free cloud providers) ----------


class _Reply:
    reason = "Error"
    text = ""

    def __init__(self, payload=None, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise tc.requests.HTTPError(f"{self.status_code} error")

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


OPENAI_SERVER = {
    "name": "groq",
    "type": "openai",
    "url": "https://api.example.test/openai/v1/",
    "model": "some-model",
    "api_key_env": "TEST_LLM_KEY",
}


def test_openai_call_sends_bearer_key_from_the_environment(monkeypatch):
    monkeypatch.setenv("TEST_LLM_KEY", "sekret-value")
    reply = _Reply({"choices": [{"message": {"content": "nodes: []"}}]})
    with patch.object(tc.requests, "post", return_value=reply) as post:
        text = tc.call_llm("make a topology", OPENAI_SERVER, 30)
    assert text == "nodes: []"
    assert (
        post.call_args.args[0] == "https://api.example.test/openai/v1/chat/completions"
    )
    assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer sekret-value"
    body = post.call_args.kwargs["json"]
    assert body["model"] == "some-model"
    assert body["messages"] == [{"role": "user", "content": "make a topology"}]


def test_openai_call_without_its_key_is_a_failover_error_not_a_crash(monkeypatch):
    monkeypatch.delenv("TEST_LLM_KEY", raising=False)
    with patch.object(tc.requests, "post") as post:
        try:
            tc.call_llm("x", OPENAI_SERVER, 30)
        except tc.requests.RequestException as exc:
            assert "TEST_LLM_KEY" in str(exc)
        else:
            raise AssertionError("expected RequestException")
    post.assert_not_called()


def test_failover_moves_past_a_rate_limited_cloud_server_to_ollama(monkeypatch, caplog):
    monkeypatch.setenv("TEST_LLM_KEY", "sekret-value")
    servers = [
        OPENAI_SERVER,
        {"name": "local", "type": "ollama", "url": "http://ollama:11434", "model": "m"},
    ]

    def fake_post(url, **kwargs):
        if "chat/completions" in url:
            return _Reply(status=429)
        return _Reply({"response": "from ollama"})

    with (
        patch.object(tc.requests, "post", side_effect=fake_post),
        caplog.at_level("DEBUG"),
    ):
        text, used = tc.call_ollama_with_failover("p", servers, 30)
    assert (text, used) == ("from ollama", "local")
    assert "sekret-value" not in caplog.text


def test_openai_reply_of_unexpected_shape_counts_as_a_failure(monkeypatch):
    monkeypatch.setenv("TEST_LLM_KEY", "k")
    with patch.object(tc.requests, "post", return_value=_Reply({"oops": 1})):
        try:
            tc.call_llm("x", OPENAI_SERVER, 30)
        except tc.requests.RequestException:
            pass
        else:
            raise AssertionError("expected RequestException")


def _write_config(tmp_path, servers_yaml):
    (tmp_path / "prompt_template_file.txt").write_text("{user_request}")
    (tmp_path / "prompt_template.txt").write_text("{user_request}")
    (tmp_path / "example_topology.yml").write_text("nodes: []\nedges: []\n")
    path = tmp_path / "topology_config.yml"
    path.write_text(f"gns3_api_url: http://x\nollama_servers:\n{servers_yaml}")
    return str(path)


def test_load_config_accepts_both_server_types_and_base_url(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    path = _write_config(
        tmp_path,
        "  - {name: local, url: 'http://o:11434', model: m}\n"
        "  - {name: gemini, type: openai, base_url: 'https://g.test/v1beta/openai',"
        " model: gm, api_key_env: GEMINI_API_KEY}\n",
    )
    servers = tc.load_config(path)["ollama_servers"]
    assert servers[0]["type"] == "ollama"
    assert servers[1]["type"] == "openai"
    assert servers[1]["url"] == "https://g.test/v1beta/openai"


def test_load_config_rejects_an_unknown_server_type(tmp_path):
    path = _write_config(
        tmp_path, "  - {name: x, type: banana, url: 'http://x', model: m}\n"
    )
    try:
        tc.load_config(path)
    except ValueError as exc:
        assert "unknown type" in str(exc)
    else:
        raise AssertionError("expected ValueError")


# ---------- Internet uplink (GNS3 NAT node) ----------

NAT_DEVICES = DEVICES + [{"name": "NAT"}, {"name": "Cisco IOSv"}]


def _validate_nat(text):
    fixes = []
    parsed, warnings, _corrected = tc.validate_topology(
        text, NAT_DEVICES, [], fixes=fixes
    )
    return parsed, warnings, fixes


def test_nat_interface_is_renamed_to_its_only_port():
    parsed, warnings, fixes = _validate_nat(
        """nodes:
  - {image: NAT, role: FW, names: [Internet]}
  - {image: Cisco IOSv, role: ROUTER, names: [R1]}
edges:
  - [Internet, gi0/0, R1, gi0/0]
"""
    )
    assert warnings == []
    assert parsed["edges"][0] == ["Internet", "nat0", "R1", "gi0/0"]
    assert any("nat0" in fix for fix in fixes)


def test_nat_with_two_connections_is_reported():
    _, warnings, _ = _validate_nat(
        """nodes:
  - {image: NAT, role: FW, names: [Internet]}
  - {image: Cisco IOSv, role: ROUTER, names: [R1, R2]}
edges:
  - [Internet, nat0, R1, gi0/0]
  - [Internet, nat0, R2, gi0/0]
"""
    )
    assert any("only one port" in w for w in warnings)


def test_correct_nat_edge_needs_no_repair():
    _parsed, warnings, fixes = _validate_nat(
        """nodes:
  - {image: NAT, role: FW, names: [Internet]}
  - {image: Cisco IOSv, role: ROUTER, names: [R1]}
edges:
  - [Internet, nat0, R1, gi0/0]
"""
    )
    assert warnings == [] and fixes == []


def test_http_errors_carry_the_providers_message_but_never_the_key(monkeypatch):
    monkeypatch.setenv("TEST_LLM_KEY", "sekret-value")

    class Rejected(_Reply):
        reason = "Not Found"
        text = (
            '{"error": {"message": "models/foo is not found for API version v1beta"}}'
        )

    with patch.object(tc.requests, "post", return_value=Rejected(status=404)):
        try:
            tc.call_llm("x", OPENAI_SERVER, 30)
        except tc.requests.RequestException as exc:
            assert "404 Not Found" in str(exc)
            assert "models/foo is not found" in str(exc)
            assert "sekret-value" not in str(exc)
        else:
            raise AssertionError("expected RequestException")


def test_extra_body_is_merged_but_never_overrides_model_or_messages(monkeypatch):
    monkeypatch.setenv("TEST_LLM_KEY", "k")
    server = {
        **OPENAI_SERVER,
        "extra_body": {"reasoning_effort": "low", "model": "evil"},
    }
    reply = _Reply({"choices": [{"message": {"content": "x"}}]})
    with patch.object(tc.requests, "post", return_value=reply) as post:
        tc.call_llm("p", server, 30)
    body = post.call_args.kwargs["json"]
    assert body["reasoning_effort"] == "low"
    assert body["model"] == "some-model"

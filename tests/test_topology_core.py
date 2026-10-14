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

import topology_core as tc  # noqa: E402

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

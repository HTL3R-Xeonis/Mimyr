"""The release workflow (test_release_workflow.py): starts only on a pushed version tag, writes only in the release job.

Nothing here runs the workflow; it pins the properties that keep the public repository safe and the assets named as
the appliance's update manifest expects (Mimyr-<version>.tar.gz and Mimyr-<version>.tar.gz.sha256).
"""

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / ".github" / "workflows" / "release.yml"


def _workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text())


def test_starts_only_on_a_pushed_version_tag():
    document = _workflow()
    # YAML 1.1 reads a bare `on` key as True
    triggers = document.get("on", document.get(True))
    assert triggers == {"push": {"tags": ["v*"]}}
    assert "pull_request_target" not in WORKFLOW.read_text()


def test_grants_write_access_to_the_release_job_only():
    document = _workflow()
    assert document["permissions"] == {}
    writers = [
        name
        for name, job in document["jobs"].items()
        if job.get("permissions", {}).get("contents") == "write"
    ]
    assert writers == ["release"]
    assert document["jobs"]["release"]["needs"] == "verify"


def test_pins_every_action_to_a_full_commit():
    steps = [
        step
        for job in _workflow()["jobs"].values()
        for step in job["steps"]
        if "uses" in step
    ]
    assert steps
    for step in steps:
        assert re.fullmatch(r"[\w./-]+@[0-9a-f]{40}", step["uses"]), step["uses"]


def test_names_the_component_like_the_update_manifest():
    assert _workflow()["env"]["COMPONENT"] == "Mimyr"
    assert "git archive" in WORKFLOW.read_text()

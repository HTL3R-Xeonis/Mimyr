"""The release version: the one-line VERSION file is valid SemVer, is shipped in the image and shown by /health."""

import importlib
import re
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
# SemVer 2.0 without build metadata: X.Y.Z, or X.Y.Z-rc.N for a release candidate (the release tag must equal it).
SEMVER = re.compile(
    r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(-(0|[1-9]\d*|\d*[A-Za-z-][0-9A-Za-z-]*)(\.(0|[1-9]\d*|\d*[A-Za-z-][0-9A-Za-z-]*))*)?"
)


def test_the_version_file_is_one_semver_line():
    text = (ROOT / "VERSION").read_text()
    assert text.endswith("\n") and text.count("\n") == 1
    assert SEMVER.fullmatch(text.strip())


def test_the_semver_pattern_takes_release_candidates_and_refuses_the_rest():
    assert SEMVER.fullmatch("1.3.0-rc.1")
    for bad in ("1.3", "v1.3.0", "01.3.0", "1.3.0-", "1.3.0-rc..1", "1.3.0 "):
        assert not SEMVER.fullmatch(bad), bad


def test_the_docker_image_ships_the_version_file():
    dockerfile = (ROOT / "topology.Dockerfile").read_text()
    assert re.search(r"^COPY .*\bVERSION\b", dockerfile, re.MULTILINE)


def test_health_shows_the_version_in_both_states(monkeypatch, tmp_path):
    monkeypatch.setenv("CONFIG_PATH", str(tmp_path / "missing.yml"))
    api = importlib.reload(importlib.import_module("topology_api"))
    expected = (ROOT / "VERSION").read_text().strip()
    assert api.__version__ == expected
    client = TestClient(api.app)
    assert client.get("/health").json()["version"] == expected  # degraded (503) body
    api.CONFIG = {"ollama_servers": [], "max_retries": 1}
    assert client.get("/health").json()["version"] == expected  # ok body

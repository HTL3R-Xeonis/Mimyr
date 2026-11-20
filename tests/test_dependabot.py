"""The Dependabot configuration (.github/dependabot.yml): weekly and grouped, python base images pinned for now.

Nothing here talks to GitHub; it pins the agreements so a later edit cannot silently reopen them: one weekly PR for
patch/minor updates (majors stay separate), GitHub Actions grouped, and no python base-image bump until it is done
deliberately with an image rebuild and acceptance.
"""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / ".github" / "dependabot.yml"


def _updates() -> list[dict]:
    document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert document["version"] == 2
    return document["updates"]


def _of(ecosystem: str) -> list[dict]:
    return [update for update in _updates() if update["package-ecosystem"] == ecosystem]


def _directories(update: dict) -> list[str]:
    return update.get("directories") or [update["directory"]]


def test_every_ecosystem_is_checked_weekly() -> None:
    for update in _updates():
        assert update["schedule"]["interval"] == "weekly"


def test_pip_groups_patch_and_minor_together_and_leaves_majors_separate() -> None:
    updates = _of("pip")
    assert updates
    for update in updates:
        group = update["groups"]["pip-minor-and-patch"]
        assert sorted(group["update-types"]) == ["minor", "patch"]


def test_github_actions_updates_are_grouped() -> None:
    updates = _of("github-actions")
    assert len(updates) == 1
    assert updates[0]["groups"]["github-actions"]["patterns"] == ["*"]


def test_python_base_image_bumps_are_ignored_until_done_deliberately() -> None:
    updates = _of("docker")
    assert updates
    for update in updates:
        (rule,) = [
            rule for rule in update["ignore"] if rule["dependency-name"] == "python"
        ]
        assert sorted(rule["update-types"]) == [
            "version-update:semver-major",
            "version-update:semver-minor",
        ]
        for directory in _directories(update):
            folder = ROOT / directory.lstrip("/")
            assert any(
                "dockerfile" in path.name.lower() for path in folder.iterdir()
            ), directory


def test_every_configured_directory_exists() -> None:
    for update in _updates():
        for directory in _directories(update):
            assert (ROOT / directory.lstrip("/")).is_dir(), directory

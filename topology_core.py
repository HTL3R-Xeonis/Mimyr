"""
Kernlogik für die Topologie-Generierung per Ollama LLM.
Komplett config-getrieben: Prompt-Template, Beispiel-Format, Ollama-Server-
Liste, Retry-Anzahl etc. kommen aus externen Dateien statt aus dem Code.

Wird sowohl vom API-Endpoint (topology_api.py) als auch vom
CLI-Script (generate_topology_cli.py) genutzt.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import requests
import yaml

log = logging.getLogger("topology_core")


# ---------- Config laden ----------
def load_config(config_path: str) -> dict:
    """
    Lädt die Haupt-Config sowie die darin referenzierten Prompt- und
    Beispiel-Dateien (Pfade werden relativ zum Config-Verzeichnis aufgelöst).
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config-Datei nicht gefunden: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    base_dir = path.parent

    prompt_template_file = config.get("prompt_template_file", "prompt_template.txt")
    prompt_path = base_dir / prompt_template_file
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt-Template-Datei nicht gefunden: {prompt_path}")
    config["prompt_template"] = prompt_path.read_text(encoding="utf-8")

    example_file = config.get("example_topology_file", "example_topology.yml")
    example_path = base_dir / example_file
    if not example_path.exists():
        raise FileNotFoundError(f"Beispiel-Topologie-Datei nicht gefunden: {example_path}")
    config["example_topology"] = example_path.read_text(encoding="utf-8")

    config.setdefault("ollama_servers", [])
    config.setdefault("max_retries", 3)
    config.setdefault("request_timeout", 300)
    config.setdefault("api_timeout", 15)

    if not config["ollama_servers"]:
        raise ValueError("Config enthält keine 'ollama_servers'.")
    if not config.get("gns3_api_url"):
        raise ValueError("Config: 'gns3_api_url' fehlt.")

    for i, server in enumerate(config["ollama_servers"]):
        if "url" not in server or "model" not in server:
            raise ValueError(
                f"ollama_servers[{i}] braucht mindestens 'url' und 'model'."
            )
        server.setdefault("name", server["url"])

    return config


# ---------- Geräte-Quellen abrufen ----------
def fetch_available_devices(gns3_api_url: str, timeout: int) -> list[dict]:
    """Holt die aktuell verfügbaren GNS3-Templates von der GNS3-Template-API."""
    url = f"{gns3_api_url.rstrip('/')}/api/templates"
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data.get("templates", [])


def fetch_nfs_templates(template_api_url: str, timeout: int) -> list[dict]:
    """Holt die Liste der VM-Templates (.ova-Dateien) von der NFS-Template-API."""
    if not template_api_url:
        return []
    url = f"{template_api_url.rstrip('/')}/api/templates"
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return data.get("templates", [])
    except requests.RequestException as e:
        log.warning("Konnte NFS-Template-API nicht abfragen (%s): %s", url, e)
        return []


# ---------- Prompt bauen ----------
def build_device_list_text(devices: list[dict]) -> str:
    names = sorted({d.get("name") for d in devices if d.get("name")})
    if not names:
        return "  (keine Geräte gefunden)"
    return "\n".join(f"  - {name}" for name in names)


def build_prompt(
    user_request: str,
    gns3_devices: list[dict],
    vm_templates: list[dict],
    prompt_template: str,
    example_topology: str,
) -> str:
    return prompt_template.format(
        example=example_topology,
        gns3_device_list=build_device_list_text(gns3_devices),
        vm_device_list=build_device_list_text(vm_templates),
        user_request=user_request.strip(),
    )


# ---------- Ollama mit Failover ----------
def call_ollama(prompt: str, url: str, model: str, timeout: int) -> str:
    resp = requests.post(
        f"{url.rstrip('/')}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=(CONNECT_TIMEOUT_SECONDS, timeout),
    )
    resp.raise_for_status()
    return resp.json().get("response", "")


CONNECT_TIMEOUT_SECONDS = 10
"""Wie lange auf den TCP-Connect zu einem Ollama-Server gewartet wird,
getrennt vom (viel größeren) request_timeout für die eigentliche
Generierung. Ohne das hängt ein Server, der Pakete nur verwirft statt die
Verbindung aktiv abzulehnen, den kompletten request_timeout lang - und das
bei jedem Retry erneut, da ollama_servers[0] sonst immer wieder als
Erstes probiert wird."""


def call_ollama_with_failover(
    prompt: str, servers: list[dict], timeout: int, skip: set[str] | None = None
) -> tuple[str, str]:
    """
    Probiert die konfigurierten Ollama-Server der Reihe nach durch, bis
    einer erfolgreich antwortet. Gibt (roher_text, servername) zurück.
    Wirft RuntimeError, falls ALLE Server fehlschlagen.

    `skip` (vom Aufrufer über mehrere Versuche hinweg wiederverwendet)
    markiert Server, die in einem früheren Versuch bereits fehlgeschlagen
    sind, damit sie nicht bei jedem Retry erneut angefragt werden.
    """
    last_error: Optional[Exception] = None
    skip = skip if skip is not None else set()
    candidates = [s for s in servers if s["name"] not in skip]

    if not candidates:
        raise RuntimeError(
            f"Alle {len(servers)} konfigurierten Ollama-Server sind bereits "
            f"in einem früheren Versuch als nicht erreichbar markiert worden."
        )

    for server in candidates:
        name = server["name"]
        url = server["url"]
        model = server["model"]
        log.info("Versuche Ollama-Server '%s' (%s, model=%s) ...", name, url, model)
        try:
            response = call_ollama(prompt, url, model, timeout)
            log.info("Antwort von '%s' erhalten.", name)
            return response, name
        except requests.RequestException as e:
            log.warning("Server '%s' nicht erreichbar/Fehler: %s", name, e)
            last_error = e
            skip.add(name)
            continue

    raise RuntimeError(
        f"Alle {len(servers)} konfigurierten Ollama-Server nicht erreichbar. "
        f"Letzter Fehler: {last_error}"
    )


# ---------- Ausgabe bereinigen & validieren ----------
def strip_code_fences(text: str) -> str:
    """Entfernt ```yaml ... ``` Umrandungen sowie vom Modell hinzugefügte
    Marker-/Trennzeilen (z.B. '--- START ---' oder '=== ENDE ===')."""
    text = text.strip()

    match = re.match(r"^```(?:yaml)?\s*\n(.*?)\n```$", text, re.DOTALL)
    if match:
        text = match.group(1).strip()

    return _strip_marker_lines(text)


def _strip_marker_lines(text: str) -> str:
    lines = text.splitlines()
    marker_pattern = re.compile(r"^[-=]{2,}.*[-=]{2,}$|^[-=]{3,}$")

    start = 0
    while start < len(lines) and (
        not lines[start].strip() or marker_pattern.match(lines[start].strip())
    ):
        start += 1

    end = len(lines)
    while end > start and (
        not lines[end - 1].strip() or marker_pattern.match(lines[end - 1].strip())
    ):
        end -= 1

    return "\n".join(lines[start:end]).strip()


def validate_topology(
    yaml_text: str, gns3_devices: list[dict], vm_templates: list[dict]
) -> tuple[Optional[dict], list[str]]:
    """
    Prüft die generierte YAML auf Gültigkeit und ob alle 'image'-Werte zur
    jeweils passenden Quelle passen:
      - role VM             -> muss in vm_templates (NFS-Share) vorkommen
      - alle anderen Rollen -> muss in gns3_devices (GNS3-Templates) vorkommen
    """
    warnings: list[str] = []
    gns3_names = {d.get("name") for d in gns3_devices if d.get("name")}
    vm_names = {d.get("name") for d in vm_templates if d.get("name")}

    cleaned = strip_code_fences(yaml_text)

    try:
        parsed = yaml.safe_load(cleaned)
    except yaml.YAMLError as e:
        return None, [f"Ausgabe ist kein gültiges YAML: {e}"]

    if not isinstance(parsed, dict) or "nodes" not in parsed:
        return parsed, ["Ausgabe enthält keinen 'nodes'-Schlüssel."]

    node_names: set[str] = set()

    for node in parsed.get("nodes", []):
        if not isinstance(node, dict):
            warnings.append(f"Ungültiger Node-Eintrag: {node!r}")
            continue

        image = node.get("image")
        role = node.get("role")

        if role == "VM":
            if image not in vm_names:
                warnings.append(
                    f"Image '{image}' (role=VM) ist nicht in der Liste der "
                    f"verfügbaren VM-Images auf dem NFS-Share vorhanden."
                )
        else:
            if image not in gns3_names:
                warnings.append(
                    f"Image '{image}' (role={role}) ist nicht in der Liste "
                    f"der verfügbaren GNS3-Templates vorhanden."
                )

        for n in node.get("names") or []:
            node_names.add(n)

    for edge in parsed.get("edges", []):
        if not isinstance(edge, list) or len(edge) != 4:
            warnings.append(f"Ungültiger Edge-Eintrag: {edge!r}")
            continue
        device_a, _, device_b, _ = edge
        if device_a not in node_names:
            warnings.append(f"Edge referenziert unbekanntes Gerät: '{device_a}'")
        if device_b not in node_names:
            warnings.append(f"Edge referenziert unbekanntes Gerät: '{device_b}'")

    return parsed, warnings


# ---------- Hauptablauf ----------
def generate_topology(user_request: str, config: dict) -> dict:
    """
    Führt den kompletten Ablauf aus (inkl. Retries bei ungültiger Ausgabe
    und Failover zwischen mehreren Ollama-Servern) und gibt zurück:
    {
        "yaml": <roher YAML-Text des letzten Versuchs>,
        "parsed": <geparste Struktur oder None>,
        "warnings": [...],
        "valid": bool,
        "devices_considered": <Anzahl GNS3-Geräte>,
        "vm_templates_considered": <Anzahl NFS-VM-Images>,
        "ollama_server_used": <Name des Servers, der zuletzt geantwortet hat>,
        "attempts": [{"attempt": 1, "server": "...", "warnings": [...]}, ...],
    }
    """
    gns3_api_url = config["gns3_api_url"]
    template_api_url = config.get("template_api_url")
    servers = config["ollama_servers"]
    max_retries = config.get("max_retries", 3)
    request_timeout = config.get("request_timeout", 300)
    api_timeout = config.get("api_timeout", 15)

    log.info("Hole verfügbare GNS3-Geräte von %s ...", gns3_api_url)
    gns3_devices = fetch_available_devices(gns3_api_url, api_timeout)
    log.info("%d GNS3-Geräte gefunden.", len(gns3_devices))

    vm_templates: list[dict] = []
    if template_api_url:
        log.info("Hole verfügbare VM-Templates von %s ...", template_api_url)
        vm_templates = fetch_nfs_templates(template_api_url, api_timeout)
        log.info("%d VM-Template(s) gefunden.", len(vm_templates))
    else:
        log.warning(
            "Keine template_api_url konfiguriert - role 'VM' hat keine Bildquelle."
        )

    prompt = build_prompt(
        user_request,
        gns3_devices,
        vm_templates,
        config["prompt_template"],
        config["example_topology"],
    )

    attempts: list[dict] = []
    parsed = None
    warnings: list[str] = []
    raw_response = ""
    used_server: Optional[str] = None
    dead_servers: set[str] = set()

    for attempt in range(1, max_retries + 1):
        log.info("Generierungsversuch %d/%d ...", attempt, max_retries)
        try:
            raw_response, used_server = call_ollama_with_failover(
                prompt, servers, request_timeout, skip=dead_servers
            )
        except RuntimeError as e:
            log.error(str(e))
            warnings = [str(e)]
            attempts.append({"attempt": attempt, "server": None, "warnings": warnings})
            break

        parsed, warnings = validate_topology(raw_response, gns3_devices, vm_templates)
        attempts.append(
            {"attempt": attempt, "server": used_server, "warnings": list(warnings)}
        )

        if parsed is not None and not warnings:
            log.info("Versuch %d erfolgreich, gültige Topologie erhalten.", attempt)
            break

        log.warning(
            "Versuch %d/%d lieferte kein gültiges Ergebnis (%d Warnung(en)).",
            attempt,
            max_retries,
            len(warnings),
        )

    return {
        "yaml": strip_code_fences(raw_response),
        "parsed": parsed,
        "warnings": warnings,
        "valid": parsed is not None and len(warnings) == 0,
        "devices_considered": len(gns3_devices),
        "vm_templates_considered": len(vm_templates),
        "ollama_server_used": used_server,
        "attempts": attempts,
    }

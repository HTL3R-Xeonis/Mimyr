"""
Core logic for topology generation via an Ollama LLM.
Fully config-driven: prompt template, example format, Ollama server list,
retry count etc. all come from external files instead of the code.

Used by both the API endpoint (topology_api.py) and the CLI script
(generate_topology_cli.py).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import requests
import yaml

log = logging.getLogger("topology_core")


# ---------- Load config ----------
def load_config(config_path: str) -> dict:
    """
    Loads the main config plus the prompt and example files it references
    (paths are resolved relative to the config file's own directory).
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    base_dir = path.parent

    prompt_template_file = config.get("prompt_template_file", "prompt_template.txt")
    prompt_path = base_dir / prompt_template_file
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt template file not found: {prompt_path}")
    config["prompt_template"] = prompt_path.read_text(encoding="utf-8")

    example_file = config.get("example_topology_file", "example_topology.yml")
    example_path = base_dir / example_file
    if not example_path.exists():
        raise FileNotFoundError(f"Example topology file not found: {example_path}")
    config["example_topology"] = example_path.read_text(encoding="utf-8")

    config.setdefault("ollama_servers", [])
    config.setdefault("max_retries", 3)
    config.setdefault("request_timeout", 300)
    config.setdefault("api_timeout", 15)

    if not config["ollama_servers"]:
        raise ValueError("Config contains no 'ollama_servers'.")
    if not config.get("gns3_api_url"):
        raise ValueError("Config: 'gns3_api_url' is missing.")

    for i, server in enumerate(config["ollama_servers"]):
        if "url" not in server or "model" not in server:
            raise ValueError(f"ollama_servers[{i}] needs at least 'url' and 'model'.")
        server.setdefault("name", server["url"])

    return config


# ---------- Fetch device sources ----------
def fetch_available_devices(gns3_api_url: str, timeout: int) -> list[dict]:
    """Fetches the currently available GNS3 templates from the GNS3 Template API."""
    url = f"{gns3_api_url.rstrip('/')}/api/templates"
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data.get("templates", [])


def fetch_nfs_templates(template_api_url: str, timeout: int) -> list[dict]:
    """Fetches the list of VM templates (.ova files) from the NFS Template API."""
    if not template_api_url:
        return []
    url = f"{template_api_url.rstrip('/')}/api/templates"
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return data.get("templates", [])
    except requests.RequestException as e:
        log.warning("Could not query NFS Template API (%s): %s", url, e)
        return []


# ---------- Build prompt ----------
def build_device_list_text(devices: list[dict]) -> str:
    names = sorted({d.get("name") for d in devices if d.get("name")})
    if not names:
        return "  (no devices found)"
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


# ---------- Ollama with failover ----------
def call_ollama(prompt: str, url: str, model: str, timeout: int) -> str:
    resp = requests.post(
        f"{url.rstrip('/')}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=(CONNECT_TIMEOUT_SECONDS, timeout),
    )
    resp.raise_for_status()
    return resp.json().get("response", "")


CONNECT_TIMEOUT_SECONDS = 10
"""How long to wait for the TCP connect to an Ollama server, kept separate
from the (much larger) request_timeout for the actual generation. Without
this, a server that only drops packets instead of actively refusing the
connection would block for the full request_timeout - and do so again on
every retry, since ollama_servers[0] is otherwise always tried first."""


def call_ollama_with_failover(
    prompt: str, servers: list[dict], timeout: int, skip: set[str] | None = None
) -> tuple[str, str]:
    """
    Tries the configured Ollama servers in order until one responds
    successfully. Returns (raw_text, server_name). Raises RuntimeError if
    ALL servers fail.

    `skip` (reused by the caller across multiple attempts) marks servers
    that already failed in an earlier attempt, so they aren't queried
    again on every retry.
    """
    last_error: Optional[Exception] = None
    skip = skip if skip is not None else set()
    candidates = [s for s in servers if s["name"] not in skip]

    if not candidates:
        raise RuntimeError(
            f"All {len(servers)} configured Ollama servers were already "
            f"marked unreachable in an earlier attempt."
        )

    for server in candidates:
        name = server["name"]
        url = server["url"]
        model = server["model"]
        log.info("Trying Ollama server '%s' (%s, model=%s) ...", name, url, model)
        try:
            response = call_ollama(prompt, url, model, timeout)
            log.info("Got a response from '%s'.", name)
            return response, name
        except requests.RequestException as e:
            log.warning("Server '%s' unreachable/error: %s", name, e)
            last_error = e
            skip.add(name)
            continue

    raise RuntimeError(
        f"All {len(servers)} configured Ollama servers are unreachable. "
        f"Last error: {last_error}"
    )


# ---------- Clean up & validate output ----------
def strip_code_fences(text: str) -> str:
    """Strips ```yaml ... ``` fences as well as separator/marker lines the
    model added on its own (e.g. '--- START ---' or '=== END ===')."""
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
) -> tuple[Optional[dict], list[str], bool]:
    """
    Checks the generated YAML for validity and whether every 'image' value
    matches its expected source:
      - role VM     -> must appear in vm_templates (NFS share)
      - every other role -> must appear in gns3_devices (GNS3 templates)

    A node whose image only exists as a VM-list image gets its role
    force-corrected to VM regardless of what the model output, instead of
    just being flagged as a warning: TopologyBuilder itself guarantees
    VM-list and GNS3-list image names never collide (it refuses to run
    otherwise), so "image is a known VM image" unambiguously means role
    VM is correct - no need to burn a whole retry over a mistake that has
    exactly one possible fix. The model has repeatedly mislabeled a
    firewall/OS VM image (e.g. OPNsense, Windows 11 Pro Education) with a
    GNS3-only role like FW/PC despite the prompt explicitly saying not to.
    :return: (parsed structure, warnings, whether any role was
        auto-corrected)
    """
    warnings: list[str] = []
    corrected = False
    gns3_names = {d.get("name") for d in gns3_devices if d.get("name")}
    vm_names = {d.get("name") for d in vm_templates if d.get("name")}

    cleaned = strip_code_fences(yaml_text)

    try:
        parsed = yaml.safe_load(cleaned)
    except yaml.YAMLError as e:
        return None, [f"Output is not valid YAML: {e}"], False

    if not isinstance(parsed, dict) or "nodes" not in parsed:
        return parsed, ["Output has no 'nodes' key."], False

    node_names: set[str] = set()

    for node in parsed.get("nodes", []):
        if not isinstance(node, dict):
            warnings.append(f"Invalid node entry: {node!r}")
            continue

        image = node.get("image")
        role = node.get("role")

        if image in vm_names and role != "VM":
            log.info(
                "Auto-correcting role for image '%s': %r -> 'VM' (only "
                "exists as a VM-list image).",
                image,
                role,
            )
            node["role"] = "VM"
            role = "VM"
            corrected = True

        if role == "VM":
            if image not in vm_names:
                warnings.append(
                    f"Image '{image}' (role=VM) is not in the list of "
                    f"available VM images on the NFS share."
                )
        else:
            if image not in gns3_names:
                warnings.append(
                    f"Image '{image}' (role={role}) is not in the list "
                    f"of available GNS3 templates."
                )

        for n in node.get("names") or []:
            node_names.add(n)

    for edge in parsed.get("edges", []):
        if not isinstance(edge, list) or len(edge) != 4:
            warnings.append(f"Invalid edge entry: {edge!r}")
            continue
        device_a, _, device_b, _ = edge
        if device_a not in node_names:
            warnings.append(f"Edge references unknown device: '{device_a}'")
        if device_b not in node_names:
            warnings.append(f"Edge references unknown device: '{device_b}'")

    return parsed, warnings, corrected


# ---------- Main flow ----------
def generate_topology(user_request: str, config: dict) -> dict:
    """
    Runs the full flow (including retries on invalid output and failover
    across multiple Ollama servers) and returns:
    {
        "yaml": <raw YAML text of the last attempt>,
        "parsed": <parsed structure or None>,
        "warnings": [...],
        "valid": bool,
        "devices_considered": <number of GNS3 devices>,
        "vm_templates_considered": <number of NFS VM images>,
        "ollama_server_used": <name of the server that last responded>,
        "attempts": [{"attempt": 1, "server": "...", "warnings": [...]}, ...],
    }
    """
    gns3_api_url = config["gns3_api_url"]
    template_api_url = config.get("template_api_url")
    servers = config["ollama_servers"]
    max_retries = config.get("max_retries", 3)
    request_timeout = config.get("request_timeout", 300)
    api_timeout = config.get("api_timeout", 15)

    log.info("Fetching available GNS3 devices from %s ...", gns3_api_url)
    gns3_devices = fetch_available_devices(gns3_api_url, api_timeout)
    log.info("Found %d GNS3 device(s).", len(gns3_devices))

    vm_templates: list[dict] = []
    if template_api_url:
        log.info("Fetching available VM templates from %s ...", template_api_url)
        vm_templates = fetch_nfs_templates(template_api_url, api_timeout)
        log.info("Found %d VM template(s).", len(vm_templates))
    else:
        log.warning("No template_api_url configured - role 'VM' has no image source.")

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
    corrected = False
    raw_response = ""
    used_server: Optional[str] = None
    dead_servers: set[str] = set()

    for attempt in range(1, max_retries + 1):
        log.info("Generation attempt %d/%d ...", attempt, max_retries)
        try:
            raw_response, used_server = call_ollama_with_failover(
                prompt, servers, request_timeout, skip=dead_servers
            )
        except RuntimeError as e:
            log.error(str(e))
            warnings = [str(e)]
            attempts.append({"attempt": attempt, "server": None, "warnings": warnings})
            break

        parsed, warnings, corrected = validate_topology(
            raw_response, gns3_devices, vm_templates
        )
        attempts.append(
            {"attempt": attempt, "server": used_server, "warnings": list(warnings)}
        )

        if parsed is not None and not warnings:
            log.info("Attempt %d succeeded, got a valid topology.", attempt)
            break

        log.warning(
            "Attempt %d/%d did not produce a valid result (%d warning(s)).",
            attempt,
            max_retries,
            len(warnings),
        )

    valid = parsed is not None and len(warnings) == 0
    # If validate_topology auto-corrected a node's role, the raw model
    # output still has the wrong value - re-serialize the corrected
    # structure instead of handing back text that doesn't match "parsed".
    final_yaml = (
        yaml.safe_dump(parsed, sort_keys=False)
        if valid and corrected
        else strip_code_fences(raw_response)
    )

    return {
        "yaml": final_yaml,
        "parsed": parsed,
        "warnings": warnings,
        "valid": valid,
        "devices_considered": len(gns3_devices),
        "vm_templates_considered": len(vm_templates),
        "ollama_server_used": used_server,
        "attempts": attempts,
    }

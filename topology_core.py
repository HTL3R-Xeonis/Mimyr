"""
Core logic for topology generation via an Ollama LLM.
Fully config-driven: prompt template, example format, Ollama server list,
retry count etc. all come from external files instead of the code.

Used by both the API endpoint (topology_api.py) and the CLI script
(generate_topology_cli.py).
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

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
        server.setdefault("type", "ollama")
        if server["type"] not in LLM_TYPES:
            raise ValueError(
                f"ollama_servers[{i}]: unknown type {server['type']!r} "
                f"(expected one of {', '.join(LLM_TYPES)})."
            )
        if "base_url" in server and "url" not in server:
            server["url"] = server["base_url"]
        if "url" not in server or "model" not in server:
            raise ValueError(f"ollama_servers[{i}] needs at least 'url' and 'model'.")
        server.setdefault("name", server["url"])
        key_env = server.get("api_key_env")
        if key_env and not os.environ.get(key_env):
            # Not fatal: the server is then skipped at call time (failover), so
            # one missing key doesn't take the other servers down with it.
            log.warning(
                "Server '%s': environment variable %s (api_key_env) is not set.",
                server["name"],
                key_env,
            )

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


# ---------- LLM servers with failover ----------
LLM_TYPES = ("ollama", "openai")
"""Server types: "ollama" (a self-hosted Ollama, /api/generate) and "openai"
(anything speaking the OpenAI chat-completions API - Groq, Google Gemini's
OpenAI-compatible endpoint, OpenRouter, Cerebras, Mistral, GitHub Models, a
local vLLM/LM Studio ...). See the README's config section."""


def call_openai_compatible(prompt: str, server: dict, timeout: int) -> str:
    """
    One chat-completion call against an OpenAI-compatible endpoint. The API
    key is read from the environment variable named by the server's
    `api_key_env` - never from the config file - and never logged.
    Every failure (missing key, HTTP error incl. 429 rate limit, a reply of an
    unexpected shape) is raised as requests.RequestException so the failover
    in call_ollama_with_failover moves on to the next server.
    """
    key_env = server.get("api_key_env")
    headers = {}
    if key_env:
        key = os.environ.get(key_env)
        if not key:
            raise requests.RequestException(
                f"environment variable {key_env} is not set"
            )
        headers["Authorization"] = f"Bearer {key}"
    headers.update(server.get("headers") or {})
    body = {
        "model": server["model"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": server.get("temperature", 0.2),
    }
    if server.get("max_tokens"):
        body["max_tokens"] = server["max_tokens"]
    # Provider-specific extras, e.g. {"reasoning_effort": "low"} to make a
    # "thinking" model answer in seconds instead of minutes. Never overrides the
    # model/messages.
    for key, value in (server.get("extra_body") or {}).items():
        body.setdefault(key, value)
    resp = requests.post(
        f"{server['url'].rstrip('/')}/chat/completions",
        json=body,
        headers=headers,
        timeout=(CONNECT_TIMEOUT_SECONDS, timeout),
    )
    try:
        resp.raise_for_status()
    except requests.HTTPError as exc:
        # The provider's own error text ("model not found", "quota exceeded", ...) is
        # what tells the operator what to fix; raise_for_status alone drops it. It
        # never contains the key (that only travels in the request header).
        detail = " ".join((resp.text or "").split())[:300]
        raise requests.RequestException(
            f"{resp.status_code} {resp.reason}: {detail}"
        ) from exc
    try:
        content = resp.json()["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise requests.RequestException(
            "unexpected reply shape from the server (no choices[0].message.content)"
        ) from exc
    return content or ""


def call_llm(prompt: str, server: dict, timeout: int) -> str:
    if server.get("type", "ollama") == "openai":
        return call_openai_compatible(prompt, server, timeout)
    return call_ollama(prompt, server["url"], server["model"], timeout)


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
    last_error: Exception | None = None
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
        log.info(
            "Trying %s server '%s' (%s, model=%s) ...",
            server.get("type", "ollama"),
            name,
            url,
            model,
        )
        try:
            response = call_llm(prompt, server, timeout)
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


VALID_ROLES = ("PC", "VM", "ROUTER", "SWITCH", "FW")
"""The only roles TopologyBuilder accepts - anything else is rejected there."""

_TRAILING_NUMBER = re.compile(r"^(.*?)(\d+)$")


UPLINK_IMAGE = "nat"
UPLINK_INTERFACE = "nat0"


def _uplink_names(parsed: dict) -> set[str]:
    """Names of the Internet-uplink devices (GNS3's built-in NAT node)."""
    names: set[str] = set()
    for group in parsed.get("nodes") or []:
        if (
            isinstance(group, dict)
            and str(group.get("image", "")).strip().lower() == UPLINK_IMAGE
        ):
            for name in group.get("names") or []:
                names.add(str(name))
    return names


def repair_uplink_interfaces(parsed: dict) -> list[str]:
    """
    The NAT node has exactly one port, called nat0 - not gi0/0 like the
    prompt's other devices - so a model following the general interface rule
    names it wrongly. With exactly one edge on the device there is exactly one
    sensible fix. Mutates parsed["edges"] in place.
    :return: a description of each rename done
    """
    uplinks = _uplink_names(parsed)
    fixes: list[str] = []
    edges = [
        e for e in parsed.get("edges") or [] if isinstance(e, list) and len(e) == 4
    ]
    for name in uplinks:
        ends = [(e, i) for e in edges for d, i in ((0, 1), (2, 3)) if str(e[d]) == name]
        if len(ends) != 1:
            continue  # unconnected, or too many - validate_topology reports the latter
        edge, index = ends[0]
        if str(edge[index]) != UPLINK_INTERFACE:
            fixes.append(
                f"Renamed interface '{edge[index]}' on the NAT device '{name}' to "
                f"'{UPLINK_INTERFACE}' (its only port)."
            )
            edge[index] = UPLINK_INTERFACE
    return fixes


def repair_duplicate_interfaces(parsed: dict) -> list[str]:
    """
    Renumbers an interface name a device uses in more than one edge to
    the next free one of the same pattern (gi0/1 -> gi0/2, eth0 ->
    eth1, ...). The model - especially the small 8B one this service runs
    - keeps restarting at gi0/1 for each further link of the same
    switch, and there is exactly one sensible fix for that (the prompt
    even tells the model to do it itself), so repairing it here saves a
    whole LLM round trip of several minutes. Edges are processed in
    order; the first use of a name keeps it. A name without a trailing
    number can't be renumbered and is left alone for
    validate_topology to flag.
    Mutates parsed["edges"] in place.
    :return: a human-readable description of each renumbering done
    """
    edges = parsed.get("edges")
    if not isinstance(edges, list):
        return []

    wellformed = [e for e in edges if isinstance(e, list) and len(e) == 4]
    taken: dict[str, set[str]] = {}
    for device_a, interface_a, device_b, interface_b in wellformed:
        taken.setdefault(str(device_a), set()).add(str(interface_a))
        taken.setdefault(str(device_b), set()).add(str(interface_b))

    seen: dict[str, set[str]] = {}
    fixes: list[str] = []
    for edge in wellformed:
        for device_index, interface_index in ((0, 1), (2, 3)):
            device = str(edge[device_index])
            interface = str(edge[interface_index])
            used = seen.setdefault(device, set())
            if interface in used:
                match = _TRAILING_NUMBER.match(interface)
                if match is None:
                    continue
                prefix, number = match.group(1), int(match.group(2))
                candidate = number + 1
                while f"{prefix}{candidate}" in taken[device] | used:
                    candidate += 1
                replacement = f"{prefix}{candidate}"
                edge[interface_index] = replacement
                taken[device].add(replacement)
                interface = replacement
                fixes.append(
                    f"Renumbered the reused interface '{match.group(0)}' on "
                    f"'{device}' to '{replacement}'."
                )
            used.add(interface)
    return fixes


def validate_topology(
    yaml_text: str,
    gns3_devices: list[dict],
    vm_templates: list[dict],
    fixes: list[str] | None = None,
) -> tuple[dict | None, list[str], bool]:
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
    :param fixes: if given, descriptions of automatic repairs (see
        repair_duplicate_interfaces) are appended to it
    :return: (parsed structure, warnings, whether anything was
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

        for required in ("image", "role", "names"):
            if required not in node:
                warnings.append(f"Node entry is missing the '{required}' key: {node!r}")

        names = node.get("names")
        if names is not None and (
            not isinstance(names, list) or not all(isinstance(n, str) for n in names)
        ):
            warnings.append(f"'names' must be a list of text values: {names!r}")
            names = None
        if role is not None and role not in VALID_ROLES:
            warnings.append(
                f"Role '{role}' is not valid - it must be exactly one of "
                f"{', '.join(VALID_ROLES)}."
            )

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

        for n in names or []:
            if n in node_names:
                warnings.append(f"Device name '{n}' is used more than once.")
            node_names.add(n)

    # Same rule TopologyBuilder's own validation enforces (and rejects the
    # whole topology over): a device may not use the same interface name
    # in two edges. Without checking it here too, this service reported
    # such a topology as valid and the caller only found out afterwards -
    # large topologies with many switches hit this repeatedly (the model
    # restarts at gi0/1 for each new link of the same switch).
    repairs = repair_uplink_interfaces(parsed) + repair_duplicate_interfaces(parsed)
    if repairs:
        corrected = True
        for repair in repairs:
            log.info("Auto-repair: %s", repair)
        if fixes is not None:
            fixes.extend(repairs)

    for uplink in sorted(_uplink_names(parsed)):
        connections = sum(
            1
            for e in parsed.get("edges") or []
            if isinstance(e, list) and len(e) == 4 and uplink in (str(e[0]), str(e[2]))
        )
        if connections > 1:
            warnings.append(
                f"The NAT device '{uplink}' has {connections} connections but only "
                f"one port ({UPLINK_INTERFACE}) - connect it to exactly one router or switch."
            )

    used_interfaces: dict[str, set[str]] = {}

    def _check_interface_unused(device: str, interface: str) -> None:
        used = used_interfaces.setdefault(device, set())
        if interface in used:
            warnings.append(
                f"Interface '{interface}' is used twice on device '{device}' - "
                f"every edge of a device needs its own, distinct interface name."
            )
        used.add(interface)

    for edge in parsed.get("edges", []):
        if not isinstance(edge, list) or len(edge) != 4:
            warnings.append(f"Invalid edge entry: {edge!r}")
            continue
        device_a, interface_a, device_b, interface_b = edge
        if not all(isinstance(v, str) for v in edge):
            warnings.append(
                f"Edge {edge!r} must contain only text values "
                f"[DeviceA, InterfaceA, DeviceB, InterfaceB]."
            )
            continue
        if device_a == device_b:
            warnings.append(f"Edge {edge!r} connects device '{device_a}' to itself.")
        if device_a not in node_names:
            warnings.append(f"Edge references unknown device: '{device_a}'")
        if device_b not in node_names:
            warnings.append(f"Edge references unknown device: '{device_b}'")
        _check_interface_unused(str(device_a), str(interface_a))
        _check_interface_unused(str(device_b), str(interface_b))

    return parsed, warnings, corrected


# ---------- Retry feedback ----------
MAX_FEEDBACK_WARNINGS = 12


def build_retry_prompt(
    base_prompt: str, previous_response: str, warnings: list[str]
) -> str:
    """
    The prompt for a retry: the original prompt plus the model's own
    previous answer and exactly why it was rejected. Resending the bare
    original prompt made the model repeat the same mistake attempt after
    attempt (e.g. restarting a switch's interface numbering) - telling it
    what was wrong lets it fix that instead of rolling the dice again.
    """
    shown = warnings[:MAX_FEEDBACK_WARNINGS]
    problems = "\n".join(f"- {w}" for w in shown)
    if len(warnings) > len(shown):
        problems += f"\n- ... and {len(warnings) - len(shown)} more problem(s)"
    return (
        f"{base_prompt}\n\n"
        f"Your previous answer was rejected:\n"
        f"=== PREVIOUS ANSWER START ===\n{strip_code_fences(previous_response)}\n"
        f"=== PREVIOUS ANSWER END ===\n\n"
        f"Problems found:\n{problems}\n\n"
        f"Fix exactly these problems and answer again with the complete "
        f"corrected YAML, following all rules above. Start directly with 'nodes:'."
    )


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
        "attempts": [{"attempt": 1, "server": "...", "warnings": [...], "auto_fixes": [...]}, ...],
        "auto_fixes": <what was repaired automatically in the attempt returned>,
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
    auto_fixes: list[str] = []
    parsed = None
    warnings: list[str] = []
    corrected = False
    raw_response = ""
    used_server: str | None = None
    dead_servers: set[str] = set()

    for attempt in range(1, max_retries + 1):
        log.info("Generation attempt %d/%d ...", attempt, max_retries)
        attempt_prompt = (
            build_retry_prompt(prompt, raw_response, warnings)
            if attempt > 1 and raw_response and warnings
            else prompt
        )
        try:
            raw_response, used_server = call_ollama_with_failover(
                attempt_prompt, servers, request_timeout, skip=dead_servers
            )
        except RuntimeError as e:
            log.error(str(e))
            warnings = [str(e)]
            attempts.append({"attempt": attempt, "server": None, "warnings": warnings})
            break

        attempt_fixes: list[str] = []
        parsed, warnings, corrected = validate_topology(
            raw_response, gns3_devices, vm_templates, fixes=attempt_fixes
        )
        auto_fixes = attempt_fixes
        attempts.append(
            {
                "attempt": attempt,
                "server": used_server,
                "warnings": list(warnings),
                "auto_fixes": list(attempt_fixes),
            }
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
        "auto_fixes": auto_fixes,
    }

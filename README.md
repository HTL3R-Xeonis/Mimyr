# Topology Generator

Turns a natural-language prompt into a GNS3 topology config YAML, using an
Ollama LLM or a hosted one with a free tier (Gemini, Groq, OpenRouter, ... - any
OpenAI-compatible API). Given a request like "two PCs behind a switch", it queries the
GNS3 and NFS template catalogs for available device images, builds a prompt
around them, and asks an LLM to produce a topology file in the project's
`nodes`/`edges` format - the same format
[TopologyBuilder](https://github.com/HTL3R-Xeonis/TopologyBuilder) deploys
from.

## How it works

1. Fetches the current GNS3 device templates (`gns3_api_url`) and NFS VM
   images (`template_api_url`).
2. Builds a prompt from `config/prompt_template.txt`, filling in those
   device lists plus the example topology (`config/example_topology.yml`)
   and the user's request.
3. Sends the prompt to the first available server in `ollama_servers`
   (Ollama or an OpenAI-compatible hosted one), failing over to the next if a
   server is unreachable, out of quota or has no API key.
4. Validates the LLM's YAML output - the same things
   [TopologyBuilder](https://github.com/HTL3R-Xeonis/TopologyBuilder) would
   reject later, so a topology reported as `valid` really deploys: valid YAML,
   every node has `image`, `role` and `names`, `role` is one of `PC`, `VM`,
   `ROUTER`, `SWITCH`, `FW`, `names` are text and used only once, every
   `image` exists in the matching catalog (GNS3 templates for non-VM roles,
   NFS images for `role: VM`), every edge is four text values referencing
   defined devices, connects two *different* devices, and no device uses an
   interface name in more than one edge.
5. Repairs what has exactly one sensible fix instead of spending another
   (minutes-long) LLM call on it: a role on a VM-list image becomes `VM`, and
   an interface name a device reuses is renumbered to the next free one of
   the same pattern (`gi0/1` -> `gi0/2`; a name without a trailing number
   can't be renumbered and is reported instead). The small 8B model in use
   restarts at `gi0/1` for every further link of a switch, which is why this
   matters. What was repaired is returned as `auto_fixes`.
6. If validation still fails, retries (up to `max_retries` times) - and each
   retry's prompt carries the model's previous answer plus exactly what was
   wrong with it, instead of resending the bare prompt. Servers already
   found unreachable in an earlier attempt of the same request are skipped.

## Running it

```bash
docker compose up -d --build
```

Reads `config/topology_config.yml` (bind-mounted read-only into the
container) on startup. After editing that file, either restart the
container or hot-reload without downtime:

```bash
curl -X POST http://localhost:8002/api/reload-config
```

## Config (`config/topology_config.yml`)

| Key | Purpose |
|---|---|
| `gns3_api_url` | URL of the GNS3 Template API |
| `template_api_url` | URL of the NFS Template API (VM images) |
| `ollama_servers` | List of `{name, url, model}` - tried in order, first success wins |
| `max_retries` | How many times to retry on invalid LLM output |
| `request_timeout` | Per-request timeout (seconds) for one Ollama generation call |
| `api_timeout` | Timeout (seconds) for the GNS3/NFS template lookups |
| `prompt_template_file` / `example_topology_file` | Paths (relative to this config directory) to the prompt template and example topology |

### Internet uplink

If the request asks for internet access and the GNS3 catalog has the built-in `NAT` template, the
prompt (rule 8) makes the model add one `NAT` node named `Internet` (role `FW`) with a single edge
from its only port `nat0`. A wrongly named port is renamed to `nat0` automatically; more than one
connection on the NAT device is reported as a validation problem.

### Hosted LLMs with a free tier

Each `ollama_servers` entry has a `type`: `ollama` (default, as before) or `openai`
for anything speaking the OpenAI chat-completions API. Example entries for Gemini,
Groq and OpenRouter are in `config/topology_config.yml`.

| Key (type `openai`) | Meaning |
|---|---|
| `base_url` | API base, e.g. `https://api.groq.com/openai/v1` (`url` works too) |
| `model` | the provider's model name |
| `api_key_env` | NAME of the environment variable holding the key - never the key itself |
| `temperature`, `max_tokens`, `headers` | optional |

Keys go into a `.env` file next to `docker-compose.yml` (`GROQ_API_KEY=...`, gitignored,
loaded by compose; `docker compose up -d` after changing it - `reload-config` does not
re-read the environment). `GET /api/config` shows each server's `type` and whether its
key is set (`api_key_set`), never the key. A server without key, or answering 429/5xx,
is skipped by the failover. Put hosted servers *after* your own Ollama machines if
the prompts should stay inside the network, *before* them if speed matters more.

## API

| Endpoint | Purpose |
|---|---|
| `GET /` | Service info, whether config loaded successfully |
| `GET /api/config` | Currently loaded config (without the full prompt/example text) |
| `POST /api/reload-config` | Reloads the config files without restarting the container |
| `POST /api/generate-topology` | Body: `{"prompt": "..."}`. Generates a topology; see response shape below |

`POST /api/generate-topology` response:

```json
{
  "yaml": "nodes:\n  - image: ...",
  "parsed": { "nodes": [...], "edges": [...] },
  "warnings": [],
  "valid": true,
  "devices_considered": 12,
  "vm_templates_considered": 4,
  "ollama_server_used": "ollama-2",
  "attempts": [{"attempt": 1, "server": "ollama-2", "warnings": [], "auto_fixes": []}],
  "auto_fixes": ["Renumbered the reused interface 'gi0/1' on 'SW1' to 'gi0/2'."]
}
```

## CLI

For generating a topology file without going through the API:

```bash
python generate_topology_cli.py \
  --config config/topology_config.yml \
  --prompt "Build a small topology with 2 PCs behind a switch" \
  --output topology.yml
```

## Files

- `topology_core.py` - the actual generation/validation/failover logic
- `topology_api.py` - FastAPI wrapper around it
- `generate_topology_cli.py` - standalone CLI wrapper around it
- `tests/test_topology_core.py` - validation, auto-repair and retry-feedback tests
  (`python -m pytest tests`; pytest is not in `requirements.txt`, and the
  Docker image ships only `topology_core.py` and `topology_api.py`)
- `config/topology_config.yml` - server URLs, Ollama server list, timeouts
- `config/prompt_template.txt` - the prompt sent to the LLM
- `config/example_topology.yml` - the example format shown to the LLM

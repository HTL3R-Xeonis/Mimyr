# Topology Generator

Turns a natural-language prompt into a GNS3 topology config YAML, using an
Ollama LLM. Given a request like "two PCs behind a switch", it queries the
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
3. Sends the prompt to the first available Ollama server in
   `ollama_servers`, failing over to the next one if a server is
   unreachable.
4. Validates the LLM's YAML output - is it valid YAML, does every node's
   `image` actually exist in the matching catalog (GNS3 templates for
   non-VM roles, NFS images for `role: VM`), do all edges reference
   defined nodes.
5. If validation fails, retries from the first Ollama server again (up to
   `max_retries` times) - servers already found unreachable in an earlier
   attempt of the same request are skipped on retries.

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
  "attempts": [{"attempt": 1, "server": "ollama-2", "warnings": []}]
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
- `config/topology_config.yml` - server URLs, Ollama server list, timeouts
- `config/prompt_template.txt` - the prompt sent to the LLM
- `config/example_topology.yml` - the example format shown to the LLM

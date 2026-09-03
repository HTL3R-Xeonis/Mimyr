from __future__ import annotations

import logging
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from topology_core import generate_topology, load_config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("topology_api")

app = FastAPI(title="Topology Generator API", version="2.0")

CONFIG_PATH = os.environ.get("CONFIG_PATH", "/config/topology_config.yml")

CONFIG: dict | None = None
CONFIG_ERROR: str | None = None

try:
    CONFIG = load_config(CONFIG_PATH)
    log.info(
        "Config geladen von %s (%d Ollama-Server, max_retries=%d)",
        CONFIG_PATH,
        len(CONFIG["ollama_servers"]),
        CONFIG["max_retries"],
    )
except Exception as e:
    CONFIG_ERROR = str(e)
    log.error("Konnte Config nicht laden: %s", e)


class TopologyRequest(BaseModel):
    prompt: str


@app.get("/")
def root():
    return {
        "name": "Topology Generator API",
        "version": "2.0",
        "config_loaded": CONFIG is not None,
        "config_error": CONFIG_ERROR,
    }


@app.get("/api/config")
def get_config():
    """Zeigt die aktuell geladene Config (ohne Prompt-/Beispiel-Volltext)."""
    if CONFIG is None:
        raise HTTPException(500, f"Config nicht geladen: {CONFIG_ERROR}")

    return {
        "gns3_api_url": CONFIG["gns3_api_url"],
        "template_api_url": CONFIG.get("template_api_url"),
        "ollama_servers": [
            {"name": s["name"], "url": s["url"], "model": s["model"]}
            for s in CONFIG["ollama_servers"]
        ],
        "max_retries": CONFIG["max_retries"],
        "request_timeout": CONFIG["request_timeout"],
    }


@app.post("/api/reload-config")
def reload_config():
    """Lädt die Config-Dateien neu, ohne den Container neu starten zu müssen."""
    global CONFIG, CONFIG_ERROR
    try:
        CONFIG = load_config(CONFIG_PATH)
        CONFIG_ERROR = None
    except Exception as e:
        CONFIG_ERROR = str(e)
        raise HTTPException(500, f"Konnte Config nicht neu laden: {e}")

    return {
        "status": "ok",
        "ollama_servers": [s["name"] for s in CONFIG["ollama_servers"]],
    }


@app.post("/api/generate-topology")
def generate(request: TopologyRequest):
    if CONFIG is None:
        raise HTTPException(
            500, f"Config konnte nicht geladen werden: {CONFIG_ERROR}"
        )

    if not request.prompt.strip():
        raise HTTPException(400, "Feld 'prompt' darf nicht leer sein.")

    try:
        result = generate_topology(request.prompt, CONFIG)
    except Exception as e:
        log.exception("Fehler bei der Topologie-Generierung")
        raise HTTPException(502, f"Fehler bei der Generierung: {e}")

    return result

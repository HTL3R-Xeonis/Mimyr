from __future__ import annotations

import logging
import os
from secrets import compare_digest
from typing import Annotated, Literal

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from attachment_inputs import (
    AttachmentBatch,
    AttachmentError,
    UnsupportedAttachmentError,
    read_attachment,
    screenshot_dimensions,
)
from topology_core import (
    ConfigDraftError,
    generate_config_draft,
    generate_snippet_settings,
    generate_topology,
    load_config,
    suggest_fix,
    summarize_text,
    supports_attachments,
)

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
        "Config loaded from %s (%d Ollama server(s), max_retries=%d)",
        CONFIG_PATH,
        len(CONFIG["ollama_servers"]),
        CONFIG["max_retries"],
    )
except Exception as e:  # noqa: BLE001 - any problem with the config is reported by /health instead of crashing
    CONFIG_ERROR = str(e)
    log.error("Could not load config: %s", e)


class TopologyRequest(BaseModel):
    prompt: str


class FixSuggestionRequest(BaseModel):
    device_role: str
    device_image: str
    context: str
    goal: str


class SummarizeTextRequest(BaseModel):
    text: str
    goal: str = ""


class SnippetSettingsRequest(BaseModel):
    topic: Literal["vlan", "interface_description"]
    prompt: str = Field(min_length=1, max_length=1500)


@app.get("/")
def root():
    return {
        "name": "Topology Generator API",
        "version": "2.0",
        "config_loaded": CONFIG is not None,
        "config_error": CONFIG_ERROR,
    }


@app.get("/health")
def health():
    """Whether the generator can work at all: its config is loaded. The LLM servers are not asked (that can be slow)."""
    if CONFIG is None:
        return JSONResponse(
            {
                "status": "degraded",
                "servers": 0,
                "problems": [f"config not loaded: {CONFIG_ERROR}"],
            },
            status_code=503,
        )
    return {"status": "ok", "servers": len(CONFIG["ollama_servers"]), "problems": []}


@app.get("/api/config")
def get_config():
    """Shows the currently loaded config (without the full prompt/example text)."""
    if CONFIG is None:
        raise HTTPException(500, f"Config not loaded: {CONFIG_ERROR}")

    return {
        "gns3_api_url": CONFIG["gns3_api_url"],
        "template_api_url": CONFIG.get("template_api_url"),
        "ollama_servers": [
            {
                "name": s["name"],
                "type": s.get("type", "ollama"),
                "url": s["url"],
                "model": s["model"],
                "api_key_set": bool(
                    s.get("api_key_env") and os.environ.get(s["api_key_env"])
                ),
            }
            for s in CONFIG["ollama_servers"]
        ],
        "max_retries": CONFIG["max_retries"],
        "request_timeout": CONFIG["request_timeout"],
    }


@app.post("/api/reload-config")
def reload_config():
    """Reloads the config files without needing to restart the container."""
    global CONFIG, CONFIG_ERROR
    try:
        CONFIG = load_config(CONFIG_PATH)
        CONFIG_ERROR = None
    except Exception as e:  # noqa: BLE001 - any config problem is reported to the caller
        CONFIG_ERROR = str(e)
        raise HTTPException(500, f"Could not reload config: {e}")

    return {
        "status": "ok",
        "ollama_servers": [s["name"] for s in CONFIG["ollama_servers"]],
    }


@app.post("/api/generate-topology")
def generate(request: TopologyRequest):
    if CONFIG is None:
        raise HTTPException(500, f"Could not load config: {CONFIG_ERROR}")

    if not request.prompt.strip():
        raise HTTPException(400, "Field 'prompt' must not be empty.")

    try:
        result = generate_topology(request.prompt, CONFIG)
    except Exception as e:
        log.exception("Error during topology generation")
        raise HTTPException(502, f"Error during generation: {e}")

    return result


@app.post("/api/suggest-fix")
def suggest_fix_endpoint(request: FixSuggestionRequest):
    """
    One best-effort device-config suggestion for a failed health check -
    see topology_core.suggest_fix's own docstring. The caller (TopologyOperator's
    app/fix_jobs.py) retries by calling this again; a single call here never retries.
    """
    if os.environ.get("XEONIS_CLASSROOM_MODE") == "1":
        raise HTTPException(
            403, "Health-check fix suggestions are disabled in classroom mode."
        )
    if CONFIG is None:
        raise HTTPException(500, f"Could not load config: {CONFIG_ERROR}")
    if CONFIG.get("fix_prompt_template") is None:
        raise HTTPException(
            503,
            "This Topology Generator has no fix_prompt_template_file configured.",
        )
    if not request.device_role.strip() or not request.device_image.strip():
        raise HTTPException(400, "'device_role' and 'device_image' must not be empty.")

    try:
        return suggest_fix(
            request.device_role,
            request.device_image,
            request.context,
            request.goal,
            CONFIG,
        )
    except ValueError as e:
        raise HTTPException(503, str(e))
    except Exception as e:
        log.exception("Error during fix suggestion")
        raise HTTPException(502, f"Error during fix suggestion: {e}")


@app.post("/api/summarize-text")
def summarize_text_endpoint(request: SummarizeTextRequest):
    """
    One best-effort text summary - see topology_core.summarize_text's own
    docstring. The caller (TopologyOperator's app/log_summary.py, for its
    admin panel's log summary) does not retry; a single call here never
    retries either.
    """
    if CONFIG is None:
        raise HTTPException(500, f"Could not load config: {CONFIG_ERROR}")
    if CONFIG.get("summary_prompt_template") is None:
        raise HTTPException(
            503,
            "This Topology Generator has no summary_prompt_template_file configured.",
        )
    if not request.text.strip():
        raise HTTPException(400, "'text' must not be empty.")
    try:
        return summarize_text(request.text, request.goal, CONFIG)
    except ValueError as e:
        raise HTTPException(503, str(e))
    except Exception as e:
        log.exception("Error during text summarization")
        raise HTTPException(502, f"Error during text summarization: {e}")


def _require_snippet_token(authorization: str | None) -> None:
    token = os.environ.get("XEONIS_AI_SNIPPETS_TOKEN", "")
    if (
        len(token) < 32
        or not token.isascii()
        or not token.isprintable()
        or " " in token
        or not compare_digest(
            (authorization or "").encode(), f"Bearer {token}".encode()
        )
    ):
        raise HTTPException(403, "Snippet generation requires service authorization.")


@app.post("/api/config-snippets/generate")
def generate_config_snippet_settings(
    request: SnippetSettingsRequest,
    authorization: Annotated[str | None, Header()] = None,
) -> dict:
    _require_snippet_token(authorization)
    if CONFIG is None:
        raise HTTPException(503, "Generator configuration is unavailable.")
    if not request.prompt.strip():
        raise HTTPException(422, "Enter a prompt.")
    try:
        return generate_snippet_settings(request.topic, request.prompt.strip(), CONFIG)
    except ValueError as exc:
        raise HTTPException(
            422, "The prompt did not yield settings for the selected exercise."
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(502, "The model service is unavailable.") from exc


class ConfigLink(BaseModel):
    model_config = ConfigDict(extra="forbid")
    port: str = Field(max_length=200)
    neighbor: str = Field(max_length=200)
    neighbor_port: str = Field(max_length=200)


ConfigPort = Annotated[str, Field(max_length=200)]


class ConfigContext(BaseModel):
    model_config = ConfigDict(extra="forbid")
    device: str = Field(max_length=200)
    role: str = Field(max_length=100)
    image: str = Field(max_length=300)
    # Known platforms (iosv, iosvl2, linux, windows, routeros, vpcs) get a real syntax checker,
    # any other name is accepted and only gets the generic line checks.
    platform: str = Field(min_length=1, max_length=40, pattern=r"^[A-Za-z0-9_.+-]+$")
    ports: list[ConfigPort] = Field(max_length=100)
    links: list[ConfigLink] = Field(max_length=100)


class ConfigDraftRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=1, max_length=1500)
    context: ConfigContext


class ConfigDraftResponse(BaseModel):
    commands: list[str]
    warnings: list[str]


@app.post("/api/config-draft")
def generate_config_draft_endpoint(
    request: ConfigDraftRequest,
    authorization: Annotated[str | None, Header()] = None,
) -> ConfigDraftResponse:
    """
    Real device commands for any request, syntax-checked per platform (see
    topology_core.generate_config_draft and config_syntax). Nothing is executed here.
    """
    _require_snippet_token(authorization)
    if CONFIG is None:
        raise HTTPException(503, "Generator configuration is unavailable.")
    prompt = request.prompt.strip()
    if not prompt:
        raise HTTPException(422, "Enter a prompt.")
    context = request.context.model_dump()
    context["platform"] = context["platform"].lower()
    try:
        return generate_config_draft(prompt, context, CONFIG)
    except ConfigDraftError as exc:
        raise HTTPException(422, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(502, "The model service is unavailable.") from exc


MAX_ATTACHMENT_COUNT = 5
MAX_ATTACHMENT_BYTES = 10 * 1024 * 1024


@app.post("/api/generate-topology/attachments")
async def generate_from_attachments(
    files: Annotated[list[UploadFile], File()],
    prompt: Annotated[str, Form()] = "",
    ollama_only: Annotated[bool, Form()] = False,
    match_screenshot_layout: Annotated[bool, Form()] = False,
):
    config = CONFIG
    if config is None:
        raise HTTPException(503, "Topology Generator configuration is unavailable.")
    if not 1 <= len(files) <= MAX_ATTACHMENT_COUNT:
        raise HTTPException(400, "Choose between 1 and 5 files.")
    if not any(
        supports_attachments(server, ollama_only=ollama_only)
        for server in config["ollama_servers"]
    ):
        message = (
            "No Ollama model is configured for file attachments."
            if ollama_only
            else "No Gemini or Ollama model is configured for file attachments."
        )
        raise HTTPException(503, message)

    uploads = []
    remaining = MAX_ATTACHMENT_BYTES
    for upload in files:
        data = await upload.read(remaining + 1)
        remaining -= len(data)
        if remaining < 0:
            raise HTTPException(413, "Attachments may total at most 10 MiB.")
        if not data:
            raise HTTPException(400, "Attachments must not be empty.")
        uploads.append((data, upload.filename or ""))

    try:
        return await run_in_threadpool(
            _generate_from_uploads,
            prompt,
            config,
            uploads,
            ollama_only,
            match_screenshot_layout,
        )
    except UnsupportedAttachmentError as exc:
        raise HTTPException(415, str(exc)) from exc
    except AttachmentError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        log.error("Attachment generation failed (%s)", type(exc).__name__)
        raise HTTPException(
            502, "Topology generation from attachments failed."
        ) from exc


def _generate_from_uploads(
    prompt: str,
    config: dict,
    uploads: list[tuple[bytes, str]],
    ollama_only: bool,
    match_screenshot_layout: bool = False,
):
    screenshot_size = None
    if match_screenshot_layout:
        screenshots = [
            upload
            for upload in uploads
            if upload[1].lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        if len(screenshots) != 1:
            raise AttachmentError(
                "Choose exactly one PNG or JPEG screenshot to match its layout."
            )
        screenshot = screenshots[0]
        screenshot_size = screenshot_dimensions(screenshot[0])
        uploads = [
            screenshot,
            *(upload for upload in uploads if upload is not screenshot),
        ]
    sections = []
    media = []
    for data, filename in uploads:
        text, parts = read_attachment(data, filename)
        if text:
            sections.append(text)
        media.extend(parts)
    attachments = AttachmentBatch(
        media,
        ollama_only=ollama_only,
        native_text="\n\n".join(sections),
        screenshot_size=screenshot_size,
    )
    return generate_topology(prompt, config, attachments=attachments)

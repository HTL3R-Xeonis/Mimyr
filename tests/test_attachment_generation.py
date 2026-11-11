import base64
from io import BytesIO
from unittest.mock import Mock

import pytest
from docx import Document
from fastapi.testclient import TestClient
from PIL import Image

import attachment_inputs
import topology_api
import topology_core as core

PNG_HEADER = b"\x89PNG\r\n\x1a\n"
GEMINI = {
    "type": "openai",
    "url": f"{core.GEMINI_API_BASE}/openai",
    "name": "gemini",
    "model": "gemini-test",
    "api_key_env": "TEST_GEMINI_KEY",
}
OLLAMA = {"url": "http://ollama.test", "name": "ollama", "model": "test-model"}
VALID_TOPOLOGY = "nodes: [{image: VPCS, role: PC, names: [PC1]}]\nedges: []"


@pytest.fixture
def config(monkeypatch):
    config = {
        "gns3_api_url": "http://templates.test",
        "ollama_servers": [GEMINI, OLLAMA],
        "prompt_template": "{user_request}",
        "example_topology": "",
        "max_retries": 1,
    }
    monkeypatch.setattr(topology_api, "CONFIG", config)
    monkeypatch.setattr(core, "fetch_available_devices", lambda *_: [{"name": "VPCS"}])
    return config


def test_file_only_request_sends_both_files_to_gemini_and_validates(
    config, monkeypatch
):
    monkeypatch.setenv("TEST_GEMINI_KEY", "test")
    reply = Mock(status_code=200)
    reply.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"thought": True, "text": "hidden reasoning"},
                        {"text": VALID_TOPOLOGY},
                    ]
                }
            }
        ]
    }
    post = Mock(return_value=reply)
    monkeypatch.setattr(core.requests, "post", post)
    with TestClient(topology_api.app) as client:
        response = client.post(
            "/api/generate-topology/attachments",
            files=[
                ("files", ("network.pdf", b"%PDF-1.7", "application/pdf")),
                ("files", ("plan.png", PNG_HEADER, "application/octet-stream")),
            ],
        )
    assert response.status_code == 200
    assert response.json()["valid"] is True
    parts = post.call_args.kwargs["json"]["contents"][0]["parts"]
    assert [part["inlineData"]["mimeType"] for part in parts[:-1]] == [
        "application/pdf",
        "image/png",
    ]
    assert base64.b64decode(parts[1]["inlineData"]["data"]) == PNG_HEADER
    assert "Derive the topology" in parts[-1]["text"]
    assert post.call_args.args[0].endswith(":generateContent")


def test_ollama_only_renders_pdf_locally_and_excludes_gemini(config, monkeypatch):
    pdf = BytesIO()
    with Image.new("RGB", (96, 72), "white") as picture:
        picture.save(pdf, format="PDF")

    metadata = Mock(status_code=200)
    metadata.json.return_value = {"capabilities": ["completion", "vision"]}
    result = Mock(status_code=200)
    result.json.return_value = {"response": VALID_TOPOLOGY}
    post = Mock(side_effect=[metadata, result])
    monkeypatch.setattr(core.requests, "post", post)
    with TestClient(topology_api.app) as client:
        response = client.post(
            "/api/generate-topology/attachments",
            files=[("files", ("network.pdf", pdf.getvalue(), "application/pdf"))],
            data={"ollama_only": "true", "prompt": "Keep device names."},
        )
    assert response.status_code == 200
    assert response.json()["valid"] is True
    assert [call.args[0] for call in post.call_args_list] == [
        "http://ollama.test/api/show",
        "http://ollama.test/api/generate",
    ]
    payload = post.call_args.kwargs["json"]
    assert "Keep device names." in payload["prompt"]
    assert base64.b64decode(payload["images"][0]).startswith(PNG_HEADER)


def test_ollama_text_model_uses_local_ocr(config, monkeypatch):
    metadata = Mock(status_code=200)
    metadata.json.return_value = {"capabilities": ["completion"]}
    result = Mock(status_code=200)
    result.json.return_value = {"response": VALID_TOPOLOGY}
    post = Mock(side_effect=[metadata, result])
    monkeypatch.setattr(core.requests, "post", post)

    png = BytesIO()
    with Image.new("RGB", (160, 90), "white") as picture:
        picture.save(png, format="PNG")
    ocr = Mock(return_value=Mock(stdout=b"PC1 connected to SW1"))
    monkeypatch.setattr(attachment_inputs.subprocess, "run", ocr)

    with TestClient(topology_api.app) as client:
        response = client.post(
            "/api/generate-topology/attachments",
            files=[("files", ("plan.png", png.getvalue(), "image/png"))],
            data={"ollama_only": "true"},
        )
    assert response.status_code == 200
    assert response.json()["valid"] is True
    ocr.assert_called_once()
    assert ocr.call_args.args[0] == [
        "tesseract",
        "stdin",
        "stdout",
        "-l",
        "eng",
        "--psm",
        "11",
    ]
    payload = post.call_args.kwargs["json"]
    assert "PC1 connected to SW1" in payload["prompt"]
    assert "images" not in payload
    assert post.call_args.args[0] == "http://ollama.test/api/generate"


def test_ollama_text_model_reads_word_paragraphs_and_tables(config, monkeypatch):
    document = Document()
    document.add_paragraph("The topology contains PC1 and SW1.")
    cells = document.add_table(rows=1, cols=2).rows[0].cells
    cells[0].text = "PC1 Ethernet0"
    cells[1].text = "SW1 Gi0/1"
    data = BytesIO()
    document.save(data)
    result = Mock(status_code=200)
    result.json.return_value = {"response": VALID_TOPOLOGY}
    post = Mock(return_value=result)
    monkeypatch.setattr(core.requests, "post", post)

    with TestClient(topology_api.app) as client:
        response = client.post(
            "/api/generate-topology/attachments",
            files=[("files", ("design.docx", data.getvalue()))],
            data={"ollama_only": "true"},
        )
    assert response.status_code == 200
    assert response.json()["valid"] is True
    post.assert_called_once()
    prompt = post.call_args.kwargs["json"]["prompt"]
    assert prompt.count("The topology contains PC1 and SW1.") == 1
    assert "PC1 Ethernet0 | SW1 Gi0/1" in prompt


def test_ollama_ocr_falls_back_on_scanned_pdf(config, monkeypatch):
    with Image.new("RGB", (96, 72), "white") as picture, BytesIO() as pdf:
        picture.save(pdf, format="PDF")
        batch = attachment_inputs.AttachmentBatch(
            [
                {
                    "mimeType": "application/pdf",
                    "data": base64.b64encode(pdf.getvalue()).decode(),
                }
            ]
        )
    ocr = Mock(return_value="PC1 -> SW1")
    monkeypatch.setattr(attachment_inputs, "_recognize_text", ocr)
    assert batch.text_for_ollama == "PC1 -> SW1"
    assert ocr.call_args.args[0].startswith(PNG_HEADER)


@pytest.mark.parametrize(
    ("data", "size_limit", "status"),
    [(b"not an image", 100, 415), (PNG_HEADER, 3, 413), (b"", 100, 400)],
)
def test_rejects_unsupported_oversized_or_empty_file(
    config, monkeypatch, data, size_limit, status
):
    monkeypatch.setattr(topology_api, "MAX_ATTACHMENT_BYTES", size_limit)
    with TestClient(topology_api.app) as client:
        response = client.post(
            "/api/generate-topology/attachments",
            files=[("files", ("plan.png", data, "image/png"))],
        )
    assert response.status_code == status

# Third-party notices

This project is licensed under the GNU GPL v3 (see `LICENSE`). No third-party source code is copied into the repository.
Licenses were read from the installed packages (`pip-licenses`, November 2026) and the Docker base images.

## Python dependencies (`requirements.txt`)

| Package | License | Note |
| --- | --- | --- |
| FastAPI, pydantic, PyYAML, python-docx, tree-sitter, tree-sitter-pwsh | MIT | Permissive. |
| uvicorn | BSD-3-Clause | Permissive. |
| requests, python-multipart | Apache-2.0 | Compatible with GPL-3.0. |
| Pillow | MIT-CMU (HPND) | Permissive. |
| pypdfium2 | BSD-3-Clause / Apache-2.0 (bundles Google's PDFium, with its own third-party licenses) | Keep the notices of the bundled PDFium libraries when redistributing a built image. |

## Docker image

`topology.Dockerfile` starts from `python:3.12-slim` (Python: PSF license; Debian packages with their own licenses) and
installs `tesseract-ocr` from the Debian archive (Apache-2.0). The image is built by the user, not distributed from this
repository.

## LLM services

The generator talks to Ollama servers or hosted LLM APIs (Gemini, Groq, OpenRouter, ...). Using them is subject to
their own terms of use; no API key is part of this repository.

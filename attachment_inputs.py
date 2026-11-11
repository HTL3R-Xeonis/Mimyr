"""Convert uploaded topology sources to text and model-ready media."""

from __future__ import annotations

import subprocess
from base64 import b64decode, b64encode
from contextlib import closing
from dataclasses import dataclass
from functools import cached_property
from io import BytesIO
from threading import Lock
from zipfile import BadZipFile, ZipFile

import pypdfium2 as pdfium
from docx import Document
from docx.opc.constants import RELATIONSHIP_TYPE
from docx.text.paragraph import Paragraph
from lxml.etree import XMLSyntaxError
from PIL import Image

MAX_ATTACHMENT_BYTES = 10 * 1024 * 1024
MAX_VISUAL_PAGES = 12
MAX_TEXT_LENGTH = 40_000
MAX_WORD_EXPANDED_BYTES = 50 * 1024 * 1024
_PDFIUM_LOCK = Lock()
_SIGNATURES = {
    b"%PDF-": "application/pdf",
    b"\x89PNG\r\n\x1a\n": "image/png",
    b"\xff\xd8\xff": "image/jpeg",
}


class AttachmentError(ValueError):
    """An uploaded document could not be used."""


def screenshot_dimensions(data: bytes) -> tuple[int, int]:
    try:
        with Image.open(BytesIO(data)) as image:
            if image.format not in {"PNG", "JPEG"}:
                raise AttachmentError("Screenshot layout requires a PNG or JPEG image.")
            size = image.size
            image.verify()
            return size
    except (OSError, SyntaxError, Image.DecompressionBombError) as exc:
        raise AttachmentError("Could not read the screenshot image.") from exc


class UnsupportedAttachmentError(AttachmentError):
    """The uploaded file type is unsupported."""


def _media_part(data: bytes, mime: str) -> dict[str, str]:
    return {"mimeType": mime, "data": b64encode(data).decode("ascii")}


def _document_text(container):
    for block in container.iter_inner_content():
        if isinstance(block, Paragraph):
            yield block.text
        else:
            for row in block.rows:
                yield " | ".join(" ".join(_document_text(cell)) for cell in row.cells)


def _read_word(data: bytes) -> tuple[str, list[dict[str, str]]]:
    try:
        with ZipFile(BytesIO(data)) as archive:
            if (
                sum(entry.file_size for entry in archive.infolist())
                > MAX_WORD_EXPANDED_BYTES
            ):
                raise AttachmentError("A Word file exceeds the 50 MiB unpacked limit.")
            if "word/document.xml" not in archive.namelist():
                raise AttachmentError("The attachment is not a Word .docx document.")
        document = Document(BytesIO(data))
        text = "\n".join(_document_text(document)).strip()
        media = []
        for relationship in document.part.rels.values():
            if (
                relationship.is_external
                or relationship.reltype != RELATIONSHIP_TYPE.IMAGE
            ):
                continue
            part = relationship.target_part
            if part.content_type in {"image/png", "image/jpeg"}:
                media.append(_media_part(part.blob, part.content_type))
        if not text and not media:
            raise AttachmentError(
                "The Word document has no readable text or supported images."
            )
        return text, media
    except (BadZipFile, KeyError, OSError, XMLSyntaxError) as exc:
        raise AttachmentError("Could not read the Word .docx attachment.") from exc


def read_attachment(data: bytes, filename: str) -> tuple[str, list[dict[str, str]]]:
    for signature, mime_type in _SIGNATURES.items():
        if data.startswith(signature):
            return "", [_media_part(data, mime_type)]
    if filename.lower().endswith(".docx"):
        return _read_word(data)
    if filename.lower().endswith((".txt", ".md", ".csv")):
        try:
            text = data.decode("utf-8-sig").strip()
        except UnicodeError as exc:
            raise AttachmentError("Text attachments must be saved as UTF-8.") from exc
        if not text or "\x00" in text:
            raise AttachmentError(
                "The text attachment is empty or contains binary data."
            )
        return text, []
    raise UnsupportedAttachmentError(
        "Choose PDF, PNG, JPEG, DOCX, TXT, Markdown or CSV files."
    )


def _render_page(page) -> bytes:
    longest_side = max(page.get_size())
    if longest_side <= 0:
        raise AttachmentError("A PDF page has invalid dimensions.")
    with (
        closing(page.render(scale=min(3, 1600 / longest_side))) as bitmap,
        BytesIO() as output,
    ):
        bitmap.to_pil().save(output, format="PNG")
        return output.getvalue()


def _recognize_text(image: bytes) -> str:
    try:
        # Decode within Pillow's size checks before handing the raster to Tesseract.
        with Image.open(BytesIO(image)) as picture, BytesIO() as raster:
            picture.thumbnail((2400, 2400))
            picture.convert("RGB").save(raster, format="PNG")
            result = subprocess.run(
                [
                    "tesseract",
                    "stdin",
                    "stdout",
                    "-l",
                    "eng",
                    "--psm",
                    "11",
                ],
                input=raster.getvalue(),
                capture_output=True,
                timeout=12,
                check=True,
            )
        return result.stdout.decode("utf-8", errors="replace").strip()
    except FileNotFoundError as exc:
        raise AttachmentError(
            "OCR requires Tesseract on the generator server."
        ) from exc
    except (OSError, Image.DecompressionBombError, subprocess.SubprocessError) as exc:
        raise AttachmentError("OCR could not read an attachment image.") from exc


def _pdf_pages(
    data: bytes, remaining: int, *, text_first: bool
) -> list[tuple[str, bytes]]:
    try:
        # PDFium calls must be serialized across request threads.
        with _PDFIUM_LOCK, pdfium.PdfDocument(data) as pdf:
            if not 1 <= len(pdf) <= remaining:
                raise AttachmentError(
                    f"Ollama and OCR accept up to {MAX_VISUAL_PAGES} pages/images per request."
                )
            pages = []
            for page in pdf:
                with closing(page):
                    text = ""
                    if text_first:
                        with closing(page.get_textpage()) as text_page:
                            text = text_page.get_text_bounded().strip()
                    pages.append((text, b"" if text else _render_page(page)))
            return pages
    except pdfium.PdfiumError as exc:
        raise AttachmentError("Could not read a PDF attachment.") from exc


@dataclass
class AttachmentBatch:
    files: list[dict[str, str]]
    ollama_only: bool = False
    native_text: str = ""
    screenshot_size: tuple[int, int] | None = None

    def __post_init__(self):
        max_encoded_bytes = 4 * ((MAX_ATTACHMENT_BYTES + 2) // 3 + MAX_VISUAL_PAGES)
        if sum(len(part["data"]) for part in self.files) > max_encoded_bytes:
            raise AttachmentError("Attachment images may total at most 10 MiB.")
        if len(self.files) > MAX_VISUAL_PAGES:
            raise AttachmentError(
                f"Choose at most {MAX_VISUAL_PAGES} attachment images."
            )
        if len(self.native_text) > MAX_TEXT_LENGTH:
            raise AttachmentError(
                "Attachments contain too much text. Choose the relevant sections."
            )

    def _visual_pages(self, *, text_first: bool) -> list[tuple[str, bytes]]:
        pages: list[tuple[str, bytes]] = []
        for part in self.files:
            data = b64decode(part["data"])
            if part["mimeType"] == "application/pdf":
                pages.extend(
                    _pdf_pages(
                        data, MAX_VISUAL_PAGES - len(pages), text_first=text_first
                    )
                )
            else:
                pages.append(("", data))
            if len(pages) > MAX_VISUAL_PAGES:
                raise AttachmentError(
                    f"Ollama and OCR accept up to {MAX_VISUAL_PAGES} pages/images per request."
                )
        return pages

    @cached_property
    def ollama_images(self) -> list[str]:
        return [
            b64encode(image).decode("ascii")
            for _, image in self._visual_pages(text_first=False)
        ]

    @cached_property
    def text_for_ollama(self) -> str:
        sections = [self.native_text]
        length = len(self.native_text)
        for text, image in self._visual_pages(text_first=True):
            recognized = text or _recognize_text(image)
            length += len(recognized)
            if length > MAX_TEXT_LENGTH:
                raise AttachmentError(
                    "Attachments contain too much text. Choose the relevant sections."
                )
            if recognized:
                sections.append(recognized)
        result = "\n\n".join(section for section in sections if section).strip()
        if not result:
            raise AttachmentError(
                "No text was recognized. Use an image-capable model for this attachment."
            )
        return result

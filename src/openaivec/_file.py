"""File path detection and multimodal content utilities."""

from __future__ import annotations

import base64
import mimetypes
import os
from pathlib import Path

from openai.types.responses.response_input_message_content_list_param import ResponseInputContentParam

__all__: list[str] = []

_FILE_EXTENSIONS = frozenset(
    {
        ".csv",
        ".tsv",
        ".json",
        ".jsonl",
        ".ndjson",
        ".parquet",
        ".pq",
        ".xlsx",
        ".xls",
        ".feather",
        ".arrow",
        ".ipc",
        ".orc",
        ".txt",
        ".log",
    }
)

_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".svg"})

_DOCUMENT_EXTENSIONS = frozenset({".pdf", ".csv", ".txt", ".json", ".md", ".html", ".xml", ".log"})

_SUPPORTED_MEDIA_EXTENSIONS = _IMAGE_EXTENSIONS | _DOCUMENT_EXTENSIONS

_MIME_OVERRIDES: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".svg": "image/svg+xml",
    ".md": "text/markdown",
}


def is_file_path(value: str) -> bool:
    """Check whether *value* looks like a file path.

    Returns ``True`` when any of the following hold:

    * The string points to an existing file on disk.
    * It contains a path separator (``/`` or ``os.sep``).
    * It ends with a recognised data-file extension.

    Args:
        value (str): The string to inspect.

    Returns:
        bool: ``True`` if *value* appears to be a file path.
    """
    if not value or not isinstance(value, str):
        return False

    if os.path.isfile(value):
        return True

    if "/" in value or (os.sep != "/" and os.sep in value):
        return True

    suffix = Path(value).suffix.lower()
    if suffix in _FILE_EXTENSIONS | _SUPPORTED_MEDIA_EXTENSIONS:
        return True

    return False


def is_image_path(value: str) -> bool:
    """Return ``True`` if *value* ends with a recognised image extension."""
    return Path(value).suffix.lower() in _IMAGE_EXTENSIONS


def is_url(value: str) -> bool:
    """Return ``True`` if *value* starts with ``http://`` or ``https://``."""
    return isinstance(value, str) and value.startswith(("http://", "https://"))


def _mime_type(path: str) -> str:
    """Return the MIME type for a file path, with sensible fallbacks."""
    suffix = Path(path).suffix.lower()
    if suffix in _MIME_OVERRIDES:
        return _MIME_OVERRIDES[suffix]
    mime, _ = mimetypes.guess_type(path)
    return mime or "application/octet-stream"


def encode_file_to_data_uri(path: str) -> str:
    """Read a file from disk and return a ``data:`` URI with base64 encoding.

    Args:
        path (str): Path to the file.

    Returns:
        str: A ``data:<mime>;base64,<data>`` URI.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    mime = _mime_type(path)
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{data}"


def build_multimodal_content(value: str) -> list[ResponseInputContentParam]:
    """Convert a string to a list of OpenAI Responses API content parts.

    If *value* is a URL or local file path pointing to a supported media
    type, the returned list contains the appropriate ``input_image`` or
    ``input_file`` content part.  Otherwise a single ``input_text`` part
    is returned.

    Args:
        value (str): Plain text, URL, or local file path.

    Returns:
        list[ResponseInputContentParam]: Content parts suitable for the
        ``content`` field of an OpenAI Responses API user message.
    """
    if is_url(value):
        if is_image_path(value):
            return [{"type": "input_image", "image_url": value, "detail": "auto"}]
        return [{"type": "input_file", "file_url": value, "filename": Path(value).name}]

    if os.path.isfile(value):
        data_uri = encode_file_to_data_uri(value)
        if is_image_path(value):
            return [{"type": "input_image", "image_url": data_uri, "detail": "auto"}]
        return [{"type": "input_file", "file_data": data_uri, "filename": Path(value).name}]

    return [{"type": "input_text", "text": value}]


def is_multimodal_input(value: str) -> bool:
    """Return ``True`` if *value* is a URL or existing file that needs multimodal handling."""
    if is_url(value):
        return True
    if os.path.isfile(value) and Path(value).suffix.lower() in _SUPPORTED_MEDIA_EXTENSIONS:
        return True
    return False

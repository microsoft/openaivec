"""File path detection and multimodal content utilities."""

from __future__ import annotations

import base64
import mimetypes
import os
from pathlib import Path
from urllib.parse import urlparse

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

_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".svg", ".tiff", ".tif"})

_DOCUMENT_EXTENSIONS = frozenset({
    ".pdf",
    ".csv", ".tsv",
    ".txt", ".log", ".md", ".rst",
    ".json", ".jsonl",
    ".html", ".htm", ".xml",
    ".docx", ".doc",
    ".pptx", ".ppt",
    ".xlsx", ".xls",
    ".rtf",
    ".epub",
    ".yaml", ".yml",
    ".tex",
})

_SUPPORTED_MEDIA_EXTENSIONS = _IMAGE_EXTENSIONS | _DOCUMENT_EXTENSIONS

_MIME_OVERRIDES: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".svg": "image/svg+xml",
    ".md": "text/markdown",
    ".rst": "text/x-rst",
    ".yml": "application/x-yaml",
    ".yaml": "application/x-yaml",
    ".tex": "application/x-tex",
    ".jsonl": "application/jsonl",
    ".tsv": "text/tab-separated-values",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".epub": "application/epub+zip",
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
    """Return the MIME type for a file path, with sensible fallbacks.

    Args:
        path (str): File path to inspect.

    Returns:
        str: MIME type string (e.g. ``"image/png"``).
    """
    suffix = Path(path).suffix.lower()
    if suffix in _MIME_OVERRIDES:
        return _MIME_OVERRIDES[suffix]
    mime, _ = mimetypes.guess_type(path)
    return mime or "application/octet-stream"


_MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB


def encode_file_to_data_uri(path: str) -> str:
    """Read a file from disk and return a ``data:`` URI with base64 encoding.

    Args:
        path (str): Path to the file.

    Returns:
        str: A ``data:<mime>;base64,<data>`` URI.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the file exceeds 20 MB.
    """
    size = os.path.getsize(path)
    if size > _MAX_FILE_SIZE_BYTES:
        raise ValueError(f"File {path} is {size / 1024 / 1024:.1f} MB, exceeding the 20 MB limit for base64 encoding.")
    mime = _mime_type(path)
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{data}"


def _url_has_media_extension(url: str) -> bool:
    """Return ``True`` if a URL path ends with a recognised media extension.

    Args:
        url (str): Full URL string.

    Returns:
        bool: ``True`` when the URL path has a known image or document suffix.
    """
    path = urlparse(url).path
    return Path(path).suffix.lower() in _SUPPORTED_MEDIA_EXTENSIONS | _IMAGE_EXTENSIONS


def build_multimodal_content(value: str) -> list[ResponseInputContentParam]:
    """Convert a string to a list of OpenAI Responses API content parts.

    URLs are treated as multimodal only when they end with a recognised media
    extension (e.g. ``.jpg``, ``.pdf``).  URLs without an extension are
    treated as plain text.

    Args:
        value (str): Plain text, URL, or local file path.

    Returns:
        list[ResponseInputContentParam]: Content parts suitable for the
        ``content`` field of an OpenAI Responses API user message.
    """
    if is_url(value) and _url_has_media_extension(value):
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
    """Return ``True`` if *value* needs multimodal handling.

    URLs are multimodal only when they have a recognised media extension.
    Local files are multimodal when they exist and have a supported extension.
    """
    if is_url(value):
        return _url_has_media_extension(value)
    if os.path.isfile(value) and Path(value).suffix.lower() in _SUPPORTED_MEDIA_EXTENSIONS:
        return True
    return False

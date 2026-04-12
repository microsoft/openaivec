"""File path detection and multimodal content utilities.

Provides helpers for detecting file types (image, audio, document) and
building OpenAI Responses API input messages from local files or URLs.

Audio files (``.mp3``, ``.wav``) are encoded as base64 and sent via the
``input_audio`` item type.  Images are inlined as ``data:`` URIs.
Documents are uploaded through the Files API.
"""

from __future__ import annotations

import base64
import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from openai import AsyncOpenAI, OpenAI
from openai.types.responses.response_input_param import ResponseInputParam

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

_AUDIO_EXTENSIONS = frozenset({".mp3", ".wav"})

_AUDIO_FORMAT: dict[str, str] = {
    ".mp3": "mp3",
    ".wav": "wav",
}

_DOCUMENT_EXTENSIONS = frozenset(
    {
        # Responses API ``input_file`` (context stuffing) supported formats.
        # Source: API error response for unsupported file types.
        # Note: .svg is also in _IMAGE_EXTENSIONS and handled as input_image.
        # --- Office / document ---
        ".doc",
        ".docx",
        ".dot",
        ".hwp",
        ".hwpx",
        ".keynote",
        ".odt",
        ".pages",
        ".pdf",
        ".pot",
        ".ppa",
        ".pps",
        ".ppt",
        ".pptx",
        ".pwz",
        ".rtf",
        ".wiz",
        ".xla",
        ".xlb",
        ".xlc",
        ".xlm",
        ".xls",
        ".xlsx",
        ".xlt",
        ".xlw",
        # --- Text / markup ---
        ".csv",
        ".eml",
        ".htm",
        ".html",
        ".ics",
        ".ifb",
        ".json",
        ".ltx",
        ".mail",
        ".markdown",
        ".md",
        ".mht",
        ".mhtml",
        ".nws",
        ".rst",
        ".shtml",
        ".srt",
        ".sty",
        ".svgz",
        ".tex",
        ".text",
        ".txt",
        ".vcf",
        ".vtt",
        ".xml",
        ".yaml",
        ".yml",
        # --- Source code ---
        ".art",
        ".bat",
        ".brf",
        ".c",
        ".cls",
        ".css",
        ".diff",
        ".es",
        ".h",
        ".hs",
        ".java",
        ".js",
        ".ksh",
        ".mjs",
        ".patch",
        ".pl",
        ".pm",
        ".py",
        ".scala",
        ".sh",
    }
)

_SUPPORTED_MEDIA_EXTENSIONS = _IMAGE_EXTENSIONS | _AUDIO_EXTENSIONS | _DOCUMENT_EXTENSIONS

_MIME_OVERRIDES: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".svg": "image/svg+xml",
    ".md": "text/markdown",
    ".rst": "text/x-rst",
    ".yml": "application/x-yaml",
    ".yaml": "application/x-yaml",
    ".tex": "application/x-tex",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}

_MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB


# ---------------------------------------------------------------------------
# Pure utility functions
# ---------------------------------------------------------------------------


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
    return suffix in _FILE_EXTENSIONS | _SUPPORTED_MEDIA_EXTENSIONS


def is_image_path(value: str) -> bool:
    """Return ``True`` if *value* ends with a recognised image extension.

    Args:
        value (str): File path or URL to inspect.

    Returns:
        bool: ``True`` when the path has an image suffix.
    """
    return Path(value).suffix.lower() in _IMAGE_EXTENSIONS


def is_audio_path(value: str) -> bool:
    """Return ``True`` if *value* ends with a recognised audio extension.

    Currently recognised formats are ``.mp3`` and ``.wav``, the only
    formats supported by the OpenAI ``input_audio`` item type.

    Args:
        value (str): File path or URL to inspect.

    Returns:
        bool: ``True`` when the path has an audio suffix.
    """
    return Path(value).suffix.lower() in _AUDIO_EXTENSIONS


def is_url(value: str) -> bool:
    """Return ``True`` if *value* starts with ``http://`` or ``https://``.

    Args:
        value (str): String to inspect.

    Returns:
        bool: ``True`` when the string is an HTTP(S) URL.
    """
    return isinstance(value, str) and value.startswith(("http://", "https://"))


def is_multimodal_input(value: str) -> bool:
    """Return ``True`` if *value* needs multimodal handling.

    URLs are multimodal only when they have a recognised media extension.
    Local files are multimodal when they exist and have a supported extension.

    Args:
        value (str): String to inspect.

    Returns:
        bool: ``True`` when *value* should be routed through multimodal handling.
    """
    if is_url(value):
        return _url_has_media_extension(value)
    if os.path.isfile(value) and Path(value).suffix.lower() in _SUPPORTED_MEDIA_EXTENSIONS:
        return True
    return False


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
    _check_file_size(path)
    mime = _mime_type(path)
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{data}"


def encode_file_to_base64(path: str) -> str:
    """Read a file from disk and return raw base64 string (no ``data:`` prefix).

    Used for audio encoding where the API expects plain base64.

    Args:
        path (str): Path to the file.

    Returns:
        str: Base64-encoded file contents.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the file exceeds 20 MB.
    """
    _check_file_size(path)
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def _check_file_size(path: str) -> None:
    """Raise ``ValueError`` if *path* exceeds the 20 MB limit.

    Args:
        path (str): File path to check.

    Raises:
        ValueError: If the file exceeds 20 MB.
    """
    size = os.path.getsize(path)
    if size > _MAX_FILE_SIZE_BYTES:
        raise ValueError(f"File {path} is {size / 1024 / 1024:.1f} MB, exceeding the 20 MB limit for base64 encoding.")


def _audio_format(path: str) -> str:
    """Return the audio format string for the ``input_audio`` item type.

    Args:
        path (str): File path or URL.

    Returns:
        str: ``"mp3"`` or ``"wav"``.

    Raises:
        ValueError: If the extension is not a supported audio format.
    """
    suffix = Path(path).suffix.lower()
    fmt = _AUDIO_FORMAT.get(suffix)
    if fmt is None:
        raise ValueError(f"Unsupported audio format: {suffix}")
    return fmt


def _url_has_media_extension(url: str) -> bool:
    """Return ``True`` if a URL path ends with a recognised media extension.

    Args:
        url (str): Full URL string.

    Returns:
        bool: ``True`` when the URL path has a known image, audio, or document suffix.
    """
    path = urlparse(url).path
    return Path(path).suffix.lower() in _SUPPORTED_MEDIA_EXTENSIONS


def _reject_audio(path_or_url: str) -> None:
    """Raise ``ValueError`` if *path_or_url* is an audio file.

    The Responses API does not accept ``input_audio`` items.  Audio must
    be processed through the Realtime API or Chat Completions API instead.

    Args:
        path_or_url (str): File path or URL to check.

    Raises:
        ValueError: If the path has an audio extension (``.mp3`` or ``.wav``).
    """
    if is_audio_path(path_or_url):
        ext = Path(path_or_url).suffix.lower()
        raise ValueError(
            f"Audio files ({ext}) are not supported by the Responses API. "
            f"Use the Realtime API or Chat Completions API for audio input."
        )


def _wrap_content_as_message(*content_parts: dict[str, Any]) -> ResponseInputParam:
    """Wrap content parts in a user message for the Responses API.

    Args:
        *content_parts: Content part dicts (``input_text``, ``input_image``, etc.).

    Returns:
        ResponseInputParam: A single-element list containing the user message.
    """
    return [{"role": "user", "content": list(content_parts)}]  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# Multimodal content builder
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MultimodalContentBuilder:
    """Build OpenAI Responses API input messages from strings.

    Converts plain text, URLs, and local file paths into properly formatted
    ``ResponseInputParam`` lists ready for ``responses.create(input=...)``.

    * **Images** — inlined as base64 ``data:`` URIs via ``input_image``.
    * **Documents** (PDF, DOCX, etc.) — uploaded via the Files API and
      referenced by ``file_id``.
    * **Plain text** — wrapped as ``input_text``.

    Note:
        Audio files (``.mp3``, ``.wav``) are **not supported** by the
        Responses API.  Passing an audio file raises ``ValueError``.

    Attributes:
        client (OpenAI): Sync OpenAI client used for Files API uploads.
    """

    client: OpenAI

    def build(self, value: str) -> ResponseInputParam:
        """Convert *value* to Responses API input messages.

        Args:
            value (str): Plain text, URL, or local file path.

        Returns:
            ResponseInputParam: Input messages for ``responses.create(input=...)``.

        Raises:
            ValueError: If *value* is an audio file (not supported by Responses API).
        """
        if is_url(value) and _url_has_media_extension(value):
            return self._build_url(value)

        if os.path.isfile(value):
            return self._build_local_file(value)

        return _wrap_content_as_message({"type": "input_text", "text": value})

    def _build_url(self, url: str) -> ResponseInputParam:
        """Build input messages for a media URL.

        Args:
            url (str): URL with a recognised media extension.

        Returns:
            ResponseInputParam: Input messages for the URL.

        Raises:
            ValueError: If the URL points to an audio file.
        """
        _reject_audio(url)
        if is_image_path(url):
            return _wrap_content_as_message({"type": "input_image", "image_url": url, "detail": "auto"})
        return _wrap_content_as_message({"type": "input_file", "file_url": url})

    def _build_local_file(self, path: str) -> ResponseInputParam:
        """Build input messages for a local file.

        Args:
            path (str): Path to an existing local file.

        Returns:
            ResponseInputParam: Input messages for the file.

        Raises:
            ValueError: If the file is an audio file.
        """
        _reject_audio(path)
        if is_image_path(path):
            data_uri = encode_file_to_data_uri(path)
            return _wrap_content_as_message({"type": "input_image", "image_url": data_uri, "detail": "auto"})
        file_id = self._upload(path)
        return _wrap_content_as_message({"type": "input_file", "file_id": file_id})

    def _upload(self, path: str) -> str:
        """Upload a document via the Files API.

        Args:
            path (str): Local file path.

        Returns:
            str: The ``file_id`` of the uploaded file.
        """
        with open(path, "rb") as f:
            uploaded = self.client.files.create(file=f, purpose="assistants")
        return uploaded.id


@dataclass(frozen=True)
class AsyncMultimodalContentBuilder:
    """Async variant of :class:`MultimodalContentBuilder`.

    Attributes:
        client (AsyncOpenAI): Async OpenAI client used for Files API uploads.
    """

    client: AsyncOpenAI

    async def build(self, value: str) -> ResponseInputParam:
        """Convert *value* to Responses API input messages (async).

        Args:
            value (str): Plain text, URL, or local file path.

        Returns:
            ResponseInputParam: Input messages for ``responses.create(input=...)``.

        Raises:
            ValueError: If *value* is an audio file (not supported by Responses API).
        """
        if is_url(value) and _url_has_media_extension(value):
            return self._build_url(value)

        if os.path.isfile(value):
            return await self._build_local_file(value)

        return _wrap_content_as_message({"type": "input_text", "text": value})

    def _build_url(self, url: str) -> ResponseInputParam:
        """Build input messages for a media URL.

        Args:
            url (str): URL with a recognised media extension.

        Returns:
            ResponseInputParam: Input messages for the URL.

        Raises:
            ValueError: If the URL points to an audio file.
        """
        _reject_audio(url)
        if is_image_path(url):
            return _wrap_content_as_message({"type": "input_image", "image_url": url, "detail": "auto"})
        return _wrap_content_as_message({"type": "input_file", "file_url": url})

    async def _build_local_file(self, path: str) -> ResponseInputParam:
        """Build input messages for a local file (async).

        Args:
            path (str): Path to an existing local file.

        Returns:
            ResponseInputParam: Input messages for the file.

        Raises:
            ValueError: If the file is an audio file.
        """
        _reject_audio(path)
        if is_image_path(path):
            data_uri = encode_file_to_data_uri(path)
            return _wrap_content_as_message({"type": "input_image", "image_url": data_uri, "detail": "auto"})
        file_id = await self._upload(path)
        return _wrap_content_as_message({"type": "input_file", "file_id": file_id})

    async def _upload(self, path: str) -> str:
        """Upload a document via the Files API (async).

        Args:
            path (str): Local file path.

        Returns:
            str: The ``file_id`` of the uploaded file.
        """
        with open(path, "rb") as f:
            uploaded = await self.client.files.create(file=f, purpose="assistants")
        return uploaded.id

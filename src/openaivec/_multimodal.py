"""File path detection and multimodal content utilities.

Provides helpers for detecting file types (image, audio, document) and
building OpenAI Responses API input messages from local files or URLs.

Audio files (``.mp3``, ``.wav``) are rejected because the Responses API
does not accept ``input_audio`` items. Images are inlined as ``data:``
URIs. Documents are uploaded through the Files API.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import mimetypes
import os
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, cast
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

_BINARY_DOCUMENT_EXTENSIONS = frozenset(
    {
        # Office / binary document formats — require Files API upload.
        ".doc",
        ".docx",
        ".dot",
        ".hwp",
        ".hwpx",
        ".keynote",
        ".mht",
        ".mhtml",
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
        ".svgz",
        ".wiz",
        ".xla",
        ".xlb",
        ".xlc",
        ".xlm",
        ".xls",
        ".xlsx",
        ".xlt",
        ".xlw",
    }
)

_TEXT_DOCUMENT_EXTENSIONS = frozenset(
    {
        # Text / markup — readable as strings, eligible for batching.
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
        ".nws",
        ".rst",
        ".shtml",
        ".srt",
        ".sty",
        ".tex",
        ".text",
        ".txt",
        ".vcf",
        ".vtt",
        ".xml",
        ".yaml",
        ".yml",
        # Source code
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

_DOCUMENT_EXTENSIONS = _BINARY_DOCUMENT_EXTENSIONS | _TEXT_DOCUMENT_EXTENSIONS

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
    return _path_suffix(value) in _IMAGE_EXTENSIONS


def is_audio_path(value: str) -> bool:
    """Return ``True`` if *value* ends with a recognised audio extension.

    Currently recognised formats are ``.mp3`` and ``.wav``, the only
    formats supported by the OpenAI ``input_audio`` item type.

    Args:
        value (str): File path or URL to inspect.

    Returns:
        bool: ``True`` when the path has an audio suffix.
    """
    return _path_suffix(value) in _AUDIO_EXTENSIONS


def is_url(value: str) -> bool:
    """Return ``True`` if *value* starts with ``http://`` or ``https://``.

    Args:
        value (str): String to inspect.

    Returns:
        bool: ``True`` when the string is an HTTP(S) URL.
    """
    return isinstance(value, str) and value.startswith(("http://", "https://"))


def _path_suffix(value: str) -> str:
    """Return the extension of a local path or a URL's parsed path."""
    return Path(urlparse(value).path if is_url(value) else value).suffix.lower()


def is_multimodal_input(value: str) -> bool:
    """Return ``True`` if *value* needs multimodal API handling.

    Returns ``True`` for images, audio, and binary documents that cannot
    be inlined as text.  Text-readable files (source code, markup, plain
    text) return ``False`` — they should be read as strings and sent
    through the batched text path for dedup and batching benefits.

    Args:
        value (str): String to inspect.

    Returns:
        bool: ``True`` when *value* requires individual multimodal API calls.
    """
    if is_url(value):
        return _url_has_media_extension(value)
    if os.path.isfile(value):
        suffix = Path(value).suffix.lower()
        if suffix in _IMAGE_EXTENSIONS | _AUDIO_EXTENSIONS | _BINARY_DOCUMENT_EXTENSIONS:
            return True
    return False


def is_readable_text_file(value: str) -> bool:
    """Return ``True`` if *value* is a local text file that can be read as a string.

    Text-readable files are inlined into the batched text path instead of
    going through the Files API, enabling deduplication and batching.

    Args:
        value (str): String to inspect.

    Returns:
        bool: ``True`` when *value* is a local file with a text-readable extension.
    """
    if is_url(value):
        return False
    if not os.path.isfile(value):
        return False
    suffix = Path(value).suffix.lower()
    return suffix in _TEXT_DOCUMENT_EXTENSIONS


def read_text_file(path: str, *, file_bytes: bytes | None = None) -> str:
    """Read a local text file and return its content with a filename header.

    The returned string has the format::

        [File: filename.ext]
        <file content>

    This preserves filename context for the model while allowing the
    content to go through the batched text path.

    Args:
        path (str): Path to a text-readable file.
        file_bytes (bytes | None): Verified contents from a content-versioned cache key,
            if available. Avoids reading a different version of the file.

    Returns:
        str: File content prefixed with a ``[File: ...]`` header.
    """
    name = Path(path).name
    content = (
        file_bytes.decode("utf-8", errors="replace")
        if file_bytes is not None
        else Path(path).read_text(encoding="utf-8", errors="replace")
    )
    return f"[File: {name}]\n{content}"


def _file_version(stat: os.stat_result) -> tuple[int, int, int, int, int]:
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def local_file_cache_key(value: str) -> str:
    """Return a content-versioned key for an existing local file.

    The complete file is hashed, rather than its modification time, so edits
    remain visible even on filesystems with coarse timestamps. Non-files keep
    their original key.
    """
    if is_url(value) or not os.path.isfile(value):
        return value
    digest = hashlib.sha256()
    before = _file_version(os.stat(value))
    with open(value, "rb") as file:
        if _file_version(os.fstat(file.fileno())) != before:
            raise ValueError(f"File changed while computing cache key: {value}")
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
        if _file_version(os.fstat(file.fileno())) != before:
            raise ValueError(f"File changed while computing cache key: {value}")
    if _file_version(os.stat(value)) != before:
        raise ValueError(f"File changed while computing cache key: {value}")
    return f"{value}\0{digest.hexdigest()}"


def read_local_file_for_cache_key(path: str, key: str) -> bytes:
    """Read bytes only if they still match the path's content-versioned cache key.

    Args:
        path (str): Local file path used to compute the key.
        key (str): Result of ``local_file_cache_key(path)``.

    Returns:
        bytes: The exact bytes represented by the cache key.

    Raises:
        ValueError: If the file changed during hashing or reading.
    """
    prefix = f"{path}\0"
    if not key.startswith(prefix):
        raise ValueError(f"Invalid cache key for file: {path}")
    try:
        before = _file_version(os.stat(path))
        with open(path, "rb") as file:
            if _file_version(os.fstat(file.fileno())) != before:
                raise ValueError(f"File changed since computing cache key: {path}")
            contents = file.read()
            if _file_version(os.fstat(file.fileno())) != before:
                raise ValueError(f"File changed while reading cached input: {path}")
        if _file_version(os.stat(path)) != before or hashlib.sha256(contents).hexdigest() != key[len(prefix) :]:
            raise ValueError(f"File changed since computing cache key: {path}")
    except (FileNotFoundError, IsADirectoryError) as error:
        raise ValueError(f"File changed since computing cache key: {path}") from error
    return contents


def _mime_type(path: str) -> str:
    """Return the MIME type for a file path, with sensible fallbacks.

    Args:
        path (str): File path to inspect.

    Returns:
        str: MIME type string (e.g. ``"image/png"``).
    """
    suffix = _path_suffix(path)
    if suffix in _MIME_OVERRIDES:
        return _MIME_OVERRIDES[suffix]
    mime, _ = mimetypes.guess_type(path)
    return mime or "application/octet-stream"


def encode_file_to_data_uri(path: str, *, file_bytes: bytes | None = None) -> str:
    """Read a file from disk and return a ``data:`` URI with base64 encoding.

    Args:
        path (str): Path to the file.
        file_bytes (bytes | None): Verified contents, if already read for a cache key.

    Returns:
        str: A ``data:<mime>;base64,<data>`` URI.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the file exceeds 20 MB.
    """
    if file_bytes is None:
        _check_file_size(path)
        with open(path, "rb") as f:
            file_bytes = f.read()
    _check_file_size(path, size=len(file_bytes))
    mime = _mime_type(path)
    data = base64.b64encode(file_bytes).decode("ascii")
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


def _check_file_size(path: str, *, size: int | None = None) -> None:
    """Raise ``ValueError`` if *path* exceeds the 20 MB limit.

    Args:
        path (str): File path to check.

    Raises:
        ValueError: If the file exceeds 20 MB.
    """
    size = os.path.getsize(path) if size is None else size
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
    suffix = _path_suffix(path)
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
    return _path_suffix(url) in _SUPPORTED_MEDIA_EXTENSIONS


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
        ext = _path_suffix(path_or_url)
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
      referenced by ``file_id``. ``build`` transfers ownership of uploads
      to its caller. Use ``build_with_uploads`` and ``cleanup_uploads`` for
      temporary request-scoped uploads.
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

        Files uploaded by this method are caller-owned. Use
        ``build_with_uploads`` for request-scoped temporary files.

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

    def build_with_uploads(
        self, value: str, *, file_bytes: bytes | None = None
    ) -> tuple[ResponseInputParam, tuple[str, ...]]:
        """Build a request and return IDs of temporary Files API uploads.

        The caller must delete these IDs after the request (including validation
        corrections) completes or fails. No uploads are created for URLs or images.
        Pass verified ``file_bytes`` to send the version identified by the cache key.
        """
        messages = self._build_local_file(value, file_bytes=file_bytes) if file_bytes is not None else self.build(value)
        content = cast(list[dict[str, Any]], messages)[0]["content"]
        uploads = tuple(part["file_id"] for part in content if "file_id" in part)
        return messages, uploads

    def cleanup_uploads(self, file_ids: tuple[str, ...]) -> None:
        """Delete request-scoped files after the final API attempt."""
        for file_id in file_ids:
            self.client.files.delete(file_id)

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

    def _build_local_file(self, path: str, *, file_bytes: bytes | None = None) -> ResponseInputParam:
        """Build input messages for a local file.

        Args:
            path (str): Path to an existing local file.
            file_bytes (bytes | None): Verified contents for the cache key.

        Returns:
            ResponseInputParam: Input messages for the file.

        Raises:
            ValueError: If the file is an audio file.
        """
        _reject_audio(path)
        if is_image_path(path):
            data_uri = encode_file_to_data_uri(path, file_bytes=file_bytes)
            return _wrap_content_as_message({"type": "input_image", "image_url": data_uri, "detail": "auto"})
        file_id = self._upload(path, file_bytes=file_bytes)
        return _wrap_content_as_message({"type": "input_file", "file_id": file_id})

    def _upload(self, path: str, *, file_bytes: bytes | None = None) -> str:
        """Upload a document via the Files API.

        Args:
            path (str): Local file path.
            file_bytes (bytes | None): Verified contents for the cache key.

        Returns:
            str: The ``file_id`` of the uploaded file.
        """
        if file_bytes is None:
            with open(path, "rb") as f:
                uploaded = self.client.files.create(file=f, purpose="assistants")
        else:
            with BytesIO(file_bytes) as f:
                setattr(f, "name", Path(path).name)
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

        Files uploaded by this method are caller-owned. Use
        ``build_with_uploads`` for request-scoped temporary files.

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

    async def build_with_uploads(
        self, value: str, *, file_bytes: bytes | None = None
    ) -> tuple[ResponseInputParam, tuple[str, ...]]:
        """Build a request and return IDs of temporary Files API uploads.

        Pass verified ``file_bytes`` to send the version identified by the cache key.
        The caller owns and must clean up the returned uploads.
        """
        messages = (
            await self._build_local_file(value, file_bytes=file_bytes)
            if file_bytes is not None
            else await self.build(value)
        )
        content = cast(list[dict[str, Any]], messages)[0]["content"]
        uploads = tuple(part["file_id"] for part in content if "file_id" in part)
        return messages, uploads

    async def cleanup_uploads(self, file_ids: tuple[str, ...]) -> None:
        """Delete request-scoped files, even if the caller is cancelled."""
        if not file_ids:
            return

        async def delete_files() -> None:
            for file_id in file_ids:
                await self.client.files.delete(file_id)

        pending = asyncio.create_task(delete_files())
        cancelled = False
        while True:
            try:
                await asyncio.shield(pending)
                break
            except asyncio.CancelledError:
                cancelled = True
                if pending.done():
                    break
        pending.result()
        if cancelled:
            raise asyncio.CancelledError

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

    async def _build_local_file(self, path: str, *, file_bytes: bytes | None = None) -> ResponseInputParam:
        """Build input messages for a local file (async).

        Args:
            path (str): Path to an existing local file.
            file_bytes (bytes | None): Verified contents for the cache key.

        Returns:
            ResponseInputParam: Input messages for the file.

        Raises:
            ValueError: If the file is an audio file.
        """
        _reject_audio(path)
        if is_image_path(path):
            data_uri = encode_file_to_data_uri(path, file_bytes=file_bytes)
            return _wrap_content_as_message({"type": "input_image", "image_url": data_uri, "detail": "auto"})
        file_id = await self._upload(path, file_bytes=file_bytes)
        return _wrap_content_as_message({"type": "input_file", "file_id": file_id})

    async def _upload(self, path: str, *, file_bytes: bytes | None = None) -> str:
        """Upload a document via the Files API (async).

        Args:
            path (str): Local file path.
            file_bytes (bytes | None): Verified contents for the cache key.

        Returns:
            str: The ``file_id`` of the uploaded file.
        """
        if file_bytes is None:
            with open(path, "rb") as f:
                return await self._upload_stream(f)
        with BytesIO(file_bytes) as f:
            setattr(f, "name", Path(path).name)
            return await self._upload_stream(f)

    async def _upload_stream(self, file: Any) -> str:
        pending = asyncio.create_task(self.client.files.create(file=file, purpose="assistants"))
        try:
            return (await asyncio.shield(pending)).id
        except asyncio.CancelledError:
            while True:
                try:
                    uploaded = await asyncio.shield(pending)
                    break
                except asyncio.CancelledError:
                    if pending.done():
                        raise
            await self.cleanup_uploads((uploaded.id,))
            raise

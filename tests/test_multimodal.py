import builtins
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from openaivec import _multimodal
from openaivec._multimodal import (
    AsyncMultimodalContentBuilder,
    MultimodalContentBuilder,
    is_audio_path,
    is_image_path,
    is_multimodal_input,
    local_file_cache_key,
    read_local_file_for_cache_key,
)


@pytest.mark.parametrize("builder_type", [MultimodalContentBuilder, AsyncMultimodalContentBuilder])
@pytest.mark.parametrize(
    ("path", "expected_type", "expected_field"),
    [
        ("photo.png", "input_image", "image_url"),
        ("document.pdf", "input_file", "file_url"),
        ("data.csv", "input_file", "file_url"),
    ],
)
@pytest.mark.asyncio
async def test_signed_url_media_routing(builder_type, path, expected_type, expected_field):
    url = f"https://cdn.example.com/{path}?sig=abc&expires=10"
    assert is_multimodal_input(url)
    builder = builder_type(client=MagicMock())
    if builder_type is AsyncMultimodalContentBuilder:
        messages = await builder.build(url)
    else:
        messages = builder.build(url)
    part = messages[0]["content"][0]
    assert part["type"] == expected_type
    assert part[expected_field] == url


@pytest.mark.parametrize("builder_type", [MultimodalContentBuilder, AsyncMultimodalContentBuilder])
@pytest.mark.asyncio
async def test_signed_audio_url_is_rejected(builder_type):
    url = "https://cdn.example.com/voice.mp3?sig=abc"
    assert is_audio_path(url)
    assert not is_image_path(url)
    assert is_multimodal_input(url)
    builder = builder_type(client=MagicMock())
    with pytest.raises(ValueError, match=r"Audio files \(\.mp3\)"):
        if builder_type is AsyncMultimodalContentBuilder:
            await builder.build(url)
        else:
            builder.build(url)


@pytest.mark.parametrize("builder_type", [MultimodalContentBuilder, AsyncMultimodalContentBuilder])
@pytest.mark.asyncio
async def test_extensionless_url_remains_text(builder_type):
    url = "https://cdn.example.com/download?sig=abc"
    assert not is_multimodal_input(url)
    assert not is_multimodal_input("https://cdn.example.com/download?next=photo.png")
    builder = builder_type(client=MagicMock())
    messages = await builder.build(url) if builder_type is AsyncMultimodalContentBuilder else builder.build(url)
    assert messages[0]["content"][0] == {"type": "input_text", "text": url}


def test_local_file_cache_key_tracks_bytes_without_mtime_dependency(tmp_path):
    path = tmp_path / "document.pdf"
    path.write_bytes(b"first")
    first = local_file_cache_key(str(path))
    assert local_file_cache_key(str(path)) == first
    timestamp = path.stat().st_mtime_ns
    path.write_bytes(b"other")
    os.utime(path, ns=(timestamp, timestamp))
    assert local_file_cache_key(str(path)) != first
    assert local_file_cache_key("https://example.com/file.pdf") == "https://example.com/file.pdf"


def test_file_bytes_must_match_cache_key_even_when_mtime_is_unchanged(tmp_path):
    path = tmp_path / "document.pdf"
    path.write_bytes(b"first")
    key = local_file_cache_key(str(path))
    assert read_local_file_for_cache_key(str(path), key) == b"first"

    timestamp = path.stat().st_mtime_ns
    path.write_bytes(b"other")
    os.utime(path, ns=(timestamp, timestamp))
    with pytest.raises(ValueError, match="changed"):
        read_local_file_for_cache_key(str(path), key)


def test_file_removed_after_cache_key_fails_closed(tmp_path):
    path = tmp_path / "document.pdf"
    path.write_bytes(b"first")
    key = local_file_cache_key(str(path))
    path.unlink()
    with pytest.raises(ValueError, match="changed"):
        read_local_file_for_cache_key(str(path), key)


def test_change_during_cache_key_read_fails_closed(tmp_path, monkeypatch):
    path = tmp_path / "document.pdf"
    path.write_bytes(b"first")
    timestamp = path.stat().st_mtime_ns
    original_open = builtins.open

    def replacing_open(filename, mode="r", *args, **kwargs):
        real_file = original_open(filename, mode, *args, **kwargs)
        if str(filename) != str(path) or mode != "rb":
            return real_file

        wrapped = MagicMock(wraps=real_file)
        wrapped.__enter__.return_value = wrapped
        wrapped.__exit__.side_effect = real_file.__exit__
        wrapped.fileno.side_effect = real_file.fileno

        def read_and_replace(*read_args):
            data = real_file.read(*read_args)
            path.write_bytes(b"other")
            os.utime(path, ns=(timestamp, timestamp))
            return data

        wrapped.read.side_effect = read_and_replace
        return wrapped

    monkeypatch.setattr(_multimodal, "open", replacing_open, raising=False)
    with pytest.raises(ValueError, match="changed"):
        local_file_cache_key(str(path))


def test_change_during_verified_file_read_fails_closed(tmp_path, monkeypatch):
    path = tmp_path / "document.pdf"
    path.write_bytes(b"first")
    key = local_file_cache_key(str(path))
    timestamp = path.stat().st_mtime_ns
    original_open = builtins.open

    def replacing_open(filename, mode="r", *args, **kwargs):
        real_file = original_open(filename, mode, *args, **kwargs)
        if str(filename) != str(path) or mode != "rb":
            return real_file

        wrapped = MagicMock(wraps=real_file)
        wrapped.__enter__.return_value = wrapped
        wrapped.__exit__.side_effect = real_file.__exit__
        wrapped.fileno.side_effect = real_file.fileno

        def read_and_replace(*read_args):
            data = real_file.read(*read_args)
            path.write_bytes(b"other")
            os.utime(path, ns=(timestamp, timestamp))
            return data

        wrapped.read.side_effect = read_and_replace
        return wrapped

    monkeypatch.setattr(_multimodal, "open", replacing_open, raising=False)
    with pytest.raises(ValueError, match="changed"):
        read_local_file_for_cache_key(str(path), key)


@pytest.mark.asyncio
async def test_temporary_upload_ids_are_returned_and_deleted(tmp_path):
    path = tmp_path / "document.pdf"
    path.write_bytes(b"pdf")
    sync_client = MagicMock()
    sync_client.files.create.return_value = SimpleNamespace(id="sync-file")
    sync_builder = MultimodalContentBuilder(client=sync_client)
    messages, uploads = sync_builder.build_with_uploads(str(path))
    assert messages[0]["content"][0]["file_id"] == "sync-file"
    assert uploads == ("sync-file",)
    sync_client.files.delete.assert_not_called()
    sync_builder.cleanup_uploads(uploads)
    sync_client.files.delete.assert_called_once_with("sync-file")

    async_client = MagicMock()
    async_client.files.create = AsyncMock(return_value=SimpleNamespace(id="async-file"))
    async_client.files.delete = AsyncMock()
    async_builder = AsyncMultimodalContentBuilder(client=async_client)
    messages, uploads = await async_builder.build_with_uploads(str(path))
    assert messages[0]["content"][0]["file_id"] == "async-file"
    assert uploads == ("async-file",)
    await async_builder.cleanup_uploads(uploads)
    async_client.files.delete.assert_awaited_once_with("async-file")


@pytest.mark.asyncio
async def test_direct_build_leaves_caller_owned_upload_available(tmp_path):
    path = tmp_path / "document.pdf"
    path.write_bytes(b"pdf")
    sync_client = MagicMock()
    sync_client.files.create.return_value = SimpleNamespace(id="sync-file")
    assert MultimodalContentBuilder(client=sync_client).build(str(path))[0]["content"][0]["file_id"] == "sync-file"
    sync_client.files.delete.assert_not_called()

    async_client = MagicMock()
    async_client.files.create = AsyncMock(return_value=SimpleNamespace(id="async-file"))
    async_client.files.delete = AsyncMock()
    assert (await AsyncMultimodalContentBuilder(client=async_client).build(str(path)))[0]["content"][0][
        "file_id"
    ] == "async-file"
    async_client.files.delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_cleanup_failure_is_propagated(tmp_path):
    path = tmp_path / "document.pdf"
    path.write_bytes(b"pdf")
    client = MagicMock()
    client.files.create.return_value = SimpleNamespace(id="file-1")
    client.files.delete.side_effect = RuntimeError("delete failed")
    builder = MultimodalContentBuilder(client=client)
    _, uploads = builder.build_with_uploads(str(path))
    with pytest.raises(RuntimeError, match="delete failed"):
        builder.cleanup_uploads(uploads)

    async_client = MagicMock()
    async_client.files.create = AsyncMock(return_value=SimpleNamespace(id="file-2"))
    async_client.files.delete = AsyncMock(side_effect=RuntimeError("delete failed"))
    async_builder = AsyncMultimodalContentBuilder(client=async_client)
    _, uploads = await async_builder.build_with_uploads(str(path))
    with pytest.raises(RuntimeError, match="delete failed"):
        await async_builder.cleanup_uploads(uploads)

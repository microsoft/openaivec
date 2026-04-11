from logging import Handler, StreamHandler, basicConfig
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import BaseModel, ValidationError

from openaivec import BatchResponses
from openaivec._responses import AsyncBatchResponses, Request

_h: Handler = StreamHandler()

basicConfig(handlers=[_h], level="DEBUG")


def _build_validation_error() -> ValidationError:
    class Fruit(BaseModel):
        name: str
        color: str
        taste: str

    class MessageT(BaseModel):
        id: int
        body: Fruit

    class ResponseT(BaseModel):
        assistant_messages: list[MessageT]

    try:
        ResponseT.model_validate({"assistant_messages": [{"id": 0, "body": {"name": "apple", "color": "red"}}]})
    except ValidationError as err:
        return err
    raise RuntimeError("Expected ValidationError")


class TestStructuredValidationRetries:
    def test_sync_retry_with_feedback(self):
        class Fruit(BaseModel):
            name: str
            color: str
            taste: str

        parse = Mock(
            side_effect=[
                _build_validation_error(),
                SimpleNamespace(
                    output_parsed=SimpleNamespace(
                        assistant_messages=[SimpleNamespace(id=0, body=Fruit(name="apple", color="red", taste="sweet"))]
                    )
                ),
            ]
        )
        client = BatchResponses(
            client=SimpleNamespace(responses=SimpleNamespace(parse=parse)),  # type: ignore[arg-type]
            model_name="gpt-4.1-mini",
            system_message="return fruit attributes",
            response_format=Fruit,
        )

        parsed = client._predict_chunk(["apple"])

        assert parse.call_count == 2
        first_instructions = parse.call_args_list[0].kwargs["instructions"]
        second_instructions = parse.call_args_list[1].kwargs["instructions"]
        assert "--- PRIOR VALIDATION FEEDBACK ---" not in first_instructions
        assert "--- PRIOR VALIDATION FEEDBACK ---" in second_instructions
        assert "assistant_messages[0].body.taste" in second_instructions
        assert parsed[0] == Fruit(name="apple", color="red", taste="sweet")

    def test_sync_retry_serializes_request_once(self, monkeypatch):
        class Fruit(BaseModel):
            name: str
            color: str
            taste: str

        validation_error = _build_validation_error()
        parse = Mock(
            side_effect=[
                validation_error,
                SimpleNamespace(
                    output_parsed=SimpleNamespace(
                        assistant_messages=[SimpleNamespace(id=0, body=Fruit(name="apple", color="red", taste="sweet"))]
                    )
                ),
            ]
        )
        serialized_calls = 0
        original_model_dump_json = Request.model_dump_json

        def counting_model_dump_json(self, *args, **kwargs):
            nonlocal serialized_calls
            serialized_calls += 1
            return original_model_dump_json(self, *args, **kwargs)

        monkeypatch.setattr(Request, "model_dump_json", counting_model_dump_json)

        client = BatchResponses(
            client=SimpleNamespace(responses=SimpleNamespace(parse=parse)),  # type: ignore[arg-type]
            model_name="gpt-4.1-mini",
            system_message="return fruit attributes",
            response_format=Fruit,
        )

        parsed = client._predict_chunk(["apple"])

        assert parsed[0] == Fruit(name="apple", color="red", taste="sweet")
        assert parse.call_count == 2
        assert serialized_calls == 1


class TestResponsesCachingAndErrors:
    def test_sync_parse_none_result_is_cached(self):
        parse = Mock(return_value=SimpleNamespace(output_parsed=None))
        client = BatchResponses(
            client=SimpleNamespace(responses=SimpleNamespace(parse=parse)),  # type: ignore[arg-type]
            model_name="gpt-4.1-mini",
            system_message="repeat user input",
            response_format=str,
        )

        first = client.parse(["hello"])
        second = client.parse(["hello"])

        assert first == [None]
        assert second == [None]
        assert parse.call_count == 1

    @pytest.mark.asyncio
    async def test_async_parse_none_result_is_cached(self):
        parse = AsyncMock(return_value=SimpleNamespace(output_parsed=None))
        client = AsyncBatchResponses(
            client=SimpleNamespace(responses=SimpleNamespace(parse=parse)),  # type: ignore[arg-type]
            model_name="gpt-4.1-mini",
            system_message="repeat user input",
            response_format=str,
        )

        first = await client.parse(["hello"])
        second = await client.parse(["hello"])

        assert first == [None]
        assert second == [None]
        assert parse.call_count == 1

    def test_sync_retry_exhaustion_raises(self):
        class Fruit(BaseModel):
            name: str
            color: str
            taste: str

        validation_error = _build_validation_error()
        parse = Mock(side_effect=[validation_error, validation_error, validation_error])
        client = BatchResponses(
            client=SimpleNamespace(responses=SimpleNamespace(parse=parse)),  # type: ignore[arg-type]
            model_name="gpt-4.1-mini",
            system_message="return fruit attributes",
            response_format=Fruit,
            max_validation_retries=2,
        )

        with pytest.raises(ValidationError):
            client._predict_chunk(["apple"])
        assert parse.call_count == 3

    @pytest.mark.asyncio
    async def test_async_retry_with_feedback(self):
        class Fruit(BaseModel):
            name: str
            color: str
            taste: str

        parse = AsyncMock(
            side_effect=[
                _build_validation_error(),
                SimpleNamespace(
                    output_parsed=SimpleNamespace(
                        assistant_messages=[SimpleNamespace(id=0, body=Fruit(name="apple", color="red", taste="sweet"))]
                    )
                ),
            ]
        )
        client = AsyncBatchResponses(
            client=SimpleNamespace(responses=SimpleNamespace(parse=parse)),  # type: ignore[arg-type]
            model_name="gpt-4.1-mini",
            system_message="return fruit attributes",
            response_format=Fruit,
        )

        parsed = await client._predict_chunk(["apple"])

        assert parse.call_count == 2
        first_instructions = parse.call_args_list[0].kwargs["instructions"]
        second_instructions = parse.call_args_list[1].kwargs["instructions"]
        assert "--- PRIOR VALIDATION FEEDBACK ---" not in first_instructions
        assert "--- PRIOR VALIDATION FEEDBACK ---" in second_instructions
        assert "assistant_messages[0].body.taste" in second_instructions
        assert parsed[0] == Fruit(name="apple", color="red", taste="sweet")

    @pytest.mark.asyncio
    async def test_async_retry_serializes_request_once(self, monkeypatch):
        class Fruit(BaseModel):
            name: str
            color: str
            taste: str

        parse = AsyncMock(
            side_effect=[
                _build_validation_error(),
                SimpleNamespace(
                    output_parsed=SimpleNamespace(
                        assistant_messages=[SimpleNamespace(id=0, body=Fruit(name="apple", color="red", taste="sweet"))]
                    )
                ),
            ]
        )
        serialized_calls = 0
        original_model_dump_json = Request.model_dump_json

        def counting_model_dump_json(self, *args, **kwargs):
            nonlocal serialized_calls
            serialized_calls += 1
            return original_model_dump_json(self, *args, **kwargs)

        monkeypatch.setattr(Request, "model_dump_json", counting_model_dump_json)

        client = AsyncBatchResponses(
            client=SimpleNamespace(responses=SimpleNamespace(parse=parse)),  # type: ignore[arg-type]
            model_name="gpt-4.1-mini",
            system_message="return fruit attributes",
            response_format=Fruit,
        )

        parsed = await client._predict_chunk(["apple"])

        assert parsed[0] == Fruit(name="apple", color="red", taste="sweet")
        assert parse.call_count == 2
        assert serialized_calls == 1


@pytest.mark.requires_api
class TestVectorizedResponsesOpenAI:
    @pytest.fixture(autouse=True)
    def setup_client(self, openai_client, responses_model_name):
        self.openai_client = openai_client
        self.model_name = responses_model_name
        yield

    def test_predict_str(self):
        system_message = """
        just repeat the user message
        """.strip()
        client = BatchResponses(
            client=self.openai_client,
            model_name=self.model_name,
            system_message=system_message,
        )
        response = client._predict_chunk(["hello", "world"])

        assert response == ["hello", "world"]

    def test_predict_structured(self):
        system_message = """
        return the color and taste of given fruit
        #example
        ## input
        apple

        ## output
        {
            "name": "apple",
            "color": "red",
            "taste": "sweet"
        }
        """

        class Fruit(BaseModel):
            name: str
            color: str
            taste: str

        client = BatchResponses(
            client=self.openai_client, model_name=self.model_name, system_message=system_message, response_format=Fruit
        )

        response = client._predict_chunk(["apple", "banana"])

        assert all(isinstance(item, Fruit) for item in response)

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_predict_with_batch_sizes(self, batch_size):
        """Test BatchResponses with different batch sizes."""
        system_message = "just repeat the user message"
        client = BatchResponses(
            client=self.openai_client,
            model_name=self.model_name,
            system_message=system_message,
        )

        test_inputs = ["test1", "test2", "test3", "test4"][:batch_size]
        response = client._predict_chunk(test_inputs)

        assert len(response) == len(test_inputs)
        assert all(isinstance(item, str) for item in response)


@pytest.mark.requires_api
class TestAsyncBatchResponses:
    @pytest.fixture(autouse=True)
    def setup_client(self, async_openai_client):
        self.openai_client = async_openai_client
        self.model_name = "gpt-4.1-mini"
        yield

    @pytest.mark.asyncio
    async def test_parse_str(self):
        system_message = """
        just repeat the user message
        """.strip()
        client = AsyncBatchResponses.of(
            client=self.openai_client,
            model_name=self.model_name,
            system_message=system_message,
            batch_size=1,
        )
        response = await client.parse(["apple", "orange", "banana", "pineapple"])
        assert response == ["apple", "orange", "banana", "pineapple"]

    @pytest.mark.asyncio
    async def test_parse_structured(self):
        system_message = """
        return the color and taste of given fruit
        #example
        ## input
        apple

        ## output
        {
            "name": "apple",
            "color": "red",
            "taste": "sweet"
        }
        """
        input_fruits = ["apple", "banana", "orange", "pineapple"]

        class Fruit(BaseModel):
            name: str
            color: str
            taste: str

        client = AsyncBatchResponses.of(
            client=self.openai_client,
            model_name=self.model_name,
            system_message=system_message,
            response_format=Fruit,
            batch_size=1,
        )
        response = await client.parse(input_fruits)
        assert len(response) == len(input_fruits)
        for i, item in enumerate(response):
            assert isinstance(item, Fruit)
            assert item.name.lower() == input_fruits[i].lower()
            assert isinstance(item.color, str)
            assert len(item.color) > 0
            assert isinstance(item.taste, str)
            assert len(item.taste) > 0

    @pytest.mark.asyncio
    async def test_parse_structured_empty_input(self):
        system_message = """
        return the color and taste of given fruit
        """

        class Fruit(BaseModel):
            name: str
            color: str
            taste: str

        client = AsyncBatchResponses.of(
            client=self.openai_client,
            model_name=self.model_name,
            system_message=system_message,
            response_format=Fruit,
            batch_size=1,
        )
        response = await client.parse([])
        assert response == []

    @pytest.mark.asyncio
    async def test_parse_structured_batch_size(self):
        system_message = """
        return the color and taste of given fruit
        #example
        ## input
        apple

        ## output
        {
            "name": "apple",
            "color": "red",
            "taste": "sweet"
        }
        """
        input_fruits = ["apple", "banana", "orange", "pineapple"]

        class Fruit(BaseModel):
            name: str
            color: str
            taste: str

        client_bs2 = AsyncBatchResponses.of(
            client=self.openai_client,
            model_name=self.model_name,
            system_message=system_message,
            response_format=Fruit,
            batch_size=2,
        )
        response_bs2 = await client_bs2.parse(input_fruits)
        assert len(response_bs2) == len(input_fruits)
        for i, item in enumerate(response_bs2):
            assert isinstance(item, Fruit)
            assert item.name.lower() == input_fruits[i].lower()
            assert isinstance(item.color, str)
            assert len(item.color) > 0
            assert isinstance(item.taste, str)
            assert len(item.taste) > 0

        client_bs4 = AsyncBatchResponses.of(
            client=self.openai_client,
            model_name=self.model_name,
            system_message=system_message,
            response_format=Fruit,
            batch_size=4,
        )
        response_bs4 = await client_bs4.parse(input_fruits)
        assert len(response_bs4) == len(input_fruits)
        for i, item in enumerate(response_bs4):
            assert isinstance(item, Fruit)
            assert item.name.lower() == input_fruits[i].lower()
            assert isinstance(item.color, str)
            assert len(item.color) > 0
            assert isinstance(item.taste, str)
            assert len(item.taste) > 0


# ---------------------------------------------------------------------------
# Multimodal routing tests
# ---------------------------------------------------------------------------


class TestMultimodalRouting:
    """Test that multimodal=True routes file/URL inputs correctly."""

    def test_text_only_uses_batch_path(self):
        from unittest.mock import MagicMock, patch

        from openaivec._cache import BatchCache
        from openaivec._responses import BatchResponses, Message, Response

        batch = BatchResponses(
            client=MagicMock(),
            model_name="gpt-4.1-mini",
            system_message="test",
            response_format=str,
            cache=BatchCache(batch_size=10),
            multimodal=True,
        )

        def mock_llm(self, msgs):
            resp = MagicMock()
            resp.output_parsed = Response(assistant_messages=[Message(id=m.id, body="ok") for m in msgs])
            return resp

        with patch.object(BatchResponses, "_request_llm", mock_llm):
            results = batch.parse(["hello", "world"])

        assert results == ["ok", "ok"]

    def test_multimodal_false_treats_urls_as_text(self):
        from unittest.mock import MagicMock, patch

        from openaivec._cache import BatchCache
        from openaivec._responses import BatchResponses, Message, Response

        batch = BatchResponses(
            client=MagicMock(),
            model_name="gpt-4.1-mini",
            system_message="test",
            response_format=str,
            cache=BatchCache(batch_size=10),
            multimodal=False,
        )

        def mock_llm(self, msgs):
            resp = MagicMock()
            resp.output_parsed = Response(
                assistant_messages=[Message(id=m.id, body=f"text:{m.body[:10]}") for m in msgs]
            )
            return resp

        with patch.object(BatchResponses, "_request_llm", mock_llm):
            results = batch.parse(["https://example.com/photo.jpg", "plain text"])

        assert results[0].startswith("text:https://e")
        assert results[1].startswith("text:plain tex")

    def test_mixed_routing_with_local_file(self, tmp_path):
        from unittest.mock import MagicMock, patch

        from openaivec._cache import BatchCache
        from openaivec._responses import BatchResponses, Message, Response

        img = tmp_path / "test.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)

        batch = BatchResponses(
            client=MagicMock(),
            model_name="gpt-4.1-mini",
            system_message="test",
            response_format=str,
            cache=BatchCache(batch_size=10),
            multimodal=True,
        )

        calls = {"llm": 0, "mm": 0}

        def mock_llm(self, msgs):
            calls["llm"] += 1
            resp = MagicMock()
            resp.output_parsed = Response(assistant_messages=[Message(id=m.id, body="batch") for m in msgs])
            return resp

        def mock_mm(self, parts):
            calls["mm"] += 1
            return "individual"

        with (
            patch.object(BatchResponses, "_request_llm", mock_llm),
            patch.object(BatchResponses, "_request_multimodal", mock_mm),
        ):
            results = batch.parse(["plain text", str(img)])

        assert calls["llm"] == 1
        assert calls["mm"] == 1
        assert results == ["batch", "individual"]

    def test_url_without_extension_is_text(self):
        from openaivec._file import is_multimodal_input

        assert not is_multimodal_input("https://api.example.com/v1/data?key=abc")
        assert is_multimodal_input("https://cdn.example.com/image.png")
        assert not is_multimodal_input("https://example.com/")

    def test_file_size_limit(self, tmp_path):
        from openaivec._file import encode_file_to_data_uri

        big_file = tmp_path / "huge.txt"
        big_file.write_bytes(b"x" * (21 * 1024 * 1024))

        with pytest.raises(ValueError, match="exceeding the 20 MB limit"):
            encode_file_to_data_uri(str(big_file))

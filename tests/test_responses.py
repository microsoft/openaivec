import asyncio
import base64
import json
import os
from logging import Handler, StreamHandler, basicConfig
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
import tiktoken
from pydantic import BaseModel, ValidationError

from openaivec import BatchResponses, ResponseLimits, _responses
from openaivec._responses import AsyncBatchResponses, Message, Request, _plan_response_batches
from openaivec._retry import RetryPolicy

_h: Handler = StreamHandler()

basicConfig(handlers=[_h], level="DEBUG")


class TestResponseTokenPlanning:
    @pytest.mark.asyncio
    async def test_every_sync_and_async_request_fits_measured_token_budget(self):
        inputs = ["one " * 300, "short", "two " * 230, "short"]
        limits = ResponseLimits(
            max_request_tokens=900,
            max_inputs=3,
            expected_output_tokens_per_item=64,
            validation_feedback_tokens=48,
        )
        sync_calls = []
        async_calls = []

        def response_for(kwargs):
            envelope = json.loads(kwargs["input"])
            return SimpleNamespace(
                output_parsed=SimpleNamespace(
                    assistant_messages=[
                        SimpleNamespace(id=message["id"], body=message["body"])
                        for message in reversed(envelope["user_messages"])
                    ]
                )
            )

        def sync_parse(**kwargs):
            sync_calls.append(kwargs)
            return response_for(kwargs)

        async def async_parse(**kwargs):
            async_calls.append(kwargs)
            return response_for(kwargs)

        sync_client = BatchResponses.of(
            SimpleNamespace(responses=SimpleNamespace(parse=sync_parse)),
            "gpt-4.1-mini",
            "Echo.",
            batch_size=0,
            limits=limits,
            max_validation_retries=1,
            max_output_tokens=70,
        )
        async_client = AsyncBatchResponses.of(
            SimpleNamespace(responses=SimpleNamespace(parse=async_parse)),
            "gpt-4.1-mini",
            "Echo.",
            batch_size=0,
            limits=limits,
            max_validation_retries=1,
            max_output_tokens=70,
        )
        assert sync_client.parse(inputs) == inputs
        assert await async_client.parse(inputs) == inputs
        assert len(sync_calls) > 1
        assert [[m["id"] for m in json.loads(c["input"])["user_messages"]] for c in sync_calls] == [
            [m["id"] for m in json.loads(c["input"])["user_messages"]] for c in async_calls
        ]

        encoding = tiktoken.encoding_for_model("gpt-4.1-mini")
        for call in [*sync_calls, *async_calls]:
            request = json.loads(call["input"])
            schema = call["text_format"].model_json_schema()
            measured_cost = (
                len(encoding.encode_ordinary(call["instructions"]))
                + len(encoding.encode_ordinary(json.dumps(schema, separators=(",", ":"))))
                + len(encoding.encode_ordinary(json.dumps(request, ensure_ascii=False, separators=(",", ":"))))
                + max(70, limits.expected_output_tokens_per_item * len(request["user_messages"]))
                + limits.validation_feedback_tokens
                + 32
            )
            assert len(request["user_messages"]) <= limits.max_inputs
            assert measured_cost <= limits.max_request_tokens, (measured_cost, limits.max_request_tokens)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("async_mode", [False, True])
    async def test_validation_feedback_cannot_exceed_request_budget(self, async_mode):
        validation_error = _build_validation_error()
        parse = AsyncMock(side_effect=validation_error) if async_mode else Mock(side_effect=validation_error)
        client = SimpleNamespace(responses=SimpleNamespace(parse=parse))
        high_limits = ResponseLimits(expected_output_tokens_per_item=16, validation_feedback_tokens=0)
        if async_mode:
            probe = AsyncBatchResponses.of(
                client, "gpt-4.1-mini", "Echo.", max_validation_retries=0, limits=high_limits
            )
            with pytest.raises(ValidationError):
                await probe.parse(["fruit"])
        else:
            probe = BatchResponses.of(client, "gpt-4.1-mini", "Echo.", max_validation_retries=0, limits=high_limits)
            with pytest.raises(ValidationError):
                probe.parse(["fruit"])

        call = parse.call_args.kwargs
        encoding = tiktoken.encoding_for_model("gpt-4.1-mini")
        estimated_initial = (
            len(encoding.encode_ordinary(call["instructions"]))
            + len(encoding.encode_ordinary(json.dumps(call["text_format"].model_json_schema(), separators=(",", ":"))))
            + len(
                encoding.encode_ordinary(
                    json.dumps(json.loads(call["input"]), ensure_ascii=False, separators=(",", ":"))
                )
            )
            + high_limits.expected_output_tokens_per_item
            + 32
        )
        limits = ResponseLimits(
            max_request_tokens=estimated_initial + 1,
            expected_output_tokens_per_item=high_limits.expected_output_tokens_per_item,
            validation_feedback_tokens=0,
        )
        parse.reset_mock()
        if async_mode:
            wrapper = AsyncBatchResponses.of(client, "gpt-4.1-mini", "Echo.", limits=limits, max_validation_retries=1)
            with pytest.raises(ValueError, match="budget"):
                await wrapper.parse(["fruit"])
        else:
            wrapper = BatchResponses.of(client, "gpt-4.1-mini", "Echo.", limits=limits, max_validation_retries=1)
            with pytest.raises(ValueError, match="budget"):
                wrapper.parse(["fruit"])
        assert parse.call_count == 1
        assert not wrapper.cache._inflight

    def test_split_by_tokens_and_count_preserves_ids_and_order(self, monkeypatch):
        inputs = ["short", "long " * 500, "middle", "another", "short"]
        limits = ResponseLimits(
            max_request_tokens=1200,
            max_inputs=2,
            expected_output_tokens_per_item=32,
            validation_feedback_tokens=0,
        )
        calls = []

        def request(self, batch):
            calls.append([message.id for message in batch])
            return SimpleNamespace(
                output_parsed=SimpleNamespace(
                    assistant_messages=[
                        SimpleNamespace(id=message.id, body=message.body) for message in reversed(batch)
                    ]
                )
            )

        monkeypatch.setattr(BatchResponses, "_request_llm", request)
        client = BatchResponses.of(
            SimpleNamespace(),
            "gpt-4.1-mini",
            "Echo.",
            batch_size=0,
            max_validation_retries=0,
            limits=limits,
        )
        assert client.parse(inputs) == inputs
        assert [identity for batch in calls for identity in batch] == [0, 1, 2, 3]
        assert len(calls) >= 2
        assert all(len(batch) <= 2 for batch in calls)
        planned = _plan_response_batches(
            [Message(id=i, body=text) for i, text in enumerate(inputs[:-1])],
            client.model_name,
            client._vectorized_system_message,
            str,
            limits,
            0,
            {},
        )
        assert calls == [[message.id for message in batch] for batch in planned]

    @pytest.mark.asyncio
    async def test_async_split_matches_sync(self, monkeypatch):
        inputs = ["short", "long " * 500, "middle", "another", "short"]
        limits = ResponseLimits(
            max_request_tokens=1200,
            max_inputs=2,
            expected_output_tokens_per_item=32,
            validation_feedback_tokens=0,
        )
        calls = []

        async def request(self, batch):
            calls.append([message.id for message in batch])
            return SimpleNamespace(
                output_parsed=SimpleNamespace(
                    assistant_messages=[
                        SimpleNamespace(id=message.id, body=message.body) for message in reversed(batch)
                    ]
                )
            )

        monkeypatch.setattr(AsyncBatchResponses, "_request_llm", request)
        client = AsyncBatchResponses.of(
            SimpleNamespace(),
            "gpt-4.1-mini",
            "Echo.",
            batch_size=0,
            max_validation_retries=0,
            limits=limits,
        )
        assert await client.parse(inputs) == inputs
        assert calls == [
            [message.id for message in batch]
            for batch in _plan_response_batches(
                [Message(id=i, body=text) for i, text in enumerate(inputs[:-1])],
                client.model_name,
                client._vectorized_system_message,
                str,
                limits,
                0,
                {},
            )
        ]

    @pytest.mark.parametrize("async_mode", [False, True])
    @pytest.mark.asyncio
    async def test_oversized_item_fails_before_request(self, monkeypatch, async_mode):
        limits = ResponseLimits(
            max_request_tokens=800,
            expected_output_tokens_per_item=32,
            validation_feedback_tokens=0,
        )
        sync_request = Mock()
        async_request = AsyncMock()
        monkeypatch.setattr(BatchResponses, "_request_llm", sync_request)
        monkeypatch.setattr(AsyncBatchResponses, "_request_llm", async_request)
        inputs = ["short", "long " * 2000]
        if async_mode:
            client = AsyncBatchResponses.of(
                SimpleNamespace(),
                "gpt-4.1-mini",
                "Echo.",
                batch_size=0,
                max_validation_retries=0,
                limits=limits,
            )
            with pytest.raises(ValueError, match="Response input ID 1 exceeds"):
                await client.parse(inputs)
        else:
            client = BatchResponses.of(
                SimpleNamespace(),
                "gpt-4.1-mini",
                "Echo.",
                batch_size=0,
                max_validation_retries=0,
                limits=limits,
            )
            with pytest.raises(ValueError, match="Response input ID 1 exceeds"):
                client.parse(inputs)
        sync_request.assert_not_called()
        async_request.assert_not_called()

    def test_schema_instructions_and_output_allowance_reduce_capacity(self):
        class RichResult(BaseModel):
            title: str
            explanation: str
            category: str

        messages = [Message(id=i, body="hello") for i in range(3)]

        def plan(instructions, response_format, allowance, budget=1000, kwargs=None):
            return _plan_response_batches(
                messages,
                "gpt-4.1-mini",
                instructions,
                response_format,
                ResponseLimits(
                    max_request_tokens=budget,
                    max_inputs=3,
                    expected_output_tokens_per_item=allowance,
                    validation_feedback_tokens=0,
                ),
                0,
                kwargs or {},
            )

        assert len(plan("short", str, 32)) == 1
        assert len(plan("short", RichResult, 32, budget=300)) > 1
        assert len(plan("short", str, 32, budget=300)) == 1
        assert len(plan("long " * 80, str, 32, budget=300)) > 1
        assert len(plan("short", str, 400)) > 1
        assert len(plan("short", str, 32, kwargs={"max_output_tokens": 850})) > 1

    @pytest.mark.parametrize(
        "field,value,error",
        [
            ("max_inputs", 0, ValueError),
            ("max_request_tokens", -1, ValueError),
            ("expected_output_tokens_per_item", True, TypeError),
            ("validation_feedback_tokens", -1, ValueError),
        ],
    )
    def test_invalid_limits(self, field, value, error):
        with pytest.raises(error):
            ResponseLimits(**{field: value})

    @pytest.mark.parametrize("invalid_cap", [False, 0, 0.0, None])
    def test_invalid_max_output_tokens_is_rejected(self, invalid_cap):
        with pytest.raises(ValueError, match="max_output_tokens"):
            _plan_response_batches(
                [Message(id=0, body="hello")],
                "gpt-4.1-mini",
                "short",
                str,
                ResponseLimits(),
                0,
                {"max_output_tokens": invalid_cap},
            )


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
    def setup_client(self, async_openai_client, responses_model_name):
        self.openai_client = async_openai_client
        self.model_name = responses_model_name
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

    def test_text_files_batched_with_text(self, tmp_path):
        """Text-readable files (.py, .js, etc.) are inlined and batched."""
        from unittest.mock import MagicMock, patch

        from openaivec._cache import BatchCache
        from openaivec._responses import BatchResponses, Message, Response

        py_file = tmp_path / "app.py"
        py_file.write_text("def hello(): return 42")
        js_file = tmp_path / "app.js"
        js_file.write_text("const x = 7;")

        batch = BatchResponses(
            client=MagicMock(),
            model_name="gpt-4.1-mini",
            system_message="test",
            response_format=str,
            cache=BatchCache(batch_size=10),
            multimodal=True,
        )

        captured_bodies: list[str] = []
        calls = {"llm": 0, "mm": 0}

        def mock_llm(self, msgs):
            calls["llm"] += 1
            for m in msgs:
                captured_bodies.append(m.body)
            resp = MagicMock()
            resp.output_parsed = Response(assistant_messages=[Message(id=m.id, body="ok") for m in msgs])
            return resp

        def mock_mm(self, parts):
            calls["mm"] += 1
            return "mm"

        with (
            patch.object(BatchResponses, "_request_llm", mock_llm),
            patch.object(BatchResponses, "_request_multimodal", mock_mm),
        ):
            results = batch.parse(["hello", str(py_file), str(js_file)])

        assert calls["llm"] == 1, "All text inputs should be in one batch call"
        assert calls["mm"] == 0, "No multimodal calls for text files"
        assert results == ["ok", "ok", "ok"]
        assert any("[File: app.py]" in b for b in captured_bodies)
        assert any("[File: app.js]" in b for b in captured_bodies)
        assert any("def hello" in b for b in captured_bodies)

    def test_binary_file_goes_multimodal(self, tmp_path):
        """Binary document files (PDF) still go through Files API."""
        from openaivec._multimodal import is_multimodal_input, is_readable_text_file

        pdf = tmp_path / "report.pdf"
        pdf.write_bytes(b"%PDF-1.4 test")
        assert is_multimodal_input(str(pdf))
        assert not is_readable_text_file(str(pdf))

        py = tmp_path / "code.py"
        py.write_text("x = 1")
        assert not is_multimodal_input(str(py))
        assert is_readable_text_file(str(py))

    def test_url_without_extension_is_text(self):
        from openaivec._multimodal import is_multimodal_input

        assert not is_multimodal_input("https://api.example.com/v1/data?key=abc")
        assert is_multimodal_input("https://cdn.example.com/image.png")
        assert not is_multimodal_input("https://example.com/")

    def test_audio_url_is_multimodal(self):
        from openaivec._multimodal import is_audio_path, is_multimodal_input

        assert is_multimodal_input("https://cdn.example.com/speech.mp3")
        assert is_multimodal_input("https://cdn.example.com/audio.wav")
        assert is_audio_path("recording.mp3")
        assert is_audio_path("file.wav")
        assert not is_audio_path("photo.png")

    def test_audio_local_file_raises_error(self, tmp_path):
        from openaivec._cache import BatchCache
        from openaivec._responses import BatchResponses

        mp3 = tmp_path / "test.mp3"
        mp3.write_bytes(b"\xff\xfb\x90\x00" + b"\x00" * 50)

        batch = BatchResponses(
            client=MagicMock(),
            model_name="gpt-4.1-mini",
            system_message="test",
            response_format=str,
            cache=BatchCache(batch_size=10),
            multimodal=True,
        )

        with pytest.raises(ValueError, match="Audio files.*not supported by the Responses API"):
            batch.parse([str(mp3)])

    def test_audio_url_raises_error(self):
        from openaivec._multimodal import MultimodalContentBuilder

        builder = MultimodalContentBuilder(client=MagicMock())

        with pytest.raises(ValueError, match="Audio files.*not supported"):
            builder.build("https://cdn.example.com/speech.mp3")

    def test_builder_returns_response_input_param(self, tmp_path):
        from openaivec._multimodal import MultimodalContentBuilder

        builder = MultimodalContentBuilder(client=MagicMock())

        result = builder.build("plain text")
        assert isinstance(result, list)
        assert result[0]["role"] == "user"

        img = tmp_path / "photo.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 50)
        result = builder.build(str(img))
        assert isinstance(result, list)
        assert result[0]["role"] == "user"
        assert result[0]["content"][0]["type"] == "input_image"

    def test_document_url_no_filename(self):
        from openaivec._multimodal import MultimodalContentBuilder

        builder = MultimodalContentBuilder(client=MagicMock())
        result = builder.build("https://example.com/report.pdf")
        assert result[0]["role"] == "user"
        content = result[0]["content"][0]
        assert content["type"] == "input_file"
        assert content["file_url"] == "https://example.com/report.pdf"
        assert "filename" not in content

    def test_file_size_limit(self, tmp_path):
        from openaivec._multimodal import encode_file_to_data_uri

        big_file = tmp_path / "huge.txt"
        big_file.write_bytes(b"x" * (21 * 1024 * 1024))

        with pytest.raises(ValueError, match="exceeding the 20 MB limit"):
            encode_file_to_data_uri(str(big_file))


class _MediaResult(BaseModel):
    name: str
    color: str


def _media_validation_error() -> ValidationError:
    try:
        _MediaResult.model_validate({"name": "apple"})
    except ValidationError as error:
        return error
    raise AssertionError("Expected missing color to fail validation")


class TestMultimodalAcceptance:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("async_mode", [False, True])
    @pytest.mark.parametrize("suffix", [".txt", ".png", ".pdf"])
    async def test_cache_tracks_local_content_and_uses_matching_bytes(self, tmp_path, suffix, async_mode):
        path = tmp_path / f"document{suffix}"
        path.write_bytes(b"first")
        uploads: dict[str, bytes] = {}

        def upload(*, file, purpose):
            assert purpose == "assistants"
            file_id = f"file-{len(uploads) + 1}"
            uploads[file_id] = file.read()
            return SimpleNamespace(id=file_id)

        def text_response(**kwargs):
            messages = json.loads(kwargs["input"])["user_messages"]
            return SimpleNamespace(
                output_parsed=SimpleNamespace(
                    assistant_messages=[
                        SimpleNamespace(id=message["id"], body=message["body"].splitlines()[-1]) for message in messages
                    ]
                )
            )

        def media_response(**kwargs):
            content = kwargs["input"][0]["content"][0]
            if suffix == ".pdf":
                body = uploads[content["file_id"]]
            else:
                body = base64.b64decode(content["image_url"].split(",", 1)[1])
            return SimpleNamespace(output_text=body.decode())

        files = SimpleNamespace(
            create=AsyncMock(side_effect=upload) if async_mode else Mock(side_effect=upload),
            delete=AsyncMock() if async_mode else Mock(),
        )
        responses = SimpleNamespace(
            parse=AsyncMock(side_effect=text_response) if async_mode else Mock(side_effect=text_response),
            create=AsyncMock(side_effect=media_response) if async_mode else Mock(side_effect=media_response),
        )
        client = SimpleNamespace(files=files, responses=responses)
        if async_mode:
            wrapper = AsyncBatchResponses.of(client, "gpt-4.1-mini", "Echo.", batch_size=0, multimodal=True)

            async def predict(values):
                return await wrapper.parse(values)

        else:
            wrapper = BatchResponses.of(client, "gpt-4.1-mini", "Echo.", batch_size=0, multimodal=True)

            async def predict(values):
                return wrapper.parse(values)

        assert await predict([str(path), str(path)]) == ["first", "first"]
        assert await predict([str(path)]) == ["first"]
        timestamp = path.stat().st_mtime_ns
        path.write_bytes(b"other")
        os.utime(path, ns=(timestamp, timestamp))
        assert await predict([str(path)]) == ["other"]
        assert responses.parse.call_count == (2 if suffix == ".txt" else 0)
        assert responses.create.call_count == (0 if suffix == ".txt" else 2)
        assert list(uploads.values()) == ([b"first", b"other"] if suffix == ".pdf" else [])
        assert files.delete.call_count == (2 if suffix == ".pdf" else 0)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("async_mode", [False, True])
    async def test_file_changed_after_cache_lookup_cannot_return_stale_hit(self, tmp_path, monkeypatch, async_mode):
        path = tmp_path / "message.txt"
        path.write_text("first")

        def echo(**kwargs):
            body = json.loads(kwargs["input"])["user_messages"][0]["body"]
            return SimpleNamespace(output_parsed=SimpleNamespace(assistant_messages=[SimpleNamespace(id=0, body=body)]))

        parse = AsyncMock(side_effect=echo) if async_mode else Mock(side_effect=echo)
        client = SimpleNamespace(responses=SimpleNamespace(parse=parse))
        if async_mode:
            wrapper = AsyncBatchResponses.of(client, "gpt-4.1-mini", "Echo.", multimodal=True)

            async def predict():
                return await wrapper.parse([str(path)])

        else:
            wrapper = BatchResponses.of(client, "gpt-4.1-mini", "Echo.", multimodal=True)

            async def predict():
                return wrapper.parse([str(path)])

        assert await predict() == ["[File: message.txt]\nfirst"]
        original_cache_key = _responses.local_file_cache_key
        changed = False

        def mutate_on_lookup(value):
            nonlocal changed
            key = original_cache_key(value)
            if not changed:
                path.write_text("other")
                changed = True
            return key

        monkeypatch.setattr(_responses, "local_file_cache_key", mutate_on_lookup)
        with pytest.raises(ValueError, match="changed"):
            await predict()
        assert parse.call_count == 1
        assert not wrapper.cache._inflight
        assert await predict() == ["[File: message.txt]\nother"]
        assert parse.call_count == 2

    @pytest.mark.asyncio
    @pytest.mark.parametrize("async_mode", [False, True])
    async def test_literal_nul_content_cannot_collide_with_local_file_key(self, tmp_path, async_mode):
        path = tmp_path / "message.txt"
        path.write_text("first")
        literal = _responses.local_file_cache_key(str(path))

        def echo(**kwargs):
            messages = json.loads(kwargs["input"])["user_messages"]
            return SimpleNamespace(
                output_parsed=SimpleNamespace(
                    assistant_messages=[SimpleNamespace(id=message["id"], body=message["body"]) for message in messages]
                )
            )

        parse = AsyncMock(side_effect=echo) if async_mode else Mock(side_effect=echo)
        client = SimpleNamespace(responses=SimpleNamespace(parse=parse))
        if async_mode:
            wrapper = AsyncBatchResponses.of(client, "gpt-4.1-mini", "Echo.", multimodal=True)
            result = await wrapper.parse([str(path), literal])
        else:
            wrapper = BatchResponses.of(client, "gpt-4.1-mini", "Echo.", multimodal=True)
            result = wrapper.parse([str(path), literal])
        assert result == ["[File: message.txt]\nfirst", literal]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("async_mode", [False, True])
    @pytest.mark.parametrize("suffix", [".txt", ".pdf"])
    async def test_mutation_between_digest_and_file_read_fails_without_upload(
        self, tmp_path, monkeypatch, async_mode, suffix
    ):
        path = tmp_path / f"document{suffix}"
        path.write_bytes(b"first")
        original_cache_key = _responses.local_file_cache_key

        def mutate_after_digest(value):
            key = original_cache_key(value)
            path.write_bytes(b"other")
            return key

        monkeypatch.setattr(_responses, "local_file_cache_key", mutate_after_digest)
        client = SimpleNamespace(
            files=SimpleNamespace(create=AsyncMock() if async_mode else Mock()),
            responses=SimpleNamespace(parse=AsyncMock() if async_mode else Mock()),
        )
        if async_mode:
            wrapper = AsyncBatchResponses.of(client, "gpt-4.1-mini", "Echo.", batch_size=0, multimodal=True)
            with pytest.raises(ValueError, match="changed"):
                await wrapper.parse([str(path)])
        else:
            wrapper = BatchResponses.of(client, "gpt-4.1-mini", "Echo.", batch_size=0, multimodal=True)
            with pytest.raises(ValueError, match="changed"):
                wrapper.parse([str(path)])
        client.files.create.assert_not_called()
        client.responses.parse.assert_not_called()
        assert not wrapper.cache._inflight

    @pytest.mark.asyncio
    @pytest.mark.parametrize("async_mode", [False, True])
    async def test_structured_retry_reuses_upload_and_deletes_after_last_attempt(self, tmp_path, async_mode):
        path = tmp_path / "document.pdf"
        path.write_bytes(b"pdf content")
        events = []
        instructions = []

        def upload(*, file, purpose):
            events.append(("upload", file.read()))
            return SimpleNamespace(id="uploaded-1")

        def parse_media(**kwargs):
            part = kwargs["input"][0]["content"][0]
            events.append(("parse", part["file_id"]))
            instructions.append(kwargs["instructions"])
            if len(instructions) == 1:
                raise _media_validation_error()
            return SimpleNamespace(output_parsed=_MediaResult(name="apple", color="red"))

        def delete(file_id):
            events.append(("delete", file_id))

        files = SimpleNamespace(
            create=AsyncMock(side_effect=upload) if async_mode else Mock(side_effect=upload),
            delete=AsyncMock(side_effect=delete) if async_mode else Mock(side_effect=delete),
        )
        responses = SimpleNamespace(
            parse=AsyncMock(side_effect=parse_media) if async_mode else Mock(side_effect=parse_media)
        )
        client = SimpleNamespace(files=files, responses=responses)
        if async_mode:
            wrapper = AsyncBatchResponses.of(
                client,
                "gpt-4.1-mini",
                "Extract fruit",
                _MediaResult,
                batch_size=0,
                multimodal=True,
                max_validation_retries=1,
            )
            result = await wrapper.parse([str(path)])
        else:
            wrapper = BatchResponses.of(
                client,
                "gpt-4.1-mini",
                "Extract fruit",
                _MediaResult,
                batch_size=0,
                multimodal=True,
                max_validation_retries=1,
            )
            result = wrapper.parse([str(path)])

        assert result == [_MediaResult(name="apple", color="red")]
        assert events == [
            ("upload", b"pdf content"),
            ("parse", "uploaded-1"),
            ("parse", "uploaded-1"),
            ("delete", "uploaded-1"),
        ]
        assert "--- PRIOR VALIDATION FEEDBACK ---" not in instructions[0]
        assert "color" in instructions[1]
        assert "assistant_messages" not in instructions[1]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("async_mode", [False, True])
    async def test_structured_retry_exhaustion_still_deletes_upload(self, tmp_path, async_mode):
        path = tmp_path / "document.pdf"
        path.write_bytes(b"pdf")
        files = SimpleNamespace(
            create=AsyncMock(return_value=SimpleNamespace(id="uploaded-1"))
            if async_mode
            else Mock(return_value=SimpleNamespace(id="uploaded-1")),
            delete=AsyncMock() if async_mode else Mock(),
        )
        responses = SimpleNamespace(
            parse=AsyncMock(side_effect=_media_validation_error())
            if async_mode
            else Mock(side_effect=_media_validation_error())
        )
        client = SimpleNamespace(files=files, responses=responses)
        if async_mode:
            wrapper = AsyncBatchResponses.of(
                client,
                "gpt-4.1-mini",
                "Extract fruit",
                _MediaResult,
                batch_size=0,
                multimodal=True,
                max_validation_retries=2,
            )
            with pytest.raises(ValidationError):
                await wrapper.parse([str(path)])
        else:
            wrapper = BatchResponses.of(
                client,
                "gpt-4.1-mini",
                "Extract fruit",
                _MediaResult,
                batch_size=0,
                multimodal=True,
                max_validation_retries=2,
            )
            with pytest.raises(ValidationError):
                wrapper.parse([str(path)])
        assert responses.parse.call_count == 3
        files.create.assert_called_once()
        files.delete.assert_called_once_with("uploaded-1")
        assert not wrapper.cache._inflight

    @pytest.mark.asyncio
    @pytest.mark.parametrize("async_mode", [False, True])
    @pytest.mark.parametrize("parse_failure", [False, True])
    async def test_cleanup_failure_is_explicit_after_success_or_parse_error(self, tmp_path, async_mode, parse_failure):
        path = tmp_path / "document.pdf"
        path.write_bytes(b"pdf")
        files = SimpleNamespace(
            create=AsyncMock(return_value=SimpleNamespace(id="uploaded-1"))
            if async_mode
            else Mock(return_value=SimpleNamespace(id="uploaded-1")),
            delete=AsyncMock(side_effect=RuntimeError("delete failed"))
            if async_mode
            else Mock(side_effect=RuntimeError("delete failed")),
        )
        parse_result = SimpleNamespace(output_parsed=_MediaResult(name="apple", color="red"))
        responses = SimpleNamespace(
            parse=AsyncMock(side_effect=_media_validation_error() if parse_failure else None, return_value=parse_result)
            if async_mode
            else Mock(side_effect=_media_validation_error() if parse_failure else None, return_value=parse_result)
        )
        client = SimpleNamespace(files=files, responses=responses)
        if async_mode:
            wrapper = AsyncBatchResponses.of(
                client,
                "gpt-4.1-mini",
                "Extract fruit",
                _MediaResult,
                batch_size=0,
                multimodal=True,
                max_validation_retries=0,
            )
            with pytest.raises(RuntimeError, match="delete failed") as caught:
                await wrapper.parse([str(path)])
        else:
            wrapper = BatchResponses.of(
                client,
                "gpt-4.1-mini",
                "Extract fruit",
                _MediaResult,
                batch_size=0,
                multimodal=True,
                max_validation_retries=0,
            )
            with pytest.raises(RuntimeError, match="delete failed") as caught:
                wrapper.parse([str(path)])
        assert isinstance(caught.value.__context__, ValidationError) == parse_failure
        files.delete.assert_called_once_with("uploaded-1")
        assert not wrapper.cache._inflight

    @pytest.mark.asyncio
    @pytest.mark.parametrize("async_mode", [False, True])
    async def test_multimodal_validation_attempts_share_one_transport_deadline(self, monkeypatch, async_mode):
        policy = RetryPolicy(max_attempts=1, max_elapsed=5)
        deadlines = []
        deadline_policies = []

        def deadline_for(current):
            deadline_policies.append(current)
            return 42.0

        def sync_transport(client, current, operation, options, *, deadline=None):
            assert current is policy
            deadlines.append(deadline)
            return operation(client, options)

        async def async_transport(client, current, operation, options, *, deadline=None):
            assert current is policy
            deadlines.append(deadline)
            return await operation(client, options)

        monkeypatch.setattr(_responses, "retry_deadline", deadline_for)
        monkeypatch.setattr(_responses, "call_with_retry", sync_transport)
        monkeypatch.setattr(_responses, "call_with_retry_async", async_transport)
        result = SimpleNamespace(output_parsed=_MediaResult(name="apple", color="red"))
        parse = (
            AsyncMock(side_effect=[_media_validation_error(), result])
            if async_mode
            else Mock(side_effect=[_media_validation_error(), result])
        )
        client = SimpleNamespace(responses=SimpleNamespace(parse=parse))
        input_messages = [
            {"role": "user", "content": [{"type": "input_image", "image_url": "https://example.com/a.png"}]}
        ]
        if async_mode:
            wrapper = AsyncBatchResponses(
                client=client,
                model_name="gpt-4.1-mini",
                system_message="Extract fruit",
                response_format=_MediaResult,
                retry_policy=policy,
                max_validation_retries=1,
            )
            parsed = await wrapper._request_multimodal(input_messages)
        else:
            wrapper = BatchResponses(
                client=client,
                model_name="gpt-4.1-mini",
                system_message="Extract fruit",
                response_format=_MediaResult,
                retry_policy=policy,
                max_validation_retries=1,
            )
            parsed = wrapper._request_multimodal(input_messages)
        assert parsed == _MediaResult(name="apple", color="red")
        assert deadlines == [42.0, 42.0]
        assert deadline_policies == [policy]
        assert parse.call_count == 2

    @pytest.mark.asyncio
    async def test_async_media_concurrency_is_bounded_and_results_keep_input_order(self):
        urls = [f"https://example.com/photo-{i}.png?sig=abc" for i in range(8)]
        release = asyncio.Event()
        at_limit = asyncio.Event()
        active = 0
        peak = 0

        async def create(**kwargs):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            if active == 3:
                at_limit.set()
            try:
                await release.wait()
                return SimpleNamespace(output_text=kwargs["input"][0]["content"][0]["image_url"])
            finally:
                active -= 1

        client = SimpleNamespace(responses=SimpleNamespace(create=AsyncMock(side_effect=create)))
        wrapper = AsyncBatchResponses.of(
            client, "gpt-4.1-mini", "Describe image", batch_size=0, max_concurrency=3, multimodal=True
        )
        pending = asyncio.create_task(wrapper.parse(urls))
        try:
            await asyncio.wait_for(at_limit.wait(), timeout=2)
            assert peak == 3
            assert active == 3
        finally:
            release.set()
        assert await asyncio.wait_for(pending, timeout=2) == urls
        assert peak == 3
        assert active == 0

    @pytest.mark.asyncio
    async def test_parallel_media_error_cancels_peers_and_cleans_their_uploads(self, tmp_path):
        path = tmp_path / "pending.pdf"
        path.write_bytes(b"pdf")
        waiting_on_document = asyncio.Event()
        document_cancelled = asyncio.Event()

        async def create(**kwargs):
            content = kwargs["input"][0]["content"][0]
            if content["type"] == "input_file":
                waiting_on_document.set()
                try:
                    await asyncio.Future()
                except asyncio.CancelledError:
                    document_cancelled.set()
                    raise
            await waiting_on_document.wait()
            raise RuntimeError("image failed")

        client = SimpleNamespace(
            files=SimpleNamespace(
                create=AsyncMock(return_value=SimpleNamespace(id="uploaded-1")),
                delete=AsyncMock(),
            ),
            responses=SimpleNamespace(create=AsyncMock(side_effect=create)),
        )
        wrapper = AsyncBatchResponses.of(
            client, "gpt-4.1-mini", "Describe", batch_size=0, max_concurrency=2, multimodal=True
        )
        with pytest.raises(RuntimeError, match="image failed"):
            await asyncio.wait_for(wrapper.parse([str(path), "https://example.com/fail.png"]), timeout=2)
        assert document_cancelled.is_set()
        client.files.delete.assert_awaited_once_with("uploaded-1")
        assert not wrapper.cache._inflight

    @pytest.mark.asyncio
    @pytest.mark.parametrize("cleanup_fails", [False, True])
    async def test_cancel_during_multimodal_parse_cleans_uploads_and_waiters(self, tmp_path, cleanup_fails):
        path = tmp_path / "document.pdf"
        path.write_bytes(b"pdf")
        started = asyncio.Event()
        delete = AsyncMock(side_effect=RuntimeError("delete failed")) if cleanup_fails else AsyncMock()

        async def parse_media(**kwargs):
            started.set()
            await asyncio.Future()

        client = SimpleNamespace(
            files=SimpleNamespace(create=AsyncMock(return_value=SimpleNamespace(id="uploaded-1")), delete=delete),
            responses=SimpleNamespace(parse=AsyncMock(side_effect=parse_media)),
        )
        wrapper = AsyncBatchResponses.of(
            client,
            "gpt-4.1-mini",
            "Extract fruit",
            _MediaResult,
            batch_size=0,
            max_concurrency=2,
            multimodal=True,
        )
        pending = asyncio.create_task(wrapper.parse([str(path)]))
        await asyncio.wait_for(started.wait(), timeout=2)
        pending.cancel()
        if cleanup_fails:
            with pytest.raises(RuntimeError, match="delete failed"):
                await asyncio.wait_for(pending, timeout=2)
        else:
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(pending, timeout=2)
        delete.assert_awaited_once_with("uploaded-1")
        assert not wrapper.cache._inflight

        client.responses.parse = AsyncMock(
            return_value=SimpleNamespace(output_parsed=_MediaResult(name="apple", color="red"))
        )
        client.files.delete = AsyncMock()
        assert await asyncio.wait_for(wrapper.parse([str(path)]), timeout=2) == [
            _MediaResult(name="apple", color="red")
        ]

    @pytest.mark.asyncio
    async def test_cancel_during_async_upload_deletes_created_file(self, tmp_path):
        path = tmp_path / "document.pdf"
        path.write_bytes(b"pdf")
        upload_started = asyncio.Event()
        finish_upload = asyncio.Event()

        async def upload(*, file, purpose):
            upload_started.set()
            await finish_upload.wait()
            return SimpleNamespace(id="uploaded-1")

        client = SimpleNamespace(
            files=SimpleNamespace(create=AsyncMock(side_effect=upload), delete=AsyncMock()),
            responses=SimpleNamespace(create=AsyncMock()),
        )
        wrapper = AsyncBatchResponses.of(client, "gpt-4.1-mini", "Describe", batch_size=0, multimodal=True)
        pending = asyncio.create_task(wrapper.parse([str(path)]))
        await asyncio.wait_for(upload_started.wait(), timeout=2)
        pending.cancel()
        finish_upload.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(pending, timeout=2)
        client.files.delete.assert_awaited_once_with("uploaded-1")
        client.responses.create.assert_not_called()
        assert not wrapper.cache._inflight

    @pytest.mark.asyncio
    @pytest.mark.parametrize("cleanup_fails", [False, True])
    async def test_cancel_during_cleanup_waits_for_delete_and_reports_failure(self, tmp_path, cleanup_fails):
        path = tmp_path / "document.pdf"
        path.write_bytes(b"pdf")
        delete_started = asyncio.Event()
        finish_delete = asyncio.Event()

        async def delete(file_id):
            assert file_id == "uploaded-1"
            delete_started.set()
            await finish_delete.wait()
            if cleanup_fails:
                raise RuntimeError("delete failed")

        client = SimpleNamespace(
            files=SimpleNamespace(
                create=AsyncMock(return_value=SimpleNamespace(id="uploaded-1")),
                delete=AsyncMock(side_effect=delete),
            ),
            responses=SimpleNamespace(create=AsyncMock(return_value=SimpleNamespace(output_text="done"))),
        )
        wrapper = AsyncBatchResponses.of(client, "gpt-4.1-mini", "Describe", multimodal=True)
        pending = asyncio.create_task(wrapper.parse([str(path)]))
        await asyncio.wait_for(delete_started.wait(), timeout=2)
        pending.cancel()
        finish_delete.set()
        if cleanup_fails:
            with pytest.raises(RuntimeError, match="delete failed"):
                await asyncio.wait_for(pending, timeout=2)
        else:
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(pending, timeout=2)
        client.files.delete.assert_awaited_once_with("uploaded-1")
        assert not wrapper.cache._inflight

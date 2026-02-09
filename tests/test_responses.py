from logging import Handler, StreamHandler, basicConfig
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import BaseModel, ValidationError

from openaivec import BatchResponses
from openaivec._responses import AsyncBatchResponses

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
                        assistant_messages=[
                            SimpleNamespace(id=0, body=Fruit(name="apple", color="red", taste="sweet"))
                        ]
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
                        assistant_messages=[
                            SimpleNamespace(id=0, body=Fruit(name="apple", color="red", taste="sweet"))
                        ]
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
        response: list[str] = client._predict_chunk(["hello", "world"])

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

        response: list[Fruit] = client._predict_chunk(["apple", "banana"])

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
        response: list[str] = client._predict_chunk(test_inputs)

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
        response: list[str] = await client.parse(["apple", "orange", "banana", "pineapple"])
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
        response: list[Fruit] = await client.parse(input_fruits)
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
        response: list[Fruit] = await client.parse([])
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
        response_bs2: list[Fruit] = await client_bs2.parse(input_fruits)
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
        response_bs4: list[Fruit] = await client_bs4.parse(input_fruits)
        assert len(response_bs4) == len(input_fruits)
        for i, item in enumerate(response_bs4):
            assert isinstance(item, Fruit)
            assert item.name.lower() == input_fruits[i].lower()
            assert isinstance(item.color, str)
            assert len(item.color) > 0
            assert isinstance(item.taste, str)
            assert len(item.taste) > 0

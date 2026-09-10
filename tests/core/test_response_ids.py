from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from openai import AsyncOpenAI, OpenAI
from pydantic import ValidationError

from openaivec import AsyncBatchResponses, BatchResponses


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("multimodal", [False, True])
@pytest.mark.parametrize("invalid_ids", [[0, 0], [0, 999], [0], [], [0, 1, 2]])
async def test_invalid_response_ids_do_not_poison_cache(monkeypatch, asynchronous, multimodal, invalid_ids):
    response_ids = invalid_ids

    def respond(**kwargs):
        parsed = kwargs["text_format"].model_validate(
            {"assistant_messages": [{"id": identity, "body": f"value-{identity}"} for identity in response_ids]}
        )
        return SimpleNamespace(output_parsed=parsed)

    client = AsyncOpenAI(api_key="test") if asynchronous else OpenAI(api_key="test")
    parse = AsyncMock(side_effect=respond) if asynchronous else Mock(side_effect=respond)
    monkeypatch.setattr(client.responses, "parse", parse)
    wrapper_type = AsyncBatchResponses if asynchronous else BatchResponses
    wrapper = wrapper_type.of(
        client=client,
        model_name="gpt-4.1-mini",
        system_message="echo",
        max_validation_retries=1,
        multimodal=multimodal,
    )
    try:
        with pytest.raises(ValidationError, match="IDs"):
            result = wrapper.parse(["first", "second", "first"])
            if asynchronous:
                await result
        assert parse.call_count == 2
        assert len(wrapper.cache.cache) == 0
        assert "PRIOR VALIDATION FEEDBACK" in parse.call_args.kwargs["instructions"]
        response_ids = [1, 0]
        result = wrapper.parse(["first", "second", "first"])
        if asynchronous:
            result = await result
        assert result == ["value-0", "value-1", "value-0"]
        assert parse.call_count == 3
    finally:
        if asynchronous:
            await client.close()
        else:
            client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("multimodal", [False, True])
async def test_response_ids_are_corrected_within_validation_budget(monkeypatch, asynchronous, multimodal):
    attempts = iter([[0, 0], [1, 0]])

    def respond(**kwargs):
        parsed = kwargs["text_format"].model_validate(
            {"assistant_messages": [{"id": identity, "body": str(identity)} for identity in next(attempts)]}
        )
        return SimpleNamespace(output_parsed=parsed)

    client = AsyncOpenAI(api_key="test") if asynchronous else OpenAI(api_key="test")
    parse = AsyncMock(side_effect=respond) if asynchronous else Mock(side_effect=respond)
    monkeypatch.setattr(client.responses, "parse", parse)
    wrapper_type = AsyncBatchResponses if asynchronous else BatchResponses
    wrapper = wrapper_type.of(
        client=client,
        model_name="gpt-4.1-mini",
        system_message="echo",
        max_validation_retries=1,
        multimodal=multimodal,
    )
    try:
        result = wrapper.parse(["first", "second"])
        if asynchronous:
            result = await result
        assert result == ["0", "1"]
        assert parse.call_count == 2
        assert "PRIOR VALIDATION FEEDBACK" in parse.call_args.kwargs["instructions"]
        assert parse.call_args_list[0].kwargs["input"] == parse.call_args_list[1].kwargs["input"]
    finally:
        if asynchronous:
            await client.close()
        else:
            client.close()

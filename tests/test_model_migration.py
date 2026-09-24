"""Bounded live checks for the model selected by OPENAIVEC_TEST_RESPONSES_MODEL."""

from typing import Literal

import pytest
from pydantic import BaseModel, ConfigDict

from openaivec import AsyncBatchResponses, BatchResponses


class ReviewSignal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sentiment: Literal["positive", "negative", "neutral"]
    order_id: int


_REVIEWS = [
    ("Order 101: I love this product. Excellent quality!", "positive", 101),
    ("注文番号202。壊れた商品が届いて最悪でした。", "negative", 202),
    ("Order 303 was placed on Monday. No evaluation provided.", "neutral", 303),
    ("注文番号404。とても良い商品で大満足です。", "positive", 404),
    ("Order 505: Terrible service. I am very disappointed.", "negative", 505),
    ("注文番号606。火曜日に注文しました。評価は記載されていません。", "neutral", 606),
]


@pytest.mark.requires_api
@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("effort", [None, "none", "low"])
async def test_bilingual_structured_batch(
    openai_client, async_openai_client, responses_model_name, asynchronous, effort
):
    """Check semantic extraction and duplicate/order preservation on real responses."""
    if effort is not None and not responses_model_name.startswith("gpt-6-"):
        pytest.skip("Explicit GPT-6 reasoning cases; the baseline uses its own defaults")
    options = {"store": False, "max_output_tokens": 4096, "timeout": 90}
    if effort is not None:
        options["reasoning"] = {"effort": effort}
    cls = AsyncBatchResponses if asynchronous else BatchResponses
    wrapper = cls.of(
        client=async_openai_client if asynchronous else openai_client,
        model_name=responses_model_name,
        system_message="Extract the numeric order ID and review sentiment. No evaluation means neutral.",
        response_format=ReviewSignal,
        batch_size=6,
        **options,
    )
    cases = [_REVIEWS[2], *_REVIEWS, _REVIEWS[0]]
    try:
        pending = wrapper.parse([text for text, _, _ in cases])
        results = await pending if asynchronous else pending
        assert results == [ReviewSignal(sentiment=sentiment, order_id=order_id) for _, sentiment, order_id in cases]
    finally:
        if asynchronous:
            await wrapper.cache.aclose()
        else:
            wrapper.cache.clear()

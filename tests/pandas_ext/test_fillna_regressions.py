"""Regression tests for pandas fillna row alignment and few-shot preparation."""

import json
from xml.etree import ElementTree

import pandas as pd
import pytest
import tiktoken

from openaivec import pandas_ext
from openaivec._prompt import FewShotPromptBuilder
from openaivec.pandas_ext._common import _df_rows_to_json_series
from openaivec.task.table import FillNaResponse, fillna


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_fillna_duplicate_labels_only_update_missing_positions(monkeypatch, asynchronous):
    def fake_task(self, task, **kwargs):
        assert self._obj.index.tolist() == [17, 17, 99]
        assert json.loads(_df_rows_to_json_series(self._obj).iloc[0]) == {"value": None, "context": 1}
        return [FillNaResponse(output="first"), FillNaResponse(output=None), FillNaResponse(output="last")]

    async def fake_async_task(self, task, **kwargs):
        return fake_task(self, task, **kwargs)

    monkeypatch.setattr(pandas_ext.OpenAIVecDataFrameAccessor, "task", fake_task)
    monkeypatch.setattr(pandas_ext.AsyncOpenAIVecDataFrameAccessor, "task", fake_async_task)
    df = pd.DataFrame(
        {"value": [None, "keep", None, None, "also keep"], "context": [1, 2, 3, 4, 5]},
        index=[17, 17, 17, 99, 99],
    )
    result = await df.aio.fillna("value") if asynchronous else df.ai.fillna("value")
    assert result.index.tolist() == df.index.tolist()
    assert result["value"].tolist()[:2] == ["first", "keep"]
    assert pd.isna(result["value"].iloc[2])
    assert result["value"].tolist()[3:] == ["last", "also keep"]
    assert pd.isna(df["value"].iloc[0])
    assert df["value"].iloc[1] == "keep"


def test_fillna_preparation_is_local_and_matches_runtime_rows(monkeypatch):
    def fail_improve(self, *args, **kwargs):
        raise AssertionError("default task construction must not call the API")

    monkeypatch.setattr(FewShotPromptBuilder, "improve", fail_improve)
    df = pd.DataFrame({"value": ["A", None, "B"], "context": [1, 2, 3]}, index=[8, 8, 42])
    task = fillna(df, "value")
    assert task.response_format is FillNaResponse
    assert set(FillNaResponse.model_fields) == {"output"}
    example_inputs = [
        node.text for node in ElementTree.fromstring(task.instructions).findall("./Examples/Example/Input")
    ]
    assert example_inputs, task.instructions
    for example in example_inputs:
        assert set(json.loads(example)) == set(json.loads(_df_rows_to_json_series(df.iloc[[1]]).iloc[0]))
        assert json.loads(example)["value"] is None
    for output in ElementTree.fromstring(task.instructions).findall("./Examples/Example/Output"):
        assert set(json.loads(output.text)) == {"output"}


def test_fillna_optional_improvement_and_example_budget(monkeypatch):
    calls = []

    def fake_improve(self, *args, **kwargs):
        calls.append(self)
        return self

    monkeypatch.setattr(FewShotPromptBuilder, "improve", fake_improve)
    df = pd.DataFrame({"value": ["A"] * 2000 + [None], "context": list(range(2001))})
    first = fillna(df, "value", max_examples=1000, improve_prompt=True)
    second = fillna(df, "value", max_examples=1000)
    assert len(calls) == 1
    assert first.instructions == second.instructions
    assert first.instructions.count("<Example>") <= 1000
    assert len(first.instructions) < 8000
    assert len(tiktoken.get_encoding("o200k_base").encode_ordinary(first.instructions)) <= 1800


def test_fillna_nested_values_and_bounded_sampling(monkeypatch):
    def reject_full_shuffle(self, *args, **kwargs):
        raise AssertionError("Do not shuffle the complete DataFrame")

    monkeypatch.setattr(pd.DataFrame, "sample", reject_full_shuffle)
    df = pd.DataFrame(
        {
            "value": ["A", "B", "C", None],
            "items": [[1, 2], [3], [4], [5]],
            "metadata": [{"tag": "a"}, {"tag": "b"}, {"tag": "c"}, {"tag": "missing"}],
        }
    )
    task = fillna(df, "value", max_examples=2)
    assert task.instructions.count("<Example>") == 2
    assert '"items"' in task.instructions
    assert '"metadata"' in task.instructions


def test_fillna_wide_rows_fall_back_to_zero_shot_within_budget():
    df = pd.DataFrame({f"column_{i}": ["x" * 200, "y" * 200] for i in range(60)})
    df["value"] = ["known", None]
    task = fillna(df, "value")
    assert "value" in task.instructions
    assert len(task.instructions) < 1000


def test_fillna_examples_have_a_token_budget_not_just_a_character_budget():
    df = pd.DataFrame(
        {
            "value": ["known", "known", None],
            "context": ["^!@#$%&*?~" * 540, "short", "missing"],
        }
    )

    task = fillna(df, "value", max_examples=2)
    examples = ElementTree.fromstring(task.instructions).findall("./Examples/Example")
    assert len(examples) == 1
    assert len(tiktoken.get_encoding("o200k_base").encode_ordinary(task.instructions)) <= 1800


def test_fillna_examples_accept_text_that_looks_like_a_special_token():
    df = pd.DataFrame({"value": ["known", None], "context": ["<|endoftext|>", "missing"]})
    task = fillna(df, "value")
    example = ElementTree.fromstring(task.instructions).find("./Examples/Example/Input")
    assert json.loads(example.text)["context"] == "<|endoftext|>"

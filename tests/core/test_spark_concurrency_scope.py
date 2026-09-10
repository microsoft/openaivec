import asyncio
import inspect
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pandas as pd
import pytest
from openai import AsyncOpenAI

from openaivec import PreparedTask, SchemaInferenceInput

pytest.importorskip("pyspark")
from openaivec import spark_ext


@pytest.mark.parametrize("name", ["responses_udf", "task_udf", "parse_udf", "embeddings_udf"])
def test_concurrency_documentation_describes_invocation_scope(name):
    documentation = inspect.getdoc(getattr(spark_ext, name))
    assert "partition invocation" in documentation
    assert "PER EXECUTOR" not in documentation
    assert "max_concurrency × executors" not in documentation


@pytest.mark.parametrize("mode", ["responses", "structured", "task", "parse", "embeddings"])
def test_each_partition_invocation_has_independent_concurrency_and_cache(monkeypatch, mode):
    barrier = threading.Barrier(5)
    lock = threading.Lock()
    active = {}
    peaks = {}
    requests = {}
    total_peak = 0
    client = AsyncOpenAI(api_key="test")

    async def request(**options):
        nonlocal total_peak
        identity = threading.get_ident()
        with lock:
            active[identity] = active.get(identity, 0) + 1
            requests[identity] = requests.get(identity, 0) + 1
            peaks[identity] = max(peaks.get(identity, 0), active[identity])
            total_peak = max(total_peak, sum(active.values()))
        try:
            await asyncio.to_thread(barrier.wait, timeout=5)
            if mode == "embeddings":
                return SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[1.0, 0.0])])
            return SimpleNamespace(output_parsed=None)
        finally:
            with lock:
                active[identity] -= 1

    monkeypatch.setattr(client.responses, "parse", request)
    monkeypatch.setattr(client.embeddings, "create", request)
    monkeypatch.setattr(spark_ext, "get_async_client", lambda: client)
    monkeypatch.setattr(spark_ext, "pandas_udf", lambda **options: lambda function: function)
    monkeypatch.setattr(spark_ext.CONTAINER, "is_registered", lambda kind: False)
    options = {"model_name": "test-model", "batch_size": 1, "max_concurrency": 2}
    if mode == "embeddings":
        udf = spark_ext.embeddings_udf(**options)
    elif mode == "task":
        udf = spark_ext.task_udf(PreparedTask("echo", str), **options)
    elif mode == "parse":
        udf = spark_ext.parse_udf("echo", response_format=str, **options)
    else:
        udf = spark_ext.responses_udf(
            "echo", response_format=SchemaInferenceInput if mode == "structured" else str, **options
        )
    data = pd.Series(["first", "second", "first"], index=[9, 3, 7])
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(list, udf(iter([data, data]))) for _ in range(2)]
            barrier.wait(timeout=5)
            outputs = [future.result(timeout=5) for future in futures]
        assert [[len(batch) for batch in output] for output in outputs] == [[3, 3], [3, 3]]
        assert sorted(peaks.values()) == [2, 2]
        assert sorted(requests.values()) == [2, 2]
        assert total_peak == 4
        assert not any(active.values())
        assert not client.is_closed()
    finally:
        asyncio.run(client.close())

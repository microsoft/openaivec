# Embedding Request Limits

`BatchEmbeddings` and `AsyncBatchEmbeddings` split cache batches into requests
with at most **2,048 inputs**, **8,192 tokens per input**, and **300,000 total
tokens**. These defaults match the
[OpenAI Embeddings contract](https://developers.openai.com/api/reference/resources/embeddings/methods/create).

`batch_size=None` still enables adaptive batching. Positive sizes remain cache
chunk sizes, and nonpositive sizes still select all unique inputs at the cache
layer. In every mode the embedding layer may split that chunk further. These
limits do not constrain generic `BatchCache` use. Splits retain input order,
deduplication, and the existing concurrency bound; responses are ordered by
their SDK `index` before caching.

Empty strings or a single input exceeding either token limit raise `ValueError`
before sending any requests for the affected cache chunk. Text is never
silently truncated. Earlier chunks in a long operation may already have run.

For a compatible provider with different limits, pass `limits=EmbeddingLimits(...)`
to either core factory or constructor, pandas `.ai` / `.aio` `embeddings()` or
`embeddings_with_cache()`, or Spark/DuckDB `embeddings_udf()`. All numeric limits
must be positive. This option is not sent to the SDK as an API parameter.

```python
from openaivec import BatchEmbeddings, EmbeddingLimits

limits = EmbeddingLimits(
    max_inputs=512,
    max_input_tokens=4096,
    max_request_tokens=100000,
    encoding_name="cl100k_base",
)
embedder = BatchEmbeddings.of(client, "my-embedding-deployment", limits=limits)
```

By default, token counting uses tiktoken's model encoding. Unknown deployment
names use `cl100k_base`, shared by the supported OpenAI embedding models. Custom
providers must specify the correct encoding and limits; these local checks do
not replace provider quotas or rate limits.

::: openaivec.EmbeddingLimits

::: openaivec.BatchEmbeddings

::: openaivec.AsyncBatchEmbeddings
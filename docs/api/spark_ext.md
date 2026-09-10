# Spark Extension

## Concurrency and Cache Scope

`max_concurrency=8` limits concurrent batch requests within one partition
invocation of one UDF. It is not an executor-wide or cluster-wide limit.
With `P` simultaneous invocations at the same setting, the aggregate upper
bound is `max_concurrency * P`. One executor can run several tasks, so executor
count alone is not enough to calculate this bound.

The invocation reuses its limiter, event loop, and bounded cache across Arrow
batches. Separate invocations do not share caches or a limiter, even when they
use the same UDF or encounter identical inputs. Repeated Spark actions, task
retries, speculative attempts, and multiple UDF expressions can cause new API
requests; caching is not an exactly-once execution guarantee.

Start with `max_concurrency=1` for a small validation workload, then tune it
against simultaneous task slots and provider request/token quotas. Lower the
per-invocation limit or reduce active Spark task slots when necessary. This is
a concurrency control, not a requests-per-second limiter. A cluster-wide quota
requires coordination outside these UDFs. `batch_size` and embedding request
limits determine request contents independently.

## Fabric Built-in Models

Call `openaivec.spark_ext.setup_fabric(spark)` before creating AI UDFs in a Fabric
notebook. Install the package in a published Fabric Environment attached to the
notebook so the driver and Python workers use the same dependencies. Do not install
the `spark` extra over Fabric's managed PySpark.

```python
from openaivec.spark_ext import embeddings_udf, setup_fabric

setup_fabric(spark)
spark.udf.register("ai_embeddings", embeddings_udf(batch_size=2, max_concurrency=1))
result = spark.sql("""
	SELECT id, text, ai_embeddings(text) AS embedding
	FROM VALUES (0, 'apple'), (1, 'banana'), (2, 'apple') AS inputs(id, text)
""")
result.show()
```

Register each UDF once per Spark session before calling it from `spark.sql` or a
notebook `%%sql` cell. Importing `openaivec` does not select an authentication route
or register SQL functions automatically. These are Spark SQL functions, not
T-SQL functions in the Lakehouse SQL analytics endpoint.

Each partition owns its Fabric async client and closes it with its event loop.
Clients supplied through the existing OpenAI/Azure configuration remain caller-owned.
See [Fabric authentication](../authentication.md#spark-udfs) for setup, model defaults,
and preview limitations.

::: openaivec.spark_ext
# Spark Extension

## Fabric Built-in Models

Call `openaivec.spark_ext.setup_fabric(spark)` before creating AI UDFs in a Fabric
notebook. Install the package in a published Fabric Environment attached to the
notebook so the driver and Python workers use the same dependencies. Do not install
the `spark` extra over Fabric's managed PySpark.

```python
from openaivec.spark_ext import embeddings_udf, setup_fabric

setup_fabric(spark)
embed = embeddings_udf(batch_size=2, max_concurrency=1)
result = df.withColumn("embedding", embed("text"))
```

Each partition owns its Fabric async client and closes it with its event loop.
Clients supplied through the existing OpenAI/Azure configuration remain caller-owned.
See [Fabric authentication](../authentication.md#spark-udfs) for setup, model defaults,
and preview limitations.

::: openaivec.spark_ext
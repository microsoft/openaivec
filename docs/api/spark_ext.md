# Spark Extension

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
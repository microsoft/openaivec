"""DuckDB integration for openaivec.

Provides helpers that bridge openaivec's batched AI capabilities with DuckDB:

- **UDF registration** – register ``responses``, ``embeddings``, ``task``,
  ``parse`` and ``count_tokens``
  functions directly as DuckDB scalar UDFs for SQL queries.
- **Schema inference** – infer a Pydantic response model once from a bounded
  table sample before registering a typed ``parse`` UDF.
- **Persistent caching** – pass ``DuckDBCacheBackend`` as the ``cache`` field
  of ``BatchCache`` for cross-session cache persistence.
- **Vector similarity** – ``similarity_search`` performs top-k cosine similarity
  queries against an embedding table using DuckDB's built-in
  ``list_cosine_similarity``.
- **Schema → DDL** – ``pydantic_to_duckdb_ddl`` converts a Pydantic model to a
  ``CREATE TABLE`` statement for immediate SQL analysis of structured-output
  results.

## Quick Start

```python
import duckdb
from openaivec.duckdb_ext import responses_udf, embeddings_udf

conn = duckdb.connect()
responses_udf(conn, "translate", instructions="Translate to French", reasoning={"effort": "none"})
embeddings_udf(conn, "embed")

conn.sql("SELECT translate(review) FROM products")
conn.sql("SELECT text, embed(text) FROM documents")
```
"""

from openaivec.duckdb_ext._cache import DuckDBCacheBackend
from openaivec.duckdb_ext._schema import infer_schema, parse_udf
from openaivec.duckdb_ext._similarity import similarity_search
from openaivec.duckdb_ext._tokens import count_tokens_udf
from openaivec.duckdb_ext._types import (
    _pydantic_to_struct_type as _pydantic_to_struct_type,
)
from openaivec.duckdb_ext._types import (
    _python_type_to_duckdb as _python_type_to_duckdb,
)
from openaivec.duckdb_ext._types import (
    _serialize_for_duckdb as _serialize_for_duckdb,
)
from openaivec.duckdb_ext._types import (
    pydantic_to_duckdb_ddl,
)
from openaivec.duckdb_ext._udfs import embeddings_udf, responses_udf, task_udf

__all__ = [
    "DuckDBCacheBackend",
    "count_tokens_udf",
    "infer_schema",
    "parse_udf",
    "pydantic_to_duckdb_ddl",
    "embeddings_udf",
    "responses_udf",
    "task_udf",
    "similarity_search",
]

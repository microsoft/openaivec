"""Assistant-only, in-memory orchestration for a registered openaivec Arrow UDF."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from time import monotonic
from typing import Any

import duckdb

__all__ = []


def _identifier(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise ValueError("Use a simple local identifier for source, function, and run names.")
    return f'"{value}"'


def _fetch_row(cursor: duckdb.DuckDBPyConnection | duckdb.DuckDBPyRelation) -> tuple[Any, ...]:
    row = cursor.fetchone()
    if row is None:
        raise ValueError("A local run validation query returned no row.")
    return row


@dataclass(frozen=True)
class BulkPlan:
    source_rows: int
    non_null_rows: int
    unique_inputs: int

    @property
    def repeated_evaluations_avoided(self) -> int:
        return self.non_null_rows - self.unique_inputs


@dataclass(frozen=True)
class BulkProgress:
    phase: str
    completed_inputs: int
    total_inputs: int
    succeeded_inputs: int
    unresolved_inputs: int
    elapsed_seconds: float


@dataclass
class BulkRunner:
    """Keep accepted pilot and completed chunk results outside the bounded UDF cache.

    The caller must register an openaivec UDF, stage an immutable temporary table
    with unique non-NULL ``row_id`` and VARCHAR ``input_text`` columns, and obtain
    user consent before setting either approval flag. No file is written here.
    """

    conn: duckdb.DuckDBPyConnection
    source_table: str
    udf_name: str
    prefix: str
    checkpoint_size: int
    plan: BulkPlan
    on_progress: Callable[[BulkProgress], None] | None
    clock: Callable[[], float]
    _elapsed_seconds: float = field(default=0.0, init=False)
    _completed_inputs: int = field(default=0, init=False)
    _succeeded_inputs: int = field(default=0, init=False)
    _pilot_ready: bool = field(default=False, init=False)
    _complete: bool = field(default=False, init=False)

    @classmethod
    def of(
        cls,
        conn: duckdb.DuckDBPyConnection,
        *,
        source_table: str,
        udf_name: str,
        prefix: str = "bulk",
        checkpoint_size: int = 2048,
        on_progress: Callable[[BulkProgress], None] | None = None,
        clock: Callable[[], float] = monotonic,
    ) -> BulkRunner:
        """Prepare local work IDs and typed empty results without invoking the UDF.

        Args:
            conn (duckdb.DuckDBPyConnection): In-memory processing connection with disk spill disabled.
            source_table (str): Immutable temporary staging table.
            udf_name (str): Already registered public openaivec UDF.
            prefix (str): Unused namespace for temporary run tables.
            checkpoint_size (int): Maximum distinct inputs in a checkpoint, not the API batch size.
            on_progress (Callable | None): Structured events for the assistant to translate.
            clock (Callable): Monotonic processing timer.

        Returns:
            BulkRunner: Prepared run; no remote work or persistent output.

        Raises:
            ValueError: Unsafe options, staging, or an occupied namespace.
        """
        source = _identifier(source_table)
        udf = _identifier(udf_name)
        _identifier(prefix)
        if prefix == source_table:
            raise ValueError("The run prefix must differ from the source table.")
        if isinstance(checkpoint_size, bool) or not isinstance(checkpoint_size, int) or checkpoint_size <= 0:
            raise ValueError("checkpoint_size must be a positive integer.")
        if _fetch_row(conn.sql("SELECT current_setting('temp_directory')"))[0]:
            raise ValueError("Disable disk spill, or use a separately approved disk-backed workflow.")
        is_temporary = _fetch_row(
            conn.execute(
                """
                SELECT count(*) FROM duckdb_tables()
                WHERE database_name = 'temp' AND schema_name = 'main' AND table_name = ? AND temporary
                """,
                [source_table],
            )
        )[0]
        if is_temporary != 1:
            raise ValueError("The source must be a materialized temporary table.")
        staged = f"temp.main.{source}"
        relation = conn.sql(f"SELECT * FROM {staged} LIMIT 0")
        schema = dict(zip(relation.columns, relation.types))
        if "row_id" not in schema or "input_text" not in schema:
            raise ValueError("The source needs row_id and input_text columns.")
        if str(schema["input_text"]) != "VARCHAR":
            raise ValueError("input_text must be VARCHAR; prepare an explicit approved input first.")
        if "ai_result" in schema:
            raise ValueError("ai_result is reserved for the new result; preserve existing fields under another name.")
        cls._check_keys(conn, staged)
        names = [f"{prefix}_work", f"{prefix}_results", f"{prefix}_processed"]
        existing = _fetch_row(
            conn.execute(
                """
                SELECT count(*) FROM (
                    SELECT table_name AS name FROM duckdb_tables()
                    UNION ALL SELECT view_name AS name FROM duckdb_views()
                ) WHERE name IN (?, ?, ?)
                """,
                names,
            )
        )[0]
        if existing:
            raise ValueError("A run table already exists; use a new run prefix without replacing it.")
        work, results = (_identifier(name) for name in names[:2])
        conn.execute(
            f"""
            CREATE TEMP TABLE {work} AS
            SELECT row_number() OVER (ORDER BY input_text) AS work_id, input_text
            FROM (SELECT DISTINCT input_text FROM {staged} WHERE input_text IS NOT NULL)
            """
        )
        conn.execute(
            f"""
            CREATE TEMP TABLE {results} AS
            SELECT work_id, input_text, {udf}(input_text) AS ai_result FROM {work} WHERE false
            """
        )
        conn.execute(f"ALTER TABLE {results} ADD PRIMARY KEY (work_id)")
        source_rows, non_null_rows = _fetch_row(conn.sql(f"SELECT count(*), count(input_text) FROM {staged}"))
        unique_inputs = _fetch_row(conn.sql(f"SELECT count(*) FROM {work}"))[0]
        return cls(
            conn=conn,
            source_table=source_table,
            udf_name=udf_name,
            prefix=prefix,
            checkpoint_size=checkpoint_size,
            plan=BulkPlan(source_rows, non_null_rows, unique_inputs),
            on_progress=on_progress,
            clock=clock,
        )

    @staticmethod
    def _check_keys(conn: duckdb.DuckDBPyConnection, source: str) -> None:
        total, non_null, distinct = _fetch_row(
            conn.sql(f"SELECT count(*), count(row_id), count(DISTINCT row_id) FROM {source}")
        )
        if non_null != total:
            raise ValueError("row_id must be non-NULL for every source row.")
        if distinct != total:
            raise ValueError("row_id must be unique; keep the business key in a separate column.")

    def _table(self, suffix: str) -> str:
        return f"temp.main.{_identifier(f'{self.prefix}_{suffix}')}"

    def progress(self, phase: str = "processing") -> BulkProgress:
        """Read committed progress locally; a missing result is unresolved, not retried."""
        return BulkProgress(
            phase=phase,
            completed_inputs=self._completed_inputs,
            total_inputs=self.plan.unique_inputs,
            succeeded_inputs=self._succeeded_inputs,
            unresolved_inputs=self._completed_inputs - self._succeeded_inputs,
            elapsed_seconds=self._elapsed_seconds,
        )

    def _emit(self, phase: str) -> None:
        if self.on_progress is not None:
            self.on_progress(self.progress(phase))

    def _evaluate(self, work_ids: list[int], phase: str) -> None:
        committed = {
            row[0]
            for row in self.conn.execute(
                f"SELECT work_id FROM {self._table('results')} WHERE work_id IN (SELECT unnest(?::BIGINT[]))",
                [work_ids],
            ).fetchall()
        }
        work_ids = [work_id for work_id in work_ids if work_id not in committed]
        if not work_ids:
            return
        self._emit(phase)
        started = self.clock()
        try:
            self.conn.execute(
                f"""
                INSERT INTO {self._table("results")}
                SELECT w.work_id, w.input_text, {_identifier(self.udf_name)}(w.input_text) AS ai_result
                FROM {self._table("work")} AS w
                WHERE w.work_id IN (SELECT unnest(?::BIGINT[]))
                ORDER BY w.work_id
                """,
                [work_ids],
            )
        finally:
            self._elapsed_seconds += self.clock() - started
        completed, succeeded = _fetch_row(
            self.conn.sql(
                f"""
                SELECT count(*), count(ai_result) FROM {self._table("results")}
                WHERE work_id IN (SELECT unnest(?::BIGINT[]))
                """,
                params=[work_ids],
            )
        )
        if completed != len(work_ids):
            raise ValueError("The checkpoint mapping is incomplete; do not continue or export.")
        self._completed_inputs += completed
        self._succeeded_inputs += succeeded
        self._emit(phase)

    def pilot(
        self, *, approved: bool = False, size: int = 5, work_ids: Sequence[int] | None = None
    ) -> duckdb.DuckDBPyRelation:
        """Materialize up to ten approved pilot inputs; re-reading them is local."""
        if not approved:
            raise PermissionError("Obtain remote-call approval before the pilot.")
        if isinstance(size, bool) or not isinstance(size, int) or not 1 <= size <= 10:
            raise ValueError("The pilot size must be between 1 and 10.")
        if work_ids is None:
            selected = [
                row[0]
                for row in self.conn.execute(
                    f"""
                    SELECT work_id FROM (
                        SELECT work_id, row_number() OVER (PARTITION BY bucket ORDER BY work_id) AS candidate
                        FROM (
                            SELECT work_id, ntile(?) OVER (ORDER BY length(input_text), work_id) AS bucket
                            FROM {self._table("work")}
                        )
                    ) WHERE candidate = 1 ORDER BY work_id
                    """,
                    [size],
                ).fetchall()
            ]
        else:
            selected = list(work_ids)
            if (
                not 1 <= len(selected) <= 10
                or any(isinstance(i, bool) or not isinstance(i, int) for i in selected)
                or len(set(selected)) != len(selected)
                or any(i < 1 or i > self.plan.unique_inputs for i in selected)
            ):
                raise ValueError("Choose one to ten distinct valid pilot work IDs.")
        self._evaluate(selected, "pilot")
        self._pilot_ready = True
        return self.conn.sql(
            f"SELECT * FROM {self._table('results')} WHERE work_id IN (SELECT unnest(?::BIGINT[])) ORDER BY work_id",
            params=[selected],
        )

    def run(self, *, pilot_approved: bool = False) -> duckdb.DuckDBPyRelation:
        """Process only pending inputs after approval, then restore every source row.

        Transport failures propagate. Committed checkpoints remain in this process;
        rerunning after an explicit recovery decision does not repeat those inputs.
        A failed statement may already have sent billable requests.
        """
        if not pilot_approved:
            raise PermissionError("Obtain pilot acceptance and full run approval before continuing.")
        if self.plan.unique_inputs and not self._pilot_ready:
            raise ValueError("Complete a pilot before the full run.")
        if self._complete:
            return self.result()
        source = f"temp.main.{_identifier(self.source_table)}"
        self._check_keys(self.conn, source)
        total, non_null, distinct = _fetch_row(
            self.conn.sql(f"SELECT count(*), count(input_text), count(DISTINCT input_text) FROM {source}")
        )
        if BulkPlan(total, non_null, distinct) != self.plan:
            raise ValueError("The staged source changed; prepare a new run and pilot.")
        unmatched = _fetch_row(
            self.conn.sql(
                f"""
                SELECT count(*) FROM {source} AS s
                ANTI JOIN {self._table("work")} AS w USING (input_text)
                WHERE s.input_text IS NOT NULL
                """
            )
        )[0]
        if unmatched:
            raise ValueError("The staged source changed; prepare a new run and pilot.")
        for first in range(1, self.plan.unique_inputs + 1, self.checkpoint_size):
            pending = list(range(first, min(first + self.checkpoint_size, self.plan.unique_inputs + 1)))
            self._evaluate(pending, "processing")
        mapped_inputs = _fetch_row(self.conn.sql(f"SELECT count(*) FROM {self._table('results')}"))[0]
        if mapped_inputs != self.plan.unique_inputs or self._completed_inputs != mapped_inputs:
            raise ValueError("The mapping is incomplete; no final result was materialized.")
        self._emit("restoring")
        self.conn.execute(
            f"""
            CREATE TEMP TABLE {_identifier(f"{self.prefix}_processed")} AS
            SELECT s.*, r.ai_result FROM {source} AS s
            LEFT JOIN {self._table("results")} AS r USING (input_text)
            """
        )
        output_rows = _fetch_row(self.conn.sql(f"SELECT count(*) FROM {self._table('processed')}"))[0]
        if output_rows != self.plan.source_rows:
            raise ValueError("Source and result counts differ; do not export this run.")
        self._complete = True
        self._emit("complete")
        return self.result()

    def result(self) -> duckdb.DuckDBPyRelation:
        """Return the validated temporary result in staged order without remote work."""
        if not self._complete:
            raise ValueError("The run is not complete; a partial result is not a final output.")
        return self.conn.table(self._table("processed")).order("row_id")

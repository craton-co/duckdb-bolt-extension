# W1-A-006: Temporal pushdown gaps (TIMESTAMPTZ, TIMESTAMP_S/MS, DATE vs TIMESTAMP)

- **Severity:** low (performance; results are correct)
- **Status:** blocked-upstream
- **Files:** craton-bolt SQL frontend; `crates/core/src/table_functions.rs` (`tz_constants_safe`, `value_to_bolt_sql_literal`)

## Description

The engine's `TIMESTAMP '…'` literal is `Timestamp(Nanosecond, None)`. The engine refuses to
compare it with:

- `Timestamp(Microsecond, Some("UTC"))`: what `TIMESTAMPTZ` becomes once W1-C keeps the
  time zone;
- `Timestamp(Second | Millisecond, None)`: `TIMESTAMP_S` and `TIMESTAMP_MS` columns;
- `Date32`.

These predicates are rejected by the EXPLAIN validation in `pushdown_filter` and evaluated by
DuckDB. The engine has no `TIMESTAMPTZ` literal: `TIMESTAMPTZ '… +00'` is a parse error and
`CAST(… AS TIMESTAMPTZ)` is unsupported.

What W1-A guarantees:

- TIMESTAMPTZ constants are rendered from the raw UTC epoch value, not the session-time-zone
  display string.
- They are only pushed when compared directly with a column stored as `Timestamp(us, UTC)`.
- Casts between TIMESTAMP and TIMESTAMPTZ are never stripped, because they depend on the time
  zone.

So the pushdown is correct whether the column comes back naive or UTC.

Caveat for the host fallback path (`execute_sql_host_fallback`, used only after two GPU
failures): the embedded DuckDB evaluates a naive `TIMESTAMP` literal against a TIMESTAMPTZ
column in *its* session time zone. If such a predicate is ever pushed, the fallback should run
with `TimeZone='UTC'`.

## Suggested fix

- Engine: coerce `TIMESTAMP` literals to the column unit and accept UTC-zoned columns. Add a
  `TIMESTAMPTZ` literal.
- Runtime (W1-B): set `TimeZone='UTC'` on the fallback DuckDB connection.

## W2-C note (RFC-008)

COPY TO now records `duckdb.logical_type = TIMESTAMPTZ` on the naive `Timestamp(us, None)` storage
column, and `arrow_bridge::arrow_field_to_duckdb_type` maps it (and any `Timestamp(_, Some(tz))`
column) to `TIMESTAMPTZ`. When W2-C-001 wires that into `bolt_scan`, pushed `TIMESTAMPTZ`
constants must be rendered as UTC values against the naive storage column; see W2-C-001 step 5.
## W2-B notes

- DATE vs TIMESTAMP is not a gap in practice. DuckDB's optimizer folds
  `CAST(dt AS TIMESTAMP) op TIMESTAMP 'c'` into a DATE comparison before filter pushdown, so
  the engine receives `dt op DATE '...'`. `w2b_date_column_vs_timestamp_constant` checks every
  operator in both orientations (including non-midnight and pre-1970 constants) and asserts
  the pushed SQL.
- TIMESTAMP_S, TIMESTAMP_MS and TIMESTAMPTZ predicates are still not pushed, because the engine
  has no literal of those types.

## W3-B notes (commit 71e05e7)

- RFC-008 is wired (W2-C-001): `TIMESTAMPTZ` columns are declared TIMESTAMPTZ and
  `tz_constants_safe` translates TIMESTAMPTZ constants as UTC `TIMESTAMP '...'` literals
  against the naive UTC storage column (correct on the engine and on the host fallback, whose
  mirror uses storage types; see W3-B-002).
- Measured on craton-bolt 0.9.0 (RTX 2060): `bolt_explain` of `"tstz" = TIMESTAMP '...'` (and
  of `"ts" = TIMESTAMP '...'` on a plain TIMESTAMP column) fails with
  `cannot compare Timestamp(Microsecond, None) with Timestamp(Nanosecond, None)`. So **no**
  microsecond timestamp predicate is pushed today (TIMESTAMP, TIMESTAMPTZ, TIMESTAMP_S/MS); the
  plan check keeps them with DuckDB and results are correct. DATE predicates are pushed.
- Engine fix needed: coerce `TIMESTAMP` literals to the column's unit.

Upstream report: [craton-bolt-005](../upstream/craton-bolt-005-timestamp-literal-unit.md) (W4-D).
## W4-B re-check (RTX 2060, craton-bolt 0.9.0): blocked-upstream

`bolt_explain` on the engine, unchanged:
- `ts = TIMESTAMP '...'` (TIMESTAMP column) -> `cannot compare Timestamp(Microsecond, None) with Timestamp(Nanosecond, None)`;
- `tsms = TIMESTAMP '...'` (TIMESTAMP_MS) -> `cannot compare Timestamp(Millisecond, None) with Timestamp(Nanosecond, None)`;
- `tstz = TIMESTAMP '...'` (TIMESTAMPTZ, naive UTC storage) -> same as TIMESTAMP;
- `dt = TIMESTAMP '...'` (DATE column, when DuckDB does not fold the cast) -> `cannot compare Date32 with Timestamp(Nanosecond, None)`;
- `tstz = TIMESTAMPTZ '... +00'` -> `TIMESTAMP literal ... has malformed time component`;
- `dt = DATE '...'` plans (`Filter: (dt = Date32(19725))`).

So only DATE predicates reach the engine. DuckDB evaluates every timestamp predicate, and the
results are correct: `test/sql/bolt_timestamptz.test` and the W4-B fuzzer check TIMESTAMP,
TIMESTAMP_MS and TIMESTAMPTZ predicates in three session time zones. This waits for an engine
that coerces `TIMESTAMP` literals to the column unit (W1-O-003).

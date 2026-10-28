# Craton Bolt DuckDB Extension — SQL Reference

Complete SQL reference for the `bolt` DuckDB extension by [Craton Software Company](https://craton.com.ar).
Behavior described here is that of `duckdb-bolt-extension` 0.1.x with `craton-bolt` 0.9.0.

---

## 1. Loading the Extension

```sql
LOAD 'build/release/bolt.duckdb_extension';   -- duckdb -unsigned, DuckDB v2.0.0 only
```

---

## 2. Concepts

### 2.1 Table names

- The Bolt catalog is **process-wide**: every connection and every DuckDB database in the process
  sees the same tables.
- Table references (`bolt_scan`, `bolt_aggregate`, `bolt_drop_table`, `bolt_register_sql`,
  `bolt_load_*`, the `TABLE` COPY option, replacement scans) strip surrounding quotes and one
  leading `bolt://`, `bolt:`, `bolt.main.` or `bolt.` prefix (any case), nothing else. `tmp_x` and
  `x` are different tables.
- Names are matched exactly first, then ASCII case-insensitively.
- Names starting with `__bolt_` (any case) are reserved for internal use and rejected.
- Column names must be non-empty and unique; names that differ only in case are allowed and are
  exposed by `bolt_scan` with a `_<n>` suffix (`ID`, `id_1`).

### 2.2 Types (RFC-008)

Each column is stored in a Craton Bolt storage type and records its original DuckDB type in the
Arrow field metadata key `duckdb.logical_type`:

| DuckDB type | Storage type | Original type kept by `bolt_scan` / `COPY FROM` / `bolt_tables()` |
| :--- | :--- | :--- |
| `BOOLEAN` | `Boolean` | yes |
| `TINYINT`, `SMALLINT`, `INTEGER`, `UTINYINT`, `USMALLINT` | `Int32` | yes |
| `BIGINT`, `UINTEGER`, `UBIGINT`, `HUGEINT` | `Int64` (`UBIGINT` / `HUGEINT` values outside `i64` are an error) | yes |
| `FLOAT`, `DOUBLE` | `Float32`, `Float64` | yes |
| `VARCHAR` | `Utf8` | yes |
| `DATE` | `Date32` | yes |
| `TIMESTAMP_S` / `TIMESTAMP_MS` / `TIMESTAMP` / `TIMESTAMP_NS` | `Timestamp(s / ms / us / ns, None)` | yes |
| `TIMESTAMPTZ` / `TIMESTAMP_TZ_NS` | `Timestamp(us / ns, None)`, the UTC instant | yes |
| `DECIMAL(1..38, s)` | `Decimal128(p, s)` | yes |

Every other type is rejected. All conversions are checked: a value that does not fit the target is
an error, never `NULL`.

- `bolt_scan`, replacement scans and `COPY ... FROM` work with the original types.
  `bolt_tables().schema` shows the original type (`v: SMALLINT`) where one is recorded, else the
  storage type (`v: Int32`).
- `bolt_query`, `bolt_sql` and `bolt_aggregate` return DuckDB's result types for the query over
  the original column types (5), except that `TIMESTAMPTZ` / `TIMESTAMP_TZ_NS` columns are
  computed on as the naive UTC `TIMESTAMP` / `TIMESTAMP_NS` they are stored as.
- Tables registered without metadata (`bolt_register_sql`, the Rust API) expose their storage
  types; a zoned Arrow timestamp is exposed as `TIMESTAMPTZ`.

### 2.3 Backends

| Backend | When | Dialect |
| :--- | :--- | :--- |
| `cuda-gpu` | A CUDA device is visible, the build has no `cuda-stub`, and the engine guard accepts the query | Craton Bolt SQL |
| `host-planner-fallback` | No GPU engine, or the guard routes the query, or the engine rejects / fails it | DuckDB SQL, one `SELECT` |

The guard sends a query to the host fallback when craton-bolt 0.9.0 could crash on it or
compute it differently from DuckDB. Its rules, in the order they are checked:

1. The query names a table the engine does not hold (over `bolt_memory_budget`, more than 2 GiB
   of strings in a column, or a failed upload).
2. It has more than 4,096 `AND` / `OR` terms.
3. Bolt's planner cannot plan it. The guard fails closed here: the engine would run such shapes
   (`WITH RECURSIVE`, `LATERAL`, correlated subqueries, `VALUES`, `generate_series`,
   `DISTINCT ON`, `COUNT(DISTINCT)` with `GROUP BY`) as host-orchestrated GPU sub-plans that the
   guard cannot check.
4. It has a `LIKE` / string-ordering predicate over a table with more than 4,096 distinct strings
   in a column.
5. Type hazards, whatever the NULLs, checked before and after Bolt's optimizer (so constant
   folding cannot hide them):
   - a cast from or to `DECIMAL`, or arithmetic on `DECIMAL` values;
   - arithmetic on a column stored widened (`TINYINT`, `SMALLINT`, unsigned, `HUGEINT`), which
     then runs on the original types so overflow is an error as in DuckDB;
   - arithmetic on `FLOAT` values, and a cast from `FLOAT` / `DOUBLE` to an integer (the engine
     truncates, DuckDB rounds);
   - integer `/` and `%` (DuckDB's `/` is float division, and `x % 0` is an error);
   - arithmetic between a fractional literal (`0.1`, a `DECIMAL` in DuckDB) and an integer;
   - `TIMESTAMP - TIMESTAMP`;
   - an ungrouped `count(*)` next to `SUM` of an `INTEGER` column;
   - an ungrouped `SUM` / `MIN` / `MAX` / `STDDEV_POP` over an input that may be empty (a filter,
     a join, `LIMIT 0`, an `OFFSET`, `EXCEPT` / `INTERSECT`, an empty table);
   - `SUM` of a `FLOAT`, and an integer `SUM` / `AVG` whose total may exceed the `BIGINT` range.
6. No referenced table holds a NULL, but the query can produce one (outer join, `CASE`, `NULL`
   literal) and its plan is not null-safe (7).
7. A referenced table holds a NULL, and the plan is not null-safe or has a subquery. Null-safe
   means: scans, projections of plain columns and literals, `LIMIT`, `DISTINCT`, `UNION ALL`,
   inner / cross equi-joins without an extra filter; sorts whose `NULLS FIRST` keys (the
   engine's default for ascending keys) cannot hold NULLs; `COUNT` / `AVG` of plain columns and
   ungrouped `SUM` / `MIN` / `MAX` of plain columns; grouped `SUM` / `MIN` / `MAX` only of columns
   that cannot hold NULLs. An aggregate of a column that may hold NULLs is null-safe only directly
   over the scan (not below a sort, `LIMIT` or `DISTINCT`), and never next to an ungrouped
   `count(*)`. Whether a column may hold NULLs is followed
   through the plan (aliases, `UNION ALL` branches, NULL literals). Filters, window functions and
   `EXCEPT` / `INTERSECT` are never null-safe.

See [ARCHITECTURE.md](ARCHITECTURE.md#4-gpu-engine-guard-and-host-fallback).

---

## 3. `COPY ... (FORMAT bolt)`

### 3.1 Upload (`COPY ... TO`)

```sql
COPY lineitem TO 'gpu_lineitem' (FORMAT bolt);

COPY (
    SELECT l_orderkey, l_quantity, l_extendedprice
    FROM lineitem
    WHERE l_shipdate >= DATE '1995-01-01'
) TO 'bolt://lineitem_1995.bolt' (FORMAT bolt, TABLE 'gpu_lineitem_1995', MODE 'replace');
```

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `TABLE` (or `TABLE_NAME`) | `VARCHAR` | derived from the target path | Bolt table name, resolved like every table reference (2.1): never path-cleaned. Giving both spellings is an error. |
| `MODE` | `VARCHAR` | `'replace'` | `'replace'` (alias `'overwrite'`), `'append'`, or `'create'` (alias `'error'`, fails if the table exists); case-insensitive. |

- Without `TABLE`, the name is the target path as a table reference without its directory part
  and a trailing `.bolt`: `'dir/x.bolt'` registers `x`, `'tmp_y'` registers `tmp_y`.
- `MODE 'append'` creates the table if it does not exist. Appended columns are matched **by name**
  (exact, then case-insensitive); a missing or extra column is an error. Values are cast to the
  stored types with checked casts and must also fit the column's original type.
- Any other option, a `NULL` option value, and multi-file output (`PARTITION_BY`,
  `PER_THREAD_OUTPUT`, `FILE_SIZE_BYTES`) are rejected, before anything is registered.
- `USE_TMP_FILE` is rejected with an explanatory error: a Bolt table is not a file, so DuckDB's
  rename would fail after the registration. DuckDB enables it by default only when a local file
  with the target's path exists; add `USE_TMP_FILE FALSE` then.
- `RETURN_STATS` works; `file_size_bytes` reports the bytes of Arrow memory registered (there is no
  file).
- The data is registered as DuckDB's exported batches, without concatenation, in one atomic
  catalog operation.

### 3.2 Copy into DuckDB (`COPY ... FROM`)

```sql
CREATE TABLE local_lineitem (l_orderkey BIGINT, l_quantity DOUBLE, l_extendedprice DOUBLE);
COPY local_lineitem FROM 'gpu_lineitem_1995' (FORMAT bolt);
COPY local_lineitem FROM 'unused' (FORMAT bolt, TABLE 'gpu_lineitem_1995');
```

- Source: the `TABLE` / `TABLE_NAME` option (a table reference), else the path, first as a table
  reference, then cleaned like a COPY path (`dir/x.bolt`, `tmp_x` -> `x`). `MODE` and any other
  option are rejected.
- Columns are matched **by name** (exact, then case-insensitive) when the destination has the same
  number of columns and every name resolves to a distinct Bolt column, otherwise **by position**
  (the counts must then match).
- Every value is converted to the destination's declared type with a checked cast (`SMALLINT`,
  `UINTEGER`, `HUGEINT`, `DECIMAL`, ... destinations work; out-of-range values are an error).
  Conversions that depend on the session time zone (`TIMESTAMPTZ` <-> `TIMESTAMP` / `DATE` /
  `VARCHAR`) are done by DuckDB under the statement's `TimeZone`, exactly like
  `INSERT INTO t SELECT CAST(col AS <type>)`. Destination types Craton Bolt cannot produce
  (`BLOB`, `INTERVAL`, `LIST`, ...) are rejected at bind time.
- A prepared `COPY ... FROM` reads the table's current rows on every `EXECUTE`.

---

## 4. Replacement Scans

```sql
SELECT * FROM bolt.gpu_lineitem;
SELECT * FROM bolt.main.gpu_lineitem;
SELECT * FROM 'bolt:gpu_lineitem';
SELECT * FROM 'bolt://gpu_lineitem';
SELECT * FROM gpu_lineitem;          -- bare name
```

All forms become `bolt_scan('<table_name>')`. Any other qualifier (`bolt.other.t`,
`mydb.bolt.t`) is left to DuckDB, which reports its own catalog error. The bare-name form applies only when DuckDB itself
cannot resolve the name (a DuckDB table, view or CTE always wins) and the name matches a Bolt
table exactly or case-insensitively; nothing is stripped from a bare name. Because the catalog is
process-wide, a Bolt table can satisfy a misspelled or not-yet-created DuckDB table name; write
`bolt.<name>` when you mean the Bolt table.

---

## 5. Table Functions

### `bolt_query(sql VARCHAR)` / `bolt_sql(sql VARCHAR)`

Runs `sql` (a constant string) on Craton Bolt and returns the result.

- The result schema is inferred without executing the query, so `DESCRIBE`, `EXPLAIN` and
  `PREPARE` do not run it: the SQL is bound by DuckDB against empty tables with the Bolt tables'
  original column types. Only a query DuckDB cannot bind that way (Bolt-only syntax) is planned
  by Bolt instead, or, if that fails too, executed once at bind time; that result serves the
  first execution if the catalog is unchanged, and every later `EXECUTE` of a prepared statement
  runs the query again.
- Result types are DuckDB's: `7 / 2` is `DOUBLE` (`3.5`), `x / 0` is `inf` or `NaN`,
  `SUM(<integer>)` is `HUGEINT`, `SUM(FLOAT)` is `DOUBLE`, `AVG` is `DOUBLE`, a `TINYINT` /
  unsigned / `HUGEINT` column keeps its type, and `TIMESTAMP - TIMESTAMP` is an `INTERVAL`. The
  exceptions: `TIMESTAMPTZ` / `TIMESTAMP_TZ_NS` columns are `TIMESTAMP` / `TIMESTAMP_NS` (the UTC
  instant, 2.2), and a query DuckDB cannot bind (above) gets the Bolt planner's types (storage
  types). GPU results are converted to these types with checked casts (a value that does not fit
  is an error, which also names a catalog change when one happened after the bind). Column names
  are the Bolt planner's (`count`, `sum_c`, `__expr_1`) when it plans the query, else DuckDB's.
  Columns are declared nullable; empty names become `col_<i>` and case-insensitive duplicates get
  a `_<n>` suffix.
- The query runs on the host fallback when there is no GPU engine, when the guard routes it (2.3),
  or when the engine rejects or fails it (if the host fails too, both errors are reported). On the
  host fallback the SQL must be a single `SELECT` in DuckDB's dialect.
- Projection pushdown and progress reporting are supported; results are scanned in parallel in
  2,048-row chunks.

```sql
SELECT * FROM bolt_query('
    SELECT region, COUNT(*) AS cnt, SUM(amount) AS revenue
    FROM gpu_sales GROUP BY region ORDER BY revenue DESC
');
```

### `bolt_scan(table_name VARCHAR)`

Scans a Bolt table in its **original** DuckDB types (2.2) with projection pushdown and, when the
GPU backend is active, filter pushdown. Without filters the stored batches are streamed directly
(no engine round trip, no copy), in parallel chunks.

A predicate is pushed only when the engine is expected to evaluate it exactly like DuckDB; the
rest stays with DuckDB. Nothing is pushed without a GPU engine.

| Pushed | Evaluated by DuckDB |
| :--- | :--- |
| `=`, `!=`, `<`, `>`, `<=`, `>=` with constants, and between non-float, non-string columns; `IS NOT NULL` guards are added for nullable columns | `FLOAT` / `DOUBLE` column-vs-column, NaN / infinite constants, Utf8 column-vs-column |
| `FLOAT` / `DOUBLE` vs a finite constant (NaN ordered like DuckDB: above every value, equal to itself); float constants are written in scientific notation | any predicate on a `DECIMAL` column ([W1-A-002](issues/W1-A-002-decimal-gpu-predicates.md)) |
| integer columns vs `DECIMAL` constants, as exact integer bounds (`=` / `!=` only for integral constants) | `=` / `!=` between an integer column and a fractional `DECIMAL` constant |
| `IN` / `NOT IN` with up to 64 non-NULL constants | `IN` lists with columns or `NULL`, longer lists |
| `IS NULL` / `IS NOT NULL` over a column, constant, cast, or pushed `DOUBLE` arithmetic | `IS [NOT] NULL` over a predicate; `IS NULL` on `VARCHAR` (`IS NOT NULL` is pushed as `LIKE '%'`) |
| `AND`, `OR`; `NOT` over NULL-strict operands | `NOT` over `AND` / `OR` / `IS [NOT] NULL` of `VARCHAR` |
| `CAST`s that are lossless widenings (`INTEGER` -> `BIGINT`, `SMALLINT` -> `DOUBLE`, ...) | narrowing, float-to-integer, numeric-to-`VARCHAR`, time-zone dependent casts |
| `DOUBLE` `+`, `-`, `*` | integer, `FLOAT` and `DECIMAL` arithmetic (DuckDB raises overflow errors the engine would not) |
| `LIKE` / `NOT LIKE` / `prefix` / `suffix` / `contains` with a constant pattern without `\`, and `VARCHAR` ordering vs a literal, on columns with at most 64 distinct values | `ILIKE`, `ESCAPE` / backslash patterns, other functions, larger dictionaries |
| `DATE` constants; `TIMESTAMPTZ` constants only against columns stored as UTC instants | most timestamp comparisons: craton-bolt 0.9 only compares nanosecond timestamp literals ([W1-A-006](issues/W1-A-006-temporal-pushdown-gaps.md)) |
| `BETWEEN`, as an inclusive **pre-filter** (DuckDB still applies the exact filter, because the C API does not report the bounds' inclusivity) | |

At most 128 predicate terms are pushed per scan. All candidates are validated with one engine
`EXPLAIN` (one per predicate if the combination is rejected) before DuckDB is told they were
pushed. Every pushed text means the same in the engine's dialect and in DuckDB's, because it may
run on either.

Execution of a filtered scan: if no projected column holds NULLs, the projected columns are
selected on the engine; otherwise (or if that fails) only the referenced columns plus a row-id
column are evaluated on an engine-only scratch table and the matching rows are gathered from the
stored batches ("filter-take"); if the engine cannot run the filter, the host fallback evaluates
it. Every result is converted to the declared types.

The schema is captured at bind time. If the table is dropped or replaced with a different schema
before the scan starts, it fails with "was dropped or replaced after the query was bound; re-run
the query".

```sql
SELECT id, amount FROM bolt_scan('gpu_sales')
WHERE amount >= 100.0 AND amount <= 500.0 AND active = true;
```

### `bolt_aggregate(table_name VARCHAR, group_by VARCHAR, aggregates VARCHAR)`

Runs `SELECT <group_by>, <aggregates> FROM "<table>" GROUP BY <group_by>` (or
`SELECT <aggregates> FROM "<table>"` when `group_by` is empty) like `bolt_query`. `table_name` is
resolved like `bolt_scan` and the SQL uses the resolved catalog name. `group_by` and `aggregates`
are inserted as written, but fragments with statement separators, comments, unbalanced
parentheses or unterminated quotes are rejected, as are backtick identifiers, `$` (dollar-quoted
strings) and backslashes inside `E'...'` escape strings.

```sql
SELECT grp, total_amount, row_cnt
FROM bolt_aggregate('gpu_sales', 'grp', 'sum(amount) AS total_amount, count(*) AS row_cnt');
```

### `bolt_tables()`

| Column | Type | Description |
| :--- | :--- | :--- |
| `table_name` | `VARCHAR` | Catalog name |
| `num_rows` | `BIGINT` | Rows across all batches |
| `num_columns` | `INTEGER` | Columns |
| `bytes_in_memory` | `BIGINT` | Arrow buffer bytes on the host |
| `schema` | `VARCHAR` | `col: TYPE, ...` with original DuckDB types where recorded |
| `on_gpu` | `BOOLEAN` | Whether the GPU engine holds the table: `false` on the host fallback or when the engine could not take it (see `gpu_note`); NULL while the backend is not known yet (no engine build since the last configuration change or `PRAGMA bolt_reset`) |
| `gpu_note` | `VARCHAR` | Why the engine does not hold this table (over `bolt_memory_budget`, an upload error), else NULL; an engine-wide reason is in `bolt_pool_stats().fallback_reason` |

### `bolt_pool_stats()`

One row: `backend VARCHAR` (`cuda-gpu`, `host-planner-fallback`, `wasm-stub`), `device_id
INTEGER`, `device_name VARCHAR` (the engine-initialization error when the device is unavailable),
`memory_budget_bytes`, `registered_tables`, `total_pooled_bytes`, `bucket_count`,
`oom_recovery_count`, `proactive_eviction_count`, `ptx_cache_hits`, `ptx_cache_misses`,
`ptx_cache_evictions`, `plan_cache_hits`, `plan_cache_misses`, `plan_cache_evictions` (all
`BIGINT`), and `fallback_reason VARCHAR`: why queries or tables run on the host fallback (the
engine-initialization error, and one `table 'x' not on GPU: <reason>` entry per table the engine
could not take, joined with `; `), NULL when everything is on the GPU. The first call after a
configuration change may initialize CUDA; later calls are answered without waiting for running
queries.

### `CALL` forms of the catalog mutations

`bolt_clear_tables()`, `bolt_drop_table(table_name)`, `bolt_register_sql(table_name, sql [, mode])`,
`bolt_load_parquet(table_name, path [, mode])` and `bolt_load_arrow_ipc(table_name, path [, mode])`
(not on wasm32) are also table functions, used with `CALL` or `SELECT * FROM`. Unlike the scalar
forms of the same names (6), they act **exactly once per execution**: the side effect runs when the
statement starts executing, not at bind time, so `EXPLAIN` and a `PREPARE` have no side effect and
every `EXECUTE` of a prepared statement acts again.

- Arguments must be constants (a prepared-statement parameter is bound at `EXECUTE`) and must not
  be `NULL`. `mode` is optional (positional or `mode := '...'`), parsed like the scalar `mode`, and
  validated at bind time.
- One result row: `removed BIGINT` (`bolt_clear_tables`), `dropped BOOLEAN` (`bolt_drop_table`) or
  `rows BIGINT` (the table's row count after `bolt_register_sql` / `bolt_load_*`).
- File loads apply the same file-access checks as the scalar `bolt_load_*`.

```sql
CALL bolt_load_parquet('gpu_trips', 'trips.parquet');
CALL bolt_register_sql('gpu_top', 'SELECT * FROM gpu_trips ORDER BY fare DESC LIMIT 100', mode := 'replace');
CALL bolt_drop_table('gpu_top');
```

### `bolt_metrics()`

`metric VARCHAR, category VARCHAR, value BIGINT`: craton-bolt's process-global counters
(`category = 'counter'`) and, per pipeline phase, `<phase>_latency_sum` (`histogram_ns`) and
`<phase>_latency_count` (`histogram_count`).

### `bolt_devices()`

`device_id INTEGER, name VARCHAR, total_memory_bytes BIGINT, is_active BOOLEAN`, from the CUDA
driver. Empty in `cuda-stub` builds and on hosts without a driver.

---

## 6. Scalar Functions

All scalar functions are `VOLATILE`, write exactly one value per input row, and return `NULL`
without side effects when any argument is `NULL`. Side-effecting functions (`bolt_register_sql`,
`bolt_load_*`, `bolt_drop_table`, `bolt_clear_tables`) act once per row DuckDB evaluates; the
optimizer may skip or reorder volatile calls (`WHERE false`, `LIMIT`, short-circuiting), as for
`nextval()`. `mode` is `'replace'` (default, alias `'overwrite'`), `'append'` or `'create'` (alias
`'error'`), case-insensitive, and is validated before any work is done. For exactly one side effect
per statement, use the `CALL` forms (5).

| Function | Returns | Description |
| :--- | :--- | :--- |
| `bolt_register_sql(table_name, sql [, mode])` | `BIGINT` | Runs `sql` like `bolt_query` and registers the result as `table_name`; returns the table's row count. |
| `bolt_load_parquet(table_name, path [, mode])` | `BIGINT` | Loads a Parquet file and registers it; returns the table's row count. Rows of one chunk that load the same path decode the file once. Not available on wasm32. |
| `bolt_load_arrow_ipc(table_name, path [, mode])` | `BIGINT` | Same for Arrow IPC (Feather v2) files. |
| `bolt_drop_table(table_name)` | `BOOLEAN` | `true` if the table existed and was dropped. |
| `bolt_clear_tables()` | `BIGINT` | Drops every table and returns how many were removed. Clears once per DuckDB worker thread per query; use `SELECT bolt_clear_tables()` or `PRAGMA bolt_clear_tables` for one well-defined clear. |
| `bolt_explain(sql)` | `VARCHAR` | Craton Bolt's logical and physical plan. Prefixed `=== Executed on the host fallback (DuckDB): <reason> ===` when the guard routes the query; if the engine cannot explain a routed query, that header is followed by `=== Host fallback (DuckDB) plan ===` and DuckDB's plan. Without a GPU engine, the Bolt plan under `=== No GPU engine: executed on the host fallback (DuckDB) ===`, or DuckDB's plan (`=== Host fallback (DuckDB) plan ===`) for shapes Bolt cannot plan. |
| `bolt_execution_tier(sql)` | `VARCHAR` | Where the query would run, without executing it: `'Host'` when there is no GPU engine or the guard routes it (and the host fallback can bind it); otherwise the planner's tier (`'Gpu'`, `'Hybrid'`, `'Host'`; `'Host'` there is the engine's own CPU tier). Host-orchestrated shapes (`LATERAL`, recursive CTEs, correlated subqueries, `VALUES`, `generate_series`, `DISTINCT ON`) are routed by the guard (2.3) and report `'Host'`. A run-time engine failure, retried on the host, cannot be predicted. |
| `bolt_version()` | `VARCHAR` | `duckdb-bolt-extension v<x> (craton-bolt v0.9.0, backend=<b>, device=<n>: <name>)`; `backend=not-initialized` until a GPU operation ran. Does not initialize the GPU. Notes an unusable `bolt_ptx_cache_dir`. |
| `bolt_prometheus_metrics()` / `bolt_metrics_prometheus()` | `VARCHAR` | craton-bolt telemetry in Prometheus text format. |

### File loading (`bolt_load_*`)

- **Local paths** are checked against the calling connection's `enable_external_access`,
  `allowed_directories` and `allowed_paths` on canonical paths (symlinks and `..` cannot escape; a
  refused or missing path is the same Permission Error), the opened handle is re-checked, and the
  file is also opened once through DuckDB's file system, so `disabled_filesystems` applies and `~`
  is expanded by DuckDB.
- **`file:` URLs** that DuckDB maps to local paths (`file:///x`, `file:/x`, `file://localhost/x`)
  are treated as local paths.
- **Other URLs** (`s3://`, `https://`, ...) are read through DuckDB's file system only (httpfs,
  secrets and DuckDB's own access checks apply). A remote Parquet file is read with range
  requests (the footer, then one request per column chunk it decodes), never buffered whole; Arrow
  IPC is read with positional reads through a 1 MiB buffer. For any Parquet load, a column chunk
  larger than `memory_limit`, or one that lies outside the file, is refused before decoding.
- **Codecs**: Parquet Snappy, Gzip, ZSTD, LZ4 (`LZ4_RAW` and Hadoop `LZ4`) and Brotli; Arrow IPC
  LZ4_FRAME and ZSTD. Arrow IPC files are validated before decoding (blocks inside the file,
  buffers inside their block, bounded declared uncompressed sizes), so a corrupt file is an error.
- **Types**: as 2.2; each column records the type DuckDB's own reader would return. Every field
  is nullable. `Date64` values that are not whole days are an error. Zoned timestamps are stored
  as naive UTC (`s` / `ms` converted to `us`).

---

## 7. Configuration Options (`SET` / `RESET`)

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `bolt_device` | `INTEGER` | `0` | CUDA device ordinal (`>= 0`). craton-bolt 0.9 binds the process to the first device it used: changing it later makes engine builds fail with an explanation (host fallback) until it is set back or the process restarts. A non-existent ordinal is reported with the visible ordinals. |
| `bolt_memory_budget` | `BIGINT` | `0` | Bytes (`>= 0`, `0` = uncapped). Enforced by the extension for uploads: a table whose upload would bring the engine-resident tables (Arrow bytes) above the budget stays on the host and is served by the host fallback. Query intermediates are not capped; craton-bolt also uses it to split streaming scans. |
| `bolt_ptx_cache_dir` | `VARCHAR` | `''` | Persistent on-disk PTX kernel cache (`''` = disabled). Refused while `enable_external_access = false`. Must be absolute and contain no `..` component; an existing path must be a directory. Unix: the directory (or, if missing, its nearest existing ancestor) must be owned by the process user or root and not group- or world-writable, and every parent must be owned by the user or root and not world-writable unless sticky; if the user id cannot be determined only root-owned directories are accepted. Windows: it must lie inside `%USERPROFILE%`, `%LOCALAPPDATA%` or `%APPDATA%`, and not inside `%PUBLIC%`, `%ALLUSERSPROFILE%` or `%ProgramData%`. See 7.1. |
| `bolt_enable_tracing` | `BOOLEAN` | `false` | craton-bolt tracing spans. Turning it off restores the process `log` level seen before it was turned on. |

- Options are **GLOBAL only** (`SET SESSION bolt_*` is rejected) and configure the single
  process-wide runtime: a `SET` in one database affects every database in the process.
- `SET bolt_* = NULL` is rejected; `RESET bolt_*` restores the default above. A refused value
  leaves the setting unchanged.
- Setting an option to its current value does nothing; any other change drops the engine, which
  is rebuilt lazily on next use and forgets a cached initialization failure.
- `bolt_*` options cannot be passed when a database is opened: DuckDB 2.0 checks startup options
  before extensions are loaded and refuses them ("The following options were not recognized").
  Use `SET GLOBAL bolt_* = ...` after `LOAD bolt`. A setting another database changed is
  refreshed on the next Bolt call, so `current_setting('bolt_*')` shows what the runtime uses.

### 7.1 PTX cache directory checks

The directory is checked at `SET` time, again before every engine build (a failing directory is
not used; `bolt_version()` says why), and on Bolt calls (a full check when the directory's
identity or permissions changed, else at most every 5 seconds). A directory that fails the
use-time check disables the cache (`bolt_ptx_cache_dir` becomes `''`) and fails that one call with
the reason. When the option is `''`, craton-bolt's `BOLT_PTX_CACHE_DIR` environment variable is
subject to the same rules; an unsafe value makes the engine build fail.

---

## 8. `PRAGMA` Statements

- `PRAGMA bolt_info;` — `SELECT * FROM bolt_pool_stats()`.
- `PRAGMA bolt_clear_tables;` — drops every Bolt table.
- `PRAGMA bolt_reset;` — drops and rebuilds the engine with the current settings, re-uploads every
  table, and forgets a cached initialization failure. Device memory of the old engine is released;
  the process-wide PTX / plan caches and their counters are not cleared. **Fails** whenever the
  engine cannot be built while at least one CUDA device is visible (driver or context error,
  unusable `BOLT_PTX_CACHE_DIR`, a `bolt_device` the process cannot switch to, or an ordinal that
  does not exist); the host fallback stays active. Succeeds without an engine only when no CUDA
  device is visible (always in `cuda-stub` builds). A table the engine cannot upload never fails it.
- `PRAGMA bolt_request_timeout(<milliseconds>);` — limits how long a statement waits for one Bolt
  request (`0` = no limit, the default; negative or `NULL` is rejected). Process-wide. On timeout:
  a request still queued behind other work is abandoned and has no effect; a running query fails,
  and if it runs on the host fallback that query is interrupted (a GPU query cannot be cancelled
  and finishes in the background); a running catalog or configuration change is waited for, so a
  statement never fails while its change still takes effect.

`PRAGMA bolt_reset` and `PRAGMA bolt_request_timeout` change process-wide runtime configuration, so
both are refused in a database running with `lock_configuration = true`, like `SET bolt_*`.
`bolt_info` and `bolt_clear_tables` stay available.

---

## 9. Restrictions inside the host fallback

The host fallback is a separate DuckDB instance with `enable_external_access=false`, extension
auto-loading/-installing and community extensions disabled, `lock_configuration=true`, at most 4
threads and a `memory_limit` of 25%. It accepts one `SELECT` statement. Bolt functions, the `bolt`
namespace and `'bolt:'` table names inside a query that runs there are refused ("cannot be used
inside a query that runs on the Craton Bolt host fallback").

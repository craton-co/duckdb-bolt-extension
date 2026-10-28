# duckdb-bolt-extension (`bolt`)

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

A DuckDB v2.0 C API extension written in Rust that embeds the **[Craton Bolt](https://github.com/craton-co/craton-bolt)** JIT-compiled GPU SQL engine (`craton-bolt` 0.9.0) inside DuckDB, developed by **[Craton Software Company](https://craton.com.ar)**.

`duckdb-bolt-extension` lets you:
- **Upload** any DuckDB table, view, file scan, or query result into the Bolt catalog with `COPY ... TO '<bolt_table>' (FORMAT bolt)`, or load Parquet / Arrow IPC files directly with `bolt_load_parquet` / `bolt_load_arrow_ipc`.
- **Execute** SQL on the GPU engine (filters, projections, hash joins, `GROUP BY` aggregations, sorting, window functions) with `bolt_query('<sql>')` / `bolt_sql('<sql>')` / `bolt_aggregate(...)`.
- **Scan** Bolt tables back into DuckDB with `bolt_scan('<bolt_table>')` or a replacement scan (`SELECT ... FROM bolt.<table> WHERE ...`), with projection pushdown and, on a GPU, filter pushdown. Columns come back with their **original DuckDB types**.
- **Copy** Bolt tables into DuckDB tables with `COPY <duckdb_table> FROM '<bolt_table>' (FORMAT bolt)`.
- **Inspect** the GPU memory pool, the PTX JIT and SQL plan caches, execution tiers, plans, and Prometheus metrics from SQL.

Full reference: [docs/SQL_REFERENCE.md](docs/SQL_REFERENCE.md). Internals: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## Requirements & backends

- **DuckDB v2.0.0 exactly.** The extension uses the unstable C API (`C_STRUCT_UNSTABLE`), so a binary only loads in the DuckDB version it was built for.
- **Unsigned binaries.** Locally built binaries are unsigned: start DuckDB with `-unsigned` (CLI) or `allow_unsigned_extensions=true`.
- **GPU backend (`cuda-gpu`)** needs an NVIDIA GPU and driver and a build *without* the `cuda-stub` feature. The Makefile enables `cuda-stub` (and prints a warning) when neither `CUDA_PATH` nor `/usr/local/cuda` exists, so binaries from the DuckDB extension CI (no CUDA toolkit) are CPU-only. Pass `BOLT_CUDA_STUB=0` to force a GPU build.
- **Host fallback (`host-planner-fallback`)**: an embedded, long-lived in-memory DuckDB instance (external access and extension auto-loading/-installing disabled, configuration locked, at most 4 threads and 25% of RAM) into which the Bolt tables a query names are mirrored. It runs exactly one `SELECT` statement with **DuckDB's** dialect and semantics. It is used when no GPU engine can be built, and on GPU hosts for every query the engine guard does not trust (see [GPU vs host fallback](#gpu-vs-host-fallback)). `SELECT backend FROM bolt_pool_stats()` shows the active backend.
- **Wasm** (`crates/wasm`): compiles (checked in CI) and runs every request inline on the calling thread; there is no GPU, and SQL execution, `EXPLAIN` and file loading are unavailable. Wasm targets are not distributed yet (the build has not been verified in DuckDB-Wasm).
- **Process-wide state.** The Bolt catalog, every `bolt_*` setting, and `PRAGMA bolt_request_timeout` are shared by every connection and every DuckDB database in the process. Any connection can read, replace or drop any Bolt table.

---

## Quick start

```sql
LOAD 'build/release/bolt.duckdb_extension';

SELECT bolt_version();
SELECT * FROM bolt_devices();

-- 1. Upload a DuckDB table or query into the Bolt catalog
CREATE TABLE sales AS
SELECT
    i::INTEGER AS id,
    CASE WHEN i % 3 = 0 THEN 'north' WHEN i % 3 = 1 THEN 'south' ELSE 'east' END AS region,
    (i * 12.5)::DOUBLE AS amount,
    (i % 2 = 0) AS active
FROM range(1, 10001) t(i);

COPY sales TO 'gpu_sales' (FORMAT bolt);

-- 2. Inspect registered tables
SELECT * FROM bolt_tables();

-- 3. Run SQL on Craton Bolt
SELECT * FROM bolt_query('
    SELECT region, SUM(amount) AS total_revenue, COUNT(id) AS orders
    FROM gpu_sales
    WHERE active = TRUE
    GROUP BY region
    ORDER BY total_revenue DESC
');

-- 4. Query Bolt tables from DuckDB with projection and filter pushdown
SELECT id, amount
FROM bolt.gpu_sales
WHERE region = 'north' AND amount >= 5000.0
ORDER BY id
LIMIT 5;

-- 5. Where would a query run, and how?
SELECT bolt_execution_tier('SELECT region, SUM(amount) FROM gpu_sales GROUP BY region');
SELECT bolt_explain('SELECT region, SUM(amount) FROM gpu_sales GROUP BY region');

-- 6. Pool, cache and backend telemetry
PRAGMA bolt_info;
```

If a local file with the COPY target's name exists, DuckDB turns on `USE_TMP_FILE`, which Bolt
rejects; add `USE_TMP_FILE FALSE` to the options in that case.

---

## SQL overview

Details, options and edge cases: [docs/SQL_REFERENCE.md](docs/SQL_REFERENCE.md).

| Kind | Name | Purpose |
| :--- | :--- | :--- |
| COPY | `COPY <table\|(query)> TO '<bolt_table>' (FORMAT bolt [, MODE ...] [, TABLE ...])` | Upload (`MODE` `'replace'` default, `'append'`, `'create'`). |
| COPY | `COPY <duckdb_table> FROM '<bolt_table>' (FORMAT bolt [, TABLE ...])` | Copy a Bolt table into a DuckDB table (columns by name, checked casts). |
| Table function | `bolt_query(sql)` / `bolt_sql(sql)` | Run SQL on Craton Bolt; the result has engine (storage) types. |
| Table function | `bolt_scan(table_name)` | Scan a Bolt table in its original DuckDB types; projection and (GPU only) filter pushdown. |
| Table function | `bolt_aggregate(table_name, group_by, aggregates)` | `SELECT <group_by>, <aggregates> FROM <table> GROUP BY <group_by>` on Craton Bolt. |
| Table function | `bolt_tables()`, `bolt_pool_stats()`, `bolt_metrics()`, `bolt_devices()` | Catalog; pool / cache / backend telemetry; execution counters; CUDA devices. |
| Replacement scan | `bolt.<t>`, `bolt.main.<t>`, `'bolt:<t>'`, `'bolt://<t>'`, bare `<t>` | Rewritten to `bolt_scan('<t>')` (a bare name only when no DuckDB object has it). |
| Scalar | `bolt_register_sql(table_name, sql [, mode])` | Run SQL and register the result as a table. |
| Scalar | `bolt_load_parquet(table_name, path [, mode])`, `bolt_load_arrow_ipc(table_name, path [, mode])` | Load a local file or a URL (through DuckDB's file system). Not on wasm. |
| Scalar | `bolt_drop_table(table_name)`, `bolt_clear_tables()` | Drop one / all tables. |
| Table function (`CALL`) | `bolt_register_sql`, `bolt_load_parquet`, `bolt_load_arrow_ipc`, `bolt_drop_table`, `bolt_clear_tables` | The same catalog mutations, exactly once per execution: `CALL bolt_drop_table('t')`. |
| Scalar | `bolt_explain(sql)`, `bolt_execution_tier(sql)` | Plan text (says when the host fallback runs the query) and tier (`'Gpu'`, `'Hybrid'`, `'Host'`). |
| Scalar | `bolt_version()`, `bolt_prometheus_metrics()` / `bolt_metrics_prometheus()` | Version and backend; Prometheus text. |
| Option | `SET [GLOBAL] bolt_device`, `bolt_memory_budget`, `bolt_ptx_cache_dir`, `bolt_enable_tracing` | Engine configuration; GLOBAL scope only, process-wide. |
| Pragma | `PRAGMA bolt_info`, `bolt_reset`, `bolt_clear_tables`, `bolt_request_timeout(<ms>)` | Telemetry, engine rebuild, clear the catalog, per-request wait limit. |

Table names are matched case-insensitively. Names starting with `__bolt_` are reserved, and empty
or duplicate column names are rejected.

---

## Supported data types

Values are stored in the subset of Arrow types Craton Bolt supports, and every column records its
original DuckDB type in Arrow field metadata (`duckdb.logical_type`, RFC-008). `bolt_scan`,
replacement scans, `bolt_tables().schema` and `COPY ... FROM` use the original type;
`bolt_query` / `bolt_sql` / `bolt_aggregate` return the engine's storage types.

| DuckDB type | Stored as | `bolt_scan` returns | `bolt_query` returns |
| :--- | :--- | :--- | :--- |
| `BOOLEAN` | `Boolean` | `BOOLEAN` | `BOOLEAN` |
| `TINYINT`, `SMALLINT`, `INTEGER`, `UTINYINT`, `USMALLINT` | `Int32` | original type | `INTEGER` |
| `BIGINT`, `UINTEGER` | `Int64` | original type | `BIGINT` |
| `UBIGINT`, `HUGEINT` | `Int64` (values outside the `i64` range are rejected on upload) | original type | `BIGINT` |
| `FLOAT`, `DOUBLE` | `Float32`, `Float64` | same | same |
| `VARCHAR` | `Utf8` | `VARCHAR` | `VARCHAR` |
| `DATE` | `Date32` | `DATE` | `DATE` |
| `TIMESTAMP_S`, `TIMESTAMP_MS`, `TIMESTAMP`, `TIMESTAMP_NS` | `Timestamp(s / ms / us / ns)` | same | same |
| `TIMESTAMPTZ`, `TIMESTAMP_TZ_NS` | naive `Timestamp(us / ns)` holding the UTC instant | `TIMESTAMPTZ`, `TIMESTAMP_TZ_NS` | `TIMESTAMP`, `TIMESTAMP_NS` (UTC wall clock) |
| `DECIMAL(1..38, s)` | `Decimal128(p, s)` | same | same |

Other types (`BLOB`, `UUID`, `INTERVAL`, `TIME`, `ENUM`, `LIST`, `STRUCT`, `MAP`, ...) are
rejected. Files loaded with `bolt_load_*` record the type DuckDB's own reader would return
(`SMALLINT` for `Int16`, `TIMESTAMPTZ` for a zoned timestamp, ...); `LargeUtf8`, `Utf8View`,
dictionaries, `Float16` and whole-day `Date64` are converted. Every conversion is checked: a value
that does not fit is an error, never a silent `NULL`. Appending to a table also checks each value
against the column's original type (appending `70000` to a column copied from `SMALLINT` fails).

---

## GPU vs host fallback

On a GPU host, a query sent to the engine (`bolt_query`, `bolt_sql`, `bolt_aggregate`,
`bolt_register_sql`, filtered `bolt_scan`) runs on the host fallback instead when the engine
guard (`gpu_crash_guard`) finds that craton-bolt 0.9.0 could crash or return wrong results:

- it references a table that holds NULLs and its plan is not "null-safe" (anything beyond plain
  column scans, projections, sorts, `LIMIT`, `DISTINCT`, `UNION ALL`, inner equi-joins and
  `COUNT` / `SUM` / `MIN` / `MAX` / `AVG` of plain columns; not grouped `SUM` / `MIN` / `MAX`,
  ascending sort keys or aggregates below `ORDER BY` / `LIMIT` / `DISTINCT` over a nullable
  column), or it can produce NULLs itself (outer
  join, `CASE`, `NULL` literal) ([W2-A-001](docs/issues/W2-A-001-gpu-ignores-null-validity.md),
  upstream [001](docs/upstream/craton-bolt-001-crash-filtered-aggregate-over-nulls.md),
  [002](docs/upstream/craton-bolt-002-gpu-kernels-ignore-null-validity.md));
- the guard's Bolt planner cannot plan it (correlated subqueries, `LATERAL`, recursive CTEs,
  `DISTINCT ON`, `VALUES`, ...), or it has a type hazard the engine computes wrongly (see Known
  limitations);
- it has more than 4,096 `AND` / `OR` terms, or a `LIKE` / string-ordering predicate over a table
  with more than 4,096 distinct strings in a column (deep predicate trees overflow the engine,
  upstream [007](docs/upstream/craton-bolt-007-deep-predicate-stack-overflow.md));
- it references a table the engine does not hold: one whose upload would bring the uploaded tables
  above `bolt_memory_budget`, one with more than 2 GiB of string data in a column, or one whose
  upload failed;
- the engine rejects or fails the query (planning gaps, unsupported functions, device errors).

Tables without NULLs and queries the guard accepts run on the GPU. `bolt_explain` prefixes the
plan with `=== Executed on the host fallback (DuckDB): <reason> ===` when the guard routes a
query (this is also where you see why a table is not on the GPU), and `bolt_execution_tier`
returns `'Host'`. Multi-batch tables are uploaded as one coalesced engine batch per table.
Results can differ between backends where the engine's semantics differ from DuckDB's; see
[Known limitations](#known-limitations).

---

## Security and sandboxing

- `bolt_load_parquet` / `bolt_load_arrow_ipc` honor the calling connection's
  `enable_external_access`, `allowed_directories` and `allowed_paths` on canonical paths
  (symlinks and `..` cannot escape), re-check the opened handle, and also open local files once
  through DuckDB's file system (so `disabled_filesystems` applies). `file:` URLs that DuckDB maps to
  local paths get the same checks; other URLs (`s3://`, `https://`, ...) are read through DuckDB's
  file system with its own checks, secrets and extensions (httpfs).
- `SET bolt_ptx_cache_dir` is refused while `enable_external_access = false`, must be an absolute,
  private directory (rules in [SECURITY.md](SECURITY.md)), and is re-checked when it is used.
- The host fallback cannot reach files, the network or extensions. A Bolt function used inside a
  query that runs on the host fallback is refused.

See [SECURITY.md](SECURITY.md) for the threat model and how to report a vulnerability.

---

## Known limitations

Open items are tracked in [`docs/issues/`](docs/issues/README.md); engine defects are written up
for craton-bolt in [`docs/upstream/`](docs/upstream/README.md). The most visible ones:

- **Engine results that differ from DuckDB** (`bolt_query`, `bolt_sql`, `bolt_aggregate` when
  they run on the GPU): NaN comparisons and `INTEGER` / `BIGINT` overflow follow IEEE / wrapping
  rules ([W1-A-003](docs/issues/W1-A-003-nan-and-overflow-semantics.md)). Shapes the engine is
  known to get wrong are routed to the host fallback, which runs them on the original DuckDB
  types: NULLs in filters, projections, grouped aggregates, default-ordered sort keys and
  aggregates below `ORDER BY` / `LIMIT` / `DISTINCT`; an ungrouped `count(*)` next to any
  aggregate of a nullable column; `DECIMAL` casts and arithmetic; `CAST(<FLOAT / DOUBLE> AS
  <integer>)` (the engine truncates, DuckDB rounds); arithmetic
  on narrow, unsigned or `FLOAT` columns; integer `/` and `%`; fractional literals in integer
  arithmetic; `SUM` of `FLOAT` and integer sums that may leave the `BIGINT` range; `TIMESTAMP`
  subtraction; `STDDEV_POP` and `SUM` / `MIN` / `MAX` over a possibly empty input; and every
  query the Bolt planner cannot plan (correlated subqueries, `LATERAL`, recursive CTEs,
  `DISTINCT ON`, `VALUES`, ...). The guard is checked by a differential fuzzer against DuckDB.
  `bolt_scan` and replacement scans are not affected: DuckDB evaluates everything the engine
  might get wrong.
- **Result types**: `bolt_query`, `bolt_sql` and `bolt_aggregate` return DuckDB's result types
  (integer `SUM` is `HUGEINT`, `7 / 2` is `3.5`, `TIMESTAMP - TIMESTAMP` is an `INTERVAL`, narrow
  and unsigned integer columns keep their types), bound against an empty copy of the catalog's
  schemas. Exceptions: `TIMESTAMPTZ` columns come back as naive UTC `TIMESTAMP`, and output column
  names follow the Bolt planner (`count`, `sum_c`) when it plans the query.
- **Startup options**: `bolt_*` options cannot be passed when a database is opened (DuckDB 2.0
  refuses options that are not registered yet); use `SET GLOBAL` after `LOAD`.
- **Pushdown gaps** (performance only): predicates on `DECIMAL` columns
  ([W1-A-002](docs/issues/W1-A-002-decimal-gpu-predicates.md)) and most timestamp comparisons
  ([W1-A-006](docs/issues/W1-A-006-temporal-pushdown-gaps.md)) are evaluated by DuckDB.
- **No cancellation**: `Ctrl-C` does not stop a Bolt request. `PRAGMA bolt_request_timeout`
  bounds the wait and interrupts a host-fallback query, but a GPU query finishes in the background
  ([W2-E-005](docs/issues/W2-E-005-bolt-requests-not-interruptible.md)).
- **Side-effecting scalars** (`bolt_register_sql`, `bolt_load_*`, `bolt_drop_table`,
  `bolt_clear_tables`) act once per row DuckDB evaluates, and the optimizer may skip or reorder
  them. Their `CALL` forms (`CALL bolt_load_parquet('t', 'f.parquet')`, `CALL bolt_clear_tables()`,
  ...) act exactly once per execution.
- **Dropped tables** stay in the engine as empty placeholders until a rebuild
  ([W1-B-006](docs/issues/W1-B-006-engine-per-table-unregister.md)), and `bolt_device` cannot be
  changed after the first engine build in a process.
- **File loading**: remote (`https://`) loads go through DuckDB's file system like local ones,
  but have no automated test against a real HTTP server (the tests run offline); a column chunk larger
  than `memory_limit` is refused rather than streamed, and page headers inside a chunk are not
  bounded before `parquet` allocates for them
  ([W7-A-001](docs/issues/W7-A-001-parquet-page-header-sizes-unbounded.md)); residual PTX-cache
  gaps ([W2-D-003](docs/issues/W2-D-003-ptx-cache-dir-residual-hardening.md)); some Arrow types cannot be
  loaded ([W1-D-007](docs/issues/W1-D-007-ingest-type-coverage.md)).
- Most engine workarounds can go away with craton-bolt 0.10, which is not published yet
  ([W1-O-003](docs/issues/W1-O-003-upgrade-craton-bolt-0-10.md)).

---

## Building & testing

The workspace has three crates: `duckdb-bolt-core` (`crates/core`, all logic, the default
workspace member), `duckdb-bolt-extension` (`crates/native`, the loadable cdylib) and
`duckdb-bolt-wasm` (`crates/wasm`). `duckdb-neo`'s `build-extension` (native, wasm) and
`build-extension-static` (core tests) features are mutually exclusive, so always select a crate
with `-p`; **`cargo test --workspace` does not build**.

```bash
# Unit, integration and SQLLogicTest (test/sql/*.test) suites, plus the differential fuzzer
cargo test -p duckdb-bolt-core --features cuda-stub   # no GPU / CUDA toolkit needed
cargo test -p duckdb-bolt-core                        # GPU host: expects backend cuda-gpu

# Differential fuzzer only, with more cases or another seed
BOLT_FUZZ_CASES=20000 BOLT_FUZZ_SEED=42 \
  cargo test -p duckdb-bolt-core --features cuda-stub --test w4b_differential_fuzz

# Loadable extension, option A: Cargo + packager (Windows / Linux / macOS)
cargo build --release -p duckdb-bolt-extension        # add --features cuda-stub for CPU-only
python scripts/package_extension.py --profile release # -> build/release/bolt.duckdb_extension

# Option B: DuckDB extension-ci-tools Makefile (also runs DuckDB's SQLLogicTest runner)
make configure
make release          # -> build/release/bolt.duckdb_extension
make test_release
```

CI (`.github/workflows/RustCI.yml`) runs rustfmt; clippy and `cargo test --features cuda-stub`
on Linux, Windows and macOS; a wasm `cargo check` and clippy; DuckDB's SQLLogicTest runner on a
debug build; an MSRV (Rust 1.88) build and test; and `cargo deny`. No CI job has a GPU. See
[CONTRIBUTING.md](CONTRIBUTING.md) for the test conventions.

---

## Documentation & governance

- [Architecture](docs/ARCHITECTURE.md) and [SQL reference](docs/SQL_REFERENCE.md)
- [Roadmap & RFCs](docs/ROADMAP.md)
- [Issue tracker](docs/issues/README.md) and [upstream craton-bolt reports](docs/upstream/README.md)
- [Contributing](CONTRIBUTING.md), [Security policy](SECURITY.md), [Code of Conduct](CODE_OF_CONDUCT.md)
- [Changelog](CHANGELOG.md), [Maintainers](MAINTAINERS.md), [Releasing](RELEASING.md)
- [Developer Certificate of Origin](DCO) (sign off every commit with `git commit -s`)
- Dependency policy: [`deny.toml`](deny.toml) (`cargo deny --all-features check`)

---

## License

Copyright 2026 [Craton Software Company](https://craton.com.ar).
Licensed under the [Apache License, Version 2.0](LICENSE).

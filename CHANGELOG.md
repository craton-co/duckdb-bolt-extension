# Changelog

All notable changes to `duckdb-bolt-extension` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Summary of review and fix waves 1-12. Issue pages that are still open are linked from
[`docs/issues/`](docs/issues/README.md); engine defects are written up for craton-bolt in
[`docs/upstream/`](docs/upstream/README.md).

### Added
- `CALL` (table-function) forms of `bolt_register_sql`, `bolt_load_parquet`, `bolt_load_arrow_ipc`,
  `bolt_drop_table` and `bolt_clear_tables` that act exactly once per execution (not at bind, so
  `EXPLAIN` / `PREPARE` have no side effect; `mode` is validated at bind).
- `PRAGMA bolt_request_timeout(<ms>)`: per-request wait limit (process-wide); abandons queued
  requests, interrupts host-fallback queries (only the timed-out request's own query), waits for
  running mutations.
- `bolt_register_sql(table_name, sql, mode)` overload.
- `bolt_load_*` read URLs through DuckDB's file system (httpfs, S3, HTTPS).
- `bolt_metrics()` includes per-phase latency histograms.
- `bolt_pool_stats().fallback_reason` (why queries or tables run on the host fallback, NULL when
  everything is on the GPU) and `bolt_tables().on_gpu` / `gpu_note` (per table, never initializes
  CUDA), appended as the last columns.
- Tests: SQLLogicTest runner for every `test/sql/*.test` file; SLT suites `bolt_types`,
  `bolt_pushdown`, `bolt_semantics`, `bolt_sandbox`, `bolt_prepared`, `bolt_timestamptz`,
  `bolt_call`, `bolt_qualifiers_and_lock` and `bolt_w9d_validation`; per-wave Rust regression
  suites (`w1a_*` ... `w10a_*`); a probe that builds, packages and `LOAD`s the real
  `bolt.duckdb_extension` (`w5b_packaged_probe`); and a seeded differential fuzzer
  (`BOLT_FUZZ_CASES`, `BOLT_FUZZ_SEED`, `BOLT_FUZZ_TABLE_SEEDS`, `BOLT_FUZZ_VERBOSE`) with the
  crash-guard shapes of wave 8 and a route-consistency oracle (`BOLT_FUZZ_ROUTE_EVERY`,
  `BOLT_FUZZ_STRICT_TYPES`, opt-in `BOLT_FUZZ_WIDE_LIKE`).
- CI: rustfmt, clippy and tests on Linux / Windows / macOS, wasm compile check, DuckDB's
  SQLLogicTest runner, MSRV, `cargo deny`; Dependabot; extension-ci-tools pinned to a commit.
- `deny.toml`, `DCO`, `MAINTAINERS.md`, `RELEASING.md`, `CODEOWNERS`, per-crate `LICENSE` /
  `NOTICE`, the in-repo issue tracker (`docs/issues/`) and upstream reports (`docs/upstream/`);
  `scripts/package_extension.py` gained `--suffix`, `--output-name`, `--target-dir`, `--platform`
  and `CARGO_TARGET_DIR` support, and warns when an unknown `--target` falls back to the host's
  library name and platform.
- Documentation rewritten for the current behavior (README, SQL reference, architecture, roadmap,
  security, contributing).

### Changed
- **Original types on the read side (RFC-008)**: `bolt_scan`, replacement scans and
  `bolt_tables().schema` now return the original DuckDB type of each column (`SMALLINT`,
  `UINTEGER`, `HUGEINT`, `TIMESTAMPTZ`, ...) instead of the widened storage type. `COPY ... TO` and
  `bolt_load_*` record it in the Arrow field metadata key `duckdb.logical_type`.
- **`bolt_query` / `bolt_sql` / `bolt_aggregate` return DuckDB's result types** (W9-A-001): the
  query is bound against an empty DuckDB holding zero-row copies of the referenced tables'
  original schemas, and GPU results are cast to those types with checked casts. Integer `SUM` is
  now `HUGEINT`, `7 / 2` is `3.5`, `SUM(FLOAT)` / `AVG` are `DOUBLE`, and narrow, unsigned and
  `DECIMAL` columns keep their types. `TIMESTAMPTZ` still comes back as naive UTC `TIMESTAMP`.
  `INTERVAL` results (`TIMESTAMP - TIMESTAMP`) can be returned (W9-A-002).
- **GPU result guard**: queries the engine could crash on or answer wrongly run on the host
  fallback instead (see Fixed). A query the guard's Bolt planner cannot plan (`LATERAL`,
  `WITH RECURSIVE`, correlated subqueries, `VALUES`, `DISTINCT ON`, ...) now always runs on the
  host fallback (it used to run as host-orchestrated GPU sub-plans); `bolt_explain` shows DuckDB's
  plan for it.
- **`bolt_execution_tier`** reports `'Host'` when the query will run on the host fallback;
  `bolt_explain` says so (with the reason) and explains host-only shapes with DuckDB.
- **`COPY ... FROM`** matches columns by name (exact, then case-insensitive) when the column counts
  match and every name resolves, by position otherwise; accepts `TABLE` / `TABLE_NAME`; converts
  into the destination's declared types with checked casts; time-zone-dependent casts use the
  session `TimeZone`. Appends (`MODE 'append'`) match columns by name and check values against the
  original type.
- **Table names**: only quotes and a `bolt://` / `bolt:` / `bolt.main.` / `bolt.` prefix (any case)
  are stripped. `COPY ... TO 'tmp_y'` now registers `tmp_y` (not `y`); a target path still loses its
  directory part and `.bolt` suffix; the `TABLE` option is never path-cleaned. Names starting with
  `__bolt_` are reserved; empty and duplicate column names are rejected.
- **Replacement scan**: only `bolt.<t>`, `bolt.main.<t>` and an unqualified `'bolt:<t>'` /
  `'bolt://<t>'` are explicit Bolt references; other qualified names (`bolt.other.<t>`,
  `mydb.bolt.<t>`) are left to DuckDB, which reports its own catalog error.
- **Options** are `GLOBAL`-only (`SET SESSION bolt_*` is rejected), `SET bolt_* = NULL` is rejected,
  `RESET` restores the documented default, unchanged values do not rebuild the engine, and
  `current_setting()` shows what the process-wide runtime uses (a setting another database changed
  is refreshed on the next Bolt call).
- **`PRAGMA bolt_reset`** fails whenever a CUDA device is visible but no engine can be built,
  including a `bolt_device` ordinal that does not exist or that the process cannot switch to; it
  succeeds without an engine only when no CUDA device is visible.
- **`bolt_memory_budget`** is enforced for uploads: a table that would exceed it is served by the
  host fallback.
- **`USE_TMP_FILE`** with `COPY ... TO (FORMAT bolt)` is rejected with an explanation (it used to
  fail after registering); `RETURN_STATS` reports Arrow bytes as `file_size_bytes`;
  `PARTITION_BY` / `PER_THREAD_OUTPUT` / `FILE_SIZE_BYTES` are rejected before anything is
  registered.
- **`bolt_aggregate`** rejects `group_by` / `aggregates` fragments with statement separators,
  comments, unbalanced parentheses or unterminated quotes, and fragments with backtick
  identifiers, `$` (dollar quoting) or backslashes inside `E'...'` strings.
- The distribution pipeline excludes `wasm_*`, `linux_amd64_musl`, `windows_amd64_rtools` and
  `windows_amd64_mingw`; `duckdb-bolt-extension` is `publish = false` (only `duckdb-bolt-core` is
  published); the Makefile warns when it enables `cuda-stub`.

### Fixed
- Decimal casts that lower the precision but keep the scale added one unit in the last place
  (arrow-cast 53.4.1: `DECIMAL(12,2)` 21.25 became 21.26). All Arrow casts in the crate now go
  through `arrow_bridge::cast_with_options`, which rescales `Decimal128` exactly and rounds half
  away from zero like DuckDB; affected `bolt_query` result types, `bolt_scan` type restoration,
  `COPY ... FROM`, appends and ingest (W12-A-001).
- `CAST(<FLOAT / DOUBLE> AS <integer>)` truncated on the GPU; it is routed to the host fallback,
  which rounds like DuckDB (W12-A-002).
- An ungrouped `count(*)` next to any aggregate of a nullable column (`count(c)`, `min(c)`, ...)
  gave wrong results on the GPU; it is routed to the host fallback (W11-A-001).
- **Wrong results and crashes from the GPU engine** (craton-bolt 0.9.0 ignores NULL validity and
  crashes on filtered aggregates over NULLs): the engine guard routes to the host fallback
  non-null-safe queries over tables with NULLs, queries that produce NULLs (outer joins, `CASE`,
  `NULL` literals), very deep predicates, and tables the engine does not hold. Engine failures are
  retried on the host.
- **Engine result hazards found by the differential fuzzer and the wave-8 review** also run on the
  host fallback: grouped `SUM` / `MIN` / `MAX` over a column with NULLs; ascending (`NULLS FIRST`)
  sort keys over nullable columns; aggregates of nullable columns below `ORDER BY` / `LIMIT` /
  `DISTINCT`; `CAST` from or to `DECIMAL` and `DECIMAL` arithmetic; arithmetic on widened integers
  (`TINYINT`, `SMALLINT`, unsigned, `HUGEINT`; mirrored with their original types, so overflow is
  an error) and on `FLOAT`; integer `/` and `%`; fractional literals in integer arithmetic;
  `TIMESTAMP - TIMESTAMP`; an ungrouped `count(*)` next to `SUM` of an `INTEGER` column; ungrouped
  `SUM` / `MIN` / `MAX` / `STDDEV_POP` over a possibly empty input; `SUM` of `FLOAT` and integer
  `SUM` / `AVG` that may leave `BIGINT`. Nullability, widened integers and integer bounds are
  tracked through the plan, so aliases, `UNION` branches and `NULL` literals cannot hide them, and
  type hazards are checked before and after the optimizer.
- **Filter pushdown** is semantics-preserving and dialect-neutral: only lossless widening casts are
  stripped; integer arithmetic and `DECIMAL` columns stay with DuckDB; NaN follows DuckDB's
  ordering; `(<predicate>) IS [NOT] NULL` is no longer pushed with inner guards; integer columns vs
  `DECIMAL` constants become exact integer bounds; `BETWEEN` is only a pre-filter; float literals
  are written in scientific notation so the host fallback reads them as `DOUBLE`.
- **Prepared statements**: `bolt_query` and `COPY ... FROM (FORMAT bolt)` read current data on every
  `EXECUTE`; `bolt_scan` fails cleanly when its table was replaced with another schema after bind.
- **Arrow bridge**: exact DuckDB types and checked casts (no silent wrap or `NULL`), invalid decimal
  scales rejected, `HUGEINT` through the unaligned `i128` path, imported buffers aligned and
  validated, stream-import panics caught, output chunks sized from the chunk capacity, dictionaries
  of any supported value type, `Utf8View` input; a batch whose column count differs from the
  schema is an error instead of leaving vectors uninitialized.
- **Runtime robustness**: worker panics (and panics during their recovery) no longer kill the
  worker; a failed thread spawn is an error, not a panic; a Bolt call inside a host-fallback query
  is refused instead of deadlocking; tables with more than 2 GiB of strings in a column no longer
  panic; one table the fallback cannot mirror no longer breaks unrelated queries; the 512 MiB worker
  stack prevents deep-predicate stack overflows; a late timeout interrupt no longer fails the next
  host-fallback query.
- `bolt_tables().on_gpu` / `gpu_note` and `bolt_pool_stats().fallback_reason` keep reporting the
  tables the GPU engine keeps on the host while the engine waits for a lazy rebuild.
- **Options**: `SET bolt_ptx_cache_dir = ''` and `SET bolt_enable_tracing = false` take effect;
  changing `bolt_device` after the first engine build is reported instead of silently ignored;
  concurrent `SET`s cannot lose updates.
- **Scalars** write one value per input row and return `NULL` without side effects for `NULL`
  arguments; `bolt_clear_tables()` clears once per worker thread; the `mode` argument and `MODE`
  option share one parser and message; `bolt_version()` does not initialize the GPU.
- **Ingestion**: Parquet ZSTD / LZ4 / Brotli and Arrow IPC LZ4 / ZSTD files load; `Date64` values
  that are not whole days are an error; zoned timestamps are stored as naive UTC (also for tables
  registered through the runtime, with `TIMESTAMPTZ` / `TIMESTAMP_TZ_NS` metadata); projected
  Parquet loads keep the file's schema metadata; a DuckDB file-system read error is reported with
  its own message instead of the decoder's.
- `aarch64` Linux builds compile; the wasm side module links with the C API v2 entrypoint and the
  runtime runs inline on wasm.

### Security
- `bolt_load_parquet` / `bolt_load_arrow_ipc` honor `enable_external_access`,
  `allowed_directories` and `allowed_paths` on canonical paths, re-check the opened handle, probe
  local files through DuckDB's file system (`disabled_filesystems`), and treat local `file:` URLs
  like paths. On Windows, paths are compared with a per-UTF-16-unit upcase that never folds wider
  than NTFS's ordinal comparison (full Unicode lower-casing let `KELVIN SIGN` match ASCII `k`).
- Hostile Arrow IPC files (out-of-file blocks, out-of-block buffers, huge declared decompressed
  sizes) are rejected before decoding; decoder panics are contained. Parquet column chunks that lie
  outside the file or exceed `memory_limit` are refused before decoding (local files too).
- `bolt_ptx_cache_dir`: refused without external access; must be absolute and contain no `..`
  component; Unix ownership / permission checks on the directory and its parents (fail-closed when
  the user id is unknown); Windows profile-directory rule; re-validation before every engine build
  and during calls; `BOLT_PTX_CACHE_DIR` checked.
- `PRAGMA bolt_reset` and `PRAGMA bolt_request_timeout` are refused in a database running with
  `lock_configuration = true` (DuckDB already refuses `SET bolt_*` there).
- The host fallback runs with `enable_external_access=false`, no extension auto-loading or
  community extensions, `lock_configuration=true`, at most 4 threads and `memory_limit=25%`.

### Performance
- Parquet files read through DuckDB's file system (`s3://`, `https://`, ...) are read with range
  requests instead of being buffered whole, so remote files larger than `memory_limit` load: two
  requests for the footer, then one per decoded column chunk (a chunk above 256 MiB, or one that
  does not fit `memory_limit` next to the chunks already held, streams through a 64 KiB
  read-ahead instead).
- Multi-batch tables are uploaded as one coalesced engine batch (was ~0.6 s per batch), and the
  catalog shares the engine's copy through zero-copy slices.
- `COPY ... TO` and `bolt_load_*` register batches without concatenating them; Parquet and IPC are
  decoded and normalized batch by batch.
- Dropping or clearing tables no longer rebuilds the engine (empty placeholders); engine build
  failures are cached; metadata reads, telemetry, metrics, devices and unfiltered `bolt_scan` are
  served from a snapshot without waiting for running queries.
- Scans are parallel (chunked results claimed by several DuckDB threads, insertion order kept);
  unfiltered `bolt_scan` streams stored batches directly and caches restored columns; filtered
  scans use an engine-only scratch table for filter-take; `count(*)` reads one narrow column.
- The host fallback mirrors only the tables a query names, re-mirrors only changed tables, copies
  mirrors in slices of 2^20 rows, bounds idle mirrors to 256 MiB, and reads results through
  DuckDB's Arrow stream; `bolt_query`, `bolt_register_sql` and host-filtered `bolt_scan` keep the
  stream's batches instead of concatenating them.
- A table with widened integer columns (`TINYINT`, `SMALLINT`, unsigned, `HUGEINT`) is copied into
  the host fallback once (schema `__bolt_mirror_base`, behind a view), not again on every switch
  between queries with and without arithmetic on those columns.
- Settings sync on every Bolt call is lock-free when nothing changed.

## [0.1.0] - 2026-10-25

### Added
- **Bidirectional `COPY (FORMAT bolt)`**:
  - `COPY <table_or_query> TO '<bolt_table>' (FORMAT bolt, [TABLE '<name>'], [MODE 'replace'|'append'|'create'])` to stream DuckDB `DataChunk`s into Craton Bolt GPU memory.
  - `COPY <duckdb_table> FROM '<bolt_table>' (FORMAT bolt)` to copy GPU-resident tables back into DuckDB tables.
- **Table Functions**:
  - `bolt_query(sql)` and `bolt_sql(sql)` for executing SQL queries directly on the Craton Bolt GPU SQL engine with projection pushdown and progress reporting.
  - `bolt_scan(table_name)` for scanning GPU-resident tables with column projection pushdown, SQL filter pushdown (`=`, `!=`, `<`, `>`, `<=`, `>=`, `BETWEEN`, `IN`, `NOT IN`, `IS NULL`, `IS NOT NULL`, `AND`, `OR`, `NOT`), and GPU row-ID gather (`execute_gpu_filter_take`) when `Utf8` columns are projected alongside GPU filter predicates.
  - `bolt_aggregate(table_name, group_by, aggregates)` for running aggregations on Craton Bolt.
  - `bolt_tables()`, `bolt_pool_stats()`, `bolt_metrics()`, and `bolt_devices()` for GPU catalog and telemetry inspection.
- **Schema-only bind** for `bolt_query` / `bolt_sql` (RFC-002): result schemas are inferred by planning, so `DESCRIBE` / `PREPARE` do not execute the query when it can be planned statically.
- **Extended filter pushdown** (RFC-003): `LIKE`-family predicates, integer/float arithmetic, and `TIMESTAMP` / `UBIGINT` / `HUGEINT` / `DECIMAL` / `DATE` literals.
- **Replacement Scans**:
  - Transparently query GPU tables via `SELECT * FROM bolt.<table_name>`, `SELECT * FROM 'bolt:<table_name>'`, `SELECT * FROM 'bolt://<table_name>'`, or the bare `SELECT * FROM <table_name>` when no DuckDB object has that name.
- **Scalar Functions**:
  - `bolt_load_parquet(table_name, path [, mode])` and `bolt_load_arrow_ipc(table_name, path [, mode])` for direct file ingestion (RFC-004).
  - `bolt_register_sql(table_name, sql)`, `bolt_drop_table(table_name)`, `bolt_clear_tables()`, `bolt_explain(sql)`, `bolt_execution_tier(sql)`, `bolt_version()`, and `bolt_prometheus_metrics()` (`bolt_metrics_prometheus()`).
- **Configuration Options & PRAGMAs**:
  - `SET bolt_device`, `SET bolt_memory_budget`, `SET bolt_ptx_cache_dir`, and `SET bolt_enable_tracing`.
  - `PRAGMA bolt_info`, `PRAGMA bolt_clear_tables`, and `PRAGMA bolt_reset`.
- **Rich Arrow-DuckDB Type Bridge**:
  - Bidirectional zero-copy / vectorized conversion for `BOOLEAN`, `TINYINT`..`HUGEINT` (with unaligned 128-bit pointer safety), `UTINYINT`..`UBIGINT`, `FLOAT`, `DOUBLE`, `DECIMAL(1..38, s)`, `DATE`, `TIMESTAMP_S`, `TIMESTAMP_MS`, `TIMESTAMP`, `TIMESTAMP_NS`, `TIMESTAMPTZ`, `VARCHAR`, `LargeUtf8`, and `Dictionary(Int32, Utf8)`.

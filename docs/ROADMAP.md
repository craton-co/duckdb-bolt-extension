# Craton Bolt DuckDB Extension — Engineering Roadmap & Feature Proposals

This document tracks architectural enhancements and feature proposals for `duckdb-bolt-extension`, maintained by [Craton Software Company](https://craton.com.ar). Defects are tracked in [`docs/issues/`](issues/README.md); engine defects reported to craton-bolt are in [`docs/upstream/`](upstream/README.md).

---

## Status Summary

| ID | Feature / Proposal | Target Module(s) | Status |
| :--- | :--- | :--- | :--- |
| **RFC-001** | Zero-copy Arrow C Data Interface streaming | `arrow_bridge.rs`, `table_functions.rs`, `copy_function.rs` | **Implemented** |
| **RFC-002** | Deferred execution & schema-only bind in `bolt_query` / `bolt_sql` | `runtime.rs`, `table_functions.rs` | **Implemented** |
| **RFC-003** | Extended filter pushdown & `bolt_aggregate` | `table_functions.rs` | **Implemented** (narrowed to semantics-preserving predicates) |
| **RFC-004** | Direct Parquet & Arrow IPC ingestion (`bolt_load_parquet` / `bolt_load_arrow_ipc`) | `ingest.rs`, `scalar_functions.rs` | **Implemented** (benchmark open) |
| **RFC-005** | Streaming results & zero-copy host scans | `runtime.rs`, `table_functions.rs` | **Partly implemented** |
| **RFC-006** | Incremental engine catalog | `runtime.rs` | **Partly implemented** (needs craton-bolt 0.10) |
| **RFC-007** | Sandboxing: honor DuckDB's file-access settings | `sandbox.rs`, `ingest.rs`, `options.rs`, `runtime/fallback.rs` | **Implemented** |
| **RFC-008** | Faithful type round-trip (original DuckDB type metadata) | `arrow_bridge.rs`, `copy_function.rs`, `ingest.rs`, `table_functions.rs` | **Implemented** |
| **RFC-009** | Per-database runtime isolation | `runtime.rs`, `options.rs` | Proposed (deferred) |
| **RFC-010** | GPU-enabled CI and signed binary distribution | `.github/workflows`, `Makefile` | **Partly implemented** |
| **RFC-011** | Upgrade to craton-bolt 0.10 and drop engine workarounds | `Cargo.toml`, `runtime.rs` | Proposed (blocked upstream) |
| **RFC-012** | Interruptible and cancellable Bolt requests | `runtime.rs`, all callbacks | Proposed (partly blocked upstream) |
| **RFC-013** | `CALL` forms for catalog mutation | `table_functions.rs` | **Implemented** |
| **RFC-014** | DuckDB-compatible engine semantics for `bolt_query` | `runtime.rs`, upstream | Proposed |

---

## Implemented

### RFC-001: Zero-copy Arrow C Data Interface streaming

DuckDB chunks are exported through DuckDB's Arrow exporter / Arrow result stream and imported with `arrow_array::ffi`; Bolt results are handed to DuckDB with `to_ffi` + `ArrowImporter::append_consumed` and referenced from the output vectors. Imported buffers are aligned and validated, stream-import panics are caught, and output chunks respect the chunk capacity. See [ARCHITECTURE.md §5](ARCHITECTURE.md#5-arrow-bridge-and-rfc-008-types-arrow_bridgers).

### RFC-002: Schema-only bind

`BoltRuntime::infer_sql_schema` plans without executing; `bolt_query` declares the inferred schema and executes in `init_global_state`, casting to it with checked casts. Queries the planner cannot type are executed once at bind time; that result is reused only by the first execution and only if the catalog is unchanged, so prepared statements see current data.

### RFC-003: Extended filter pushdown & `bolt_aggregate`

`bolt_scan` pushes comparisons, `IN` lists, `IS [NOT] NULL`, `AND` / `OR` / `NOT`, lossless widening casts, `DOUBLE` arithmetic, `LIKE`-family predicates and `BETWEEN` (as a pre-filter), only where the engine and DuckDB agree, written so both dialects read the SQL the same way. Integer arithmetic, `DECIMAL` columns and most timestamp comparisons stay with DuckDB (engine limitations: [W1-A-002](issues/W1-A-002-decimal-gpu-predicates.md), [W1-A-003](issues/W1-A-003-nan-and-overflow-semantics.md), [W1-A-006](issues/W1-A-006-temporal-pushdown-gaps.md)). `bolt_aggregate(table_name, group_by, aggregates)` builds the SQL from the resolved table name and rejects fragments that could escape it.

### RFC-004: Direct Parquet & Arrow IPC ingestion

`bolt_load_parquet` / `bolt_load_arrow_ipc` decode on the host, normalize to Bolt storage types with checked casts, and register the decoded batches without concatenation. Local paths, `file:` URLs, and any URL DuckDB's file system can open (httpfs) are supported; Parquet Snappy / Gzip / ZSTD / LZ4 / Brotli and Arrow IPC LZ4 / ZSTD decode; Arrow IPC files are validated before decoding. Remaining: the ingestion benchmark acceptance criterion, an automated `https://` load test (needs httpfs in CI) and the Arrow types Bolt cannot store ([W1-D-007](issues/W1-D-007-ingest-type-coverage.md)).

### RFC-007: Sandboxing

`bolt_load_*` apply DuckDB's `enable_external_access`, `allowed_directories` and `allowed_paths` on canonical paths (on Windows compared with a per-UTF-16-unit upcase that never folds wider than NTFS's ordinal comparison), re-check the opened handle (Linux `/proc/self/fd`, Windows `GetFinalPathNameByHandleW`, identity checks elsewhere), and probe local files through DuckDB's file system (`disabled_filesystems`); URLs go through DuckDB's file system. `SET bolt_ptx_cache_dir` is refused without external access, must be absolute without `..` components, and must be a private directory (Unix ownership / permission rules, Windows profile rule), re-checked at use time. The host fallback runs without external access, extension loading or configuration changes. A database with `lock_configuration = true` cannot change the process-wide runtime: DuckDB refuses `SET bolt_*`, and the extension refuses `PRAGMA bolt_reset` / `bolt_request_timeout`. Residual gaps: [W2-D-003](issues/W2-D-003-ptx-cache-dir-residual-hardening.md).

### RFC-008: Faithful type round-trip

Every column records its original DuckDB type in the Arrow field metadata key `duckdb.logical_type` (written by `COPY ... TO` and `bolt_load_*`). `bolt_scan`, replacement scans, `bolt_tables().schema` and `COPY ... FROM` use it; appends are checked against it; `COPY ... FROM` converts into the destination's declared types, with session-time-zone casts done by DuckDB. `bolt_query` still returns engine (storage) types; queries that do arithmetic on narrow or unsigned integers run on the host fallback over the original types.

### RFC-013: `CALL` forms for catalog mutation

`bolt_clear_tables()`, `bolt_drop_table(table_name)`, `bolt_register_sql(table_name, sql [, mode])`, `bolt_load_parquet(table_name, path [, mode])` and `bolt_load_arrow_ipc(table_name, path [, mode])` are also table functions (`CALL ...` or `SELECT * FROM ...`) that act exactly once per execution, in the global init: not at bind, so `EXPLAIN` and prepared-statement re-binds have no side effect, and again on every `EXECUTE`. The scalar forms of the same names keep their once-per-evaluated-row behavior.

---

## Partly implemented

### RFC-005: Streaming results & zero-copy host scans

Done: results are exposed as chunked, multi-batch `ChunkedResult`s scanned by several DuckDB threads in insertion order; unfiltered `bolt_scan` streams the stored batches from the metadata snapshot without an engine or host-fallback round trip; COPY and ingestion register batches without concatenating them; host-fallback results are read through DuckDB's Arrow stream.
Remaining: stream batches from the engine instead of one materialized `RecordBatch`; push `bolt_query` projections into the engine SQL.

### RFC-006: Incremental engine catalog

Done: drops and clears no longer rebuild the engine (empty placeholders, rebuild only when a query may see one or more than 128 accumulate); a failed engine build is cached until a configuration change or `PRAGMA bolt_reset`; unchanged `SET` values are no-ops; appends upload only the new rows (one coalesced `register_batch`).
Remaining: real per-table unregister and no rebuild on `bolt_memory_budget` / tracing changes, which need craton-bolt 0.10's `deregister_table` ([W1-B-006](issues/W1-B-006-engine-per-table-unregister.md), [craton-bolt-009](upstream/craton-bolt-009-no-public-unregister-table.md)).

### RFC-010: GPU-enabled CI & distribution

Done: Rust CI on Linux, Windows and macOS (rustfmt, clippy, `cargo test --features cuda-stub`), a wasm compile check, DuckDB's SQLLogicTest runner on a debug build, MSRV, `cargo deny`, a seeded differential fuzzer in the test suite, Dependabot, and the DuckDB distribution pipeline pinned to an extension-ci-tools commit.
Remaining: a CUDA build image or self-hosted GPU runner for GPU release binaries and GPU tests (the distribution pipeline's binaries are CPU-only); verifying and distributing wasm builds; publishing through the DuckDB community-extensions repository for signed binaries (`INSTALL bolt FROM community`).

---

## Proposed

### RFC-009: Per-database runtime isolation *(deferred)*

The catalog and `bolt_*` options are process-global, because craton-bolt allows one engine per process and the CUDA context is per process. Key the catalog and configuration by DuckDB database instance (sharing one engine), so `SET` in one database and tables registered by one database do not affect another. Until then the options are GLOBAL-only and settings are kept truthful across databases by `sync_with_runtime`.

### RFC-011: Upgrade to craton-bolt 0.10

craton-bolt 0.10 (unpublished) fixes the NULL crash and NULL-validity defects, adds `deregister_table`, and improves `register_batch`. Once it is on crates.io, upgrade and remove the workarounds it makes unnecessary: the NULL guard, the 512 MiB worker stack and predicate limits, upload coalescing, tombstones ([W1-O-003](issues/W1-O-003-upgrade-craton-bolt-0-10.md), [upstream README](upstream/README.md)).

### RFC-012: Interruptible requests

`PRAGMA bolt_request_timeout` bounds the wait and interrupts host-fallback queries, but `Ctrl-C` / `duckdb_interrupt()` do not reach a running Bolt request, and a GPU query cannot be cancelled. Poll DuckDB's interrupt state while waiting, abandon or interrupt the request like a timeout, and add engine cancellation once craton-bolt offers a hook ([W2-E-005](issues/W2-E-005-bolt-requests-not-interruptible.md)).

### RFC-014: DuckDB-compatible engine semantics

`bolt_query` / `bolt_sql` / `bolt_aggregate` follow the engine's semantics where they differ from DuckDB: NaN ordering and wrapping integer overflow ([W1-A-003](issues/W1-A-003-nan-and-overflow-semantics.md), [craton-bolt-010](upstream/craton-bolt-010-nan-ordering-and-integer-overflow.md)). NULL ordering, all-NULL group aggregates, empty-input aggregates, DECIMAL casts and narrow-integer / FLOAT arithmetic are already routed to the host fallback (wave 5), and so are integer `/` and `%`, fractional literals in integer arithmetic, `SUM` of `FLOAT`, integer sums that may leave `BIGINT`, `TIMESTAMP` subtraction, `STDDEV_POP` over a possibly empty input, aggregates of nullable columns below `ORDER BY` / `LIMIT` / `DISTINCT`, and every query the guard's planner cannot plan (wave 9). Ask craton-bolt for a SQL-compatible mode so these shapes can run on the GPU again.

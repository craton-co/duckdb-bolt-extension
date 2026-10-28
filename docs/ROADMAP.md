<!-- SPDX-License-Identifier: Apache-2.0 -->

# Craton Bolt DuckDB Extension — Engineering Roadmap & Feature Proposals

This document tracks architectural enhancements, feature proposals, and implementation statuses for `duckdb-bolt-extension`, maintained by [Craton Software Company](https://craton.com.ar).

---

## Status Summary

| ID | Feature / Proposal | Target Module(s) | Status |
| :--- | :--- | :--- | :--- |
| **RFC-001** | Zero-copy Arrow C Data Interface streaming | `arrow_bridge.rs`, `table_functions.rs`, `copy_function.rs` | **Implemented** |
| **RFC-002** | Deferred execution & schema-only bind in `bolt_query` / `bolt_sql` | `runtime.rs`, `table_functions.rs` | **Implemented** |
| **RFC-003** | Extended filter pushdown & `bolt_aggregate` | `table_functions.rs` | **Implemented** |
| **RFC-004** | Direct Parquet & Arrow IPC ingestion (`bolt_load_parquet` / `bolt_load_arrow_ipc`) | `ingest.rs`, `scalar_functions.rs` | **Implemented** |
| **RFC-005** | Streaming results & zero-copy host scans | `runtime.rs`, `table_functions.rs` | **Implemented** |
| **RFC-006** | Incremental engine catalog & direct memory reclaim | `runtime.rs` | **Implemented (craton-bolt 0.11.0)** |
| **RFC-007** | Sandboxing: honor DuckDB's file-access settings | `sandbox.rs`, `ingest.rs`, `options.rs`, `runtime/fallback.rs` | **Implemented** |
| **RFC-008** | Faithful type round-trip (original DuckDB type metadata) | `arrow_bridge.rs`, `copy_function.rs`, `ingest.rs`, `table_functions.rs` | **Implemented** |
| **RFC-009** | Per-database runtime isolation | `runtime.rs`, `options.rs` | Proposed (deferred) |
| **RFC-010** | GPU-enabled CI and signed binary distribution | `.github/workflows`, `Makefile`, `scripts` | **Partly implemented** |
| **RFC-011** | Upgrade to craton-bolt 0.11.0 & retire CPU fallback gatekeepers | `Cargo.toml`, `runtime.rs`, `arrow_bridge.rs` | **Implemented (craton-bolt 0.11.0)** |
| **RFC-012** | Interruptible and cancellable Bolt requests via `CancellationToken` | `runtime.rs`, all callbacks | **Implemented (craton-bolt 0.11.0)** |
| **RFC-013** | `CALL` forms for catalog mutation | `table_functions.rs` | **Implemented** |
| **RFC-014** | DuckDB-compatible standard SQL semantics for `bolt_query` | `runtime.rs`, engine config | **Implemented (craton-bolt 0.11.0)** |

---

## Implemented

### RFC-001: Zero-copy Arrow C Data Interface streaming

DuckDB chunks are exported through DuckDB's Arrow exporter / Arrow result stream and imported with `arrow_array::ffi`; Bolt results are handed to DuckDB with `to_ffi` + `ArrowImporter::append_consumed` and referenced from the output vectors. Imported buffers are aligned and validated, stream-import panics are caught, and output chunks respect chunk capacity. See [ARCHITECTURE.md §5](ARCHITECTURE.md#5-arrow-bridge-and-rfc-008-types-arrow_bridgers).

### RFC-002: Schema-only bind

`BoltRuntime::infer_sql_schema` plans without executing; `bolt_query` declares the inferred schema and executes in `init_global_state`, casting to it with checked casts. Queries the planner cannot type are executed once at bind time; that result is reused only by the first execution and only if the catalog is unchanged, ensuring prepared statements see current data.

### RFC-003: Extended filter pushdown & `bolt_aggregate`

`bolt_scan` pushes comparisons, `IN` lists, `IS [NOT] NULL`, `AND` / `OR` / `NOT`, lossless widening casts, `DOUBLE` arithmetic, `LIKE`-family predicates, and `BETWEEN` (as a pre-filter) directly to the engine where supported. Extended Arrow data types (`Extended`) and GPU-safe NULL predicates evaluate natively on CUDA. `bolt_aggregate(table_name, group_by, aggregates)` constructs SQL from the resolved table identifier and rejects injection fragments.

### RFC-004: Direct Parquet & Arrow IPC ingestion

`bolt_load_parquet` / `bolt_load_arrow_ipc` decode on the host, normalize to Bolt storage types with checked casts, and register decoded batches without concatenation. Local paths, `file:` URLs, and any URL DuckDB's file system can open (e.g. `httpfs`) are supported; Parquet Snappy, Gzip, ZSTD, LZ4, and Brotli and Arrow IPC LZ4 / ZSTD decode; Arrow IPC files are validated before decoding.

### RFC-005: Streaming results & zero-copy host scans

Results are exposed as chunked, multi-batch `ChunkedResult` structures scanned by multiple DuckDB threads in parallel and preserving insertion order. Unfiltered `bolt_scan` streams stored batches from the metadata snapshot without an engine or host-fallback round trip. `COPY ... TO` and ingestion register batches without concatenating them, and host-fallback results are read directly through DuckDB's Arrow stream.

### RFC-006: Incremental engine catalog & direct memory reclaim

With the adoption of `craton-bolt 0.11.0`, dropped tables and cleared catalogs call `Engine::drop_table(&key)` and `Engine::deregister_table(&key)` directly. This releases device memory immediately and eliminates the legacy tombstone empty-batch mechanism. Unchanged configuration updates remain zero-cost no-ops, failed engine builds are cached until configuration changes or `PRAGMA bolt_reset`, and table appends upload only newly added rows.

### RFC-007: Sandboxing

`bolt_load_*` enforce DuckDB's security boundaries (`enable_external_access`, `allowed_directories`, and `allowed_paths`) on canonical paths. On Windows, comparisons use case-insensitive per-unit ordinal matching aligned with NTFS semantics. Opened file handles are re-checked via operating system APIs (`/proc/self/fd` on Linux, `GetFinalPathNameByHandleW` on Windows, device/inode identity on macOS/BSD). `SET bolt_ptx_cache_dir` requires external access, rejects relative paths or `..` traversals, and must be a private directory owned by the user. Databases with `lock_configuration = true` cannot mutate process-wide settings.

### RFC-008: Faithful type round-trip

Every column records its original DuckDB type in the Arrow field metadata key `duckdb.logical_type` (written by `COPY ... TO` and `bolt_load_*`). `bolt_scan`, replacement scans, `bolt_tables().schema`, and `COPY ... FROM` read this metadata; table appends are validated against it; and `COPY ... FROM` converts back into the destination table's declared types with session-time-zone conversions performed by DuckDB.

### RFC-011: Upgrade to craton-bolt 0.11.0 & retire CPU fallback gatekeepers

Upgraded dependency to `craton-bolt 0.11.0`. The legacy CPU fallback gatekeepers were retired:
- **Stack Guards**: Removed `MAX_GPU_PREDICATE_TERMS` and `MAX_GPU_DICTIONARY_LOWERED_VALUES` stack-guard limits. Deep predicate trees now evaluate on the GPU.
- **NULL & Join Guard Relaxation**: Relaxed `gpu_crash_guard` so queries over tables with NULLs and outer joins stay on the GPU.
- **Extended Types**: Added support for `DataType::Extended` across `arrow_bridge.rs` and `runtime.rs` (Lists, Structs, Maps, Unions, Decimals, Durations, Intervals, Time32/64, BinaryView).
- **Differential Fuzz Verification**: Verified zero divergence against DuckDB across 2,400 randomized queries on 200 tables on CUDA.

### RFC-012: Interruptible and cancellable Bolt requests via `CancellationToken`

Wired DuckDB's interrupt handler into `craton_bolt::CancellationToken`. Queries executed via `handle_execute_sql` pass the token to `Engine::sql_with_cancel(sql, &token)`. When a query times out under `PRAGMA bolt_request_timeout` or is interrupted by the user, the token is triggered, cancelling the GPU query in-flight without leaking background execution threads.

### RFC-013: `CALL` forms for catalog mutation

`bolt_clear_tables()`, `bolt_drop_table(table_name)`, `bolt_register_sql(table_name, sql [, mode])`, `bolt_load_parquet(table_name, path [, mode])`, and `bolt_load_arrow_ipc(table_name, path [, mode])` are registered as both scalar functions and table functions (`CALL ...` or `SELECT * FROM ...`). The table-function variants execute exactly once in `init_global_state`, preventing side-effects during `EXPLAIN` or prepared-statement re-binds.

### RFC-014: DuckDB-compatible standard SQL semantics

Configured `EngineBuilder::new().sql_semantics(SqlSemantics::Standard)` on all GPU engines. This instructs `craton-bolt` to apply standard SQL semantics matching DuckDB:
- IEEE total ordering for floating-point `NaN` comparisons.
- Standard `NULLS LAST` ordering for ascending sorts.
- Checked integer arithmetic with overflow detection.

---

## Partly Implemented

### RFC-010: GPU-enabled CI & binary distribution

- **Completed**: Rust CI on Linux, Windows, and macOS (rustfmt, clippy, `cargo test --features cuda-stub`), WebAssembly compilation check, DuckDB SQLLogicTest test runner integration, MSRV (Rust 1.88), `cargo deny` license and dependency auditing, seeded differential fuzzer in the test suite, and packaging scripts (`scripts/package_extension.py`).
- **In Progress**: Dedicated GPU self-hosted runners for continuous automated CUDA execution testing in CI, publication to the DuckDB community extensions repository for signed binary distribution (`INSTALL bolt FROM community`), and automated WebAssembly artifact deployment.

---

## Proposed

### RFC-009: Per-database runtime isolation *(deferred)*

The Bolt catalog and configuration pragmas are process-global because `craton-bolt` operates with a single engine and CUDA context per host process. Future work will key catalog entries and session options by DuckDB database instance (while sharing the underlying device memory pool), so multiple attached databases do not mutate each other's registered tables or configuration. Until then, settings are global and synchronized across connections via `sync_with_runtime`.

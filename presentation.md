<!--
applies-to: main
owner: maintainers
product: duckdb-bolt-extension
version-neutral: crate version and MSRV live in Cargo.toml
-->

# DuckDB Bolt Extension (`bolt`) — Product Presentation

---

## Slide 1 · What Is DuckDB Bolt Extension?

> An in-process DuckDB v2.0 extension written in Rust that embeds the **[Craton Bolt](https://github.com/craton-co/craton-bolt)** JIT-compiled GPU SQL engine inside DuckDB.

- **What**: A loadable DuckDB extension (`bolt`) that accelerates analytical SQL workloads using NVIDIA GPUs directly inside the DuckDB process space.
- **Language & Stack**: Pure Rust (2024 edition, MSRV 1.88) built on DuckDB's C API v2 (`duckdb-neo`, `C_STRUCT_UNSTABLE`), Apache Arrow 53, and the CUDA Driver API.
- **Engine Core**: Powered by `craton-bolt` 0.10.0—compiling SQL queries into fused NVIDIA PTX kernels at runtime.
- **Zero IPC Overhead**: Runs entirely in-process; data exchanges between DuckDB vectors and GPU device buffers occur over the standard Arrow C Data Interface without intermediate copies.
- **Status**: Version 0.1.0, active development, licensed under Apache-2.0. Developed by [Craton Software Company](https://craton.com.ar).

---

## Slide 2 · The Five Distinguishing Ideas

**1 · In-Process GPU Compute with Isolated Worker Runtime**  
DuckDB worker threads schedule parallel tasks dynamically, while CUDA contexts are bound to a single OS thread. The extension solves this with a dedicated background worker (`duckdb-bolt-worker`) configured with a 512 MiB virtual call stack to handle deep expression trees without stack overflows.

**2 · Zero-Copy Arrow C Data Interface Bridging**  
Tables uploaded from DuckDB (`COPY ... TO (FORMAT bolt)`) or scanned back (`bolt_scan` / replacement scan) interchange record batches using Arrow C Data Interface pointers (`to_ffi` / `from_ffi`). DuckDB output vectors reference ref-counted Arrow buffers directly (`Vector::reference`).

**3 · Faithful RFC-008 Type Preservation**  
GPU execution engines normalize types to physical Arrow layouts (e.g., storing `SMALLINT` in `Int32`, `TIMESTAMPTZ` in naive UTC timestamps). RFC-008 embeds original DuckDB logical types in Arrow field metadata (`duckdb.logical_type`). Round-tripping tables through Bolt preserves original types with zero precision loss.

**4 · Fail-Closed GPU Crash Guard & Host Fallback**  
GPU kernels are vulnerable to crashes or silent correctness issues on edge cases (NULL validity bitmaps, IEEE vs SQL NaN ordering, integer overflow, deep predicate trees). The `gpu_crash_guard` statically analyzes query plans. If a query is unsafe for GPU kernels, it transparently executes on an embedded, sandboxed DuckDB instance (`FallbackDb`) with bit-exact DuckDB semantics.

**5 · Native DuckDB SQL Ergonomics**  
No custom client libraries or daemons required. Query Bolt tables directly using replacement scans (`SELECT ... FROM bolt.my_table WHERE ...`), complete with automatic projection pushdown and GPU filter-take acceleration.

---

## Slide 3 · Architecture & Runtime Pipeline

```
+-------------------------------------------------------------------------+
|                  DuckDB Worker Threads (Queries / COPY)                 |
+-------------------------------------------------------------------------+
            │                                             ▲
     Task Envelopes                               Snapshot Read
   (Worker Commands)                             (Lock-free Arcs)
            │                                             │
            ▼                                             │
+─────────────────────────────────────────────────────────────────────────+
|               duckdb-bolt-worker (Dedicated OS Thread)                  |
|                                                                         |
|  +───────────────────────────────────────────────────────────────────+  |
|  |                gpu_crash_guard (Static Plan Analysis)             |  |
|  +─────────────────┬─────────────────────────────────┬───────────────+  |
|                    │ (Safe Shapes)                   │ (Risky Shapes)|
|                    ▼                                 ▼               |
|  +──────────────────────────────────+  +─────────────────────────────+  |
|  |     craton_bolt::Engine (CUDA)   |  | FallbackDb (DuckDB host-fb) |  |
|  |  • PTX JIT Kernel Compilation    |  | • Isolated embedded DuckDB  |  |
|  |  • VRAM Pool & Arrow Buffers     |  | • Exact DuckDB semantics    |  |
|  +──────────────────────────────────+  +─────────────────────────────+  |
+─────────────────────────────────────────────────────────────────────────+
```

### Zero-Contention Metadata Snapshots
After every catalog mutation, the worker publishes an immutable snapshot (`MetaSnapshot`). Read-only operations (`bolt_tables()`, `bolt_pool_stats()`, bind-time schema resolution, and raw table scans) read directly from this snapshot on the calling thread without queuing behind running GPU queries.

---

## Slide 4 · What's In the Box

The workspace consists of three Cargo crates targeting DuckDB's C API v2:

| Crate | Path | Output | Purpose |
|:---|:---|:---|:---|
| `duckdb-bolt-core` | `crates/core` | `rlib` (crates.io) | Core runtime logic, engine guard, Arrow bridge, fallback DB, functions. |
| `duckdb-bolt-extension` | `crates/native` | `cdylib` | Loadable extension binary (`bolt.duckdb_extension`). |
| `duckdb-bolt-wasm` | `crates/wasm` | `staticlib` | WebAssembly build target (`bolt_init_c_api_v2`). |

### Key Core Modules

| Module | What it does |
|:---|:---|
| `runtime.rs` | `BoltRuntime` singleton, worker thread, command dispatch, request state machine, shadow catalog, metadata snapshot. |
| `runtime/engine_support.rs` | Batch coalescing, string capacity checks, plan inspection for null-safety and type hazards. |
| `runtime/plan_facts.rs` | Column-level plan inference (`NodeFacts`, `ColumnFact`, nullability, integer bounds, emptiness checks). |
| `runtime/fallback.rs` | `FallbackDb` embedded DuckDB instance with mirrored tables and dialect compatibility. |
| `arrow_bridge.rs` | DuckDB <-> Arrow C Data Interface conversion, RFC-008 original-type serialization and restoration. |
| `table_functions.rs` | `bolt_query`, `bolt_sql`, `bolt_scan`, `bolt_aggregate`, `bolt_tables`, `bolt_pool_stats`, `bolt_devices`. |
| `replacement_scan.rs` | Rewrites `bolt.<t>`, `'bolt:<t>'`, and bare `<t>` to `bolt_scan('<t>')` with pushdown. |
| `copy_function.rs` | `COPY ... TO / FROM (FORMAT bolt)` execution with checked appends. |
| `ingest.rs` | Host-side Parquet and Arrow IPC decoding (local paths, VFS, `s3://`, `https://`). |
| `sandbox.rs` | Sandboxing: enforces `enable_external_access`, `allowed_directories`, and Windows NTFS Unicode normalization. |

---

## Slide 5 · SQL Surface & Ergonomics

| Kind | Syntax / Name | Purpose |
|:---|:---|:---|
| **COPY** | `COPY table TO 'bolt_table' (FORMAT bolt [, MODE ...])` | Upload table into Bolt catalog (`MODE 'replace'`, `'append'`, `'create'`). |
| **COPY** | `COPY duckdb_table FROM 'bolt_table' (FORMAT bolt)` | Copy Bolt table into DuckDB table (column matching, checked casts). |
| **Table Function** | `bolt_query('<sql>')` / `bolt_sql('<sql>')` | Execute SQL directly on Craton Bolt GPU engine. |
| **Table Function** | `bolt_scan('bolt_table')` | Scan Bolt table into DuckDB with projection & filter pushdown. |
| **Table Function** | `bolt_aggregate('table', 'group_by', 'aggs')` | Run grouped aggregations on Bolt with sanitization. |
| **Replacement Scan** | `SELECT ... FROM bolt.table WHERE ...` | Transparent query rewriting with pushdown to `bolt_scan`. |
| **Scalar / CALL** | `bolt_load_parquet('table', 'path')` | Ingest Parquet directly into Bolt without landing in DuckDB. |
| **Scalar / CALL** | `bolt_load_arrow_ipc('table', 'path')` | Ingest Arrow IPC / Feather v2 directly into Bolt. |
| **Scalar / CALL** | `bolt_drop_table('table')`, `bolt_clear_tables()` | Drop single or all tables from catalog. |
| **Introspection** | `bolt_explain('<sql>')`, `bolt_execution_tier('<sql>')` | Inspect plan routing (`'Gpu'`, `'Hybrid'`, `'Host'`) and fallback reasons. |
| **Telemetry** | `bolt_tables()`, `bolt_pool_stats()`, `bolt_devices()` | Catalog inventory, VRAM pool stats, PTX cache hits, CUDA device info. |
| **Pragma** | `PRAGMA bolt_info`, `bolt_reset`, `bolt_clear_tables` | Full engine telemetry, eager GPU reset, catalog wipe. |
| **Metrics** | `bolt_prometheus_metrics()` | Prometheus text exporter for Grafana and monitoring alerts. |

---

## Slide 6 · Data Types & RFC-008 Type Fidelity

The extension maps DuckDB types to Arrow storage while preserving logical definitions via RFC-008 metadata (`duckdb.logical_type`):

| DuckDB Type | Physical Storage | `bolt_scan` Output | `bolt_query` Output |
|:---|:---|:---|:---|
| `BOOLEAN` | `Boolean` | `BOOLEAN` | `BOOLEAN` |
| `TINYINT`, `SMALLINT`, `INTEGER` | `Int32` | Original Type | `INTEGER` |
| `BIGINT`, `UINTEGER` | `Int64` | Original Type | `BIGINT` |
| `UBIGINT`, `HUGEINT` | `Int64` (range checked) | Original Type | `BIGINT` |
| `FLOAT`, `DOUBLE` | `Float32`, `Float64` | `FLOAT`, `DOUBLE` | `FLOAT`, `DOUBLE` |
| `VARCHAR` | `Utf8` (dictionary/raw) | `VARCHAR` | `VARCHAR` |
| `DATE` | `Date32` | `DATE` | `DATE` |
| `TIMESTAMP_S / MS / US / NS` | `Timestamp(s/ms/us/ns)` | Same | Same |
| `TIMESTAMPTZ` | naive `Timestamp(us)` (UTC instant) | `TIMESTAMPTZ` (session tz) | `TIMESTAMP` (UTC wall clock) |
| `DECIMAL(p, s)` | `Decimal128(p, s)` | Same | Same |

*Note: Unsupported types (`BLOB`, `UUID`, `INTERVAL`, nested lists/structs) fail fast on upload. Conversions never produce silent NULLs.*

---

## Slide 7 · Safety Architecture: The GPU Crash Guard

Hardware acceleration must never compromise database stability. `gpu_crash_guard(sql)` runs before any query reaches the GPU engine:

```
                            Incoming SQL Query
                                    │
                                    ▼
                     ┌─────────────────────────────┐
                     │   Static Plan Analyzer      │
                     │      gpu_crash_guard        │
                     └──────────────┬──────────────┘
                                    │
           ┌────────────────────────┴────────────────────────┐
           ▼                                                 ▼
      [Passes Guard]                                  [Fails Guard]
 • Null-safe query structure                    • Nullable aggregate hazard
 • Supported types & bounds                     • Type hazards (overflow / cast)
 • < 4,096 predicate terms                      • Complex subquery / lateral join
 • Uploaded & admitted table                    • Memory limit / missing table
           │                                                 │
           ▼                                                 ▼
┌─────────────────────────────┐               ┌─────────────────────────────┐
│  craton_bolt::Engine (GPU)  │               │    FallbackDb (DuckDB CPU)  │
│  • NVIDIA PTX JIT           │               │  • Embedded DuckDB engine   │
│  • Massive SIMT Parallelism │               │  • Bit-exact DuckDB logic   │
└─────────────────────────────┘               └─────────────────────────────┘
```

- **`bolt_execution_tier(sql)`**: Returns `'Gpu'` or `'Host'` dynamically without executing the query.
- **`bolt_explain(sql)`**: Explains the plan and prefixes fallback queries with the exact safety reason (e.g., `=== Executed on the host fallback (DuckDB): nullable grouped aggregate hazard ===`).

---

## Slide 8 · Security & Sandboxing

`duckdb-bolt-extension` fully adheres to DuckDB's multi-tenant security configuration:

1. **Filesystem Sandbox Enforcement**:
   - `bolt_load_parquet` and `bolt_load_arrow_ipc` strictly enforce `enable_external_access`, `allowed_directories`, and `allowed_paths`.
   - File handles are re-checked after canonicalization to defeat symlink race attacks (`/proc/self/fd` on Linux, `GetFinalPathNameByHandleW` on Windows).
2. **Windows NTFS Unicode Case-Folding Protection**:
   - Path comparisons on Windows use a per-UTF-16-unit uppercase transform that prevents NTFS ordinal case-folding bypasses.
3. **PTX Cache Directory Validation**:
   - `SET bolt_ptx_cache_dir` requires `enable_external_access = true`, must be an absolute path, and enforces private directory ownership (Unix `0700`, Windows user SID check).
4. **Configuration Lock**:
   - When DuckDB sets `lock_configuration = true`, `SET bolt_*` options, `PRAGMA bolt_reset`, and timeout modifications are strictly prohibited.

---

## Slide 9 · Verification Discipline & Differential Fuzzing

To guarantee 100% semantic correctness against DuckDB, we built a layered testing regimen:

- **SQLLogicTest Suite**: End-to-end tests covering options, table functions, replacement scans, pushdown, transactions, and types (`test/sql/*.test`).
- **Offline CPU Testing**: Built and tested in CI via the `cuda-stub` feature without requiring physical CUDA hardware.
- **Differential Fuzzing (`tests/w4b_differential_fuzz.rs`)**:
  - Automatically synthesizes random, complex SQL queries across mixed numeric, temporal, and string schemas.
  - Executes queries against native DuckDB and `bolt_scan` / `bolt_query`.
  - Verifies that the symmetric difference is strictly zero:
    ```sql
    (SELECT * FROM local_tbl EXCEPT ALL SELECT * FROM bolt_scan('gpu_tbl'))
    UNION ALL
    (SELECT * FROM bolt_scan('gpu_tbl') EXCEPT ALL SELECT * FROM local_tbl)
    -- Must return 0 rows
    ```
  - Identified and surfaced 10 upstream engine issues to `craton-bolt` during pre-release audits.

---

## Slide 10 · Production Observability

Built-in telemetry surfaces hardware metrics and execution states directly to SQL:

```sql
-- 1. Full engine telemetry overview
PRAGMA bolt_info;

-- 2. VRAM pool allocations and plan cache stats
SELECT allocated_bytes, peak_bytes, ptx_cache_hits, ptx_cache_misses
FROM bolt_pool_stats();

-- 3. Hardware device inventory
SELECT device_id, name, total_memory_bytes, compute_capability
FROM bolt_devices();

-- 4. Native Prometheus metrics exporter
SELECT bolt_prometheus_metrics();
```

Exported metrics include `bolt_pool_allocated_bytes`, `bolt_pool_peak_bytes`, `bolt_queries_total`, `bolt_ptx_cache_hits`, and `bolt_host_fallback_total` for immediate Grafana / Prometheus integration.

---

## Slide 11 · Quick Start & SQL Example

```sql
-- Load the extension
LOAD 'build/release/bolt.duckdb_extension';

-- 1. Create a large dataset in DuckDB
CREATE TABLE telemetry AS
SELECT
    i::INTEGER AS device_id,
    CASE WHEN i % 2 = 0 THEN 'sensor_a' ELSE 'sensor_b' END AS sensor_type,
    (i * 0.42)::DOUBLE AS reading,
    (i % 5 != 0) AS is_valid
FROM range(1, 10000001) t(i);

-- 2. Ingest 10 million rows into Craton Bolt GPU memory
COPY telemetry TO 'gpu_telemetry' (FORMAT bolt);

-- 3. Run high-throughput SQL on the GPU
SELECT * FROM bolt_query('
    SELECT sensor_type,
           COUNT(device_id) AS sample_count,
           SUM(reading) AS total_reading,
           AVG(reading) AS avg_reading
    FROM gpu_telemetry
    WHERE is_valid = TRUE
    GROUP BY sensor_type
');

-- 4. Query seamlessly with DuckDB replacement scan & filter pushdown
SELECT device_id, reading
FROM bolt.gpu_telemetry
WHERE sensor_type = 'sensor_a' AND reading > 100000.0
LIMIT 10;
```

---

## Summary

| | |
|:---|:---|
| **What** | DuckDB v2.0 C API extension embedding Craton Bolt 0.10.0 JIT-compiled GPU SQL engine |
| **Architecture** | In-process dedicated worker OS thread (512 MiB virtual stack) + lock-free metadata snapshots |
| **Data Bridge** | Zero-copy Apache Arrow C Data Interface with ref-counted buffer sharing |
| **Type System** | RFC-008 original DuckDB logical type preservation |
| **Safety** | Static `gpu_crash_guard` with automatic transparent `FallbackDb` (DuckDB CPU) |
| **Ergonomics** | `COPY (FORMAT bolt)`, `bolt_query`, `bolt_scan`, and `bolt.table` replacement scans |
| **Security** | DuckDB external access, allowed paths sandbox, NTFS case-folding defense, PTX hardening |
| **Verification** | Offline `cuda-stub` CI, SQLLogicTests, and differential fuzzing against native DuckDB |
| **License** | Apache-2.0 |
| **Repository** | [https://github.com/craton-co/duckdb-bolt-extension](https://github.com/craton-co/duckdb-bolt-extension) |
| **Maintainer** | [Craton Software Company](https://craton.com.ar) (`opensource@craton.com.ar`) |

---

Copyright 2026 Craton Software Company. Licensed under Apache-2.0.

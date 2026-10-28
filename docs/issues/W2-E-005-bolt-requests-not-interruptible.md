<!-- SPDX-License-Identifier: Apache-2.0 -->

# W2-E-005: Bolt work cannot be interrupted or timed out

| Field | Value |
| :--- | :--- |
| Title | Bolt work cannot be interrupted or timed out |
| Severity | medium |
| Status | blocked-upstream |
| Kind | bug |
| Confidence | high (by reading) |
| Files | `crates/core/src/runtime.rs:345-359` (`BoltRuntime::request`), `crates/core/src/table_functions.rs:182-208` / `:308-447` (`init_global_state` of `bolt_query` / `bolt_scan`), `crates/core/src/scalar_functions.rs` |

## Description

Every runtime call sends a command to the single worker thread and blocks in
`reply_rx.recv()` with no timeout and no cancellation path. The work happens inside one DuckDB
callback (`bind` or `init_global_state` for `bolt_query` / `bolt_scan` / `bolt_aggregate`, the
scalar `exec` for `bolt_register_sql` / `bolt_load_*`, `finalize` for `COPY ... TO`). DuckDB only
checks for interrupts between callbacks, so:

- `Ctrl-C` in the CLI, `duckdb_interrupt()`, Python `KeyboardInterrupt`, or a client-side query
  timeout has no effect until the GPU query (or the host-fallback DuckDB query, or the Parquet
  decode) finishes, however long it takes.
- An abandoned request still runs to completion on the worker, and every other Bolt call from any
  connection queues behind it (W1-B-004 covers the queueing, not the missing cancellation).
- The host fallback runs on its own embedded DuckDB, which the host's interrupt does not reach.

## Reproduction

```sql
-- cuda-stub build (host fallback); any long-running query works
COPY (SELECT range AS i FROM range(100000)) TO 'gpu_cancel' (FORMAT bolt, USE_TMP_FILE FALSE);
SELECT count(*) FROM bolt_query('SELECT a.i FROM gpu_cancel a, gpu_cancel b WHERE a.i + b.i = -1');
-- press Ctrl-C: expected "INTERRUPT Error"; by reading the CLI waits for the full 10^10-row join
```

## Suggested fix

- Give each request a cancellation token (`Arc<AtomicBool>`) and wait with
  `recv_timeout` in a loop, checking DuckDB's interrupt state between waits (duckdb-neo needs to
  expose `ClientContext::interrupted` / `duckdb_v2_..._is_interrupted`; file upstream if missing).
- On interrupt, set the token and call `Connection::interrupt()` on the fallback connection
  (`FallbackDb`); for the GPU path, check the token between plan stages if craton-bolt offers a
  hook, otherwise at least stop waiting and return `INTERRUPT` to DuckDB.
- Long term, RFC-005 streaming scans make the work per chunk, which DuckDB can interrupt.

## W3-A notes

Partly addressed in 6dcd2b9 (W3-A): new
`PRAGMA bolt_request_timeout(<ms>)` (`0` = no limit, process-wide) and
`BoltRuntime::set_request_timeout` / `request_timeout`. A caller waits at most that long; on
timeout it gets an error and a host-fallback query in flight is interrupted
(`Connection::interrupt_handle` of the fallback connection). A GPU query cannot be cancelled
(craton-bolt 0.9 has no hook) and keeps running; later requests queue behind it. Test
`crates/core/tests/w3a_runtime_regression.rs::request_timeout_interrupts_host_fallback_queries`. Still open: honoring DuckDB's own
interrupt (Ctrl-C) needs a duckdb-neo API to poll the client context's interrupt state, and an
interrupt may hit a later fallback query of another request if the timed-out one already
finished (benign race).

## W4-A notes (blocked upstream)

Re-checked in b1158f0 (W4-A):
- **Ctrl-C / `duckdb_interrupt()`**: duckdb-neo 0.20000 `Context` exposes only `config()`;
  the C API (`libduckdb-sys` 1.20000) has `duckdb_v2_context_get_config`, `_get_factory`,
  `_get_file_system`, `_log`, but no way to read the client context's interrupt state.
  **Upstream ask (duckdb C API v2 / duckdb-neo):** `bool duckdb_v2_context_is_interrupted(duckdb_v2_context)`
  (or an interrupt-handle / callback registration on the client context), exposed as
  `Context::is_interrupted()`. With it, `BoltRuntime::request` can poll between
  `recv_timeout` slices and abandon / interrupt exactly like the timeout path does now.
- **GPU cancellation**: craton-bolt 0.9 `Engine::sql` has no cancellation token or
  per-stage hook. **Upstream ask (craton-bolt):** `Engine::sql_with_cancel(&str, &AtomicBool)`
  or a `CancellationToken` checked between physical operators / kernel launches.
- Timeout semantics were tightened (W4-D-003): a request that has not started is abandoned
  (no effect), a running query interrupts only its own fallback query, a running mutation is
  waited for.

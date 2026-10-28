<!-- SPDX-License-Identifier: Apache-2.0 -->

# W2-A-001: craton-bolt 0.9.0 GPU filters and expressions ignore NULL validity

| Field | Value |
| :--- | :--- |
| Title | craton-bolt 0.9.0 GPU filters and expressions ignore NULL validity |
| Severity | high |
| Status | blocked-upstream |
| Kind | bug (upstream craton-bolt 0.9.0; mitigated here) |
| Files | upstream `craton_bolt::Engine::sql`; mitigation `crates/core/src/runtime.rs` (`plan_is_null_safe`, `BoltWorkerState::gpu_crash_guard`) |

## Description

Found while fixing W1-O-002 (verified by running craton-bolt alone on an RTX 2060). The GPU
expression evaluator reads the values buffer and ignores the Arrow validity bitmap. With a
nullable `d DOUBLE` column whose NULL slots hold non-zero values (DuckDB-exported arrays do):

| Query | craton-bolt 0.9.0 | Correct |
| :--- | :--- | :--- |
| `SELECT d FROM t WHERE d > 0` | 79 rows, 0 NULLs | 67 rows |
| `SELECT d FROM t WHERE id > 20` | 79 rows, **0 NULLs** | 79 rows, 12 NULLs |
| `SELECT d FROM t WHERE d IS NULL` | 15 rows, **0 NULLs** (garbage values) | 15 NULLs |
| `SELECT d + 1 FROM t` | 0 NULLs | 15 NULLs |
| `SELECT CASE WHEN d > 0 THEN 1 ELSE 0 END FROM t` | 79 ones | 67 ones |

Correct: plain projections, `ORDER BY` on a column with `DESC` or explicit `NULLS LAST`,
`LIMIT`, `DISTINCT`, `UNION ALL`, equi joins, `COUNT`/`AVG` of plain columns, and ungrouped
`SUM`/`MIN`/`MAX` over a non-empty input. Wave 5 found that grouped `SUM`/`MIN`/`MAX` over a
column with NULLs (W4-B-001), the default NULLS FIRST sort order (W4-B-003), `count(*)` next to
an INTEGER `SUM` (W5-A-001) and ungrouped aggregates over an empty input (W5-A-003) are also
wrong; the guard now routes those shapes to the host too.
With arrow-rs arrays built from `Option` values (zeros under NULLs) many of these happen to give
the right answer, which is why the upstream test suite does not catch it.

## Mitigation

Commit 6be420d routes every query over a table with NULLs to the host fallback unless the plan is
null-safe (see W1-O-002). Cost: such queries run on the CPU.

## Suggested fix

Upstream: honour validity in the filter/projection kernels (or upgrade, W1-O-003), then relax
`plan_is_null_safe`. Re-run `w2a_engine_crash_guard.rs` with the guard disabled to verify.

## W3-A notes

6dcd2b9 (W3-A): guard coverage extended. Besides stored
NULLs, the guard now plans queries that can produce NULLs themselves (outer joins, `CASE`, NULL
literals; `plan_may_produce_nulls`), treats outer joins as not null-safe (`plan_is_null_safe`),
and recognizes tables whose names need escaping (W3-E-002). Tests:
`crates/core/tests/w3a_runtime_regression.rs::query_generated_nulls_and_escaped_names_are_guarded` plus the existing
`w2a_engine_crash_guard.rs`. The root cause is upstream; this page stays open.

Upstream reports: [craton-bolt-002](../upstream/craton-bolt-002-gpu-kernels-ignore-null-validity.md), [craton-bolt-001](../upstream/craton-bolt-001-crash-filtered-aggregate-over-nulls.md) (W4-D).
## W4-A notes (blocked upstream)

Re-verified in b1158f0: still craton-bolt 0.9.0; the guard (`plan_is_null_safe`,
`plan_may_produce_nulls`, `gpu_crash_guard`) is unchanged and its tests pass on the RTX 2060.
**Upstream ask (craton-bolt-002):** honour the Arrow validity bitmap in the filter, projection
and expression kernels (and propagate validity to outputs), with tests built from arrays whose
NULL slots hold non-zero values; then release it. Relax `plan_is_null_safe` after re-running
`w2a_engine_crash_guard.rs` with the guard disabled.

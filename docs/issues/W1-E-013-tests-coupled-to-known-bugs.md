<!-- SPDX-License-Identifier: Apache-2.0 -->

# W1-E-013: Tests that must change when known bugs are fixed

| Field | Value |
| :--- | :--- |
| Title | Tests that must change when known bugs are fixed |
| Severity | low |
| Status | blocked-upstream |
| Kind | test |
| Files | `test/sql/bolt.test` (`PRAGMA bolt_reset`), `crates/core/tests/integration_test.rs`, `test/sql/bolt_types.test`, `test/sql/bolt_pushdown.test` |

## Description

To keep the suites green on both backends while report bugs are open, W1-E left out or tolerated
the following; each must be tightened by whoever fixes the bug.

### Still open

| Report item | Where | Current handling | When fixed |
| :--- | :--- | :--- | :--- |
| 1.1.2 TIMESTAMPTZ storage (pushdown part fixed in W1-A, storage still naive `Timestamp(us)`) | `bolt_types.test`, `bolt_pushdown.test` headers (`KNOWN-BUG (report 1.1.2)`) | TIMESTAMPTZ columns left out | add a TIMESTAMPTZ column, a non-UTC `SET TimeZone`, and pushdown on it |
| W1-O-002 engine crash on filtered aggregates over NULLs | `test/sql/bolt_types.test` (`bolt_query('SELECT SUM(v) ... WHERE d IS NULL')`) | kept as is (W2-A mitigates) | keep; add the reverse (`d IS NOT NULL`) once craton-bolt 0.10 lands |

The pushdown differential tests are only meaningful on `cuda-gpu`; CI (`cuda-stub`) passes them
trivially. A GPU CI job (RFC-010) is needed for them to guard against regressions.

### Tightened by W2-E (branch `bolt/w2-e`)

| Report item | Change |
| :--- | :--- |
| 1.1.1 explicit CAST stripped | `bolt_pushdown.test`: 13 differential `CAST(...)` predicates (narrowing, float-to-int, widening, to `VARCHAR`, `TIMESTAMP` -> `DATE`, `DATE` -> `TIMESTAMP`) |
| 1.1.3 integer overflow | `bolt_pushdown.test`: `statement error` ... `Overflow in multiplication` for `u * u` and `id * id * id * id * 1000` |
| 1.1.4 narrow-type COPY FROM | `bolt_types.test` section 4b/4c: `COPY types_src_copy FROM 'gpu_types'` into the original types, out-of-range and `BLOB` destinations |
| 1.1.5 bolt_aggregate names | `test_bolt_aggregate_and_extended_literals`: `'GPU_AGG_SRC'`, `'bolt.gpu_agg_src'`, `'BOLT.Gpu_Agg_Src'`, `'bolt:gpu_agg_src'` |
| 1.1.6 LATERAL on fallback | `test_rfc002_schema_only_bind_and_eager_fallback`: tolerant match arm removed (strict on both backends) |
| 1.1.13 bolt_reset | The implemented policy errors only when the configured device exists but cannot be initialized, so `bolt.test` keeps `statement ok PRAGMA bolt_reset`; the `bolt_reset()` helper in `integration_test.rs` no longer tolerates an error in `cuda-stub` builds |
| W1-E-004 FLOAT literals | `bolt_semantics.test`: `f = 0.1::FLOAT`, `f > 0.3::FLOAT`, `f <= 0.7::FLOAT`, `f != 0.5::FLOAT`, `f IN (...)`, `f = 0.3` differential queries (plus NaN / infinity cases for DOUBLE) |

### Tightened by W3-E (branch `bolt/w3-e`)

Review of `integration_test.rs`, `sqllogictest_and_loadable.rs`, and `test/sql/*.test` for
assertions that pass on any outcome:

| Where | Before | After |
| :--- | :--- | :--- |
| `integration_test.rs` negative tests (SET of a negative `bolt_memory_budget` / `bolt_device`, `MODE 'create'` on an existing table, UBIGINT / HUGEINT overflow in COPY TO, unknown COPY option, COPY FROM column count, append schema mismatch, empty `bolt_query`, unknown `bolt_scan` / `bolt_aggregate` table, empty `bolt_aggregate` arguments) | bare `.is_err()`: an unrelated failure (typo, missing table) also passed | new `assert_err_contains` helper names the expected message; table-function errors are drained with `query_first_chunk` so execution-time errors count too |
| `integration_test.rs` `queries_total >= 0` | always true for a counter the next check already bounds | asserts the metric is exported exactly once |
| `test/sql/bolt.test` section 11 | `statement error` with an empty message block | expects `is not registered in Craton Bolt` and `Invalid Bolt registration mode 'invalid_mode'` |

Left as is, with a reason: the `is_err()` checks for `LATERAL bolt_query(x::VARCHAR)` /
`LATERAL bolt_scan(x::VARCHAR)` and for `COPY ... TO ''` (DuckDB may reject these before the
extension's bind runs, so the message is not ours to pin), the `infer_sql_schema` unit-level
errors, and the GPU-only branches that only print a note on the host fallback (they are the
backend split documented at the top of the file). The SLT runner itself is strict: unknown
directives and `statement ok` failures panic, and labels after the sort mode are ignored as in
DuckDB.

## Reproduction

N/A (bookkeeping).

## Suggested fix

Tighten each "still open" row in the pull request that fixes the corresponding bug; close this page
when the table is empty.

## W4-B (commit 82bdbb3)

- Row "1.1.2 TIMESTAMPTZ storage" tightened. RFC-008 keeps the type, so the `KNOWN-BUG` headers in
  `bolt_types.test` / `bolt_pushdown.test` now point to the new `test/sql/bolt_timestamptz.test`.
  It has a TIMESTAMPTZ column with NULLs and a NULL-free copy, runs under
  `SET TimeZone = 'America/New_York'` and `'Asia/Kolkata'`, and checks the declared type, exact
  value rendering, and differential `=`, `<`, `<=`, `>`, `BETWEEN` and `IN` predicates against
  TIMESTAMPTZ (`+00`, `+05:30`) and TIMESTAMP constants, plus casts to DATE / TIMESTAMP.
- The only row left is W1-O-002: add the reverse `d IS NOT NULL` filtered aggregate once
  craton-bolt 0.10 lands. The page is therefore `blocked-upstream` on W1-O-003.
- The W4-B fuzzer now runs differential checks on every `cargo test`. On `cuda-stub` it still
  compares `bolt_scan`, the replacement scan and `bolt_query` with DuckDB on the host fallback.

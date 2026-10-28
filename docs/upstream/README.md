<!-- SPDX-License-Identifier: Apache-2.0 -->

# Upstream reports (craton-bolt)

Defects in the embedded GPU SQL engine [craton-bolt](https://crates.io/crates/craton-bolt)
found while building and reviewing duckdb-bolt-extension. Each page is self-contained and
written to be pasted into the craton-bolt issue tracker: title, affected versions,
environment, a minimal Rust reproduction that uses only `craton_bolt::Engine` and arrow 53,
expected vs actual results, and the status in the unpublished 0.10.0 source.

- Extension dependency: `craton-bolt = "0.9.0"` (the only published release).
- 0.10.0 status was determined by **reading** the local source (`C:/craton/bolt`, commit
  `248e130`, 2026-10-26) unless a page says "measured". "Unverified" means the source did not
  settle the question.
- Measured environment: NVIDIA GeForce RTX 2060 (6 GiB, sm_75), CUDA 13.3, Windows 11 Pro.
- Extension-side tracking pages are in [`docs/issues/`](../issues/README.md); each page here
  names the page it came from.

## Naming

`craton-bolt-NNN-<slug>.md`, `NNN` a three-digit sequence number that is never reused. When a
report is filed upstream, add its upstream issue URL to the page's header table.

## Index

| Id | Title | Affected | 0.10.0 | Extension page | Severity |
| :--- | :--- | :--- | :--- | :--- | :--- |
| [001](craton-bolt-001-crash-filtered-aggregate-over-nulls.md) | Filtered aggregate over a nullable column with NULLs crashes the process | 0.9.0 | fixed (measured) | W1-O-002 (retired), W1-O-003 | critical |
| [002](craton-bolt-002-gpu-kernels-ignore-null-validity.md) | GPU filters and computed expressions ignore the validity bitmap | 0.9.0 | fixed (by reading) | W2-A-001 | high |
| [003](craton-bolt-003-scalar-sum-min-max-over-empty-input.md) | Ungrouped `SUM` / `MIN` / `MAX` over an empty filtered input return the identity, not NULL | 0.10.0 | present (measured) | W1-O-003 | high |
| [004](craton-bolt-004-decimal128-predicates.md) | Decimal128: wrong `IS NULL`, no literal comparison, filtered plans fail | 0.9.0 | literal/CAST present; rest unverified | W1-A-002 | high |
| [005](craton-bolt-005-timestamp-literal-unit.md) | `TIMESTAMP` literals are nanosecond-only; µs / ms / s / zoned columns cannot be compared | 0.9.0 | present (by reading) | W1-A-006 | medium |
| [006](craton-bolt-006-utf8-is-null-in-and-not-equal.md) | Utf8 `IS [NOT] NULL` and `IN` fail; `!=` matches NULL rows; EXPLAIN accepts unrunnable plans | 0.9.0 | partly fixed; `IS NULL` / `IN` unverified | W1-A-004 (retired) | high |
| [007](craton-bolt-007-deep-predicate-stack-overflow.md) | Long OR chains / dictionary-lowered predicates overflow the stack | 0.9.0 | probably present (by reading) | W2-B-001 (retired) | critical |
| [008](craton-bolt-008-register-batch-quadratic.md) | `register_batch` is O(table) per call; N appends are O(N^2) | 0.9.0 | improved, still O(table) per call | W2-O-001, W3-A-001 (both retired) | high |
| [009](craton-bolt-009-no-public-unregister-table.md) | No public API to unregister one table | 0.9.0 | fixed (`deregister_table`, by reading) | W1-B-006 | low |
| [010](craton-bolt-010-nan-ordering-and-integer-overflow.md) | IEEE NaN comparisons and wrapping integer arithmetic; no SQL-compatible mode | 0.9.0 | unverified | W1-A-003 | medium |

Not reported: BETWEEN inclusivity (W1-A-001) was a DuckDB-side translation bug in the
extension, not an engine defect.

The engine defects found by the wave-4 and wave-5 differential fuzzer (grouped `MIN` / `MAX` /
`SUM` of an all-NULL group, ascending `ORDER BY` sorting NULLs first, `CAST(<negative DECIMAL> AS
DOUBLE)`, `count(*)` next to an `INTEGER` `SUM`, `FLOAT` arithmetic in double precision, and
aggregates over an empty input) are noted in reports 002, 003, 004 and 010. The wave-8 review
found more (integer `/` and `%`, `SUM` accumulators that lose precision or wrap, `STDDEV_POP` over
an empty input, fractional literals typed as `DOUBLE` instead of `DECIMAL`, `TIMESTAMP -
TIMESTAMP` returning raw microsecond ticks); the first three are noted in reports 003 and 010,
the last two are not reported yet. In wave 9 the fuzzer also found that an aggregate below a
sort, limit or distinct drops the validity of its column (`count(v)` counts NULLs; W9-A-003, not
reported yet). The extension routes each of these shapes to the host fallback.

## Cross-cutting request

Publish 0.10.x to crates.io. 001, 002, 009 and part of 006 are fixed only in unpublished
source, so duckdb-bolt-extension (which must build from crates.io) keeps its workarounds:
the NULL crash guard (`gpu_crash_guard`), the 512 MiB worker stack and predicate limits,
caller-side batch coalescing, and drop tombstones.

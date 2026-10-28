<!-- SPDX-License-Identifier: Apache-2.0 -->

# Issue tracker (in-repo)

Open defects, proposals, documentation problems, and missing tests for
`duckdb-bolt-extension` are tracked as one Markdown page per item in this directory.
The pages are reviewed like code: open a page in the same pull request that
discovers the problem, and update its `Status` in the pull request that fixes it.

## File naming

```
<WAVE>-<AGENT>-<NNN>-<slug>.md
```

| Part | Meaning | Example |
| :--- | :--- | :--- |
| `WAVE` | Review / fix wave that filed the item (`W1`, `W2`, ...). Use `GH` for items imported from GitHub issues. | `W1` |
| `AGENT` | Author or work-stream id inside the wave (a letter, or a GitHub handle). | `E` |
| `NNN` | Three-digit sequence number, unique per `WAVE-AGENT`. Never reused. | `007` |
| `slug` | A few lowercase words joined with `-`. | `append-cast-silently-nulls` |

Example: `W1-E-007-append-cast-silently-nulls.md`.

## Page format

Every page starts with this header block, followed by free-form sections.

```markdown
# W1-E-007: Short title

| Field | Value |
| :--- | :--- |
| Title | Short title (same as the heading) |
| Severity | critical \| high \| medium \| low |
| Status | open \| in-progress \| blocked-upstream \| fixed \| wontfix |
| Kind | bug \| proposal \| docs \| test |
| Files | `crates/core/src/runtime.rs:1149-1186`, ... |

## Description
What is wrong and why it matters. Quote the code (with file and line) when it
helps; say whether the problem was verified by running something or by reading.

## Reproduction
SQL / Rust / shell steps, expected result, and actual result. For proposals,
the situation that motivates the change.

## Suggested fix
The smallest change that fixes it, plus the test that should cover it.
```

Severity guide:

- **critical**: data corruption, memory unsafety, or a security boundary bypass that is
  reachable with default settings.
- **high**: wrong query results, a security weakness that needs a non-default setup, or
  a feature that does not work at all on a supported platform.
- **medium**: errors where there should be results (or vice versa), misleading
  diagnostics, performance cliffs, CI or packaging gaps.
- **low**: cosmetic issues, naming, documentation drift, nice-to-have tests.

## Life cycle

| Status | Meaning |
| :--- | :--- |
| `open` | Filed; nobody is working on it. A note in parentheses may say which items are already fixed. |
| `in-progress` | A branch or pull request is addressing it; add its link. |
| `blocked-upstream` | The remaining fix needs a change in a dependency, usually craton-bolt (an engine defect, or an API craton-bolt 0.9.0 lacks). The extension-side mitigation, if any, is in place and described on the page; the engine report lives in [`docs/upstream/`](../upstream/README.md). Revisit when the dependency is upgraded (see [W1-O-003](W1-O-003-upgrade-craton-bolt-0-10.md)). |
| `fixed` | The fix is merged; add the commit hash and the test that covers it. |
| `wontfix` | Closed without a change; the page says why (and names any follow-up page). |

Pages may use either the header table shown above or the equivalent bullet list
(`- **Severity:** ...`). Fixed and `wontfix` pages are retired out of git into
`docs/internal/retired/` by the orchestrator (that directory is ignored by `.gitignore`), and
their row is removed from the index below. Do not delete or renumber pages yourself, and do not
link retired pages from other documents.

Report-level findings that predate this tracker are listed in the internal review
(`docs/internal/CODE_REVIEW_2026-10-26.md`, not in git); pages here reference those
items as "report 1.1.4" and so on instead of duplicating them.

## Index

<!-- Generated from the page headers: one row per page in this directory, fixed pages included until the orchestrator retires them. -->

| Id | Title | Severity | Status | Kind |
| :--- | :--- | :--- | :--- | :--- |
| [W1-A-002](W1-A-002-decimal-gpu-predicates.md) | GPU engine gets `IS NULL` wrong on Decimal128 and cannot compare decimals to literals | high (engine returns wrong results; `bolt_scan` now works around it) | blocked-upstream |  |
| [W1-A-003](W1-A-003-nan-and-overflow-semantics.md) | Engine NaN and integer-overflow semantics differ from DuckDB (bolt_query / bolt_sql) | medium | blocked-upstream |  |
| [W1-A-006](W1-A-006-temporal-pushdown-gaps.md) | Temporal pushdown gaps (TIMESTAMPTZ, TIMESTAMP_S/MS, DATE vs TIMESTAMP) | low (performance; results are correct) | blocked-upstream |  |
| [W1-B-006](W1-B-006-engine-per-table-unregister.md) | craton-bolt has no public way to unregister one table, so drops use empty placeholders | low (performance edge cases; upstream API gap) | blocked-upstream |  |
| [W1-D-007](W1-D-007-ingest-type-coverage.md) | Ingestion type coverage and lossy conversions | low | blocked-upstream (items 2, 3, 4 fixed; item 1 needs Craton Bolt types) |  |
| [W1-E-013](W1-E-013-tests-coupled-to-known-bugs.md) | Tests that must change when known bugs are fixed | low | blocked-upstream | test |
| [W1-O-003](W1-O-003-upgrade-craton-bolt-0-10.md) | Upgrade to craton-bolt 0.10 once it is published | high | blocked-upstream | proposal |
| [W2-A-001](W2-A-001-gpu-ignores-null-validity.md) | craton-bolt 0.9.0 GPU filters and expressions ignore NULL validity | high | blocked-upstream | bug (upstream craton-bolt 0.9.0; mitigated here) |
| [W2-D-003](W2-D-003-ptx-cache-dir-residual-hardening.md) | `bolt_ptx_cache_dir` is not checked on Windows or when the engine uses it | medium | blocked-upstream (items 1, 2, 4 fixed; item 2 engine-build hook in W3-D-004; item 3 wontfix; item 5 needs craton-bolt) | bug |
| [W2-E-005](W2-E-005-bolt-requests-not-interruptible.md) | Bolt work cannot be interrupted or timed out | medium | blocked-upstream | bug |
| [W7-A-001](W7-A-001-parquet-page-header-sizes-unbounded.md) | A Parquet page header can still make `parquet` allocate up to 2 GiB before a short read | low | blocked-upstream | bug |

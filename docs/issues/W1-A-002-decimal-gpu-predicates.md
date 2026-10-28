# W1-A-002: GPU engine gets `IS NULL` wrong on Decimal128 and cannot compare decimals to literals

- **Severity:** high (engine returns wrong results; `bolt_scan` now works around it)
- **Status:** blocked-upstream
- **Files:** craton-bolt engine; `crates/core/src/table_functions.rs` (`translate_expr_to_bolt_sql`, `BoundColumnRef` arm)

## Description

1. `"dec" IS NULL` on a nullable `Decimal128(10,2)` column returns the wrong rows on the GPU:
   2,391 rows instead of 715 in the matrix test. The same predicate on Int32, Utf8, Boolean
   and Timestamp columns is correct.
2. `"dec" >= 2.50` (or any numeric literal) fails with
   `type error: cannot compare Decimal128(10, 2) with Float64`. The engine parses the literal
   as Float64 and has no decimal literal or implicit cast for it.

W1-A no longer pushes any predicate that references a DECIMAL column, so DuckDB evaluates
them. If the engine is fixed, remove the `Decimal128 | Decimal256` check in the
`BoundColumnRef` arm.

## Reproduction

```sql
CREATE TABLE t AS SELECT i::INTEGER AS id,
  CASE WHEN i % 7 = 0 THEN NULL ELSE ((i % 1000) * 1.25)::DECIMAL(10,2) END AS dec
FROM range(0, 5000) t(i);
COPY t TO 'gt' (FORMAT bolt, USE_TMP_FILE FALSE);
SELECT * FROM bolt_query('SELECT count(*) FROM gt WHERE dec IS NULL');  -- expected 715
SELECT * FROM bolt_query('SELECT count(*) FROM gt WHERE dec >= 2.50');  -- type error
```

## Suggested fix

In craton-bolt, check the validity-bitmap handling for Decimal128 in the `IS NULL` kernel. Add
decimal literals (or coerce numeric literals to the column's decimal type) in the SQL frontend.

## W2-B findings (RTX 2060, craton-bolt 0.9.0)

Re-probed: decimal predicates still cannot be pushed, so they stay with DuckDB.
- **Validity:** on 100 rows with 15 NULLs, `dec IS NULL` returned 87 rows and `IS NOT NULL`
  returned 13. The engine reads Decimal128 validity wrongly, so no NULL guard can work on a
  DECIMAL column.
- **Literals and casts:** every way of comparing a decimal fails:
  - `dec >= 2.50` -> `cannot compare Decimal128(10, 2) with Float64`.
  - `dec >= CAST(2.50 AS DECIMAL(10,2))` -> `CAST target type not supported: DECIMAL(10,2)`.
  - `CAST(dec AS DOUBLE) >= 2.5` passes EXPLAIN but fails at run time
    (`scan_kernel: Decimal128 ... CAST ... not lowered into a predicate kernel`).

  The engine cannot evaluate a scaled-integer comparison either.
- **Whole-table plans:** a filtered engine plan that materializes the whole table fails
  whenever the table *contains* a Decimal128 column
  (`PhysicalPlan::Filter: unsupported Arrow dtype Decimal128`). Examples: projecting a Utf8
  column, or a Utf8 `IN`. `bolt_scan` then falls back to filter-take, which only uploads the
  referenced columns.

## W3-B notes (commit 71e05e7)

No change possible in `table_functions.rs`: the engine-side defects (Decimal128 validity,
no decimal literal or cast) are unchanged in craton-bolt 0.9.0, so DECIMAL columns are still
never pushed. DECIMAL *constants* against integer columns are now pushed exactly (W2-E-003).
Stays open until craton-bolt is fixed (W1-O-003).

Upstream report: [craton-bolt-004](../upstream/craton-bolt-004-decimal128-predicates.md) (W4-D).
## W4-B re-check (RTX 2060, craton-bolt 0.9.0): blocked-upstream

Engine behavior is unchanged (measured with `bolt_explain` on a NULL-free table):
- `dec >= 2.50` -> `type error: cannot compare Decimal128(10, 2) with Float64`;
- `dec >= CAST(2.50 AS DECIMAL(10,2))` -> `SQL parse error: CAST target type not supported: DECIMAL(10,2)`.

`bolt_scan` keeps every DECIMAL-column predicate with DuckDB, so results are correct. The W4-B
differential fuzzer checks DECIMAL(9,2) / DECIMAL(18,4) columns with and without NULLs on every
run. Nothing is left in `table_functions.rs`; this waits for craton-bolt (W1-O-003). The fuzzer
found a related engine defect: `CAST(<negative decimal> AS DOUBLE)` in projections is wrong
(W4-B-004).

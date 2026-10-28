<!-- SPDX-License-Identifier: Apache-2.0 -->

# craton-bolt-004: Decimal128 predicates: wrong `IS NULL`, no way to compare with a literal, and filtered plans fail on tables that contain a Decimal128 column

| Field | Value |
| :--- | :--- |
| Affected | craton-bolt 0.9.0 (crates.io); 0.10.0 source (`248e130`) for items 2 and 3 by reading |
| Fixed in | not fixed |
| Severity | high (item 1: wrong results); medium (items 2, 3: errors) |
| Environment | NVIDIA GeForce RTX 2060, CUDA 13.3, Windows 11 Pro, Rust stable |
| Found by | duckdb-bolt-extension, internal page W1-A-002 |

## Summary

1. `dec IS NULL` / `dec IS NOT NULL` on a nullable `Decimal128(10, 2)` column return the wrong
   rows. The same predicates on Int32, Utf8, Boolean and Timestamp columns are correct.
2. A Decimal128 column cannot be compared with any constant:
   - `dec >= 2.50` fails with `type error: cannot compare Decimal128(10, 2) with Float64` (the
     SQL frontend has no decimal literal; `2.50` becomes Float64);
   - `dec >= CAST(2.50 AS DECIMAL(10,2))` fails with
     `CAST target type not supported: DECIMAL(10,2)`;
   - `CAST(dec AS DOUBLE) >= 2.5` passes `explain_sql` but fails in `sql` with
     `scan_kernel: Decimal128 ... CAST ... not lowered into a predicate kernel`.
3. A filtered plan that materializes the whole table fails when the table merely *contains* a
   Decimal128 column, even if the query does not reference it:
   `PhysicalPlan::Filter: unsupported Arrow dtype Decimal128` (for example projecting a Utf8
   column under a filter, or a Utf8 `IN`).

## Minimal reproduction (Rust, `craton_bolt::Engine` + arrow 53 only)

```toml
[dependencies]
craton-bolt = "0.9.0"
arrow = "53"
```

```rust
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Decimal128Array, Int32Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use craton_bolt::Engine;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 100 rows: id = i, dec = (i * 1.25) as DECIMAL(10,2), NULL when i % 7 == 0 (15 NULLs),
    // s = 'v' || (i % 6).
    let n = 100;
    let id = Int32Array::from((0..n).collect::<Vec<i32>>());
    let dec = Decimal128Array::from(
        (0..n)
            .map(|i| (i % 7 != 0).then_some(i as i128 * 125))
            .collect::<Vec<_>>(),
    )
    .with_precision_and_scale(10, 2)?;
    let s = StringArray::from((0..n).map(|i| format!("v{}", i % 6)).collect::<Vec<_>>());
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("dec", DataType::Decimal128(10, 2), true),
        Field::new("s", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(id) as ArrayRef, Arc::new(dec), Arc::new(s)],
    )?;

    let mut engine = Engine::new()?;
    engine.register_table("t", batch)?;

    for q in [
        "SELECT id FROM t WHERE dec IS NULL",                          // expected 15 rows
        "SELECT id FROM t WHERE dec IS NOT NULL",                      // expected 85 rows
        "SELECT id FROM t WHERE dec >= 2.50",                          // expected 84 rows
        "SELECT id FROM t WHERE dec >= CAST(2.50 AS DECIMAL(10,2))",   // expected 84 rows
        "SELECT id FROM t WHERE CAST(dec AS DOUBLE) >= 2.5",           // expected 84 rows
        "SELECT s FROM t WHERE id > 50",                               // expected 49 rows
    ] {
        match engine.sql(q) {
            Ok(out) => println!("{q}\n  rows = {}", out.record_batch().column(0).len()),
            Err(e) => println!("{q}\n  error: {e}"),
        }
    }
    Ok(())
}
```

(Rows with `i >= 2` and `i % 7 != 0` satisfy `dec >= 2.50`: 98 rows minus the 14 NULLs at 7, 14, ..., 98 = 84.)

## Actual (0.9.0, measured on the extension's 100-row probe with 15 NULLs)

- `dec IS NULL` returned 87 rows and `dec IS NOT NULL` 13 rows (expected 15 and 85); on a
  5,000-row table with 715 NULLs, `IS NULL` returned 2,391 rows.
- The three comparisons fail with the errors listed in the summary.
- `SELECT s FROM t WHERE id > 50` fails with `PhysicalPlan::Filter: unsupported Arrow dtype
  Decimal128`; dropping the `dec` column from the table makes it succeed.

## Status in 0.10.0

- Item 2 **still present by reading**: `DataType::is_numeric` excludes `Decimal128`
  (`src/plan/logical_plan.rs`, "Not part of `is_numeric()` yet"), so a comparison with a
  Float64 literal still hits `cannot compare {l:?} with {r:?}`; the SQL frontend never produces
  `Literal::Decimal128` from a numeric literal; and `lower_cast_data_type`
  (`src/plan/sql_frontend.rs`) still accepts only INT / BIGINT / REAL / DOUBLE / BOOLEAN.
- Items 1 and 3: **unverified** (0.10's `plan::null_semantics` may change item 1).

## Suggested fix

- Read the Decimal128 validity bitmap in the `IS [NOT] NULL` kernel (the offset/stride for
  16-byte values looks wrong).
- Coerce numeric literals compared with a `Decimal128(p, s)` column to that type (exact when
  the literal has at most `s` fractional digits; otherwise compare in a wider scale), and
  accept `CAST(... AS DECIMAL(p, s))`.
- Skip unreferenced columns when materializing a filter input, or lower Decimal128 as an
  opaque 16-byte passthrough.

Mitigated in duckdb-bolt-extension (partly) by `plan_type_hazard`
(`crates/core/src/runtime/engine_support.rs`, W5-A, commit `ea8d17e`): every CAST from or to
DECIMAL and every arithmetic operator with a DECIMAL operand runs on the host fallback, because
0.9.0 also converts negative Decimal128 values to DOUBLE wrongly (`CAST(-0.25 AS DOUBLE)` ->
`-0.2048`, W4-B-004). The predicate items above are handled by the scan pushdown rules (W1-A-002).

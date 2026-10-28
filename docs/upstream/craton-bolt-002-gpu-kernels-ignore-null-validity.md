<!-- SPDX-License-Identifier: Apache-2.0 -->

# craton-bolt-002: GPU filters and computed expressions ignore the Arrow validity bitmap (NULLs read as stored values)

| Field | Value |
| :--- | :--- |
| Affected | craton-bolt 0.9.0 (crates.io) |
| Fixed in | 0.10.0 source (`248e130`), per its CHANGELOG ("GPU kernels ignored SQL NULLs of nullable primitive columns", `plan::null_semantics`); not yet re-measured by us |
| Severity | high (silently wrong query results) |
| Environment | NVIDIA GeForce RTX 2060, CUDA 13.3, Windows 11 Pro, Rust stable |
| Found by | duckdb-bolt-extension, internal page W2-A-001 |

## Summary

The fused filter / projection kernels read the values buffer of nullable primitive columns and
never consult the validity bitmap (except in `IS [NOT] NULL`). A NULL slot is evaluated as
whatever bytes the values buffer holds. Arrays built with `PrimitiveArray::from(Vec<Option<_>>)`
store `0` under NULLs, which hides the bug for many predicates; arrays exported by other
systems (DuckDB, Arrow C Data Interface producers) keep arbitrary values under NULLs, and then
NULL rows pass comparisons, computed outputs lose their NULLs, and `CASE` is not Kleene.

## Minimal reproduction (Rust, `craton_bolt::Engine` + arrow 53 only)

```toml
[dependencies]
craton-bolt = "0.9.0"
arrow = "53"
```

```rust
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Float64Array, Int32Array};
use arrow::buffer::{NullBuffer, ScalarBuffer};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use craton_bolt::Engine;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 100 rows: id = i, d = i - 20, NULL when i % 7 == 0 (15 NULLs).
    // The values buffer keeps `i - 20` under the NULL slots (valid per the Arrow spec).
    let n = 100;
    let id = Int32Array::from((0..n).collect::<Vec<i32>>());
    let values: Vec<f64> = (0..n).map(|i| (i - 20) as f64).collect();
    let validity: Vec<bool> = (0..n).map(|i| i % 7 != 0).collect();
    let d = Float64Array::new(ScalarBuffer::from(values), Some(NullBuffer::from(validity)));
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("d", DataType::Float64, true),
    ]));
    let batch = RecordBatch::try_new(schema, vec![Arc::new(id) as ArrayRef, Arc::new(d)])?;

    let mut engine = Engine::new()?;
    engine.register_table("t", batch)?;

    for q in [
        "SELECT d FROM t WHERE d > 0",
        "SELECT d FROM t WHERE id > 20",
        "SELECT d FROM t WHERE d IS NULL",
        "SELECT d + 1 FROM t",
        "SELECT CASE WHEN d > 0 THEN 1 ELSE 0 END FROM t",
    ] {
        let out = engine.sql(q)?;
        let col = out.record_batch().column(0);
        println!("{q}\n  rows = {}, nulls = {}", col.len(), col.null_count());
    }
    // Do not add an aggregate over a filter here: on 0.9.0 that crashes the process
    // (craton-bolt-001).
    Ok(())
}
```

## Expected vs actual (0.9.0, measured)

| Query | Expected | 0.9.0 |
| :--- | :--- | :--- |
| `SELECT d FROM t WHERE d > 0` | 67 rows | 79 rows (the 12 NULL rows with `i > 20` pass) |
| `SELECT d FROM t WHERE id > 20` | 79 rows, 12 NULLs | 79 rows, **0 NULLs** (stored values returned) |
| `SELECT d FROM t WHERE d IS NULL` | 15 rows, 15 NULLs | 15 rows, **0 NULLs** (stored values returned) |
| `SELECT d + 1 FROM t` | 100 rows, 15 NULLs | 100 rows, **0 NULLs** |
| `SELECT CASE WHEN d > 0 THEN 1 ELSE 0 END FROM t` | 67 ones | 79 ones |

Correct in 0.9.0: plain projections, `ORDER BY` a column, `LIMIT`, `DISTINCT`, `UNION ALL`,
equi joins, and `COUNT` / `SUM` / `MIN` / `MAX` / `AVG` of plain columns with or without
`GROUP BY` (no filter).

The engine's own tests do not catch this because their fixtures are built from `Option` values,
which zero the NULL slots.

## Status in 0.10.0

**Fixed by reading** (not re-run): `src/plan/null_semantics.rs` rewrites every predicate over
nullable columns into "is TRUE" form and gives each projected output a hidden validity output
(`__bolt_valid_<j>`), and the CHANGELOG lists the fix. Please keep a regression test that
builds the input with non-zero values under NULL slots, as above; the `Option`-built fixtures
cannot detect a regression.

Open question for 0.10: whether the rewrite also covers `Decimal128`, `Date32` and `Timestamp`
columns (the module comment says "nullable primitive columns"); see
[craton-bolt-004](craton-bolt-004-decimal128-predicates.md) for the Decimal128 `IS NULL` case.

Mitigated in duckdb-bolt-extension by the GPU crash guard (`plan_is_null_safe`): filters and
computed expressions over tables with NULLs run on the host fallback (W2-A-001). W5-A (commit
`ea8d17e`) adds two shapes the guard used to treat as safe: grouped SUM / MIN / MAX of an all-NULL
group (W4-B-001) and ascending sort keys over NULLs, which the engine orders NULLS FIRST
(W4-B-003).

Mitigated in duckdb-bolt-extension by W9-A as well: the guard now follows each column through the
plan to the stored table it comes from (`plan_facts.rs`), so a nullable column renamed by an alias,
a UNION ALL branch or a NULL literal no longer reaches the grouped SUM / MIN / MAX or default-ordered
sort on the GPU (W8-B-004).

Mitigated in duckdb-bolt-extension by W11-A as well: an ungrouped `count(*)` next to an aggregate
of a column with NULLs runs on the host fallback. The engine then ignores the column's NULLs
(`count(c) = count(*)`, and `min` / `max` / `sum` of an all-NULL column return the kernel
identity; W11-A-001).

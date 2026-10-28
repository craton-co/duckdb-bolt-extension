<!-- SPDX-License-Identifier: Apache-2.0 -->

# craton-bolt-003: Ungrouped `SUM` / `MIN` / `MAX` over an empty filtered input return the reduction identity instead of NULL

| Field | Value |
| :--- | :--- |
| Affected | craton-bolt 0.10.0 source (`248e130`, unpublished). On 0.9.0 the same queries crash instead (craton-bolt-001). |
| Fixed in | not fixed |
| Severity | high (silently wrong results: `0` where SQL requires `NULL`) |
| Environment | NVIDIA GeForce RTX 2060, CUDA 13.3, Windows 11 Pro, Rust stable |
| Found by | duckdb-bolt-extension, internal page W1-O-003 |

## Summary

SQL defines `SUM`, `MIN`, `MAX` (and `AVG`, `STDDEV_*`, `VAR_*`) over zero non-NULL input values
as `NULL`. 0.10.0 fixed this for the plain scalar path (`exec/aggregate.rs`, the
`non_null_count_for_input(...) == 0` check) and for grouped aggregates (`null_empty_groups`),
but not for the **aggregate pre-stage** path that runs when the aggregate input is filtered or
computed (`exec/agg_with_pre.rs`). There, `build_one_aggregate` calls `reduce_host_col` on the
NULL-stripped, filtered column, and `reduce_gpu_vec` returns `T::identity_scalar` for zero rows
(`agg_with_pre.rs`, `if n_rows == 0 { return T::identity_scalar(op, dtype); }`), which is then
packed as a non-NULL value.

## Minimal reproduction (Rust, `craton_bolt::Engine` + arrow 53 only)

```toml
[dependencies]
craton-bolt = { path = "../bolt" }   # 0.10.0 source; 0.9.0 crashes on these queries
arrow = "53"
```

```rust
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Float64Array, Int32Array, Int64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use craton_bolt::Engine;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 100 rows: id = i, d = i as f64 / v = i as i64, both NULL when i % 7 == 0.
    let n = 100;
    let id = Int32Array::from((0..n).collect::<Vec<i32>>());
    let d = Float64Array::from((0..n).map(|i| (i % 7 != 0).then_some(i as f64)).collect::<Vec<_>>());
    let v = Int64Array::from((0..n).map(|i| (i % 7 != 0).then_some(i as i64)).collect::<Vec<_>>());
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("d", DataType::Float64, true),
        Field::new("v", DataType::Int64, true),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(id) as ArrayRef, Arc::new(d), Arc::new(v)],
    )?;

    let mut engine = Engine::new()?;
    engine.register_table("t", batch)?;

    for q in [
        "SELECT SUM(d) FROM t WHERE d IS NULL", // all-NULL input after the filter
        "SELECT SUM(v) FROM t WHERE id < 0",    // empty input after the filter
        "SELECT MIN(v) FROM t WHERE id < 0",
        "SELECT MAX(d) FROM t WHERE id < 0",
        "SELECT STDDEV_POP(d) FROM t WHERE id < 0",
        "SELECT SUM(d) FROM t WHERE id < 0 AND d > 0",
    ] {
        let out = engine.sql(q)?;
        let col = out.record_batch().column(0);
        println!("{q}\n  rows = {}, null = {}, value = {:?}", col.len(), col.is_null(0), col);
    }
    Ok(())
}
```

## Expected

Each query returns one row whose value is `NULL` (`col.is_null(0) == true`).

## Actual (0.10.0)

- `SELECT SUM(d) FROM t WHERE d IS NULL` returns `0` (measured on the RTX 2060).
- By reading `agg_with_pre.rs` (not run): the other `SUM` queries return `0`, `MIN(v)` returns
  `i64::MAX`, `MAX(d)` returns `-inf`, and `STDDEV_POP(d)` returns `0.0`
  (`stddev_to_array_with_pre`, "Empty / all-NULL input -> 0.0").

## Suggested fix

In `build_one_aggregate`, mirror the scalar path: when `out_field.nullable` and the resolved
input's `non_null_count() == 0`, return `null_scalar_array(out_field.dtype)` for
`SUM` / `MIN` / `MAX`; return NULL for `STDDEV_*` / `VAR_*` on empty input (and `VAR_SAMP` /
`STDDEV_SAMP` on one value). Add the queries above to the DuckDB differential lane.

Mitigated in duckdb-bolt-extension by the GPU crash guard: aggregates over a filtered input with
NULLs run on the host fallback (W2-A-001), and since W5-A (commit `ea8d17e`) so do grouped SUM /
MIN / MAX over a column with NULLs, the grouped form of this bug (an all-NULL group returns `0`,
`i32::MAX`, `-inf` or `NaN`; W4-B-001).

Mitigated in duckdb-bolt-extension by W9-A as well: an ungrouped `STDDEV_POP` over an input that may
be empty runs on the host fallback (the engine returns `0.0`, W8-B-006; `VAR_POP`, `VAR_SAMP` and
`STDDEV_SAMP` were measured NULL), the "may be empty" test is a structural walk of the plan (W8-B-005),
and host-orchestrated shapes such as `LATERAL (SELECT max(..) ... WHERE <no match>)`, which returned
`i64::MIN`, are no longer sent to the engine (W8-B-003).

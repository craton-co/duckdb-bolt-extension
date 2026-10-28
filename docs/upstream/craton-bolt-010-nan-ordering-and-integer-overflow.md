<!-- SPDX-License-Identifier: Apache-2.0 -->

# craton-bolt-010: Float comparisons use IEEE NaN semantics and integer arithmetic wraps; no SQL-compatible mode

| Field | Value |
| :--- | :--- |
| Affected | craton-bolt 0.9.0 (crates.io); 0.10.0 source (`248e130`): unverified (no CHANGELOG entry) |
| Fixed in | not fixed |
| Severity | medium (results differ from DuckDB / PostgreSQL; documented behavior would be acceptable) |
| Environment | NVIDIA GeForce RTX 2060, CUDA 13.3, Windows 11 Pro, Rust stable |
| Found by | duckdb-bolt-extension, internal page W1-A-003 |

## Summary

- DuckDB and PostgreSQL order NaN above every other value and treat `NaN = NaN` as TRUE. The
  engine uses IEEE comparisons: every comparison with NaN is FALSE, and `!=` is an *ordered*
  not-equal, so `NaN != 3` is FALSE too. `NOT (d > 3)` and `d > 3` therefore do not partition
  the non-NULL rows.
- Integer `+`, `-`, `*` wrap on overflow on the GPU; DuckDB and PostgreSQL raise an
  out-of-range error.

This is a semantic choice, but it is not documented, and embedders that translate SQL from
another engine get silently different rows.

## Minimal reproduction (Rust, `craton_bolt::Engine` + arrow 53 only)

```toml
[dependencies]
craton-bolt = "0.9.0"
arrow = "53"
```

```rust
use std::sync::Arc;

use arrow::array::{ArrayRef, Float64Array, Int32Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use craton_bolt::Engine;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // d = NaN when i % 4 == 0, else i + 0.4, for i in 0..10 (3 NaNs: i = 0, 4, 8).
    let d = Float64Array::from(
        (0..10).map(|i| if i % 4 == 0 { f64::NAN } else { i as f64 + 0.4 }).collect::<Vec<_>>(),
    );
    let big = Int32Array::from(vec![i32::MAX; 10]);
    let schema = Arc::new(Schema::new(vec![
        Field::new("d", DataType::Float64, false),
        Field::new("big", DataType::Int32, false),
    ]));
    let batch = RecordBatch::try_new(schema, vec![Arc::new(d) as ArrayRef, Arc::new(big)])?;

    let mut engine = Engine::new()?;
    engine.register_table("p", batch)?;
    for q in [
        "SELECT d FROM p WHERE d > 3",        // DuckDB: 8 rows (5 values > 3 plus 3 NaN)
        "SELECT d FROM p WHERE NOT (d > 3)",  // DuckDB: 2 rows
        "SELECT d FROM p WHERE d = d",        // DuckDB: 10 rows
        "SELECT d FROM p WHERE d != 3.4",     // DuckDB: 9 rows
        "SELECT big + 1 FROM p",              // DuckDB: Out of Range error
    ] {
        match engine.sql(q) {
            Ok(out) => println!("{q}\n  rows = {} {:?}", out.record_batch().num_rows(),
                                out.record_batch().column(0)),
            Err(e) => println!("{q}\n  error: {e}"),
        }
    }
    Ok(())
}
```

## Actual (0.9.0, measured)

| Query | DuckDB | craton-bolt 0.9.0 |
| :--- | ---: | ---: |
| `d > 3` | 8 | 5 |
| `NOT (d > 3)` | 2 | 5 |
| `d = d` | 10 | 7 |
| `d != 3.4` | 9 | 6 |
| `big + 1` | error | `-2147483648` (wrapped) |

(The `d != 3.4` row follows from the ordered not-equal measured as `d != d` = 0 rows and
`NaN != 3` = FALSE; the other rows were measured directly.)

## Status in 0.10.0

**Unverified**. The 0.10.0 CHANGELOG has no entry for NaN ordering or checked integer
arithmetic, and `src/jit/ptx_gen.rs` still comments "NaN compares false" for its `setp`
comparisons.

## Suggested fix

Either document IEEE / wrapping semantics in `SQL_REFERENCE.md` / `LIMITATIONS.md`, or add an
`EngineBuilder` option for SQL-compatible semantics: total-order float comparisons (NaN
greatest, `NaN = NaN`), unordered `!=`, and checked integer arithmetic that returns an error
(or NULL, configurable) on overflow.

Mitigated in duckdb-bolt-extension (partly) since W5-A (commit `ea8d17e`): arithmetic on integer
columns stored widened (TINYINT, SMALLINT, unsigned, HUGEINT) runs on the host fallback with the
original DuckDB types, so their overflow raises an error (W4-B-002). INTEGER / BIGINT wrap-around
and NaN ordering on the engine are unchanged (documented, W1-A-003).

Mitigated in duckdb-bolt-extension by W9-A for SUM: an integer `SUM` / `AVG` whose bound (largest
absolute value times row count) may exceed `i64::MAX`, and every `SUM` of a FLOAT (accumulated in
single precision), runs on the host fallback (W8-B-007). Integer `/` and `%` (truncating, NULL for a
zero divisor) run there too (W8-B-001).

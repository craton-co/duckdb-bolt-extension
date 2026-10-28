<!-- SPDX-License-Identifier: Apache-2.0 -->

# craton-bolt-005: `TIMESTAMP '...'` literals are nanosecond-only and cannot be compared with microsecond, millisecond, second or zoned timestamp columns

| Field | Value |
| :--- | :--- |
| Affected | craton-bolt 0.9.0 (crates.io); 0.10.0 source (`248e130`) by reading |
| Fixed in | not fixed |
| Severity | medium (queries error; the most common Arrow timestamp unit, microseconds, cannot be filtered by a constant) |
| Environment | NVIDIA GeForce RTX 2060, CUDA 13.3, Windows 11 Pro, Rust stable |
| Found by | duckdb-bolt-extension, internal page W1-A-006 |

## Summary

The SQL frontend parses `TIMESTAMP 'YYYY-MM-DD HH:MM:SS[.f]'` into
`Literal::Timestamp(ticks, TimeUnit::Nanosecond, None)`, and the type checker allows a
comparison only when both sides have exactly the same `DataType` (or both are numeric). So
every comparison between a timestamp literal and a column of type
`Timestamp(Microsecond | Millisecond | Second, _)` or `Timestamp(_, Some(tz))` fails with

```
type error: cannot compare Timestamp(Microsecond, None) with Timestamp(Nanosecond, None)
```

`Timestamp(Microsecond, _)` is what DuckDB, Parquet (`TIMESTAMP_MICROS`), Spark and pandas
(`datetime64[us]` since pandas 2) produce. There is also no `TIMESTAMPTZ '...'` literal
(parse error) and `CAST(... AS TIMESTAMP)` / `AS TIMESTAMPTZ` is not supported, so there is no
spelling that works. `DATE '...'` against `Date32` works.

## Minimal reproduction (Rust, `craton_bolt::Engine` + arrow 53 only)

```toml
[dependencies]
craton-bolt = "0.9.0"
arrow = "53"
```

```rust
use std::sync::Arc;

use arrow::array::{
    ArrayRef, Int32Array, TimestampMicrosecondArray, TimestampMillisecondArray,
    TimestampNanosecondArray,
};
use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use arrow::record_batch::RecordBatch;
use craton_bolt::Engine;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 10 rows, one per day from 2024-01-01 00:00:00 UTC.
    let n = 10i64;
    let day_us = 86_400_000_000i64;
    let base_us = 1_704_067_200_000_000i64; // 2024-01-01
    let us: Vec<i64> = (0..n).map(|i| base_us + i * day_us).collect();
    let id = Int32Array::from((0..n as i32).collect::<Vec<_>>());
    let ts_ns = TimestampNanosecondArray::from(us.iter().map(|v| v * 1000).collect::<Vec<_>>());
    let ts_us = TimestampMicrosecondArray::from(us.clone());
    let ts_ms = TimestampMillisecondArray::from(us.iter().map(|v| v / 1000).collect::<Vec<_>>());
    let ts_tz = TimestampMicrosecondArray::from(us.clone()).with_timezone("UTC");
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("ts_ns", DataType::Timestamp(TimeUnit::Nanosecond, None), false),
        Field::new("ts_us", DataType::Timestamp(TimeUnit::Microsecond, None), false),
        Field::new("ts_ms", DataType::Timestamp(TimeUnit::Millisecond, None), false),
        Field::new("ts_tz", DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())), false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(id) as ArrayRef,
            Arc::new(ts_ns),
            Arc::new(ts_us),
            Arc::new(ts_ms),
            Arc::new(ts_tz),
        ],
    )?;

    let mut engine = Engine::new()?;
    engine.register_table("t", batch)?;

    for col in ["ts_ns", "ts_us", "ts_ms", "ts_tz"] {
        let q = format!("SELECT id FROM t WHERE {col} >= TIMESTAMP '2024-01-05 00:00:00'");
        match engine.sql(&q) {
            Ok(out) => println!("{q}\n  rows = {}", out.record_batch().num_rows()),
            Err(e) => println!("{q}\n  error: {e}"),
        }
    }
    Ok(())
}
```

## Expected

All four queries return 6 rows (ids 4..9). For `ts_tz`, a naive literal compared with a UTC
column should be interpreted in UTC (or the engine should offer a `TIMESTAMPTZ` literal).

## Actual (0.9.0, measured)

Only `ts_ns` works. The other three fail at planning time with
`cannot compare Timestamp(<unit>, <tz>) with Timestamp(Nanosecond, None)` (both in
`explain_sql` and `sql`).

## Status in 0.10.0

**Still present by reading**: `parse_timestamp_literal` (`src/plan/sql_frontend.rs`) still
returns `Literal::Timestamp(ticks, TimeUnit::Nanosecond, None)`; `Expr::dtype_depth`
(`src/plan/logical_plan.rs`) still requires `l == r` for non-numeric comparisons; and
`lower_cast_data_type` has no TIMESTAMP target.

## Suggested fix

When a timestamp literal meets a timestamp column, rescale the literal to the column's unit
(exactly when the literal has no sub-unit digits; otherwise round toward the correct side for
`<` / `<=` / `>` / `>=` and fold `=` to FALSE), and adopt the column's time zone for zoned
columns. Add `TIMESTAMPTZ '... +hh[:mm]'` and `CAST(... AS TIMESTAMP[TZ])`.

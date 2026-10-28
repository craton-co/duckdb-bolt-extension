<!-- SPDX-License-Identifier: Apache-2.0 -->

# craton-bolt-001: Filtered aggregate over a nullable column with NULLs crashes the process (CUDA 700 / access violation)

| Field | Value |
| :--- | :--- |
| Affected | craton-bolt 0.9.0 (crates.io) |
| Fixed in | 0.10.0 source (`248e130`): measured not to crash; see "Status in 0.10.0" |
| Severity | critical (the host process dies; no error is returned) |
| Environment | NVIDIA GeForce RTX 2060 (6 GiB, sm_75), CUDA 13.3, Windows 11 Pro, Rust stable, debug and release |
| Found by | duckdb-bolt-extension, internal pages W1-O-002 (retired), W2-A-001, W1-O-003 |

## Summary

`Engine::sql` on an aggregate whose input is filtered, over a table with a nullable column that
holds at least one NULL, fails inside the aggregate pre-stage with
`CUDA driver error 700: an illegal memory access` or `CudaWithCode { code: 1, "invalid argument" }`
and then terminates the process with `STATUS_ACCESS_VIOLATION`. The same table without NULLs,
and the same aggregates without a `WHERE`, are fine.

## Minimal reproduction (Rust, `craton_bolt::Engine` + arrow 53 only)

```toml
[dependencies]
craton-bolt = "0.9.0"
arrow = "53"
```

```rust
use std::sync::Arc;

use arrow::array::{ArrayRef, Float64Array, Int32Array};
use arrow::buffer::{NullBuffer, ScalarBuffer};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use craton_bolt::Engine;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 100 rows: id = i, d = i as f64, NULL when i % 7 == 0 (15 NULLs).
    let n = 100;
    let id = Int32Array::from((0..n).collect::<Vec<i32>>());
    let values: Vec<f64> = (0..n).map(|i| i as f64).collect();
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
        "SELECT COUNT(*) FROM t",                          // ok: 100
        "SELECT SUM(d) FROM t",                            // ok: 4215
        "SELECT COUNT(*) FROM t WHERE d > 0",              // crash (expected 85)
        "SELECT COUNT(*) FROM t WHERE d IS NOT NULL AND d > 0", // crash (expected 85)
        "SELECT SUM(d) FROM t WHERE d IS NULL",            // crash (expected NULL)
        "SELECT SUM(d) FROM t WHERE id > 5",               // crash (expected 4200)
        "SELECT COUNT(*) FROM (SELECT d FROM t WHERE d > 0) s", // crash (expected 85)
    ] {
        eprintln!("running: {q}");
        match engine.sql(q) {
            Ok(out) => eprintln!("  -> {:?}", out.record_batch().column(0)),
            Err(e) => eprintln!("  -> error: {e}"),
        }
    }
    Ok(())
}
```

## Expected

Every query returns one row, in order `100`, `4215`, `85`, `85`, `NULL`, `4200`, `85` (DuckDB /
PostgreSQL semantics), or at worst an `Err`; the process must never die.

## Actual (0.9.0)

The first two queries succeed. The first filtered aggregate either returns
`Err(Cuda(... 700 illegal memory access ...))` / `CudaWithCode { code: 1, "invalid argument" }`
and the next CUDA call aborts the process, or the process aborts directly with
`STATUS_ACCESS_VIOLATION` (exit code `0xc0000005`). Reproduced with `d` as Float64, Float32,
Int32 and Int64; also for `... WHERE d > 0 GROUP BY id`. A filter on a NULL-free column with an
aggregate over NULL-free columns does not crash.

## Likely cause

The fused aggregate pre-stage predicate kernel is compiled for validity parameters it is not
given (garbage validity pointers when the input holds NULLs). 0.10.0's CHANGELOG lists this
fix: "Aggregates filtered by `IS [NOT] NULL` read garbage validity pointers in the fused
pre-stage predicate kernel".

## Status in 0.10.0

**Fixed** (measured): against the 0.10.0 source on the same machine, `SELECT COUNT(*) FROM t
WHERE d > 0` returns `85` and nothing crashes. `SELECT SUM(d) FROM t WHERE d IS NULL` no longer
crashes but returns `0` instead of `NULL`; that is reported separately as
[craton-bolt-003](craton-bolt-003-scalar-sum-min-max-over-empty-input.md).

## Request

Publish 0.10.x to crates.io (0.9.0 is the only published release with this crash), and add a
regression test with NULL-bearing columns under every pre-stage predicate shape listed above.

Mitigated in duckdb-bolt-extension by the GPU crash guard (`gpu_crash_guard` /
`plan_is_null_safe`, `crates/core/src/runtime.rs`): any `Filter` over a table with NULLs runs on
the host fallback, so the crashing shapes never reach the engine (W2-A-001; column-level checks
added in W5-A, commit `ea8d17e`).

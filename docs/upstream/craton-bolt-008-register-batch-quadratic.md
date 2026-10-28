<!-- SPDX-License-Identifier: Apache-2.0 -->

# craton-bolt-008: `Engine::register_batch` costs O(table size) per call (about 0.6 s per batch on a 300k-row table), so appending N batches is O(N^2)

| Field | Value |
| :--- | :--- |
| Affected | craton-bolt 0.9.0 (crates.io); 0.10.0 source (`248e130`): still O(table) per call by reading, constant factor unverified |
| Fixed in | not fixed |
| Severity | high (performance: a 300,000-row COPY took 91 s on GPU vs 43 ms on the CPU stub) |
| Environment | NVIDIA GeForce RTX 2060, CUDA 13.3, Windows 11 Pro, Rust stable (debug build) |
| Found by | duckdb-bolt-extension, internal page W2-O-001 (retired; mitigated by concatenating batches before upload) |

## Summary

Registering a table as `register_table(first)` followed by one `register_batch` per further
batch is the natural way to stream a table into the engine (Arrow producers emit
2,048- to 65,536-row batches). In 0.9.0 every `register_batch` call:

1. concatenates **all** batches of the table (`materialize_table`),
2. unregisters and rebuilds the Utf8 dictionary registry from the concatenated batch,
3. re-derives the provider schema and per-column nullability from it.

So each call costs O(rows already in the table), and N appends cost O(N^2). The cost is on the
host (dictionary build over every Utf8 column), not in the HtoD copy, and shows up even
without a query between appends.

## Minimal reproduction (Rust, `craton_bolt::Engine` + arrow 53 only)

```toml
[dependencies]
craton-bolt = "0.9.0"
arrow = "53"
```

```rust
use std::sync::Arc;
use std::time::Instant;

use arrow::array::{ArrayRef, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use craton_bolt::Engine;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("s", DataType::Utf8, false),
    ]));
    let rows_per_batch = 2_048i64;
    let batches = 147i64; // ~300,000 rows
    let make = |b: i64| -> RecordBatch {
        let start = b * rows_per_batch;
        let id = Int64Array::from((start..start + rows_per_batch).collect::<Vec<_>>());
        let s = StringArray::from(
            (start..start + rows_per_batch).map(|i| format!("s{}", i % 1000)).collect::<Vec<_>>(),
        );
        RecordBatch::try_new(schema.clone(), vec![Arc::new(id) as ArrayRef, Arc::new(s)]).unwrap()
    };

    let mut engine = Engine::new()?;
    let t0 = Instant::now();
    engine.register_table("t", make(0))?;
    for b in 1..batches {
        let t = Instant::now();
        engine.register_batch("t", make(b))?;
        if b % 25 == 0 {
            println!("batch {b}: {:?} for this call, {:?} total", t.elapsed(), t0.elapsed());
        }
    }
    let out = engine.sql("SELECT COUNT(*) FROM t")?;
    println!("count = {:?}, total {:?}", out.record_batch().column(0), t0.elapsed());
    Ok(())
}
```

## Expected

Appending a 2,048-row batch costs time proportional to the batch (microseconds to a few
milliseconds), so 147 appends finish well under a second.

## Actual (0.9.0, measured through duckdb-bolt-extension, which used exactly this call
sequence)

300,000 rows x (BIGINT, VARCHAR) in 147 batches took **91.3 s** on the RTX 2060 (about 0.6 s
per `register_batch` on average, growing with the table). Concatenating the batches on the
caller side and calling `register_table` once took **2.57 s** for the same data.

## Status in 0.10.0

**Improved but still O(table) per call (by reading; not timed)**. `register_batch`
(`src/exec/engine.rs`) now replaces the stored batches with one coalesced batch
(`concat_table_batches`), so queries no longer re-concatenate, but each call still copies the
whole table into the new coalesced batch and still rebuilds the dictionary registry from it
(`registry.unregister_table(name); registry.register_table(name, &concatenated)`). N appends
remain O(N^2) in copied bytes and dictionary work.

## Suggested fix

- Extend the dictionary registry incrementally: look up / insert only the new batch's values
  (first-occurrence order is preserved for the existing prefix, which the incremental GPU upload
  already relies on).
- Keep the table as a chunk list and coalesce lazily (or geometrically, e.g. when the tail
  exceeds the coalesced part), instead of copying the whole table on each append.
- Offer `register_batches(name, impl IntoIterator<Item = RecordBatch>)` that does the work once.

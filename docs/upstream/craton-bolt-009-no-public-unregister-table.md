<!-- SPDX-License-Identifier: Apache-2.0 -->

# craton-bolt-009: No public API to unregister one table (0.9.0)

| Field | Value |
| :--- | :--- |
| Affected | craton-bolt 0.9.0 (crates.io) |
| Fixed in | 0.10.0 source (`248e130`), by reading: `Engine::deregister_table` and `Engine::drop_table` (`src/exec/engine_ddl.rs`); needs a crates.io release |
| Severity | low (API gap; embedders work around it) |
| Environment | any (API), observed on RTX 2060, CUDA 13.3, Windows 11 Pro |
| Found by | duckdb-bolt-extension, internal page W1-B-006 |

## Summary

`craton_bolt::Engine` 0.9.0 exposes `register_table`, `register_batch`, `replace_table` and
`register_table_stream*`, but no way to remove a table. The private
`Engine::unregister_table_best_effort` (`src/exec/engine.rs`) already does the work. An
embedder that maps an external catalog onto the engine (duckdb-bolt-extension:
`bolt_drop_table`, `bolt_clear_tables`) must either rebuild the engine (new CUDA context,
re-upload of every other table) or replace the table with an empty batch and remember the name
as a tombstone. With tombstones a query that names a dropped table returns 0 rows instead of
"unknown table", so the embedder must rebuild before such a query anyway.

## Minimal reproduction (Rust, `craton_bolt::Engine` + arrow 53 only)

```toml
[dependencies]
craton-bolt = "0.9.0"
arrow = "53"
```

```rust
use std::sync::Arc;

use arrow::array::{ArrayRef, Int32Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use craton_bolt::Engine;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int32, false)]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from(vec![1, 2, 3])) as ArrayRef],
    )?;
    let mut engine = Engine::new()?;
    engine.register_table("t", batch.clone())?;

    // Desired: engine.unregister_table("t")?; -- no such public method in 0.9.0.
    // Closest workaround: keep the name, drop the rows.
    engine.replace_table("t", RecordBatch::new_empty(schema))?;
    let out = engine.sql("SELECT COUNT(*) FROM t")?; // 0, not "unknown table 't'"
    println!("{:?}", out.record_batch().column(0));

    // Re-registering under the same name also fails until the table is removed:
    let err = engine.register_table("t", batch).unwrap_err();
    println!("register_table again: {err}");
    Ok(())
}
```

## Expected

A public `Engine::unregister_table(&mut self, name) -> BoltResult<bool>` (or similar) that
frees the host and device copies, dictionaries and provider schema of one table, after which a
query naming it fails with "unknown table".

## Status in 0.10.0

**Fixed by reading**: 0.10.0 adds `pub fn deregister_table(&mut self, name: &str) ->
BoltResult<()>` (removes the host batches, streaming / storage sources, GPU table, residency,
dictionaries, provider schema and host revisions; errors if the name is not registered) and
`pub fn drop_table` (also drops a catalog table). Please publish it; duckdb-bolt-extension
will replace its tombstone logic with `deregister_table` once 0.10 is on crates.io.

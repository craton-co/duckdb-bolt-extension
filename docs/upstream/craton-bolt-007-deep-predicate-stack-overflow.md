<!-- SPDX-License-Identifier: Apache-2.0 -->

# craton-bolt-007: Long OR chains and dictionary-lowered string predicates overflow the thread stack and abort the process

| Field | Value |
| :--- | :--- |
| Affected | craton-bolt 0.9.0 (crates.io); 0.10.0 source (`248e130`) probably (by reading) |
| Fixed in | not fixed |
| Severity | critical (process abort, `STATUS_STACK_OVERFLOW`; reachable with ordinary data and SQL) |
| Environment | NVIDIA GeForce RTX 2060, CUDA 13.3, Windows 11 Pro, Rust stable (debug; release has smaller frames but the same growth) |
| Found by | duckdb-bolt-extension, internal page W2-B-001 (retired; mitigated in the extension with a 512 MiB worker stack and query limits) |

## Summary

The planner, optimizer, NULL rewrite and codegen walk predicate trees recursively. Two common
inputs build left-deep trees whose depth grows with the query or with the **data**:

1. A long `OR` list in the SQL text: `id = 0 OR id = 1 OR ... OR id = 299`.
2. A dictionary-lowered Utf8 predicate: `build_index_membership`
   (`src/plan/string_literal_rewrite.rs`) turns `s LIKE ...` or `s > 'lit'` into one
   `__idx_s = k` equality per **matching dictionary entry**, folded into a left-deep `OR`
   chain. Depth = number of distinct matching values in the column.

When the tree is deep enough the thread overflows its stack. On Windows the process is
terminated (`0xc00000fd`); there is no error to catch, and `catch_unwind` cannot help.

## Minimal reproduction (Rust, `craton_bolt::Engine` + arrow 53 only)

```toml
[dependencies]
craton-bolt = "0.9.0"
arrow = "53"
```

```rust
use std::sync::Arc;

use arrow::array::{ArrayRef, Int32Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use craton_bolt::Engine;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 5,000 rows: id = i, s = 'name_' || i (5,000 distinct strings).
    let n = 5_000;
    let id = Int32Array::from((0..n).collect::<Vec<i32>>());
    let s = StringArray::from((0..n).map(|i| format!("name_{i}")).collect::<Vec<_>>());
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("s", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(schema, vec![Arc::new(id) as ArrayRef, Arc::new(s)])?;

    // Run on a thread with the default 2 MiB stack (std::thread::spawn), as most hosts do.
    let handle = std::thread::spawn(move || {
        let mut engine = Engine::new().expect("engine");
        engine.register_table("t", batch).expect("register");
        let or_chain = (0..300).map(|i| format!("id = {i}")).collect::<Vec<_>>().join(" OR ");
        for q in [
            "SELECT id FROM t WHERE s LIKE 'name_1%'".to_string(), // 1,111 matching entries
            "SELECT id FROM t WHERE s > 'name_4'".to_string(),     // ~4,400 matching entries
            format!("SELECT id FROM t WHERE {or_chain}"),          // 300 terms
        ] {
            eprintln!("running: {}...", &q[..q.len().min(60)]);
            match engine.sql(&q) {
                Ok(out) => eprintln!("  rows = {}", out.record_batch().num_rows()),
                Err(e) => eprintln!("  error: {e}"),
            }
        }
    });
    handle.join().expect("worker thread");
    Ok(())
}
```

## Expected

Each query returns its rows (1,111; 5,000 - count(s <= 'name_4'); 300) or a clean `Err`
("predicate too complex").

## Actual (0.9.0, measured, debug build, 2 MiB thread stack)

| Predicate | Matching entries / terms | Result |
| :--- | ---: | :--- |
| `s LIKE '%'` | 100 | ok |
| `s LIKE '%'` | 250 | stack overflow, process aborted |
| `s > 'k'` | 1,500 | stack overflow |
| `id = 0 OR ... OR id = 99` | 100 | ok |
| `id = 0 OR ... OR id = 299` | 300 | stack overflow |
| `id IN (5,000 values)` | - | clean error (`IN with > 64 values not yet supported`) |

`explain_sql` overflows as well, so validating a query with EXPLAIN first does not help.

## Status in 0.10.0

**Probably still present (by reading, not run)**:

- `build_index_membership` still emits a left-deep `OR` (`acc = Or(acc, eq(idx))`), so the
  depth still grows with the number of matching dictionary entries, which no SQL-size limit
  bounds.
- 0.10 adds a pre-parse guard (`MAX_SQL_BYTES_DEFAULT` = 1 MiB, `MAX_SQL_TOKENS_DEFAULT` =
  100,000 tokens) and a lowering depth limit (`MAX_RECURSION_DEPTH` = 256) in
  `src/plan/sql_frontend.rs`. 100,000 tokens still admits an `OR` chain of about 25,000
  terms, far deeper than the 300 that overflowed a 2 MiB stack; and the new
  `plan::null_semantics` rewrite is another recursive pass over the same tree.

## Suggested fix

- Build balanced trees (or a single `InList` / bitmap-membership node) for membership tests
  and for flattened `OR` / `AND` chains; better, represent n-ary `AND` / `OR` as one node.
- Make the predicate walkers (type check, optimizer, NULL rewrite, codegen, `Drop`) iterative
  or depth-bounded with a clean error.
- Document the stack the engine needs per predicate term until then.

Mitigated in duckdb-bolt-extension by W9-A as well: queries the guard's planner cannot plan
(`DISTINCT ON`, `LATERAL`, correlated subqueries, ...) are no longer sent to the engine, so the
dictionary-lowering check cannot be skipped through them (W8-B-003).

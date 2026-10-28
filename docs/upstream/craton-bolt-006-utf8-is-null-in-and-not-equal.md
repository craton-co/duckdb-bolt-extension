<!-- SPDX-License-Identifier: Apache-2.0 -->

# craton-bolt-006: Utf8 predicates: `IS [NOT] NULL` and `IN` fail at execution, `!=` matches NULL rows, and EXPLAIN accepts plans that cannot run

| Field | Value |
| :--- | :--- |
| Affected | craton-bolt 0.9.0 (crates.io) |
| Fixed in | partly in 0.10.0 source (`248e130`): see "Status in 0.10.0" |
| Severity | high (`!=` / `NOT` return NULL rows); medium (the rest fail with an error) |
| Environment | NVIDIA GeForce RTX 2060, CUDA 13.3, Windows 11 Pro, Rust stable |
| Found by | duckdb-bolt-extension, internal page W1-A-004 (retired; root cause measured in W2-B) |

## Summary

On a nullable `Utf8` column `s` (the engine dictionary-encodes it into `__idx_s`, NULL = slot 0):

| SQL | 0.9.0 result | Expected |
| :--- | :--- | :--- |
| `SELECT id FROM t WHERE s IS NOT NULL` | `Utf8 not supported in PTX codegen yet` | the non-NULL rows |
| `SELECT id FROM t WHERE s IS NULL` | `Utf8 not supported in PTX codegen yet` | the NULL rows |
| `SELECT id FROM t WHERE s IN ('v1', 'v2')` | `expr_agg: column '__idx_s' not found in evaluator env` | rows with `v1` / `v2` |
| `SELECT id, s FROM t WHERE id > 3` (s holds NULLs) | `Utf8 not supported in PTX codegen yet` | the rows |
| `SELECT id FROM t WHERE s != 'v1'` | runs, **NULL rows included** | NULL rows excluded |
| `SELECT id FROM t WHERE NOT (s = 'v1')` | runs, **NULL rows included** | NULL rows excluded |
| `SELECT count(*) FROM t WHERE s = 'v1'` | `plan error: pre kernel input column '__idx_s' not present in table batch` | the count |
| `s = 'v1'`, `s > 'v3'`, `s LIKE 'v1%'` (projection of `id`) | correct | |

`explain_sql` succeeds for every row of the table, so a caller that validates SQL with EXPLAIN
(as duckdb-bolt-extension's filter pushdown did) accepts plans that then fail in `sql`.

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
    // 100 rows: id = i, s = 'v' || (i % 6), NULL when i % 7 == 0 (15 NULLs).
    let n = 100;
    let id = Int32Array::from((0..n).collect::<Vec<i32>>());
    let s = StringArray::from(
        (0..n)
            .map(|i| (i % 7 != 0).then(|| format!("v{}", i % 6)))
            .collect::<Vec<_>>(),
    );
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("s", DataType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(schema, vec![Arc::new(id) as ArrayRef, Arc::new(s)])?;

    let mut engine = Engine::new()?;
    engine.register_table("t", batch)?;

    for q in [
        "SELECT id FROM t WHERE s IS NOT NULL",       // expected 85 rows
        "SELECT id FROM t WHERE s IS NULL",           // expected 15 rows
        "SELECT id FROM t WHERE s IN ('v1', 'v2')",   // expected 28 rows
        "SELECT id, s FROM t WHERE id > 3",           // expected 96 rows
        "SELECT id FROM t WHERE s != 'v1'",           // expected 71 rows
        "SELECT id FROM t WHERE NOT (s = 'v1')",      // expected 71 rows
        "SELECT count(*) FROM t WHERE s = 'v1'",      // expected 14
    ] {
        let explain = engine.explain_sql(q).map(|_| "ok").unwrap_or("error");
        match engine.sql(q) {
            Ok(out) => println!("{q}\n  explain {explain}; rows = {}", out.record_batch().num_rows()),
            Err(e) => println!("{q}\n  explain {explain}; error: {e}"),
        }
    }
    Ok(())
}
```

Counts: `i % 6 == 1` for 17 values of `i` in 0..99, 3 of which (7, 49, 91) are NULL, so 14 rows
hold `v1`; `v2` likewise 14 (NULL at 14, 56, 98); `s != 'v1'` = 85 - 14 = 71.

## Status in 0.10.0

- `count(*) ... WHERE s = 'lit'` (`pre kernel input column '__idx_s' not present`): **fixed** per
  the 0.10.0 CHANGELOG ("Aggregates filtered by a string predicate execute").
- `SELECT <utf8 col>, ... WHERE <non-string predicate>`: **fixed** per the CHANGELOG
  ("`SELECT <utf8 col>, ... WHERE <non-string predicate>` executes").
- `!=` / `NOT` including NULL rows: **probably fixed** by `plan::null_semantics` (Kleene `NOT`
  over nullable columns), unverified for dictionary-folded Utf8 predicates.
- `IS [NOT] NULL` on Utf8 and Utf8 `IN`: **unverified**. `ptx_gen.rs` still rejects Utf8
  operands ("Utf8 not supported in PTX codegen yet") and Utf8 `IN` still lowers to an OR of
  equalities, so it depends on whether the dictionary rewrite now folds both shapes.
- EXPLAIN-accepts-but-execution-fails: unverified.

## Suggested fix

Lower `s IS NULL` to `__idx_s = 0` (and `IS NOT NULL` to `__idx_s <> 0`) in the dictionary
rewrite, fold Utf8 `IN` into an index membership test, and make `explain_sql` run the same
lowering (or a dry-run codegen) so plans it accepts are executable.

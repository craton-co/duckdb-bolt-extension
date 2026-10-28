# W1-B-006: craton-bolt has no public way to unregister one table, so drops use empty placeholders

- **Severity:** low (performance edge cases; upstream API gap)
- **Status:** blocked-upstream
- **Files:** `crates/core/src/runtime.rs` (`engine_forget_table`, `purge_tombstones_referenced_by`,
  `MAX_ENGINE_TOMBSTONES`); upstream `craton-bolt` 0.9.0 `src/exec/engine.rs`
  (`unregister_table_best_effort` is private)

## Description
`craton_bolt::Engine` 0.9 exposes `register_table`, `replace_table`, and `register_batch`, but no
way to remove a table. Rebuilding the engine on each `bolt_drop_table` / `bolt_clear_tables` would
mean a new CUDA context and a re-upload of every table. To avoid that, W1-B replaces the dropped
table with an empty batch, which returns its device memory to the pool, and records the name as a
tombstone. This has three consequences:
- If a query's text may reference a tombstoned name (case-insensitive substring match), the engine
  is rebuilt before the query runs. The query then fails with "unknown table" instead of returning
  0 rows. With short table names such as `t` or `id`, a drop can trigger one unneeded rebuild.
- More than 128 tombstones trigger a rebuild.
- Re-registering a tombstoned name uses `replace_table`, so no rebuild is needed.

## Suggested fix
Ask upstream for `Engine::unregister_table(&mut self, name) -> BoltResult<bool>`. The private
`unregister_table_best_effort` already does the work. Then replace the tombstone logic with the new
call.

## W3-A notes

6dcd2b9 (W3-A): tombstone checks use whole-word matching
(W2-E-007), so dropping `t` no longer rebuilds the engine for every query. The upstream
unregister API is still missing.

Upstream report: [craton-bolt-009](../upstream/craton-bolt-009-no-public-unregister-table.md) (W4-D; 0.10.0 source adds `Engine::deregister_table`).
## W4-A notes (blocked upstream)

Re-verified in b1158f0 against craton-bolt 0.9.0 (the version on crates.io and in `Cargo.lock`):
`Engine` still has only `register_table`, `register_table_stream[_lazy]`, `replace_table` and
`register_batch`; `unregister_table_best_effort` is private. According to
`docs/upstream/craton-bolt-009-no-public-unregister-table.md` (branch bolt/w4-d) the 0.10.0
source (`248e130`, `src/exec/engine_ddl.rs`) **does** add `Engine::deregister_table` and
`Engine::drop_table`. **Upstream ask:** publish 0.10.x with that API on crates.io. Then
replace `engine_forget_table` / tombstones / `MAX_ENGINE_TOMBSTONES` with `drop_table`.

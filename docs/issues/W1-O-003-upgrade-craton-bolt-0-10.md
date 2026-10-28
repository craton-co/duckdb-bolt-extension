# W1-O-003: Upgrade to craton-bolt 0.10 once it is published

- **Severity:** high
- **Status:** blocked-upstream
- **Kind:** proposal
- **Files:** `Cargo.toml` (`craton-bolt = "0.9.0"`), `crates/core/src/runtime.rs`

## Description
The local craton-bolt checkout (`C:/craton/bolt`, version 0.10.0, unpublished as of 2026-10-26)
no longer crashes on the W1-O-002 queries (filtered aggregates over columns with NULLs). crates.io
only has 0.9.0, so the extension cannot depend on 0.10 without breaking `cargo publish`.

Probe against 0.10.0 (100 rows, every 7th NULL):
- `SELECT COUNT(*) FROM t WHERE d > 0` → 85 (correct; 0.9.0 crashes).
- `SELECT SUM(d) FROM t WHERE d IS NULL` → **0** — SQL requires NULL. New upstream correctness bug to report.

0.10.0 still has no per-table unregister API (see W1-B-006).

## Suggested fix
Publish craton-bolt 0.10.x with the SUM-over-empty fix, bump the dependency, re-run the full
suite on GPU, and remove the W1-O-002 mitigation if the differential tests stay green.

## W3-A notes

6dcd2b9 (W3-A): the runtime now also works around
0.9.0's deep-recursion crashes (W2-B-001) and coalesces uploads (W2-O-001); re-check both on
0.10 (a balanced / iterative predicate walker and a cheaper `register_batch` would let
`MAX_GPU_PREDICATE_TERMS`, `MAX_GPU_DICTIONARY_LOWERED_VALUES` and the upload coalescing be
relaxed). The W3-E-002 outer-join guard must be re-verified as well.

Upstream reports: [craton-bolt-001](../upstream/craton-bolt-001-crash-filtered-aggregate-over-nulls.md), [craton-bolt-003](../upstream/craton-bolt-003-scalar-sum-min-max-over-empty-input.md), index [docs/upstream](../upstream/README.md) (W4-D; note 0.10.0 does have `Engine::deregister_table`, see craton-bolt-009).
## W4-A notes (blocked upstream)

Re-verified in b1158f0: crates.io still lists only `craton-bolt = "0.9.0"` (`cargo search`), so the
dependency cannot move. Correction to the description above: per
`docs/upstream/craton-bolt-009-no-public-unregister-table.md` (bolt/w4-d) the 0.10.0 source
adds `Engine::deregister_table` / `drop_table`, so W1-B-006 is fixed upstream once released.
**Upstream asks:** publish 0.10.x; fix `SUM` over an empty filtered input returning `0`
instead of NULL (craton-bolt-003). After the bump re-check W1-O-002 / W2-A-001 guards,
W2-B-001 limits, W2-O-001 coalescing, W3-E-002, and W3-A-001 sharing.

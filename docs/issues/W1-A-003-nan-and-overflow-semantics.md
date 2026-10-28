# W1-A-003: Engine NaN and integer-overflow semantics differ from DuckDB (bolt_query / bolt_sql)

- **Severity:** medium
- **Status:** blocked-upstream
- **Files:** craton-bolt engine; `crates/core/src/table_functions.rs` (the `bolt_scan` pushdown now compensates); `bolt_query` / `bolt_sql` / `bolt_aggregate` results

## Description

DuckDB orders NaN above every other value and treats `NaN = NaN` as TRUE. The engine uses
IEEE comparisons: every comparison with NaN is FALSE, and `!=` is an *ordered* not-equal, so
`NaN != 3` is also FALSE. Measured on a 10-row DOUBLE column with 3 NaNs:

| predicate | DuckDB | engine |
| :--- | ---: | ---: |
| `d > 3` | 8 | 5 |
| `NOT (d > 3)` | 2 | 5 |
| `d = d` | 10 (non-NULL) | 7 |
| `d != d` | 0 | 0 |

Integer `+ - *` wrap on overflow on the GPU, where DuckDB raises *Out of Range*.

`bolt_scan` no longer depends on this:
- DOUBLE `>`, `>=` and `!=` against a constant are pushed as `(e op c) OR NOT (e = e)`.
- Float column-vs-column comparisons are not pushed.
- `NOT IN` is spelled `NOT (e IN (...))`.
- Only DOUBLE arithmetic is pushed.

`bolt_query('…')`, `bolt_sql` and `bolt_aggregate` run user SQL on the engine directly, so
users still get IEEE / wrapping semantics there.

## Reproduction

```sql
CREATE TABLE p AS SELECT CASE WHEN i % 4 = 0 THEN 'NaN'::DOUBLE ELSE i + 0.4 END AS d FROM range(0,10) t(i);
COPY p TO 'gp' (FORMAT bolt, USE_TMP_FILE FALSE);
SELECT count(*) FROM p WHERE d > 3;                                   -- 8
SELECT * FROM bolt_query('SELECT count(*) FROM gp WHERE d > 3');      -- 5
```

## Suggested fix

Document the IEEE / wrapping semantics in `docs/SQL_REFERENCE.md` for `bolt_query`. Or offer a
DuckDB-compatible mode in the engine: total-order float comparisons and checked integer
arithmetic.

## W2-B notes

No change. `bolt_scan` still compensates; the W1-A matrix re-checks this and now compares full
rows. `bolt_query`, `bolt_sql` and `bolt_aggregate` need either documentation or an engine
mode, and there is nothing to fix in `table_functions.rs`. W2-A-001 (GPU filters ignore NULL
validity) is the same class of problem for `bolt_query` and is tracked there.

## W3-B notes (commit 71e05e7)

Nothing to fix in `table_functions.rs`; `bolt_scan` still compensates and the W1-A / W2-B /
W3-B differential suites now check every predicate on both the GPU engine and the host
fallback. The remaining work (documenting IEEE / wrapping semantics of `bolt_query` /
`bolt_sql` / `bolt_aggregate` in `docs/SQL_REFERENCE.md`) belongs to the docs owner. Note that
on tables with NULLs these functions now run on the host fallback (DuckDB semantics), so the
engine semantics only show on NULL-free tables.

Upstream report: [craton-bolt-010](../upstream/craton-bolt-010-nan-ordering-and-integer-overflow.md) (W4-D).
## W4-B re-check (RTX 2060, craton-bolt 0.9.0): blocked-upstream

Unchanged. On a NULL-free 10-row DOUBLE table with 3 NaNs, `bolt_query('SELECT count(*) FROM g
WHERE d > 3')` returns 5 (DuckDB: 8) and `d = d` returns 7 (DuckDB: 10).
`bolt_query('SELECT id * 2147483647 * 4 FROM g WHERE id = 9')` returns 77309411292 where DuckDB
raises an INTEGER overflow. `bolt_scan` still compensates: the W4-B fuzzer generates NaN, +-inf,
-0.0 and integer extremes in 20% of its tables and finds no `bolt_scan` difference. The engine
semantics need an engine mode or documentation (`docs/SQL_REFERENCE.md`, docs owner), so the
fuzzer does not send `bolt_query` on such tables. A related but separate difference (arithmetic
on storage types, on both backends) is W4-B-002.

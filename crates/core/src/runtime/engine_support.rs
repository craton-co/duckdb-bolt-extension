// SPDX-License-Identifier: Apache-2.0

//! Engine upload helpers and plan inspection for the GPU crash guard (craton-bolt 0.9
//! workarounds).

use super::*;

/// Total string bytes per column of `batches` (`0` for non-string columns), counted from the
/// offsets, so sliced arrays count only their own values (W2-E-004).
#[cfg(not(target_arch = "wasm32"))]
pub(super) fn utf8_bytes_per_column<'a>(
    num_columns: usize,
    batches: impl IntoIterator<Item = &'a RecordBatch>,
) -> Vec<usize> {
    use arrow_array::cast::AsArray;
    let mut totals = vec![0usize; num_columns];
    for batch in batches {
        for (total, col) in totals.iter_mut().zip(batch.columns()) {
            let bytes = match col.data_type() {
                ArrowDataType::Utf8 => {
                    let offsets = col.as_string::<i32>().value_offsets();
                    match (offsets.first(), offsets.last()) {
                        (Some(&first), Some(&last)) => (last - first).max(0) as usize,
                        _ => 0,
                    }
                }
                ArrowDataType::LargeUtf8 => {
                    let offsets = col.as_string::<i64>().value_offsets();
                    match (offsets.first(), offsets.last()) {
                        (Some(&first), Some(&last)) => (last - first).max(0) as usize,
                        _ => 0,
                    }
                }
                _ => 0,
            };
            *total = total.saturating_add(bytes);
        }
    }
    totals
}

/// Error when `batches` (with `schema`) holds more string data in one column than craton-bolt
/// 0.9 can keep in one `Utf8` array (W2-E-004): the engine concatenates every table into one
/// batch (`materialize_table`), and `concat_batches` panics on offset overflow.
#[cfg(not(target_arch = "wasm32"))]
pub(super) fn check_engine_utf8_capacity<'a>(
    name: &str,
    schema: &Schema,
    batches: impl IntoIterator<Item = &'a RecordBatch>,
) -> Result<()> {
    let totals = utf8_bytes_per_column(schema.fields().len(), batches);
    for (field, bytes) in schema.fields().iter().zip(totals) {
        if bytes > MAX_ENGINE_UTF8_BYTES {
            return Err(Error::invalid_input(format!(
                "Bolt table '{name}' is not uploaded to the GPU engine: column '{}' holds {bytes} \
                 bytes of string data, more than craton-bolt 0.9 keeps in one Utf8 array ({} \
                 bytes); the table is served by the host fallback",
                field.name(),
                MAX_ENGINE_UTF8_BYTES
            )));
        }
    }
    Ok(())
}

/// Coalesce `batches` into the one batch craton-bolt 0.9 keeps per table (W2-O-001).
///
/// craton-bolt's `register_batch` re-concatenates the whole table and rebuilds its string
/// dictionaries on every call, so uploading a COPY's ~2,048-row chunks one by one cost
/// O(batches^2) (measured: 91 s for 300,000 rows in 147 batches on an RTX 2060). The shadow
/// catalog keeps the original batches; only the engine copy is concatenated.
#[cfg(not(target_arch = "wasm32"))]
pub(super) fn coalesce_for_engine(name: &str, batches: &[RecordBatch]) -> Result<RecordBatch> {
    let Some(first) = batches.first() else {
        return Err(Error::api_error(format!(
            "Bolt table '{name}' has no batches"
        )));
    };
    if batches.len() == 1 {
        return Ok(first.clone());
    }
    let schema = first.schema();
    check_engine_utf8_capacity(name, &schema, batches)?;
    arrow_select::concat::concat_batches(&schema, batches).map_err(|e| {
        Error::invalid_input(format!(
            "Bolt engine failed to upload table '{name}': cannot combine its batches: {e}"
        ))
    })
}

/// Upload `batches` as table `name` with one `register_table` (or `replace_table` when
/// `replace`) call (W2-O-001; see [`coalesce_for_engine`]).
///
/// Returns the concatenated batch the engine now keeps on the host when `batches` had to be
/// coalesced (more than one batch), so the catalog can share it
/// ([`BoltWorkerState::share_engine_copy`], W3-A-001); `None` when the engine already holds
/// the catalog's own (single) batch.
#[cfg(not(target_arch = "wasm32"))]
pub(super) fn engine_upload(
    eng: &mut craton_bolt::Engine,
    name: &str,
    batches: &[RecordBatch],
    replace: bool,
) -> Result<Option<RecordBatch>> {
    let batch = coalesce_for_engine(name, batches)?;
    let shared = (batches.len() > 1).then(|| batch.clone());
    if replace {
        eng.replace_table(name.to_string(), batch)
    } else {
        eng.register_table(name.to_string(), batch)
    }
    .map_err(|e| {
        Error::invalid_input(format!("Bolt engine failed to upload table '{name}': {e}"))
    })?;
    Ok(shared)
}

/// Whether craton-bolt 0.9.0 evaluates `plan` correctly (and without crashing) when the
/// tables it scans contain NULLs (W1-O-002).
///
/// Measured on 0.9.0: the GPU expression evaluator ignores Arrow validity bitmaps. Filters
/// test the garbage values under NULL slots (`WHERE d > 0` keeps NULL rows whose hidden value
/// is positive), computed columns lose their NULLs (`d + 1` is never NULL), filtered columns
/// come back without their validity, and an aggregate over a filtered input crashes the
/// process. Plain column scans, projections, `LIMIT`, `DISTINCT`, `UNION ALL`, equi joins, and
/// `COUNT`/`SUM`/`MIN`/`MAX`/`AVG` of plain columns (grouped by plain columns) are correct,
/// with two exceptions found by the W4-B differential fuzzer:
///
/// - grouped `SUM` / `MIN` / `MAX` of a group whose values are all NULL return the kernel's
///   identity value (`0`, `i32::MAX`, `-inf`, `NaN`, ...) instead of NULL (W4-B-001), so a
///   grouped `SUM` / `MIN` / `MAX` over a column that may hold NULLs is unsafe;
/// - a sort key with `nulls_first` (the engine's default for ascending keys, while DuckDB
///   defaults to `NULLS LAST`; an explicit `NULLS FIRST` plans the same) puts NULLs first
///   (W4-B-003), so such a key over a column that may hold NULLs is unsafe. An explicit
///   `NULLS LAST`, and `DESC` (default `NULLS LAST` on both), stay on the GPU.
///
/// Whether a column may hold a NULL is derived through the plan ([`NodeFacts`]): a `Scan`
/// takes it from the stored table it names, and renames, UNION branches, outer-join padding
/// and NULL literals carry it up by position (W8-B-004). Anything else is treated as unsafe.
#[cfg(not(target_arch = "wasm32"))]
pub(super) fn plan_is_null_safe(
    plan: &craton_bolt::plan::LogicalPlan,
    cat: &dyn GuardCatalog,
) -> bool {
    use craton_bolt::plan::{AggregateExpr as A, Expr as E, LogicalPlan as P};
    fn plain(e: &E) -> bool {
        match e {
            E::Column(_) | E::Literal(_) => true,
            E::Alias(inner, _) => matches!(inner.as_ref(), E::Column(_) | E::Literal(_)),
            _ => false,
        }
    }
    let safe = |p: &P| plan_is_null_safe(p, cat);
    match plan {
        P::Scan { .. } => true,
        P::Project { input, exprs } => exprs.iter().all(plain) && safe(input),
        P::Sort { input, sort_exprs } => {
            let facts = NodeFacts::of(input, cat);
            sort_exprs
                .iter()
                .all(|s| plain(&s.expr) && !(s.nulls_first && facts.expr(&s.expr).nullable))
                && safe(input)
        }
        P::Limit { input, .. } | P::Distinct { input } => safe(input),
        P::Union { inputs } => inputs.iter().all(safe),
        P::Aggregate {
            input,
            group_by,
            aggregates,
        } => {
            let grouped = !group_by.is_empty();
            let facts = NodeFacts::of(input, cat);
            // W9-A-003: below a Sort, Limit or Distinct the engine aggregates a column without
            // its validity (`count(v)` counts the NULLs too), so only an input that is a scan
            // (through plain projections) keeps the NULLs of aggregated columns.
            fn scan_chain(p: &P) -> bool {
                match p {
                    P::Scan { .. } => true,
                    P::Project { input, exprs } => exprs.iter().all(plain) && scan_chain(input),
                    _ => false,
                }
            }
            let direct = scan_chain(input);
            // W11-A-001: without GROUP BY, `count(*)` next to an aggregate of a column that may
            // hold NULLs takes another engine path that ignores the NULLs: `count(c)` counts
            // them, and `min` / `max` / `sum` of an all-NULL column return the kernel identity.
            let is_star = |e: &E| !matches!(e, E::Column(c) if c != "*");
            let count_star = aggregates
                .iter()
                .any(|a| matches!(a, A::Count(e) if is_star(e)));
            let star_hazard = !grouped
                && count_star
                && aggregates.iter().any(|a| match a {
                    A::Count(e) | A::Sum(e) | A::Min(e) | A::Max(e) | A::Avg(e) => {
                        !is_star(e) && facts.expr(e).nullable
                    }
                    _ => false,
                });
            !star_hazard
                && group_by.iter().all(plain)
                && aggregates.iter().all(|a| match a {
                    A::Count(e) | A::Avg(e) => plain(e) && (direct || !facts.expr(e).nullable),
                    A::Sum(e) | A::Min(e) | A::Max(e) => {
                        plain(e) && !((grouped || !direct) && facts.expr(e).nullable)
                    }
                    _ => false,
                })
                && safe(input)
        }
        P::Join {
            left,
            right,
            on,
            filter,
            join_type,
        } => {
            // Outer joins pad unmatched rows with NULLs the GPU kernels ignore (W3-E-002).
            matches!(
                join_type,
                craton_bolt::plan::logical_plan::JoinType::Inner
                    | craton_bolt::plan::logical_plan::JoinType::Cross
            ) && filter.is_none()
                && on.iter().all(|(l, r)| plain(l) && plain(r))
                && safe(left)
                && safe(right)
        }
        P::Filter { .. } | P::Window { .. } | P::SetOp { .. } => false,
    }
}

/// Why craton-bolt 0.9.0 computes `plan` wrongly whatever the NULLs, if it does: the type
/// hazards found by the W4-B differential fuzzer and the W8-B review.
///
/// - **DECIMAL casts and arithmetic** (W4-B-004): `CAST(<negative DECIMAL> AS DOUBLE)` returns
///   wrong values (`-0.25` -> `-0.2048`, or `0.0` for `DECIMAL(9, 2)`). Every `CAST` from or to
///   a DECIMAL, and every arithmetic operator with a DECIMAL operand, is a hazard.
/// - **Arithmetic on widened integers** (W4-B-002): the engine evaluates `TINYINT` / `SMALLINT`
///   / `UTINYINT` / `USMALLINT` / `UINTEGER` / `UBIGINT` / `HUGEINT` columns on their Int32 /
///   Int64 storage type, so DuckDB's overflow errors for the original type are not raised (and
///   unsigned results go negative). An arithmetic operator over such a column (tracked through
///   the plan, [`ColumnFact::narrow`]) is a hazard. The host fallback mirrors these columns
///   with their original types ([`FallbackDb::mirror_table`]).
/// - **FLOAT arithmetic** (W5-A-002): the engine evaluates arithmetic with a `FLOAT`
///   (Float32) operand in double precision and returns `DOUBLE`, while DuckDB computes and
///   returns `FLOAT` (`c / 3.25`: `3.4615384615384617` vs `3.4615386`). Any arithmetic operator
///   with a Float32 operand is a hazard.
/// - **Integer division and modulo** (W8-B-001): DuckDB's `/` is float division (`7 / 2 =
///   3.5`) and a zero divisor is an error, while the engine truncates (`div.s32` / `div.s64`)
///   and returns NULL for `x / 0` and `x % 0`. A `/` or `%` without a floating-point operand
///   is a hazard (float division and modulo agree, `inf` / `NaN` included).
/// - **Fractional literals** (W8-B-002): DuckDB types `0.1` as an exact DECIMAL, the engine as
///   DOUBLE (`3 * 0.1 = 0.30000000000000004`). When the text has such a literal
///   (`fractional_literals`), arithmetic between a DOUBLE literal and a non-float operand (or
///   another literal) is a hazard; `<double column> * 0.75` is DOUBLE arithmetic in DuckDB too.
/// - **Floating-point to integer casts** (W12-A-002): `CAST(3.7 AS INTEGER)` is `4` in DuckDB
///   (rounded) and `3` on the engine (truncated).
/// - **Timestamp subtraction** (W8-B-008): `TIMESTAMP - TIMESTAMP` returns microsecond ticks
///   (BIGINT) on the engine and an INTERVAL in DuckDB.
/// - **Ungrouped `count(*)` next to `SUM(<INTEGER column>)`** (W5-A-001): without `GROUP BY`,
///   `SELECT count(*), sum(i) ...` sums only a prefix of an Int32 column (5,000 rows:
///   `3275520` instead of `12497500`); `count(col)`, Int64 / float columns and grouped
///   aggregates are correct.
/// - **Ungrouped `SUM` / `MIN` / `MAX` / `STDDEV_POP` over an empty input** (W5-A-003, the
///   NULL-free form of [craton-bolt-003], and W8-B-006): when a filter removes every row, the
///   engine returns the kernel's identity (`0`, `i64::MAX`, `-inf`, ...; `0.0` for
///   `STDDEV_POP`) instead of NULL; with `count(*)` in the same SELECT an empty table does too.
///   Such an aggregate is a hazard when its input may be empty ([`may_be_empty`]), or always
///   inside a subquery (`force_empty`).
/// - **`SUM` accumulators** (W8-B-007): the engine sums FLOAT in single precision (grouped
///   sums are off by thousands of units in the 1e9 range) and integers in a wrapping Int64,
///   where DuckDB uses DOUBLE / HUGEINT. Every `SUM` of a FLOAT, and every `SUM` / `AVG` of an
///   integer whose bound ([`ColumnFact::abs_max`] times [`row_bound`]) may exceed `i64::MAX`,
///   is a hazard.
///
/// An expression whose type the planner cannot derive counts as a DECIMAL operand
/// (conservative). Statistical aggregates and window functions are checked on their `Debug`
/// text.
#[cfg(not(target_arch = "wasm32"))]
pub(super) fn plan_type_hazard(
    plan: &craton_bolt::plan::LogicalPlan,
    cat: &dyn GuardCatalog,
    fractional_literals: bool,
    force_empty: bool,
) -> Option<&'static str> {
    use craton_bolt::plan::{
        AggregateExpr as A, BinaryOp as B, DataType as D, Expr as E, Literal as L,
        LogicalPlan as P, logical_plan::WindowFunc as W,
    };
    const DECIMAL_CAST: &str = "the query casts from or to DECIMAL, which craton-bolt 0.9.0 \
                                computes wrongly for negative values on the GPU (W4-B-004)";
    const DECIMAL_ARITH: &str = "the query does arithmetic on DECIMAL values, which craton-bolt \
                                 0.9.0 does not evaluate reliably on the GPU (W4-B-004)";
    const NARROW_ARITH: &str = "the query does arithmetic on a column stored in a wider integer \
                                type (TINYINT, SMALLINT, unsigned or HUGEINT), whose overflow \
                                craton-bolt 0.9.0 does not detect (W4-B-002)";
    const FLOAT_ARITH: &str = "the query does arithmetic on FLOAT values, which craton-bolt \
                               0.9.0 evaluates in double precision (W5-A-002)";
    const INT_DIV: &str = "the query divides integers (/ or %), which craton-bolt 0.9.0 \
                           truncates and evaluates to NULL for a zero divisor, while DuckDB's / \
                           is float division and a zero divisor is an error (W8-B-001)";
    const FRACTIONAL: &str = "the query does arithmetic with a fractional literal, which DuckDB \
                              evaluates as an exact DECIMAL and craton-bolt 0.9.0 as DOUBLE \
                              (W8-B-002)";
    const FLOAT_TO_INT: &str = "the query casts a floating-point value to an integer, which \
                                craton-bolt 0.9.0 truncates while DuckDB rounds (W12-A-002)";
    const TIMESTAMP_SUB: &str = "the query subtracts timestamps, which craton-bolt 0.9.0 returns \
                                 as raw microsecond ticks instead of an INTERVAL (W8-B-008)";
    const COUNT_STAR_SUM: &str = "the query combines an ungrouped count(*) with SUM of an \
                                  INTEGER column, which craton-bolt 0.9.0 sums only partially \
                                  (W5-A-001)";
    const EMPTY_AGG: &str = "the query has an ungrouped SUM, MIN, MAX or STDDEV_POP over an \
                             input that may be empty, for which craton-bolt 0.9.0 returns the \
                             kernel identity (or 0.0) instead of NULL (W5-A-003, W8-B-006)";
    const FLOAT_SUM: &str = "the query sums FLOAT values, which craton-bolt 0.9.0 accumulates in \
                             single precision (W8-B-007)";
    const INT_SUM: &str = "the query sums integers whose total may exceed the BIGINT range, \
                           which craton-bolt 0.9.0 wraps around (W8-B-007)";

    /// Expressions of one node, evaluated over the rows of `facts`.
    struct Cx<'a> {
        facts: NodeFacts,
        cat: &'a dyn GuardCatalog,
        fractional: bool,
    }
    impl Cx<'_> {
        fn is_decimal(&self, e: &E) -> bool {
            if matches!(e, E::Literal(L::Null)) {
                return false;
            }
            match self.facts.dtype(e) {
                Some(dt) => matches!(dt, D::Decimal128(..)),
                None => true,
            }
        }
        /// Whether a SUM (or AVG) of `e` over at most `rows` rows may leave the Int64 range,
        /// or sums FLOAT values.
        fn sum_hazard(&self, e: &E, rows: Option<u64>, avg: bool) -> Option<&'static str> {
            match self.facts.dtype(e) {
                Some(D::Float32) if !avg => Some(FLOAT_SUM),
                Some(D::Int32 | D::Int64) => {
                    let bound = self.facts.expr(e).abs_max.zip(rows);
                    match bound {
                        Some((b, n)) if u128::from(b) * u128::from(n) <= i64::MAX as u128 => None,
                        _ => Some(INT_SUM),
                    }
                }
                _ => None,
            }
        }
        fn expr(&self, e: &E) -> Option<&'static str> {
            let sub = |x: &E| self.expr(x);
            match e {
                E::Column(_) | E::Literal(_) => None,
                E::Cast {
                    expr: inner,
                    target,
                    ..
                } => {
                    if matches!(target, D::Decimal128(..)) || self.is_decimal(inner) {
                        Some(DECIMAL_CAST)
                    } else if matches!(target, D::Int32 | D::Int64)
                        && matches!(self.facts.dtype(inner), Some(D::Float32 | D::Float64))
                    {
                        Some(FLOAT_TO_INT)
                    } else {
                        sub(inner)
                    }
                }
                E::Binary { op, left, right } => {
                    if matches!(op, B::Add | B::Sub | B::Mul | B::Div | B::Mod) {
                        if self.is_decimal(left) || self.is_decimal(right) {
                            return Some(DECIMAL_ARITH);
                        }
                        if self.facts.expr(left).narrow || self.facts.expr(right).narrow {
                            return Some(NARROW_ARITH);
                        }
                        let (lt, rt) = (self.facts.dtype(left), self.facts.dtype(right));
                        if matches!(lt, Some(D::Float32)) || matches!(rt, Some(D::Float32)) {
                            return Some(FLOAT_ARITH);
                        }
                        let timestamp = |t: &Option<D>| matches!(t, Some(D::Timestamp(..)));
                        if matches!(op, B::Sub) && (timestamp(&lt) || timestamp(&rt)) {
                            return Some(TIMESTAMP_SUB);
                        }
                        let float = |t: &Option<D>| matches!(t, Some(D::Float64 | D::Float32));
                        if matches!(op, B::Div | B::Mod) && !float(&lt) && !float(&rt) {
                            return Some(INT_DIV);
                        }
                        // A DOUBLE literal (DECIMAL in DuckDB) next to an operand that is not
                        // a floating-point expression.
                        let fractional = |lit: &E, other: &E, other_type: &Option<D>| {
                            matches!(lit, E::Literal(L::Float64(_)))
                                && (matches!(other, E::Literal(_)) || !float(other_type))
                        };
                        if self.fractional
                            && (fractional(left, right, &rt) || fractional(right, left, &lt))
                        {
                            return Some(FRACTIONAL);
                        }
                    }
                    sub(left).or_else(|| sub(right))
                }
                E::Unary { operand: x, .. }
                | E::Like { expr: x, .. }
                | E::CastFormat { expr: x, .. }
                | E::Extract { expr: x, .. }
                | E::DateTrunc { expr: x, .. }
                | E::Alias(x, _) => sub(x),
                E::Case {
                    branches,
                    else_branch,
                } => branches
                    .iter()
                    .find_map(|(w, t)| sub(w).or_else(|| sub(t)))
                    .or_else(|| else_branch.as_deref().and_then(sub)),
                E::ScalarFn { args, .. } => args.iter().find_map(sub),
                E::ScalarSubquery(p) => plan_type_hazard(p, self.cat, self.fractional, true),
                E::InSubquery {
                    expr: x, subquery, ..
                } => sub(x).or_else(|| plan_type_hazard(subquery, self.cat, self.fractional, true)),
            }
        }
    }
    /// Conservative check of an expression only available as `Debug` text.
    fn opaque(text: &str) -> Option<&'static str> {
        (text.contains("Cast {") || text.contains("Binary {")).then_some(DECIMAL_ARITH)
    }
    let over = |p: &P| Cx {
        facts: NodeFacts::of(p, cat),
        cat,
        fractional: fractional_literals,
    };
    let below = |p: &P| plan_type_hazard(p, cat, fractional_literals, force_empty);
    match plan {
        P::Scan { .. } => None,
        P::Filter { input, predicate } => over(input).expr(predicate).or_else(|| below(input)),
        P::Project { input, exprs } => {
            let cx = over(input);
            exprs
                .iter()
                .find_map(|e| cx.expr(e))
                .or_else(|| below(input))
        }
        P::Aggregate {
            input,
            group_by,
            aggregates,
        } => {
            let cx = over(input);
            let ungrouped = group_by.is_empty();
            let count_star = aggregates
                .iter()
                .any(|a| matches!(a, A::Count(e) if !matches!(e, E::Column(c) if c != "*")));
            let sum_int32 = aggregates.iter().any(|a| match a {
                A::Sum(e) => matches!(cx.facts.dtype(e), Some(D::Int32) | None),
                _ => false,
            });
            if ungrouped && count_star && sum_int32 {
                return Some(COUNT_STAR_SUM);
            }
            let empty_sensitive = aggregates
                .iter()
                .any(|a| matches!(a, A::Sum(_) | A::Min(_) | A::Max(_) | A::StddevPop(_)));
            if ungrouped && empty_sensitive && (force_empty || may_be_empty(input, cat)) {
                return Some(EMPTY_AGG);
            }
            let rows = row_bound(input, cat);
            aggregates
                .iter()
                .find_map(|a| match a {
                    A::Sum(e) => cx.sum_hazard(e, rows, false),
                    A::Avg(e) => cx.sum_hazard(e, rows, true),
                    _ => None,
                })
                .or_else(|| group_by.iter().find_map(|e| cx.expr(e)))
                .or_else(|| {
                    aggregates.iter().find_map(|a| match a {
                        A::Count(e) | A::Sum(e) | A::Min(e) | A::Max(e) | A::Avg(e) => cx.expr(e),
                        other => opaque(&format!("{other:?}")),
                    })
                })
                .or_else(|| below(input))
        }
        P::Sort { input, sort_exprs } => {
            let cx = over(input);
            sort_exprs
                .iter()
                .find_map(|s| cx.expr(&s.expr))
                .or_else(|| below(input))
        }
        P::Window {
            input,
            window_exprs,
            partition_by,
            order_by,
        } => {
            let cx = over(input);
            let rows = row_bound(input, cat);
            window_exprs
                .iter()
                .find_map(|w| match &w.func {
                    W::Sum(e) => cx.sum_hazard(e, rows, false),
                    W::Avg(e) => cx.sum_hazard(e, rows, true),
                    _ => None,
                })
                .or_else(|| partition_by.iter().find_map(|e| cx.expr(e)))
                .or_else(|| order_by.iter().find_map(|o| cx.expr(&o.expr)))
                .or_else(|| opaque(&format!("{window_exprs:?}")))
                .or_else(|| below(input))
        }
        P::Limit { input, .. } | P::Distinct { input } => below(input),
        P::Union { inputs } => inputs.iter().find_map(below),
        P::SetOp { left, right, .. } => below(left).or_else(|| below(right)),
        P::Join {
            left,
            right,
            on,
            filter,
            ..
        } => {
            let cx = over(plan);
            on.iter()
                .find_map(|(l, r)| cx.expr(l).or_else(|| cx.expr(r)))
                .or_else(|| filter.as_ref().and_then(|f| cx.expr(f)))
                .or_else(|| below(left))
                .or_else(|| below(right))
        }
    }
}

/// Whether `plan` applies an arithmetic operator to a widened integer column
/// ([`ColumnFact::narrow`], W4-B-002), in any node or subquery. The host fallback then mirrors
/// widened integer columns with their original DuckDB types ([`FallbackDb::mirror_table`]), so
/// the arithmetic overflows as in DuckDB; other queries see the storage types (W3-B-002).
#[cfg(not(target_arch = "wasm32"))]
pub(super) fn plan_has_narrow_arithmetic(
    plan: &craton_bolt::plan::LogicalPlan,
    cat: &dyn GuardCatalog,
) -> bool {
    use craton_bolt::plan::{AggregateExpr as A, BinaryOp as B, Expr as E, LogicalPlan as P};
    fn expr(e: &E, facts: &NodeFacts, cat: &dyn GuardCatalog) -> bool {
        let sub = |x: &E| expr(x, facts, cat);
        match e {
            E::Column(_) | E::Literal(_) => false,
            E::Binary { op, left, right } => {
                (matches!(op, B::Add | B::Sub | B::Mul | B::Div | B::Mod)
                    && (facts.expr(left).narrow || facts.expr(right).narrow))
                    || sub(left)
                    || sub(right)
            }
            E::Unary { operand: x, .. }
            | E::Like { expr: x, .. }
            | E::Cast { expr: x, .. }
            | E::CastFormat { expr: x, .. }
            | E::Extract { expr: x, .. }
            | E::DateTrunc { expr: x, .. }
            | E::Alias(x, _) => sub(x),
            E::Case {
                branches,
                else_branch,
            } => {
                branches.iter().any(|(w, t)| sub(w) || sub(t))
                    || else_branch.as_deref().is_some_and(sub)
            }
            E::ScalarFn { args, .. } => args.iter().any(sub),
            E::ScalarSubquery(p) => plan_has_narrow_arithmetic(p, cat),
            E::InSubquery {
                expr: x, subquery, ..
            } => sub(x) || plan_has_narrow_arithmetic(subquery, cat),
        }
    }
    let over = |p: &P, exprs: &mut dyn Iterator<Item = &E>| {
        let facts = NodeFacts::of(p, cat);
        for e in exprs {
            if expr(e, &facts, cat) {
                return true;
            }
        }
        false
    };
    let below = |p: &P| plan_has_narrow_arithmetic(p, cat);
    match plan {
        P::Scan { .. } => false,
        P::Filter { input, predicate } => {
            over(input, &mut std::iter::once(predicate)) || below(input)
        }
        P::Project { input, exprs } => over(input, &mut exprs.iter()) || below(input),
        P::Aggregate {
            input,
            group_by,
            aggregates,
        } => {
            over(
                input,
                &mut group_by
                    .iter()
                    .chain(aggregates.iter().filter_map(|a| match a {
                        A::Count(e) | A::Sum(e) | A::Min(e) | A::Max(e) | A::Avg(e) => Some(e),
                        _ => None,
                    })),
            ) || below(input)
        }
        P::Sort { input, sort_exprs } => {
            over(input, &mut sort_exprs.iter().map(|s| &s.expr)) || below(input)
        }
        P::Window {
            input,
            partition_by,
            order_by,
            ..
        } => {
            over(
                input,
                &mut partition_by.iter().chain(order_by.iter().map(|s| &s.expr)),
            ) || below(input)
        }
        P::Limit { input, .. } | P::Distinct { input } => below(input),
        P::Union { inputs } => inputs.iter().any(below),
        P::SetOp { left, right, .. } => below(left) || below(right),
        P::Join {
            left,
            right,
            on,
            filter,
            ..
        } => {
            over(
                plan,
                &mut on.iter().flat_map(|(l, r)| [l, r]).chain(filter.iter()),
            ) || below(left)
                || below(right)
        }
    }
}

/// Whether `field` is an integer column stored in a wider type than its original DuckDB type
/// (RFC-008 metadata; W4-B-002): `TINYINT` / `SMALLINT` / `UTINYINT` / `USMALLINT` as Int32,
/// `UINTEGER` / `UBIGINT` / `HUGEINT` as Int64.
#[cfg(not(target_arch = "wasm32"))]
pub(super) fn widened_integer_original(
    field: &arrow_schema::Field,
) -> Option<crate::arrow_bridge::OriginalDuckDbType> {
    use crate::arrow_bridge::OriginalDuckDbType as O;
    crate::arrow_bridge::original_duckdb_type(field).filter(|original| {
        matches!(
            original,
            O::TinyInt
                | O::SmallInt
                | O::UTinyInt
                | O::USmallInt
                | O::UInteger
                | O::UBigInt
                | O::HugeInt
        ) && field.data_type() == &original.storage_arrow_type()
    })
}

/// Whether `plan` has a predicate craton-bolt 0.9 lowers to one `OR` term per matching
/// dictionary entry: `LIKE` / `ILIKE` over a column, or an ordering comparison between a column
/// and a string literal (`string_literal_rewrite::build_index_membership`, W2-B-001).
///
/// Works on the plan's `Debug` text, which covers every node and expression kind without
/// enumerating them; `Binary { op: Gt, left: Column("s"), right: Literal(Utf8("k")) }`.
#[cfg(not(target_arch = "wasm32"))]
pub(super) fn plan_has_dictionary_lowered_predicate(plan: &craton_bolt::plan::LogicalPlan) -> bool {
    let text = format!("{plan:?}");
    if text.contains("Like {") {
        return true;
    }
    /// Skip an operand `<prefix>"..."<suffix>` (Debug escapes inner quotes and backslashes),
    /// returning the text after it.
    pub(super) fn skip_quoted<'a>(s: &'a str, prefix: &str, suffix: &str) -> Option<&'a str> {
        let mut rest = s.strip_prefix(prefix)?;
        loop {
            let end = rest.find('"')?;
            let backslashes = rest[..end].chars().rev().take_while(|&c| c == '\\').count();
            rest = &rest[end + 1..];
            if backslashes % 2 == 0 {
                return rest.strip_prefix(suffix);
            }
        }
    }
    pub(super) const COLUMN: (&str, &str) = ("Column(\"", ")");
    pub(super) const STRING: (&str, &str) = ("Literal(Utf8(\"", "))");
    for op in ["Lt", "LtEq", "Gt", "GtEq"] {
        let marker = format!("op: {op}, left: ");
        let mut search = text.as_str();
        while let Some(pos) = search.find(&marker) {
            let operands = &search[pos + marker.len()..];
            for (left, right) in [(COLUMN, STRING), (STRING, COLUMN)] {
                if let Some(rest) = skip_quoted(operands, left.0, left.1)
                    && let Some(rest) = rest.strip_prefix(", right: ")
                    && skip_quoted(rest, right.0, right.1).is_some()
                {
                    return true;
                }
            }
            search = operands;
        }
    }
    false
}

/// Whether `plan` can produce NULLs from NULL-free inputs: outer joins, `CASE`, or a NULL
/// literal (W3-E-002). Works on the `Debug` text like
/// [`plan_has_dictionary_lowered_predicate`].
#[cfg(not(target_arch = "wasm32"))]
pub(super) fn plan_may_produce_nulls(plan: &craton_bolt::plan::LogicalPlan) -> bool {
    let text = format!("{plan:?}");
    [
        "LeftOuter",
        "RightOuter",
        "FullOuter",
        "Case {",
        "Literal(Null)",
    ]
    .iter()
    .any(|marker| text.contains(marker))
}

/// Convert a (normalized) Arrow schema to a Bolt planner schema.
#[cfg(not(target_arch = "wasm32"))]
pub(super) fn plan_schema_of(schema: &Schema) -> Result<craton_bolt::plan::Schema> {
    use craton_bolt::plan::{
        DataType as PlanDataType, Field as PlanField, Schema as PlanSchema,
        TimeUnit as PlanTimeUnit,
    };
    // A dictionary is planned as its value type (W1-E-014): `Dictionary(Int32, Int64)` is an
    // integer column, not text. (Catalog schemas are normalized, so this only matters for
    // callers that bypass `normalize_schema_for_bolt`.)
    pub(super) fn plan_dtype(dt: &ArrowDataType) -> Result<PlanDataType> {
        Ok(match dt {
            ArrowDataType::Boolean => PlanDataType::Bool,
            ArrowDataType::Int32 => PlanDataType::Int32,
            ArrowDataType::Int64 => PlanDataType::Int64,
            ArrowDataType::Float32 => PlanDataType::Float32,
            ArrowDataType::Float64 => PlanDataType::Float64,
            ArrowDataType::Utf8 | ArrowDataType::LargeUtf8 | ArrowDataType::Utf8View => {
                PlanDataType::Utf8
            }
            ArrowDataType::Dictionary(_, value)
                if !matches!(**value, ArrowDataType::Dictionary(..)) =>
            {
                plan_dtype(value)?
            }
            ArrowDataType::Date32 => PlanDataType::Date32,
            ArrowDataType::Timestamp(unit, tz) => {
                let plan_unit = match unit {
                    ArrowTimeUnit::Second => PlanTimeUnit::Second,
                    ArrowTimeUnit::Millisecond => PlanTimeUnit::Millisecond,
                    ArrowTimeUnit::Microsecond => PlanTimeUnit::Microsecond,
                    ArrowTimeUnit::Nanosecond => PlanTimeUnit::Nanosecond,
                };
                let plan_tz = match tz.as_deref() {
                    Some("UTC") | Some("utc") => Some("UTC"),
                    _ => None,
                };
                PlanDataType::Timestamp(plan_unit, plan_tz)
            }
            ArrowDataType::Decimal128(p, s) => PlanDataType::Decimal128(*p, *s),
            other => {
                return Err(Error::invalid_input(format!(
                    "Unsupported Arrow type in Bolt planner provider: {other:?}"
                )));
            }
        })
    }
    let mut fields = Vec::with_capacity(schema.fields().len());
    for f in schema.fields() {
        fields.push(PlanField::new(
            f.name(),
            plan_dtype(f.data_type())?,
            f.is_nullable(),
        ));
    }
    Ok(PlanSchema::new(fields))
}

// SPDX-License-Identifier: Apache-2.0

//! Per-column facts the GPU result guard derives through a Bolt logical plan (W9-A).
//!
//! The guard used to answer "may this column hold a NULL?" and "is this a widened integer?"
//! by looking a plan column's *name* up in the stored tables, so an alias, a UNION branch or a
//! NULL literal under a stored column's name hid a nullable value (W8-B-004). The facts here
//! are computed bottom-up instead: a `Scan` takes them from the stored table it names, and every
//! other node maps its input's facts through its expressions by position. They also bound the
//! row counts and integer magnitudes the SUM overflow check needs (W8-B-007), and answer "may
//! this input be empty?" structurally (W8-B-005).

use craton_bolt::plan::{
    AggregateExpr as A, BinaryOp as B, Expr as E, Literal as L, LogicalPlan as P, Schema as S,
    logical_plan::{JoinType, WindowFunc as W},
};

use super::*;

/// What the guard knows about the stored tables a plan scans.
pub(super) trait GuardCatalog {
    /// Schema, NULL count per column and row count of the stored table a `Scan` names.
    fn table(&self, name: &str) -> Option<(&Schema, &[usize], usize)>;
    /// Largest absolute value of integer column `index` of table `name`; `None` when unknown or
    /// not an integer column.
    fn int_abs_max(&self, name: &str, index: usize) -> Option<u64>;
}

/// What is known about one output column of a plan node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ColumnFact {
    /// The column may hold a NULL.
    pub(super) nullable: bool,
    /// The column is (or is computed from) an integer stored in a wider type than its
    /// original DuckDB type (W4-B-002).
    pub(super) narrow: bool,
    /// Upper bound of the absolute value of an integer column; `None` when unknown.
    pub(super) abs_max: Option<u64>,
}

impl ColumnFact {
    /// Nothing is known: may be NULL, may be narrow, unbounded.
    pub(super) const UNKNOWN: Self = Self {
        nullable: true,
        narrow: true,
        abs_max: None,
    };
    const COMPUTED: Self = Self {
        nullable: true,
        narrow: false,
        abs_max: None,
    };

    fn not_null(abs_max: Option<u64>) -> Self {
        Self {
            nullable: false,
            narrow: false,
            abs_max,
        }
    }

    /// The fact for a column that is either `self` or `other` (UNION, set operations).
    fn merge(self, other: Self) -> Self {
        Self {
            nullable: self.nullable || other.nullable,
            narrow: self.narrow || other.narrow,
            abs_max: self.abs_max.zip(other.abs_max).map(|(a, b)| a.max(b)),
        }
    }
}

/// A node's output schema with its column facts (same order).
pub(super) struct NodeFacts {
    schema: Option<S>,
    columns: Vec<ColumnFact>,
}

impl NodeFacts {
    /// The facts of `plan`'s output columns.
    pub(super) fn of(plan: &P, cat: &dyn GuardCatalog) -> Self {
        Self {
            schema: plan.schema().ok(),
            columns: column_facts(plan, cat),
        }
    }

    /// The type of expression `e` over this node's rows (`None`: the planner cannot derive it).
    pub(super) fn dtype(&self, e: &E) -> Option<craton_bolt::plan::DataType> {
        self.schema.as_ref().and_then(|s| e.dtype(s).ok())
    }

    /// The fact of the column `name` resolves to (unknown when it does not resolve).
    fn column(&self, name: &str) -> ColumnFact {
        self.schema
            .as_ref()
            .and_then(|s| s.index_of(name).ok())
            .and_then(|i| self.columns.get(i).copied())
            .unwrap_or(ColumnFact::UNKNOWN)
    }

    /// The fact of expression `e` evaluated over this node's rows.
    pub(super) fn expr(&self, e: &E) -> ColumnFact {
        match e {
            E::Column(name) => self.column(name),
            E::Literal(L::Null) => ColumnFact {
                nullable: true,
                narrow: false,
                abs_max: Some(0),
            },
            E::Literal(L::Int32(v)) => ColumnFact::not_null(Some(u64::from(v.unsigned_abs()))),
            E::Literal(L::Int64(v)) => ColumnFact::not_null(Some(v.unsigned_abs())),
            E::Literal(_) => ColumnFact::not_null(None),
            E::Alias(inner, _) => self.expr(inner),
            E::Cast {
                expr: inner,
                target,
                ..
            } => {
                let f = self.expr(inner);
                ColumnFact {
                    nullable: true,
                    narrow: f.narrow,
                    abs_max: f.abs_max.filter(|_| {
                        matches!(
                            target,
                            craton_bolt::plan::DataType::Int32 | craton_bolt::plan::DataType::Int64
                        )
                    }),
                }
            }
            E::Binary { op, left, right } => {
                let (l, r) = (self.expr(left), self.expr(right));
                let abs_max = match op {
                    B::Add | B::Sub => l.abs_max.zip(r.abs_max).map(|(a, b)| a.saturating_add(b)),
                    B::Mul => l.abs_max.zip(r.abs_max).map(|(a, b)| a.saturating_mul(b)),
                    _ => None,
                };
                ColumnFact {
                    nullable: true,
                    narrow: l.narrow || r.narrow,
                    abs_max,
                }
            }
            E::Case {
                branches,
                else_branch,
            } => {
                let mut narrow = false;
                let mut abs_max = Some(0u64);
                for (w, t) in branches {
                    let (w, t) = (self.expr(w), self.expr(t));
                    narrow |= w.narrow || t.narrow;
                    abs_max = abs_max.zip(t.abs_max).map(|(a, b)| a.max(b));
                }
                if let Some(x) = else_branch {
                    let x = self.expr(x);
                    narrow |= x.narrow;
                    abs_max = abs_max.zip(x.abs_max).map(|(a, b)| a.max(b));
                }
                ColumnFact {
                    nullable: true,
                    narrow,
                    abs_max,
                }
            }
            E::Unary { operand: x, .. }
            | E::Like { expr: x, .. }
            | E::CastFormat { expr: x, .. }
            | E::Extract { expr: x, .. }
            | E::DateTrunc { expr: x, .. }
            | E::InSubquery { expr: x, .. } => ColumnFact {
                narrow: self.expr(x).narrow,
                ..ColumnFact::COMPUTED
            },
            E::ScalarFn { args, .. } => ColumnFact {
                narrow: args.iter().any(|a| self.expr(a).narrow),
                ..ColumnFact::COMPUTED
            },
            E::ScalarSubquery(_) => ColumnFact::COMPUTED,
        }
    }
}

/// The facts of `plan`'s output columns, by position.
fn column_facts(plan: &P, cat: &dyn GuardCatalog) -> Vec<ColumnFact> {
    match plan {
        P::Scan {
            table,
            projection,
            schema,
        } => {
            let names: Vec<&str> = match projection {
                Some(cols) => cols.iter().map(String::as_str).collect(),
                None => schema.fields.iter().map(|f| f.name.as_str()).collect(),
            };
            let Some((stored, nulls, _)) = cat.table(table) else {
                return vec![ColumnFact::UNKNOWN; names.len()];
            };
            names
                .iter()
                .map(|name| {
                    let fields = stored.fields();
                    let index = fields.iter().position(|f| f.name() == name).or_else(|| {
                        fields
                            .iter()
                            .position(|f| f.name().eq_ignore_ascii_case(name))
                    });
                    match index {
                        Some(i) => ColumnFact {
                            nullable: nulls.get(i).is_none_or(|&n| n > 0),
                            narrow: widened_integer_original(&fields[i]).is_some(),
                            abs_max: cat.int_abs_max(table, i),
                        },
                        None => ColumnFact::UNKNOWN,
                    }
                })
                .collect()
        }
        P::Filter { input, .. }
        | P::Sort { input, .. }
        | P::Limit { input, .. }
        | P::Distinct { input } => column_facts(input, cat),
        P::Project { input, exprs } => {
            let facts = NodeFacts::of(input, cat);
            exprs.iter().map(|e| facts.expr(e)).collect()
        }
        P::Aggregate {
            input,
            group_by,
            aggregates,
        } => {
            let facts = NodeFacts::of(input, cat);
            let rows = row_bound(input, cat);
            // An ungrouped aggregate of no rows is NULL (COUNT: 0).
            let empty = group_by.is_empty() && may_be_empty(input, cat);
            group_by
                .iter()
                .map(|g| facts.expr(g))
                .chain(aggregates.iter().map(|a| match a {
                    A::Count(_) => ColumnFact::not_null(rows),
                    A::Sum(e) => {
                        let f = facts.expr(e);
                        ColumnFact {
                            nullable: f.nullable || empty,
                            narrow: false,
                            abs_max: f.abs_max.zip(rows).map(|(a, n)| a.saturating_mul(n)),
                        }
                    }
                    A::Min(e) | A::Max(e) => {
                        let f = facts.expr(e);
                        ColumnFact {
                            nullable: f.nullable || empty,
                            ..f
                        }
                    }
                    A::Avg(e) => ColumnFact {
                        nullable: facts.expr(e).nullable || empty,
                        narrow: false,
                        abs_max: None,
                    },
                    A::VarPop(_) | A::VarSamp(_) | A::StddevPop(_) | A::StddevSamp(_) => {
                        ColumnFact::COMPUTED
                    }
                }))
                .collect()
        }
        P::Window {
            input,
            window_exprs,
            ..
        } => {
            let facts = NodeFacts::of(input, cat);
            let rows = row_bound(input, cat);
            let mut columns = facts.columns.clone();
            columns.extend(window_exprs.iter().map(|w| match &w.func {
                W::RowNumber | W::Rank | W::DenseRank | W::Count(_) => ColumnFact::not_null(rows),
                W::Sum(e) => {
                    let f = facts.expr(e);
                    ColumnFact {
                        nullable: f.nullable,
                        narrow: false,
                        abs_max: f.abs_max.zip(rows).map(|(a, n)| a.saturating_mul(n)),
                    }
                }
                W::Min(e) | W::Max(e) => facts.expr(e),
                W::Avg(e) => ColumnFact {
                    nullable: facts.expr(e).nullable,
                    narrow: false,
                    abs_max: None,
                },
            }));
            columns
        }
        P::Union { inputs } => merge_by_position(inputs.iter().map(|p| column_facts(p, cat))),
        P::SetOp { left, right, .. } => {
            merge_by_position([column_facts(left, cat), column_facts(right, cat)].into_iter())
        }
        P::Join {
            left,
            right,
            join_type,
            ..
        } => {
            let pad = |mut facts: Vec<ColumnFact>, padded: bool| {
                if padded {
                    facts.iter_mut().for_each(|f| f.nullable = true);
                }
                facts
            };
            let mut columns = pad(
                column_facts(left, cat),
                matches!(join_type, JoinType::RightOuter | JoinType::FullOuter),
            );
            columns.extend(pad(
                column_facts(right, cat),
                matches!(join_type, JoinType::LeftOuter | JoinType::FullOuter),
            ));
            columns
        }
    }
}

fn merge_by_position(mut branches: impl Iterator<Item = Vec<ColumnFact>>) -> Vec<ColumnFact> {
    let mut merged = branches.next().unwrap_or_default();
    for branch in branches {
        if branch.len() != merged.len() {
            return vec![ColumnFact::UNKNOWN; merged.len()];
        }
        for (m, b) in merged.iter_mut().zip(branch) {
            *m = m.merge(b);
        }
    }
    merged
}

/// Upper bound of the number of rows `plan` produces (`None`: unknown).
pub(super) fn row_bound(plan: &P, cat: &dyn GuardCatalog) -> Option<u64> {
    match plan {
        P::Scan { table, .. } => cat.table(table).map(|(_, _, rows)| rows as u64),
        P::Filter { input, .. }
        | P::Project { input, .. }
        | P::Sort { input, .. }
        | P::Distinct { input }
        | P::Window { input, .. } => row_bound(input, cat),
        P::Limit { input, limit, .. } => {
            Some(row_bound(input, cat).map_or(*limit as u64, |n| n.min(*limit as u64)))
        }
        P::Aggregate {
            input, group_by, ..
        } => {
            if group_by.is_empty() {
                Some(1)
            } else {
                row_bound(input, cat)
            }
        }
        P::Union { inputs } => inputs.iter().try_fold(0u64, |total, p| {
            Some(total.saturating_add(row_bound(p, cat)?))
        }),
        // EXCEPT / INTERSECT keep at most the left input's rows.
        P::SetOp { left, .. } => row_bound(left, cat),
        P::Join { left, right, .. } => {
            Some(row_bound(left, cat)?.saturating_mul(row_bound(right, cat)?))
        }
    }
}

/// Whether `plan` may produce no rows (W8-B-005): a filter, a join, `EXCEPT` / `INTERSECT`, a
/// `LIMIT 0` or any `OFFSET`, or a scan of an empty (or unknown) table below it.
pub(super) fn may_be_empty(plan: &P, cat: &dyn GuardCatalog) -> bool {
    match plan {
        P::Scan { table, .. } => cat.table(table).is_none_or(|(_, _, rows)| rows == 0),
        P::Filter { .. } | P::Join { .. } | P::SetOp { .. } => true,
        P::Limit {
            input,
            limit,
            offset,
        } => *limit == 0 || *offset > 0 || may_be_empty(input, cat),
        P::Project { input, .. }
        | P::Sort { input, .. }
        | P::Distinct { input }
        | P::Window { input, .. } => may_be_empty(input, cat),
        P::Aggregate {
            input, group_by, ..
        } => !group_by.is_empty() && may_be_empty(input, cat),
        P::Union { inputs } => inputs.iter().all(|p| may_be_empty(p, cat)),
    }
}

/// Largest absolute value per column of `batches` for Int32 / Int64 columns (`None` for other
/// columns). Values under NULL slots count too: that can only raise the bound.
pub(super) fn int_abs_max_per_column(
    num_columns: usize,
    batches: &[RecordBatch],
) -> Vec<Option<u64>> {
    use arrow_array::{
        cast::AsArray,
        types::{Int32Type, Int64Type},
    };
    (0..num_columns)
        .map(|index| {
            let mut max = 0u64;
            for batch in batches {
                let col = batch.column(index);
                match col.data_type() {
                    ArrowDataType::Int32 => {
                        for v in col.as_primitive::<Int32Type>().values().iter() {
                            max = max.max(u64::from(v.unsigned_abs()));
                        }
                    }
                    ArrowDataType::Int64 => {
                        for v in col.as_primitive::<Int64Type>().values().iter() {
                            max = max.max(v.unsigned_abs());
                        }
                    }
                    _ => return None,
                }
            }
            Some(max)
        })
        .collect()
}

/// The guard's view of the catalog, plus the engine-only scratch table of a scratch query.
pub(super) struct GuardTables<'a> {
    catalog: &'a BTreeMap<String, RegisteredTable>,
    scratch: Option<Scratch<'a>>,
}

struct Scratch<'a> {
    name: &'a str,
    batch: &'a RecordBatch,
    column_nulls: Vec<usize>,
    int_abs_max: OnceLock<Vec<Option<u64>>>,
}

impl<'a> GuardTables<'a> {
    pub(super) fn new(
        catalog: &'a BTreeMap<String, RegisteredTable>,
        scratch: Option<(&'a str, &'a RecordBatch)>,
    ) -> Self {
        Self {
            catalog,
            scratch: scratch.map(|(name, batch)| Scratch {
                name,
                batch,
                column_nulls: batch_column_nulls(batch),
                int_abs_max: OnceLock::new(),
            }),
        }
    }

    fn scratch(&self, name: &str) -> Option<&Scratch<'a>> {
        self.scratch
            .as_ref()
            .filter(|s| s.name.eq_ignore_ascii_case(name))
    }

    fn entry(&self, name: &str) -> Option<&'a RegisteredTable> {
        self.catalog.get(name).or_else(|| {
            self.catalog
                .iter()
                .find(|(key, _)| key.eq_ignore_ascii_case(name))
                .map(|(_, t)| t)
        })
    }

    /// Whether the scratch table holds a NULL.
    pub(super) fn scratch_has_nulls(&self) -> bool {
        self.scratch
            .as_ref()
            .is_some_and(|s| s.column_nulls.iter().any(|&n| n > 0))
    }
}

impl GuardCatalog for GuardTables<'_> {
    fn table(&self, name: &str) -> Option<(&Schema, &[usize], usize)> {
        if let Some(s) = self.scratch(name) {
            return Some((
                s.batch.schema_ref().as_ref(),
                s.column_nulls.as_slice(),
                s.batch.num_rows(),
            ));
        }
        self.entry(name)
            .map(|t| (t.schema.as_ref(), t.column_nulls.as_slice(), t.num_rows))
    }

    fn int_abs_max(&self, name: &str, index: usize) -> Option<u64> {
        if let Some(s) = self.scratch(name) {
            let bounds = s.int_abs_max.get_or_init(|| {
                int_abs_max_per_column(s.batch.num_columns(), std::slice::from_ref(s.batch))
            });
            return bounds.get(index).copied().flatten();
        }
        self.entry(name)?
            .int_abs_max()
            .get(index)
            .copied()
            .flatten()
    }
}

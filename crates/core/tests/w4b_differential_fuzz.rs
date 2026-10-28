// SPDX-License-Identifier: Apache-2.0

//! Randomized differential fuzzer (W4-B): `bolt_scan`, the replacement scan and `bolt_query`
//! against plain DuckDB.
//!
//! A deterministic, seeded generator (a small xorshift PRNG; no extra dependencies) builds
//! tables with random column types, random NULL density (none, so the GPU engine evaluates
//! pushed SQL, and some, so the host fallback does), special values (NaN, infinities, integer
//! extremes) in some tables, and one to three appended batches. For each table it runs random
//! queries: filters (comparisons, BETWEEN, IN / NOT IN, IS [NOT] NULL, LIKE, AND / OR / NOT,
//! CAST, arithmetic), projections, `count(*)`, GROUP BY aggregates and ORDER BY + LIMIT.
//!
//! W9-B (W8-B-010) adds shapes aimed at the GPU crash guard: integer `/`, `//` and `%` with
//! zero divisors and fractional literals next to integer columns (W8-B-001 / -002);
//! host-orchestrated wrappers (`DISTINCT ON`, correlated `EXISTS` and scalar subqueries,
//! `LATERAL`, `WITH RECURSIVE`, `VALUES`, `generate_series`, grouped `count(DISTINCT)`,
//! W8-B-003); renames over nullable columns (aliases, column swaps, `ORDER BY` an alias,
//! `UNION ALL` branches with other names or a NULL literal, over tables whose NULLs are drawn
//! per group so that all-NULL groups exist, W8-B-004); aggregates over inputs that are empty
//! without a filter (`LIMIT 0`, `OFFSET` past the end, `EXCEPT`, `INTERSECT`, W8-B-005);
//! `stddev_pop` / `stddev_samp` / `var_pop` / `var_samp` (W8-B-006); tables with big values
//! (FLOAT sums beyond 2^24, BIGINT sums beyond `i64`, W8-B-007); and temporal arithmetic
//! (`ts - ts`, `date - date`, `date_trunc`, `extract`, W8-B-008).
//!
//! W11-A (W8-B-010 items 10 and 11) adds `CASE` with and without `ELSE`, `coalesce`,
//! `nullif`, string functions with edge arguments (`substring` from 0 or a negative start,
//! `lpad` / `rpad` shorter than the input, `upper` / `lower` / `concat` with non-ASCII text,
//! `length`, `strpos` of the empty string), `GROUP BY ... HAVING`, and window functions
//! (`row_number`, `rank`, `dense_rank`, running `sum` / `count` / `min` / `max` / `avg` with ties
//! and nullable keys); and a routing oracle (`w4b_safe_shapes_stay_on_the_gpu`) that requires a
//! fixed list of shapes that are correct on the engine to stay there. Large floating-point
//! results are compared at 12 significant digits (6 decimals below 1e6), and an evaluation
//! error that only DuckDB's evaluation order decides is not a `bolt_query` failure.
//!
//! Every query runs against the DuckDB source table and against the Bolt copy through
//! `bolt_scan('<t>')` and the replacement scan (`<t>`); the results must be equal as multisets
//! (`EXCEPT ALL` in both directions). The same query is also sent to the engine through
//! `bolt_query`, except where its documented semantics differ (see `bolt_query_eligible`): a
//! Bolt error there is a skip, a wrong result is a failure.
//!
//! Route-consistency oracle (W8-B-010 item 1): for a table without NULLs (whose `bolt_query`
//! SQL may run on the GPU), a twin Bolt table holds the same rows plus one extra column with a
//! NULL, which the query does not use (so the guard routes it to the host fallback). The query
//! runs through `bolt_query` on both; the output types must be equal, and the values must be
//! equal exactly (floating-point columns rounded to 6 decimals, as everywhere). A difference
//! means the result depends on the route. The types of the GPU route are also compared with
//! DuckDB's own result types; those differences are counted and printed (`bolt_query` returns
//! the engine's types by design), and fail the test only with `BOLT_FUZZ_STRICT_TYPES`.
//!
//! Environment:
//! - `BOLT_FUZZ_CASES`: number of queries (default [`DEFAULT_CASES`]).
//! - `BOLT_FUZZ_SEED`: base seed (default fixed, so runs are reproducible).
//! - `BOLT_FUZZ_VERBOSE`: print every failing case before minimizing.
//! - `BOLT_FUZZ_TABLE_SEEDS`: comma-separated table seeds (as printed in reports) to replay.
//! - `BOLT_FUZZ_ROUTE_EVERY`: run the route-consistency oracle on every n-th query of each
//!   eligible table (default [`DEFAULT_ROUTE_EVERY`]; `1`: every query; `0`: off).
//! - `BOLT_FUZZ_STRICT_TYPES`: a `bolt_query` result type that differs from DuckDB's (found by
//!   the route oracle) fails the test instead of only being counted.
//! - `BOLT_FUZZ_WIDE_LIKE`: also generate some tables of 6,000 rows with a VARCHAR column of
//!   6,000 distinct values (above the guard's 4,096-value dictionary limit) for LIKE predicates
//!   inside host-orchestrated wrappers (W8-B-003). Opt-in: a failure there can abort the
//!   process (craton-bolt-007).
//!
//! Failures that are known and tracked live in [`known_issue`], each with the page that
//! tracks it; anything else fails the test with a minimized reproduction.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::sync::{Mutex, Once};
use std::time::Instant;

use duckdb_bolt_core::runtime::BoltRuntime;
use duckdb_loadable_macros::duckdb_entrypoint_c_api_v2;
use duckdb_neo::{
    Parameters, Result,
    connection::Connection,
    environment::{Environment, StorageLocation},
    extension::{Extension, register_static_extension},
};

#[duckdb_entrypoint_c_api_v2(extension_name = "bolt", api_version = "2.0.0")]
fn bolt_test_entrypoint(extension: &Extension) -> Result<()> {
    duckdb_bolt_core::register(extension)
}

static REGISTER_ONCE: Once = Once::new();
static TEST_MUTEX: Mutex<()> = Mutex::new(());

/// Default number of queries per run (tables are generated every [`QUERIES_PER_TABLE`]).
const DEFAULT_CASES: usize = 2400;
const QUERIES_PER_TABLE: usize = 12;
const DEFAULT_SEED: u64 = 0x5EED_B017_F022_0004;

/// Default period of the route-consistency oracle (every n-th query of an eligible table).
const DEFAULT_ROUTE_EVERY: usize = 3;

const SRC: &str = "fz_src";
const BOLT: &str = "fz_b";
/// The Bolt twin of [`BOLT`] with one extra, partly NULL column (route-consistency oracle).
/// Not a prefix of [`BOLT`] or the reverse, so name-based table lookups cannot confuse them.
const TWIN: &str = "fz_twin";

/// Statistical aggregates (W8-B-006).
const STAT_AGGS: [&str; 4] = ["stddev_pop", "stddev_samp", "var_pop", "var_samp"];

fn setup_connection() -> Result<(Environment, Connection)> {
    REGISTER_ONCE.call_once(|| {
        // SAFETY: `duckdb_extension_bolt_describe` is emitted by `#[duckdb_entrypoint_c_api_v2]`,
        // which fills the descriptor as `duckdb_static_extension.h` specifies (the contract of
        // `register_static_extension`); `Once` makes this a single call.
        unsafe { register_static_extension(duckdb_extension_bolt_describe) }
            .expect("Failed to register static bolt extension");
    });
    let env = Environment::new()?;
    let instance = env.open(StorageLocation::InMemory)?;
    let conn = instance.connect()?;
    conn.execute("PRAGMA bolt_clear_tables", Parameters::None)?;
    Ok((env, conn))
}

fn gpu() -> bool {
    BoltRuntime::global().telemetry().unwrap().backend == "cuda-gpu"
}

fn exec(conn: &Connection, sql: &str) -> Result<()> {
    conn.execute(sql, Parameters::None)?;
    Ok(())
}

fn query_i64(conn: &Connection, sql: &str) -> Result<i64> {
    let mut result = conn.query(sql, Parameters::None)?;
    let chunk = result.next_chunk()?.expect("Expected at least one chunk");
    let vec = chunk.get_vector_at::<i64>(0)?;
    Ok(*vec.reader()?.get(0)?.expect("Expected non-null i64"))
}

fn query_string(conn: &Connection, sql: &str) -> Result<String> {
    let mut result = conn.query(sql, Parameters::None)?;
    let chunk = result.next_chunk()?.expect("Expected at least one chunk");
    let vec = chunk.get_vector_at::<String>(0)?;
    Ok(vec.reader()?.get(0)?.expect("non-null").to_string())
}

// ---------------------------------------------------------------------------------------------
// PRNG
// ---------------------------------------------------------------------------------------------

/// xorshift64* (Vigna); deterministic across platforms.
#[derive(Clone)]
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        // splitmix64 scramble so nearby seeds diverge immediately; never zero.
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        Self((z ^ (z >> 31)) | 1)
    }
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n.max(1)
    }
    fn range(&mut self, lo: i64, hi: i64) -> i64 {
        lo + self.below((hi - lo + 1) as u64) as i64
    }
    fn chance(&mut self, percent: u64) -> bool {
        self.below(100) < percent
    }
    fn pick<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        &items[self.below(items.len() as u64) as usize]
    }
}

// ---------------------------------------------------------------------------------------------
// Schema and data
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Ty {
    TinyInt,
    SmallInt,
    Integer,
    BigInt,
    HugeInt,
    UTinyInt,
    USmallInt,
    UInteger,
    UBigInt,
    Float,
    Double,
    Dec9,
    Dec18,
    Varchar,
    Date,
    Timestamp,
    TimestampMs,
    TimestampTz,
    Bool,
}

const ALL_TYPES: [Ty; 19] = [
    Ty::TinyInt,
    Ty::SmallInt,
    Ty::Integer,
    Ty::BigInt,
    Ty::HugeInt,
    Ty::UTinyInt,
    Ty::USmallInt,
    Ty::UInteger,
    Ty::UBigInt,
    Ty::Float,
    Ty::Double,
    Ty::Dec9,
    Ty::Dec18,
    Ty::Varchar,
    Ty::Date,
    Ty::Timestamp,
    Ty::TimestampMs,
    Ty::TimestampTz,
    Ty::Bool,
];

/// Comparison class of an expression.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Int,
    Float,
    Dec,
    Str,
    Date,
    Ts,
    Tstz,
    Bool,
}

impl Ty {
    fn sql(self) -> &'static str {
        match self {
            Ty::TinyInt => "TINYINT",
            Ty::SmallInt => "SMALLINT",
            Ty::Integer => "INTEGER",
            Ty::BigInt => "BIGINT",
            Ty::HugeInt => "HUGEINT",
            Ty::UTinyInt => "UTINYINT",
            Ty::USmallInt => "USMALLINT",
            Ty::UInteger => "UINTEGER",
            Ty::UBigInt => "UBIGINT",
            Ty::Float => "FLOAT",
            Ty::Double => "DOUBLE",
            Ty::Dec9 => "DECIMAL(9, 2)",
            Ty::Dec18 => "DECIMAL(18, 4)",
            Ty::Varchar => "VARCHAR",
            Ty::Date => "DATE",
            Ty::Timestamp => "TIMESTAMP",
            Ty::TimestampMs => "TIMESTAMP_MS",
            Ty::TimestampTz => "TIMESTAMPTZ",
            Ty::Bool => "BOOLEAN",
        }
    }
    fn kind(self) -> Kind {
        match self {
            Ty::TinyInt
            | Ty::SmallInt
            | Ty::Integer
            | Ty::BigInt
            | Ty::HugeInt
            | Ty::UTinyInt
            | Ty::USmallInt
            | Ty::UInteger
            | Ty::UBigInt => Kind::Int,
            Ty::Float | Ty::Double => Kind::Float,
            Ty::Dec9 | Ty::Dec18 => Kind::Dec,
            Ty::Varchar => Kind::Str,
            Ty::Date => Kind::Date,
            Ty::Timestamp | Ty::TimestampMs => Kind::Ts,
            Ty::TimestampTz => Kind::Tstz,
            Ty::Bool => Kind::Bool,
        }
    }
    fn unsigned(self) -> bool {
        matches!(
            self,
            Ty::UTinyInt | Ty::USmallInt | Ty::UInteger | Ty::UBigInt
        )
    }
    /// `(min, max)` literal text for integer types (UBIGINT / HUGEINT: the i64 range Bolt stores).
    fn extremes(self) -> (&'static str, &'static str) {
        match self {
            Ty::TinyInt => ("-128", "127"),
            Ty::SmallInt => ("-32768", "32767"),
            Ty::Integer => ("-2147483648", "2147483647"),
            Ty::BigInt | Ty::HugeInt => ("-9223372036854775808", "9223372036854775807"),
            Ty::UTinyInt => ("0", "255"),
            Ty::USmallInt => ("0", "65535"),
            Ty::UInteger => ("0", "4294967295"),
            Ty::UBigInt => ("0", "9223372036854775807"),
            _ => ("0", "0"),
        }
    }
}

const STRINGS: [&str; 12] = [
    "", "a", "ab", "abc", "b", "ba", "B", "x_1", "x%2", "ñu", "zz", "a b",
];

#[derive(Clone, Debug)]
struct Column {
    name: String,
    ty: Ty,
    /// Percentage of NULLs (0 ..= 100).
    null_pct: u64,
    /// High-cardinality strings (above the engine's dictionary pushdown limit).
    wide_strings: bool,
    /// With [`Table::groups`]: NULLs are drawn per group (`i % groups`), not per row, so a
    /// GROUP BY on a per-group integer column (whose value is the group) has all-NULL groups
    /// (W8-B-004). A per-group integer column holds `i % groups`.
    per_group: bool,
}

#[derive(Clone, Debug)]
struct Table {
    columns: Vec<Column>,
    rows: usize,
    /// Row boundaries of the appended batches.
    splits: Vec<usize>,
    /// NaN / infinities / -0.0 in float columns and type extremes in integer columns.
    special: bool,
    /// Big values (W8-B-007): FLOAT / DOUBLE columns near 1e5 (a FLOAT sum of a few hundred
    /// rows is beyond 2^24) and BIGINT columns near 4e18 (a sum of three rows overflows i64).
    big: bool,
    /// Number of groups for [`Column::per_group`] columns (0: none).
    groups: u64,
    /// Distinct values of a wide-string column (`BOLT_FUZZ_WIDE_LIKE` raises it above 4,096).
    wide_card: u64,
    time_zone: &'static str,
}

impl Table {
    fn has_nulls(&self) -> bool {
        self.rows > 0 && self.columns.iter().any(|c| c.null_pct > 0)
    }
    fn column(&self, idx: usize) -> &Column {
        &self.columns[idx]
    }
}

/// `BOLT_FUZZ_WIDE_LIKE`: generate some tables above the dictionary limit (see the header).
fn wide_like_enabled() -> bool {
    std::env::var_os("BOLT_FUZZ_WIDE_LIKE").is_some()
}

fn gen_table(rng: &mut Rng) -> Table {
    let ncols = rng.range(1, 5) as usize;
    let nulls = rng.chance(50);
    let special = rng.chance(20);
    let big = !special && rng.chance(8);
    let groups = if rng.chance(30) {
        rng.range(2, 7) as u64
    } else {
        0
    };
    let mut columns = vec![Column {
        name: "id".into(),
        ty: Ty::Integer,
        null_pct: 0,
        wide_strings: false,
        per_group: false,
    }];
    for i in 0..ncols {
        let ty = *rng.pick(&ALL_TYPES);
        let null_pct = if nulls {
            *rng.pick(&[0, 5, 5, 20, 50, 100])
        } else {
            0
        };
        columns.push(Column {
            name: format!("c{i}"),
            ty,
            null_pct,
            wide_strings: rng.chance(25),
            per_group: groups > 0 && rng.chance(50),
        });
    }
    let mut rows = *rng.pick(&[0, 1, 7, 50, 200, 200, 600, 600, 2047, 2049, 3000]);
    if groups > 0 {
        // The group key itself (`i % groups`), so per-group NULLs give all-NULL groups.
        columns.push(Column {
            name: "grp".into(),
            ty: Ty::Integer,
            null_pct: 0,
            wide_strings: false,
            per_group: true,
        });
    }
    let mut wide_card = 300;
    if wide_like_enabled() && rng.chance(30) {
        columns.push(Column {
            name: "cw".into(),
            ty: Ty::Varchar,
            null_pct: 0,
            wide_strings: true,
            per_group: false,
        });
        rows = 6000;
        wide_card = 6000;
    }
    let mut splits = vec![0];
    let parts = rng.range(1, 3) as usize;
    for _ in 1..parts {
        splits.push(rng.range(0, rows as i64) as usize);
    }
    splits.push(rows);
    splits.sort_unstable();
    let time_zone = *rng.pick(&["UTC", "UTC", "America/New_York", "Asia/Kolkata"]);
    Table {
        columns,
        rows,
        splits,
        special,
        big,
        groups,
        wide_card,
        time_zone,
    }
}

/// DuckDB expression producing column `col` (index `idx`) of table `t` for row `i`.
fn column_expr(col: &Column, idx: usize, t: &Table) -> String {
    let special = t.special;
    let grouped = col.per_group && t.groups > 0;
    let h = format!("hash(i, {})", idx * 7919 + 13);
    let value = match col.ty.kind() {
        Kind::Int if grouped => format!("CAST(i % {} AS {})", t.groups, col.ty.sql()),
        Kind::Int => {
            let narrow = if col.ty.unsigned() {
                format!("({h} % 41)")
            } else {
                format!("(({h} % 41)::BIGINT - 20)")
            };
            let (lo, hi) = col.ty.extremes();
            let v = if special {
                format!(
                    "CASE WHEN {h} % 37 = 0 THEN {hi} WHEN {h} % 37 = 1 THEN {lo} ELSE {narrow} END"
                )
            } else if t.big && col.ty == Ty::BigInt {
                format!("(4000000000000000000 + {narrow})")
            } else {
                narrow
            };
            format!("CAST({v} AS {})", col.ty.sql())
        }
        Kind::Float => {
            let narrow = format!("((({h} % 321)::BIGINT - 160) / 8.0)");
            let v = if special {
                format!(
                    "CASE {h} % 23 WHEN 0 THEN 'NaN'::DOUBLE WHEN 1 THEN 'inf'::DOUBLE \
                     WHEN 2 THEN '-inf'::DOUBLE WHEN 3 THEN -0.0::DOUBLE ELSE {narrow} END"
                )
            } else if t.big {
                // Multiples of 1/8 near 1e5: exact in FLOAT, and exact in a DOUBLE sum.
                format!("(100000 + {narrow})")
            } else {
                narrow
            };
            format!("CAST({v} AS {})", col.ty.sql())
        }
        Kind::Dec => {
            let (modulus, divisor) = if col.ty == Ty::Dec9 {
                (4001, "100.0")
            } else {
                (40001, "1000.0")
            };
            format!(
                "CAST((({h} % {modulus})::BIGINT - {}) / {divisor} AS {})",
                modulus / 2,
                col.ty.sql()
            )
        }
        Kind::Str => {
            if col.wide_strings {
                format!("('s' || ({h} % {})::VARCHAR)", t.wide_card)
            } else {
                let list = STRINGS
                    .iter()
                    .map(|s| format!("'{s}'"))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("([{list}])[({h} % {})::INTEGER + 1]", STRINGS.len())
            }
        }
        Kind::Date => format!("(DATE '1970-01-01' + (({h} % 2001)::INTEGER - 1000))"),
        Kind::Ts | Kind::Tstz => {
            let ts = format!(
                "(TIMESTAMP '1970-01-01 00:00:00' + to_seconds((({h} % 20001)::BIGINT - 10000) * 4111) \
                 + to_microseconds((({h} // 7) % 4)::BIGINT * 250000))"
            );
            format!("CAST({ts} AS {})", col.ty.sql())
        }
        Kind::Bool => format!("({h} % 3 = 0)"),
    };
    if col.null_pct == 0 {
        value
    } else {
        let row = if grouped {
            format!("i % {}", t.groups)
        } else {
            "i".to_string()
        };
        format!(
            "CASE WHEN hash({row}, {}) % 100 < {} THEN NULL ELSE {value} END",
            idx * 104729 + 7,
            col.null_pct
        )
    }
}

/// Create the DuckDB source table and its Bolt copy (one COPY per appended batch).
fn build_table(conn: &Connection, table: &Table) -> Result<()> {
    exec(conn, &format!("SET TimeZone = '{}'", table.time_zone))?;
    let select = table
        .columns
        .iter()
        .enumerate()
        .map(|(idx, c)| {
            if idx == 0 {
                "CAST(i AS INTEGER) AS id".to_string()
            } else {
                format!("{} AS {}", column_expr(c, idx, table), c.name)
            }
        })
        .collect::<Vec<_>>()
        .join(", ");
    exec(
        conn,
        &format!(
            "CREATE OR REPLACE TABLE {SRC} AS SELECT {select} FROM range(0, {}) t(i)",
            table.rows
        ),
    )?;
    for (part, bounds) in table.splits.windows(2).enumerate() {
        let mode = if part == 0 { "replace" } else { "append" };
        exec(
            conn,
            &format!(
                "COPY (SELECT * FROM {SRC} WHERE id >= {} AND id < {} ORDER BY id) TO '{BOLT}' \
                 (FORMAT bolt, USE_TMP_FILE FALSE, MODE '{mode}')",
                bounds[0], bounds[1]
            ),
        )?;
    }
    Ok(())
}

/// The route-consistency twin of the current table: the same rows plus a column `zn` that is
/// NULL in row 0 and that no query references.
fn build_twin(conn: &Connection) -> Result<()> {
    exec(
        conn,
        &format!(
            "COPY (SELECT *, CAST(CASE WHEN id = 0 THEN NULL ELSE 0 END AS INTEGER) AS zn \
             FROM {SRC} ORDER BY id) TO '{TWIN}' (FORMAT bolt, USE_TMP_FILE FALSE, MODE 'replace')"
        ),
    )
}

// ---------------------------------------------------------------------------------------------
// Expressions and predicates
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Debug)]
enum Expr {
    Col(usize),
    Lit(String),
    Cast(Box<Expr>, &'static str),
    Arith(Box<Expr>, &'static str, Box<Expr>),
    /// `name(args)` (W8-B-008: `date_trunc`).
    Func(&'static str, Vec<Expr>),
    /// `extract(part FROM e)` (W8-B-008).
    Extract(&'static str, Box<Expr>),
    /// `CASE WHEN w op r THEN t [ELSE e] END` (W8-B-010 item 10: without `ELSE` it is NULL
    /// for the other rows).
    Case {
        when: Box<Expr>,
        op: &'static str,
        rhs: Box<Expr>,
        then: Box<Expr>,
        otherwise: Option<Box<Expr>>,
    },
}

#[derive(Clone, Debug)]
enum Pred {
    Cmp(Expr, &'static str, Expr),
    Between(Expr, Expr, Expr, bool),
    In(Expr, Vec<Expr>, bool),
    IsNull(Expr, bool),
    Like(Expr, String, bool),
    BoolExpr(Expr),
    And(Box<Pred>, Box<Pred>),
    Or(Box<Pred>, Box<Pred>),
    Not(Box<Pred>),
    NullTest(Box<Pred>, bool),
}

fn render_expr(e: &Expr, t: &Table) -> String {
    match e {
        Expr::Col(i) => t.column(*i).name.clone(),
        Expr::Lit(s) => s.clone(),
        Expr::Cast(inner, ty) => format!("CAST({} AS {ty})", render_expr(inner, t)),
        Expr::Arith(a, op, b) => format!("({} {op} {})", render_expr(a, t), render_expr(b, t)),
        Expr::Func(name, args) => format!(
            "{name}({})",
            args.iter()
                .map(|a| render_expr(a, t))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Expr::Extract(part, e) => format!("extract({part} FROM {})", render_expr(e, t)),
        Expr::Case {
            when,
            op,
            rhs,
            then,
            otherwise,
        } => format!(
            "(CASE WHEN {} {op} {} THEN {}{} END)",
            render_expr(when, t),
            render_expr(rhs, t),
            render_expr(then, t),
            otherwise
                .as_ref()
                .map(|e| format!(" ELSE {}", render_expr(e, t)))
                .unwrap_or_default()
        ),
    }
}

fn render_pred(p: &Pred, t: &Table) -> String {
    match p {
        Pred::Cmp(a, op, b) => format!("{} {op} {}", render_expr(a, t), render_expr(b, t)),
        Pred::Between(e, lo, hi, neg) => format!(
            "{} {}BETWEEN {} AND {}",
            render_expr(e, t),
            if *neg { "NOT " } else { "" },
            render_expr(lo, t),
            render_expr(hi, t)
        ),
        Pred::In(e, items, neg) => format!(
            "{} {}IN ({})",
            render_expr(e, t),
            if *neg { "NOT " } else { "" },
            items
                .iter()
                .map(|i| render_expr(i, t))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Pred::IsNull(e, neg) => format!(
            "{} IS {}NULL",
            render_expr(e, t),
            if *neg { "NOT " } else { "" }
        ),
        Pred::Like(e, pat, neg) => format!(
            "{} {}LIKE '{}'",
            render_expr(e, t),
            if *neg { "NOT " } else { "" },
            pat.replace('\'', "''")
        ),
        Pred::BoolExpr(e) => render_expr(e, t),
        Pred::And(a, b) => format!("({} AND {})", render_pred(a, t), render_pred(b, t)),
        Pred::Or(a, b) => format!("({} OR {})", render_pred(a, t), render_pred(b, t)),
        Pred::Not(a) => format!("(NOT {})", render_pred(a, t)),
        Pred::NullTest(a, neg) => format!(
            "(({}) IS {}NULL)",
            render_pred(a, t),
            if *neg { "NOT " } else { "" }
        ),
    }
}

/// Proleptic Gregorian date of `days` since 1970-01-01 (Hinnant's `civil_from_days`).
fn civil(days: i64) -> (i64, i64, i64) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y }, m, d)
}

fn date_text(days: i64) -> String {
    let (y, m, d) = civil(days);
    format!("{y:04}-{m:02}-{d:02}")
}

fn ts_text(micros: i64) -> String {
    let secs = micros.div_euclid(1_000_000);
    let frac = micros.rem_euclid(1_000_000);
    let days = secs.div_euclid(86_400);
    let sod = secs.rem_euclid(86_400);
    let mut out = format!(
        "{} {:02}:{:02}:{:02}",
        date_text(days),
        sod / 3600,
        sod / 60 % 60,
        sod % 60
    );
    if frac != 0 {
        let _ = write!(out, ".{frac:06}");
    }
    out
}

fn rand_ts_micros(rng: &mut Rng) -> i64 {
    let base = rng.range(-10_005, 10_005) * 4111 * 1_000_000;
    let frac = if rng.chance(70) {
        rng.range(0, 3) * 250_000
    } else {
        rng.range(0, 999_999)
    };
    base + frac
}

/// A literal of `kind`, mostly from the data's domain (`ty` refines integer literals).
fn gen_lit(rng: &mut Rng, kind: Kind, ty: Option<Ty>, special: bool) -> String {
    match kind {
        Kind::Int => {
            let ty = ty.unwrap_or(Ty::Integer);
            match rng.below(14) {
                0 => {
                    let (lo, hi) = ty.extremes();
                    if rng.chance(50) { lo } else { hi }.to_string()
                }
                1 => format!("{}.5", rng.range(-21, 21)),
                2 => format!("{}.0", rng.range(-21, 21)),
                3 => format!("CAST({} AS BIGINT)", rng.range(-21, 41)),
                4 => format!("{}.25", rng.range(0, 21)),
                // Exact decimals such as 0.3 / 2.1 (W8-B-002: `id * 0.1 = 0.3`).
                5 => format!("{}.{}", rng.range(0, 4), rng.range(1, 9)),
                _ if ty.unsigned() => rng.range(-1, 42).to_string(),
                _ => rng.range(-22, 22).to_string(),
            }
        }
        Kind::Float => match rng.below(12) {
            0 if special => rng
                .pick(&["'NaN'::DOUBLE", "'inf'::DOUBLE", "'-inf'::DOUBLE"])
                .to_string(),
            1 => "0.1".to_string(),
            2 => rng.range(-21, 21).to_string(),
            3 => format!("{}e0", rng.range(-160, 160) as f64 / 8.0),
            _ => format!("{}", rng.range(-170, 170) as f64 / 8.0),
        },
        Kind::Dec => match rng.below(6) {
            0 => rng.range(-21, 21).to_string(),
            1 => format!("{:.3}", rng.range(-21_000, 21_000) as f64 / 1000.0),
            _ => format!("{:.2}", rng.range(-2100, 2100) as f64 / 100.0),
        },
        Kind::Str => match rng.below(5) {
            0 => format!("'s{}'", rng.range(0, 310)),
            1 => "'q'".to_string(),
            _ => format!("'{}'", rng.pick(&STRINGS).replace('\'', "''")),
        },
        Kind::Date => match rng.below(8) {
            0 => format!(
                "TIMESTAMP '{}'",
                ts_text(rng.range(-1010, 1010) * 86_400_000_000)
            ),
            1 => format!(
                "TIMESTAMP '{}'",
                ts_text(rng.range(-1010, 1010) * 86_400_000_000 + 43_200_000_000)
            ),
            _ => format!("DATE '{}'", date_text(rng.range(-1010, 1010))),
        },
        Kind::Ts => match rng.below(8) {
            0 => format!("DATE '{}'", date_text(rng.range(-520, 520))),
            1 => format!("TIMESTAMPTZ '{}+00'", ts_text(rand_ts_micros(rng))),
            _ => format!("TIMESTAMP '{}'", ts_text(rand_ts_micros(rng))),
        },
        Kind::Tstz => match rng.below(6) {
            0 => format!("TIMESTAMP '{}'", ts_text(rand_ts_micros(rng))),
            1 => format!("TIMESTAMPTZ '{}+05:30'", ts_text(rand_ts_micros(rng))),
            _ => format!("TIMESTAMPTZ '{}+00'", ts_text(rand_ts_micros(rng))),
        },
        Kind::Bool => rng.pick(&["true", "false"]).to_string(),
    }
}

/// Columns of `kind`.
fn columns_of(t: &Table, kind: Kind) -> Vec<usize> {
    (0..t.columns.len())
        .filter(|&i| t.column(i).ty.kind() == kind)
        .collect()
}

/// An expression of `kind` with its integer type hint: a column, a cast, or arithmetic.
fn gen_value_expr(rng: &mut Rng, t: &Table, kind: Kind, depth: u32) -> Option<(Expr, Option<Ty>)> {
    if depth == 0
        && rng.chance(10)
        && let Some(e) = gen_null_or_string_expr(rng, t, kind)
    {
        return Some((e, None));
    }
    let cols = columns_of(t, kind);
    let roll = rng.below(10);
    if depth < 2 && roll >= 7 {
        // Casts into `kind` from another class.
        let from = |k: Kind| columns_of(t, k);
        let cast = match kind {
            Kind::Float => {
                let src = [from(Kind::Int), from(Kind::Dec), from(Kind::Float)].concat();
                (!src.is_empty()).then(|| {
                    let c = *rng.pick(&src);
                    (Expr::Cast(Box::new(Expr::Col(c)), "DOUBLE"), None)
                })
            }
            Kind::Int => {
                let src = [from(Kind::Int), from(Kind::Bool), from(Kind::Float)].concat();
                (!src.is_empty()).then(|| {
                    let c = *rng.pick(&src);
                    let target = *rng.pick(&["INTEGER", "BIGINT", "SMALLINT"]);
                    let ty = match target {
                        "BIGINT" => Ty::BigInt,
                        "SMALLINT" => Ty::SmallInt,
                        _ => Ty::Integer,
                    };
                    (Expr::Cast(Box::new(Expr::Col(c)), target), Some(ty))
                })
            }
            Kind::Str => {
                let src = [from(Kind::Int), from(Kind::Str), from(Kind::Date)].concat();
                (!src.is_empty()).then(|| {
                    let c = *rng.pick(&src);
                    (Expr::Cast(Box::new(Expr::Col(c)), "VARCHAR"), None)
                })
            }
            Kind::Date => {
                let src = [from(Kind::Ts), from(Kind::Date)].concat();
                (!src.is_empty()).then(|| {
                    let c = *rng.pick(&src);
                    (Expr::Cast(Box::new(Expr::Col(c)), "DATE"), None)
                })
            }
            Kind::Ts => {
                let src = [from(Kind::Date), from(Kind::Ts)].concat();
                (!src.is_empty()).then(|| {
                    let c = *rng.pick(&src);
                    (Expr::Cast(Box::new(Expr::Col(c)), "TIMESTAMP"), None)
                })
            }
            Kind::Dec => {
                // Into DECIMAL from integers and other DECIMALs (W4-B-004 shapes).
                let src = [from(Kind::Int), from(Kind::Dec)].concat();
                (!src.is_empty()).then(|| {
                    let c = *rng.pick(&src);
                    let target = *rng.pick(&["DECIMAL(18, 4)", "DECIMAL(9, 2)", "DECIMAL(38, 6)"]);
                    (Expr::Cast(Box::new(Expr::Col(c)), target), None)
                })
            }
            _ => None,
        };
        if cast.is_some() {
            return cast;
        }
    }
    if cols.is_empty() {
        return None;
    }
    let c = *rng.pick(&cols);
    let ty = t.column(c).ty;
    let col = Expr::Col(c);
    if depth < 2 && roll >= 5 {
        let arith = match kind {
            Kind::Int => {
                // `/` is DOUBLE division in DuckDB, `//` integer division (W8-B-001).
                let op = *rng.pick(&["+", "-", "*", "+", "-", "*", "/", "//", "%"]);
                let division = matches!(op, "/" | "//" | "%");
                let rhs = match rng.below(10) {
                    // W8-B-001: zero divisors, as a literal and computed from a column.
                    0 | 1 if division => Some(if rng.chance(50) {
                        Expr::Lit("0".into())
                    } else {
                        Expr::Arith(Box::new(col.clone()), "-", Box::new(col.clone()))
                    }),
                    // W8-B-002: fractional literals (DECIMAL in DuckDB, DOUBLE in the engine).
                    2 | 3 if op != "//" && op != "%" => Some(Expr::Lit(
                        rng.pick(&["0.1", "0.5", "0.7", "1.1", "1.15", "2.5"])
                            .to_string(),
                    )),
                    4..=6 if !division => {
                        gen_value_expr(rng, t, Kind::Int, depth + 1).map(|(e, _)| e)
                    }
                    _ => Some(Expr::Lit(rng.range(1, 5).to_string())),
                };
                let fractional = matches!(&rhs, Some(Expr::Lit(s)) if s.contains('.'));
                let hint = (op != "/" && !fractional).then_some(Ty::BigInt);
                rhs.map(|r| (Expr::Arith(Box::new(col.clone()), op, Box::new(r)), hint))
            }
            Kind::Float => {
                let op = *rng.pick(&["+", "-", "*", "/"]);
                let lit = format!("{}", rng.range(1, 16) as f64 / 4.0);
                Some((
                    Expr::Arith(Box::new(col.clone()), op, Box::new(Expr::Lit(lit))),
                    None,
                ))
            }
            Kind::Dec => Some((
                Expr::Arith(
                    Box::new(col.clone()),
                    rng.pick(&["+", "-"]),
                    Box::new(Expr::Lit(format!(
                        "{:.2}",
                        rng.range(0, 300) as f64 / 100.0
                    ))),
                ),
                None,
            )),
            Kind::Date => Some((
                Expr::Arith(
                    Box::new(col.clone()),
                    "+",
                    Box::new(Expr::Lit(rng.range(-5, 5).to_string())),
                ),
                None,
            )),
            _ => None,
        };
        if arith.is_some() {
            return arith;
        }
    }
    Some((col, Some(ty)))
}

/// W8-B-010 item 10: an expression of class `kind` that can produce NULLs from non-NULL
/// inputs or has edge-case semantics: `CASE` without (or with) `ELSE`, `coalesce`, `nullif`,
/// and string functions with edge arguments (`substring` from 0 or a negative start, `lpad`
/// / `rpad` shorter than the input, `length` / `upper` of non-ASCII text, `strpos` of the empty
/// string).
fn gen_null_or_string_expr(rng: &mut Rng, t: &Table, kind: Kind) -> Option<Expr> {
    let cols = columns_of(t, kind);
    let strs = columns_of(t, Kind::Str);
    let lit = |rng: &mut Rng| Expr::Lit(gen_lit(rng, kind, None, false));
    match (kind, rng.below(6)) {
        (Kind::Str, 0 | 1) if !strs.is_empty() => {
            let s = Box::new(Expr::Col(*rng.pick(&strs)));
            let n = |v: i64| Expr::Lit(v.to_string());
            Some(match rng.below(6) {
                0 => Expr::Func("substring", vec![*s, n(*rng.pick(&[0, -2, 1, 3])), n(2)]),
                1 => Expr::Func("lpad", vec![*s, n(2), Expr::Lit("'x'".into())]),
                2 => Expr::Func(
                    "rpad",
                    vec![*s, n(*rng.pick(&[1, 3])), Expr::Lit("'.'".into())],
                ),
                3 => Expr::Func("upper", vec![*s]),
                4 => Expr::Func("lower", vec![*s]),
                _ => Expr::Func("concat", vec![*s, Expr::Lit("'ñ'".into())]),
            })
        }
        (Kind::Int, 0) if !strs.is_empty() => {
            let s = Expr::Col(*rng.pick(&strs));
            Some(match rng.below(3) {
                0 => Expr::Func("length", vec![s]),
                1 => Expr::Func("strpos", vec![s, Expr::Lit("''".into())]),
                _ => Expr::Func("strpos", vec![s, Expr::Lit("'a'".into())]),
            })
        }
        (Kind::Bool | Kind::Ts | Kind::Tstz, _) => None,
        (_, 2 | 3) if !cols.is_empty() => {
            let w = Expr::Col(*rng.pick(&cols));
            let then = Expr::Col(*rng.pick(&cols));
            let rhs = lit(rng);
            let otherwise = rng.chance(40).then(|| Box::new(lit(rng)));
            Some(Expr::Case {
                when: Box::new(w),
                op: rng.pick(&["=", "<>", "<", ">"]),
                rhs: Box::new(rhs),
                then: Box::new(then),
                otherwise,
            })
        }
        (_, 4) if !cols.is_empty() => {
            let c = Expr::Col(*rng.pick(&cols));
            Some(Expr::Func("coalesce", vec![c, lit(rng)]))
        }
        (_, 5) if !cols.is_empty() => {
            let c = Expr::Col(*rng.pick(&cols));
            Some(Expr::Func("nullif", vec![c, lit(rng)]))
        }
        _ => None,
    }
}

/// A TIMESTAMP-typed operand for column `c` (TIMESTAMP_MS is cast to TIMESTAMP first).
fn temporal_operand(t: &Table, c: usize) -> Expr {
    if t.column(c).ty == Ty::TimestampMs {
        Expr::Cast(Box::new(Expr::Col(c)), "TIMESTAMP")
    } else {
        Expr::Col(c)
    }
}

/// W8-B-008 shapes: `ts - ts` (INTERVAL), `date - date` (BIGINT days), `date + n`,
/// `date_trunc` and `extract`, with a literal of the result's class for comparisons.
fn gen_temporal(rng: &mut Rng, t: &Table) -> Option<(Expr, String)> {
    let ts = columns_of(t, Kind::Ts);
    let dates = columns_of(t, Kind::Date);
    let any = [ts.clone(), dates.clone()].concat();
    if any.is_empty() {
        return None;
    }
    for _ in 0..6 {
        match rng.below(5) {
            0 if !ts.is_empty() => {
                let a = *rng.pick(&ts);
                let b = *rng.pick(&ts);
                let e = Expr::Arith(
                    Box::new(temporal_operand(t, a)),
                    "-",
                    Box::new(temporal_operand(t, b)),
                );
                return Some((
                    e,
                    format!("INTERVAL '{} hours'", rng.range(-20_000, 20_000)),
                ));
            }
            1 if !dates.is_empty() => {
                let a = *rng.pick(&dates);
                let b = *rng.pick(&dates);
                let e = Expr::Arith(Box::new(Expr::Col(a)), "-", Box::new(Expr::Col(b)));
                return Some((e, rng.range(-2000, 2000).to_string()));
            }
            2 if !dates.is_empty() => {
                let a = *rng.pick(&dates);
                let e = Expr::Arith(
                    Box::new(Expr::Col(a)),
                    "+",
                    Box::new(Expr::Lit(rng.range(-40, 40).to_string())),
                );
                return Some((e, format!("DATE '{}'", date_text(rng.range(-1010, 1010)))));
            }
            3 => {
                let c = *rng.pick(&any);
                let is_ts = t.column(c).ty.kind() == Kind::Ts;
                let part = if is_ts {
                    *rng.pick(&["year", "month", "day", "hour"])
                } else {
                    *rng.pick(&["year", "month", "day"])
                };
                let e = Expr::Func(
                    "date_trunc",
                    vec![Expr::Lit(format!("'{part}'")), temporal_operand(t, c)],
                );
                return Some((e, format!("TIMESTAMP '{}'", ts_text(rand_ts_micros(rng)))));
            }
            4 => {
                let c = *rng.pick(&any);
                let is_ts = t.column(c).ty.kind() == Kind::Ts;
                let part = if is_ts {
                    *rng.pick(&["year", "month", "day", "hour"])
                } else {
                    *rng.pick(&["year", "month", "day"])
                };
                let lit = match part {
                    "year" => rng.range(1966, 1973),
                    "month" => rng.range(1, 12),
                    "day" => rng.range(1, 31),
                    _ => rng.range(0, 23),
                };
                let e = Expr::Extract(part, Box::new(temporal_operand(t, c)));
                return Some((e, lit.to_string()));
            }
            _ => {}
        }
    }
    None
}

const KINDS: [Kind; 8] = [
    Kind::Int,
    Kind::Float,
    Kind::Dec,
    Kind::Str,
    Kind::Date,
    Kind::Ts,
    Kind::Tstz,
    Kind::Bool,
];

fn gen_leaf(rng: &mut Rng, t: &Table) -> Pred {
    if rng.chance(6)
        && let Some((e, lit)) = gen_temporal(rng, t)
    {
        let op = *rng.pick(&["=", "<>", "<", "<=", ">", ">="]);
        return Pred::Cmp(e, op, Expr::Lit(lit));
    }
    for _ in 0..8 {
        let kind = *rng.pick(&KINDS);
        let Some((e, ty)) = gen_value_expr(rng, t, kind, 0) else {
            continue;
        };
        let lit = |rng: &mut Rng| Expr::Lit(gen_lit(rng, kind, ty, t.special));
        let p = match (kind, rng.below(10)) {
            (Kind::Bool, 0..=3) => Pred::BoolExpr(e),
            (Kind::Str, 0..=2) => {
                let pat = if rng.chance(30) {
                    format!("s{}%", rng.range(0, 30))
                } else {
                    rng.pick(&[
                        "a%", "%b", "%a%", "_", "a_", "%", "", "%%", "B%", "ñ%", "x_%", "_b%",
                    ])
                    .to_string()
                };
                Pred::Like(e, pat, rng.chance(30))
            }
            (_, 0) => Pred::IsNull(e, rng.chance(50)),
            (_, 1) => {
                let lo = lit(rng);
                let hi = lit(rng);
                Pred::Between(e, lo, hi, rng.chance(25))
            }
            (_, 2) => {
                let n = rng.range(1, 5);
                let mut items: Vec<Expr> = (0..n).map(|_| lit(rng)).collect();
                if rng.chance(10) {
                    items.push(Expr::Lit("NULL".into()));
                }
                Pred::In(e, items, rng.chance(35))
            }
            (_, 3) if kind != Kind::Bool => {
                // Column vs column of the same class.
                match gen_value_expr(rng, t, kind, 1) {
                    Some((other, _)) => {
                        Pred::Cmp(e, rng.pick(&["=", "<>", "<", "<=", ">", ">="]), other)
                    }
                    None => continue,
                }
            }
            _ => {
                let op = *rng.pick(&["=", "<>", "<", "<=", ">", ">=", "!="]);
                let l = lit(rng);
                if rng.chance(20) {
                    Pred::Cmp(l, op, e)
                } else {
                    Pred::Cmp(e, op, l)
                }
            }
        };
        return p;
    }
    Pred::IsNull(Expr::Col(0), true)
}

fn gen_pred(rng: &mut Rng, t: &Table, depth: u32) -> Pred {
    if depth >= 3 || rng.chance(45) {
        return gen_leaf(rng, t);
    }
    match rng.below(10) {
        0..=3 => Pred::And(
            Box::new(gen_pred(rng, t, depth + 1)),
            Box::new(gen_pred(rng, t, depth + 1)),
        ),
        4..=6 => Pred::Or(
            Box::new(gen_pred(rng, t, depth + 1)),
            Box::new(gen_pred(rng, t, depth + 1)),
        ),
        7..=8 => Pred::Not(Box::new(gen_pred(rng, t, depth + 1))),
        _ => Pred::NullTest(Box::new(gen_pred(rng, t, depth + 1)), rng.chance(50)),
    }
}

// ---------------------------------------------------------------------------------------------
// Queries
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Debug)]
enum Shape {
    /// `SELECT <cols / exprs> FROM T WHERE p`.
    Filter { select: Vec<Expr> },
    /// `SELECT count(*) FROM T WHERE p`.
    Count,
    /// `SELECT g, aggs FROM T WHERE p GROUP BY g`, or `SELECT aggs FROM T WHERE p` without a
    /// key. Aggregate arguments are columns or (numeric) expressions.
    Group {
        key: Option<usize>,
        aggs: Vec<(String, Expr)>,
    },
    /// `SELECT cols FROM T WHERE p ORDER BY key, id LIMIT n`.
    TopN {
        select: Vec<usize>,
        key: usize,
        desc: bool,
        /// Explicit `NULLS FIRST` (`Some(true)`) / `NULLS LAST` (`Some(false)`).
        nulls: Option<bool>,
        limit: usize,
    },
    /// `SELECT aggs FROM (<input>) s` over an input that is empty (or not) without a filter
    /// (W8-B-005, W8-B-006). Aggregates are `(function, column)`; never has a predicate.
    EmptyAgg {
        aggs: Vec<(String, usize)>,
        src: EmptySrc,
    },
    /// Renames over nullable columns (W8-B-004).
    Rename(Rename),
    /// Host-orchestrated wrappers around the existing expressions (W8-B-003).
    Wrapped(Wrap),
    /// `SELECT k, aggs FROM T WHERE p GROUP BY k HAVING h` (W8-B-010 item 10); `h` is rendered
    /// SQL over the aggregates (`count(*) > n`, `<agg> IS [NOT] NULL`, `<agg> > 0`).
    Having {
        key: usize,
        aggs: Vec<(String, Expr)>,
        having: String,
    },
    /// `SELECT id, part, f(arg) OVER (PARTITION BY part ORDER BY order[, id]) FROM T WHERE p`
    /// (W8-B-010 item 10): ranking and running aggregates with ties and nullable keys.
    Window {
        func: &'static str,
        arg: usize,
        part: usize,
        order: usize,
        desc: bool,
    },
}

/// The input of [`Shape::EmptyAgg`] (`cols`: `id` plus the aggregated columns).
#[derive(Clone, Debug)]
enum EmptySrc {
    /// `SELECT cols FROM T LIMIT 0`.
    Limit0,
    /// `SELECT cols FROM T ORDER BY id LIMIT l OFFSET o` (often `o >= rows`).
    Offset { limit: usize, offset: usize },
    /// `SELECT cols FROM T EXCEPT SELECT cols FROM T` (always empty).
    Except,
    /// `SELECT cols FROM T INTERSECT SELECT cols FROM T`, or with an empty (`LIMIT 0`) right
    /// branch.
    Intersect { empty: bool },
}

#[derive(Clone, Debug)]
enum Rename {
    /// `SELECT k, f(a) FROM (SELECT k, src AS a FROM T WHERE p) s GROUP BY k`, where `a` is the
    /// name of another (preferably NULL-free) column and `src` is preferably nullable.
    Agg {
        key: usize,
        src: usize,
        alias: usize,
        func: &'static str,
    },
    /// `SELECT a AS b, b AS a FROM T WHERE p ORDER BY b, id LIMIT n` (a column swap).
    Swap {
        a: usize,
        b: usize,
        desc: bool,
        limit: usize,
    },
    /// `SELECT a FROM (SELECT src AS a FROM T WHERE p) s ORDER BY a LIMIT n`.
    Order {
        src: usize,
        alias: usize,
        desc: bool,
        limit: usize,
    },
    /// `SELECT k, f(a) FROM (SELECT k, a FROM T UNION ALL SELECT k, b FROM T) u GROUP BY k`,
    /// where the second branch names the column differently (`b`, same type) or is
    /// `NULL AS a` (`b == None`). Never has a predicate.
    Union {
        key: usize,
        a: usize,
        b: Option<usize>,
        func: &'static str,
    },
}

#[derive(Clone, Debug)]
enum Wrap {
    /// `SELECT DISTINCT ON (k) k, id, exprs FROM T WHERE p ORDER BY k, id`.
    DistinctOn { key: usize, select: Vec<Expr> },
    /// `SELECT exprs FROM T o WHERE [NOT] EXISTS (SELECT 1 FROM T u WHERE u.k = o.k AND p)`.
    Exists {
        key: usize,
        negated: bool,
        select: Vec<Expr>,
    },
    /// `SELECT exprs FROM T o WHERE o.x > (SELECT max(u.x) FROM T u WHERE u.k = o.k AND p)`.
    ScalarSub {
        key: usize,
        x: usize,
        select: Vec<Expr>,
    },
    /// `SELECT o.k, l.m FROM T o, LATERAL (SELECT max(u.x) AS m FROM T u WHERE u.k = o.k) l
    /// WHERE p`.
    Lateral { key: usize, x: usize },
    /// `WITH RECURSIVE r(n) AS (... n < depth) SELECT aggs FROM T WHERE id IN (SELECT n * step
    /// FROM r) AND p`.
    Recursive {
        aggs: Vec<(String, Expr)>,
        depth: i64,
        step: i64,
    },
    /// `SELECT v.x, exprs FROM (VALUES ...) v(x) JOIN T o ON o.id = v.x WHERE p`.
    Values { ids: Vec<i64>, select: Vec<Expr> },
    /// `SELECT g.n, exprs FROM generate_series(a, b, s) g(n) JOIN T o ON o.id = g.n WHERE p`.
    Series {
        start: i64,
        stop: i64,
        step: i64,
        select: Vec<Expr>,
    },
    /// `SELECT k, count(DISTINCT c) FROM T WHERE p GROUP BY k`.
    CountDistinct { key: usize, col: usize },
}

#[derive(Clone, Debug)]
struct Query {
    shape: Shape,
    pred: Option<Pred>,
}

fn render_agg(f: &str, e: &Expr, t: &Table) -> String {
    if f == "count_star" {
        "count(*)".to_string()
    } else {
        format!("{f}({})", render_expr(e, t))
    }
}

/// Whether aggregate `f` of `e` is floating point (compared rounded).
fn agg_is_float(f: &str, e: &Expr, t: &Table) -> bool {
    f == "avg"
        || STAT_AGGS.contains(&f)
        || (f != "count_star" && f != "count" && expr_is_float(e, t))
}

impl Query {
    /// SQL over table `from`, and the number of output columns.
    fn render(&self, t: &Table, from: &str) -> (String, usize) {
        let pred_sql = self.pred.as_ref().map(|p| render_pred(p, t));
        let where_clause = pred_sql
            .as_ref()
            .map(|p| format!(" WHERE {p}"))
            .unwrap_or_default();
        let and_pred = pred_sql
            .as_ref()
            .map(|p| format!(" AND ({p})"))
            .unwrap_or_default();
        let name = |c: usize| t.column(c).name.clone();
        let list = |select: &[Expr]| {
            select
                .iter()
                .map(|e| render_expr(e, t))
                .collect::<Vec<_>>()
                .join(", ")
        };
        let agg_list = |aggs: &[(String, Expr)]| {
            aggs.iter()
                .map(|(f, e)| render_agg(f, e, t))
                .collect::<Vec<_>>()
                .join(", ")
        };
        match &self.shape {
            Shape::Filter { select } => (
                format!(
                    "SELECT {} FROM {from}{where_clause}",
                    list(select.as_slice())
                ),
                select.len(),
            ),
            Shape::Count => (format!("SELECT count(*) FROM {from}{where_clause}"), 1),
            Shape::Having { key, aggs, having } => {
                let k = name(*key);
                (
                    format!(
                        "SELECT {k}, {} FROM {from}{where_clause} GROUP BY {k} HAVING {having}",
                        agg_list(aggs.as_slice())
                    ),
                    1 + aggs.len(),
                )
            }
            Shape::Window {
                func,
                arg,
                part,
                order,
                desc,
            } => {
                let call = match *func {
                    "row_number" | "rank" | "dense_rank" => format!("{func}()"),
                    other => format!("{other}({})", name(*arg)),
                };
                // `row_number` needs a total order; the others are defined over ties.
                let tie = if *func == "row_number" { ", id" } else { "" };
                (
                    format!(
                        "SELECT id, {p}, {call} OVER (PARTITION BY {p} ORDER BY {o}{}{tie}) FROM {from}{where_clause}",
                        if *desc { " DESC" } else { "" },
                        p = name(*part),
                        o = name(*order),
                    ),
                    3,
                )
            }
            Shape::Group { key, aggs } => {
                let aggs_sql = agg_list(aggs.as_slice());
                match key {
                    Some(key) => {
                        let key_name = name(*key);
                        (
                            format!(
                                "SELECT {key_name}, {aggs_sql} FROM {from}{where_clause} GROUP BY {key_name}"
                            ),
                            1 + aggs.len(),
                        )
                    }
                    None => (
                        format!("SELECT {aggs_sql} FROM {from}{where_clause}"),
                        aggs.len(),
                    ),
                }
            }
            Shape::TopN {
                select,
                key,
                desc,
                nulls,
                limit,
            } => (
                format!(
                    "SELECT {} FROM {from}{where_clause} ORDER BY {}{}{}, id LIMIT {limit}",
                    select
                        .iter()
                        .map(|&c| name(c))
                        .collect::<Vec<_>>()
                        .join(", "),
                    name(*key),
                    if *desc { " DESC" } else { "" },
                    match nulls {
                        Some(true) => " NULLS FIRST",
                        Some(false) => " NULLS LAST",
                        None => "",
                    }
                ),
                select.len(),
            ),
            Shape::EmptyAgg { aggs, src } => {
                let mut cols: Vec<usize> = vec![0];
                for (_, c) in aggs {
                    if !cols.contains(c) {
                        cols.push(*c);
                    }
                }
                let cols = cols.iter().map(|&c| name(c)).collect::<Vec<_>>().join(", ");
                let inner = match src {
                    EmptySrc::Limit0 => format!("SELECT {cols} FROM {from} LIMIT 0"),
                    EmptySrc::Offset { limit, offset } => format!(
                        "SELECT {cols} FROM {from} ORDER BY id LIMIT {limit} OFFSET {offset}"
                    ),
                    EmptySrc::Except => {
                        format!("SELECT {cols} FROM {from} EXCEPT SELECT {cols} FROM {from}")
                    }
                    EmptySrc::Intersect { empty: false } => {
                        format!("SELECT {cols} FROM {from} INTERSECT SELECT {cols} FROM {from}")
                    }
                    EmptySrc::Intersect { empty: true } => format!(
                        "(SELECT {cols} FROM {from}) INTERSECT (SELECT {cols} FROM {from} LIMIT 0)"
                    ),
                };
                let aggs_sql = aggs
                    .iter()
                    .map(|(f, c)| render_agg(f, &Expr::Col(*c), t))
                    .collect::<Vec<_>>()
                    .join(", ");
                (format!("SELECT {aggs_sql} FROM ({inner}) s"), aggs.len())
            }
            Shape::Rename(r) => match r {
                Rename::Agg {
                    key,
                    src,
                    alias,
                    func,
                } => {
                    let (k, s, a) = (name(*key), name(*src), name(*alias));
                    (
                        format!(
                            "SELECT {k}, {func}({a}) FROM (SELECT {k}, {s} AS {a} FROM {from}{where_clause}) s GROUP BY {k}"
                        ),
                        2,
                    )
                }
                Rename::Swap { a, b, desc, limit } => {
                    let (an, bn) = (name(*a), name(*b));
                    (
                        format!(
                            "SELECT {an} AS {bn}, {bn} AS {an} FROM {from}{where_clause} ORDER BY {bn}{}, id LIMIT {limit}",
                            if *desc { " DESC" } else { "" }
                        ),
                        2,
                    )
                }
                Rename::Order {
                    src,
                    alias,
                    desc,
                    limit,
                } => {
                    let (s, a) = (name(*src), name(*alias));
                    (
                        format!(
                            "SELECT {a} FROM (SELECT {s} AS {a} FROM {from}{where_clause}) s ORDER BY {a}{} LIMIT {limit}",
                            if *desc { " DESC" } else { "" }
                        ),
                        1,
                    )
                }
                Rename::Union { key, a, b, func } => {
                    let (k, an) = (name(*key), name(*a));
                    let other = match b {
                        Some(b) => name(*b),
                        None => format!("NULL AS {an}"),
                    };
                    (
                        format!(
                            "SELECT {k}, {func}({an}) FROM (SELECT {k}, {an} FROM {from} UNION ALL SELECT {k}, {other} FROM {from}) u GROUP BY {k}"
                        ),
                        2,
                    )
                }
            },
            Shape::Wrapped(w) => match w {
                Wrap::DistinctOn { key, select } => {
                    let k = name(*key);
                    let (id, order, ids) = if *key == 0 {
                        ("", String::new(), 0)
                    } else {
                        (", id", format!("{k}, "), 1)
                    };
                    (
                        format!(
                            "SELECT DISTINCT ON ({k}) {k}{id}, {} FROM {from}{where_clause} ORDER BY {order}id",
                            list(select.as_slice())
                        ),
                        1 + ids + select.len(),
                    )
                }
                Wrap::Exists {
                    key,
                    negated,
                    select,
                } => {
                    let k = name(*key);
                    (
                        format!(
                            "SELECT {} FROM {from} o WHERE {}EXISTS (SELECT 1 FROM {from} u WHERE u.{k} = o.{k}{and_pred})",
                            list(select.as_slice()),
                            if *negated { "NOT " } else { "" }
                        ),
                        select.len(),
                    )
                }
                Wrap::ScalarSub { key, x, select } => {
                    let (k, x) = (name(*key), name(*x));
                    (
                        format!(
                            "SELECT {} FROM {from} o WHERE o.{x} > (SELECT max(u.{x}) FROM {from} u WHERE u.{k} = o.{k}{and_pred})",
                            list(select.as_slice())
                        ),
                        select.len(),
                    )
                }
                Wrap::Lateral { key, x } => {
                    let (k, x) = (name(*key), name(*x));
                    (
                        format!(
                            "SELECT o.{k}, l.m FROM {from} o, LATERAL (SELECT max(u.{x}) AS m FROM {from} u WHERE u.{k} = o.{k}) l{where_clause}"
                        ),
                        2,
                    )
                }
                Wrap::Recursive { aggs, depth, step } => (
                    format!(
                        "WITH RECURSIVE r(n) AS (SELECT 0 UNION ALL SELECT n + 1 FROM r WHERE n < {depth}) \
                         SELECT {} FROM {from} WHERE id IN (SELECT n * {step} FROM r){and_pred}",
                        agg_list(aggs.as_slice())
                    ),
                    aggs.len(),
                ),
                Wrap::Values { ids, select } => (
                    format!(
                        "SELECT v.x, {} FROM (VALUES {}) v(x) JOIN {from} o ON o.id = v.x{where_clause}",
                        list(select.as_slice()),
                        ids.iter()
                            .map(|i| format!("({i})"))
                            .collect::<Vec<_>>()
                            .join(", ")
                    ),
                    1 + select.len(),
                ),
                Wrap::Series {
                    start,
                    stop,
                    step,
                    select,
                } => (
                    format!(
                        "SELECT g.n, {} FROM generate_series({start}, {stop}, {step}) g(n) JOIN {from} o ON o.id = g.n{where_clause}",
                        list(select.as_slice())
                    ),
                    1 + select.len(),
                ),
                Wrap::CountDistinct { key, col } => {
                    let (k, c) = (name(*key), name(*col));
                    (
                        format!(
                            "SELECT {k}, count(DISTINCT {c}) FROM {from}{where_clause} GROUP BY {k}"
                        ),
                        2,
                    )
                }
            },
        }
    }

    /// Whether each output column is floating point (compared rounded).
    fn output_kinds(&self, t: &Table) -> Vec<bool> {
        let float = |c: usize| t.column(c).ty.kind() == Kind::Float;
        let exprs = |select: &[Expr]| {
            select
                .iter()
                .map(|e| expr_is_float(e, t))
                .collect::<Vec<_>>()
        };
        match &self.shape {
            Shape::Filter { select } => exprs(select.as_slice()),
            Shape::Count => vec![false],
            Shape::Having { key, aggs, .. } => std::iter::once(float(*key))
                .chain(aggs.iter().map(|(f, e)| agg_is_float(f, e, t)))
                .collect(),
            Shape::Window {
                func, arg, part, ..
            } => vec![
                false,
                float(*part),
                matches!(*func, "sum" | "avg" | "min" | "max") && float(*arg) || *func == "avg",
            ],
            Shape::Group { key, aggs } => key
                .map(float)
                .into_iter()
                .chain(aggs.iter().map(|(f, e)| agg_is_float(f, e, t)))
                .collect(),
            Shape::TopN { select, .. } => select.iter().map(|&c| float(c)).collect(),
            Shape::EmptyAgg { aggs, .. } => aggs
                .iter()
                .map(|(f, c)| agg_is_float(f, &Expr::Col(*c), t))
                .collect(),
            Shape::Rename(r) => match r {
                Rename::Agg { key, src, func, .. } => {
                    vec![float(*key), *func != "count" && float(*src)]
                }
                Rename::Swap { a, b, .. } => vec![float(*a), float(*b)],
                Rename::Order { src, .. } => vec![float(*src)],
                Rename::Union { key, a, func, .. } => {
                    vec![float(*key), *func != "count" && float(*a)]
                }
            },
            Shape::Wrapped(w) => match w {
                Wrap::DistinctOn { key, select } => {
                    let mut out = vec![float(*key)];
                    if *key != 0 {
                        out.push(false);
                    }
                    out.extend(exprs(select.as_slice()));
                    out
                }
                Wrap::Exists { select, .. } | Wrap::ScalarSub { select, .. } => {
                    exprs(select.as_slice())
                }
                Wrap::Lateral { key, x } => vec![float(*key), float(*x)],
                Wrap::Recursive { aggs, .. } => {
                    aggs.iter().map(|(f, e)| agg_is_float(f, e, t)).collect()
                }
                Wrap::Values { select, .. } | Wrap::Series { select, .. } => {
                    let mut out = vec![false];
                    out.extend(exprs(select.as_slice()));
                    out
                }
                Wrap::CountDistinct { key, .. } => vec![float(*key), false],
            },
        }
    }

    /// Every expression the shape evaluates (not the predicate), with referenced columns as
    /// [`Expr::Col`]. `count(*)` placeholders are left out.
    fn shape_exprs(&self) -> Vec<Expr> {
        let agg_args = |aggs: &[(String, Expr)]| {
            aggs.iter()
                .filter(|(f, _)| f != "count_star")
                .map(|(_, e)| e.clone())
                .collect::<Vec<_>>()
        };
        match &self.shape {
            Shape::Filter { select } => select.clone(),
            Shape::Count => Vec::new(),
            Shape::Having { key, aggs, .. } => std::iter::once(Expr::Col(*key))
                .chain(agg_args(aggs.as_slice()))
                .collect(),
            Shape::Window {
                arg, part, order, ..
            } => vec![
                Expr::Col(0),
                Expr::Col(*arg),
                Expr::Col(*part),
                Expr::Col(*order),
            ],
            Shape::Group { key, aggs } => key
                .iter()
                .map(|&k| Expr::Col(k))
                .chain(agg_args(aggs.as_slice()))
                .collect(),
            Shape::TopN { select, key, .. } => select
                .iter()
                .chain(std::iter::once(key))
                .map(|&c| Expr::Col(c))
                .collect(),
            Shape::EmptyAgg { aggs, .. } => std::iter::once(0)
                .chain(aggs.iter().map(|(_, c)| *c))
                .map(Expr::Col)
                .collect(),
            Shape::Rename(r) => match r {
                Rename::Agg { key, src, .. } => vec![Expr::Col(*key), Expr::Col(*src)],
                Rename::Swap { a, b, .. } => vec![Expr::Col(0), Expr::Col(*a), Expr::Col(*b)],
                Rename::Order { src, .. } => vec![Expr::Col(*src)],
                Rename::Union { key, a, b, .. } => {
                    let mut out = vec![Expr::Col(*key), Expr::Col(*a)];
                    out.extend(b.map(Expr::Col));
                    out
                }
            },
            Shape::Wrapped(w) => match w {
                Wrap::DistinctOn { key, select } | Wrap::Exists { key, select, .. } => {
                    let mut out = vec![Expr::Col(*key), Expr::Col(0)];
                    out.extend(select.iter().cloned());
                    out
                }
                Wrap::ScalarSub { key, x, select } => {
                    let mut out = vec![Expr::Col(*key), Expr::Col(*x)];
                    out.extend(select.iter().cloned());
                    out
                }
                Wrap::Lateral { key, x } => vec![Expr::Col(*key), Expr::Col(*x)],
                Wrap::Recursive { aggs, .. } => {
                    let mut out = vec![Expr::Col(0)];
                    out.extend(agg_args(aggs.as_slice()));
                    out
                }
                Wrap::Values { select, .. } | Wrap::Series { select, .. } => {
                    let mut out = vec![Expr::Col(0)];
                    out.extend(select.iter().cloned());
                    out
                }
                Wrap::CountDistinct { key, col } => vec![Expr::Col(*key), Expr::Col(*col)],
            },
        }
    }

    fn columns(&self) -> Vec<usize> {
        let mut out = Vec::new();
        if let Some(p) = &self.pred {
            pred_columns(p, &mut out);
        }
        for e in self.shape_exprs() {
            expr_columns(&e, &mut out);
        }
        out
    }

    /// Whether `f` holds for any expression (or sub-expression) of the query.
    fn any_expr(&self, f: &dyn Fn(&Expr) -> bool) -> bool {
        self.shape_exprs().iter().any(|e| expr_any(e, f))
            || self.pred.as_ref().is_some_and(|p| pred_any(p, f))
    }

    /// The aggregate functions the query computes (`count_star` for `count(*)`).
    #[allow(dead_code)] // For shape-based `known_issue` entries (none open).
    fn agg_funcs(&self) -> Vec<String> {
        match &self.shape {
            Shape::Group { aggs, .. } | Shape::Wrapped(Wrap::Recursive { aggs, .. }) => {
                aggs.iter().map(|(f, _)| f.clone()).collect()
            }
            Shape::EmptyAgg { aggs, .. } => aggs.iter().map(|(f, _)| f.clone()).collect(),
            Shape::Rename(Rename::Agg { func, .. } | Rename::Union { func, .. }) => {
                vec![func.to_string()]
            }
            _ => Vec::new(),
        }
    }
}

fn expr_is_float(e: &Expr, t: &Table) -> bool {
    match e {
        Expr::Col(c) => t.column(*c).ty.kind() == Kind::Float,
        Expr::Lit(_) => false,
        Expr::Cast(_, ty) => *ty == "DOUBLE",
        // `/` is DOUBLE division in DuckDB, also on integers (W8-B-001).
        Expr::Arith(a, op, b) => *op == "/" || expr_is_float(a, t) || expr_is_float(b, t),
        // `coalesce` / `nullif` have their first argument's type.
        Expr::Func("coalesce" | "nullif", args) => {
            args.first().is_some_and(|a| expr_is_float(a, t))
        }
        Expr::Func(..) | Expr::Extract(..) => false,
        Expr::Case {
            then, otherwise, ..
        } => expr_is_float(then, t) || otherwise.as_ref().is_some_and(|e| expr_is_float(e, t)),
    }
}

/// Whether `e` is an integer-typed expression in DuckDB (W8-B-001 / -002 classification).
#[allow(dead_code)] // For shape-based `known_issue` entries (none open).
fn expr_is_intlike(e: &Expr, t: &Table) -> bool {
    match e {
        Expr::Col(c) => t.column(*c).ty.kind() == Kind::Int,
        Expr::Lit(s) => s.parse::<i64>().is_ok(),
        Expr::Cast(_, ty) => matches!(*ty, "INTEGER" | "BIGINT" | "SMALLINT"),
        Expr::Arith(a, op, b) => *op != "/" && expr_is_intlike(a, t) && expr_is_intlike(b, t),
        Expr::Func(..) | Expr::Extract(..) | Expr::Case { .. } => false,
    }
}

/// Whether `e` is a TIMESTAMP / DATE expression (W8-B-008 classification).
#[allow(dead_code)] // For shape-based `known_issue` entries (none open).
fn expr_is_temporal(e: &Expr, t: &Table) -> bool {
    match e {
        Expr::Col(c) => matches!(t.column(*c).ty.kind(), Kind::Date | Kind::Ts | Kind::Tstz),
        Expr::Cast(_, ty) => matches!(*ty, "DATE" | "TIMESTAMP"),
        Expr::Func(name, _) => *name == "date_trunc",
        _ => false,
    }
}

fn expr_any(e: &Expr, f: &dyn Fn(&Expr) -> bool) -> bool {
    f(e) || match e {
        Expr::Col(_) | Expr::Lit(_) => false,
        Expr::Cast(a, _) | Expr::Extract(_, a) => expr_any(a, f),
        Expr::Arith(a, _, b) => expr_any(a, f) || expr_any(b, f),
        Expr::Func(_, args) => args.iter().any(|a| expr_any(a, f)),
        Expr::Case {
            when,
            rhs,
            then,
            otherwise,
            ..
        } => {
            expr_any(when, f)
                || expr_any(rhs, f)
                || expr_any(then, f)
                || otherwise.as_ref().is_some_and(|e| expr_any(e, f))
        }
    }
}

fn pred_any(p: &Pred, f: &dyn Fn(&Expr) -> bool) -> bool {
    match p {
        Pred::Cmp(a, _, b) => expr_any(a, f) || expr_any(b, f),
        Pred::Between(a, b, c, _) => expr_any(a, f) || expr_any(b, f) || expr_any(c, f),
        Pred::In(a, items, _) => expr_any(a, f) || items.iter().any(|i| expr_any(i, f)),
        Pred::IsNull(a, _) | Pred::Like(a, _, _) | Pred::BoolExpr(a) => expr_any(a, f),
        Pred::And(a, b) | Pred::Or(a, b) => pred_any(a, f) || pred_any(b, f),
        Pred::Not(a) | Pred::NullTest(a, _) => pred_any(a, f),
    }
}

fn expr_columns(e: &Expr, out: &mut Vec<usize>) {
    match e {
        Expr::Col(c) => out.push(*c),
        Expr::Lit(_) => {}
        Expr::Cast(a, _) | Expr::Extract(_, a) => expr_columns(a, out),
        Expr::Arith(a, _, b) => {
            expr_columns(a, out);
            expr_columns(b, out);
        }
        Expr::Func(_, args) => args.iter().for_each(|a| expr_columns(a, out)),
        Expr::Case {
            when,
            rhs,
            then,
            otherwise,
            ..
        } => {
            expr_columns(when, out);
            expr_columns(rhs, out);
            expr_columns(then, out);
            if let Some(e) = otherwise {
                expr_columns(e, out);
            }
        }
    }
}

fn pred_columns(p: &Pred, out: &mut Vec<usize>) {
    match p {
        Pred::Cmp(a, _, b) => {
            expr_columns(a, out);
            expr_columns(b, out);
        }
        Pred::Between(a, b, c, _) => {
            expr_columns(a, out);
            expr_columns(b, out);
            expr_columns(c, out);
        }
        Pred::In(a, items, _) => {
            expr_columns(a, out);
            items.iter().for_each(|i| expr_columns(i, out));
        }
        Pred::IsNull(a, _) | Pred::Like(a, _, _) | Pred::BoolExpr(a) => expr_columns(a, out),
        Pred::And(a, b) | Pred::Or(a, b) => {
            pred_columns(a, out);
            pred_columns(b, out);
        }
        Pred::Not(a) | Pred::NullTest(a, _) => pred_columns(a, out),
    }
}

/// One aggregate `(function, argument)` over a random column of `t`.
fn gen_agg(rng: &mut Rng, t: &Table) -> (String, Expr) {
    let c = rng.below(t.columns.len() as u64) as usize;
    let kind = t.column(c).ty.kind();
    let numeric = matches!(kind, Kind::Int | Kind::Float | Kind::Dec);
    let f = if numeric {
        if rng.chance(15) {
            // W8-B-006 (grouped ones are an engine error, so they run on the host).
            *rng.pick(&STAT_AGGS)
        } else {
            *rng.pick(&["count_star", "count", "sum", "min", "max", "avg"])
        }
    } else if kind == Kind::Bool {
        *rng.pick(&["count_star", "count"])
    } else {
        *rng.pick(&["count_star", "count", "min", "max"])
    };
    // Numeric aggregates sometimes take an expression (arithmetic, a cast such as
    // CAST(<decimal> AS DOUBLE)) of the same class.
    let arg = if numeric && rng.chance(30) {
        gen_value_expr(rng, t, kind, 1).map(|(e, _)| e)
    } else {
        None
    };
    (f.to_string(), arg.unwrap_or(Expr::Col(c)))
}

/// One or two output expressions for a wrapper: plain columns, and often a DECIMAL cast,
/// FLOAT arithmetic or integer arithmetic (the W4-B-004 / W5-A-002 / W8-B-001 shapes).
fn gen_select_exprs(rng: &mut Rng, t: &Table) -> Vec<Expr> {
    let ncols = t.columns.len() as u64;
    let n = rng.range(1, 2);
    let mut out: Vec<Expr> = (0..n)
        .map(|_| Expr::Col(rng.below(ncols) as usize))
        .collect();
    if rng.chance(60) {
        let kind = *rng.pick(&[Kind::Dec, Kind::Float, Kind::Int, Kind::Dec]);
        if let Some((e, _)) = gen_value_expr(rng, t, kind, 0) {
            out.push(e);
        }
    }
    out
}

/// W8-B-005 / W8-B-006: aggregates over an input that is empty (or not) without a filter.
fn gen_empty_agg(rng: &mut Rng, t: &Table) -> Shape {
    let ncols = t.columns.len() as u64;
    let n = rng.range(1, 3);
    let aggs = (0..n)
        .map(|_| {
            let c = rng.below(ncols) as usize;
            let f = match t.column(c).ty.kind() {
                Kind::Int | Kind::Float | Kind::Dec => *rng.pick(&[
                    "count_star",
                    "count",
                    "sum",
                    "sum",
                    "min",
                    "max",
                    "max",
                    "avg",
                    "stddev_pop",
                    "stddev_samp",
                    "var_pop",
                    "var_samp",
                ]),
                Kind::Bool => *rng.pick(&["count_star", "count"]),
                _ => *rng.pick(&["count_star", "count", "min", "max"]),
            };
            (f.to_string(), c)
        })
        .collect();
    let src = match rng.below(5) {
        0 => EmptySrc::Limit0,
        1 | 2 => EmptySrc::Offset {
            limit: *rng.pick(&[1usize, 5, 100]),
            offset: match rng.below(4) {
                0 => t.rows,
                1 => t.rows + 3,
                2 => t.rows / 2,
                _ => 0,
            },
        },
        3 => EmptySrc::Except,
        _ => EmptySrc::Intersect {
            empty: rng.chance(60),
        },
    };
    Shape::EmptyAgg { aggs, src }
}

/// An index from `preferred` other than `exclude`, else any column other than `exclude`.
fn pick_other(rng: &mut Rng, preferred: &[usize], exclude: usize, ncols: usize) -> Option<usize> {
    let pool: Vec<usize> = preferred
        .iter()
        .copied()
        .filter(|&c| c != exclude)
        .collect();
    let pool = if pool.is_empty() {
        (0..ncols).filter(|&c| c != exclude).collect()
    } else {
        pool
    };
    (!pool.is_empty()).then(|| *rng.pick(&pool))
}

/// An aggregate over a renamed column of `kind` (W4-B-001 class: MIN / MAX / SUM).
fn rename_agg(rng: &mut Rng, kind: Kind) -> &'static str {
    match kind {
        Kind::Int | Kind::Float | Kind::Dec => rng.pick(&["sum", "min", "max", "max", "count"]),
        Kind::Bool => "count",
        _ => rng.pick(&["min", "max", "count"]),
    }
}

/// W8-B-004: renames that give a nullable value the name of a NULL-free column.
fn gen_rename(rng: &mut Rng, t: &Table) -> Option<Shape> {
    let ncols = t.columns.len();
    if ncols < 2 {
        return None;
    }
    let nullable: Vec<usize> = (0..ncols).filter(|&i| t.column(i).null_pct > 0).collect();
    let null_free: Vec<usize> = (0..ncols).filter(|&i| t.column(i).null_pct == 0).collect();
    let grouped: Vec<usize> = (0..ncols)
        .filter(|&i| t.column(i).per_group && t.column(i).ty.kind() == Kind::Int)
        .collect();
    let src = if !nullable.is_empty() && rng.chance(80) {
        *rng.pick(&nullable)
    } else {
        rng.below(ncols as u64) as usize
    };
    let key = if !grouped.is_empty() && rng.chance(70) {
        *rng.pick(&grouped)
    } else {
        rng.below(ncols as u64) as usize
    };
    let limit = *rng.pick(&[1usize, 3, 5, 100, 3000]);
    let rename = match rng.below(4) {
        0 => {
            let alias = pick_other(rng, &null_free, key, ncols)?;
            Rename::Agg {
                key,
                src,
                alias,
                func: rename_agg(rng, t.column(src).ty.kind()),
            }
        }
        1 => {
            // Two distinct non-id columns (the `id` tie-breaker keeps its name).
            if ncols < 3 {
                return None;
            }
            let a = rng.range(1, ncols as i64 - 1) as usize;
            let mut b = rng.range(1, ncols as i64 - 1) as usize;
            if b == a {
                b = if a + 1 < ncols { a + 1 } else { 1 };
            }
            Rename::Swap {
                a,
                b,
                desc: rng.chance(50),
                limit,
            }
        }
        2 => Rename::Order {
            src,
            alias: *rng.pick(&null_free),
            desc: rng.chance(30),
            limit,
        },
        _ => {
            let a = pick_other(rng, &null_free, key, ncols)?;
            let ty = t.column(a).ty;
            let same: Vec<usize> = (0..ncols)
                .filter(|&i| i != key && t.column(i).ty == ty)
                .collect();
            let b = (!same.is_empty() && rng.chance(70)).then(|| *rng.pick(&same));
            Rename::Union {
                key,
                a,
                b,
                func: rename_agg(rng, ty.kind()),
            }
        }
    };
    Some(Shape::Rename(rename))
}

/// W8-B-003: host-orchestrated wrappers. The correlated shapes (which the engine may evaluate
/// per outer row) are drawn only for small tables, to keep the run fast.
fn gen_wrapped(rng: &mut Rng, t: &Table) -> Shape {
    let ncols = t.columns.len() as u64;
    let small = t.rows <= 50;
    let key = rng.below(ncols) as usize;
    let wrap = match rng.below(8) {
        0 => Wrap::DistinctOn {
            key,
            select: gen_select_exprs(rng, t),
        },
        1 if small => Wrap::Exists {
            key,
            negated: rng.chance(30),
            select: gen_select_exprs(rng, t),
        },
        2 if small => Wrap::ScalarSub {
            key,
            x: rng.below(ncols) as usize,
            select: gen_select_exprs(rng, t),
        },
        3 if small => Wrap::Lateral {
            key,
            x: rng.below(ncols) as usize,
        },
        4 => Wrap::Recursive {
            aggs: (0..rng.range(1, 2)).map(|_| gen_agg(rng, t)).collect(),
            depth: rng.range(0, 8),
            step: rng.range(1, 400),
        },
        5 => Wrap::Values {
            ids: (0..rng.range(1, 4)).map(|_| rng.range(-1, 3100)).collect(),
            select: gen_select_exprs(rng, t),
        },
        6 => Wrap::Series {
            start: rng.range(-5, 100),
            stop: rng.range(0, 3100),
            step: rng.range(1, 700),
            select: gen_select_exprs(rng, t),
        },
        _ => Wrap::CountDistinct {
            key,
            col: rng.below(ncols) as usize,
        },
    };
    Shape::Wrapped(wrap)
}

fn gen_query(rng: &mut Rng, t: &Table) -> Query {
    let ncols = t.columns.len();
    let mut pred = (!rng.chance(8)).then(|| gen_pred(rng, t, 0));
    let shape = match rng.below(20) {
        0..=6 => {
            let n = rng.range(1, ncols as i64 + 1) as usize;
            let mut select: Vec<Expr> = (0..n)
                .map(|_| Expr::Col(rng.below(ncols as u64) as usize))
                .collect();
            if rng.chance(30) {
                let kind = *rng.pick(&[Kind::Int, Kind::Float, Kind::Dec, Kind::Str, Kind::Date]);
                if let Some((e, _)) = gen_value_expr(rng, t, kind, 0) {
                    select.push(e);
                }
            }
            // W8-B-008: temporal arithmetic in projections.
            if rng.chance(12)
                && let Some((e, _)) = gen_temporal(rng, t)
            {
                select.push(e);
            }
            Shape::Filter { select }
        }
        7..=9 => Shape::Count,
        10..=12 => {
            // Mostly grouped; sometimes one aggregate row over the whole (filtered) table.
            let key = (!rng.chance(20)).then(|| rng.below(ncols as u64) as usize);
            let n = rng.range(1, 3);
            let aggs = (0..n).map(|_| gen_agg(rng, t)).collect();
            Shape::Group { key, aggs }
        }
        13..=15 => {
            let n = rng.range(1, ncols as i64) as usize;
            let select = (0..n).map(|_| rng.below(ncols as u64) as usize).collect();
            Shape::TopN {
                select,
                key: rng.below(ncols as u64) as usize,
                desc: rng.chance(50),
                nulls: match rng.below(5) {
                    0 => Some(true),
                    1 => Some(false),
                    _ => None,
                },
                limit: *rng.pick(&[0, 1, 5, 100, 3000]),
            }
        }
        16 => gen_empty_agg(rng, t),
        17 => gen_rename(rng, t).unwrap_or(Shape::Count),
        18 if rng.chance(50) => gen_having(rng, t),
        18 => gen_window(rng, t),
        _ => gen_wrapped(rng, t),
    };
    if matches!(
        shape,
        Shape::EmptyAgg { .. } | Shape::Rename(Rename::Union { .. })
    ) {
        pred = None;
    }
    Query { shape, pred }
}

/// W8-B-010 item 10: `GROUP BY k HAVING ...` over the grouped aggregates.
fn gen_having(rng: &mut Rng, t: &Table) -> Shape {
    let ncols = t.columns.len() as u64;
    let key = rng.below(ncols) as usize;
    let n = rng.range(1, 2);
    let aggs: Vec<(String, Expr)> = (0..n).map(|_| gen_agg(rng, t)).collect();
    let (f, e) = &aggs[0];
    let first = render_agg(f, e, t);
    // `> 0` only for aggregates with a numeric result.
    let numeric_arg = |e: &Expr| {
        let mut cols = Vec::new();
        expr_columns(e, &mut cols);
        cols.iter()
            .all(|&c| matches!(t.column(c).ty.kind(), Kind::Int | Kind::Float | Kind::Dec))
    };
    let numeric = f.starts_with("count") || numeric_arg(e);
    let having = match rng.below(4) {
        0 => format!("count(*) > {}", rng.pick(&[0, 1, 5, 20])),
        1 => format!("{first} IS NULL"),
        2 => format!("{first} IS NOT NULL"),
        _ if numeric => format!("{first} > 0"),
        _ => format!("count({}) >= 1", t.column(key).name),
    };
    Shape::Having { key, aggs, having }
}

/// W8-B-010 item 10: window functions with ties and nullable partition / order keys.
fn gen_window(rng: &mut Rng, t: &Table) -> Shape {
    let ncols = t.columns.len() as u64;
    let part = rng.below(ncols) as usize;
    let order = rng.below(ncols) as usize;
    let numeric: Vec<usize> = [columns_of(t, Kind::Int), columns_of(t, Kind::Float)].concat();
    let (func, arg) = if !numeric.is_empty() && rng.chance(50) {
        (
            *rng.pick(&["sum", "count", "min", "max", "avg"]),
            *rng.pick(&numeric),
        )
    } else {
        (*rng.pick(&["row_number", "rank", "dense_rank"]), 0)
    };
    Shape::Window {
        func,
        arg,
        part,
        order,
        desc: rng.chance(30),
    }
}

// ---------------------------------------------------------------------------------------------
// Differential check
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Target {
    BoltScan,
    Replacement,
    BoltQuery,
    /// Route-consistency oracle: `bolt_query` over the NULL-free table vs over its twin with a
    /// NULL in an unused column (see the header).
    Route,
}

/// `q0, q1, ...` for `n` output columns.
fn aliases(n: usize) -> String {
    (0..n)
        .map(|i| format!("q{i}"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Wrap `inner` (a query or a `bolt_query(...)` call) so floats are compared rounded.
fn normalized(inner: &str, is_subquery: bool, floats: &[bool]) -> String {
    let aliases = aliases(floats.len());
    let select = floats
        .iter()
        .enumerate()
        .map(|(i, &f)| {
            if f {
                // 6 decimals, or 12 significant digits for large values (sums and variances
                // of extreme values depend on the summation order in the last bits).
                let v = format!("CAST(q{i} AS DOUBLE)");
                let scale = format!("pow(10, floor(log10(abs({v}))) - 11)");
                format!(
                    "CASE WHEN NOT isfinite({v}) OR abs({v}) < 1e6 THEN round({v}, 6) \
                     ELSE round({v} / {scale}) * {scale} END"
                )
            } else {
                format!("q{i}")
            }
        })
        .collect::<Vec<_>>()
        .join(", ");
    let from = if is_subquery {
        format!("({inner})")
    } else {
        inner.to_string()
    };
    format!("SELECT {select} FROM {from} AS q({aliases})")
}

#[derive(Debug)]
enum Outcome {
    Same,
    /// Both sides raised an error.
    BothError,
    /// Only one side raised an expression error, and the other raises it too with DuckDB's
    /// optimizer disabled.
    OptimizerDependentError,
    /// Bolt raised an error DuckDB did not (`bolt_query`: counted as a skip).
    BoltError(String),
    /// DuckDB raised an error Bolt did not.
    MissingError(String),
    /// `n` rows in the symmetric difference.
    Differ(i64),
    /// Route oracle: the output column types differ between the two routes.
    TypeDiffer(String),
    /// Route oracle: only one of the two routes raised an error.
    RouteError(String),
}

/// Errors raised while evaluating an expression on some row.
fn is_evaluation_error(message: &str) -> bool {
    message.starts_with("Out of Range Error")
        || message.starts_with("Conversion Error")
        // Integer `/`, `//` and `%` by zero (W8-B-001 shapes): whether it is raised depends on
        // conjunct order and statistics just like overflow.
        || message.starts_with("Invalid Input Error: Division by zero")
}

fn compare(conn: &Connection, t: &Table, q: &Query, target: Target) -> Outcome {
    if target == Target::Route {
        return compare_routes(conn, t, q);
    }
    let floats = q.output_kinds(t);
    let (reference_sql, _) = q.render(t, SRC);
    let reference = normalized(&reference_sql, true, &floats);
    let actual = match target {
        Target::BoltScan => normalized(
            &q.render(t, &format!("bolt_scan('{BOLT}')")).0,
            true,
            &floats,
        ),
        Target::Replacement => normalized(&q.render(t, BOLT).0, true, &floats),
        Target::BoltQuery | Target::Route => {
            let sql = q.render(t, BOLT).0;
            normalized(
                &format!("bolt_query('{}')", sql.replace('\'', "''")),
                false,
                &floats,
            )
        }
    };
    let outcome = run_pair(conn, &reference, &actual);
    // Whether an expression error (overflow, failed cast) is raised depends on what DuckDB's
    // optimizer folds and in which order it evaluates conjuncts: the source table has
    // statistics and table filters, a Bolt scan has pushed (or declined) filters. Both are
    // legitimate DuckDB behaviors when the two sides agree with the optimizer disabled
    // (bolt_scan then receives no filters and DuckDB evaluates them in query order).
    let evaluation_error = match &outcome {
        Outcome::BoltError(m) | Outcome::MissingError(m) => is_evaluation_error(m),
        _ => false,
    };
    if evaluation_error {
        // `bolt_query` runs on the engine or in the host fallback's own DuckDB, which this
        // pragma does not reach; there it only changes the reference's evaluation order, which
        // is enough when the reference then agrees.
        exec(conn, "PRAGMA disable_optimizer").expect("disable_optimizer");
        let plain = run_pair(conn, &reference, &actual);
        exec(conn, "PRAGMA enable_optimizer").expect("enable_optimizer");
        let agrees = if target == Target::BoltQuery {
            matches!(plain, Outcome::Same)
        } else {
            matches!(plain, Outcome::Same | Outcome::BothError)
        };
        if agrees {
            return Outcome::OptimizerDependentError;
        }
        // The host fallback's DuckDB evaluates the conjuncts in its own order (W11-A). When
        // DuckDB itself, over the Bolt copy (`bolt_scan`, so with the same data and no table
        // statistics), returns the same rows without an error, the difference is DuckDB's
        // evaluation order and not a wrong result.
        if target == Target::BoltQuery && matches!(outcome, Outcome::MissingError(_)) {
            let over_copy = normalized(
                &q.render(t, &format!("bolt_scan('{BOLT}')")).0,
                true,
                &floats,
            );
            if matches!(run_pair(conn, &over_copy, &actual), Outcome::Same) {
                return Outcome::OptimizerDependentError;
            }
            // When the guard routed the query to the host fallback, the fallback is DuckDB over
            // the original column types (widened integers are mirrored with them), so a missing
            // evaluation error can only come from that DuckDB evaluating the conjuncts in another
            // order (e.g. a `nullif(..) = ..` conjunct before an unsigned subtraction). Storage-
            // typed arithmetic in the fallback is caught by the route oracle's type check and the
            // W4-B-002 tests instead.
            let sql = q.render(t, BOLT).0;
            let tier = query_string(
                conn,
                &format!("SELECT bolt_execution_tier('{}')", sql.replace('\'', "''")),
            );
            if matches!(tier.as_deref(), Ok("Host")) {
                return Outcome::OptimizerDependentError;
            }
        }
    }
    outcome
}

/// The DuckDB types of `table`'s columns, in order, separated by `|`.
fn column_types(conn: &Connection, table: &str) -> Result<String> {
    query_string(
        conn,
        &format!(
            "SELECT coalesce(string_agg(data_type, '|' ORDER BY column_index), '') \
             FROM duckdb_columns() WHERE table_name = '{table}'"
        ),
    )
}

/// Route-consistency oracle: run `q` through `bolt_query` over the NULL-free table (GPU
/// route) into `fz_route_a` and over its NULL twin (host route) into `fz_route_b`, without
/// normalizing; compare the column types, then the values (floating-point columns rounded).
fn compare_routes(conn: &Connection, t: &Table, q: &Query) -> Outcome {
    let floats = q.output_kinds(t);
    let names = aliases(floats.len());
    let raw = |from: &str| {
        let sql = q.render(t, from).0;
        format!(
            "SELECT * FROM bolt_query('{}') AS q({names})",
            sql.replace('\'', "''")
        )
    };
    let gpu = exec(
        conn,
        &format!("CREATE OR REPLACE TEMP TABLE fz_route_a AS {}", raw(BOLT)),
    );
    let host = exec(
        conn,
        &format!("CREATE OR REPLACE TEMP TABLE fz_route_b AS {}", raw(TWIN)),
    );
    match (gpu, host) {
        (Err(_), Err(_)) => return Outcome::BothError,
        (Ok(()), Err(e)) => {
            return Outcome::RouteError(format!(
                "only the NULL twin (host route) failed: {}",
                e.message
            ));
        }
        (Err(e), Ok(())) => {
            return Outcome::RouteError(format!(
                "only the NULL-free table (GPU route) failed: {}",
                e.message
            ));
        }
        (Ok(()), Ok(())) => {}
    }
    match (
        column_types(conn, "fz_route_a"),
        column_types(conn, "fz_route_b"),
    ) {
        (Ok(a), Ok(b)) if a != b => {
            return Outcome::TypeDiffer(format!("NULL-free table [{a}], NULL twin [{b}]"));
        }
        (Ok(_), Ok(_)) => {}
        (Err(e), _) | (_, Err(e)) => {
            return Outcome::BoltError(format!("type query failed: {}", e.message));
        }
    }
    run_pair(
        conn,
        &normalized("fz_route_a", false, &floats),
        &normalized("fz_route_b", false, &floats),
    )
}

/// After a [`Target::Route`] check: the output columns whose `bolt_query` type (GPU route,
/// `fz_route_a`) differs from DuckDB's own result type, as `"<duckdb> -> <bolt_query>"`.
fn duckdb_type_gaps(conn: &Connection, t: &Table, q: &Query) -> Result<Vec<String>> {
    let (reference_sql, n) = q.render(t, SRC);
    exec(
        conn,
        &format!(
            "CREATE OR REPLACE TEMP TABLE fz_route_ref AS SELECT * FROM ({reference_sql}) AS q({})",
            aliases(n)
        ),
    )?;
    let duck = column_types(conn, "fz_route_ref")?;
    let bolt = column_types(conn, "fz_route_a")?;
    let duck: Vec<&str> = duck.split('|').collect();
    let bolt: Vec<&str> = bolt.split('|').collect();
    if duck.len() != bolt.len() {
        return Ok(vec![format!(
            "{} columns -> {} columns",
            duck.len(),
            bolt.len()
        )]);
    }
    Ok(duck
        .iter()
        .zip(&bolt)
        .filter(|(d, b)| d != b)
        .map(|(d, b)| format!("{d} -> {b}"))
        .collect())
}

/// Run both sides once (each into a temporary table) and compare them as multisets.
fn run_pair(conn: &Connection, reference: &str, actual: &str) -> Outcome {
    let expected = exec(
        conn,
        &format!("CREATE OR REPLACE TEMP TABLE fz_expected AS {reference}"),
    );
    let got = exec(
        conn,
        &format!("CREATE OR REPLACE TEMP TABLE fz_actual AS {actual}"),
    );
    match (expected, got) {
        (Err(_), Err(_)) => Outcome::BothError,
        (Ok(()), Err(e)) => Outcome::BoltError(e.message),
        (Err(e), Ok(())) => Outcome::MissingError(e.message),
        (Ok(()), Ok(())) => {
            let diff = concat!(
                "SELECT count(*) FROM ((FROM fz_expected EXCEPT ALL FROM fz_actual) ",
                "UNION ALL (FROM fz_actual EXCEPT ALL FROM fz_expected))"
            );
            match query_i64(conn, diff) {
                Ok(0) => Outcome::Same,
                Ok(n) => Outcome::Differ(n),
                Err(e) => Outcome::BoltError(format!("diff query failed: {}", e.message)),
            }
        }
    }
}

fn is_failure(target: Target, outcome: &Outcome) -> bool {
    match outcome {
        Outcome::Same | Outcome::BothError | Outcome::OptimizerDependentError => false,
        // `bolt_query` runs only the SQL the engine supports: its errors are skips.
        Outcome::BoltError(_) => !matches!(target, Target::BoltQuery | Target::Route),
        Outcome::MissingError(_)
        | Outcome::Differ(_)
        | Outcome::TypeDiffer(_)
        | Outcome::RouteError(_) => true,
    }
}

/// Whether `bolt_query` is expected to agree with DuckDB on this query. Documented
/// differences (not defects of this extension) are excluded:
/// - IEEE NaN comparisons and wrapping integer arithmetic on the engine (W1-A-003): tables with
///   special values, and arithmetic or casts on tables with big values (W8-B-007 tables are
///   meant for sums, not for overflowing `c * 4` or `CAST(c AS INTEGER)`);
/// - `bolt_query` returns the engine's storage types (RFC-008 metadata is not attached to
///   engine results), so TIMESTAMPTZ columns come back as naive UTC timestamps: queries over
///   TIMESTAMPTZ columns;
/// - literals that only DuckDB types (`TIMESTAMPTZ '...'`, `'NaN'::DOUBLE`) are skipped by
///   the TIMESTAMPTZ rule and the special-table rule.
fn bolt_query_eligible(t: &Table, q: &Query) -> bool {
    !t.special
        && !q
            .columns()
            .iter()
            .any(|&c| t.column(c).ty.kind() == Kind::Tstz)
        && !q.render(t, BOLT).0.contains("TIMESTAMPTZ")
        && !(t.big && q.any_expr(&|e: &Expr| matches!(e, Expr::Arith(..) | Expr::Cast(..))))
}

/// Known, tracked failures: `Some(page)` when the failure matches one.
///
/// W4-B-001 .. W4-B-004 are fixed since W5-A and the W8-B guard holes since W9-A, so the
/// fuzzer checks them on every run. Match entries by the generator shape that exposes them
/// (on `bolt_query` and the route oracle only: `bolt_scan` and the replacement scan run these
/// shapes in DuckDB itself). Add an entry here (with its page) only for a failure that is
/// tracked.
fn known_issue(t: &Table, q: &Query, target: Target, outcome: &Outcome) -> Option<&'static str> {
    let _ = outcome;
    if !matches!(target, Target::BoltQuery | Target::Route) {
        return None;
    }
    let _ = (t, q);
    None
}

/// Every predicate obtained from `p` by replacing one AND / OR / NOT / IS NULL node with one
/// of its children.
fn reductions(p: &Pred) -> Vec<Pred> {
    let mut out = Vec::new();
    match p {
        Pred::And(a, b) | Pred::Or(a, b) => {
            out.push((**a).clone());
            out.push((**b).clone());
            let and = matches!(p, Pred::And(..));
            let make = |x: Pred, y: Pred| {
                if and {
                    Pred::And(Box::new(x), Box::new(y))
                } else {
                    Pred::Or(Box::new(x), Box::new(y))
                }
            };
            for r in reductions(a) {
                out.push(make(r, (**b).clone()));
            }
            for r in reductions(b) {
                out.push(make((**a).clone(), r));
            }
        }
        Pred::Not(a) => {
            out.push((**a).clone());
            out.extend(reductions(a).into_iter().map(|r| Pred::Not(Box::new(r))));
        }
        Pred::NullTest(a, neg) => {
            out.push((**a).clone());
            out.extend(
                reductions(a)
                    .into_iter()
                    .map(|r| Pred::NullTest(Box::new(r), *neg)),
            );
        }
        _ => {}
    }
    out
}

/// Shrink a failing query's predicate (drop it, or reduce it one node at a time) while the
/// failure persists.
fn minimize(conn: &Connection, t: &Table, q: &Query, target: Target) -> Query {
    let mut best = q.clone();
    for _ in 0..64 {
        let Some(pred) = &best.pred else {
            return best;
        };
        let mut candidates: Vec<Option<Pred>> = vec![None];
        candidates.extend(reductions(pred).into_iter().map(Some));
        let next = candidates.into_iter().find_map(|cand| {
            let trial = Query {
                shape: best.shape.clone(),
                pred: cand,
            };
            is_failure(target, &compare(conn, t, &trial, target)).then_some(trial)
        });
        match next {
            Some(trial) => best = trial,
            None => return best,
        }
    }
    best
}

#[derive(Default)]
struct Stats {
    queries: usize,
    tables: usize,
    /// Tables with NULLs (pushed SQL runs on the host fallback).
    null_tables: usize,
    checks: BTreeMap<&'static str, usize>,
    bolt_query_skips: usize,
    both_error: usize,
    optimizer_dependent: usize,
    build_secs: f64,
    secs: BTreeMap<String, f64>,
    known: BTreeMap<&'static str, usize>,
    skip_reasons: BTreeMap<String, usize>,
    /// Tables for which the route-consistency twin was built.
    twin_tables: usize,
    /// `"<duckdb type> -> <bolt_query type>"` differences found by the route oracle.
    type_gaps: BTreeMap<String, usize>,
}

fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name).ok().and_then(|v| v.trim().parse().ok())
}

fn env_u64(name: &str) -> Option<u64> {
    std::env::var(name).ok().and_then(|v| {
        let v = v.trim();
        v.strip_prefix("0x")
            .map(|h| u64::from_str_radix(h, 16).ok())
            .unwrap_or_else(|| v.parse().ok())
    })
}

/// First words of an error message, numbers and quoted parts removed (skip histogram key).
fn error_class(message: &str) -> String {
    let mut out = String::new();
    let mut in_quote = false;
    for ch in message.chars().take(90) {
        match ch {
            '\'' | '"' => in_quote = !in_quote,
            _ if in_quote => {}
            '0'..='9' => {}
            '\n' => break,
            _ => out.push(ch),
        }
    }
    out
}

#[test]
fn w4b_differential_fuzz() -> Result<()> {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let (_env, conn) = setup_connection()?;
    let cases = env_usize("BOLT_FUZZ_CASES").unwrap_or(DEFAULT_CASES);
    let base_seed = env_u64("BOLT_FUZZ_SEED").unwrap_or(DEFAULT_SEED);
    let verbose = std::env::var_os("BOLT_FUZZ_VERBOSE").is_some();
    let route_every = env_usize("BOLT_FUZZ_ROUTE_EVERY").unwrap_or(DEFAULT_ROUTE_EVERY);
    let strict_types = std::env::var_os("BOLT_FUZZ_STRICT_TYPES").is_some();
    let started = Instant::now();
    let mut stats = Stats::default();
    let mut failures: Vec<String> = Vec::new();
    let mut failure_classes: BTreeMap<String, usize> = BTreeMap::new();

    // `BOLT_FUZZ_TABLE_SEEDS=0xa,0xb`: replay only these table seeds (as printed in reports).
    let only: Vec<u64> = std::env::var("BOLT_FUZZ_TABLE_SEEDS")
        .map(|v| {
            v.split(',')
                .filter_map(|s| {
                    let s = s.trim();
                    s.strip_prefix("0x")
                        .map_or_else(|| s.parse().ok(), |h| u64::from_str_radix(h, 16).ok())
                })
                .collect()
        })
        .unwrap_or_default();
    let seeds: Vec<u64> = if only.is_empty() {
        (0..cases.div_ceil(QUERIES_PER_TABLE))
            .map(|i| base_seed ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
            .collect()
    } else {
        only
    };
    let cases = if seeds.len() * QUERIES_PER_TABLE < cases {
        seeds.len() * QUERIES_PER_TABLE
    } else {
        cases
    };
    for seed in seeds {
        let mut rng = Rng::new(seed);
        let table = gen_table(&mut rng);
        let build_started = Instant::now();
        build_table(&conn, &table).unwrap_or_else(|e| {
            panic!(
                "seed {seed:#x}: building the table failed: {}\n{table:?}",
                e.message
            )
        });
        stats.tables += 1;
        stats.build_secs += build_started.elapsed().as_secs_f64();
        stats.null_tables += usize::from(table.has_nulls());
        // The route oracle needs rows (for the twin's NULL) and no NULLs in the table itself
        // (so its `bolt_query` SQL may run on the GPU). The twin is built on first use.
        let route_table = route_every > 0 && table.rows > 0 && !table.has_nulls();
        let mut twin_built = false;
        for query_idx in 0..QUERIES_PER_TABLE {
            if stats.queries >= cases {
                break;
            }
            stats.queries += 1;
            let query = gen_query(&mut rng, &table);
            // The replacement scan binds `bolt_scan`; check it on every other query.
            let mut targets = vec![Target::BoltScan];
            if query_idx % 2 == 0 {
                targets.push(Target::Replacement);
            }
            let eligible = bolt_query_eligible(&table, &query);
            if eligible {
                targets.push(Target::BoltQuery);
            }
            if eligible && route_table && query_idx % route_every == route_every - 1 {
                if !twin_built {
                    let twin_started = Instant::now();
                    build_twin(&conn).unwrap_or_else(|e| {
                        panic!(
                            "seed {seed:#x}: building the twin failed: {}\n{table:?}",
                            e.message
                        )
                    });
                    twin_built = true;
                    stats.twin_tables += 1;
                    stats.build_secs += twin_started.elapsed().as_secs_f64();
                }
                targets.push(Target::Route);
            }
            for target in targets {
                let check_started = Instant::now();
                let outcome = compare(&conn, &table, &query, target);
                *stats.secs.entry(format!("{target:?}")).or_default() +=
                    check_started.elapsed().as_secs_f64();
                *stats
                    .checks
                    .entry(match target {
                        Target::BoltScan => "bolt_scan",
                        Target::Replacement => "replacement scan",
                        Target::BoltQuery => "bolt_query",
                        Target::Route => "route twin",
                    })
                    .or_default() += 1;
                if target == Target::Route && matches!(outcome, Outcome::Same) {
                    // Types of the GPU route vs DuckDB's own result types: counted (and
                    // failures only with BOLT_FUZZ_STRICT_TYPES; `bolt_query` returns the
                    // engine's types by design).
                    for gap in duckdb_type_gaps(&conn, &table, &query).unwrap_or_default() {
                        *stats.type_gaps.entry(gap.clone()).or_default() += 1;
                        if strict_types {
                            let class = format!("Route/type differs from DuckDB: {gap}");
                            let count = failure_classes.entry(class.clone()).or_default();
                            *count += 1;
                            if *count <= 2 {
                                failures.push(format!(
                                    "[{class}] seed {seed:#x} query {query_idx}\n  reference: {}",
                                    query.render(&table, SRC).0
                                ));
                            }
                        }
                    }
                }
                match &outcome {
                    Outcome::BothError => stats.both_error += 1,
                    Outcome::OptimizerDependentError => stats.optimizer_dependent += 1,
                    Outcome::BoltError(m) if target == Target::BoltQuery => {
                        stats.bolt_query_skips += 1;
                        *stats.skip_reasons.entry(error_class(m)).or_default() += 1;
                    }
                    _ => {}
                }
                if !is_failure(target, &outcome) {
                    continue;
                }
                if let Some(page) = known_issue(&table, &query, target, &outcome) {
                    *stats.known.entry(page).or_default() += 1;
                    continue;
                }
                if verbose {
                    eprintln!(
                        "FAIL seed {seed:#x} query {query_idx} {target:?}: {outcome:?}\n  {}",
                        query.render(&table, SRC).0
                    );
                }
                let class = format!(
                    "{target:?}/{}",
                    match &outcome {
                        Outcome::BoltError(m) => format!("error: {}", error_class(m)),
                        Outcome::MissingError(m) => format!("missing error: {}", error_class(m)),
                        Outcome::Differ(n) =>
                            format!("wrong result ({} rows)", if *n > 0 { "some" } else { "no" }),
                        Outcome::TypeDiffer(m) => format!("type differs: {m}"),
                        Outcome::RouteError(m) => format!("route error: {}", error_class(m)),
                        _ => unreachable!(),
                    }
                );
                let count = failure_classes.entry(class.clone()).or_default();
                *count += 1;
                // Minimize and report the first few of each class.
                if *count <= 2 {
                    let small = minimize(&conn, &table, &query, target);
                    let outcome = compare(&conn, &table, &small, target);
                    failures.push(format!(
                        "[{class}] seed {seed:#x} query {query_idx}: {outcome:?}\n  table: rows={} splits={:?} special={} big={} groups={} tz={} columns={:?}\n  reference: {}\n  minimized: {}",
                        table.rows,
                        table.splits,
                        table.special,
                        table.big,
                        table.groups,
                        table.time_zone,
                        table
                            .columns
                            .iter()
                            .map(|c| format!("{} {} nulls={}% wide={} per_group={}", c.name, c.ty.sql(), c.null_pct, c.wide_strings, c.per_group))
                            .collect::<Vec<_>>(),
                        query.render(&table, SRC).0,
                        small.render(&table, SRC).0,
                    ));
                }
            }
        }
    }
    exec(&conn, "SET TimeZone = 'UTC'")?;
    exec(&conn, "PRAGMA bolt_clear_tables")?;

    eprintln!(
        "w4b fuzz: backend={} seed={base_seed:#x} tables={} (with NULLs: {}) queries={} checks={:?} both-error={} optimizer-dependent-error={} \
         bolt_query skips={} known={:?} failures={} in {:.1}s",
        if gpu() { "cuda-gpu" } else { "host" },
        stats.tables,
        stats.null_tables,
        stats.queries,
        stats.checks,
        stats.both_error,
        stats.optimizer_dependent,
        stats.bolt_query_skips,
        stats.known,
        failure_classes.values().sum::<usize>(),
        started.elapsed().as_secs_f64()
    );
    eprintln!(
        "w4b fuzz timing: table builds {:.1}s, checks {:?}",
        stats.build_secs, stats.secs
    );
    eprintln!(
        "w4b fuzz route oracle: twin tables={} bolt_query type differences from DuckDB={} (kinds: {})",
        stats.twin_tables,
        stats.type_gaps.values().sum::<usize>(),
        stats.type_gaps.len()
    );
    if verbose {
        for (reason, n) in &stats.skip_reasons {
            eprintln!("  bolt_query skip x{n}: {reason}");
        }
        for (gap, n) in &stats.type_gaps {
            eprintln!("  bolt_query type x{n}: {gap}");
        }
    }
    assert!(
        failures.is_empty(),
        "{} failure class(es) {failure_classes:?}:\n{}",
        failure_classes.len(),
        failures.join("\n\n")
    );
    Ok(())
}

/// The generator is deterministic: the same seed builds the same table and queries.
#[test]
fn w4b_fuzz_generator_is_deterministic() {
    let render = |seed: u64| {
        let mut rng = Rng::new(seed);
        let t = gen_table(&mut rng);
        let qs: Vec<String> = (0..20)
            .map(|_| gen_query(&mut rng, &t).render(&t, SRC).0)
            .collect();
        (format!("{t:?}"), qs)
    };
    assert_eq!(render(42), render(42));
    assert_ne!(render(42), render(43));
    assert_eq!(date_text(0), "1970-01-01");
    assert_eq!(date_text(-1), "1969-12-31");
    assert_eq!(date_text(19_782), "2024-02-29");
    assert_eq!(ts_text(-1), "1969-12-31 23:59:59.999999");
    assert_eq!(
        ts_text(86_400_000_000 + 3_723_000_000),
        "1970-01-02 01:02:03"
    );
}

/// Routing oracle (W8-B-010 item 11): shapes that are correct on the engine over NULL-free
/// tables must keep running there. This catches over-routing by the guard's checks before it
/// costs users GPU execution. With a GPU only (without one everything runs on the host).
#[test]
fn w4b_safe_shapes_stay_on_the_gpu() -> Result<()> {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|p| p.into_inner());
    let (_env, conn) = setup_connection()?;
    if !gpu() {
        return Ok(());
    }
    exec(
        &conn,
        "CREATE OR REPLACE TABLE fz_route AS SELECT i::INTEGER AS id, (i % 7)::INTEGER AS g, \
         (i * 3)::BIGINT AS b, i / 8 AS d, 's' || (i % 50) AS s, DATE '2026-01-01' + (i % 30)::INTEGER AS dt \
         FROM range(0, 5000) r(i)",
    )?;
    exec(
        &conn,
        "COPY fz_route TO 'fz_route_b' (FORMAT bolt, USE_TMP_FILE FALSE)",
    )?;
    let mut over_routed = Vec::new();
    for sql in [
        "SELECT * FROM fz_route_b",
        "SELECT id, b, s FROM fz_route_b WHERE id > 100 AND g = 3",
        "SELECT g, count(*) FROM fz_route_b GROUP BY g",
        "SELECT g, count(*), sum(b), min(id), max(d), avg(b) FROM fz_route_b GROUP BY g",
        "SELECT sum(b), max(id), avg(d) FROM fz_route_b",
        "SELECT count(*) FROM fz_route_b WHERE s = 's7'",
        "SELECT id, b + 1, b * 2, d * 0.5 FROM fz_route_b",
        "SELECT id FROM fz_route_b ORDER BY b DESC LIMIT 10",
        "SELECT DISTINCT g FROM fz_route_b",
        "SELECT a.id, z.b FROM fz_route_b a JOIN fz_route_b z ON a.id = z.id WHERE a.g = 1",
        "SELECT id, b FROM fz_route_b WHERE dt >= DATE '2026-01-10'",
        "SELECT g, b FROM fz_route_b UNION ALL SELECT g, b FROM fz_route_b",
    ] {
        let explain = query_string(
            &conn,
            &format!("SELECT bolt_explain('{}')", sql.replace('\'', "''")),
        )?;
        let tier = query_string(
            &conn,
            &format!("SELECT bolt_execution_tier('{}')", sql.replace('\'', "''")),
        )?;
        if explain.contains("Executed on the host fallback") {
            over_routed.push(format!("{sql}: {}", explain.lines().next().unwrap_or("")));
        } else if !matches!(tier.as_str(), "Gpu" | "Hybrid") {
            over_routed.push(format!("{sql}: bolt_execution_tier = {tier}"));
        } else {
            eprintln!("w4b routing: {tier:6} {sql}");
        }
    }
    exec(&conn, "PRAGMA bolt_clear_tables")?;
    assert!(over_routed.is_empty(), "{over_routed:#?}");
    Ok(())
}

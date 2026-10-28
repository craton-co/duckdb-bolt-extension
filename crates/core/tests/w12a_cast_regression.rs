// SPDX-License-Identifier: Apache-2.0

//! Wave 12 (W12-A) regression tests for the two `bolt_query` wrong-result classes the W4-B
//! fuzzer found on seed `0xfee10011`: a decimal cast that lowers the precision added one unit in
//! the last place (arrow-cast 53.4.1, W12-A-001), and the engine truncates floating-point to
//! integer casts (W12-A-002). Both backends; with a GPU the routing is checked too.

use std::sync::{Mutex, Once};

use duckdb_bolt_core::runtime::BoltRuntime;
use duckdb_loadable_macros::duckdb_entrypoint_c_api_v2;
use duckdb_neo::{
    Parameters, Result,
    connection::Connection,
    environment::{Environment, StorageLocation},
    extension::{Extension, register_static_extension},
};

#[duckdb_entrypoint_c_api_v2(extension_name = "bolt", api_version = "2.0.0")]
fn bolt_w12a_cast_entrypoint(extension: &Extension) -> Result<()> {
    duckdb_bolt_core::register(extension)
}

static REGISTER_ONCE: Once = Once::new();
// Serialize tests because Craton Bolt's runtime catalog is process-wide.
static TEST_MUTEX: Mutex<()> = Mutex::new(());

fn lock() -> std::sync::MutexGuard<'static, ()> {
    TEST_MUTEX.lock().unwrap_or_else(|p| p.into_inner())
}

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

fn exec(conn: &Connection, sql: &str) -> Result<()> {
    conn.execute(sql, Parameters::None).map(|_| ())
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
    Ok(vec
        .reader()?
        .get(0)?
        .map(|s| s.to_string())
        .unwrap_or_default())
}

fn gpu() -> bool {
    BoltRuntime::global().telemetry().unwrap().backend == "cuda-gpu"
}

/// `CREATE TABLE <src> AS <select>` and its Bolt copy `<bolt>`.
fn build(conn: &Connection, src: &str, bolt: &str, select: &str) -> Result<()> {
    exec(conn, &format!("CREATE OR REPLACE TABLE {src} AS {select}"))?;
    exec(
        conn,
        &format!("COPY {src} TO '{bolt}' (FORMAT bolt, USE_TMP_FILE FALSE)"),
    )
}

/// `bolt_query(<sql>)` as a SQL string literal argument.
fn bolt_query(sql: &str) -> String {
    format!("FROM bolt_query('{}')", sql.replace('\'', "''"))
}

/// Asserts that `bolt_query(<sql over bolt>)` returns the rows of `<sql over src>` (as a
/// multiset). `{t}` in `sql` names the table.
fn assert_same(conn: &Connection, sql: &str, src: &str, bolt: &str) -> Result<()> {
    let reference = sql.replace("{t}", src);
    let actual = bolt_query(&sql.replace("{t}", bolt));
    exec(
        conn,
        &format!("CREATE OR REPLACE TEMP TABLE w12a_expected AS {reference}"),
    )?;
    exec(
        conn,
        &format!("CREATE OR REPLACE TEMP TABLE w12a_actual AS {actual}"),
    )?;
    let differing = query_i64(
        conn,
        "SELECT count(*) FROM ((FROM w12a_expected EXCEPT ALL FROM w12a_actual) \
         UNION ALL (FROM w12a_actual EXCEPT ALL FROM w12a_expected))",
    )?;
    if differing != 0 {
        let show = |table: &str| {
            query_string(
                conn,
                &format!("SELECT CAST(list(x) AS VARCHAR) FROM {table} x"),
            )
            .unwrap_or_else(|e| e.message)
        };
        panic!(
            "{sql}: DuckDB {} vs bolt_query {}",
            show("w12a_expected"),
            show("w12a_actual")
        );
    }
    Ok(())
}

/// With a GPU: why the crash guard routes `sql` to the host fallback (`None`: it runs on the
/// engine). Read from `bolt_explain`.
fn host_reason(sql: &str) -> Option<String> {
    let text = BoltRuntime::global().explain_sql(sql).expect("explain");
    let marker = "Executed on the host fallback (DuckDB): ";
    text.find(marker).map(|at| {
        let rest = &text[at + marker.len()..];
        rest[..rest.find(" ===").unwrap_or(rest.len())].to_string()
    })
}

fn routed_to_host(sql: &str) -> bool {
    host_reason(sql).is_some()
}

#[test]
fn decimal_precision_narrowing_keeps_the_value() -> Result<()> {
    // W12-A-001 (W4-B fuzzer, seed 0xfee10011, table 0xe53eb6eaeeb6d3ab): arrow-cast 53.4.1
    // adds one unit in the last place when a decimal cast keeps the scale and lowers the
    // precision. `coalesce(<USMALLINT>, 21.25)` is DECIMAL(7, 2) in DuckDB, the host fallback
    // computed it over the INTEGER mirror as DECIMAL(12, 2), and the cast to the bound type
    // returned 21.26.
    use arrow_array::{Array, ArrayRef, Decimal128Array};
    use arrow_schema::DataType;
    let _guard = lock();
    let (_env, conn) = setup_connection()?;
    build(
        &conn,
        "w12a_q",
        "w12a_qb",
        "SELECT i::INTEGER AS id, NULL::USMALLINT AS c0, \
         CASE WHEN i % 5 = 0 THEN NULL ELSE i END::BIGINT AS c2, (i % 3 = 0) AS c3, \
         (i * 7)::USMALLINT AS c5 FROM range(0, 2049) r(i)",
    )?;
    for sql in [
        "SELECT c2, c3, c2, coalesce(c0, 21.25) FROM {t}",
        "SELECT coalesce(c5, 21.25) AS x, coalesce(c2, -0.05) AS y FROM {t}",
        "SELECT id, CAST(c2 AS DECIMAL(9, 2)) AS d FROM {t}",
    ] {
        assert_same(&conn, sql, "w12a_q", "w12a_qb")?;
    }
    // The cast itself, for every scale direction.
    let cast = |v: i128, from: (u8, i8), to: (u8, i8)| -> Option<i128> {
        let a: ArrayRef = std::sync::Arc::new(
            Decimal128Array::from(vec![Some(v)])
                .with_precision_and_scale(from.0, from.1)
                .unwrap(),
        );
        duckdb_bolt_core::arrow_bridge::cast_array_to_exact_type(
            &a,
            &DataType::Decimal128(to.0, to.1),
        )
        .ok()
        .map(|c| {
            c.as_any()
                .downcast_ref::<Decimal128Array>()
                .unwrap()
                .value(0)
        })
    };
    assert_eq!(cast(2125, (12, 2), (7, 2)), Some(2125));
    assert_eq!(cast(-2125, (12, 2), (4, 2)), Some(-2125));
    assert_eq!(cast(12345, (38, 0), (18, 0)), Some(12345));
    assert_eq!(cast(2125, (12, 2), (7, 3)), Some(21250));
    assert_eq!(cast(21255, (12, 3), (7, 2)), Some(2126));
    assert_eq!(cast(-21255, (12, 3), (7, 2)), Some(-2126));
    assert_eq!(cast(21254, (12, 3), (7, 2)), Some(2125));
    assert_eq!(cast(99_999, (12, 2), (4, 2)), None, "overflow is an error");
    exec(&conn, "PRAGMA bolt_clear_tables")?;
    Ok(())
}

#[test]
fn float_to_integer_casts_round_like_duckdb() -> Result<()> {
    // W12-A-002 (W4-B fuzzer, seed 0xfee10011, table 0xd5b1333837279104): the engine truncates
    // `CAST(<FLOAT or DOUBLE> AS INTEGER)`, DuckDB rounds (`3.7` -> `4`).
    let _guard = lock();
    let (_env, conn) = setup_connection()?;
    build(
        &conn,
        "w12a_p",
        "w12a_pb",
        "SELECT i::INTEGER AS id, (i * 7)::BIGINT AS c4, (i * 0.37)::FLOAT AS c2, \
         (i * -0.37)::DOUBLE AS d FROM range(0, 3000) r(i)",
    )?;
    for sql in [
        "SELECT id, c4, c4, id, c4, CAST(c2 AS INTEGER) FROM {t}",
        "SELECT CAST(d AS BIGINT) AS x, CAST(c2 AS SMALLINT) AS y FROM {t}",
        "SELECT count(*) FROM {t} WHERE CAST(d AS INTEGER) = -4",
    ] {
        assert_same(&conn, sql, "w12a_p", "w12a_pb")?;
    }
    if gpu() {
        assert!(routed_to_host("SELECT CAST(c2 AS INTEGER) FROM w12a_pb"));
        // Casts to floating point, and plain columns, stay on the GPU.
        assert!(!routed_to_host(
            "SELECT CAST(id AS DOUBLE), c4 FROM w12a_pb"
        ));
    }
    exec(&conn, "PRAGMA bolt_clear_tables")?;
    Ok(())
}

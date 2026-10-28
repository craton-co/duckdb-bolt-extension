// SPDX-License-Identifier: Apache-2.0

//! Wave 5 (W5-A) regression tests for the engine result hazards the W4-B differential fuzzer
//! found through `bolt_query` / `bolt_sql` / `bolt_aggregate`:
//!
//! - W4-B-001: grouped MIN / MAX / SUM of an all-NULL group must be NULL;
//! - W4-B-002: arithmetic on integers stored widened (TINYINT, unsigned, ...) must overflow as
//!   in DuckDB;
//! - W4-B-003: ascending ORDER BY puts NULLs last (DuckDB's default);
//! - W4-B-004: `CAST(<negative DECIMAL> AS DOUBLE)` must be exact.
//!
//! Every check compares `bolt_query` with the same query on the DuckDB source table, on both
//! backends. With a GPU the routing is checked too: hazards run on the host fallback, the
//! shapes that stay safe keep running on the engine.

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
fn bolt_w5a_hazards_entrypoint(extension: &Extension) -> Result<()> {
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

/// Rows in the symmetric difference of `bolt_query(<sql over bolt>)` and `<sql over src>`.
fn differing_rows(conn: &Connection, sql: &str, src: &str, bolt: &str) -> Result<i64> {
    let reference = sql.replace("{t}", src);
    let actual = format!(
        "FROM bolt_query('{}')",
        sql.replace("{t}", bolt).replace('\'', "''")
    );
    exec(
        conn,
        &format!("CREATE OR REPLACE TEMP TABLE w5a_expected AS {reference}"),
    )?;
    exec(
        conn,
        &format!("CREATE OR REPLACE TEMP TABLE w5a_actual AS {actual}"),
    )?;
    query_i64(
        conn,
        "SELECT count(*) FROM ((FROM w5a_expected EXCEPT ALL FROM w5a_actual) \
         UNION ALL (FROM w5a_actual EXCEPT ALL FROM w5a_expected))",
    )
}

/// With a GPU: whether the crash guard routes `sql` to the host fallback.
fn routed_to_host(sql: &str) -> bool {
    BoltRuntime::global()
        .explain_sql(sql)
        .expect("explain")
        .contains("Executed on the host fallback")
}

#[test]
fn grouped_min_max_sum_of_an_all_null_group_is_null() -> Result<()> {
    let _guard = lock();
    let (_env, conn) = setup_connection()?;
    build(
        &conn,
        "w5a_s",
        "w5a_sb",
        "SELECT 0::INTEGER AS id, 5::TINYINT AS c, 1.5::DOUBLE AS d, 2 AS g, 7 AS k \
         UNION ALL SELECT 1, NULL, NULL, 3, 8",
    )?;
    let sql = "SELECT g, max(c), min(c), sum(c), max(d), min(d), sum(d) FROM {t} GROUP BY g";
    assert_eq!(differing_rows(&conn, sql, "w5a_s", "w5a_sb")?, 0);
    let nulls = query_i64(
        &conn,
        "SELECT count(*) FROM bolt_query('SELECT g, max(c) AS m, sum(d) AS s FROM w5a_sb \
         GROUP BY g') WHERE g = 3 AND m IS NULL AND s IS NULL",
    )?;
    assert_eq!(nulls, 1, "the all-NULL group must aggregate to NULL");
    if gpu() {
        assert!(routed_to_host("SELECT g, max(c) FROM w5a_sb GROUP BY g"));
        // NULL-free aggregated column, COUNT / AVG, and ungrouped aggregates stay on the GPU.
        assert!(!routed_to_host(
            "SELECT g, max(k), sum(k) FROM w5a_sb GROUP BY g"
        ));
        assert!(!routed_to_host(
            "SELECT g, count(c), avg(d) FROM w5a_sb GROUP BY g"
        ));
        assert!(!routed_to_host("SELECT max(c), sum(d) FROM w5a_sb"));
    }
    exec(&conn, "PRAGMA bolt_clear_tables")?;
    Ok(())
}

#[test]
fn ascending_order_by_puts_nulls_last() -> Result<()> {
    let _guard = lock();
    let (_env, conn) = setup_connection()?;
    build(
        &conn,
        "w5a_d",
        "w5a_db",
        "SELECT i::INTEGER AS id, CASE WHEN i % 20 = 0 THEN NULL ELSE 100 - i END AS v, \
         CASE WHEN i % 7 = 0 THEN NULL ELSE 'v' || i END AS s FROM range(0, 50) t(i)",
    )?;
    for sql in [
        "SELECT id, v FROM {t} ORDER BY v, id LIMIT 3",
        "SELECT id, s FROM {t} ORDER BY s, id LIMIT 5",
        "SELECT id, v FROM {t} ORDER BY v DESC, id LIMIT 3",
        "SELECT id, v FROM {t} ORDER BY v NULLS FIRST, id LIMIT 3",
        "SELECT id, v FROM {t} ORDER BY v NULLS LAST, id LIMIT 3",
    ] {
        assert_eq!(differing_rows(&conn, sql, "w5a_d", "w5a_db")?, 0, "{sql}");
    }
    let first = query_i64(
        &conn,
        "SELECT id::BIGINT FROM bolt_query('SELECT id, v FROM w5a_db ORDER BY v, id LIMIT 1')",
    )?;
    assert_eq!(first, 49, "the smallest non-NULL value comes first");
    if gpu() {
        assert!(routed_to_host(
            "SELECT id, v FROM w5a_db ORDER BY v, id LIMIT 3"
        ));
        // DESC, explicit NULLS LAST and keys without NULLs stay on the GPU.
        assert!(!routed_to_host(
            "SELECT id, v FROM w5a_db ORDER BY v DESC LIMIT 3"
        ));
        assert!(!routed_to_host(
            "SELECT id, v FROM w5a_db ORDER BY v NULLS LAST LIMIT 3"
        ));
        assert!(!routed_to_host(
            "SELECT id, v FROM w5a_db ORDER BY id LIMIT 3"
        ));
    }
    exec(&conn, "PRAGMA bolt_clear_tables")?;
    Ok(())
}

#[test]
fn negative_decimal_casts_are_exact() -> Result<()> {
    let _guard = lock();
    let (_env, conn) = setup_connection()?;
    build(
        &conn,
        "w5a_a",
        "w5a_ab",
        "SELECT i::INTEGER AS id, CAST(-i / 4.0 AS DECIMAL(18, 4)) AS n, \
         CAST(-i / 4.0 AS DECIMAL(9, 2)) AS n9 FROM range(0, 2049) t(i)",
    )?;
    for sql in [
        "SELECT id, n, CAST(n AS DOUBLE), CAST(n9 AS DOUBLE) FROM {t}",
        "SELECT id, n + 1, n9 - 0.5 FROM {t}",
        "SELECT id FROM {t} WHERE CAST(n AS DOUBLE) < -100.0",
        "SELECT sum(CAST(n9 AS DOUBLE)) FROM {t}",
    ] {
        assert_eq!(differing_rows(&conn, sql, "w5a_a", "w5a_ab")?, 0, "{sql}");
    }
    if gpu() {
        assert!(routed_to_host("SELECT CAST(n AS DOUBLE) FROM w5a_ab"));
        assert!(routed_to_host("SELECT n + 1 FROM w5a_ab"));
        // Plain DECIMAL columns, and casts of other types, stay on the GPU.
        assert!(!routed_to_host("SELECT id, n FROM w5a_ab"));
        assert!(!routed_to_host("SELECT CAST(id AS DOUBLE) FROM w5a_ab"));
    }
    exec(&conn, "PRAGMA bolt_clear_tables")?;
    Ok(())
}

#[test]
fn arithmetic_on_widened_integers_overflows_like_duckdb() -> Result<()> {
    let _guard = lock();
    let (_env, conn) = setup_connection()?;
    build(
        &conn,
        "w5a_u",
        "w5a_ub",
        "SELECT i::INTEGER AS id, (i % 5)::UTINYINT AS c0, (i * 10)::TINYINT AS t, \
         i::UINTEGER AS u FROM range(10) t(i)",
    )?;
    for sql in [
        "SELECT count(*) FROM w5a_ub WHERE c0 - 2 = -4.5",
        "SELECT t * 2 FROM w5a_ub",
        "SELECT u - 5 FROM w5a_ub",
    ] {
        let err = conn
            .execute(
                format!("FROM bolt_query('{}')", sql.replace('\'', "''")),
                Parameters::None,
            )
            .expect_err(sql);
        assert!(
            err.message.contains("Out of Range Error") || err.message.contains("Overflow"),
            "{sql}: {}",
            err.message
        );
        // DuckDB raises it too.
        assert!(
            conn.execute(sql.replace("w5a_ub", "w5a_u"), Parameters::None)
                .is_err()
        );
    }
    // In-range arithmetic and plain reads agree, with the storage types bolt_query reports.
    for sql in [
        "SELECT id, c0 + 1, t + 1, u + 1 FROM {t}",
        "SELECT c0, count(*) FROM {t} GROUP BY c0",
        "SELECT id, CAST(t AS INTEGER) * 2 FROM {t}",
    ] {
        assert_eq!(differing_rows(&conn, sql, "w5a_u", "w5a_ub")?, 0, "{sql}");
    }
    if gpu() {
        assert!(routed_to_host("SELECT c0 + 1 FROM w5a_ub"));
        assert!(!routed_to_host("SELECT c0, id + 1 FROM w5a_ub"));
    }
    exec(&conn, "PRAGMA bolt_clear_tables")?;
    Ok(())
}

#[test]
fn ungrouped_count_star_with_integer_sum_is_exact() -> Result<()> {
    let _guard = lock();
    let (_env, conn) = setup_connection()?;
    build(
        &conn,
        "w5a_c",
        "w5a_cb",
        "SELECT i::INTEGER AS id, i::BIGINT AS b, (i % 3)::INTEGER AS g FROM range(0, 5000) t(i)",
    )?;
    for sql in [
        "SELECT count(*), sum(id) FROM {t}",
        "SELECT sum(g), count(1) FROM {t}",
        "SELECT count(*), sum(id), sum(b), avg(id) FROM {t} WHERE id > 10",
        "SELECT g, count(*), sum(id) FROM {t} GROUP BY g",
    ] {
        assert_eq!(differing_rows(&conn, sql, "w5a_c", "w5a_cb")?, 0, "{sql}");
    }
    if gpu() {
        assert!(routed_to_host("SELECT count(*), sum(id) FROM w5a_cb"));
        // count(col), Int64 sums and grouped aggregates stay on the GPU.
        assert!(!routed_to_host("SELECT count(id), sum(id) FROM w5a_cb"));
        assert!(!routed_to_host("SELECT count(*), sum(b) FROM w5a_cb"));
        assert!(!routed_to_host(
            "SELECT g, count(*), sum(id) FROM w5a_cb GROUP BY g"
        ));
    }
    exec(&conn, "PRAGMA bolt_clear_tables")?;
    Ok(())
}

#[test]
fn float_arithmetic_is_computed_in_float() -> Result<()> {
    let _guard = lock();
    let (_env, conn) = setup_connection()?;
    build(
        &conn,
        "w5a_f",
        "w5a_fb",
        "SELECT i::INTEGER AS id, ((i % 321) - 160)::FLOAT / 8 AS f, i::DOUBLE / 8 AS d \
         FROM range(0, 600) t(i)",
    )?;
    for sql in [
        "SELECT id, f / 3.25, f * 0.75 FROM {t}",
        "SELECT id FROM {t} WHERE f / 3.25 > 1.5",
        "SELECT id, d / 3.25 FROM {t}",
    ] {
        assert_eq!(differing_rows(&conn, sql, "w5a_f", "w5a_fb")?, 0, "{sql}");
    }
    if gpu() {
        assert!(routed_to_host("SELECT f / 3.25 FROM w5a_fb"));
        assert!(!routed_to_host("SELECT d / 3.25, f FROM w5a_fb"));
    }
    exec(&conn, "PRAGMA bolt_clear_tables")?;
    Ok(())
}

#[test]
fn ungrouped_sum_min_max_over_an_empty_input_is_null() -> Result<()> {
    let _guard = lock();
    let (_env, conn) = setup_connection()?;
    build(
        &conn,
        "w5a_e",
        "w5a_eb",
        "SELECT i::INTEGER AS id, i::BIGINT AS b, i::DOUBLE AS d FROM range(0, 600) t(i)",
    )?;
    build(&conn, "w5a_z", "w5a_zb", "FROM w5a_e LIMIT 0")?;
    for (sql, src, bolt) in [
        (
            "SELECT min(b), max(b), sum(b), min(d), max(d), sum(d) FROM {t} WHERE id = -5",
            "w5a_e",
            "w5a_eb",
        ),
        (
            "SELECT min(id), max(id), sum(id) FROM {t} WHERE id < 0",
            "w5a_e",
            "w5a_eb",
        ),
        (
            "SELECT count(*), max(b), min(d) FROM {t}",
            "w5a_z",
            "w5a_zb",
        ),
        ("SELECT max(b), sum(d) FROM {t}", "w5a_z", "w5a_zb"),
    ] {
        assert_eq!(differing_rows(&conn, sql, src, bolt)?, 0, "{sql}");
    }
    if gpu() {
        assert!(routed_to_host("SELECT max(b) FROM w5a_eb WHERE id = -5"));
        assert!(routed_to_host("SELECT max(b) FROM w5a_zb"));
        // Unfiltered aggregates of a non-empty table and filtered counts stay on the GPU.
        assert!(!routed_to_host("SELECT max(b), sum(d) FROM w5a_eb"));
        assert!(!routed_to_host("SELECT count(*) FROM w5a_eb WHERE id = -5"));
    }
    exec(&conn, "PRAGMA bolt_clear_tables")?;
    Ok(())
}

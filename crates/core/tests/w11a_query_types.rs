// SPDX-License-Identifier: Apache-2.0

//! Wave 11 (W11-A) regression tests for `bolt_query` result types (W9-A-001) and INTERVAL
//! results (W9-A-002): `bolt_query` binds DuckDB's own result types, so `7 / 2` is `3.5`,
//! `SUM(BIGINT)` is a `HUGEINT`, `SUM(FLOAT)` a `DOUBLE`, widened integer columns keep their
//! original types and `ts - ts` is an `INTERVAL`, on both backends and both routes.

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
fn bolt_w11a_types_entrypoint(extension: &Extension) -> Result<()> {
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
        &format!("CREATE OR REPLACE TEMP TABLE w11a_expected AS {reference}"),
    )?;
    exec(
        conn,
        &format!("CREATE OR REPLACE TEMP TABLE w11a_actual AS {actual}"),
    )?;
    let differing = query_i64(
        conn,
        "SELECT count(*) FROM ((FROM w11a_expected EXCEPT ALL FROM w11a_actual) \
         UNION ALL (FROM w11a_actual EXCEPT ALL FROM w11a_expected))",
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
            show("w11a_expected"),
            show("w11a_actual")
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

/// DuckDB's column types of `<sql over src>` and of `bolt_query(<sql over bolt>)`.
fn types(conn: &Connection, sql: &str, src: &str, bolt: &str) -> Result<(String, String)> {
    exec(
        conn,
        &format!(
            "CREATE OR REPLACE TABLE w11a_te AS {}",
            sql.replace("{t}", src)
        ),
    )?;
    exec(
        conn,
        &format!(
            "CREATE OR REPLACE TABLE w11a_ta AS {}",
            bolt_query(&sql.replace("{t}", bolt))
        ),
    )?;
    let of = |t: &str| {
        query_string(
            conn,
            &format!(
                "SELECT CAST(list(data_type ORDER BY ordinal_position) AS VARCHAR) \
                 FROM information_schema.columns WHERE table_name = '{t}'"
            ),
        )
    };
    Ok((of("w11a_te")?, of("w11a_ta")?))
}

fn assert_types_and_rows(conn: &Connection, sql: &str, src: &str, bolt: &str) -> Result<()> {
    let (expected, actual) = types(conn, sql, src, bolt)?;
    assert_eq!(actual, expected, "{sql}: column types");
    assert_same(conn, sql, src, bolt)
}

#[test]
fn bolt_query_binds_duckdb_result_types() -> Result<()> {
    let _guard = lock();
    let (_env, conn) = setup_connection()?;
    build(
        &conn,
        "w11a_q",
        "w11a_qb",
        "SELECT i::INTEGER AS id, (i % 3)::INTEGER AS z, i::BIGINT AS b, \
         (i * 0.5)::FLOAT AS f, i::DOUBLE AS d, (i % 100)::TINYINT AS t8, \
         (i % 200)::UTINYINT AS u8, (i * 1000)::SMALLINT AS s16, (i * 7)::USMALLINT AS u16, \
         (i * 100000)::UINTEGER AS u32, (i * 10)::UBIGINT AS u64, \
         (i::HUGEINT * 1000000000000) AS h, ((i - 5) / 4)::DECIMAL(9, 2) AS dec \
         FROM range(0, 12) r(i)",
    )?;
    for sql in [
        "SELECT 7 / 2 AS q, id / 2 AS h FROM {t}",
        "SELECT id, b / z AS x FROM {t} WHERE z = 0 AND id > 0",
        "SELECT sum(b) AS s, sum(f) AS sf, avg(id) AS a, count(*) AS n FROM {t}",
        "SELECT t8, u8, s16, u16, u32, u64, h, dec FROM {t}",
        "SELECT id, t8 + 1 AS t, u32 * 2 AS u FROM {t}",
        "SELECT z, sum(t8) AS s, max(u64) AS m, min(h) AS mh FROM {t} GROUP BY z",
        "SELECT id, id * 0.5 AS x, d * 2 AS y FROM {t}",
        "SELECT id, CAST(id AS VARCHAR) AS s FROM {t}",
    ] {
        assert_types_and_rows(&conn, sql, "w11a_q", "w11a_qb")?;
    }
    // `0 / 0` is NaN, `x / 0` inf, as in DuckDB, without a cast error.
    let nan = query_i64(
        &conn,
        &format!(
            "SELECT count(*) FROM ({}) r(id, x) WHERE isnan(x) OR isinf(x)",
            bolt_query("SELECT id, b / z FROM w11a_qb WHERE z = 0")
        ),
    )?;
    assert_eq!(nan, 4);
    // A BIGINT sum past the BIGINT range is a HUGEINT, as in DuckDB.
    build(
        &conn,
        "w11a_big",
        "w11a_bigb",
        "SELECT 6000000000000000000::BIGINT AS v FROM range(0, 2) r(i)",
    )?;
    assert_types_and_rows(
        &conn,
        "SELECT sum(v) AS s FROM {t}",
        "w11a_big",
        "w11a_bigb",
    )?;
    if gpu() {
        // Plain projections and grouped counts still run on the GPU.
        assert!(!routed_to_host("SELECT t8, u64, h FROM w11a_qb"));
        assert!(!routed_to_host(
            "SELECT z, count(*) FROM w11a_qb GROUP BY z"
        ));
    }
    exec(&conn, "PRAGMA bolt_clear_tables")?;
    Ok(())
}

#[test]
fn interval_results_are_returned() -> Result<()> {
    let _guard = lock();
    let (_env, conn) = setup_connection()?;
    build(
        &conn,
        "w11a_ts",
        "w11a_tsb",
        "SELECT TIMESTAMP '2026-01-01 00:00:00' + to_hours(i) AS t0, \
         TIMESTAMP '2026-01-02 00:30:00' + to_hours(2 * i) AS t1 FROM range(0, 3) r(i)",
    )?;
    for sql in [
        "SELECT t1 - t0 AS d FROM {t}",
        "SELECT t0, t1 - t0 AS d FROM {t} WHERE t1 - t0 > INTERVAL '1 day'",
    ] {
        assert_types_and_rows(&conn, sql, "w11a_ts", "w11a_tsb")?;
    }
    exec(
        &conn,
        "INSERT INTO w11a_ts VALUES (NULL, TIMESTAMP '2026-01-05 00:00:00')",
    )?;
    exec(
        &conn,
        "COPY w11a_ts TO 'w11a_tsb2' (FORMAT bolt, USE_TMP_FILE FALSE)",
    )?;
    assert_types_and_rows(
        &conn,
        "SELECT t1 - t0 AS d FROM {t}",
        "w11a_ts",
        "w11a_tsb2",
    )?;
    exec(&conn, "PRAGMA bolt_clear_tables")?;
    Ok(())
}

#[test]
fn ungrouped_count_star_next_to_count_of_a_nullable_column() -> Result<()> {
    // W11-A-001 (W4-B fuzzer, seed 0x77aa0010, table 0xde0497cf1269ef19): the engine returned
    // `count(c) = count(*)` for a column with NULLs when both were in one ungrouped SELECT.
    let _guard = lock();
    let (_env, conn) = setup_connection()?;
    build(
        &conn,
        "w11a_c",
        "w11a_cb",
        "SELECT i::INTEGER AS id, CASE WHEN i % 20 = 0 THEN NULL ELSE i * 0.5 END::DOUBLE AS c0, \
         CASE WHEN i % 20 = 0 THEN NULL ELSE i END::INTEGER AS c1, i % 3 AS grp, \
         NULL::HUGEINT AS cn FROM range(0, 2049) r(i)",
    )?;
    for sql in [
        "SELECT count(*), count(c0) FROM {t}",
        "SELECT count(*), count(c1), count(id) FROM {t}",
        "SELECT count(1), count(c0) FROM {t}",
        "SELECT count(c0), count(c1) FROM {t}",
        "SELECT count(*), sum(c0), max(c1), avg(c1) FROM {t}",
        "SELECT grp, count(*), count(c0) FROM {t} GROUP BY grp",
        // Table seed 0x36fbefaadfa085f2 (seed 0xb011a011): `min` / `max` / `sum` of an all-NULL
        // column next to `count(*)` returned the kernel identity (`i64::MAX`).
        "SELECT count(*), max(id), min(cn) FROM {t}",
        "SELECT count(*), min(cn), max(cn), sum(cn) FROM {t}",
        "SELECT min(cn), max(id) FROM {t}",
    ] {
        assert_types_and_rows(&conn, sql, "w11a_c", "w11a_cb")?;
    }
    if gpu() {
        assert!(routed_to_host("SELECT count(*), count(c0) FROM w11a_cb"));
        assert!(routed_to_host("SELECT count(*), min(cn) FROM w11a_cb"));
        // Counts of NULL-free columns, a count of a nullable column alone and grouped counts
        // stay on the GPU.
        assert!(!routed_to_host("SELECT count(*), count(id) FROM w11a_cb"));
        assert!(!routed_to_host("SELECT count(c0) FROM w11a_cb"));
        assert!(!routed_to_host(
            "SELECT grp, count(*), count(c0) FROM w11a_cb GROUP BY grp"
        ));
    }
    exec(&conn, "PRAGMA bolt_clear_tables")?;
    Ok(())
}

#[test]
fn hugeint_division_over_a_null_column() -> Result<()> {
    // The W4-B fuzzer's second find on seed 0x77aa0010 (table 0x7cb9db5dc8735a2b): `c2 / 4` of
    // an all-NULL HUGEINT column (bound as BIGINT, and the quotient cast back to it, before
    // W11-A).
    let _guard = lock();
    let (_env, conn) = setup_connection()?;
    build(
        &conn,
        "w11a_h",
        "w11a_hb",
        "SELECT i::INTEGER AS id, CASE WHEN i % 7 = 0 THEN 'a' ELSE 'z' END AS c0, \
         NULL::HUGEINT AS c2, (i * 1000)::HUGEINT AS c3 FROM range(0, 300) r(i)",
    )?;
    for sql in [
        "SELECT c2, c2, (c2 / 4) FROM {t}",
        "SELECT c2, (c2 / 4) FROM {t} WHERE (CAST(35 AS BIGINT) <= c2 \
         AND ((id IS NOT NULL AND c0 < 'q')) IS NOT NULL)",
        "SELECT c3, c3 / 4, c3 // 4 FROM {t}",
    ] {
        assert_types_and_rows(&conn, sql, "w11a_h", "w11a_hb")?;
    }
    exec(&conn, "PRAGMA bolt_clear_tables")?;
    Ok(())
}

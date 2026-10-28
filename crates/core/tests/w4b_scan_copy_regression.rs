// SPDX-License-Identifier: Apache-2.0

//! Regression tests for the wave-4 `bolt_scan` / COPY work (W4-B):
//!
//! - W1-B-008: `bolt_scan` resolves table references strictly (`a/x` never reads `x`); COPY
//!   keeps path cleaning, and `COPY ... TO 'tmp_y'` registers `tmp_y`.
//! - W3-D-004: table functions and COPY binds reconcile the `bolt_*` settings.
//! - W3-B-003: restored narrow columns are cached per stored column and never served stale.
//! - W3-B-002: the host-fallback mirror exposes storage types (the TIMESTAMPTZ pushdown relies
//!   on it).
//! - W4-D-005: a prepared `bolt_query` never reuses a bind-time result after replaces.

use std::sync::{Mutex, Once};

use duckdb_bolt_core::runtime::BoltRuntime;
use duckdb_bolt_core::table_functions::restored_column_cache_stats_for_testing;
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

/// The error of `sql` (drained, so execution-time errors count).
fn error_of(conn: &Connection, sql: &str) -> String {
    let outcome = conn.query(sql, Parameters::None).and_then(|mut r| {
        while r.next_chunk()?.is_some() {}
        Ok(())
    });
    match outcome {
        Ok(()) => panic!("expected `{sql}` to fail"),
        Err(e) => e.message,
    }
}

fn table_exists(name: &str) -> bool {
    BoltRuntime::global()
        .list_tables()
        .unwrap()
        .iter()
        .any(|t| t.name == name)
}

#[test]
fn w4b_bolt_scan_resolves_references_strictly() -> Result<()> {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let (_env, conn) = setup_connection()?;
    exec(
        &conn,
        "COPY (SELECT range AS a FROM range(5)) TO 'w4b_x' (FORMAT bolt, USE_TMP_FILE FALSE)",
    )?;
    assert_eq!(
        query_i64(&conn, "SELECT count(*) FROM bolt_scan('w4b_x')")?,
        5
    );
    assert_eq!(
        query_i64(&conn, "SELECT count(*) FROM bolt_scan('bolt.w4b_x')")?,
        5
    );
    // A path-like reference is not cleaned to `w4b_x` (W1-B-008).
    for reference in ["dir/w4b_x", "w4b_x.bolt", "tmp_w4b_x"] {
        let err = error_of(
            &conn,
            &format!("SELECT count(*) FROM bolt_scan('{reference}')"),
        );
        assert!(err.contains("is not registered"), "{reference}: {err}");
    }
    // COPY FROM keeps the path cleaning.
    exec(&conn, "CREATE TABLE w4b_back (a BIGINT)")?;
    exec(&conn, "COPY w4b_back FROM 'dir/w4b_x.bolt' (FORMAT bolt)")?;
    assert_eq!(query_i64(&conn, "SELECT count(*) FROM w4b_back")?, 5);
    exec(&conn, "PRAGMA bolt_clear_tables")?;
    Ok(())
}

#[test]
fn w4b_copy_to_keeps_tmp_prefix_and_strips_paths() -> Result<()> {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let (_env, conn) = setup_connection()?;
    exec(
        &conn,
        "COPY (SELECT 1 AS a) TO 'tmp_w4b_y' (FORMAT bolt, USE_TMP_FILE FALSE)",
    )?;
    assert!(table_exists("tmp_w4b_y"));
    assert!(!table_exists("w4b_y"));
    exec(
        &conn,
        "COPY (SELECT 1 AS a) TO 'some/dir/w4b_z.bolt' (FORMAT bolt, USE_TMP_FILE FALSE)",
    )?;
    assert!(table_exists("w4b_z"));
    exec(
        &conn,
        "COPY (SELECT 1 AS a) TO 'bolt.w4b_q' (FORMAT bolt, USE_TMP_FILE FALSE)",
    )?;
    assert!(table_exists("w4b_q"));
    // The tmp_ table reads back under its own name, and COPY FROM finds it.
    assert_eq!(
        query_i64(&conn, "SELECT count(*) FROM bolt_scan('tmp_w4b_y')")?,
        1
    );
    exec(&conn, "CREATE TABLE w4b_back2 (a INTEGER)")?;
    exec(&conn, "COPY w4b_back2 FROM 'tmp_w4b_y' (FORMAT bolt)")?;
    assert_eq!(query_i64(&conn, "SELECT count(*) FROM w4b_back2")?, 1);
    exec(&conn, "PRAGMA bolt_clear_tables")?;
    Ok(())
}

/// Every table function and COPY bind refreshes a setting another database made stale
/// (W3-D-004; the scalar-function version is in `w3d_options_ingest_sandbox.rs`).
#[test]
fn w4b_table_functions_and_copy_sync_settings() -> Result<()> {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let (_env_a, a) = setup_connection()?;
    let (_env_b, b) = setup_connection()?;
    exec(
        &b,
        "COPY (SELECT 1 AS a) TO 'w4b_sync' (FORMAT bolt, USE_TMP_FILE FALSE)",
    )?;
    exec(&a, "CREATE TABLE w4b_sync_back (a INTEGER)")?;
    let calls: [(&str, bool); 6] = [
        ("SELECT * FROM bolt_query('SELECT 1')", false),
        ("SELECT * FROM bolt_pool_stats()", false),
        ("SELECT * FROM bolt_tables()", false),
        ("SELECT * FROM bolt_scan('w4b_sync')", false),
        (
            "COPY (SELECT 2 AS a) TO 'w4b_sync2' (FORMAT bolt, USE_TMP_FILE FALSE)",
            true,
        ),
        ("COPY w4b_sync_back FROM 'w4b_sync' (FORMAT bolt)", true),
    ];
    for (round, (call, is_statement)) in calls.into_iter().enumerate() {
        let budget = 4_194_304 + round as i64 * 1_048_576;
        exec(&a, "SET bolt_memory_budget = 0")?;
        exec(&b, &format!("SET bolt_memory_budget = {budget}"))?;
        assert_eq!(
            query_i64(&a, "SELECT current_setting('bolt_memory_budget')::BIGINT")?,
            0,
            "{call}: precondition"
        );
        if is_statement {
            exec(&a, call)?;
        } else {
            let mut r = a.query(call, Parameters::None)?;
            while r.next_chunk()?.is_some() {}
        }
        assert_eq!(
            query_i64(&a, "SELECT current_setting('bolt_memory_budget')::BIGINT")?,
            budget,
            "{call} did not refresh the stale setting"
        );
    }
    exec(&b, "RESET bolt_memory_budget")?;
    exec(&b, "PRAGMA bolt_clear_tables")?;
    Ok(())
}

/// Restored SMALLINT / UINTEGER / HUGEINT columns are converted once per stored column and
/// shared by later scans; replace and append are never served stale data (W3-B-003).
#[test]
fn w4b_restored_columns_are_cached_per_table_version() -> Result<()> {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let (_env, conn) = setup_connection()?;
    exec(
        &conn,
        "CREATE TABLE w4b_narrow AS SELECT (i % 30000)::SMALLINT AS s, (i * 3)::UINTEGER AS u,
             i::HUGEINT AS h FROM range(0, 5000) t(i)",
    )?;
    exec(
        &conn,
        "COPY w4b_narrow TO 'w4b_n' (FORMAT bolt, USE_TMP_FILE FALSE)",
    )?;
    let sums = "SELECT sum(s)::BIGINT + sum(u)::BIGINT + sum(h)::BIGINT FROM";
    let expected = query_i64(&conn, &format!("{sums} w4b_narrow"))?;
    let (_, m0, _) = restored_column_cache_stats_for_testing();
    assert_eq!(
        query_i64(&conn, &format!("{sums} bolt_scan('w4b_n')"))?,
        expected
    );
    let (h1, m1, _) = restored_column_cache_stats_for_testing();
    assert!(m1 > m0, "the first scan converts ({m0} -> {m1})");
    assert_eq!(
        query_i64(&conn, &format!("{sums} bolt_scan('w4b_n')"))?,
        expected
    );
    let (h2, m2, _) = restored_column_cache_stats_for_testing();
    assert_eq!(m2, m1, "the second scan converts nothing");
    assert!(h2 > h1, "the second scan is served from the cache");
    assert_eq!(
        query_string(&conn, "SELECT typeof(s) FROM bolt_scan('w4b_n') LIMIT 1")?,
        "SMALLINT"
    );

    // Replace with different data: new stored columns, so new conversions and new values.
    exec(
        &conn,
        "UPDATE w4b_narrow SET s = s + 1, u = u + 1, h = h + 1",
    )?;
    exec(
        &conn,
        "COPY w4b_narrow TO 'w4b_n' (FORMAT bolt, USE_TMP_FILE FALSE, MODE 'replace')",
    )?;
    let replaced = query_i64(&conn, &format!("{sums} w4b_narrow"))?;
    assert_ne!(replaced, expected);
    assert_eq!(
        query_i64(&conn, &format!("{sums} bolt_scan('w4b_n')"))?,
        replaced
    );
    // Append: the old batch stays cached, the new one is converted.
    exec(
        &conn,
        "COPY (SELECT 7::SMALLINT AS s, 7::UINTEGER AS u, 7::HUGEINT AS h) TO 'w4b_n'
             (FORMAT bolt, USE_TMP_FILE FALSE, MODE 'append')",
    )?;
    assert_eq!(
        query_i64(&conn, &format!("{sums} bolt_scan('w4b_n')"))?,
        replaced + 21
    );
    // Eviction of entries whose stored column is gone is covered by the unit test
    // `table_functions::tests::test_restored_column_cache_hits_and_evicts`.
    exec(&conn, "PRAGMA bolt_clear_tables")?;
    Ok(())
}

/// The host-fallback mirror declares storage types (a TIMESTAMPTZ column is the naive UTC
/// TIMESTAMP it is stored as), which `tz_constants_safe` relies on (W3-B-002).
#[test]
fn w4b_host_fallback_mirror_uses_storage_types() -> Result<()> {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let (_env, conn) = setup_connection()?;
    exec(
        &conn,
        "COPY (SELECT TIMESTAMPTZ '2024-01-01 00:00:00+00' AS tstz, 1::SMALLINT AS s
               UNION ALL SELECT NULL, NULL)
         TO 'w4b_mirror' (FORMAT bolt, USE_TMP_FILE FALSE)",
    )?;
    let out = BoltRuntime::global()
        .execute_sql_host_fallback(
            "SELECT typeof(tstz) AS t, typeof(s) AS s FROM w4b_mirror LIMIT 1",
        )?
        .batch;
    let col = |i: usize| {
        let a = out.column(i);
        arrow_cast::display::array_value_to_string(a, 0).unwrap()
    };
    assert_eq!(col(0), "TIMESTAMP");
    assert_eq!(col(1), "INTEGER");
    exec(&conn, "PRAGMA bolt_clear_tables")?;
    Ok(())
}

/// A prepared `bolt_query` whose result was computed at bind time is not reused after the
/// table was replaced (twice, with the same shape) before the first EXECUTE (W4-D-005).
#[test]
fn w4b_prepared_bolt_query_never_reuses_stale_bind_result() -> Result<()> {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let (_env, conn) = setup_connection()?;
    let load = |offset: i64| {
        exec(
            &conn,
            &format!(
                "COPY (SELECT i + {offset} AS i, i AS v FROM range(100) r(i)) TO 'w4b_fp'
                 (FORMAT bolt, USE_TMP_FILE FALSE, MODE 'replace')"
            ),
        )
    };
    load(0)?;
    exec(
        &conn,
        "PREPARE w4b_p AS SELECT * FROM bolt_query('SELECT a.i FROM w4b_fp a, LATERAL (SELECT 1) s ORDER BY a.i LIMIT 1')",
    )?;
    load(1000)?;
    load(2000)?;
    assert_eq!(query_i64(&conn, "EXECUTE w4b_p")?, 2000);
    exec(&conn, "DEALLOCATE w4b_p")?;
    exec(&conn, "PRAGMA bolt_clear_tables")?;
    Ok(())
}

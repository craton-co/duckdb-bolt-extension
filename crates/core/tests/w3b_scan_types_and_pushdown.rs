// SPDX-License-Identifier: Apache-2.0

//! Regression tests for the wave-3 `bolt_scan` work (W3-B):
//!
//! - RFC-008 read side (docs/issues/W2-C-001): `bolt_scan` declares and returns the original
//!   DuckDB types recorded by `COPY ... TO (FORMAT bolt)` on every scan path.
//! - Pushed predicates mean the same on the GPU engine and on the host-fallback DuckDB
//!   (docs/issues/W2-O-002), checked on a table with NULLs (host fallback) and without
//!   (GPU engine).
//! - `(pred) IS [NOT] NULL` (W2-E-002), DECIMAL constants against integer columns (W2-E-003),
//!   DOUBLE overflow (W2-E-013) and `count(*)` over a VARCHAR-first table (W2-E-018).
//!
//! On a CUDA host the predicates are pushed to the engine; with `cuda-stub` DuckDB evaluates
//! them and the result assertions still hold.

use std::sync::{Mutex, Once};

use duckdb_bolt_core::runtime::BoltRuntime;
use duckdb_bolt_core::table_functions::last_bolt_scan_trace_for_testing;
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
// The Bolt catalog and the scan trace are process-wide.
static TEST_MUTEX: Mutex<()> = Mutex::new(());

fn setup_connection() -> Result<(Environment, Connection)> {
    REGISTER_ONCE.call_once(|| {
        // SAFETY: `duckdb_extension_bolt_describe` is emitted by `#[duckdb_entrypoint_c_api_v2]`, which
        // fills the descriptor as `duckdb_static_extension.h` specifies (the contract of
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

fn exec(conn: &Connection, sql: &str) -> Result<()> {
    conn.execute(sql, Parameters::None)?;
    Ok(())
}

/// `src` uploaded as `bolt` (with its NULLs: pushed SQL runs on the host fallback) and,
/// when `src` has NULLs, `src_nn` (its NULL-free rows) uploaded as `bolt_nn` (pushed SQL runs
/// on the GPU engine). Returns the `(reference table, bolt table)` pairs to check.
fn upload_both(conn: &Connection, src: &str, bolt: &str) -> Result<Vec<(String, String)>> {
    exec(
        conn,
        &format!("COPY {src} TO '{bolt}' (FORMAT bolt, USE_TMP_FILE FALSE, MODE 'replace')"),
    )?;
    let mut pairs = vec![(src.to_string(), bolt.to_string())];
    let rows = query_i64(conn, &format!("SELECT count(*) FROM {src}"))?;
    let complete = query_i64(
        conn,
        &format!("SELECT count(*) FROM {src} WHERE COLUMNS(*) IS NOT NULL"),
    )?;
    if complete < rows {
        exec(
            conn,
            &format!(
                "CREATE OR REPLACE TABLE {src}_nn AS SELECT * FROM {src} WHERE COLUMNS(*) IS NOT NULL"
            ),
        )?;
        exec(
            conn,
            &format!(
                "COPY {src}_nn TO '{bolt}_nn' (FORMAT bolt, USE_TMP_FILE FALSE, MODE 'replace')"
            ),
        )?;
        pairs.push((format!("{src}_nn"), format!("{bolt}_nn")));
    }
    Ok(pairs)
}

/// Rows in the symmetric multiset difference of `SELECT * FROM a WHERE pred` and the same
/// over `b` (every column, NULLs included).
fn row_diff(conn: &Connection, a: &str, b: &str, pred: &str) -> Result<i64> {
    query_i64(
        conn,
        &format!(
            "SELECT count(*) FROM (
                (SELECT * FROM {a} WHERE {pred} EXCEPT ALL SELECT * FROM {b} WHERE {pred})
                UNION ALL
                (SELECT * FROM {b} WHERE {pred} EXCEPT ALL SELECT * FROM {a} WHERE {pred}))"
        ),
    )
}

/// `bolt_scan` must return exactly the rows (and values) DuckDB returns for every predicate,
/// on both the NULL table and the NULL-free table.
fn assert_predicates(conn: &Connection, src: &str, bolt: &str, predicates: &[&str]) -> Result<()> {
    let mut failures = Vec::new();
    for (reference, bolt_name) in upload_both(conn, src, bolt)? {
        let scan = format!("bolt_scan('{bolt_name}')");
        for pred in predicates {
            let count_sql = |from: &str| format!("SELECT count(*) FROM {from} WHERE {pred}");
            let expected = query_i64(conn, &count_sql(&reference));
            let actual = query_i64(conn, &count_sql(&scan));
            match (&expected, &actual) {
                (Ok(e), Ok(a)) if e == a => {}
                // Both raise an error (e.g. an overflow): same behavior.
                (Err(_), Err(_)) => continue,
                _ => {
                    failures.push(format!(
                        "{bolt_name}: {pred}: bolt_scan {actual:?} != DuckDB {expected:?}"
                    ));
                    continue;
                }
            }
            match row_diff(conn, &scan, &reference, pred) {
                Ok(0) => {}
                Ok(n) => failures.push(format!("{bolt_name}: {pred}: {n} row(s) differ")),
                Err(e) => failures.push(format!("{bolt_name}: {pred}: row comparison error: {e}")),
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} predicate mismatch(es):\n{}",
        failures.len(),
        failures.join("\n")
    );
    Ok(())
}

/// The pushed predicates of `SELECT count(*) FROM bolt_scan(bolt) WHERE pred`.
fn pushed(conn: &Connection, bolt: &str, pred: &str) -> Result<Vec<(String, bool)>> {
    query_i64(
        conn,
        &format!("SELECT count(*) FROM bolt_scan('{bolt}') WHERE {pred}"),
    )?;
    Ok(last_bolt_scan_trace_for_testing()
        .expect("bolt_scan trace")
        .filters)
}

const TYPES_DDL: &str = "(
    id INTEGER,
    c_i8 TINYINT,
    c_i16 SMALLINT,
    c_u8 UTINYINT,
    c_u16 USMALLINT,
    c_u32 UINTEGER,
    c_u64 UBIGINT,
    c_i128 HUGEINT,
    c_f32 FLOAT,
    c_dec DECIMAL(9, 2),
    c_date DATE,
    c_ts_s TIMESTAMP_S,
    c_ts_ms TIMESTAMP_MS,
    c_ts TIMESTAMP,
    c_ts_ns TIMESTAMP_NS,
    c_tstz TIMESTAMPTZ,
    c_str VARCHAR,
    c_bool BOOLEAN
)";

const TYPED_COLUMNS: [&str; 17] = [
    "c_i8", "c_i16", "c_u8", "c_u16", "c_u32", "c_u64", "c_i128", "c_f32", "c_dec", "c_date",
    "c_ts_s", "c_ts_ms", "c_ts", "c_ts_ns", "c_tstz", "c_str", "c_bool",
];

#[test]
fn w3b_bolt_scan_returns_original_types_on_every_path() -> Result<()> {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let (_env, conn) = setup_connection()?;
    exec(&conn, &format!("CREATE TABLE ty_src {TYPES_DDL}"))?;
    exec(
        &conn,
        "INSERT INTO ty_src SELECT
            i, (i % 100 - 50)::TINYINT, (i * 7 - 3000)::SMALLINT, (i % 250)::UTINYINT,
            (i * 13)::USMALLINT, (4294967295 - i)::UINTEGER, (9223372036854775807 - i)::UBIGINT,
            (i * 1000000007)::HUGEINT, (i / 4.0)::FLOAT, (i * 1.25)::DECIMAL(9, 2),
            DATE '2024-01-01' + i::INTEGER, TIMESTAMP_S '2024-01-01 00:00:00' + INTERVAL (i) SECOND,
            TIMESTAMP_MS '2024-01-01 00:00:00' + INTERVAL (i) MILLISECOND,
            TIMESTAMP '2024-01-01 00:00:00' + INTERVAL (i) MINUTE,
            TIMESTAMP_NS '2024-01-01 00:00:00' + INTERVAL (i) MICROSECOND,
            TIMESTAMPTZ '2024-01-01 00:00:00+00' + INTERVAL (i) HOUR,
            'v' || (i % 9), i % 2 = 0
         FROM range(0, 600) t(i)",
    )?;
    exec(
        &conn,
        "INSERT INTO ty_src (id) VALUES (1000), (1001)", // rows of NULLs
    )?;
    // Fixed session time zone that differs from UTC: TIMESTAMPTZ must stay an instant.
    exec(&conn, "SET TimeZone = 'America/Argentina/Buenos_Aires'")?;
    for (reference, bolt) in upload_both(&conn, "ty_src", "w3b_ty")? {
        for col in TYPED_COLUMNS {
            let expected = query_string(
                &conn,
                &format!("SELECT typeof({col}) FROM {reference} LIMIT 1"),
            )?;
            for from in [
                format!("bolt_scan('{bolt}')"),
                format!("bolt.{bolt}"),
                format!("(SELECT * FROM bolt_scan('{bolt}') WHERE id > 3)"),
                format!("(SELECT * FROM bolt_scan('{bolt}') WHERE id % 5 = 1)"),
            ] {
                let actual =
                    query_string(&conn, &format!("SELECT typeof({col}) FROM {from} LIMIT 1"))?;
                assert_eq!(actual, expected, "{col} via {from}");
            }
        }
        // Every value round-trips without casts, unfiltered and filtered (GPU / filter-take /
        // host-fallback paths).
        for pred in [
            "true",
            "id > 3",
            "id % 5 = 1",
            "c_i16 > 0",
            "c_u32 > 4294967000",
            "c_tstz >= TIMESTAMPTZ '2024-01-05 00:00:00+00'",
            "c_str = 'v3'",
        ] {
            assert_eq!(
                row_diff(&conn, &format!("bolt_scan('{bolt}')"), &reference, pred)?,
                0,
                "{bolt}: {pred}"
            );
        }
        let projected = format!(
            "SELECT count(*) FROM (
                (SELECT c_u16, c_i16, id FROM {reference} WHERE id < 50
                 EXCEPT ALL SELECT c_u16, c_i16, id FROM bolt_scan('{bolt}') WHERE id < 50)
                UNION ALL
                (SELECT c_u16, c_i16, id FROM bolt_scan('{bolt}') WHERE id < 50
                 EXCEPT ALL SELECT c_u16, c_i16, id FROM {reference} WHERE id < 50))"
        );
        assert_eq!(
            query_i64(&conn, &projected)?,
            0,
            "{bolt}: duplicated projection"
        );
    }
    exec(&conn, "RESET TimeZone")?;
    Ok(())
}

#[test]
fn w3b_unsigned_arithmetic_overflows_like_duckdb() -> Result<()> {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let (_env, conn) = setup_connection()?;
    exec(
        &conn,
        "CREATE TABLE uo_src AS SELECT i::BIGINT AS id, (i * 1000)::UINTEGER AS u,
                (i % 120)::SMALLINT AS s, CASE WHEN i % 9 = 0 THEN NULL ELSE i END::INTEGER AS n
         FROM range(0, 100) t(i)",
    )?;
    for (_, bolt) in upload_both(&conn, "uo_src", "w3b_uo")? {
        // UINTEGER * UINTEGER overflows from u = 66000 on, in DuckDB and through bolt_scan.
        for pred in ["u * u > 0", "s * s * s * s > 0"] {
            let native = query_i64(&conn, &format!("SELECT count(*) FROM uo_src WHERE {pred}"));
            let scanned = query_i64(
                &conn,
                &format!("SELECT count(*) FROM bolt_scan('{bolt}') WHERE {pred}"),
            );
            assert!(native.is_err(), "{pred}: DuckDB {native:?}");
            let err = scanned.expect_err(pred).to_string();
            assert!(err.contains("Overflow"), "{bolt}: {pred}: {err}");
        }
    }
    assert_predicates(
        &conn,
        "uo_src",
        "w3b_uo",
        &[
            "u > 50000",
            "u = 3000",
            "u * 2 > 100000",
            "s < 10",
            "CAST(s AS TINYINT) = 7",
            "u::BIGINT > 70000",
        ],
    )
}

#[test]
fn w3b_timestamptz_predicates_follow_duckdb_in_any_time_zone() -> Result<()> {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let (_env, conn) = setup_connection()?;
    exec(
        &conn,
        "CREATE TABLE tz_src AS
         SELECT i::BIGINT AS id,
                CASE WHEN i % 10 = 9 THEN NULL
                     ELSE TIMESTAMPTZ '2024-01-01 00:00:00+00' + INTERVAL (i) HOUR END AS tstz
         FROM range(0, 96) t(i)",
    )?;
    let predicates = [
        "tstz >= TIMESTAMPTZ '2024-01-01 05:00:00+00'",
        "tstz < TIMESTAMPTZ '2024-01-01 05:00:00-03'",
        "tstz = TIMESTAMPTZ '2024-01-02 10:00:00+00'",
        "tstz != TIMESTAMPTZ '2024-01-02 10:00:00+00'",
        "tstz BETWEEN TIMESTAMPTZ '2024-01-01 03:00:00+00' AND TIMESTAMPTZ '2024-01-01 09:00:00+05'",
        "tstz IN (TIMESTAMPTZ '2024-01-01 07:00:00+00', TIMESTAMPTZ '2024-01-02 07:00:00+00')",
        "tstz IS NULL",
        "tstz::DATE = DATE '2024-01-02'",
    ];
    for zone in ["UTC", "America/Argentina/Buenos_Aires", "Asia/Kolkata"] {
        exec(&conn, &format!("SET TimeZone = '{zone}'"))?;
        assert_predicates(&conn, "tz_src", "w3b_tz", &predicates)?;
    }
    exec(&conn, "SET TimeZone = 'America/Argentina/Buenos_Aires'")?;
    assert_eq!(
        query_i64(
            &conn,
            "SELECT count(*) FROM bolt_scan('w3b_tz') WHERE tstz >= TIMESTAMPTZ '2024-01-01 05:00:00+00'"
        )?,
        91 - 9
    );
    if gpu() {
        // The predicate is translated to a UTC `TIMESTAMP` literal against the naive UTC
        // storage column, but craton-bolt 0.9 cannot compare its nanosecond `TIMESTAMP`
        // literal with a microsecond column, so the plan check keeps it with DuckDB
        // (docs/issues/W1-A-006). Either way the results above match DuckDB.
        let filters = pushed(
            &conn,
            "w3b_tz_nn",
            "tstz = TIMESTAMPTZ '2024-01-02 10:00:00+00'",
        )?;
        assert!(
            filters.iter().all(|(sql, _)| !sql.contains("+00")),
            "{filters:?}"
        );
    }
    exec(&conn, "RESET TimeZone")?;
    Ok(())
}

#[test]
fn w3b_is_null_over_predicates_is_not_guarded() -> Result<()> {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let (_env, conn) = setup_connection()?;
    exec(
        &conn,
        "CREATE TABLE isn_src AS SELECT i::BIGINT AS id,
                CASE WHEN i % 2 = 0 THEN NULL ELSE i % 7 END::INTEGER AS a,
                CASE WHEN i % 3 = 0 THEN NULL ELSE i / 4.0 END::DOUBLE AS d,
                CASE WHEN i % 5 = 0 THEN NULL ELSE 'e' || (i % 4) END AS name
         FROM range(0, 400) t(i)",
    )?;
    assert_predicates(
        &conn,
        "isn_src",
        "w3b_isn",
        &[
            "(a = 1) IS NULL",
            "(a = 1) IS NOT NULL",
            "(a IN (1, 3)) IS NOT NULL",
            "(a IN (1, 3)) IS NULL",
            "(name LIKE 'e%') IS NULL",
            "(name = 'e1') IS NOT NULL",
            "NOT ((a = 1) IS NULL)",
            "((a = 1) OR (d > 3)) IS NULL",
            "(NOT (a > 2)) IS NULL",
            "a IS NULL",
            "(d * 2) IS NULL",
            "(d + a) IS NOT NULL",
            "NOT (a IS NULL) AND d > 10",
        ],
    )?;
    if gpu() {
        // A plain column and pushed DOUBLE arithmetic are still pushed.
        let filters = pushed(&conn, "w3b_isn", "(d * 2) IS NULL")?;
        assert!(filters.iter().any(|(_, exact)| *exact), "{filters:?}");
        assert!(pushed(&conn, "w3b_isn", "(a = 1) IS NULL")?.is_empty());
    }
    Ok(())
}

#[test]
fn w3b_decimal_constants_against_integer_columns_are_exact() -> Result<()> {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let (_env, conn) = setup_connection()?;
    exec(
        &conn,
        "CREATE TABLE dl_src AS
         SELECT * FROM (VALUES (1, 2, 9007199254740993::BIGINT), (2, 3, 1::BIGINT),
                               (3, -3, -9007199254740993::BIGINT), (4, NULL, NULL),
                               (5, 0, 9007199254740992::BIGINT)) v(id, a, b)",
    )?;
    let predicates = [
        "a <= 2.99999999999999999",
        "a < 2.99999999999999999",
        "a > 2.00000000000000001",
        "a >= 2.00000000000000001",
        "a = 2.5",
        "a != 2.5",
        "a = 3.000",
        "a != 3.0",
        "a > -2.5",
        "a < -2.5",
        "a >= -3.0",
        "a BETWEEN 1.5 AND 3.5",
        "a IN (2.0, 3.5)",
        "b > 9007199254740992.5",
        "b < -9007199254740992.5",
        "b >= 9007199254740992.5",
        "2.5 < a",
    ];
    assert_predicates(&conn, "dl_src", "w3b_dl", &predicates)?;
    assert_eq!(
        query_i64(
            &conn,
            "SELECT count(*) FROM bolt_scan('w3b_dl') WHERE a <= 2.99999999999999999"
        )?,
        3
    );
    if gpu() {
        // Pushed as integer bounds, never as a (DOUBLE / DECIMAL) fractional literal.
        let filters = pushed(&conn, "w3b_dl", "a <= 2.99999999999999999")?;
        assert_eq!(filters.len(), 1, "{filters:?}");
        assert!(filters[0].0.contains("<= 2)"), "{filters:?}");
        let filters = pushed(&conn, "w3b_dl", "b > 9007199254740992.5")?;
        assert!(filters[0].0.contains("> 9007199254740992)"), "{filters:?}");
    }
    Ok(())
}

#[test]
fn w3b_double_overflow_matches_duckdb() -> Result<()> {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let (_env, conn) = setup_connection()?;
    // DuckDB 2.0 returns IEEE infinity for finite DOUBLE overflow (no error), like the engine.
    assert_eq!(query_string(&conn, "SELECT (1e308 * 10)::VARCHAR")?, "inf");
    exec(
        &conn,
        "CREATE TABLE dof_src AS SELECT * FROM (VALUES
            (1, 1e308::DOUBLE), (2, 1.7976931348623157e308::DOUBLE), (3, -1e308::DOUBLE),
            (4, 1.0::DOUBLE), (5, NULL)) v(id, d)",
    )?;
    assert_predicates(
        &conn,
        "dof_src",
        "w3b_dof",
        &[
            "d * 10 > 0",
            "d + d > 1e308",
            "d * 10 < 0",
            "d - d = 0",
            "d * 10 = 'inf'::DOUBLE",
            "d * 1e-320 > 0",
        ],
    )
}

#[test]
fn w3b_count_star_over_varchar_first_table() -> Result<()> {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let (_env, conn) = setup_connection()?;
    exec(
        &conn,
        "CREATE TABLE cs_src AS SELECT 'name_' || i AS s, (i % 1000)::DECIMAL(9, 2) AS dec,
                i::BIGINT AS id, (i % 3 = 0) AS flag
         FROM range(0, 20000) t(i)",
    )?;
    exec(
        &conn,
        "CREATE TABLE so_src AS SELECT 's' || (i % 7) AS s, 't' || i AS t FROM range(0, 500) t(i)",
    )?;
    for (src, bolt, pred) in [
        ("cs_src", "w3b_cs", "id > 10"),
        ("cs_src", "w3b_cs", "id % 3 = 1 AND id < 15000"),
        ("so_src", "w3b_so", "s = 's3'"),
        ("so_src", "w3b_so", "s != 's3'"),
    ] {
        exec(
            &conn,
            &format!("COPY {src} TO '{bolt}' (FORMAT bolt, USE_TMP_FILE FALSE, MODE 'replace')"),
        )?;
        let expected = query_i64(&conn, &format!("SELECT count(*) FROM {src} WHERE {pred}"))?;
        let actual = query_i64(
            &conn,
            &format!("SELECT count(*) FROM bolt_scan('{bolt}') WHERE {pred}"),
        )?;
        assert_eq!(actual, expected, "{bolt}: {pred}");
    }
    if gpu() {
        query_i64(
            &conn,
            "SELECT count(*) FROM bolt_scan('w3b_cs') WHERE id > 10",
        )?;
        assert_eq!(last_bolt_scan_trace_for_testing().unwrap().path, "gpu");
    }
    Ok(())
}

#[test]
fn w3b_schema_drift_detects_original_type_change() -> Result<()> {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let (_env, conn) = setup_connection()?;
    exec(
        &conn,
        "COPY (SELECT 1::SMALLINT AS v) TO 'w3b_drift' (FORMAT bolt, USE_TMP_FILE FALSE)",
    )?;
    let stmt = conn.prepare("SELECT typeof(v) FROM bolt_scan('w3b_drift')", false)?;
    {
        let mut res = stmt.execute(Parameters::None)?;
        let chunk = res.next_chunk()?.expect("chunk");
        let vec = chunk.get_vector_at::<String>(0)?;
        assert_eq!(
            vec.reader()?.get(0)?.map(|s| s.to_string()).as_deref(),
            Some("SMALLINT")
        );
    }
    // Same Int32 storage type, different declared type.
    exec(
        &conn,
        "COPY (SELECT 70000::INTEGER AS v) TO 'w3b_drift' (FORMAT bolt, USE_TMP_FILE FALSE, MODE 'replace')",
    )?;
    let err = match stmt.execute(Parameters::None) {
        Err(e) => e.to_string(),
        Ok(mut res) => match res.next_chunk() {
            Err(e) => e.to_string(),
            Ok(_) => panic!("SMALLINT plan executed against an INTEGER table"),
        },
    };
    assert!(err.contains("replaced after the query was bound"), "{err}");
    assert_eq!(
        query_string(&conn, "SELECT typeof(v) FROM bolt_scan('w3b_drift')")?,
        "INTEGER"
    );
    Ok(())
}

fn execute_i64(stmt: &duckdb_neo::statement::PreparedStatement<'_>) -> Result<i64> {
    let mut res = stmt.execute(Parameters::None)?;
    let chunk = res.next_chunk()?.expect("chunk");
    let vec = chunk.get_vector_at::<i64>(0)?;
    Ok(*vec.reader()?.get(0)?.expect("non-null"))
}

#[test]
fn w3b_prepared_eager_bolt_query_sees_current_data() -> Result<()> {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let (_env, conn) = setup_connection()?;
    exec(
        &conn,
        "COPY (SELECT 1::INTEGER AS id, 1.0::DOUBLE AS v) TO 'w3b_prep' (FORMAT bolt, USE_TMP_FILE FALSE)",
    )?;
    // LATERAL is not planned statically: the query runs at bind time (eager path).
    let eager_sql = "SELECT count(*)::BIGINT FROM bolt_query(
        'SELECT r1.id FROM w3b_prep r1, LATERAL (SELECT r2.v FROM w3b_prep r2 WHERE r2.id = r1.id) l')";
    let deferred_sql = "SELECT count(*)::BIGINT FROM bolt_query('SELECT id, v FROM w3b_prep')";
    let eager = conn.prepare(eager_sql, false)?;
    let deferred = conn.prepare(deferred_sql, false)?;
    assert_eq!(execute_i64(&eager)?, 1);
    assert_eq!(execute_i64(&eager)?, 1);
    assert_eq!(execute_i64(&deferred)?, 1);

    // Replace with the same schema.
    exec(
        &conn,
        "COPY (SELECT range::INTEGER AS id, 1.0::DOUBLE AS v FROM range(5)) TO 'w3b_prep'
         (FORMAT bolt, USE_TMP_FILE FALSE, MODE 'replace')",
    )?;
    assert_eq!(execute_i64(&eager)?, 5);
    assert_eq!(execute_i64(&deferred)?, 5);

    // Append.
    exec(
        &conn,
        "COPY (SELECT 9::INTEGER AS id, 2.0::DOUBLE AS v) TO 'w3b_prep'
         (FORMAT bolt, USE_TMP_FILE FALSE, MODE 'append')",
    )?;
    assert_eq!(execute_i64(&eager)?, 6);

    // A catalog change between PREPARE and the first EXECUTE is seen too.
    let fresh = conn.prepare(eager_sql, false)?;
    exec(
        &conn,
        "COPY (SELECT range::INTEGER AS id, 1.0::DOUBLE AS v FROM range(3)) TO 'w3b_prep'
         (FORMAT bolt, USE_TMP_FILE FALSE, MODE 'replace')",
    )?;
    assert_eq!(execute_i64(&fresh)?, 3);
    assert_eq!(query_i64(&conn, eager_sql)?, 3);
    Ok(())
}

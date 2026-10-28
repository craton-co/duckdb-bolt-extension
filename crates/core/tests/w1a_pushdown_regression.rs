// SPDX-License-Identifier: Apache-2.0

//! Differential regression tests for `bolt_scan` filter pushdown: every predicate is run
//! through plain DuckDB and through `bolt_scan`, and the results must match. On a CUDA host
//! the predicates are pushed to the Craton Bolt engine; with `cuda-stub` DuckDB evaluates
//! them, so the same assertions hold on both backends.

use std::sync::{Arc, Mutex, Once};

use arrow_array::{Int32Array, RecordBatch, TimestampMicrosecondArray};
use arrow_schema::{DataType, Field, Schema, TimeUnit};
use duckdb_bolt_core::runtime::{BoltRuntime, RegisterMode};
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
// Serialize tests because Craton Bolt's GPU runtime catalog is process-wide.
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

/// Rows in the symmetric multiset difference (EXCEPT ALL both ways) of `SELECT * FROM a WHERE
/// pred` and the same over `b`: compares every column value, including NULLs.
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

/// Count and an order-independent checksum of the `id` column for rows matching `pred`.
fn fingerprint(conn: &Connection, from: &str, pred: &str) -> Result<(i64, i64)> {
    let count = query_i64(conn, &format!("SELECT count(*) FROM {from} WHERE {pred}"))?;
    let sum = query_i64(
        conn,
        &format!("SELECT coalesce(sum(id), 0)::BIGINT FROM {from} WHERE {pred}"),
    )?;
    Ok((count, sum))
}

/// Upload `local` to Bolt as `bolt_name` and materialize an unfiltered `bolt_scan` copy
/// (`<bolt_name>_m`), which carries exactly the types `bolt_scan` exposes (storage widens
/// some types) and is the reference for every pushdown comparison.
fn upload(conn: &Connection, local: &str, bolt_name: &str) -> Result<()> {
    conn.execute(
        format!("COPY {local} TO '{bolt_name}' (FORMAT bolt, USE_TMP_FILE FALSE)"),
        Parameters::None,
    )?;
    materialize(conn, bolt_name)
}

fn materialize(conn: &Connection, bolt_name: &str) -> Result<()> {
    conn.execute(
        format!("CREATE OR REPLACE TABLE {bolt_name}_m AS SELECT * FROM bolt_scan('{bolt_name}')"),
        Parameters::None,
    )?;
    Ok(())
}

/// Assert `bolt_scan(bolt_name)` agrees with the materialized copy (and, when given, the
/// original DuckDB table) for every predicate.
///
/// When the table holds NULLs, the runtime's crash guard runs the pushed SQL on the embedded
/// host DuckDB (docs/issues/W2-A-001); every predicate is then also checked on a NULL-free
/// copy (`<bolt_name>_nn`), whose pushed SQL runs on the GPU engine, so both dialects are
/// covered (docs/issues/W2-O-002).
fn assert_predicates(
    conn: &Connection,
    original: Option<&str>,
    bolt_name: &str,
    predicates: &[&str],
) -> Result<()> {
    assert_predicates_on(conn, original, bolt_name, predicates)?;
    let materialized = format!("{bolt_name}_m");
    let rows = query_i64(conn, &format!("SELECT count(*) FROM {materialized}"))?;
    let complete = query_i64(
        conn,
        &format!("SELECT count(*) FROM {materialized} WHERE COLUMNS(*) IS NOT NULL"),
    )?;
    if complete == rows {
        return Ok(());
    }
    let nn = format!("{bolt_name}_nn");
    conn.execute(
        format!(
            "CREATE OR REPLACE TABLE {nn}_src AS SELECT * FROM {materialized} WHERE COLUMNS(*) IS NOT NULL"
        ),
        Parameters::None,
    )?;
    conn.execute(
        format!("COPY {nn}_src TO '{nn}' (FORMAT bolt, USE_TMP_FILE FALSE, MODE 'replace')"),
        Parameters::None,
    )?;
    materialize(conn, &nn)?;
    let original_nn = match original {
        Some(original) => {
            conn.execute(
                format!(
                    "CREATE OR REPLACE TABLE {original}_nn AS SELECT * FROM {original} WHERE COLUMNS(*) IS NOT NULL"
                ),
                Parameters::None,
            )?;
            Some(format!("{original}_nn"))
        }
        None => None,
    };
    assert_predicates_on(conn, original_nn.as_deref(), &nn, predicates)
}

fn assert_predicates_on(
    conn: &Connection,
    original: Option<&str>,
    bolt_name: &str,
    predicates: &[&str],
) -> Result<()> {
    let mut failures = Vec::new();
    for pred in predicates {
        let scan = format!("bolt_scan('{bolt_name}')");
        let actual = match fingerprint(conn, &scan, pred) {
            Ok(v) => v,
            Err(e) => {
                failures.push(format!("{pred}: bolt_scan error: {e}"));
                continue;
            }
        };
        let reference = fingerprint(conn, &format!("{bolt_name}_m"), pred)?;
        if actual != reference {
            failures.push(format!(
                "{pred}: bolt_scan {actual:?} != materialized copy {reference:?}"
            ));
        }
        // Full row values (every column, NULLs included), not only counts.
        match row_diff(conn, &scan, &format!("{bolt_name}_m"), pred) {
            Ok(0) => {}
            Ok(n) => failures.push(format!(
                "{pred}: {n} row(s) differ from the materialized copy"
            )),
            Err(e) => failures.push(format!("{pred}: row comparison error: {e}")),
        }
        if let Some(original) = original {
            let expected = fingerprint(conn, original, pred)?;
            if actual != expected {
                failures.push(format!(
                    "{pred}: bolt_scan {actual:?} != DuckDB table {expected:?}"
                ));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{bolt_name}: {} predicate mismatch(es):\n{}",
        failures.len(),
        failures.join("\n")
    );
    Ok(())
}

#[test]
fn w1a_explicit_cast_is_not_stripped() -> Result<()> {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let (_env, conn) = setup_connection()?;
    conn.execute(
        "CREATE TABLE cast_src AS
         SELECT i::INTEGER AS id, (i + 0.4)::DOUBLE AS x, (i * 3)::SMALLINT AS small
         FROM range(0, 10) t(i)",
        Parameters::None,
    )?;
    upload(&conn, "cast_src", "w1a_cast")?;

    // The review's reproduction: DuckDB returns exactly one row (x = 3.4).
    assert_eq!(
        query_i64(
            &conn,
            "SELECT count(*) FROM bolt_scan('w1a_cast') WHERE CAST(x AS INTEGER) = 3"
        )?,
        1
    );
    assert_predicates(
        &conn,
        Some("cast_src"),
        "w1a_cast",
        &[
            "CAST(x AS INTEGER) = 3",
            "x::INTEGER > 5",
            "CAST(x AS BIGINT) <= 2",
            "CAST(x AS DECIMAL(4,1)) = 3.4",
            "CAST(id AS VARCHAR) LIKE '1%'",
            "CAST(id AS VARCHAR) = '7'",
            "CAST(id AS DOUBLE) = 7",
            "CAST(id AS BIGINT) = 7",
            "id::BIGINT > 5000000000",
            "id::BIGINT > -5000000000",
            "small = 9",
            "small::INTEGER < 10",
            "CAST(x AS FLOAT) > 2.4",
            "TRY_CAST(x AS TINYINT) = 4",
        ],
    )
}

#[test]
fn w1a_integer_arithmetic_overflow_is_not_wrapped() -> Result<()> {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let (_env, conn) = setup_connection()?;
    conn.execute(
        "CREATE TABLE ovf_src AS
         SELECT i::INTEGER AS id, (2147483600 + i)::INTEGER AS v,
                (9223372036854775000 + i)::BIGINT AS b, (i % 7)::INTEGER AS a,
                (i * 1.5)::DOUBLE AS d
         FROM range(0, 5) t(i)",
        Parameters::None,
    )?;
    upload(&conn, "ovf_src", "w1a_ovf")?;

    // DuckDB raises an Out of Range error for integer overflow (or rewrites the predicate,
    // e.g. `v + 100 > 0` into `v > -100`); the GPU would wrap silently and return 0 rows.
    // Whatever DuckDB does, bolt_scan must do the same.
    for pred in [
        "v + 100 > 0",
        "v + id + 100 > 0",
        "v + v > 0",
        "v * 2 > 0",
        "v * v > 0",
        "b * 2 > 0",
        "b + b > 0",
        "b + 1000 > id",
        "-v - v < 0",
    ] {
        let native = query_i64(&conn, &format!("SELECT count(*) FROM ovf_src WHERE {pred}"));
        let scanned = query_i64(
            &conn,
            &format!("SELECT count(*) FROM bolt_scan('w1a_ovf') WHERE {pred}"),
        );
        match (&native, &scanned) {
            (Ok(n), Ok(s)) => assert_eq!(s, n, "count mismatch for {pred}"),
            (Err(_), Err(_)) => {}
            _ => panic!("{pred}: DuckDB {native:?} but bolt_scan {scanned:?}"),
        }
    }
    assert!(
        query_i64(
            &conn,
            "SELECT count(*) FROM bolt_scan('w1a_ovf') WHERE v + v > 0"
        )
        .is_err(),
        "INTEGER overflow must raise an error, not wrap"
    );
    assert_eq!(
        query_i64(
            &conn,
            "SELECT count(*) FROM bolt_scan('w1a_ovf') WHERE v + 100 > 0"
        )?,
        5
    );

    // Non-overflowing arithmetic still matches.
    assert_predicates(
        &conn,
        Some("ovf_src"),
        "w1a_ovf",
        &[
            "a + id > 6",
            "a * 2 - id >= 1",
            "v - 2147483600 = 3",
            "d * 2 > 5",
            "d + id > 4.5",
            "a + 0.5 > 3",
            "d - 1.5 = id * 1.5 - 1.5",
        ],
    )
}

#[test]
fn w1a_bolt_aggregate_uses_resolved_table_name() -> Result<()> {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let (_env, conn) = setup_connection()?;
    conn.execute(
        "CREATE TABLE agg_local AS SELECT i::BIGINT AS id, (i % 3)::BIGINT AS g FROM range(1, 31) t(i)",
        Parameters::None,
    )?;
    conn.execute(
        "COPY agg_local TO 'gpu_d' (FORMAT bolt, USE_TMP_FILE FALSE)",
        Parameters::None,
    )?;
    for name in ["gpu_d", "GPU_D", "bolt.gpu_d", "bolt.GPU_D"] {
        let total = query_i64(
            &conn,
            &format!("SELECT total::BIGINT FROM bolt_aggregate('{name}', '', 'sum(id) AS total')"),
        )?;
        assert_eq!(total, 465, "bolt_aggregate('{name}')");
        let groups = query_i64(
            &conn,
            &format!("SELECT count(*) FROM bolt_aggregate('{name}', 'g', 'count(*) AS c')"),
        )?;
        assert_eq!(groups, 3, "bolt_aggregate('{name}') GROUP BY");
    }
    Ok(())
}

#[test]
fn w1a_bolt_scan_rejects_schema_drift_between_bind_and_init() -> Result<()> {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let (_env, conn) = setup_connection()?;
    conn.execute(
        "CREATE TABLE drift_a AS SELECT i::INTEGER AS id, 'x' || i::VARCHAR AS s FROM range(0, 10) t(i)",
        Parameters::None,
    )?;
    conn.execute(
        "CREATE TABLE drift_b AS SELECT i::VARCHAR AS s, i::DOUBLE AS id, i::BIGINT AS extra FROM range(0, 4) t(i)",
        Parameters::None,
    )?;
    conn.execute(
        "COPY drift_a TO 'w1a_drift' (FORMAT bolt, USE_TMP_FILE FALSE)",
        Parameters::None,
    )?;

    // Bound against the first schema.
    let stmt = conn.prepare(
        "SELECT sum(id)::BIGINT FROM bolt_scan('w1a_drift') WHERE id > 2",
        false,
    )?;
    {
        let mut res = stmt.execute(Parameters::None)?;
        let chunk = res.next_chunk()?.expect("chunk");
        assert_eq!(
            chunk.get_vector_at::<i64>(0)?.reader()?.get(0)?.copied(),
            Some(42)
        );
    }

    // Replace the table with a different schema; the prepared plan must not run against it.
    conn.execute(
        "COPY drift_b TO 'w1a_drift' (FORMAT bolt, USE_TMP_FILE FALSE, MODE 'replace')",
        Parameters::None,
    )?;
    let err = match stmt.execute(Parameters::None) {
        Err(e) => e.to_string(),
        Ok(mut res) => match res.next_chunk() {
            Err(e) => e.to_string(),
            Ok(_) => panic!("stale bolt_scan plan executed against a replaced table"),
        },
    };
    assert!(
        err.contains("replaced after the query was bound"),
        "unexpected error: {err}"
    );

    // Dropping the table is reported the same way.
    conn.execute(
        "COPY drift_a TO 'w1a_drift' (FORMAT bolt, USE_TMP_FILE FALSE, MODE 'replace')",
        Parameters::None,
    )?;
    let stmt = conn.prepare("SELECT count(*) FROM bolt_scan('w1a_drift')", false)?;
    conn.execute("SELECT bolt_drop_table('w1a_drift')", Parameters::None)?;
    let failed = match stmt.execute(Parameters::None) {
        Err(_) => true,
        Ok(mut res) => res.next_chunk().is_err(),
    };
    assert!(
        failed,
        "bolt_scan must fail when the table was dropped after bind"
    );

    // A fresh query binds against the new schema and works.
    conn.execute(
        "COPY drift_b TO 'w1a_drift' (FORMAT bolt, USE_TMP_FILE FALSE, MODE 'replace')",
        Parameters::None,
    )?;
    assert_eq!(
        query_i64(
            &conn,
            "SELECT count(*) FROM bolt_scan('w1a_drift') WHERE extra >= 2"
        )?,
        2
    );
    Ok(())
}

#[test]
fn w1a_timestamptz_literals_do_not_depend_on_session_time_zone() -> Result<()> {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let (_env, conn) = setup_connection()?;
    conn.execute(
        "CREATE TABLE tz_src AS
         SELECT i::INTEGER AS id,
                TIMESTAMPTZ '2024-01-01 00:00:00+00' + INTERVAL (i) HOUR AS tstz,
                TIMESTAMP '2024-01-01 00:00:00' + INTERVAL (i) HOUR AS ts
         FROM range(0, 48) t(i)",
        Parameters::None,
    )?;
    upload(&conn, "tz_src", "w1a_tz")?;

    // A table stored as UTC instants (`Timestamp(us, "UTC")`), registered directly.
    let micros: Vec<i64> = (0..48)
        .map(|h| 1_704_067_200_000_000 + h * 3_600_000_000)
        .collect();
    let utc = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new(
                "tstz",
                DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                true,
            ),
        ])),
        vec![
            Arc::new(Int32Array::from_iter_values(0..48)),
            Arc::new(TimestampMicrosecondArray::from(micros).with_timezone("UTC")),
        ],
    )
    .expect("utc batch");
    BoltRuntime::global().register_table("w1a_tz_utc", utc, RegisterMode::Replace)?;
    materialize(&conn, "w1a_tz_utc")?;

    let predicates = [
        "tstz >= TIMESTAMPTZ '2024-01-01 05:00:00+00'",
        "tstz < TIMESTAMPTZ '2024-01-01 05:00:00-03'",
        "tstz BETWEEN TIMESTAMPTZ '2024-01-01 03:00:00+00' AND TIMESTAMPTZ '2024-01-01 09:00:00+00'",
        "tstz IN (TIMESTAMPTZ '2024-01-01 07:00:00+00', TIMESTAMPTZ '2024-01-02 07:00:00+00')",
        "tstz = TIMESTAMPTZ '2024-01-01 10:00:00+00'",
    ];
    for zone in ["UTC", "America/Argentina/Buenos_Aires", "Asia/Kolkata"] {
        conn.execute(format!("SET TimeZone = '{zone}'"), Parameters::None)?;
        assert_predicates(&conn, None, "w1a_tz", &predicates)?;
        assert_predicates(
            &conn,
            None,
            "w1a_tz",
            &[
                "ts >= TIMESTAMP '2024-01-01 05:00:00'",
                "ts::TIMESTAMPTZ >= TIMESTAMPTZ '2024-01-01 05:00:00+00'",
            ],
        )?;
        assert_predicates(&conn, None, "w1a_tz_utc", &predicates)?;
    }

    // Exact expectations for the UTC-instant table once `bolt_scan` exposes it as
    // TIMESTAMPTZ (independent of the session time zone). While it is still exposed as a
    // naive TIMESTAMP, DuckDB interprets it in the session time zone and the differential
    // checks above are the meaningful ones.
    conn.execute(
        "SET TimeZone = 'America/Argentina/Buenos_Aires'",
        Parameters::None,
    )?;
    let exposed = query_string(
        &conn,
        "SELECT typeof(tstz) FROM bolt_scan('w1a_tz_utc') LIMIT 1",
    )?;
    if exposed == "TIMESTAMP WITH TIME ZONE" {
        assert_exact_utc_counts(&conn)?;
    }
    conn.execute("RESET TimeZone", Parameters::None)?;
    Ok(())
}

fn assert_exact_utc_counts(conn: &Connection) -> Result<()> {
    assert_eq!(
        query_i64(
            conn,
            "SELECT count(*) FROM bolt_scan('w1a_tz_utc') WHERE tstz >= TIMESTAMPTZ '2024-01-01 05:00:00+00'"
        )?,
        43
    );
    assert_eq!(
        query_i64(
            conn,
            "SELECT count(*) FROM bolt_scan('w1a_tz_utc') WHERE tstz = TIMESTAMPTZ '2024-01-01 10:00:00+00'"
        )?,
        1
    );
    Ok(())
}

#[test]
fn w1a_predicate_type_matrix_matches_duckdb() -> Result<()> {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let (_env, conn) = setup_connection()?;
    // 5000 rows: several output chunks, NULL-heavy columns, NaN / infinities, quotes,
    // unicode, and every type the pushdown translator handles.
    conn.execute(
        "CREATE TABLE mx_src AS
         SELECT i::INTEGER AS id,
                CASE WHEN i % 2 = 0 THEN NULL ELSE (i % 50)::INTEGER END AS n_int,
                (i * 1000003)::BIGINT AS big,
                (i % 300)::SMALLINT AS small,
                (i % 200)::UTINYINT AS tiny_u,
                CASE WHEN i % 13 = 0 THEN NULL
                     WHEN i % 17 = 0 THEN 'NaN'::DOUBLE
                     WHEN i = 1001 THEN 'inf'::DOUBLE
                     WHEN i = 1002 THEN '-inf'::DOUBLE
                     WHEN i = 1003 THEN -0.0::DOUBLE
                     ELSE (i % 97) / 7.0 END AS d,
                CASE WHEN i % 11 = 0 THEN NULL
                     WHEN i % 19 = 0 THEN 'NaN'::FLOAT
                     ELSE ((i % 40) * 0.1)::FLOAT END AS f,
                CASE WHEN i % 7 = 0 THEN NULL ELSE ((i % 1000) * 1.25)::DECIMAL(10,2) END AS dec,
                DATE '2020-01-01' + (i % 1500)::INTEGER AS dt,
                CASE WHEN i % 23 = 0 THEN NULL
                     ELSE TIMESTAMP '2024-01-01 00:00:00' + INTERVAL (i * 37) MINUTE END AS ts,
                (TIMESTAMP '2024-01-01 00:00:00' + INTERVAL (i * 1001) MILLISECOND)::TIMESTAMP_NS AS ts_ns,
                CASE WHEN i % 5 = 0 THEN NULL
                     WHEN i % 5 = 1 THEN 'north_' || (i % 40)::VARCHAR
                     WHEN i % 5 = 2 THEN 'O''Brien ' || (i % 3)::VARCHAR
                     WHEN i % 5 = 3 THEN 'Zé' || (i % 10)::VARCHAR
                     ELSE '50%_off\\' || (i % 4)::VARCHAR END AS s,
                CASE WHEN i % 3 = 0 THEN NULL ELSE i % 2 = 0 END AS b
         FROM range(0, 5000) t(i)",
        Parameters::None,
    )?;
    upload(&conn, "mx_src", "w1a_mx")?;

    let predicates = [
        // integer comparisons, NULL-heavy column
        "id = 42",
        "id != 42",
        "id < 100",
        "id >= 4900",
        "100 > id",
        "n_int = 7",
        "n_int <> 7",
        "n_int > 25",
        "n_int <= 3",
        "n_int = id",
        "n_int < id",
        "big > 4000000000",
        "big BETWEEN 1000003 AND 5000015",
        "small = 299",
        "small > 250",
        "tiny_u >= 150",
        "tiny_u = 0",
        // BETWEEN / IN / NOT IN, including NULL bounds and NULL list items
        "id BETWEEN 10 AND 20",
        "id NOT BETWEEN 10 AND 4990",
        "n_int BETWEEN 10 AND 20",
        "NOT (n_int BETWEEN 10 AND 20)",
        "id BETWEEN NULL AND 5",
        "id BETWEEN 4995 AND NULL",
        "NOT (id BETWEEN 4995 AND NULL)",
        "id BETWEEN n_int AND 30",
        "NOT (id BETWEEN n_int AND 30)",
        "id > 10 AND id < 20",
        "id >= 10 AND id < 20",
        "id > 10 AND id <= 20",
        "id IN (1, 2, 3, 4999)",
        "id NOT IN (1, 2, 3)",
        "n_int IN (1, 3, 5)",
        "n_int NOT IN (1, 3, 5)",
        "n_int IN (1, NULL)",
        "n_int NOT IN (1, NULL)",
        "id NOT IN (1, NULL)",
        "id IN (n_int, 4)",
        // IS NULL / NOT / AND / OR
        "n_int IS NULL",
        "n_int IS NOT NULL",
        "NOT (n_int IS NULL)",
        "NOT (n_int > 25)",
        "NOT (n_int > 25 AND id < 2000)",
        "NOT (n_int > 25 OR id < 2000)",
        "NOT (n_int = id)",
        "NOT (NOT (n_int > 10))",
        "n_int IS NULL OR n_int > 40",
        "n_int > 40 OR id < 3",
        "(n_int > 10 AND n_int < 20) OR d > 10",
        "b",
        "NOT b",
        "b IS NULL",
        "b = true",
        "b != false",
        // DOUBLE with NULL / NaN / +-inf / -0.0
        "d > 3",
        "d >= 3",
        "d < 3",
        "d <= 0",
        "d = 0",
        "d != 2",
        "d = 'NaN'::DOUBLE",
        "d > 'inf'::DOUBLE",
        "d < '-inf'::DOUBLE",
        "NOT (d > 3)",
        "NOT (d != 2)",
        "d BETWEEN 1 AND 5",
        "NOT (d BETWEEN 1 AND 5)",
        "d IN (1.0, 2.0)",
        "d NOT IN (1.0, 2.0)",
        "d > id",
        "d * 2 > 10",
        "d + 1 >= 5",
        "d - id < -100",
        "d > 1e300",
        "d < 1.5e-7 AND d > -1.5e-7",
        // FLOAT literals must be exact (0.3f32 != 0.3)
        "f = 0.3",
        "f = 1.1",
        "f > 2.2",
        "f <= 0.7",
        "f != 0.5",
        "f BETWEEN 0.3 AND 0.9",
        "f IN (0.1, 0.2, 3.9)",
        "NOT (f < 1.5)",
        // DECIMAL
        "dec = 2.50",
        "dec > 1000",
        "dec BETWEEN 10.25 AND 20.75",
        "dec IN (1.25, 2.50)",
        "dec IS NULL",
        // DATE / TIMESTAMP / TIMESTAMP_NS
        "dt = DATE '2021-06-15'",
        "dt >= DATE '2023-01-01'",
        "dt BETWEEN DATE '2020-02-01' AND DATE '2020-03-01'",
        "dt > TIMESTAMP '2021-01-01 12:00:00'",
        "dt IN (DATE '2020-01-01', DATE '2020-01-05')",
        "ts >= TIMESTAMP '2024-02-01 00:00:00'",
        "ts < TIMESTAMP '2024-01-02 12:30:15.5'",
        "ts = TIMESTAMP '2024-01-01 00:37:00'",
        "ts BETWEEN TIMESTAMP '2024-01-05 00:00:00' AND TIMESTAMP '2024-01-06 00:00:00'",
        "ts IS NULL",
        "ts_ns > TIMESTAMP_NS '2024-01-01 01:00:00.001'",
        "ts_ns = TIMESTAMP_NS '2024-01-01 00:00:01.001'",
        "ts_ns < TIMESTAMP '2024-01-01 00:10:00'",
        "dt < DATE '1999-01-01' OR dt > DATE '2099-01-01'",
        // VARCHAR comparisons, quotes, LIKE escaping, unicode
        "s = 'north_7'",
        "s = 'O''Brien 1'",
        "s LIKE 'O''Brien%'",
        "s LIKE 'north_1%'",
        "s LIKE 'Z_1'",
        "s LIKE 'Z%'",
        "s NOT LIKE 'north%'",
        "s LIKE '50\\%%' ESCAPE '\\'",
        "s LIKE '%\\\\%'",
        "s LIKE '50%_off%'",
        "s = '50%_off\\1'",
        "s < 'O'",
        "s >= 'Zé5'",
        "s IN ('north_1', 'Zé3')",
        "s NOT IN ('north_1', 'Zé3')",
        "starts_with(s, 'nor')",
        "suffix(s, '1')",
        "contains(s, 'Bri')",
        "contains(s, '%_')",
        "s IS NULL",
        "s LIKE 'north%' AND id < 1000",
        "s LIKE 'north%' OR n_int = 3",
        "NOT (s LIKE 'north%')",
        "lower(s) = 'north_1'",
        "length(s) > 8",
    ];
    assert_predicates(&conn, None, "w1a_mx", &predicates)?;

    // Columns whose storage type round-trips exactly can also be compared with the
    // original DuckDB table.
    assert_predicates(
        &conn,
        Some("mx_src"),
        "w1a_mx",
        &[
            "id < 100",
            "n_int NOT IN (1, 3, 5)",
            "NOT (n_int > 25 AND id < 2000)",
            "d > 3",
            "d != 2",
            "f = 0.3",
            "dt >= DATE '2023-01-01'",
            "ts >= TIMESTAMP '2024-02-01 00:00:00'",
            "s LIKE 'O''Brien%'",
        ],
    )
}

#[test]
fn w1a_projection_with_duplicated_and_filtered_columns() -> Result<()> {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let (_env, conn) = setup_connection()?;
    conn.execute(
        "CREATE TABLE proj_src AS
         SELECT i::INTEGER AS id, (i % 10)::INTEGER AS k,
                CASE WHEN i % 4 = 0 THEN NULL ELSE 'v' || (i % 6)::VARCHAR END AS s
         FROM range(0, 3000) t(i)",
        Parameters::None,
    )?;
    upload(&conn, "proj_src", "w1a_proj")?;

    let queries = [
        "SELECT count(*) FROM (SELECT s, s, id FROM {t} WHERE s = 'v1')",
        "SELECT coalesce(sum(id + k), 0)::BIGINT FROM (SELECT id, k, k, id FROM {t} WHERE k > 6)",
        "SELECT count(*) FROM (SELECT k FROM {t} WHERE id > 100 AND s IS NOT NULL)",
        "SELECT count(*) FROM {t} WHERE s LIKE 'v%' AND k = 3",
        "SELECT count(*) FROM {t}",
        "SELECT max(length(s))::BIGINT FROM {t} WHERE s IS NOT NULL",
        "SELECT count(DISTINCT s)::BIGINT FROM {t} WHERE k IN (1, 2)",
    ];
    for q in queries {
        let expected = query_i64(&conn, &q.replace("{t}", "proj_src"))?;
        let actual = query_i64(&conn, &q.replace("{t}", "bolt_scan('w1a_proj')"))?;
        assert_eq!(actual, expected, "mismatch for {q}");
    }

    let first = query_string(
        &conn,
        "SELECT s || '|' || s FROM bolt_scan('w1a_proj') WHERE id = 5",
    )?;
    assert_eq!(first, "v5|v5");
    Ok(())
}

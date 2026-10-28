// SPDX-License-Identifier: Apache-2.0

//! W3-C regression tests:
//!
//! * W2-C-003: `COPY ... FROM (FORMAT bolt)` conversions that depend on the session `TimeZone`
//!   (`TIMESTAMPTZ -> TIMESTAMP / DATE / VARCHAR`, `TIMESTAMP / VARCHAR -> TIMESTAMPTZ`) give
//!   exactly what DuckDB's own `CAST` gives in the same session.
//! * W1-C-003: `USE_TMP_FILE` (explicit, and DuckDB's default when the target file exists) is
//!   rejected before anything is registered; `PARTITION_BY` leaves no directory behind.
//! * Unsafe / FFI audit: 128-bit unaligned read/write paths, the Arrow C Data Interface export
//!   (`collection_to_record_batches`) and the Arrow C Stream import
//!   (`arrow_query_to_record_batch`) round-trip values and NULLs across several chunks.
//!
//! Only the shadow catalog and COPY paths are used, so the tests are valid for both GPU and
//! `cuda-stub` builds.

use std::sync::{Mutex, MutexGuard, Once};

use arrow_array::{Array, Decimal128Array, Int64Array, StringArray};
use duckdb_bolt_core::{
    arrow_bridge::{
        OriginalDuckDbType as T, arrow_query_to_record_batch, conversion_depends_on_time_zone,
    },
    runtime::BoltRuntime,
};
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
// Serialize tests because Craton Bolt's runtime catalog is process-wide.
static TEST_MUTEX: Mutex<()> = Mutex::new(());

fn lock() -> MutexGuard<'static, ()> {
    TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner())
}

fn setup_connection() -> Result<(Environment, Connection)> {
    REGISTER_ONCE.call_once(|| {
        // SAFETY: `duckdb_extension_bolt_describe` is emitted by `#[duckdb_entrypoint_c_api_v2]`
        // above, which fills the descriptor as `duckdb_static_extension.h` specifies; this is
        // the contract `register_static_extension` requires. `Once` makes it a single call.
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

/// Run `sql`, require it to fail, and return the error message.
fn expect_err(conn: &Connection, sql: &str) -> String {
    match conn.execute(sql, Parameters::None) {
        Ok(_) => panic!("expected an error from: {sql}"),
        Err(e) => e.message,
    }
}

/// Both tables hold the same multiset of rows (and therefore compatible column types).
fn assert_same_rows(conn: &Connection, a: &str, b: &str, label: &str) -> Result<()> {
    for (x, y) in [(a, b), (b, a)] {
        assert_eq!(
            query_i64(
                conn,
                &format!("SELECT count(*) FROM (SELECT * FROM {x} EXCEPT ALL SELECT * FROM {y})"),
            )?,
            0,
            "{label}: rows in {x} missing from {y}"
        );
    }
    Ok(())
}

// ------------------------------------------------------------------------------- W2-C-003

#[test]
fn test_w3c_time_zone_dependent_conversions_are_classified() {
    use T::*;
    // Exactly one side is an instant.
    for (src, dst) in [
        (TimestampTz, Timestamp),
        (TimestampTz, TimestampNs),
        (TimestampTz, TimestampSec),
        (TimestampTz, Date),
        (TimestampTz, Varchar),
        (TimestampTzNs, Timestamp),
        (Timestamp, TimestampTz),
        (Date, TimestampTz),
        (Varchar, TimestampTz),
        (TimestampNs, TimestampTzNs),
    ] {
        assert!(
            conversion_depends_on_time_zone(src, dst),
            "{src:?} -> {dst:?}"
        );
    }
    // Instant -> instant (unit change only) and local -> local do not.
    for (src, dst) in [
        (TimestampTz, TimestampTz),
        (TimestampTz, TimestampTzNs),
        (Timestamp, TimestampNs),
        (Timestamp, Date),
        (SmallInt, Integer),
        (Varchar, Varchar),
    ] {
        assert!(
            !conversion_depends_on_time_zone(src, dst),
            "{src:?} -> {dst:?}"
        );
    }
    assert!(TimestampTz.is_instant() && TimestampTzNs.is_instant());
    assert!(!Timestamp.is_instant() && !Date.is_instant());
}

/// Session time zones to test: UTC, a western zone with DST, and a half-hour offset zone.
const ZONES: &[&str] = &["UTC", "America/New_York", "Asia/Kolkata"];

#[test]
fn test_w3c_copy_from_timestamptz_matches_duckdb_cast_in_session_zone() -> Result<()> {
    let _guard = lock();
    let (_env, conn) = setup_connection()?;

    // 5000 rows (three output chunks), a NULL every 7th row, instants around a DST switch.
    exec(
        &conn,
        "CREATE TABLE w3c_tz_src AS
         SELECT i::INTEGER AS id,
                CASE WHEN i % 7 = 3 THEN NULL
                     ELSE TIMESTAMPTZ '2024-03-09 20:00:00+00' + INTERVAL (i * 37) MINUTE END AS t,
                i::BIGINT AS v
         FROM range(0, 5000) r(i)",
    )?;
    exec(
        &conn,
        "COPY w3c_tz_src TO 'w3c_tz' (FORMAT bolt, USE_TMP_FILE FALSE)",
    )?;

    let result = (|| -> Result<()> {
        for zone in ZONES {
            exec(&conn, &format!("SET TimeZone = '{zone}'"))?;
            for (dest_type, name) in [
                ("TIMESTAMP", "ts"),
                ("DATE", "d"),
                ("VARCHAR", "s"),
                ("TIMESTAMPTZ", "tz"),
            ] {
                let back = format!("w3c_tz_back_{name}");
                let expected = format!("w3c_tz_exp_{name}");
                exec(&conn, &format!("DROP TABLE IF EXISTS {back}"))?;
                exec(&conn, &format!("DROP TABLE IF EXISTS {expected}"))?;
                exec(
                    &conn,
                    &format!("CREATE TABLE {back} (id INTEGER, t {dest_type}, v BIGINT)"),
                )?;
                exec(&conn, &format!("COPY {back} FROM 'w3c_tz' (FORMAT bolt)"))?;
                // DuckDB's own cast in the same session is the reference.
                exec(
                    &conn,
                    &format!(
                        "CREATE TABLE {expected} AS
                         SELECT id, CAST(t AS {dest_type}) AS t, v FROM w3c_tz_src"
                    ),
                )?;
                assert_same_rows(&conn, &back, &expected, &format!("{zone} {dest_type}"))?;
                assert_eq!(
                    query_i64(
                        &conn,
                        &format!("SELECT count(*) FROM {back} WHERE t IS NULL")
                    )?,
                    query_i64(&conn, "SELECT count(*) FROM w3c_tz_src WHERE t IS NULL")?,
                    "{zone} {dest_type}: NULLs must be preserved"
                );
            }
        }
        // Positional match (different names) takes the same path.
        exec(&conn, "SET TimeZone = 'America/New_York'")?;
        exec(
            &conn,
            "CREATE TABLE w3c_tz_pos (a INTEGER, b TIMESTAMP, c BIGINT)",
        )?;
        exec(&conn, "COPY w3c_tz_pos FROM 'w3c_tz' (FORMAT bolt)")?;
        assert_eq!(
            query_i64(
                &conn,
                "SELECT count(*) FROM w3c_tz_pos p JOIN w3c_tz_src s ON p.a = s.id
                 WHERE p.b IS DISTINCT FROM s.t::TIMESTAMP OR p.c <> s.v"
            )?,
            0
        );
        Ok(())
    })();
    exec(&conn, "RESET TimeZone")?;
    result
}

#[test]
fn test_w3c_copy_from_local_values_into_timestamptz_uses_session_zone() -> Result<()> {
    let _guard = lock();
    let (_env, conn) = setup_connection()?;

    exec(
        &conn,
        "CREATE TABLE w3c_local_src AS
         SELECT i::INTEGER AS id,
                CASE WHEN i % 5 = 0 THEN NULL
                     ELSE TIMESTAMP '2024-11-02 22:00:00' + INTERVAL (i * 13) MINUTE END AS ts,
                CASE WHEN i % 5 = 1 THEN NULL
                     ELSE DATE '2024-03-01' + (i % 60)::INTEGER END AS d,
                CASE WHEN i % 5 = 2 THEN NULL
                     ELSE strftime(TIMESTAMP '2024-06-01 12:00:00' + INTERVAL (i) HOUR, '%Y-%m-%d %H:%M:%S') END AS s
         FROM range(0, 3000) r(i)",
    )?;
    exec(
        &conn,
        "COPY w3c_local_src TO 'w3c_local' (FORMAT bolt, USE_TMP_FILE FALSE)",
    )?;

    let result = (|| -> Result<()> {
        for zone in ZONES {
            exec(&conn, &format!("SET TimeZone = '{zone}'"))?;
            exec(&conn, "DROP TABLE IF EXISTS w3c_local_back")?;
            exec(&conn, "DROP TABLE IF EXISTS w3c_local_exp")?;
            exec(
                &conn,
                "CREATE TABLE w3c_local_back (id INTEGER, ts TIMESTAMPTZ, d TIMESTAMPTZ, s TIMESTAMPTZ)",
            )?;
            exec(&conn, "COPY w3c_local_back FROM 'w3c_local' (FORMAT bolt)")?;
            exec(
                &conn,
                "CREATE TABLE w3c_local_exp AS
                 SELECT id, ts::TIMESTAMPTZ AS ts, d::TIMESTAMPTZ AS d, s::TIMESTAMPTZ AS s
                 FROM w3c_local_src",
            )?;
            assert_same_rows(&conn, "w3c_local_back", "w3c_local_exp", zone)?;
            // Compare instants, not renderings.
            assert_eq!(
                query_i64(
                    &conn,
                    "SELECT count(*) FROM w3c_local_back b JOIN w3c_local_exp e USING (id)
                     WHERE epoch_us(b.ts) IS DISTINCT FROM epoch_us(e.ts)
                        OR epoch_us(b.d) IS DISTINCT FROM epoch_us(e.d)
                        OR epoch_us(b.s) IS DISTINCT FROM epoch_us(e.s)"
                )?,
                0,
                "{zone}"
            );
        }
        Ok(())
    })();
    exec(&conn, "RESET TimeZone")?;
    result
}

#[test]
fn test_w3c_copy_from_timestamptz_unconvertible_value_is_an_error() -> Result<()> {
    let _guard = lock();
    let (_env, conn) = setup_connection()?;

    // A string that is not a timestamp: DuckDB's cast fails, so COPY FROM fails (it must not
    // insert NULL), exactly like `INSERT INTO t SELECT s::TIMESTAMPTZ`.
    exec(
        &conn,
        "COPY (SELECT 1::INTEGER AS id, 'not a timestamp'::VARCHAR AS s) TO 'w3c_badts' (FORMAT bolt, USE_TMP_FILE FALSE)",
    )?;
    exec(
        &conn,
        "CREATE TABLE w3c_badts_back (id INTEGER, s TIMESTAMPTZ)",
    )?;
    let err = expect_err(&conn, "COPY w3c_badts_back FROM 'w3c_badts' (FORMAT bolt)");
    assert!(err.contains("COPY FROM (FORMAT bolt) failed"), "{err}");
    assert_eq!(query_i64(&conn, "SELECT count(*) FROM w3c_badts_back")?, 0);
    Ok(())
}

// ------------------------------------------------------------------------------- W1-C-003

/// A fresh path under the system temp directory (removed if it exists), with `/` separators so
/// it can be embedded in SQL on every platform.
fn temp_path(name: &str) -> (std::path::PathBuf, String) {
    let path = std::env::temp_dir().join(format!("{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&path);
    let _ = std::fs::remove_file(&path);
    let sql = path
        .to_string_lossy()
        .replace('\\', "/")
        .replace('\'', "''");
    (path, sql)
}

fn table_exists(name: &str) -> bool {
    BoltRuntime::global().get_table_batches(name).is_ok()
}

#[test]
fn test_w3c_copy_to_use_tmp_file_is_rejected_before_registration() -> Result<()> {
    let _guard = lock();
    let (_env, conn) = setup_connection()?;

    // Explicit USE_TMP_FILE TRUE.
    let err = expect_err(
        &conn,
        "COPY (SELECT 1 AS a) TO 'w3c_tmp_explicit' (FORMAT bolt, USE_TMP_FILE TRUE)",
    );
    assert!(err.contains("USE_TMP_FILE"), "{err}");
    assert!(!table_exists("w3c_tmp_explicit"));

    // DuckDB's default: USE_TMP_FILE is on when a file with the target path exists.
    let (path, sql_path) = temp_path("w3c_existing_target");
    std::fs::write(&path, b"keep me").expect("write marker file");
    let err = expect_err(
        &conn,
        &format!("COPY (SELECT 1 AS a) TO '{sql_path}' (FORMAT bolt)"),
    );
    assert!(err.contains("USE_TMP_FILE FALSE"), "{err}");
    let table = path.file_name().unwrap().to_string_lossy().to_string();
    assert!(!table_exists(&table));
    assert_eq!(
        std::fs::read(&path).expect("marker file still exists"),
        b"keep me"
    );

    // With USE_TMP_FILE FALSE the COPY works and the file is left alone.
    exec(
        &conn,
        &format!("COPY (SELECT 1 AS a) TO '{sql_path}' (FORMAT bolt, USE_TMP_FILE FALSE)"),
    )?;
    assert!(table_exists(&table));
    assert_eq!(
        std::fs::read(&path).expect("marker file still exists"),
        b"keep me"
    );
    let _ = std::fs::remove_file(&path);
    Ok(())
}

#[test]
fn test_w3c_copy_to_partition_by_leaves_no_directory() -> Result<()> {
    let _guard = lock();
    let (_env, conn) = setup_connection()?;

    let (path, sql_path) = temp_path("w3c_part_dir");
    let err = expect_err(
        &conn,
        &format!(
            "COPY (SELECT (i % 2)::INTEGER AS k, i::BIGINT AS v FROM range(10) t(i))
             TO '{sql_path}' (FORMAT bolt, PARTITION_BY (k))"
        ),
    );
    assert!(err.contains("exactly one Bolt table"), "{err}");
    // DuckDB removes the output directory it created for the failed statement.
    assert!(
        !path.exists(),
        "PARTITION_BY left '{}' behind",
        path.display()
    );
    Ok(())
}

#[test]
fn test_w3c_copy_to_return_stats_reports_rows_and_arrow_bytes() -> Result<()> {
    let _guard = lock();
    let (_env, conn) = setup_connection()?;
    // RETURN_STATS: the row count is exact; `file_size_bytes` is the Arrow memory registered,
    // which is positive for a non-empty table (there is no file).
    let mut result = conn.query(
        "COPY (SELECT i::BIGINT AS v FROM range(3000) t(i)) TO 'w3c_stats' (FORMAT bolt, USE_TMP_FILE FALSE, RETURN_STATS)",
        Parameters::None,
    )?;
    let chunk = result.next_chunk()?.expect("RETURN_STATS returns one row");
    // Columns: filename, count, file_size_bytes, ...
    assert_eq!(
        chunk.get_vector_at::<u64>(1)?.reader()?.get(0)?,
        Some(&3000)
    );
    let bytes = *chunk
        .get_vector_at::<u64>(2)?
        .reader()?
        .get(0)?
        .expect("file_size_bytes is reported");
    // At least the 3000 * 8 value bytes of the BIGINT column.
    assert!(bytes >= 24_000, "file_size_bytes = {bytes}");
    Ok(())
}

// --------------------------------------------------------------------- unsafe / FFI audit

#[test]
fn test_w3c_hugeint_and_wide_decimal_round_trip_with_nulls() -> Result<()> {
    let _guard = lock();
    let (_env, conn) = setup_connection()?;

    // HUGEINT is stored as Int64 and restored through the unaligned 128-bit writer
    // (`write_unaligned_i128_column::<i128>`); DECIMAL(38, 4) keeps Decimal128 storage. 5000
    // rows cover three output chunks; NULLs on both sides of every chunk boundary.
    exec(
        &conn,
        "CREATE TABLE w3c_wide_src AS
         SELECT i::INTEGER AS id,
                CASE WHEN i % 11 IN (0, 7) OR i IN (2047, 2048, 4095) THEN NULL
                     ELSE ((i - 2500) * 3037000493)::HUGEINT END AS h,
                CASE WHEN i % 13 = 5 THEN NULL
                     ELSE (i::DECIMAL(38, 4) * 1234567890123.4567 - 99999999999999.0001)::DECIMAL(38, 4) END AS d
         FROM range(0, 5000) r(i)",
    )?;
    exec(
        &conn,
        "COPY w3c_wide_src TO 'w3c_wide' (FORMAT bolt, USE_TMP_FILE FALSE)",
    )?;
    exec(
        &conn,
        "CREATE TABLE w3c_wide_back (id INTEGER, h HUGEINT, d DECIMAL(38, 4))",
    )?;
    exec(&conn, "COPY w3c_wide_back FROM 'w3c_wide' (FORMAT bolt)")?;
    assert_same_rows(&conn, "w3c_wide_src", "w3c_wide_back", "wide")?;
    Ok(())
}

#[test]
fn test_w3c_arrow_stream_import_round_trips_values_and_nulls() -> Result<()> {
    let _guard = lock();
    let (_env, conn) = setup_connection()?;

    // Several stream batches, a wide decimal (16-byte values) and strings with NULLs.
    let batch = arrow_query_to_record_batch(
        &conn,
        "SELECT i::BIGINT AS v,
                CASE WHEN i % 3 = 0 THEN NULL ELSE (i::DECIMAL(38, 2) / 7)::DECIMAL(38, 2) END AS d,
                CASE WHEN i % 4 = 0 THEN NULL ELSE 's' || i::VARCHAR END AS s
         FROM range(0, 6000) t(i) ORDER BY i",
    )?;
    assert_eq!(batch.num_rows(), 6000);
    let v = batch
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("BIGINT column");
    let d = batch
        .column(1)
        .as_any()
        .downcast_ref::<Decimal128Array>()
        .expect("DECIMAL(38, 2) column");
    // DuckDB may export VARCHAR as Utf8 or Utf8View depending on its settings.
    let s = batch.column(2);
    for i in [0usize, 1, 2, 2047, 2048, 4095, 5999] {
        assert_eq!(v.value(i), i as i64);
        assert_eq!(d.is_null(i), i % 3 == 0, "row {i}");
        assert_eq!(s.is_null(i), i % 4 == 0, "row {i}");
    }
    if let Some(s) = s.as_any().downcast_ref::<StringArray>() {
        assert_eq!(s.value(5), "s5");
    }
    // (i / 7) rounded to 2 decimals, as DuckDB computes it.
    assert_eq!(d.value(1), 14); // 0.142857.. -> 0.14
    assert_eq!(d.value(2048), 29257); // 292.571.. -> 292.57

    // An empty result keeps the schema.
    let empty = arrow_query_to_record_batch(&conn, "SELECT 1::BIGINT AS v WHERE false")?;
    assert_eq!(empty.num_rows(), 0);
    assert_eq!(empty.num_columns(), 1);
    Ok(())
}

#[test]
fn test_w3c_collection_export_round_trips_many_chunks() -> Result<()> {
    let _guard = lock();
    let (_env, conn) = setup_connection()?;

    // COPY TO exports through `collection_to_record_batches` (Arrow C Data Interface import of
    // DuckDB's exporter output: from_ffi + align_buffers + validate). 10000 rows of mixed types.
    exec(
        &conn,
        "CREATE TABLE w3c_export_src AS
         SELECT i::INTEGER AS i32, (i * 1000003)::BIGINT AS i64,
                CASE WHEN i % 9 = 0 THEN NULL ELSE 'row ' || i::VARCHAR END AS s,
                (i::DECIMAL(18, 3) / 8)::DECIMAL(18, 3) AS d18,
                (i::DECIMAL(38, 6) * 1000000007)::DECIMAL(38, 6) AS d38,
                i % 2 = 0 AS b, (i / 3.0)::DOUBLE AS f64
         FROM range(0, 10000) t(i)",
    )?;
    exec(
        &conn,
        "COPY w3c_export_src TO 'w3c_export' (FORMAT bolt, USE_TMP_FILE FALSE)",
    )?;
    let (_, batches) = BoltRuntime::global().get_table_batches("w3c_export")?;
    assert_eq!(batches.iter().map(|b| b.num_rows()).sum::<usize>(), 10000);
    for batch in &batches {
        for col in batch.columns() {
            // Every imported buffer satisfies arrow-rs's alignment / size invariants.
            col.to_data().validate_full().expect("valid Arrow data");
        }
    }
    exec(
        &conn,
        "CREATE TABLE w3c_export_back AS SELECT * FROM w3c_export_src WHERE false",
    )?;
    exec(
        &conn,
        "COPY w3c_export_back FROM 'w3c_export' (FORMAT bolt)",
    )?;
    assert_same_rows(&conn, "w3c_export_src", "w3c_export_back", "export")?;
    Ok(())
}

// ------------------------------------------------------------------- W3-E-001 / W3-E-003

#[test]
fn test_w3c_prepared_copy_from_reads_current_rows() -> Result<()> {
    let _guard = lock();
    let (_env, conn) = setup_connection()?;

    exec(
        &conn,
        "COPY (SELECT 1::INTEGER AS id, 1.0::DOUBLE AS v) TO 'w3c_prep' (FORMAT bolt, USE_TMP_FILE FALSE)",
    )?;
    exec(&conn, "CREATE TABLE w3c_prep_dst (id INTEGER, v DOUBLE)")?;
    exec(
        &conn,
        "PREPARE w3c_c AS COPY w3c_prep_dst FROM 'w3c_prep' (FORMAT bolt)",
    )?;

    exec(&conn, "EXECUTE w3c_c")?;
    assert_eq!(query_i64(&conn, "SELECT count(*) FROM w3c_prep_dst")?, 1);

    // Replace the Bolt table with 5 rows (same schema): the next EXECUTE must read them.
    exec(
        &conn,
        "COPY (SELECT range::INTEGER AS id, 1.0::DOUBLE AS v FROM range(5)) TO 'w3c_prep' (FORMAT bolt, USE_TMP_FILE FALSE)",
    )?;
    exec(&conn, "EXECUTE w3c_c")?;
    assert_eq!(query_i64(&conn, "SELECT count(*) FROM w3c_prep_dst")?, 6);

    // Append: 5 + 2 rows.
    exec(
        &conn,
        "COPY (SELECT 7::INTEGER AS id, 2.0::DOUBLE AS v FROM range(2)) TO 'w3c_prep' (FORMAT bolt, MODE 'append', USE_TMP_FILE FALSE)",
    )?;
    exec(&conn, "EXECUTE w3c_c")?;
    assert_eq!(query_i64(&conn, "SELECT count(*) FROM w3c_prep_dst")?, 13);

    // A schema change after PREPARE is a clear error, not misread data.
    exec(
        &conn,
        "COPY (SELECT 'x'::VARCHAR AS id, 1.0::DOUBLE AS v) TO 'w3c_prep' (FORMAT bolt, USE_TMP_FILE FALSE)",
    )?;
    let err = expect_err(&conn, "EXECUTE w3c_c");
    assert!(err.contains("changed after the statement"), "{err}");
    assert_eq!(query_i64(&conn, "SELECT count(*) FROM w3c_prep_dst")?, 13);

    // A dropped table is an error too.
    exec(&conn, "SELECT bolt_drop_table('w3c_prep')")?;
    let err = expect_err(&conn, "EXECUTE w3c_c");
    assert!(err.contains("w3c_prep"), "{err}");
    exec(&conn, "DEALLOCATE w3c_c")?;
    Ok(())
}

#[test]
fn test_w3c_copy_table_option_is_a_name_not_a_path() -> Result<()> {
    let _guard = lock();
    let (_env, conn) = setup_connection()?;

    // TABLE 'tmp_sales' registers exactly `tmp_sales` (no tmp_ / directory / .bolt cleaning).
    exec(
        &conn,
        "COPY (SELECT 1::INTEGER AS a) TO 'ignored_target' (FORMAT bolt, TABLE 'tmp_w3c_sales', USE_TMP_FILE FALSE)",
    )?;
    assert!(table_exists("tmp_w3c_sales"));
    assert!(
        BoltRuntime::global()
            .table_info_exact("w3c_sales")?
            .is_none()
    );
    assert_eq!(
        query_i64(&conn, "SELECT count(*) FROM bolt_scan('tmp_w3c_sales')")?,
        1
    );
    exec(
        &conn,
        "COPY (SELECT 2::INTEGER AS a) TO 'ignored_target' (FORMAT bolt, TABLE 'w3c_t.bolt', USE_TMP_FILE FALSE)",
    )?;
    assert!(
        BoltRuntime::global()
            .table_info_exact("w3c_t.bolt")?
            .is_some()
    );
    // The `bolt.` prefix and quotes are still accepted.
    exec(
        &conn,
        "COPY (SELECT 3::INTEGER AS a) TO 'ignored_target' (FORMAT bolt, TABLE 'bolt.w3c_pref', USE_TMP_FILE FALSE)",
    )?;
    assert!(
        BoltRuntime::global()
            .table_info_exact("w3c_pref")?
            .is_some()
    );

    // COPY FROM resolves the option the same way: `tmp_w3c_sales`, never `w3c_sales`.
    exec(&conn, "CREATE TABLE w3c_sales_back (a INTEGER)")?;
    exec(
        &conn,
        "COPY w3c_sales_back FROM 'ignored_source' (FORMAT bolt, TABLE 'tmp_w3c_sales')",
    )?;
    assert_eq!(
        query_i64(&conn, "SELECT sum(a)::BIGINT FROM w3c_sales_back")?,
        1
    );
    // An option naming a missing table is an error even if its path-cleaned form exists.
    exec(
        &conn,
        "COPY (SELECT 4::INTEGER AS a) TO 'w3c_plain' (FORMAT bolt, USE_TMP_FILE FALSE)",
    )?;
    let err = expect_err(
        &conn,
        "COPY w3c_sales_back FROM 'ignored_source' (FORMAT bolt, TABLE 'tmp_w3c_plain')",
    );
    assert!(err.contains("not registered"), "{err}");
    // The source *path* is still path-cleaned.
    exec(
        &conn,
        "COPY w3c_sales_back FROM 'some/dir/w3c_plain.bolt' (FORMAT bolt)",
    )?;
    assert_eq!(
        query_i64(&conn, "SELECT sum(a)::BIGINT FROM w3c_sales_back")?,
        5
    );
    Ok(())
}

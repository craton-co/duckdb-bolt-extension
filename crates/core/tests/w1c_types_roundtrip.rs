// SPDX-License-Identifier: Apache-2.0

//! W1-C regression tests: `COPY ... TO (FORMAT bolt)` followed by `COPY ... FROM (FORMAT bolt)`
//! into the **original** (narrow) DuckDB types.
//!
//! Craton Bolt storage widens narrow types (`SMALLINT -> Int32`, `UINTEGER -> Int64`,
//! `TIMESTAMPTZ -> Timestamp(us)`, ...). `COPY FROM` must convert back to the exact destination
//! types with checked casts. These tests only use the shadow catalog paths (`COPY TO`/`COPY
//! FROM`, `bolt_scan`), so they are valid for both GPU and `cuda-stub` builds.

use std::sync::{Mutex, MutexGuard, Once};

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
    // A failing test must not poison the remaining ones.
    TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner())
}

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

fn query_single_i64(conn: &Connection, sql: &str) -> Result<i64> {
    let mut result = conn.query(sql, Parameters::None)?;
    let chunk = result.next_chunk()?.expect("Expected at least one chunk");
    let vec = chunk.get_vector_at::<i64>(0)?;
    Ok(*vec.reader()?.get(0)?.expect("Expected non-null i64"))
}

/// Assert that `a` and `b` contain the same set of rows (in both directions) and the same count.
fn assert_same_rows(conn: &Connection, a: &str, b: &str) -> Result<()> {
    let a_minus_b = query_single_i64(
        conn,
        &format!("SELECT count(*) FROM (SELECT * FROM {a} EXCEPT SELECT * FROM {b})"),
    )?;
    let b_minus_a = query_single_i64(
        conn,
        &format!("SELECT count(*) FROM (SELECT * FROM {b} EXCEPT SELECT * FROM {a})"),
    )?;
    assert_eq!(a_minus_b, 0, "rows in {a} missing from {b}");
    assert_eq!(b_minus_a, 0, "rows in {b} missing from {a}");
    assert_eq!(
        query_single_i64(conn, &format!("SELECT count(*) FROM {a}"))?,
        query_single_i64(conn, &format!("SELECT count(*) FROM {b}"))?,
        "row counts of {a} and {b} differ"
    );
    Ok(())
}

const ALL_TYPES_DDL: &str = "(
    c_i8 TINYINT,
    c_i16 SMALLINT,
    c_u8 UTINYINT,
    c_u16 USMALLINT,
    c_u32 UINTEGER,
    c_u64 UBIGINT,
    c_i128 HUGEINT,
    c_tstz TIMESTAMPTZ,
    c_dec4 DECIMAL(4, 1),
    c_dec9 DECIMAL(9, 2),
    c_dec18 DECIMAL(18, 4),
    c_dec38 DECIMAL(38, 6),
    c_date DATE,
    c_ts_s TIMESTAMP_S,
    c_ts_ms TIMESTAMP_MS,
    c_ts_us TIMESTAMP,
    c_ts_ns TIMESTAMP_NS,
    c_str VARCHAR,
    c_bool BOOLEAN
)";

#[test]
fn test_w1c_narrow_types_copy_roundtrip() -> Result<()> {
    let _guard = lock();
    let (_env, conn) = setup_connection()?;

    conn.execute(
        format!("CREATE TABLE w1c_src {ALL_TYPES_DDL}"),
        Parameters::None,
    )?;
    conn.execute(
        "INSERT INTO w1c_src VALUES
            (-128, -32768, 0, 0, 0, 0, -9223372036854775808,
             TIMESTAMPTZ '1970-01-01 00:00:00+00',
             -999.9, -9999999.99, -99999999999999.9999, -123456789012345678901234.567890,
             DATE '1900-01-01', TIMESTAMP_S '1900-01-01 00:00:00', TIMESTAMP_MS '1900-01-01 00:00:00.001',
             TIMESTAMP '1900-01-01 00:00:00.000001', TIMESTAMP_NS '1970-01-01 00:00:00.000000001',
             '', false),
            (127, 32767, 255, 65535, 4294967295, 9223372036854775807, 9223372036854775807,
             TIMESTAMPTZ '2026-03-01 01:02:03.456789+00',
             999.9, 9999999.99, 99999999999999.9999, 123456789012345678901234.567890,
             DATE '2026-03-01', TIMESTAMP_S '2026-03-01 01:02:03', TIMESTAMP_MS '2026-03-01 01:02:03.456',
             TIMESTAMP '2026-03-01 01:02:03.456789', TIMESTAMP_NS '2026-03-01 01:02:03.456789123',
             'alpha', true),
            (1, 2, 3, 4, 5, 6, 7,
             TIMESTAMPTZ '2001-02-03 04:05:06+05',
             9.9, 123.45, 1000.1234, 999999.123456,
             DATE '2001-02-03', TIMESTAMP_S '2001-02-03 04:05:06', TIMESTAMP_MS '2001-02-03 04:05:06.007',
             TIMESTAMP '2001-02-03 04:05:06.000007', TIMESTAMP_NS '2001-02-03 04:05:06.000000007',
             'beta with spaces and a longer string that is not inlined', NULL),
            (NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
             NULL, NULL, NULL, NULL, NULL, NULL, NULL)",
        Parameters::None,
    )?;

    conn.execute(
        "COPY w1c_src TO 'w1c_types' (FORMAT bolt, USE_TMP_FILE FALSE)",
        Parameters::None,
    )?;

    // Copy back into the exact original definition (previously failed with
    // "Vector logical type mismatch: expected INTEGER, got SMALLINT").
    conn.execute(
        format!("CREATE TABLE w1c_back {ALL_TYPES_DDL}"),
        Parameters::None,
    )?;
    conn.execute(
        "COPY w1c_back FROM 'w1c_types' (FORMAT bolt)",
        Parameters::None,
    )?;
    assert_same_rows(&conn, "w1c_src", "w1c_back")?;

    // NULLs survive in every column.
    assert_eq!(
        query_single_i64(
            &conn,
            "SELECT count(*) FROM w1c_back WHERE c_i8 IS NULL AND c_i128 IS NULL AND c_tstz IS NULL
               AND c_dec38 IS NULL AND c_ts_ns IS NULL AND c_str IS NULL AND c_bool IS NULL"
        )?,
        1
    );

    // A second, independent round trip of a subset of columns into a differently named table.
    conn.execute(
        "CREATE TABLE w1c_subset_src AS SELECT c_i16, c_u32, c_tstz, c_dec4 FROM w1c_src",
        Parameters::None,
    )?;
    conn.execute(
        "COPY w1c_subset_src TO 'w1c_subset' (FORMAT bolt, USE_TMP_FILE FALSE)",
        Parameters::None,
    )?;
    conn.execute(
        "CREATE TABLE w1c_subset_back (a SMALLINT, b UINTEGER, c TIMESTAMPTZ, d DECIMAL(4, 1))",
        Parameters::None,
    )?;
    conn.execute(
        "COPY w1c_subset_back FROM 'w1c_subset' (FORMAT bolt)",
        Parameters::None,
    )?;
    assert_same_rows(&conn, "w1c_subset_src", "w1c_subset_back")?;

    Ok(())
}

#[test]
fn test_w1c_copy_from_out_of_range_and_unsupported_errors() -> Result<()> {
    let _guard = lock();
    let (_env, conn) = setup_connection()?;

    // 70000 is stored as Int32 in Bolt but does not fit SMALLINT.
    conn.execute(
        "COPY (SELECT 70000::INTEGER AS v UNION ALL SELECT 1) TO 'w1c_oor' (FORMAT bolt, USE_TMP_FILE FALSE)",
        Parameters::None,
    )?;
    conn.execute("CREATE TABLE w1c_small (v SMALLINT)", Parameters::None)?;
    assert!(
        conn.execute(
            "COPY w1c_small FROM 'w1c_oor' (FORMAT bolt)",
            Parameters::None
        )
        .is_err(),
        "70000 must not be silently wrapped or NULLed into SMALLINT"
    );
    assert_eq!(
        query_single_i64(&conn, "SELECT count(*) FROM w1c_small")?,
        0
    );

    // 300 into TINYINT / UTINYINT.
    conn.execute(
        "COPY (SELECT 300::INTEGER AS v) TO 'w1c_oor_tiny' (FORMAT bolt, USE_TMP_FILE FALSE)",
        Parameters::None,
    )?;
    conn.execute("CREATE TABLE w1c_tiny (v TINYINT)", Parameters::None)?;
    assert!(
        conn.execute(
            "COPY w1c_tiny FROM 'w1c_oor_tiny' (FORMAT bolt)",
            Parameters::None
        )
        .is_err()
    );
    conn.execute("CREATE TABLE w1c_utiny (v UTINYINT)", Parameters::None)?;
    assert!(
        conn.execute(
            "COPY w1c_utiny FROM 'w1c_oor_tiny' (FORMAT bolt)",
            Parameters::None
        )
        .is_err()
    );

    // Negative values into unsigned destinations.
    conn.execute(
        "COPY (SELECT -1::BIGINT AS v) TO 'w1c_neg' (FORMAT bolt, USE_TMP_FILE FALSE)",
        Parameters::None,
    )?;
    conn.execute("CREATE TABLE w1c_ubig (v UBIGINT)", Parameters::None)?;
    assert!(
        conn.execute(
            "COPY w1c_ubig FROM 'w1c_neg' (FORMAT bolt)",
            Parameters::None
        )
        .is_err()
    );
    conn.execute("CREATE TABLE w1c_uint (v UINTEGER)", Parameters::None)?;
    assert!(
        conn.execute(
            "COPY w1c_uint FROM 'w1c_neg' (FORMAT bolt)",
            Parameters::None
        )
        .is_err()
    );

    // A decimal that does not fit the destination precision.
    conn.execute(
        "COPY (SELECT 12345.6789::DECIMAL(18, 4) AS v) TO 'w1c_dec_oor' (FORMAT bolt, USE_TMP_FILE FALSE)",
        Parameters::None,
    )?;
    conn.execute("CREATE TABLE w1c_dec4 (v DECIMAL(4, 1))", Parameters::None)?;
    assert!(
        conn.execute(
            "COPY w1c_dec4 FROM 'w1c_dec_oor' (FORMAT bolt)",
            Parameters::None
        )
        .is_err()
    );

    // Values that do fit still copy (a wider destination is fine as well).
    conn.execute("CREATE TABLE w1c_wide (v HUGEINT)", Parameters::None)?;
    conn.execute(
        "COPY w1c_wide FROM 'w1c_neg' (FORMAT bolt)",
        Parameters::None,
    )?;
    assert_eq!(
        query_single_i64(&conn, "SELECT v::BIGINT FROM w1c_wide")?,
        -1
    );

    // Destination types Craton Bolt cannot represent produce a clear error.
    conn.execute("CREATE TABLE w1c_blob (v BLOB)", Parameters::None)?;
    assert!(
        conn.execute(
            "COPY w1c_blob FROM 'w1c_neg' (FORMAT bolt)",
            Parameters::None
        )
        .is_err()
    );

    // Column count mismatch is still rejected.
    conn.execute(
        "CREATE TABLE w1c_two (a BIGINT, b BIGINT)",
        Parameters::None,
    )?;
    assert!(
        conn.execute(
            "COPY w1c_two FROM 'w1c_neg' (FORMAT bolt)",
            Parameters::None
        )
        .is_err()
    );

    // COPY FROM takes no options.
    assert!(
        conn.execute(
            "COPY w1c_wide FROM 'w1c_neg' (FORMAT bolt, MODE 'append')",
            Parameters::None
        )
        .is_err()
    );

    // TABLE and TABLE_NAME are mutually exclusive in COPY TO.
    assert!(
        conn.execute(
            "COPY (SELECT 1 AS a) TO 'w1c_x' (FORMAT bolt, TABLE 'w1c_a', TABLE_NAME 'w1c_b', USE_TMP_FILE FALSE)",
            Parameters::None
        )
        .is_err()
    );

    Ok(())
}

#[test]
fn test_w1c_multi_batch_roundtrip_preserves_rows_and_order() -> Result<()> {
    let _guard = lock();
    let (_env, conn) = setup_connection()?;

    let select = |lo: i64, hi: i64| {
        format!(
            "SELECT
                i::SMALLINT AS s,
                (i * 3)::UINTEGER AS u,
                CASE WHEN i % 5 = 0 THEN NULL ELSE (i % 100)::TINYINT END AS t,
                ('v' || i::VARCHAR) AS str,
                CASE WHEN i % 7 = 0 THEN NULL ELSE (i::DECIMAL(18, 4) + 0.1234)::DECIMAL(18, 4) END AS d,
                to_timestamp(1577836800 + i) AS tz
             FROM range({lo}, {hi}) r(i)"
        )
    };

    // Three stored batches (> 2048 rows each for the first two) via MODE 'append'.
    conn.execute(
        format!(
            "COPY ({}) TO 'w1c_multi' (FORMAT bolt, MODE 'replace', USE_TMP_FILE FALSE)",
            select(0, 3000)
        ),
        Parameters::None,
    )?;
    conn.execute(
        format!(
            "COPY ({}) TO 'w1c_multi' (FORMAT bolt, MODE 'append', USE_TMP_FILE FALSE)",
            select(3000, 6000)
        ),
        Parameters::None,
    )?;
    conn.execute(
        format!(
            "COPY ({}) TO 'w1c_multi' (FORMAT bolt, MODE 'append', USE_TMP_FILE FALSE)",
            select(6000, 6100)
        ),
        Parameters::None,
    )?;

    conn.execute(
        format!("CREATE TABLE w1c_multi_src AS {}", select(0, 6100)),
        Parameters::None,
    )?;
    conn.execute(
        "CREATE TABLE w1c_multi_back (s SMALLINT, u UINTEGER, t TINYINT, str VARCHAR, d DECIMAL(18, 4), tz TIMESTAMPTZ)",
        Parameters::None,
    )?;
    conn.execute(
        "COPY w1c_multi_back FROM 'w1c_multi' (FORMAT bolt)",
        Parameters::None,
    )?;
    assert_eq!(
        query_single_i64(&conn, "SELECT count(*) FROM w1c_multi_back")?,
        6100
    );
    assert_same_rows(&conn, "w1c_multi_src", "w1c_multi_back")?;

    // COPY FROM runs single-threaded over the stored batches in order, so insertion order matches.
    assert_eq!(
        query_single_i64(
            &conn,
            "SELECT count(*) FROM w1c_multi_back WHERE rowid <> s::BIGINT"
        )?,
        0
    );

    // A projection-free scan (DuckDB may only request the row-id column) still counts all rows.
    assert_eq!(
        query_single_i64(&conn, "SELECT count(*) FROM bolt_scan('w1c_multi')")?,
        6100
    );

    // Copying into a table twice appends (exec must keep returning empty chunks after the end).
    conn.execute(
        "COPY w1c_multi_back FROM 'w1c_multi' (FORMAT bolt)",
        Parameters::None,
    )?;
    assert_eq!(
        query_single_i64(&conn, "SELECT count(*) FROM w1c_multi_back")?,
        12200
    );

    Ok(())
}

#[test]
fn test_w1c_empty_table_roundtrip_narrow_types() -> Result<()> {
    let _guard = lock();
    let (_env, conn) = setup_connection()?;

    conn.execute(
        "COPY (SELECT 1::SMALLINT AS a, 2::UTINYINT AS b, TIMESTAMPTZ '2020-01-01 00:00:00+00' AS c WHERE 1 = 0)
         TO 'w1c_empty' (FORMAT bolt, USE_TMP_FILE FALSE, RETURN_STATS)",
        Parameters::None,
    )?;
    conn.execute(
        "CREATE TABLE w1c_empty_back (a SMALLINT, b UTINYINT, c TIMESTAMPTZ)",
        Parameters::None,
    )?;
    conn.execute(
        "COPY w1c_empty_back FROM 'w1c_empty' (FORMAT bolt)",
        Parameters::None,
    )?;
    assert_eq!(
        query_single_i64(&conn, "SELECT count(*) FROM w1c_empty_back")?,
        0
    );
    Ok(())
}

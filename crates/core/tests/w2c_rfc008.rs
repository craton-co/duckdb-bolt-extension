// SPDX-License-Identifier: Apache-2.0

//! W2-C regression tests: RFC-008 write side (original DuckDB types recorded in Arrow field
//! metadata by `COPY ... TO (FORMAT bolt)`), the RFC-008 read helpers, multi-batch COPY TO,
//! `COPY ... FROM` column matching / options, and COPY `MODE` handling.
//!
//! The read side (`bolt_scan` / `bolt.<t>` exposing the original types) is not wired yet, so
//! these tests do not assert `typeof(...)` through `bolt_scan`. They use only the shadow catalog
//! and COPY paths and are valid for both GPU and `cuda-stub` builds.

use std::sync::{Mutex, MutexGuard, Once};

use arrow_array::{Array, TimestampMicrosecondArray};
use arrow_schema::{DataType, Field, TimeUnit};
use duckdb_bolt_core::{
    arrow_bridge::{
        DUCKDB_TYPE_METADATA_KEY, arrow_field_to_duckdb_type, arrow_type_to_duckdb_type,
        duckdb_type_to_exact_arrow_type, original_duckdb_type, restore_original_types,
    },
    runtime::BoltRuntime,
};
use duckdb_loadable_macros::duckdb_entrypoint_c_api_v2;
use duckdb_neo::{
    Parameters, Result,
    connection::Connection,
    environment::{Environment, StorageLocation},
    extension::{Extension, register_static_extension},
    logical_type::{LogicalType, LogicalTypeID},
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
    Ok(vec.reader()?.get(0)?.expect("non-null").to_string())
}

/// Run `sql`, require it to fail, and return the error message.
fn expect_err(conn: &Connection, sql: &str) -> String {
    match conn.execute(sql, Parameters::None) {
        Ok(_) => panic!("expected an error from: {sql}"),
        Err(e) => e.message,
    }
}

fn assert_same_rows(conn: &Connection, a: &str, b: &str) -> Result<()> {
    for (x, y) in [(a, b), (b, a)] {
        assert_eq!(
            query_i64(
                conn,
                &format!("SELECT count(*) FROM (SELECT * FROM {x} EXCEPT ALL SELECT * FROM {y})"),
            )?,
            0,
            "rows in {x} missing from {y}"
        );
    }
    Ok(())
}

/// DuckDB's own rendering of a SQL type (so comparisons do not depend on spelling).
fn type_text(conn: &Connection, sql_type: &str) -> Result<String> {
    LogicalType::from_text(conn, sql_type)?.to_text()
}

/// `(column, SQL type, expected metadata text)`.
const TYPED_COLUMNS: &[(&str, &str, &str)] = &[
    ("c_i8", "TINYINT", "TINYINT"),
    ("c_i16", "SMALLINT", "SMALLINT"),
    ("c_i32", "INTEGER", "INTEGER"),
    ("c_i64", "BIGINT", "BIGINT"),
    ("c_u8", "UTINYINT", "UTINYINT"),
    ("c_u16", "USMALLINT", "USMALLINT"),
    ("c_u32", "UINTEGER", "UINTEGER"),
    ("c_u64", "UBIGINT", "UBIGINT"),
    ("c_i128", "HUGEINT", "HUGEINT"),
    ("c_f32", "FLOAT", "FLOAT"),
    ("c_f64", "DOUBLE", "DOUBLE"),
    ("c_str", "VARCHAR", "VARCHAR"),
    ("c_bool", "BOOLEAN", "BOOLEAN"),
    ("c_date", "DATE", "DATE"),
    ("c_ts_s", "TIMESTAMP_S", "TIMESTAMP_S"),
    ("c_ts_ms", "TIMESTAMP_MS", "TIMESTAMP_MS"),
    ("c_ts", "TIMESTAMP", "TIMESTAMP"),
    ("c_ts_ns", "TIMESTAMP_NS", "TIMESTAMP_NS"),
    ("c_tstz", "TIMESTAMPTZ", "TIMESTAMPTZ"),
    ("c_dec4", "DECIMAL(4, 1)", "DECIMAL(4,1)"),
    ("c_dec9", "DECIMAL(9, 2)", "DECIMAL(9,2)"),
    ("c_dec18", "DECIMAL(18, 4)", "DECIMAL(18,4)"),
    ("c_dec38", "DECIMAL(38, 6)", "DECIMAL(38,6)"),
];

#[test]
fn test_w2c_copy_to_records_original_types_in_field_metadata() -> Result<()> {
    let _guard = lock();
    let (_env, conn) = setup_connection()?;

    let ddl = TYPED_COLUMNS
        .iter()
        .map(|(name, ty, _)| format!("{name} {ty}"))
        .collect::<Vec<_>>()
        .join(", ");
    exec(&conn, &format!("CREATE TABLE w2c_src ({ddl})"))?;
    exec(
        &conn,
        "INSERT INTO w2c_src VALUES
            (-128, -32768, -2147483648, -9223372036854775808, 0, 0, 0, 0, -9223372036854775808,
             -1.5, -2.25, '', false, DATE '1900-01-01', TIMESTAMP_S '1900-01-01 00:00:00',
             TIMESTAMP_MS '1900-01-01 00:00:00.001', TIMESTAMP '1900-01-01 00:00:00.000001',
             TIMESTAMP_NS '1970-01-01 00:00:00.000000001', TIMESTAMPTZ '1970-01-01 00:00:00+00',
             -999.9, -9999999.99, -99999999999999.9999, -123456789012345678901234.567890),
            (127, 32767, 2147483647, 9223372036854775807, 255, 65535, 4294967295,
             9223372036854775807, 9223372036854775807, 1.5, 2.25, 'alpha', true,
             DATE '2026-03-01', TIMESTAMP_S '2026-03-01 01:02:03',
             TIMESTAMP_MS '2026-03-01 01:02:03.456', TIMESTAMP '2026-03-01 01:02:03.456789',
             TIMESTAMP_NS '2026-03-01 01:02:03.456789123',
             TIMESTAMPTZ '2026-03-01 01:02:03.456789+00',
             999.9, 9999999.99, 99999999999999.9999, 123456789012345678901234.567890),
            (NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
             NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL)",
    )?;
    exec(
        &conn,
        "COPY w2c_src TO 'w2c_meta' (FORMAT bolt, USE_TMP_FILE FALSE)",
    )?;

    let (schema, batches) = BoltRuntime::global().get_table_batches("w2c_meta")?;
    assert_eq!(schema.fields().len(), TYPED_COLUMNS.len());
    for (field, (name, sql_type, meta)) in schema.fields().iter().zip(TYPED_COLUMNS) {
        assert_eq!(field.name(), name);
        // The runtime catalog must keep the field metadata written by COPY TO.
        assert_eq!(
            field
                .metadata()
                .get(DUCKDB_TYPE_METADATA_KEY)
                .map(String::as_str),
            Some(*meta),
            "metadata of {name} (stored as {:?})",
            field.data_type()
        );
        assert!(original_duckdb_type(field).is_some(), "{name}");
        // The read-side helper resolves the original type.
        assert_eq!(
            arrow_field_to_duckdb_type(&conn, field)?.to_text()?,
            type_text(&conn, sql_type)?,
            "{name}"
        );
    }

    // Restoring the stored batches yields the exact Arrow types of the original columns.
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 3);
    for batch in &batches {
        // Every stored batch carries the full table schema (metadata included).
        assert_eq!(batch.schema(), schema);
        let restored = restore_original_types(batch)?;
        for (field, (name, sql_type, _)) in restored.schema().fields().iter().zip(TYPED_COLUMNS) {
            let expected =
                duckdb_type_to_exact_arrow_type(&LogicalType::from_text(&conn, sql_type)?)?;
            assert_eq!(field.data_type(), &expected, "{name}");
        }
        let tz_idx = TYPED_COLUMNS
            .iter()
            .position(|(n, _, _)| *n == "c_tstz")
            .unwrap();
        let tz = restored
            .column(tz_idx)
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .expect("TIMESTAMPTZ restores to Timestamp(us, UTC)");
        assert_eq!(tz.timezone(), Some("UTC"));
    }
    Ok(())
}

#[test]
fn test_w2c_arrow_field_type_fallbacks_and_dictionaries() -> Result<()> {
    let _guard = lock();
    let (_env, conn) = setup_connection()?;

    let id = |field: Field| arrow_field_to_duckdb_type(&conn, &field).and_then(|t| t.id());

    // Without metadata: storage mapping, except tz-annotated timestamps are instants.
    assert!(matches!(
        id(Field::new("a", DataType::Int32, true))?,
        LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_INTEGER
    ));
    assert!(matches!(
        id(Field::new(
            "a",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            true
        ))?,
        LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_TIMESTAMP
    ));
    assert!(matches!(
        id(Field::new(
            "a",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            true
        ))?,
        LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_TIMESTAMP_TZ
    ));
    assert!(matches!(
        id(Field::new(
            "a",
            DataType::Timestamp(TimeUnit::Millisecond, Some("+02:00".into())),
            true
        ))?,
        LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_TIMESTAMP_TZ
    ));
    assert!(matches!(
        id(Field::new(
            "a",
            DataType::Timestamp(TimeUnit::Nanosecond, Some("UTC".into())),
            true
        ))?,
        LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_TIMESTAMP_TZ_NS
    ));

    // Metadata wins; stale metadata (wrong storage type) is ignored.
    let with_meta = |dt: DataType, text: &str| {
        Field::new("a", dt, true).with_metadata(
            [(DUCKDB_TYPE_METADATA_KEY.to_string(), text.to_string())]
                .into_iter()
                .collect(),
        )
    };
    assert!(matches!(
        id(with_meta(DataType::Int32, "SMALLINT"))?,
        LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_SMALLINT
    ));
    assert!(matches!(
        id(with_meta(
            DataType::Timestamp(TimeUnit::Microsecond, None),
            "TIMESTAMPTZ"
        ))?,
        LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_TIMESTAMP_TZ
    ));
    assert_eq!(
        arrow_field_to_duckdb_type(
            &conn,
            &with_meta(DataType::Decimal128(9, 2), "DECIMAL(9,2)")
        )?
        .to_text()?,
        type_text(&conn, "DECIMAL(9, 2)")?
    );
    assert!(matches!(
        id(with_meta(DataType::Utf8, "SMALLINT"))?,
        LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_VARCHAR
    ));

    // W1-E-014: a dictionary has the logical type of its values, not VARCHAR.
    let dict = |v: DataType| DataType::Dictionary(Box::new(DataType::Int32), Box::new(v));
    assert!(matches!(
        arrow_type_to_duckdb_type(&conn, &dict(DataType::Int64))?.id()?,
        LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_BIGINT
    ));
    assert!(matches!(
        arrow_type_to_duckdb_type(&conn, &dict(DataType::Float64))?.id()?,
        LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_DOUBLE
    ));
    assert!(matches!(
        arrow_type_to_duckdb_type(
            &conn,
            &DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8))
        )?
        .id()?,
        LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_VARCHAR
    ));
    assert!(arrow_type_to_duckdb_type(&conn, &dict(DataType::Binary)).is_err());
    assert!(matches!(
        id(Field::new("a", dict(DataType::Int16), true))?,
        LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_SMALLINT
    ));
    Ok(())
}

#[test]
fn test_w2c_multi_batch_copy_to_registers_without_concatenation() -> Result<()> {
    let _guard = lock();
    let (_env, conn) = setup_connection()?;

    exec(
        &conn,
        "COPY (SELECT (i % 1000)::SMALLINT AS s, 'v' || i::VARCHAR AS str FROM range(300000) t(i))
         TO 'w2c_big' (FORMAT bolt, USE_TMP_FILE FALSE)",
    )?;
    let (schema, batches) = BoltRuntime::global().get_table_batches("w2c_big")?;
    assert_eq!(batches.iter().map(|b| b.num_rows()).sum::<usize>(), 300_000);
    for batch in &batches {
        assert!(batch.num_rows() > 0, "empty batches are not stored");
        assert_eq!(batch.schema(), schema);
    }
    assert_eq!(
        query_i64(&conn, "SELECT count(*) FROM bolt_scan('w2c_big')")?,
        300_000
    );

    // Copy back into the original types.
    exec(&conn, "CREATE TABLE w2c_big_back (s SMALLINT, str VARCHAR)")?;
    exec(&conn, "COPY w2c_big_back FROM 'w2c_big' (FORMAT bolt)")?;
    exec(
        &conn,
        "CREATE TABLE w2c_big_src AS SELECT (i % 1000)::SMALLINT AS s, 'v' || i::VARCHAR AS str FROM range(300000) t(i)",
    )?;
    assert_same_rows(&conn, "w2c_big_src", "w2c_big_back")?;

    // An empty COPY registers an empty table that still records its original types.
    exec(
        &conn,
        "COPY (SELECT 1::UTINYINT AS u WHERE 1 = 0) TO 'w2c_empty' (FORMAT bolt, USE_TMP_FILE FALSE)",
    )?;
    let (schema, batches) = BoltRuntime::global().get_table_batches("w2c_empty")?;
    assert_eq!(batches.iter().map(|b| b.num_rows()).sum::<usize>(), 0);
    assert_eq!(
        schema
            .field(0)
            .metadata()
            .get(DUCKDB_TYPE_METADATA_KEY)
            .map(String::as_str),
        Some("UTINYINT")
    );
    Ok(())
}

#[test]
fn test_w2c_timestamptz_round_trip_in_non_utc_session() -> Result<()> {
    let _guard = lock();
    let (_env, conn) = setup_connection()?;

    exec(&conn, "SET TimeZone = 'America/Argentina/Buenos_Aires'")?;
    let result = (|| -> Result<()> {
        exec(
            &conn,
            "CREATE TABLE w2c_tz_src AS
             SELECT i::INTEGER AS id,
                    TIMESTAMPTZ '2024-01-01 00:00:00+00' + INTERVAL (i * 7) MINUTE AS t
             FROM range(0, 500) r(i)",
        )?;
        exec(
            &conn,
            "COPY w2c_tz_src TO 'w2c_tz' (FORMAT bolt, USE_TMP_FILE FALSE)",
        )?;
        exec(
            &conn,
            "CREATE TABLE w2c_tz_back (id INTEGER, t TIMESTAMPTZ)",
        )?;
        exec(&conn, "COPY w2c_tz_back FROM 'w2c_tz' (FORMAT bolt)")?;
        assert_same_rows(&conn, "w2c_tz_src", "w2c_tz_back")?;
        // Same instants, independent of the session time zone.
        assert_eq!(
            query_i64(
                &conn,
                "SELECT count(*) FROM w2c_tz_src s JOIN w2c_tz_back b USING (id)
                 WHERE epoch_us(s.t) <> epoch_us(b.t)"
            )?,
            0
        );
        assert_eq!(
            query_i64(
                &conn,
                "SELECT count(*) FROM w2c_tz_back WHERE t = TIMESTAMPTZ '2024-01-01 00:07:00+00'"
            )?,
            1
        );
        Ok(())
    })();
    exec(&conn, "RESET TimeZone")?;
    result
}

#[test]
fn test_w2c_copy_from_matches_columns_by_name() -> Result<()> {
    let _guard = lock();
    let (_env, conn) = setup_connection()?;

    exec(
        &conn,
        "COPY (SELECT 1 AS a, 'x' AS b) TO 'w2c_ab' (FORMAT bolt, USE_TMP_FILE FALSE)",
    )?;

    // Reordered columns with different types: matched by name (previously a vector type error).
    exec(&conn, "CREATE TABLE w2c_ba (b VARCHAR, a INTEGER)")?;
    exec(&conn, "COPY w2c_ba FROM 'w2c_ab' (FORMAT bolt)")?;
    assert_eq!(query_i64(&conn, "SELECT a::BIGINT FROM w2c_ba")?, 1);
    assert_eq!(query_string(&conn, "SELECT b FROM w2c_ba")?, "x");

    // Case-insensitive.
    exec(
        &conn,
        "CREATE TABLE w2c_ba_upper (\"B\" VARCHAR, \"A\" INTEGER)",
    )?;
    exec(&conn, "COPY w2c_ba_upper FROM 'w2c_ab' (FORMAT bolt)")?;
    assert_eq!(
        query_i64(&conn, "SELECT \"A\"::BIGINT FROM w2c_ba_upper")?,
        1
    );

    // Same-typed columns are no longer silently swapped.
    exec(
        &conn,
        "COPY (SELECT 1 AS a, 2 AS b) TO 'w2c_ints' (FORMAT bolt, USE_TMP_FILE FALSE)",
    )?;
    exec(&conn, "CREATE TABLE w2c_ints_ba (b INTEGER, a INTEGER)")?;
    exec(&conn, "COPY w2c_ints_ba FROM 'w2c_ints' (FORMAT bolt)")?;
    assert_eq!(query_i64(&conn, "SELECT a::BIGINT FROM w2c_ints_ba")?, 1);
    assert_eq!(query_i64(&conn, "SELECT b::BIGINT FROM w2c_ints_ba")?, 2);

    // Different names: positional.
    exec(&conn, "CREATE TABLE w2c_xy (x INTEGER, y VARCHAR)")?;
    exec(&conn, "COPY w2c_xy FROM 'w2c_ab' (FORMAT bolt)")?;
    assert_eq!(query_i64(&conn, "SELECT x::BIGINT FROM w2c_xy")?, 1);
    assert_eq!(query_string(&conn, "SELECT y FROM w2c_xy")?, "x");

    // Different column count (even when the names exist): still rejected.
    exec(&conn, "CREATE TABLE w2c_only_a (a INTEGER)")?;
    expect_err(&conn, "COPY w2c_only_a FROM 'w2c_ab' (FORMAT bolt)");

    // A type mismatch names the column and both types.
    exec(&conn, "CREATE TABLE w2c_bad (a INTEGER, b DATE)")?;
    let msg = expect_err(&conn, "COPY w2c_bad FROM 'w2c_ab' (FORMAT bolt)");
    assert!(msg.contains("'b'"), "{msg}");
    Ok(())
}

#[test]
fn test_w2c_copy_from_options() -> Result<()> {
    let _guard = lock();
    let (_env, conn) = setup_connection()?;

    exec(
        &conn,
        "COPY (SELECT 42 AS v) TO 'w2c_src_opt' (FORMAT bolt, USE_TMP_FILE FALSE)",
    )?;
    exec(&conn, "CREATE TABLE w2c_dst (v INTEGER)")?;

    // TABLE picks the source Bolt table, symmetric to COPY TO.
    exec(
        &conn,
        "COPY w2c_dst FROM 'ignored_path' (FORMAT bolt, TABLE 'w2c_src_opt')",
    )?;
    assert_eq!(query_i64(&conn, "SELECT v::BIGINT FROM w2c_dst")?, 42);

    let msg = expect_err(
        &conn,
        "COPY w2c_dst FROM 'w2c_src_opt' (FORMAT bolt, MODE 'append')",
    );
    assert!(msg.contains("MODE"), "{msg}");
    let msg = expect_err(
        &conn,
        "COPY w2c_dst FROM 'w2c_src_opt' (FORMAT bolt, BOGUS_OPT 1)",
    );
    assert!(msg.to_ascii_lowercase().contains("bogus_opt"), "{msg}");
    expect_err(
        &conn,
        "COPY w2c_dst FROM 'x' (FORMAT bolt, TABLE 'w2c_src_opt', TABLE_NAME 'w2c_src_opt')",
    );
    assert_eq!(query_i64(&conn, "SELECT count(*) FROM w2c_dst")?, 1);
    Ok(())
}

#[test]
fn test_w2c_copy_to_mode_values_and_errors() -> Result<()> {
    let _guard = lock();
    let (_env, conn) = setup_connection()?;
    let count = |conn: &Connection| query_i64(conn, "SELECT count(*) FROM bolt_scan('w2c_mode')");

    exec(
        &conn,
        "COPY (SELECT 1 AS v) TO 'w2c_mode' (FORMAT bolt, MODE 'create', USE_TMP_FILE FALSE)",
    )?;
    // 'error' is an alias of 'create': the table exists.
    expect_err(
        &conn,
        "COPY (SELECT 1 AS v) TO 'w2c_mode' (FORMAT bolt, MODE 'error', USE_TMP_FILE FALSE)",
    );
    // Case-insensitive 'append'.
    exec(
        &conn,
        "COPY (SELECT 2 AS v) TO 'w2c_mode' (FORMAT bolt, MODE 'APPEND', USE_TMP_FILE FALSE)",
    )?;
    assert_eq!(count(&conn)?, 2);
    // 'overwrite' is an alias of 'replace'.
    exec(
        &conn,
        "COPY (SELECT 3 AS v) TO 'w2c_mode' (FORMAT bolt, MODE 'overwrite', USE_TMP_FILE FALSE)",
    )?;
    assert_eq!(count(&conn)?, 1);

    let msg = expect_err(
        &conn,
        "COPY (SELECT 1 AS v) TO 'w2c_mode' (FORMAT bolt, MODE 'bogus', USE_TMP_FILE FALSE)",
    );
    assert!(msg.contains("bogus") && msg.contains("MODE"), "{msg}");
    // NULL is rejected (by DuckDB or by the bind; the message source may vary).
    expect_err(
        &conn,
        "COPY (SELECT 1 AS v) TO 'w2c_mode' (FORMAT bolt, MODE NULL, USE_TMP_FILE FALSE)",
    );
    // Errors leave the table untouched.
    assert_eq!(count(&conn)?, 1);
    Ok(())
}

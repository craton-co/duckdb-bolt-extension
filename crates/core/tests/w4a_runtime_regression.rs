// SPDX-License-Identifier: Apache-2.0

//! Wave 4 (W4-A) runtime regression tests: `PRAGMA bolt_reset` policy (W1-B-001), telemetry
//! off the worker (W1-B-004), Arrow-stream host-fallback results (W1-C-005), the engine copy
//! shared with the catalog (W3-A-001), `bolt_tables().schema` original types (W3-O-001), the
//! canonical mode message (W1-E-007), tracing / PTX cache use-time checks (W2-E-009,
//! W3-D-004), pragma settings sync (W3-D-004), and the catalog snapshot after
//! `bolt_clear_tables` (W4-A-001). They run on both backends and branch on the reported
//! backend where behavior legitimately differs.

use std::{
    path::PathBuf,
    sync::{Arc, Mutex, Once},
    time::{Duration, Instant},
};

use arrow_array::{Array, ArrayRef, Int64Array, RecordBatch, cast::AsArray, types::Int64Type};
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
fn bolt_w4a_test_entrypoint(extension: &Extension) -> Result<()> {
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

fn query_single_i64(conn: &Connection, sql: &str) -> Result<i64> {
    let mut result = conn.query(sql, Parameters::None)?;
    let chunk = result.next_chunk()?.expect("Expected at least one chunk");
    let vec = chunk.get_vector_at::<i64>(0)?;
    Ok(*vec.reader()?.get(0)?.expect("Expected non-null i64"))
}

fn query_single_string(conn: &Connection, sql: &str) -> Result<String> {
    let mut result = conn.query(sql, Parameters::None)?;
    let chunk = result.next_chunk()?.expect("Expected at least one chunk");
    let vec = chunk.get_vector_at::<String>(0)?;
    Ok(vec
        .reader()?
        .get(0)?
        .expect("Expected non-null string")
        .to_string())
}

fn int_batch(name: &str, values: impl IntoIterator<Item = i64>) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![Field::new(name, DataType::Int64, false)]));
    let col: ArrayRef = Arc::new(Int64Array::from_iter_values(values));
    RecordBatch::try_new(schema, vec![col]).unwrap()
}

/// Address of the first value of the `Int64` column 0 of `batch`.
fn values_ptr(batch: &RecordBatch) -> usize {
    batch
        .column(0)
        .as_primitive::<Int64Type>()
        .values()
        .as_ptr() as usize
}

/// Whether consecutive `batches` are views into one contiguous `Int64` values buffer.
fn contiguous(batches: &[RecordBatch]) -> bool {
    batches
        .windows(2)
        .all(|w| values_ptr(&w[1]) == values_ptr(&w[0]) + w[0].num_rows() * 8)
}

#[test]
fn reset_reports_a_missing_device_when_gpus_are_visible() -> Result<()> {
    let _guard = lock();
    let (_env, conn) = setup_connection()?;
    let visible = query_single_i64(&conn, "SELECT count(*) FROM bolt_devices()")?;
    exec(&conn, "SET bolt_device = 999")?;
    let reset = conn.execute("PRAGMA bolt_reset", Parameters::None);
    if visible > 0 {
        let err = reset.expect_err("an ordinal that does not exist must fail the reset");
        assert!(
            err.message.contains("bolt_device = 999 does not exist")
                && err.message.contains("CUDA device(s) visible"),
            "{}",
            err.message
        );
    } else {
        // No CUDA device at all (cuda-stub): the host fallback is the only backend.
        reset?;
    }
    assert_eq!(
        query_single_string(&conn, "SELECT backend FROM bolt_pool_stats()")?,
        "host-planner-fallback"
    );
    exec(&conn, "SET bolt_device = 0")?;
    exec(&conn, "PRAGMA bolt_reset")?;
    let backend = query_single_string(&conn, "SELECT backend FROM bolt_pool_stats()")?;
    assert_eq!(backend == "cuda-gpu", visible > 0, "{backend}");
    Ok(())
}

#[test]
fn telemetry_does_not_wait_for_a_long_query() -> Result<()> {
    let _guard = lock();
    let (_env, _conn) = setup_connection()?;
    let rt = BoltRuntime::global();
    rt.register_table("w4a_slow", int_batch("v", 0..40_000), RegisterMode::Replace)?;
    // Make the backend known (the first call may need the worker).
    let before = rt.telemetry()?;
    rt.set_request_timeout(Some(Duration::from_secs(4)));
    let worker = std::thread::spawn(|| {
        // 1.6e9 pairs on the host fallback (no join key DuckDB could hash): runs until the
        // timeout interrupts it.
        BoltRuntime::global().execute_sql_host_fallback(
            "SELECT count(*) FROM w4a_slow a, w4a_slow b WHERE (a.v * 7 + b.v) % 1000003 = -5",
        )
    });
    std::thread::sleep(Duration::from_millis(300));
    let started = Instant::now();
    let during = rt.telemetry()?;
    let gpu = rt.backend_is_gpu()?;
    let waited = started.elapsed();
    let still_running = !worker.is_finished();
    let outcome = worker.join().expect("query thread");
    rt.set_request_timeout(None);
    rt.drop_table("w4a_slow")?;
    assert!(still_running, "the long query finished early: {outcome:?}");
    assert!(
        waited < Duration::from_secs(1),
        "telemetry waited {waited:?} behind the worker"
    );
    assert_eq!(during.backend, before.backend);
    assert_eq!(gpu, before.backend == "cuda-gpu");
    Ok(())
}

#[test]
fn host_fallback_results_keep_bolt_storage_types() -> Result<()> {
    let _guard = lock();
    let (_env, _conn) = setup_connection()?;
    let rt = BoltRuntime::global();
    let out = rt.execute_sql_host_fallback(
        "SELECT 1::SMALLINT AS a, 2::HUGEINT AS h, 3::UINTEGER AS u, 'x' AS s, \
         TIMESTAMPTZ '2020-01-01 00:00:00+00' AS t, 1.5::DECIMAL(4,1) AS d, \
         DATE '2020-01-02' AS dt, true AS b, 2.5::FLOAT AS f",
    )?;
    let types: Vec<DataType> = out
        .batch
        .schema()
        .fields()
        .iter()
        .map(|f| f.data_type().clone())
        .collect();
    assert_eq!(
        types,
        vec![
            DataType::Int32,
            DataType::Int64,
            DataType::Int64,
            DataType::Utf8,
            DataType::Timestamp(TimeUnit::Microsecond, None),
            DataType::Decimal128(4, 1),
            DataType::Date32,
            DataType::Boolean,
            DataType::Float32,
        ]
    );
    assert_eq!(out.execution_tier, "Host");
    let ts = out
        .batch
        .column(4)
        .as_primitive::<arrow_array::types::TimestampMicrosecondType>();
    assert_eq!(ts.value(0), 1_577_836_800_000_000, "naive UTC microseconds");

    // Many DuckDB chunks, with NULLs.
    let out = rt.execute_sql_host_fallback(
        "SELECT CASE WHEN range % 3 = 0 THEN NULL ELSE range END AS v FROM range(10000)",
    )?;
    assert_eq!(out.batch.num_rows(), 10_000);
    assert_eq!(out.batch.column(0).null_count(), 3_334);

    // Out-of-range values are an error, never NULL or wrapped.
    let err = rt
        .execute_sql_host_fallback("SELECT 9223372036854775808::UBIGINT AS v")
        .expect_err("UBIGINT above i64::MAX");
    assert!(err.message.contains("host fallback"), "{}", err.message);
    let err = rt
        .execute_sql_host_fallback("SELECT 'a'::BLOB AS v")
        .expect_err("BLOB is not a Bolt type");
    assert!(err.message.contains("Unsupported"), "{}", err.message);
    Ok(())
}

#[test]
fn multi_batch_tables_share_the_engine_copy() -> Result<()> {
    let _guard = lock();
    let (_env, conn) = setup_connection()?;
    let rt = BoltRuntime::global();
    let gpu = rt.backend_is_gpu()?;
    let batches: Vec<RecordBatch> = (0..5)
        .map(|i| int_batch("v", i * 1000..(i + 1) * 1000))
        .collect();
    let schema = batches[0].schema();
    let summary =
        rt.register_table_batches("w4a_shared", schema.clone(), batches, RegisterMode::Replace)?;
    let (_, stored) = rt.get_table_batches_strict("w4a_shared")?;
    assert_eq!(stored.len(), 5, "batch boundaries are kept");
    assert_eq!(
        stored.iter().map(RecordBatch::num_rows).sum::<usize>(),
        5000
    );
    // The engine's concatenated copy is the catalog's storage on the GPU backend.
    assert_eq!(contiguous(&stored), gpu);
    // `bytes_in_memory` still counts the data once.
    let expected_bytes = summary.bytes_in_memory;
    assert_eq!(
        rt.table_info("w4a_shared")?
            .expect("registered")
            .bytes_in_memory,
        expected_bytes
    );

    // Appends share the coalesced copy of the appended rows.
    let more: Vec<RecordBatch> = (5..8)
        .map(|i| int_batch("v", i * 1000..(i + 1) * 1000))
        .collect();
    rt.register_table_batches("w4a_shared", schema, more, RegisterMode::Append)?;
    let (_, stored) = rt.get_table_batches_strict("w4a_shared")?;
    assert_eq!(stored.len(), 8);
    assert_eq!(contiguous(&stored[5..]), gpu);
    assert_eq!(
        query_single_i64(&conn, "SELECT sum(v)::BIGINT FROM bolt_scan('w4a_shared')")?,
        (0..8000).sum::<i64>()
    );
    assert_eq!(
        query_single_i64(
            &conn,
            "SELECT s::BIGINT FROM bolt_query('SELECT SUM(v) AS s FROM w4a_shared')"
        )?,
        (0..8000).sum::<i64>()
    );

    // A rebuild re-shares the whole table.
    rt.reset_engine()?;
    let (_, stored) = rt.get_table_batches_strict("w4a_shared")?;
    assert_eq!(contiguous(&stored), gpu);
    rt.drop_table("w4a_shared")?;
    Ok(())
}

#[test]
fn clear_tables_releases_the_snapshot_batches() -> Result<()> {
    let _guard = lock();
    let (_env, _conn) = setup_connection()?;
    let rt = BoltRuntime::global();
    let col: ArrayRef = Arc::new(Int64Array::from_iter_values(0..100));
    let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)]));
    let batch = RecordBatch::try_new(schema, vec![col.clone()]).unwrap();
    rt.register_table("w4a_clear", batch, RegisterMode::Replace)?;
    assert!(Arc::strong_count(&col) > 1);
    assert_eq!(rt.clear_tables()?, 1);
    assert_eq!(
        Arc::strong_count(&col),
        1,
        "nothing may keep the data of a cleared table alive"
    );
    assert!(rt.get_table_batches_strict("w4a_clear").is_err());
    Ok(())
}

#[test]
fn bolt_tables_schema_shows_original_duckdb_types() -> Result<()> {
    let _guard = lock();
    let (_env, conn) = setup_connection()?;
    exec(
        &conn,
        "COPY (SELECT 1::SMALLINT AS v, 'a' AS s, 2::UINTEGER AS u, 1.5::DECIMAL(4,1) AS d) \
         TO 'w4a_types' (FORMAT bolt, USE_TMP_FILE FALSE)",
    )?;
    assert_eq!(
        query_single_string(
            &conn,
            "SELECT schema FROM bolt_tables() WHERE table_name = 'w4a_types'"
        )?,
        "v: SMALLINT, s: VARCHAR, u: UINTEGER, d: DECIMAL(4,1)"
    );
    // Without RFC-008 metadata the storage type is shown.
    BoltRuntime::global().register_table(
        "w4a_plain",
        int_batch("id", 0..3),
        RegisterMode::Replace,
    )?;
    assert_eq!(
        query_single_string(
            &conn,
            "SELECT schema FROM bolt_tables() WHERE table_name = 'w4a_plain'"
        )?,
        "id: Int64"
    );
    exec(&conn, "PRAGMA bolt_clear_tables")?;
    Ok(())
}

#[test]
fn invalid_register_mode_message_is_canonical() {
    let err = RegisterMode::parse(" Bogus ").expect_err("not a mode");
    assert_eq!(
        err.message,
        "Invalid Bolt registration mode ' Bogus ': expected 'replace', 'append', or 'create' \
         (aliases: 'overwrite' = 'replace', 'error' = 'create')"
    );
    assert_eq!(err.message, RegisterMode::invalid_mode_message(" Bogus "));
    assert_eq!(
        RegisterMode::parse(" APPEND ").unwrap(),
        RegisterMode::Append
    );
}

#[test]
fn turning_tracing_off_restores_the_log_level() -> Result<()> {
    let _guard = lock();
    let (_env, _conn) = setup_connection()?;
    let rt = BoltRuntime::global();
    log::set_max_level(log::LevelFilter::Warn);
    rt.update_config(|c| c.enable_tracing = true)?;
    let gpu = rt.backend_is_gpu()?;
    if gpu {
        // craton-bolt's `enable_tracing` raises the global level.
        assert_eq!(log::max_level(), log::LevelFilter::Info);
    }
    rt.update_config(|c| c.enable_tracing = false)?;
    assert_eq!(log::max_level(), log::LevelFilter::Warn);
    // A rebuild without tracing leaves it alone.
    assert_eq!(rt.backend_is_gpu()?, gpu);
    assert_eq!(log::max_level(), log::LevelFilter::Warn);
    Ok(())
}

#[test]
fn ptx_cache_dir_failing_the_use_time_check_is_not_used() -> Result<()> {
    let _guard = lock();
    let (_env, _conn) = setup_connection()?;
    let rt = BoltRuntime::global();
    // Bypass `SET` validation, like a directory that became unsafe after it was set.
    rt.update_config(|c| c.ptx_cache_dir = Some(PathBuf::from("w4a_relative_ptx")))?;
    let gpu = rt.backend_is_gpu()?;
    let version = rt.version_info()?;
    rt.update_config(|c| c.ptx_cache_dir = None)?;
    if gpu {
        assert!(
            version.contains("bolt_ptx_cache_dir 'w4a_relative_ptx' is not used")
                && version.contains("absolute"),
            "{version}"
        );
    }
    assert!(!std::path::Path::new("w4a_relative_ptx").exists());
    Ok(())
}

#[test]
fn pragmas_refresh_a_stale_setting() -> Result<()> {
    let _guard = lock();
    let (_env_a, a) = setup_connection()?;
    let (_env_b, b) = setup_connection()?;
    exec(&a, "SET bolt_memory_budget = 0")?;
    exec(&b, "SET bolt_memory_budget = 3145728")?;
    assert_eq!(
        query_single_i64(&a, "SELECT current_setting('bolt_memory_budget')::BIGINT")?,
        0
    );
    exec(&a, "PRAGMA bolt_request_timeout(0)")?;
    assert_eq!(
        query_single_i64(&a, "SELECT current_setting('bolt_memory_budget')::BIGINT")?,
        3_145_728
    );
    exec(&b, "SET bolt_memory_budget = 0")?;
    Ok(())
}

#[test]
fn host_fallback_accepts_bolt_prefixed_names_and_strings() -> Result<()> {
    let _guard = lock();
    let (_env, conn) = setup_connection()?;
    exec(
        &conn,
        "COPY (SELECT i AS id, CASE WHEN i % 2 = 0 THEN NULL ELSE i END AS bolt_flag, \
         'https://bolt.com/' || i AS url FROM range(10) r(i)) \
         TO 'w4a_names' (FORMAT bolt, USE_TMP_FILE FALSE)",
    )?;
    // The table has NULLs, so the filtered aggregates run on the host fallback (both backends).
    assert_eq!(
        query_single_i64(
            &conn,
            "SELECT * FROM bolt_query('SELECT count(*) AS bolt_rows FROM w4a_names WHERE bolt_flag > 3')"
        )?,
        3
    );
    assert_eq!(
        query_single_i64(
            &conn,
            "SELECT * FROM bolt_query('SELECT count(*) FROM w4a_names \
             WHERE url LIKE ''%bolt.com%'' AND bolt_flag IS NOT NULL')"
        )?,
        5
    );
    // A real call is still refused, fast.
    let err = conn
        .execute(
            "SELECT * FROM bolt_query('SELECT bolt_version() FROM w4a_names WHERE bolt_flag > 3')",
            Parameters::None,
        )
        .expect_err("re-entrant call");
    assert!(
        err.message
            .contains("cannot be used inside a query that runs on the Craton Bolt host fallback"),
        "{}",
        err.message
    );
    exec(&conn, "PRAGMA bolt_clear_tables")?;
    Ok(())
}

#[test]
fn request_timeout_only_affects_the_waiting_caller() -> Result<()> {
    let _guard = lock();
    let (_env, _conn) = setup_connection()?;
    let rt = BoltRuntime::global();
    rt.register_table("w4a_busy", int_batch("v", 0..20_000), RegisterMode::Replace)?;
    rt.set_request_timeout(None);
    // Caller A: no limit, a few seconds of host-fallback work.
    let a = std::thread::spawn(|| {
        BoltRuntime::global().execute_sql_host_fallback(
            "SELECT count(*) AS n FROM w4a_busy a, w4a_busy b WHERE (a.v * 7 + b.v) % 1000003 = -5",
        )
    });
    std::thread::sleep(Duration::from_millis(300));
    rt.set_request_timeout(Some(Duration::from_millis(200)));
    // Caller B: queued behind A, gives up; its registration must have no effect.
    let b = rt.register_table("w4a_never", int_batch("v", 0..3), RegisterMode::Create);
    let version = rt.version_info();
    rt.set_request_timeout(None);
    let a = a.join().expect("caller A");
    let b_err = b.expect_err("B waits behind A and times out");
    assert!(
        b_err.message.contains("not started and had no effect"),
        "{}",
        b_err.message
    );
    assert!(version.is_err());
    // A was not interrupted on B's behalf.
    let a = a.expect("caller A must not be interrupted by B's timeout");
    assert_eq!(a.batch.num_rows(), 1);
    // B's abandoned registration never ran.
    assert!(rt.table_info("w4a_never")?.is_none());
    rt.register_table("w4a_never", int_batch("v", 0..3), RegisterMode::Create)?;
    rt.drop_table("w4a_never")?;
    rt.drop_table("w4a_busy")?;
    Ok(())
}

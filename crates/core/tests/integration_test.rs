// SPDX-License-Identifier: Apache-2.0

//! End-to-end integration tests for `duckdb-bolt-extension` against DuckDB v2.0
//! and the Craton Bolt GPU SQL engine.
//!
//! The suite runs on both backends: `cuda-gpu` (real GPU build with a usable device) and
//! `host-planner-fallback` (`cuda-stub` builds, or no usable GPU). Tests read the active backend
//! from `bolt_pool_stats()` and assert per backend only where behavior legitimately differs.

use std::sync::{Mutex, Once};

use duckdb_loadable_macros::duckdb_entrypoint_c_api_v2;
use duckdb_neo::{
    Parameters, Result,
    connection::Connection,
    environment::{Environment, StorageLocation},
    extension::{Extension, register_static_extension},
    types::{DateValue, LongDecimal, TimestampValue, WordDecimal},
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

fn query_single_i64(conn: &Connection, sql: &str) -> Result<i64> {
    let mut result = conn.query(sql, Parameters::None)?;
    let chunk = result.next_chunk()?.expect("Expected at least one chunk");
    let vec = chunk.get_vector_at::<i64>(0)?;
    Ok(*vec.reader()?.get(0)?.expect("Expected non-null i64"))
}

/// Assert that `result` is an error whose message contains `needle`. A bare `is_err()` also
/// passes on an unrelated failure (a typo in the SQL, a missing table), so negative tests name the
/// error they expect (W1-E-013).
#[track_caller]
fn assert_err_contains<T>(result: Result<T>, needle: &str) {
    match result {
        Ok(_) => panic!("expected an error containing {needle:?}, but the call succeeded"),
        Err(e) => {
            let message = e.to_string();
            assert!(
                message.contains(needle),
                "expected an error containing {needle:?}, got: {message}"
            );
        }
    }
}

/// Run `sql` and drain its first chunk, so errors raised at execution time surface too.
fn query_first_chunk(conn: &Connection, sql: &str) -> Result<()> {
    let mut result = conn.query(sql, Parameters::None)?;
    result.next_chunk().map(|_| ())
}

fn query_single_bool(conn: &Connection, sql: &str) -> Result<bool> {
    let mut result = conn.query(sql, Parameters::None)?;
    let chunk = result.next_chunk()?.expect("Expected at least one chunk");
    let vec = chunk.get_vector_at::<bool>(0)?;
    Ok(*vec.reader()?.get(0)?.expect("Expected non-null bool"))
}

/// Total number of rows produced by `sql` (all chunks).
fn count_rows(conn: &Connection, sql: &str) -> Result<usize> {
    let mut result = conn.query(sql, Parameters::None)?;
    let mut rows = 0;
    while let Some(chunk) = result.next_chunk()? {
        rows += chunk.row_count()?;
    }
    Ok(rows)
}

const BACKEND_GPU: &str = "cuda-gpu";
const BACKEND_FALLBACK: &str = "host-planner-fallback";

/// Active execution backend as reported by `bolt_pool_stats()`.
///
/// Tests assert per backend where behavior legitimately differs: `cuda-gpu` runs queries on the
/// GPU engine and accepts filter pushdown; `host-planner-fallback` (no GPU, or a `cuda-stub`
/// build) plans with Craton Bolt and executes on an embedded DuckDB instance.
fn backend(conn: &Connection) -> Result<String> {
    let b = query_single_string(conn, "SELECT backend FROM bolt_pool_stats()")?;
    assert!(
        b == BACKEND_GPU || b == BACKEND_FALLBACK,
        "Unexpected backend '{b}'"
    );
    if cfg!(feature = "cuda-stub") {
        assert_eq!(b, BACKEND_FALLBACK, "cuda-stub builds cannot use the GPU");
    }
    Ok(b)
}

/// Value of a `bolt_metrics()` counter.
fn metric_value(conn: &Connection, metric: &str) -> Result<i64> {
    query_single_i64(
        conn,
        &format!("SELECT value FROM bolt_metrics() WHERE metric = '{metric}'"),
    )
}

/// `PRAGMA bolt_reset`.
///
/// Policy since report item 1.1.13 (`BoltWorkerState::handle_reset_engine`): the pragma errors
/// only when the configured CUDA device exists but the engine cannot be built. `cuda-stub` builds
/// see no device, so there it must always succeed. A GPU build on the host fallback (a device that
/// exists but cannot be initialized) may legitimately report the failure.
fn bolt_reset(conn: &Connection) -> Result<()> {
    match conn.execute("PRAGMA bolt_reset", Parameters::None) {
        Ok(_) => Ok(()),
        Err(e) if cfg!(feature = "cuda-stub") => Err(e),
        Err(e) if backend(conn)? == BACKEND_FALLBACK => {
            println!("PRAGMA bolt_reset reported an engine build failure on host fallback: {e}");
            Ok(())
        }
        Err(e) => Err(e),
    }
}

#[test]
fn test_version_options_and_telemetry() -> Result<()> {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let (_env, conn) = setup_connection()?;

    let active_backend = backend(&conn)?;
    let version = query_single_string(&conn, "SELECT bolt_version()")?;
    assert!(
        version.starts_with(&format!(
            "duckdb-bolt-extension v{} (craton-bolt v",
            env!("CARGO_PKG_VERSION")
        )),
        "Unexpected version string: {version}"
    );
    assert!(
        version.contains(&format!("backend={active_backend}")),
        "bolt_version() must report the active backend '{active_backend}': {version}"
    );

    assert_eq!(
        query_single_i64(&conn, "SELECT current_setting('bolt_device')::BIGINT")?,
        0
    );
    assert_eq!(
        query_single_i64(
            &conn,
            "SELECT current_setting('bolt_memory_budget')::BIGINT"
        )?,
        0
    );

    conn.execute("SET bolt_memory_budget = 268435456", Parameters::None)?;
    assert_eq!(
        query_single_i64(
            &conn,
            "SELECT current_setting('bolt_memory_budget')::BIGINT"
        )?,
        268_435_456
    );
    // The runtime configuration (not just DuckDB's setting) must follow the SET.
    assert_eq!(
        query_single_i64(&conn, "SELECT memory_budget_bytes FROM bolt_pool_stats()")?,
        268_435_456
    );

    // Negative budget must be rejected.
    assert_err_contains(
        conn.execute("SET bolt_memory_budget = -1", Parameters::None),
        "must be >= 0",
    );
    // Negative device ordinal must be rejected.
    assert_err_contains(
        conn.execute("SET bolt_device = -1", Parameters::None),
        "non-negative",
    );

    conn.execute("RESET bolt_memory_budget", Parameters::None)?;
    assert_eq!(
        query_single_i64(
            &conn,
            "SELECT current_setting('bolt_memory_budget')::BIGINT"
        )?,
        0
    );
    assert_eq!(
        query_single_i64(&conn, "SELECT memory_budget_bytes FROM bolt_pool_stats()")?,
        0
    );

    bolt_reset(&conn)?;
    assert_eq!(backend(&conn)?, active_backend);

    let pool_rows = query_single_i64(&conn, "SELECT count(*) FROM bolt_pool_stats()")?;
    assert_eq!(pool_rows, 1);
    // setup_connection() cleared the catalog.
    assert_eq!(
        query_single_i64(&conn, "SELECT registered_tables FROM bolt_pool_stats()")?,
        0
    );
    assert_eq!(
        query_single_i64(&conn, "SELECT device_id::BIGINT FROM bolt_pool_stats()")?,
        0
    );

    // PRAGMA bolt_info is an alias for bolt_pool_stats().
    let mut info_res = conn.query("PRAGMA bolt_info", Parameters::None)?;
    let info_chunk = info_res.next_chunk()?.expect("PRAGMA bolt_info row");
    assert_eq!(info_chunk.row_count()?, 1);
    assert_eq!(info_chunk.col_count(), 16);
    assert_eq!(
        info_chunk.get_vector_at::<String>(0)?.reader()?.get(0)?,
        Some(active_backend.as_str())
    );
    drop(info_res);

    // bolt_metrics() exposes every craton-bolt counter, including the query counter.
    let metrics_count = query_single_i64(
        &conn,
        "SELECT count(*) FROM bolt_metrics() WHERE category = 'counter'",
    )?;
    assert!(
        metrics_count >= 6,
        "Expected the six craton-bolt counters, got {metrics_count}"
    );
    // `queries_total` must be exported exactly once (the next check rejects negative values).
    assert_eq!(
        query_single_i64(
            &conn,
            "SELECT count(*) FROM bolt_metrics() WHERE metric = 'queries_total'"
        )?,
        1
    );
    assert_eq!(
        query_single_i64(
            &conn,
            "SELECT count(*) FROM bolt_metrics() WHERE value < 0 OR metric IS NULL OR category IS NULL"
        )?,
        0
    );

    let prom = query_single_string(&conn, "SELECT bolt_prometheus_metrics()")?;
    assert!(
        prom.contains("# TYPE craton_queries_total counter"),
        "Expected Prometheus output to contain the craton_queries_total counter, got: {prom}"
    );
    // No query runs between the two calls, so the alias must render identical text.
    assert_eq!(
        prom,
        query_single_string(&conn, "SELECT bolt_metrics_prometheus()")?,
        "bolt_metrics_prometheus() must be an alias of bolt_prometheus_metrics()"
    );

    // Devices: the GPU backend must see the configured device; a cuda-stub build sees none.
    let devices_count = query_single_i64(&conn, "SELECT count(*) FROM bolt_devices()")?;
    let active_devices =
        query_single_i64(&conn, "SELECT count(*) FROM bolt_devices() WHERE is_active")?;
    assert_eq!(
        query_single_i64(
            &conn,
            "SELECT count(*) FROM bolt_devices() WHERE device_id < 0 OR device_id >= (SELECT count(*) FROM bolt_devices()) OR name IS NULL"
        )?,
        0,
        "bolt_devices() rows must have ordinals 0..count and a name"
    );
    if active_backend == BACKEND_GPU {
        assert!(
            devices_count >= 1,
            "cuda-gpu backend without any CUDA device"
        );
        assert_eq!(active_devices, 1, "exactly one device must be active");
        assert!(query_single_bool(
            &conn,
            "SELECT is_active FROM bolt_devices() WHERE device_id = 0"
        )?);
        assert!(
            query_single_i64(
                &conn,
                "SELECT total_memory_bytes FROM bolt_devices() WHERE device_id = 0"
            )? > 0
        );
        assert!(query_single_bool(
            &conn,
            "SELECT (SELECT device_name FROM bolt_pool_stats()) = (SELECT name FROM bolt_devices() WHERE is_active)"
        )?);
    } else if cfg!(feature = "cuda-stub") {
        assert_eq!(
            devices_count, 0,
            "cuda-stub builds must not enumerate devices"
        );
    } else {
        // A real CUDA build without a usable engine: devices may or may not be visible.
        assert!(active_devices <= 1);
    }

    Ok(())
}

#[test]
fn test_copy_to_gpu_queries_and_copy_from_roundtrip() -> Result<()> {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let (_env, conn) = setup_connection()?;

    conn.execute(
        "CREATE TABLE orders (
            order_id INTEGER,
            customer_id BIGINT,
            region VARCHAR,
            price DOUBLE,
            express BOOLEAN
        )",
        Parameters::None,
    )?;

    conn.execute(
        "INSERT INTO orders VALUES
            (1, 101, 'north', 120.50, true),
            (2, 102, 'south', 80.00, false),
            (3, 101, 'north', 179.50, true),
            (4, 103, 'east',  250.00, false),
            (5, 102, 'south', 220.00, true),
            (6, 104, 'north', 50.00,  false)",
        Parameters::None,
    )?;

    // Upload to Craton Bolt via COPY ... TO (FORMAT bolt) with RETURN_STATS.
    let mut stats = conn.query(
        "COPY orders TO 'gpu_orders' (FORMAT bolt, USE_TMP_FILE FALSE, RETURN_STATS)",
        Parameters::None,
    )?;
    let stats_chunk = stats.next_chunk()?.expect("RETURN_STATS chunk");
    assert_eq!(
        stats_chunk.get_vector_at::<u64>(1)?.reader()?.get(0)?,
        Some(&6)
    );
    drop(stats);

    // Check bolt_tables() catalog entry.
    assert_eq!(
        query_single_i64(
            &conn,
            "SELECT num_rows FROM bolt_tables() WHERE table_name = 'gpu_orders'"
        )?,
        6
    );

    // Execute GPU aggregation query via bolt_query().
    let mut agg = conn.query(
        "SELECT region, total_price, order_cnt
         FROM bolt_query('SELECT region, SUM(price) AS total_price, COUNT(order_id) AS order_cnt FROM gpu_orders GROUP BY region ORDER BY region')",
        Parameters::None,
    )?;
    let chunk = agg.next_chunk()?.expect("agg chunk");
    assert_eq!(chunk.row_count()?, 3);
    let regions = chunk.get_vector_at::<String>(0)?.reader()?;
    let totals = chunk.get_vector_at::<f64>(1)?.reader()?;
    let counts = chunk.get_vector_at::<i64>(2)?.reader()?;

    assert_eq!(regions.get(0)?, Some("east"));
    assert_eq!(totals.get(0)?, Some(&250.0));
    assert_eq!(counts.get(0)?, Some(&1));

    assert_eq!(regions.get(1)?, Some("north"));
    assert_eq!(totals.get(1)?, Some(&350.0));
    assert_eq!(counts.get(1)?, Some(&3));

    assert_eq!(regions.get(2)?, Some("south"));
    assert_eq!(totals.get(2)?, Some(&300.0));
    assert_eq!(counts.get(2)?, Some(&2));
    drop(agg);

    // Append mode via COPY ... TO (FORMAT bolt, MODE 'append').
    conn.execute(
        "COPY (SELECT 7 AS order_id, 105::BIGINT AS customer_id, 'west' AS region, 500.0::DOUBLE AS price, true AS express)
         TO 'gpu_orders' (FORMAT bolt, MODE 'append', USE_TMP_FILE FALSE)",
        Parameters::None,
    )?;
    assert_eq!(
        query_single_i64(
            &conn,
            "SELECT num_rows FROM bolt_tables() WHERE table_name = 'gpu_orders'"
        )?,
        7
    );

    // MODE 'create' on an existing table must fail.
    assert_err_contains(
        conn.execute(
            "COPY orders TO 'gpu_orders' (FORMAT bolt, MODE 'create', USE_TMP_FILE FALSE)",
            Parameters::None,
        ),
        "already registered",
    );

    // Copy back from Craton Bolt into a new DuckDB table via COPY ... FROM (FORMAT bolt).
    conn.execute(
        "CREATE TABLE orders_back (
            order_id INTEGER,
            customer_id BIGINT,
            region VARCHAR,
            price DOUBLE,
            express BOOLEAN
        )",
        Parameters::None,
    )?;
    conn.execute(
        "COPY orders_back FROM 'gpu_orders' (FORMAT bolt)",
        Parameters::None,
    )?;
    assert_eq!(
        query_single_i64(&conn, "SELECT count(*) FROM orders_back")?,
        7
    );
    assert_eq!(
        query_single_i64(&conn, "SELECT SUM(price)::BIGINT FROM orders_back")?,
        1400
    );

    Ok(())
}

#[test]
fn test_bolt_scan_replacement_scan_and_pushdowns() -> Result<()> {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let (_env, conn) = setup_connection()?;

    conn.execute(
        "COPY (
            SELECT
                i::INTEGER AS id,
                (i * 10)::BIGINT AS score,
                CASE WHEN i % 2 = 0 THEN 'even' ELSE 'odd' END AS parity
            FROM range(1, 21) t(i)
        ) TO 'gpu_items' (FORMAT bolt, USE_TMP_FILE FALSE)",
        Parameters::None,
    )?;

    // 1. Direct bolt_scan with projection & filter pushdown
    let sum_filtered = query_single_i64(
        &conn,
        "SELECT SUM(score)::BIGINT FROM bolt_scan('gpu_items')
         WHERE id >= 5 AND id <= 10 AND parity = 'even'",
    )?;
    // Even ids in 5..=10 are 6, 8, 10 -> scores 60 + 80 + 100 = 240.
    assert_eq!(sum_filtered, 240);

    // 2. Replacement scan via bolt.<table_name>
    let count_bolt_ns = query_single_i64(
        &conn,
        "SELECT count(*) FROM bolt.gpu_items WHERE id IN (2, 4, 6, 8) AND score > 30",
    )?;
    // ids 4, 6, 8 -> 3 rows.
    assert_eq!(count_bolt_ns, 3);

    // 3. Replacement scan via bare <table_name> (since gpu_items is not a local DuckDB table)
    let max_score = query_single_i64(
        &conn,
        "SELECT MAX(score) FROM gpu_items WHERE id BETWEEN 10 AND 15",
    )?;
    assert_eq!(max_score, 150);

    // 4. Explain and execution tier
    let explain = query_single_string(
        &conn,
        "SELECT bolt_explain('SELECT id, score FROM gpu_items WHERE score > 50')",
    )?;
    assert!(!explain.is_empty());
    if backend(&conn)? == BACKEND_FALLBACK {
        // Host fallback renders the planner output itself (runtime.rs handle_explain_sql).
        assert!(
            explain.contains("Logical Plan") && explain.contains("Physical Plan"),
            "Unexpected host-fallback plan: {explain}"
        );
    }
    // An unknown table is a planning error, not an empty plan.
    assert!(
        conn.query(
            "SELECT bolt_explain('SELECT * FROM no_such_table_for_explain')",
            Parameters::None
        )
        .and_then(|mut r| r.next_chunk().map(|_| ()))
        .is_err()
    );

    let tier = query_single_string(
        &conn,
        "SELECT bolt_execution_tier('SELECT id, score FROM gpu_items WHERE score > 50')",
    )?;
    assert!(
        matches!(tier.as_str(), "Gpu" | "Host" | "Hybrid"),
        "Unexpected execution tier: {tier}"
    );

    // 5. bolt_register_sql derived GPU table
    let reg_rows = query_single_i64(
        &conn,
        "SELECT bolt_register_sql('gpu_high_scores', 'SELECT id, score FROM gpu_items WHERE score >= 150')",
    )?;
    assert_eq!(reg_rows, 6);

    assert_eq!(
        query_single_i64(&conn, "SELECT SUM(score)::BIGINT FROM bolt.gpu_high_scores")?,
        150 + 160 + 170 + 180 + 190 + 200
    );

    assert!(query_single_bool(
        &conn,
        "SELECT bolt_drop_table('gpu_high_scores')"
    )?);
    assert!(!query_single_bool(
        &conn,
        "SELECT bolt_drop_table('gpu_high_scores')"
    )?);

    Ok(())
}

#[test]
fn test_all_types_options_error_paths_and_edge_cases() -> Result<()> {
    use std::sync::Arc;

    use arrow_array::{
        DictionaryArray, Int8Array, Int16Array, LargeStringArray, RecordBatch, StringArray,
        UInt8Array, UInt16Array, UInt32Array, UInt64Array, types::Int32Type,
    };
    use arrow_schema::{DataType, Field, Schema};
    use duckdb_bolt_core::{
        arrow_bridge::{
            arrow_query_to_record_batch, arrow_type_to_duckdb_type, duckdb_type_to_bolt_arrow_type,
            query_result_to_record_batch, record_batches_to_collection,
        },
        runtime::{BoltRuntime, RegisterMode},
    };
    use duckdb_neo::logical_type::LogicalType;

    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let (_env, conn) = setup_connection()?;

    // 1. Options: bolt_ptx_cache_dir, bolt_enable_tracing, bolt_device
    let tmp_dir = std::env::temp_dir().join("duckdb_bolt_ptx_cache_test");
    let _ = std::fs::create_dir_all(&tmp_dir);
    let escaped_dir = tmp_dir.to_string_lossy().replace('\\', "/");
    conn.execute(
        format!("SET bolt_ptx_cache_dir = '{escaped_dir}'"),
        Parameters::None,
    )?;
    assert_eq!(
        query_single_string(&conn, "SELECT current_setting('bolt_ptx_cache_dir')")?,
        escaped_dir
    );
    conn.execute("SET bolt_enable_tracing = true", Parameters::None)?;
    assert!(query_single_bool(
        &conn,
        "SELECT current_setting('bolt_enable_tracing')::BOOLEAN"
    )?);
    conn.execute("SET bolt_device = 0", Parameters::None)?;
    bolt_reset(&conn)?;
    conn.execute("RESET bolt_ptx_cache_dir", Parameters::None)?;
    conn.execute("RESET bolt_enable_tracing", Parameters::None)?;

    // 2. Upload all sub-32-bit, unsigned, hugeint, float, decimal(1..38), date, and timestamp variants
    conn.execute(
        "COPY (
            SELECT
                1::TINYINT AS c_i8,
                2::SMALLINT AS c_i16,
                3::UTINYINT AS c_u8,
                4::USMALLINT AS c_u16,
                5::UINTEGER AS c_u32,
                6::UBIGINT AS c_u64,
                7::HUGEINT AS c_i128,
                1.5::FLOAT AS c_f32,
                2.5::DOUBLE AS c_f64,
                9.9::DECIMAL(4, 1) AS c_dec4,
                123.45::DECIMAL(9, 2) AS c_dec9,
                1000.1234::DECIMAL(18, 4) AS c_dec18,
                999999.123456::DECIMAL(38, 6) AS c_dec38,
                DATE '2026-03-01' AS c_date,
                TIMESTAMP_S '2026-03-01 01:02:03' AS c_ts_s,
                TIMESTAMP_MS '2026-03-01 01:02:03.456' AS c_ts_ms,
                TIMESTAMP '2026-03-01 01:02:03.456789' AS c_ts_us,
                TIMESTAMP_NS '2026-03-01 01:02:03.456789123' AS c_ts_ns,
                TIMESTAMPTZ '2026-03-01 01:02:03+00' AS c_tstz,
                'alpha'::VARCHAR AS c_str,
                true AS c_bool
            UNION ALL
            SELECT
                NULL, NULL, NULL, NULL, NULL, NULL, NULL,
                NULL, NULL, NULL, NULL, NULL, NULL,
                NULL, NULL, NULL, NULL, NULL, NULL,
                NULL, NULL
        ) TO 'bolt://all_types.bolt' (FORMAT bolt, TABLE 'gpu_all_types', MODE 'create', USE_TMP_FILE FALSE)",
        Parameters::None,
    )?;

    assert_eq!(
        query_single_i64(&conn, "SELECT count(*) FROM bolt.main.gpu_all_types")?,
        2
    );
    assert_eq!(
        query_single_i64(&conn, "SELECT count(*) FROM 'bolt:gpu_all_types'")?,
        2
    );

    // Exercise filter pushdown across all literal types and operators:
    // NOT, IS NULL, IS NOT NULL, !=, <, >, <=, >=, BETWEEN, IN, NOT IN, OR, AND, CAST, FLOAT, DOUBLE, DECIMAL, DATE, BOOLEAN, UINTEGER, TINYINT, SMALLINT
    let filtered = query_single_i64(
        &conn,
        "SELECT count(*) FROM bolt_scan('gpu_all_types')
         WHERE c_i8 = 1::TINYINT
           AND c_i16 != 99::SMALLINT
           AND c_u8 <= 10::UTINYINT
           AND c_u16 >= 1::USMALLINT
           AND c_u32 < 100::UINTEGER
           AND c_f32 > 0.5::FLOAT
           AND c_f64 >= 1.0::DOUBLE
           AND c_dec9 = 123.45::DECIMAL(9, 2)
           AND c_date = DATE '2026-03-01'
           AND (c_bool = true OR c_bool = false)
           AND NOT (c_i8 = 99)
           AND c_str IS NOT NULL
           AND c_i8 NOT IN (10, 20)
           AND c_i8 IN (1, 2)
           AND c_i8 BETWEEN 0 AND 5",
    )?;
    assert_eq!(filtered, 1);

    // Individual filter pushdowns on typed columns (so DuckDB doesn't fold them into a single expression)
    assert_eq!(
        query_single_i64(
            &conn,
            "SELECT count(*) FROM bolt.gpu_all_types WHERE c_f32 > 0.5::FLOAT"
        )?,
        1
    );
    assert_eq!(
        query_single_i64(
            &conn,
            "SELECT count(*) FROM bolt.gpu_all_types WHERE c_dec9 = 123.45::DECIMAL(9, 2)"
        )?,
        1
    );
    assert_eq!(
        query_single_i64(
            &conn,
            "SELECT count(*) FROM bolt.gpu_all_types WHERE c_date = DATE '2026-03-01'"
        )?,
        1
    );
    assert_eq!(
        query_single_i64(
            &conn,
            "SELECT count(*) FROM bolt.gpu_all_types WHERE c_bool = true"
        )?,
        1
    );
    assert_eq!(
        query_single_i64(
            &conn,
            "SELECT count(*) FROM bolt.gpu_all_types WHERE c_bool = false"
        )?,
        0
    );
    assert_eq!(
        query_single_i64(
            &conn,
            "SELECT count(*) FROM bolt.gpu_all_types WHERE c_i8 = 99"
        )?,
        0
    );
    assert_eq!(
        query_single_i64(
            &conn,
            "SELECT count(*) FROM bolt.gpu_all_types WHERE c_str = 'nonexistent' AND c_i8 = 99"
        )?,
        0
    );
    // Triggers execute_gpu_filter_take returning 0 rows when Utf8 column is projected
    let mut empty_take = conn.query(
        "SELECT c_str FROM bolt.gpu_all_types WHERE c_i8 = 99",
        Parameters::None,
    )?;
    assert!(empty_take.next_chunk()?.is_none());

    // Individual type filter pushdowns (UTINYINT, USMALLINT, UINTEGER, FLOAT, DECIMAL, NOT IN, OR, BETWEEN)
    assert_eq!(
        query_single_i64(
            &conn,
            "SELECT count(*) FROM bolt.gpu_all_types WHERE c_u8 = 3::UTINYINT"
        )?,
        1
    );
    assert_eq!(
        query_single_i64(
            &conn,
            "SELECT count(*) FROM bolt.gpu_all_types WHERE c_u16 = 4::USMALLINT"
        )?,
        1
    );
    assert_eq!(
        query_single_i64(
            &conn,
            "SELECT count(*) FROM bolt.gpu_all_types WHERE c_u32 = 5::UINTEGER"
        )?,
        1
    );
    assert_eq!(
        query_single_i64(
            &conn,
            "SELECT count(*) FROM bolt.gpu_all_types WHERE c_f32 > 1.0::FLOAT"
        )?,
        1
    );
    assert_eq!(
        query_single_i64(
            &conn,
            "SELECT count(*) FROM bolt.gpu_all_types WHERE c_dec18 > 10.0000::DECIMAL(18,4)"
        )?,
        1
    );
    assert_eq!(
        query_single_i64(
            &conn,
            "SELECT count(*) FROM bolt.gpu_all_types WHERE c_i8 NOT IN (99, 88)"
        )?,
        1
    );
    assert_eq!(
        query_single_i64(
            &conn,
            "SELECT count(*) FROM bolt.gpu_all_types WHERE (c_i8 = 1 OR c_i8 = 99)"
        )?,
        1
    );
    assert_eq!(
        query_single_i64(
            &conn,
            "SELECT count(*) FROM bolt.gpu_all_types WHERE c_i16 BETWEEN 1::SMALLINT AND 5::SMALLINT"
        )?,
        1
    );
    // Filter pushdown with unsupported expression (e.g. function call) gracefully falls back to DuckDB post-filtering
    assert_eq!(
        query_single_i64(
            &conn,
            "SELECT count(*) FROM bolt.gpu_all_types WHERE length(c_str) = 5"
        )?,
        1
    );

    let null_rows = query_single_i64(
        &conn,
        "SELECT count(*) FROM bolt_scan('gpu_all_types') WHERE c_str IS NULL",
    )?;
    assert_eq!(null_rows, 1);

    // Projected scan returning rich types + Utf8 with filter pushdown (triggers execute_gpu_filter_take)
    let mut rich_scan = conn.query(
        "SELECT c_str, c_dec4, c_dec38, c_ts_s, c_ts_ms, c_ts_ns, c_f32
         FROM bolt_scan('gpu_all_types')
         WHERE c_i8 = 1",
        Parameters::None,
    )?;
    let rich_chunk = rich_scan.next_chunk()?.expect("rich_scan chunk");
    assert_eq!(rich_chunk.row_count()?, 1);
    drop(rich_scan);

    // Duplicate column projection in bolt_scan (`WHERE c_str = 'alpha'` projects `c_str` in filter and SELECT)
    let dup_str = query_single_string(
        &conn,
        "SELECT c_str FROM bolt.gpu_all_types WHERE c_str = 'alpha' AND c_i8 = 1",
    )?;
    assert_eq!(dup_str, "alpha");

    // 3. Direct arrow_bridge coverage for Int8, Int16, UInt8, UInt16, UInt32, UInt64, LargeUtf8, Dictionary,
    // record_batches_to_collection, and query_result_to_record_batch
    let custom_schema = Arc::new(Schema::new(vec![
        Field::new("i8", DataType::Int8, true),
        Field::new("i16", DataType::Int16, true),
        Field::new("u8", DataType::UInt8, true),
        Field::new("u16", DataType::UInt16, true),
        Field::new("u32", DataType::UInt32, true),
        Field::new("u64", DataType::UInt64, true),
        Field::new("lstr", DataType::LargeUtf8, true),
        Field::new(
            "dict",
            DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
            true,
        ),
    ]));

    let dict_keys = arrow_array::Int32Array::from(vec![Some(0), None]);
    let dict_vals = StringArray::from(vec!["d0"]);
    let dict_arr = DictionaryArray::<Int32Type>::try_new(dict_keys, Arc::new(dict_vals)).unwrap();

    let custom_batch = RecordBatch::try_new(
        custom_schema.clone(),
        vec![
            Arc::new(Int8Array::from(vec![Some(10), None])),
            Arc::new(Int16Array::from(vec![Some(20), None])),
            Arc::new(UInt8Array::from(vec![Some(30), None])),
            Arc::new(UInt16Array::from(vec![Some(40), None])),
            Arc::new(UInt32Array::from(vec![Some(50), None])),
            Arc::new(UInt64Array::from(vec![Some(60), None])),
            Arc::new(LargeStringArray::from(vec![Some("large"), None])),
            Arc::new(dict_arr),
        ],
    )
    .unwrap();

    let collection =
        record_batches_to_collection(&conn, &custom_schema, std::slice::from_ref(&custom_batch))?;
    assert_eq!(collection.len()?, 2);

    // Test zero-copy Arrow C Stream Interface (`arrow_query_to_record_batch`) across empty, single, and multi-batch results
    let stream_empty = arrow_query_to_record_batch(&conn, "SELECT 1 AS a WHERE 1 = 0")?;
    assert_eq!(stream_empty.num_rows(), 0);

    let stream_single =
        arrow_query_to_record_batch(&conn, "SELECT 42::INTEGER AS x, 'hello' AS y")?;
    assert_eq!(stream_single.num_rows(), 1);
    assert_eq!(stream_single.num_columns(), 2);

    let stream_multi = arrow_query_to_record_batch(&conn, "SELECT * FROM range(2500)")?;
    assert_eq!(stream_multi.num_rows(), 2500);

    // Test query_result_to_record_batch across types
    let mut qr = conn.query(
        "SELECT
            1::TINYINT AS a,
            2::SMALLINT AS b,
            3::UTINYINT AS c,
            4::USMALLINT AS d,
            5::UINTEGER AS e,
            6::UBIGINT AS f,
            7::HUGEINT AS g,
            1.25::FLOAT AS h,
            2.5::DOUBLE AS i,
            'txt'::VARCHAR AS j,
            true AS k,
            DATE '2026-01-01' AS l,
            TIMESTAMP_S '2026-01-01 00:00:00' AS m,
            TIMESTAMP_MS '2026-01-01 00:00:00' AS n,
            TIMESTAMP '2026-01-01 00:00:00' AS o,
            TIMESTAMP_NS '2026-01-01 00:00:00' AS p,
            TIMESTAMPTZ '2026-01-01 00:00:00+00' AS q,
            1.2::DECIMAL(4,1) AS r,
            12.34::DECIMAL(9,2) AS s,
            123.456::DECIMAL(18,3) AS t,
            1234.5678::DECIMAL(38,4) AS u
        UNION ALL
        SELECT
            NULL, NULL, NULL, NULL, NULL, NULL, NULL,
            NULL, NULL, NULL, NULL, NULL, NULL, NULL,
            NULL, NULL, NULL, NULL, NULL, NULL, NULL",
        Parameters::None,
    )?;
    let rb = query_result_to_record_batch(&mut qr)?;
    assert_eq!(rb.num_rows(), 2);
    assert_eq!(rb.num_columns(), 21);

    let rb_collection =
        record_batches_to_collection(&conn, &rb.schema(), std::slice::from_ref(&rb))?;
    assert_eq!(rb_collection.len()?, 2);

    // Host fallback execution path (`execute_sql_host_fallback`)
    let fallback_out = BoltRuntime::global()
        .execute_sql_host_fallback("SELECT c_i8, c_str FROM gpu_all_types WHERE c_i8 = 1")?;
    assert_eq!(fallback_out.batch.num_rows(), 1);

    // 4. Schema coercion on append & multi-batch concatenation
    conn.execute(
        "COPY (SELECT 1::INTEGER AS x, 10::BIGINT AS y) TO 'gpu_append_test' (FORMAT bolt, MODE 'replace', USE_TMP_FILE FALSE)",
        Parameters::None,
    )?;
    conn.execute(
        "COPY (SELECT 2::SMALLINT AS x, 20::INTEGER AS y) TO 'gpu_append_test' (FORMAT bolt, MODE 'append', USE_TMP_FILE FALSE)",
        Parameters::None,
    )?;
    assert_eq!(
        query_single_i64(&conn, "SELECT SUM(y)::BIGINT FROM bolt.gpu_append_test")?,
        30
    );

    // Empty table COPY TO and COPY FROM
    conn.execute(
        "COPY (SELECT 1::INTEGER AS a WHERE 1 = 0) TO 'gpu_empty' (FORMAT bolt, USE_TMP_FILE FALSE)",
        Parameters::None,
    )?;
    conn.execute("CREATE TABLE empty_dest (a INTEGER)", Parameters::None)?;
    conn.execute(
        "COPY empty_dest FROM 'gpu_empty' (FORMAT bolt)",
        Parameters::None,
    )?;
    assert_eq!(
        query_single_i64(&conn, "SELECT count(*) FROM empty_dest")?,
        0
    );

    // Scalar bolt_clear_tables() returns exactly the number of tables it removed.
    let registered_before = query_single_i64(&conn, "SELECT count(*) FROM bolt_tables()")?;
    // gpu_all_types, gpu_append_test, gpu_empty (temporary filter tables are dropped after use).
    assert!(
        registered_before >= 3,
        "expected at least 3 registered tables, got {registered_before}"
    );
    let cleared = query_single_i64(&conn, "SELECT bolt_clear_tables()")?;
    assert_eq!(cleared, registered_before);
    assert_eq!(
        query_single_i64(&conn, "SELECT count(*) FROM bolt_tables()")?,
        0
    );

    // 5. Error paths & negative tests
    assert!(RegisterMode::parse("invalid_mode").is_err());
    assert!(arrow_type_to_duckdb_type(&conn, &DataType::Binary).is_err());
    let blob_lt = LogicalType::from_text(&conn, "BLOB")?;
    assert!(duckdb_type_to_bolt_arrow_type(&blob_lt).is_err());

    // UBIGINT > i64::MAX and HUGEINT > i64::MAX must error cleanly
    assert_err_contains(
        conn.execute(
            "COPY (SELECT 18446744073709551615::UBIGINT AS big) TO 'gpu_err' (FORMAT bolt, USE_TMP_FILE FALSE)",
            Parameters::None,
        ),
        "exceeds i64::MAX",
    );
    assert_err_contains(
        conn.execute(
            "COPY (SELECT (9223372036854775807::HUGEINT + 10) AS huge) TO 'gpu_err' (FORMAT bolt, USE_TMP_FILE FALSE)",
            Parameters::None,
        ),
        "exceeds i64 range",
    );

    // Unknown COPY option
    assert_err_contains(
        conn.execute(
            "COPY (SELECT 1 AS a) TO 'gpu_err' (FORMAT bolt, UNKNOWN_OPT 123, USE_TMP_FILE FALSE)",
            Parameters::None,
        ),
        "Unknown COPY TO (FORMAT bolt) option",
    );

    // Column count mismatch in COPY FROM
    conn.execute(
        "COPY (SELECT 1 AS a, 2 AS b) TO 'gpu_two_cols' (FORMAT bolt, USE_TMP_FILE FALSE)",
        Parameters::None,
    )?;
    assert_err_contains(
        conn.execute(
            "COPY empty_dest FROM 'gpu_two_cols' (FORMAT bolt)",
            Parameters::None,
        ),
        "Column count mismatch",
    );

    // NULL handling in scalar functions
    let mut null_sc = conn.query(
        "SELECT bolt_explain(NULL), bolt_execution_tier(NULL), bolt_drop_table(NULL), bolt_register_sql(NULL, 'SELECT 1')",
        Parameters::None,
    )?;
    let null_chunk = null_sc.next_chunk()?.expect("null scalar chunk");
    assert_eq!(
        null_chunk.get_vector_at::<String>(0)?.reader()?.get(0)?,
        None
    );
    assert_eq!(
        null_chunk.get_vector_at::<String>(1)?.reader()?.get(0)?,
        None
    );
    assert_eq!(null_chunk.get_vector_at::<bool>(2)?.reader()?.get(0)?, None);
    assert_eq!(null_chunk.get_vector_at::<i64>(3)?.reader()?.get(0)?, None);
    drop(null_sc);

    // Empty / invalid SQL in bolt_query and non-existent table in bolt_scan
    assert_err_contains(
        query_first_chunk(&conn, "SELECT * FROM bolt_query('')"),
        "cannot be empty",
    );
    assert_err_contains(
        query_first_chunk(&conn, "SELECT * FROM bolt_scan('non_existent_table_xyz')"),
        "is not registered in Craton Bolt",
    );

    // Duplicate column names in bolt_query output are disambiguated
    conn.execute(
        "COPY (SELECT 1::INTEGER AS id) TO 'gpu_dup_col' (FORMAT bolt, USE_TMP_FILE FALSE)",
        Parameters::None,
    )?;
    let mut dup_q = conn.query(
        "SELECT * FROM bolt_query('SELECT id, id FROM gpu_dup_col')",
        Parameters::None,
    )?;
    let dup_chunk = dup_q.next_chunk()?.expect("dup_q chunk");
    assert_eq!(dup_chunk.col_count(), 2);
    drop(dup_q);

    // Non-constant argument to bolt_query / bolt_scan must fail at bind time
    assert!(
        conn.query(
            "SELECT * FROM range(1) t(x), LATERAL bolt_query(x::VARCHAR)",
            Parameters::None
        )
        .is_err()
    );
    assert!(
        conn.query(
            "SELECT * FROM range(1) t(x), LATERAL bolt_scan(x::VARCHAR)",
            Parameters::None
        )
        .is_err()
    );

    // Empty table name in COPY TO must fail
    assert!(
        conn.execute(
            "COPY (SELECT 1 AS a) TO '' (FORMAT bolt, USE_TMP_FILE FALSE)",
            Parameters::None
        )
        .is_err()
    );
    assert!(BoltRuntime::global().drop_table("").is_err());

    // Append to non-existent table via MODE 'append' creates it, then append a second batch
    conn.execute(
        "COPY (SELECT 10::INTEGER AS k, 'v'::VARCHAR AS v) TO 'gpu_append_new' (FORMAT bolt, MODE 'append', USE_TMP_FILE FALSE)",
        Parameters::None,
    )?;
    conn.execute(
        "COPY (SELECT 20::SMALLINT AS k, 'w'::VARCHAR AS v) TO 'gpu_append_new' (FORMAT bolt, MODE 'append', USE_TMP_FILE FALSE)",
        Parameters::None,
    )?;
    assert_eq!(
        query_single_i64(
            &conn,
            "SELECT count(*) FROM bolt.gpu_append_new WHERE v = 'v'"
        )?,
        1
    );

    // Append with mismatched column count must fail
    assert_err_contains(
        conn.execute(
            "COPY (SELECT 10::INTEGER AS k) TO 'gpu_append_new' (FORMAT bolt, MODE 'append', USE_TMP_FILE FALSE)",
            Parameters::None,
        ),
        "Schema mismatch",
    );

    // Reset engine while tables are registered in the catalog re-registers them (including multi-batch tables)
    let backend_before_reset = backend(&conn)?;
    bolt_reset(&conn)?;
    assert_eq!(backend(&conn)?, backend_before_reset);
    assert_eq!(
        query_single_i64(&conn, "SELECT SUM(k)::BIGINT FROM bolt.gpu_append_new")?,
        30
    );

    // Invalid device ordinal triggers host-planner-fallback path in telemetry, explain, execution_tier, execute_sql
    conn.execute("SET bolt_device = 999", Parameters::None)?;
    bolt_reset(&conn)?;
    let fallback_backend = backend(&conn)?;
    assert_eq!(fallback_backend, BACKEND_FALLBACK);
    assert_eq!(
        query_single_i64(&conn, "SELECT device_id::BIGINT FROM bolt_pool_stats()")?,
        999
    );
    // No device can be active when the configured ordinal does not exist.
    assert_eq!(
        query_single_i64(&conn, "SELECT count(*) FROM bolt_devices() WHERE is_active")?,
        0
    );

    let fallback_plan = query_single_string(
        &conn,
        "SELECT bolt_explain('SELECT k FROM gpu_append_new WHERE k = 10')",
    )?;
    assert!(fallback_plan.contains("Logical Plan"));

    let fallback_tier = query_single_string(
        &conn,
        "SELECT bolt_execution_tier('SELECT k FROM gpu_append_new WHERE k = 10')",
    )?;
    assert!(
        matches!(fallback_tier.as_str(), "Gpu" | "Host" | "Hybrid"),
        "Unexpected execution tier: {fallback_tier}"
    );

    let fallback_q = query_single_i64(
        &conn,
        "SELECT count(*) FROM bolt_query('SELECT k FROM gpu_append_new WHERE k = 10')",
    )?;
    assert_eq!(fallback_q, 1);

    let fallback_scan = query_single_i64(
        &conn,
        "SELECT count(*) FROM bolt_scan('gpu_append_new') WHERE k = 10",
    )?;
    assert_eq!(fallback_scan, 1);

    // Restore device 0
    conn.execute("SET bolt_device = 0", Parameters::None)?;
    bolt_reset(&conn)?;
    assert_eq!(backend(&conn)?, backend_before_reset);

    Ok(())
}

#[test]
fn test_gpu_join_and_rich_data_types() -> Result<()> {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let (_env, conn) = setup_connection()?;

    // 1. Register dimension and fact tables for GPU INNER JOIN.
    conn.execute(
        "COPY (
            SELECT * FROM (VALUES
                (1::INTEGER, 'Hardware'::VARCHAR),
                (2::INTEGER, 'Software'::VARCHAR)
            ) AS dept(dept_id, dept_name)
        ) TO 'gpu_depts' (FORMAT bolt, USE_TMP_FILE FALSE)",
        Parameters::None,
    )?;

    conn.execute(
        "COPY (
            SELECT * FROM (VALUES
                (10::INTEGER, 1::INTEGER, 120000::BIGINT),
                (20::INTEGER, 2::INTEGER, 150000::BIGINT),
                (30::INTEGER, 1::INTEGER, 110000::BIGINT)
            ) AS emp_join(emp_id, dept_id, salary)
        ) TO 'gpu_emp_join' (FORMAT bolt, USE_TMP_FILE FALSE)",
        Parameters::None,
    )?;

    let joined_count = query_single_i64(
        &conn,
        "SELECT count(*) FROM bolt_query('
            SELECT gpu_emp_join.emp_id, gpu_depts.dept_name, gpu_emp_join.salary
            FROM gpu_emp_join
            INNER JOIN gpu_depts ON gpu_emp_join.dept_id = gpu_depts.dept_id
        ')",
    )?;
    assert_eq!(joined_count, 3);

    // 2. Register table with rich types (DECIMAL(4,1), DECIMAL(9,2), DECIMAL(18,4), DECIMAL(38,6), DATE, TIMESTAMP, NULLs).
    conn.execute(
        "COPY (
            SELECT * FROM (VALUES
                (10::INTEGER, 123.45::DECIMAL(9, 2), 1000.1234::DECIMAL(18, 4), DATE '2026-01-15', TIMESTAMP '2026-01-15 10:30:00'),
                (20::INTEGER, 200.00::DECIMAL(9, 2), 2000.5000::DECIMAL(18, 4), DATE '2026-02-20', TIMESTAMP '2026-02-20 12:00:00'),
                (30::INTEGER, NULL::DECIMAL(9, 2),   500.0000::DECIMAL(18, 4),  NULL::DATE,       NULL::TIMESTAMP)
            ) AS emp(emp_id, bonus_short, salary_long, hire_date, created_at)
        ) TO 'gpu_emps' (FORMAT bolt, USE_TMP_FILE FALSE)",
        Parameters::None,
    )?;

    // Scan back rich types and verify values & NULL preservation
    let mut scan = conn.query(
        "SELECT emp_id, bonus_short, salary_long, hire_date, created_at
         FROM bolt_scan('gpu_emps')
         ORDER BY emp_id",
        Parameters::None,
    )?;
    let chunk = scan.next_chunk()?.expect("rich types chunk");
    assert_eq!(chunk.row_count()?, 3);

    let emp_ids = chunk.get_vector_at::<i32>(0)?.reader()?;
    let bonuses = chunk.get_vector_at::<WordDecimal>(1)?.reader()?;
    let salaries = chunk.get_vector_at::<LongDecimal>(2)?.reader()?;
    let dates = chunk.get_vector_at::<DateValue>(3)?.reader()?;
    let timestamps = chunk.get_vector_at::<TimestampValue>(4)?.reader()?;

    assert_eq!(emp_ids.get(0)?, Some(&10));
    assert_eq!(bonuses.get(0)?, Some(&WordDecimal(12_345)));
    assert_eq!(salaries.get(0)?, Some(&LongDecimal(10_001_234)));
    // DATE '2026-01-15' is day 20468 since the epoch; TIMESTAMP '2026-01-15 10:30:00' in micros.
    assert_eq!(dates.get(0)?.map(|d| d.0), Some(20_468));
    assert_eq!(timestamps.get(0)?.map(|t| t.0), Some(1_768_473_000_000_000));

    assert_eq!(emp_ids.get(2)?, Some(&30));
    assert_eq!(bonuses.get(2)?, None);
    assert_eq!(salaries.get(2)?, Some(&LongDecimal(5_000_000)));
    assert_eq!(dates.get(2)?, None);
    assert_eq!(timestamps.get(2)?, None);
    drop(scan);

    Ok(())
}

#[test]
fn test_rfc002_schema_only_bind_and_eager_fallback() -> Result<()> {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let (_env, conn) = setup_connection()?;

    conn.execute(
        "CREATE TABLE rfc2 (id INTEGER, grp VARCHAR, v DOUBLE)",
        Parameters::None,
    )?;
    conn.execute(
        "INSERT INTO rfc2 VALUES (1,'a',1.5),(2,'a',2.5),(3,'b',4.0)",
        Parameters::None,
    )?;
    conn.execute(
        "COPY rfc2 TO 'gpu_rfc2' (FORMAT bolt, USE_TMP_FILE FALSE)",
        Parameters::None,
    )?;

    // A query with LATERAL fails Bolt's static planner, but since W11-A (W9-A-001) its schema
    // is inferred by binding it in DuckDB, still without executing it:
    let lateral = duckdb_bolt_core::runtime::BoltRuntime::global().infer_sql_schema(
        "SELECT r1.id FROM gpu_rfc2 r1, LATERAL (SELECT r2.v FROM gpu_rfc2 r2 WHERE r2.id = r1.id) l",
    )?;
    assert_eq!(lateral.fields().len(), 1);
    assert_eq!(lateral.field(0).data_type(), &arrow_schema::DataType::Int32);

    // Schema inference is plan-only and reports the right Arrow-derived types.
    let schema = duckdb_bolt_core::runtime::BoltRuntime::global()
        .infer_sql_schema("SELECT grp, SUM(v) AS total FROM gpu_rfc2 GROUP BY grp")?;
    assert_eq!(schema.fields().len(), 2);
    assert_eq!(schema.field(1).name(), "total");

    // Inference failure surfaces as an error (unknown table).
    assert!(
        duckdb_bolt_core::runtime::BoltRuntime::global()
            .infer_sql_schema("SELECT * FROM no_such_table")
            .is_err()
    );

    // DESCRIBE binds without needing the result rows.
    let described = query_single_string(
        &conn,
        "SELECT column_name FROM (DESCRIBE SELECT * FROM bolt_query('SELECT id, v FROM gpu_rfc2')) ORDER BY column_name LIMIT 1",
    )?;
    assert_eq!(described, "id");
    // ... and with the planner-inferred types.
    assert_eq!(
        query_single_string(
            &conn,
            "SELECT column_type FROM (DESCRIBE SELECT * FROM bolt_query('SELECT id, v FROM gpu_rfc2')) WHERE column_name = 'id'",
        )?,
        "INTEGER"
    );
    assert_eq!(
        query_single_string(
            &conn,
            "SELECT column_type FROM (DESCRIBE SELECT * FROM bolt_query('SELECT id, v FROM gpu_rfc2')) WHERE column_name = 'v'",
        )?,
        "DOUBLE"
    );

    // A query whose results are fetched after a deferred bind returns the right rows.
    assert_eq!(
        query_single_i64(
            &conn,
            "SELECT count(*) FROM bolt_query('SELECT id, v FROM gpu_rfc2 WHERE id >= 2')"
        )?,
        2
    );

    // A query using a host-engine path (such as LATERAL derived tables) fails static parse_sql
    // schema inference, but falls back eagerly at bind time and executes successfully.
    let lateral_sql = "SELECT count(*) FROM bolt_query('SELECT r1.id FROM gpu_rfc2 r1, LATERAL (SELECT r2.v FROM gpu_rfc2 r2 WHERE r2.id = r1.id) l')";
    // Report 1.1.6 (fixed): the host fallback no longer requires Bolt's raw planner to accept the
    // query, so LATERAL must succeed on both backends.
    assert_eq!(query_single_i64(&conn, lateral_sql)?, 3);

    // Unplannable / unknown queries surface a clear bind error (inference fails, eager fallback errors too).
    assert!(
        conn.query(
            "SELECT * FROM bolt_query('SELECT * FROM no_such_table')",
            Parameters::None
        )
        .is_err()
    );
    Ok(())
}

/// RFC-002 acceptance: `EXPLAIN`, `DESCRIBE`, and `PREPARE` of `bolt_query(...)` must not execute
/// the query when the planner can infer its schema.
///
/// Signal: craton-bolt's `queries_total` counter (exposed by `bolt_metrics()`) is bumped by every
/// `Engine::sql` call and not by `explain_sql` / planning. The embedded-DuckDB host fallback never
/// calls `Engine::sql`, so on that backend the counter only proves that nothing reached the GPU
/// engine; the positive control (an actual execution bumps the counter) runs on `cuda-gpu` only.
#[test]
fn test_rfc002_introspection_does_not_execute_query() -> Result<()> {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let (_env, conn) = setup_connection()?;

    conn.execute(
        "COPY (SELECT i::INTEGER AS id, (i * 1.5)::DOUBLE AS v FROM range(1, 6) t(i))
         TO 'gpu_rfc2_noexec' (FORMAT bolt, USE_TMP_FILE FALSE)",
        Parameters::None,
    )?;
    let inner_sql = "SELECT id, v FROM gpu_rfc2_noexec WHERE id >= 2";
    // Precondition: the schema is statically inferable, otherwise bind executes eagerly by design.
    let inferred = duckdb_bolt_core::runtime::BoltRuntime::global().infer_sql_schema(inner_sql)?;
    assert_eq!(inferred.fields().len(), 2);

    let active_backend = backend(&conn)?;
    let before = metric_value(&conn, "queries_total")?;

    // EXPLAIN (not EXPLAIN ANALYZE) binds and plans only.
    conn.execute(
        format!("EXPLAIN SELECT * FROM bolt_query('{inner_sql}')"),
        Parameters::None,
    )?;
    // DESCRIBE binds only, and reports the inferred schema.
    assert_eq!(
        count_rows(
            &conn,
            &format!("DESCRIBE SELECT * FROM bolt_query('{inner_sql}')")
        )?,
        2
    );
    // PREPARE binds only.
    conn.execute(
        format!("PREPARE rfc2_noexec AS SELECT * FROM bolt_query('{inner_sql}')"),
        Parameters::None,
    )?;

    let after_introspection = metric_value(&conn, "queries_total")?;
    assert_eq!(
        after_introspection, before,
        "EXPLAIN / DESCRIBE / PREPARE of bolt_query executed the query ({before} -> {after_introspection} engine queries)"
    );

    // Executing the prepared statement returns the rows ...
    assert_eq!(count_rows(&conn, "EXECUTE rfc2_noexec")?, 4);
    let after_execute = metric_value(&conn, "queries_total")?;
    if active_backend == BACKEND_GPU {
        // ... and is visible to the counter, which proves the signal above is meaningful.
        assert!(
            after_execute > after_introspection,
            "queries_total did not move on execution ({after_introspection} -> {after_execute}); the no-execution check is vacuous"
        );
    } else {
        println!(
            "note: host-planner-fallback has no execution counter; checked only that introspection does not reach Engine::sql"
        );
    }
    conn.execute("DEALLOCATE rfc2_noexec", Parameters::None)?;
    Ok(())
}

#[test]
fn test_rfc003_extended_filter_pushdown_matches_duckdb() -> Result<()> {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let (_env, conn) = setup_connection()?;

    conn.execute(
        "CREATE TABLE rfc3_local AS
         SELECT i::INTEGER AS id,
                (i % 7)::INTEGER AS a,
                (i % 5)::INTEGER AS b,
                CASE WHEN i % 9 = 0 THEN NULL
                     WHEN i % 3 = 0 THEN 'north_' || i::VARCHAR
                     WHEN i % 3 = 1 THEN 'south_' || i::VARCHAR
                     ELSE 'east' END AS name,
                CASE WHEN i % 11 = 0 THEN NULL
                     ELSE TIMESTAMP '2024-01-01 00:00:00' + INTERVAL (i * 7) HOUR END AS ts
         FROM range(1, 201) t(i)",
        Parameters::None,
    )?;
    conn.execute(
        "COPY rfc3_local TO 'gpu_rfc3' (FORMAT bolt, USE_TMP_FILE FALSE)",
        Parameters::None,
    )?;

    let predicates = [
        "ts >= TIMESTAMP '2024-03-01 00:00:00'",
        "ts < TIMESTAMP '2024-02-01 12:30:15'",
        "ts BETWEEN TIMESTAMP '2024-02-01 00:00:00' AND TIMESTAMP '2024-02-20 00:00:00'",
        "name LIKE 'north%'",
        "name LIKE '%_1'",
        "name NOT LIKE 'south%'",
        "name LIKE 'east'",
        "starts_with(name, 'so')",
        "suffix(name, '9')",
        "contains(name, 'th_1')",
        "a + b > 6",
        "a * 2 - b >= 5",
        "a + b = 3 AND name LIKE 'n%'",
        "(a + b) < 2 OR ts > TIMESTAMP '2024-06-01 00:00:00'",
    ];

    for pred in predicates {
        let expected = query_single_i64(
            &conn,
            &format!("SELECT count(*) FROM rfc3_local WHERE {pred}"),
        )?;
        let actual = query_single_i64(
            &conn,
            &format!("SELECT count(*) FROM bolt_scan('gpu_rfc3') WHERE {pred}"),
        )?;
        assert_eq!(actual, expected, "pushdown mismatch for predicate: {pred}");
        let expected_sum = query_single_i64(
            &conn,
            &format!("SELECT coalesce(sum(id), 0)::BIGINT FROM rfc3_local WHERE {pred}"),
        )?;
        let actual_sum = query_single_i64(
            &conn,
            &format!("SELECT coalesce(sum(id), 0)::BIGINT FROM bolt_scan('gpu_rfc3') WHERE {pred}"),
        )?;
        assert_eq!(
            actual_sum, expected_sum,
            "sum mismatch for predicate: {pred}"
        );
    }

    // Verify that translate_expr_to_bolt_sql produces pushdown SQL for supported predicates
    let backend = query_single_string(&conn, "SELECT backend FROM bolt_pool_stats()")?;
    if backend == "cuda-gpu" {
        // Under real CUDA GPU backend, explain reflects the pushed filter in the scan operator
        let mut explain_res = conn.query(
            "EXPLAIN SELECT * FROM bolt_scan('gpu_rfc3') WHERE a + b > 6 AND name LIKE 'north%'",
            Parameters::None,
        )?;
        let mut full_plan = String::new();
        while let Some(chunk) = explain_res.next_chunk()? {
            let vec = chunk.get_vector_at::<String>(1)?;
            for s in vec.reader()?.iter().flatten() {
                full_plan.push_str(s);
                full_plan.push('\n');
            }
        }
        assert!(
            full_plan.contains("BOLT_SCAN") || full_plan.contains("bolt_scan"),
            "full_plan should contain BOLT_SCAN: {full_plan}"
        );
    }

    Ok(())
}

#[test]
fn test_rfc004_load_parquet_and_arrow_ipc() -> Result<()> {
    use std::{fs::File, sync::Arc};

    use arrow_array::{Int16Array, RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema};

    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    // Scalar execution errors surface lazily, so drain the first chunk too.
    fn query_fails(conn: &Connection, sql: &str) -> bool {
        match conn.query(sql, Parameters::None) {
            Err(_) => true,
            Ok(mut r) => r.next_chunk().is_err(),
        }
    }
    let (_env, conn) = setup_connection()?;

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int16, false),
        Field::new("label", DataType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int16Array::from(vec![1, 2, 3, 4])),
            Arc::new(StringArray::from(vec![
                Some("a"),
                Some("b"),
                None,
                Some("d"),
            ])),
        ],
    )
    .unwrap();

    let dir = std::env::temp_dir();
    let pq = dir.join(format!("bolt_rfc4_{}.parquet", std::process::id()));
    let ipc = dir.join(format!("bolt_rfc4_{}.arrow", std::process::id()));
    {
        let mut w =
            parquet::arrow::ArrowWriter::try_new(File::create(&pq).unwrap(), schema.clone(), None)
                .unwrap();
        w.write(&batch).unwrap();
        w.close().unwrap();
        let mut w =
            arrow_ipc::writer::FileWriter::try_new(File::create(&ipc).unwrap(), &schema).unwrap();
        w.write(&batch).unwrap();
        w.finish().unwrap();
    }
    let pq_s = pq.to_str().unwrap().replace('\'', "''");
    let ipc_s = ipc.to_str().unwrap().replace('\'', "''");

    // Create / replace / append modes for Parquet.
    assert_eq!(
        query_single_i64(
            &conn,
            &format!("SELECT bolt_load_parquet('rfc4_pq', '{pq_s}')")
        )?,
        4
    );
    assert_eq!(
        query_single_i64(
            &conn,
            &format!("SELECT bolt_load_parquet('rfc4_pq', '{pq_s}', 'replace')")
        )?,
        4
    );
    assert_eq!(
        query_single_i64(
            &conn,
            &format!("SELECT bolt_load_parquet('rfc4_pq', '{pq_s}', 'append')")
        )?,
        8
    );
    assert!(query_fails(
        &conn,
        &format!("SELECT bolt_load_parquet('rfc4_pq', '{pq_s}', 'create')")
    ));
    assert!(query_fails(
        &conn,
        &format!("SELECT bolt_load_parquet('rfc4_pq', '{pq_s}', 'bogus')")
    ));

    // The Int16 column was normalized to INTEGER and is queryable on Bolt.
    assert_eq!(
        query_single_i64(&conn, "SELECT SUM(id)::BIGINT FROM bolt_scan('rfc4_pq')")?,
        20
    );
    assert_eq!(
        query_single_i64(
            &conn,
            "SELECT count(*) FROM bolt_query('SELECT id FROM rfc4_pq WHERE id > 2')"
        )?,
        4
    );

    // Arrow IPC.
    assert_eq!(
        query_single_i64(
            &conn,
            &format!("SELECT bolt_load_arrow_ipc('rfc4_ipc', '{ipc_s}')")
        )?,
        4
    );
    assert_eq!(
        query_single_i64(
            &conn,
            "SELECT count(*) FROM bolt_scan('rfc4_ipc') WHERE label IS NULL"
        )?,
        1
    );

    // Error paths: missing file, wrong format, NULL args.
    assert!(query_fails(
        &conn,
        "SELECT bolt_load_parquet('rfc4_x', 'no/such/file.parquet')"
    ));
    assert!(query_fails(
        &conn,
        &format!("SELECT bolt_load_parquet('rfc4_x', '{ipc_s}')")
    ));
    let mut res = conn.query("SELECT bolt_load_parquet(NULL, NULL)", Parameters::None)?;
    let chunk = res.next_chunk()?.expect("chunk");
    assert_eq!(chunk.get_vector_at::<i64>(0)?.reader()?.get(0)?, None);
    drop(res);

    let _ = std::fs::remove_file(pq);
    let _ = std::fs::remove_file(ipc);
    Ok(())
}

#[test]
fn test_bolt_aggregate_and_extended_literals() -> Result<()> {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let (_env, conn) = setup_connection()?;

    conn.execute(
        "CREATE TABLE agg_src AS
         SELECT
             i::INTEGER AS id,
             (i % 4)::BIGINT AS grp,
             (i * 10)::BIGINT AS score,
             (i * 2)::UBIGINT AS uval,
             (i * 100)::HUGEINT AS hval,
             TIMESTAMP '2024-01-01 00:00:00' + INTERVAL (i * 2) HOUR AS ts
         FROM range(1, 41) t(i)",
        Parameters::None,
    )?;
    conn.execute(
        "COPY agg_src TO 'gpu_agg_src' (FORMAT bolt, USE_TMP_FILE FALSE)",
        Parameters::None,
    )?;

    // 1. bolt_aggregate with GROUP BY
    let mut res = conn.query(
        "SELECT grp, cnt, total_score FROM bolt_aggregate('gpu_agg_src', 'grp', 'count(id) AS cnt, sum(score) AS total_score') ORDER BY grp",
        Parameters::None,
    )?;
    let chunk = res.next_chunk()?.expect("chunk");
    assert_eq!(chunk.row_count()?, 4);
    let grps = chunk.get_vector_at::<i64>(0)?.reader()?;
    let cnts = chunk.get_vector_at::<i64>(1)?.reader()?;
    assert_eq!(grps.get(0)?, Some(&0));
    assert_eq!(cnts.get(0)?, Some(&10));
    drop(res);

    // 2. bolt_aggregate without GROUP BY (global aggregate)
    let total = query_single_i64(
        &conn,
        "SELECT total::BIGINT FROM bolt_aggregate('gpu_agg_src', '', 'sum(score) AS total')",
    )?;
    let expected_total = query_single_i64(&conn, "SELECT sum(score)::BIGINT FROM agg_src")?;
    assert_eq!(total, expected_total);

    // 3. Extended literal pushdowns: UBIGINT, HUGEINT, TIMESTAMPTZ literals
    let cnt_u = query_single_i64(
        &conn,
        "SELECT count(*) FROM bolt_scan('gpu_agg_src') WHERE uval >= 20::UBIGINT",
    )?;
    let expected_cnt_u = query_single_i64(
        &conn,
        "SELECT count(*) FROM agg_src WHERE uval >= 20::UBIGINT",
    )?;
    assert_eq!(cnt_u, expected_cnt_u);

    let cnt_h = query_single_i64(
        &conn,
        "SELECT count(*) FROM bolt_scan('gpu_agg_src') WHERE hval <= 2000::HUGEINT",
    )?;
    let expected_cnt_h = query_single_i64(
        &conn,
        "SELECT count(*) FROM agg_src WHERE hval <= 2000::HUGEINT",
    )?;
    assert_eq!(cnt_h, expected_cnt_h);

    // 4. Invalid input checks for bolt_aggregate
    assert_err_contains(
        query_first_chunk(
            &conn,
            "SELECT * FROM bolt_aggregate('non_existent', 'grp', 'count(*)')",
        ),
        "is not registered in Craton Bolt",
    );
    assert_err_contains(
        query_first_chunk(&conn, "SELECT * FROM bolt_aggregate('', 'grp', 'count(*)')"),
        "bolt_aggregate table_name cannot be empty",
    );
    assert_err_contains(
        query_first_chunk(
            &conn,
            "SELECT * FROM bolt_aggregate('gpu_agg_src', 'grp', '')",
        ),
        "bolt_aggregate aggregates cannot be empty",
    );

    // 5. Report 1.1.5 (fixed): the SQL is built from the resolved catalog name, so other cases
    // and the `bolt.` / `bolt:` prefixes resolve to the same table.
    for spelling in [
        "GPU_AGG_SRC",
        "bolt.gpu_agg_src",
        "BOLT.Gpu_Agg_Src",
        "bolt:gpu_agg_src",
    ] {
        let total = query_single_i64(
            &conn,
            &format!(
                "SELECT total::BIGINT FROM bolt_aggregate('{spelling}', '', 'sum(score) AS total')"
            ),
        )?;
        assert_eq!(total, expected_total, "bolt_aggregate('{spelling}', ...)");
    }

    Ok(())
}

// SPDX-License-Identifier: Apache-2.0

//! Wave 8 (W8-A) runtime regression tests: zoned timestamps registered through the Rust API are
//! stored like `COPY` stores `TIMESTAMPTZ` (W5-C-001), the host-fallback reason is visible from
//! SQL (`bolt_pool_stats().fallback_reason`, `bolt_tables().on_gpu` / `gpu_note`; W5-C-002),
//! and host-fallback results keep DuckDB's Arrow stream batches (W4-A-002). They run on both
//! backends and branch on the reported backend where behavior legitimately differs.

use std::sync::{Arc, Mutex, Once};

use arrow_array::{
    ArrayRef, Int32Array, Int64Array, RecordBatch, TimestampMillisecondArray, cast::AsArray,
    types::Int64Type,
};
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
fn bolt_w8a_test_entrypoint(extension: &Extension) -> Result<()> {
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

/// First column of the first row as a string, `None` for SQL NULL.
fn query_opt_string(conn: &Connection, sql: &str) -> Result<Option<String>> {
    let mut result = conn.query(sql, Parameters::None)?;
    let chunk = result.next_chunk()?.expect("Expected at least one chunk");
    let vec = chunk.get_vector_at::<String>(0)?;
    Ok(vec.reader()?.get(0)?.map(|s| s.to_string()))
}

fn query_single_string(conn: &Connection, sql: &str) -> Result<String> {
    Ok(query_opt_string(conn, sql)?.expect("Expected non-null string"))
}

fn backend_is_gpu(conn: &Connection) -> Result<bool> {
    let backend = query_single_string(conn, "SELECT backend FROM bolt_pool_stats()")?;
    if cfg!(feature = "cuda-stub") {
        assert_eq!(
            backend, "host-planner-fallback",
            "cuda-stub builds cannot use the GPU"
        );
    }
    Ok(backend == "cuda-gpu")
}

fn int_batch(name: &str, values: impl IntoIterator<Item = i64>) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![Field::new(name, DataType::Int64, false)]));
    let col: ArrayRef = Arc::new(Int64Array::from_iter_values(values));
    RecordBatch::try_new(schema, vec![col]).unwrap()
}

// ------------------------------------------------------------------------------ W5-C-001

#[test]
fn zoned_timestamps_registered_from_rust_are_stored_as_naive_utc() -> Result<()> {
    let _guard = lock();
    let (_env, conn) = setup_connection()?;
    let tz = "Europe/Berlin";
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new(
                "ts",
                DataType::Timestamp(TimeUnit::Millisecond, Some(tz.into())),
                true,
            ),
        ])),
        vec![
            Arc::new(Int32Array::from(vec![1, 2])),
            Arc::new(TimestampMillisecondArray::from(vec![1_500, 500]).with_timezone(tz)),
        ],
    )
    .expect("zoned batch");
    BoltRuntime::global().register_table("w8a_tz", batch, RegisterMode::Replace)?;

    // Stored like COPY stores TIMESTAMPTZ: naive UTC microseconds with RFC-008 metadata.
    let (schema, stored) = BoltRuntime::global().get_table_batches_strict("w8a_tz")?;
    assert_eq!(
        schema.field(1).data_type(),
        &DataType::Timestamp(TimeUnit::Microsecond, None)
    );
    assert_eq!(
        schema
            .field(1)
            .metadata()
            .get("duckdb.logical_type")
            .map(String::as_str),
        Some("TIMESTAMPTZ")
    );
    let micros: Vec<i64> = stored
        .iter()
        .flat_map(|b| {
            b.column(1)
                .as_primitive::<arrow_array::types::TimestampMicrosecondType>()
                .values()
                .to_vec()
        })
        .collect();
    assert_eq!(micros, vec![1_500_000, 500_000]);

    let schema_text = query_single_string(
        &conn,
        "SELECT schema FROM bolt_tables() WHERE table_name = 'w8a_tz'",
    )?;
    assert!(schema_text.contains("ts: TIMESTAMPTZ"), "{schema_text}");
    assert_eq!(
        query_single_string(&conn, "SELECT typeof(ts) FROM bolt_scan('w8a_tz') LIMIT 1")?,
        "TIMESTAMP WITH TIME ZONE"
    );
    // A pushed TIMESTAMPTZ predicate sees the UTC instant, whatever the session time zone.
    exec(&conn, "SET TimeZone = 'Asia/Kolkata'")?;
    let result = (|| -> Result<()> {
        assert_eq!(
            query_single_i64(
                &conn,
                "SELECT count(*) FROM bolt_scan('w8a_tz') WHERE ts > TIMESTAMPTZ '1970-01-01 00:00:01+00'"
            )?,
            1
        );
        assert_eq!(
            query_single_i64(
                &conn,
                "SELECT id::BIGINT FROM bolt_scan('w8a_tz') WHERE ts = TIMESTAMPTZ '1970-01-01 00:00:01.5+00'"
            )?,
            1
        );
        Ok(())
    })();
    exec(&conn, "RESET TimeZone")?;
    result
}

// ------------------------------------------------------------------------------ W5-C-002

#[test]
fn pool_stats_reports_the_fallback_reason_last() -> Result<()> {
    let _guard = lock();
    let (_env, conn) = setup_connection()?;
    exec(&conn, "RESET bolt_memory_budget")?;
    let mut result = conn.query("SELECT * FROM bolt_pool_stats()", Parameters::None)?;
    let chunk = result.next_chunk()?.expect("bolt_pool_stats row");
    assert_eq!(chunk.col_count(), 16);
    let shown = chunk
        .get_vector_at::<String>(15)?
        .reader()?
        .get(0)?
        .map(|s| s.to_string());
    drop(result);

    let reason = BoltRuntime::global().telemetry()?.fallback_reason;
    if backend_is_gpu(&conn)? {
        // No table, no budget: everything is on the GPU.
        assert_eq!(shown, None, "{reason}");
    } else {
        // GPU-less host or cuda-stub: the cached engine-initialization error.
        let shown = shown.expect("fallback_reason must be set without a GPU");
        assert!(!shown.is_empty());
        assert_eq!(shown, reason);
    }
    // NULL, never '', so `IS NOT NULL` is the test for "something runs on the host".
    assert_eq!(
        query_single_i64(
            &conn,
            "SELECT count(*) FROM bolt_pool_stats() WHERE fallback_reason = ''"
        )?,
        0
    );
    Ok(())
}

#[test]
fn bolt_tables_reports_which_tables_the_engine_holds() -> Result<()> {
    let _guard = lock();
    let (_env, conn) = setup_connection()?;
    exec(&conn, "RESET bolt_memory_budget")?;
    BoltRuntime::global().register_table(
        "w8a_on_gpu",
        int_batch("v", 0..100),
        RegisterMode::Replace,
    )?;
    let gpu = backend_is_gpu(&conn)?;
    assert_eq!(
        query_single_i64(
            &conn,
            "SELECT (on_gpu IS NOT DISTINCT FROM true)::BIGINT FROM bolt_tables() WHERE table_name = 'w8a_on_gpu'"
        )?,
        i64::from(gpu)
    );
    assert_eq!(
        query_opt_string(
            &conn,
            "SELECT gpu_note FROM bolt_tables() WHERE table_name = 'w8a_on_gpu'"
        )?,
        None
    );
    if !gpu {
        // The reason is engine-wide: no per-table note, `on_gpu` is false (not NULL).
        assert_eq!(
            query_single_i64(
                &conn,
                "SELECT count(*) FROM bolt_tables() WHERE on_gpu IS DISTINCT FROM false"
            )?,
            0
        );
        return Ok(());
    }

    // A budget below the tables' size keeps them on the host, and says so per table and in
    // `bolt_pool_stats()`.
    exec(&conn, "SET bolt_memory_budget = 64")?;
    let result = (|| -> Result<()> {
        BoltRuntime::global().register_table(
            "w8a_over_budget",
            int_batch("v", 0..10_000),
            RegisterMode::Replace,
        )?;
        assert_eq!(
            query_single_i64(
                &conn,
                "SELECT count(*) FROM bolt_tables() WHERE table_name = 'w8a_over_budget' \
                 AND on_gpu = false AND gpu_note LIKE '%bolt_memory_budget%'"
            )?,
            1
        );
        let reason = query_single_string(&conn, "SELECT fallback_reason FROM bolt_pool_stats()")?;
        assert!(
            reason.contains("table 'w8a_over_budget' not on GPU"),
            "{reason}"
        );
        Ok(())
    })();
    exec(&conn, "RESET bolt_memory_budget")?;
    result?;
    // After the reset the engine is rebuilt with every table.
    exec(&conn, "PRAGMA bolt_reset")?;
    assert!(backend_is_gpu(&conn)?);
    assert_eq!(
        query_single_i64(
            &conn,
            "SELECT count(*) FROM bolt_tables() WHERE on_gpu IS DISTINCT FROM true OR gpu_note IS NOT NULL"
        )?,
        0
    );
    assert_eq!(
        query_opt_string(&conn, "SELECT fallback_reason FROM bolt_pool_stats()")?,
        None
    );
    Ok(())
}

// ------------------------------------------------------------------------------ W4-A-002

const ROWS: i64 = 200_000;

#[test]
fn host_fallback_results_keep_their_batches() -> Result<()> {
    let _guard = lock();
    let (_env, _conn) = setup_connection()?;
    BoltRuntime::global().register_table(
        "w8a_many",
        int_batch("v", 0..ROWS),
        RegisterMode::Replace,
    )?;
    let rt = BoltRuntime::global();

    let out = rt.execute_sql_host_fallback_batches("SELECT v FROM w8a_many ORDER BY v")?;
    assert_eq!(out.execution_tier, "Host");
    assert_eq!(out.num_rows(), ROWS as usize);
    assert!(out.batches.iter().all(|b| b.schema() == out.schema));
    assert!(out.batches.iter().all(|b| b.num_rows() > 0));
    let values: Vec<i64> = out
        .batches
        .iter()
        .flat_map(|b| b.column(0).as_primitive::<Int64Type>().values().to_vec())
        .collect();
    assert_eq!(values, (0..ROWS).collect::<Vec<_>>());

    // The single-batch API still returns the whole result.
    let single = rt.execute_sql_host_fallback("SELECT v FROM w8a_many ORDER BY v")?;
    assert_eq!(single.batch.num_rows(), ROWS as usize);
    assert_eq!(
        single.batch.column(0).as_primitive::<Int64Type>().value(0),
        0
    );

    // An empty result has a schema and no batches.
    let empty = rt.execute_sql_host_fallback_batches("SELECT v FROM w8a_many WHERE v < 0")?;
    assert_eq!(empty.num_rows(), 0);
    assert_eq!(empty.schema.fields().len(), 1);
    let empty_single = rt.execute_sql_host_fallback("SELECT v FROM w8a_many WHERE v < 0")?;
    assert_eq!(empty_single.batch.num_rows(), 0);
    assert_eq!(empty_single.batch.num_columns(), 1);
    Ok(())
}

#[test]
fn bolt_query_and_register_sql_return_every_fallback_row() -> Result<()> {
    let _guard = lock();
    let (_env, conn) = setup_connection()?;
    // One NULL: an expression over a table with a NULL is routed to the host fallback by the
    // engine guard (W1-O-002), so the queries below run on the host on every backend.
    let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, true)]));
    let values: Int64Array = (0..ROWS).map(Some).chain([None]).collect();
    let batch = RecordBatch::try_new(schema, vec![Arc::new(values) as ArrayRef]).unwrap();
    BoltRuntime::global().register_table("w8a_rows", batch, RegisterMode::Replace)?;
    let total = ROWS + 1;
    let sum = ROWS * (ROWS - 1) / 2;
    let host_sql = "SELECT v + 0 AS v FROM w8a_rows";
    assert_eq!(
        query_single_i64(
            &conn,
            &format!("SELECT count(*) FROM bolt_query('{host_sql}')")
        )?,
        total
    );
    assert_eq!(
        query_single_i64(
            &conn,
            &format!("SELECT sum(v)::BIGINT FROM bolt_query('{host_sql}')")
        )?,
        sum
    );
    exec(
        &conn,
        &format!("SELECT bolt_register_sql('w8a_copy', '{host_sql}')"),
    )?;
    assert_eq!(
        query_single_i64(
            &conn,
            "SELECT num_rows FROM bolt_tables() WHERE table_name = 'w8a_copy'"
        )?,
        total
    );
    assert_eq!(
        query_single_i64(&conn, "SELECT sum(v)::BIGINT FROM bolt_scan('w8a_copy')")?,
        sum
    );
    // An empty host result registers an empty table with the result's columns.
    exec(
        &conn,
        &format!("SELECT bolt_register_sql('w8a_empty', '{host_sql} WHERE v < 0')"),
    )?;
    assert_eq!(
        query_single_i64(
            &conn,
            "SELECT num_rows * 10 + num_columns FROM bolt_tables() WHERE table_name = 'w8a_empty'"
        )?,
        1
    );
    Ok(())
}

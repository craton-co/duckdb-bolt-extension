// SPDX-License-Identifier: Apache-2.0

//! Regression tests for the Bolt runtime (worker, host fallback, catalog naming, metrics).
//! They run on both backends (`cuda-stub` and a real GPU) and branch on the reported backend
//! where behavior legitimately differs.

use std::sync::{Arc, Barrier, Mutex, Once};

use arrow_array::{ArrayRef, Int32Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use duckdb_bolt_core::runtime::{
    BoltConfig, BoltRuntime, CRATON_BOLT_VERSION, RegisterMode, normalize_table_reference,
};
use duckdb_loadable_macros::duckdb_entrypoint_c_api_v2;
use duckdb_neo::{
    Parameters, Result,
    connection::Connection,
    environment::{Environment, StorageLocation},
    extension::{Extension, register_static_extension},
};

#[duckdb_entrypoint_c_api_v2(extension_name = "bolt", api_version = "2.0.0")]
fn bolt_w1b_test_entrypoint(extension: &Extension) -> Result<()> {
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

fn int_batch(col: &str, values: &[i32]) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![Field::new(col, DataType::Int32, true)]));
    let array: ArrayRef = Arc::new(Int32Array::from(values.to_vec()));
    RecordBatch::try_new(schema, vec![array]).unwrap()
}

fn two_col_batch(ids: &[i32], vs: &[i32]) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, true),
        Field::new("v", DataType::Int32, true),
    ]));
    let a: ArrayRef = Arc::new(Int32Array::from(ids.to_vec()));
    let b: ArrayRef = Arc::new(Int32Array::from(vs.to_vec()));
    RecordBatch::try_new(schema, vec![a, b]).unwrap()
}

fn first_i64(batch: &RecordBatch) -> i64 {
    let casted = arrow_cast::cast(batch.column(0), &DataType::Int64).unwrap();
    casted
        .as_any()
        .downcast_ref::<arrow_array::Int64Array>()
        .unwrap()
        .value(0)
}

fn gpu_backend() -> bool {
    BoltRuntime::global().telemetry().unwrap().backend == "cuda-gpu"
}

const LATERAL_SQL: &str =
    "SELECT r1.id FROM w1b_lat r1, LATERAL (SELECT r2.v FROM w1b_lat r2 WHERE r2.id = r1.id) l";

#[test]
fn lateral_query_runs_without_gpu_and_reports_host_tier() -> Result<()> {
    let _guard = lock();
    let (_env, conn) = setup_connection()?;
    let rt = BoltRuntime::global();
    rt.register_table(
        "w1b_lat",
        two_col_batch(&[1, 2, 3], &[10, 20, 30]),
        RegisterMode::Replace,
    )?;

    // Through SQL (eager bind fallback): works on both backends.
    let cnt = query_single_i64(
        &conn,
        &format!("SELECT count(*) FROM bolt_query('{LATERAL_SQL}')"),
    )?;
    assert_eq!(cnt, 3);

    // The explicit host-fallback path executes queries Bolt's raw planner rejects.
    let out = rt.execute_sql_host_fallback(LATERAL_SQL)?;
    assert_eq!(out.batch.num_rows(), 3);
    assert_eq!(out.execution_tier, "Host");

    if !gpu_backend() {
        let out = rt.execute_sql(LATERAL_SQL)?;
        assert_eq!(out.batch.num_rows(), 3);
        assert_eq!(out.execution_tier, "Host");
    }
    Ok(())
}

#[test]
fn tmp_prefixed_and_path_like_names_are_not_mangled() -> Result<()> {
    let _guard = lock();
    let (_env, conn) = setup_connection()?;
    let rt = BoltRuntime::global();

    let tmp = rt.register_table("tmp_w1b_x", int_batch("a", &[1, 2]), RegisterMode::Replace)?;
    assert_eq!(tmp.name, "tmp_w1b_x");
    let plain = rt.register_table("w1b_x", int_batch("a", &[1, 2, 3]), RegisterMode::Replace)?;
    assert_eq!(plain.name, "w1b_x");
    assert_eq!(rt.table_info("tmp_w1b_x")?.unwrap().num_rows, 2);
    assert_eq!(rt.table_info("bolt.w1b_x")?.unwrap().num_rows, 3);
    assert_eq!(
        rt.table_info("BOLT.main.TMP_W1B_X")?.unwrap().name,
        "tmp_w1b_x"
    );
    assert!(rt.table_info("w1b_x.bolt")?.is_none());

    // A path-like name must not drop the table named after its last component.
    assert!(!rt.drop_table("some/dir/w1b_x")?);
    assert!(rt.table_info("w1b_x")?.is_some());

    // SQL surface: bolt_register_sql keeps the tmp_ prefix; replacement scan resolves it.
    query_single_i64(
        &conn,
        "SELECT bolt_register_sql('tmp_w1b_sql', 'SELECT a FROM w1b_x')",
    )?;
    assert_eq!(
        query_single_i64(
            &conn,
            "SELECT count(*) FROM bolt_tables() WHERE table_name = 'tmp_w1b_sql'"
        )?,
        1
    );
    assert_eq!(
        query_single_i64(&conn, "SELECT count(*) FROM bolt.tmp_w1b_sql")?,
        3
    );
    assert_eq!(
        query_single_i64(&conn, "SELECT count(*) FROM bolt.tmp_w1b_x")?,
        2
    );

    assert_eq!(normalize_table_reference("'bolt:tmp_q'"), "tmp_q");
    Ok(())
}

#[test]
fn drop_then_query_other_tables_and_reregister() -> Result<()> {
    let _guard = lock();
    let (_env, _conn) = setup_connection()?;
    let rt = BoltRuntime::global();
    rt.register_table(
        "w1b_keep",
        int_batch("k", &[1, 2, 3, 4]),
        RegisterMode::Replace,
    )?;
    rt.register_table("w1b_gone", int_batch("g", &[7, 8]), RegisterMode::Replace)?;

    let out = rt.execute_sql("SELECT SUM(k) AS s FROM w1b_keep")?;
    assert_eq!(first_i64(&out.batch), 10);

    assert!(rt.drop_table("w1b_gone")?);
    assert!(!rt.drop_table("w1b_gone")?);

    // Remaining tables keep working after the drop.
    let out = rt.execute_sql("SELECT SUM(k) AS s FROM w1b_keep")?;
    assert_eq!(first_i64(&out.batch), 10);
    // The dropped table is unknown (not silently empty).
    assert!(rt.execute_sql("SELECT SUM(g) AS s FROM w1b_gone").is_err());
    assert!(
        rt.execute_sql_host_fallback("SELECT g FROM w1b_gone")
            .is_err()
    );

    // Re-registering the dropped name (new schema) works and shows the new data.
    rt.register_table("w1b_gone", int_batch("h", &[5, 6, 7]), RegisterMode::Create)?;
    let out = rt.execute_sql("SELECT SUM(h) AS s FROM w1b_gone")?;
    assert_eq!(first_i64(&out.batch), 18);

    // Clearing all tables keeps the runtime usable.
    assert_eq!(rt.clear_tables()?, 2);
    assert!(rt.execute_sql("SELECT SUM(k) AS s FROM w1b_keep").is_err());
    rt.register_table("w1b_keep", int_batch("k", &[2]), RegisterMode::Create)?;
    let out = rt.execute_sql("SELECT SUM(k) AS s FROM w1b_keep")?;
    assert_eq!(first_i64(&out.batch), 2);
    Ok(())
}

#[test]
fn host_fallback_resyncs_replaced_tables() -> Result<()> {
    let _guard = lock();
    let (_env, _conn) = setup_connection()?;
    let rt = BoltRuntime::global();
    rt.register_table(
        "w1b_sync",
        int_batch("x", &[1, 2, 3]),
        RegisterMode::Replace,
    )?;
    let out = rt.execute_sql_host_fallback("SELECT count(*) AS c FROM w1b_sync")?;
    assert_eq!(first_i64(&out.batch), 3);

    rt.register_table(
        "w1b_sync",
        int_batch("x", &[1, 2, 3, 4, 5]),
        RegisterMode::Replace,
    )?;
    let out = rt.execute_sql_host_fallback("SELECT count(*) AS c FROM w1b_sync")?;
    assert_eq!(first_i64(&out.batch), 5);

    rt.register_table("w1b_sync", int_batch("x", &[6]), RegisterMode::Append)?;
    let out = rt.execute_sql_host_fallback("SELECT SUM(x) AS s FROM w1b_sync")?;
    assert_eq!(first_i64(&out.batch), 21);

    // Only SELECT statements, one at a time.
    assert!(rt.execute_sql_host_fallback("DROP TABLE w1b_sync").is_err());
    assert!(rt.execute_sql_host_fallback("SELECT 1; SELECT 2").is_err());
    let out = rt.execute_sql_host_fallback("SELECT count(*) AS c FROM w1b_sync")?;
    assert_eq!(first_i64(&out.batch), 6);
    Ok(())
}

#[test]
fn host_fallback_cannot_access_files() -> Result<()> {
    let _guard = lock();
    let (_env, _conn) = setup_connection()?;
    let rt = BoltRuntime::global();

    let dir = std::env::temp_dir().join(format!("w1b_fallback_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let csv = dir.join("secret.csv");
    std::fs::write(&csv, "a,b\n1,2\n").unwrap();
    let path = csv.to_string_lossy().replace('\\', "/");

    let out = rt.execute_sql_host_fallback("SELECT 42 AS answer")?;
    assert_eq!(first_i64(&out.batch), 42);

    let read = rt.execute_sql_host_fallback(&format!("SELECT * FROM read_csv('{path}')"));
    assert!(read.is_err(), "host fallback must not read files: {read:?}");
    let read = rt.execute_sql_host_fallback(&format!("SELECT * FROM '{path}'"));
    assert!(read.is_err(), "host fallback must not read files: {read:?}");

    let ext = rt.execute_sql_host_fallback(
        "SELECT value FROM duckdb_settings() WHERE name = 'enable_external_access'",
    )?;
    let col = arrow_cast::cast(ext.batch.column(0), &DataType::Utf8).unwrap();
    let value = col
        .as_any()
        .downcast_ref::<arrow_array::StringArray>()
        .unwrap()
        .value(0)
        .to_string();
    assert_eq!(value, "false");

    let _ = std::fs::remove_dir_all(&dir);
    Ok(())
}

#[test]
fn execution_tier_never_executes_the_query() -> Result<()> {
    let _guard = lock();
    let (_env, conn) = setup_connection()?;
    let rt = BoltRuntime::global();
    rt.register_table(
        "w1b_lat",
        two_col_batch(&[1, 2, 3], &[10, 20, 30]),
        RegisterMode::Replace,
    )?;

    // Executing this query raises `boom<id>`; planning it must not.
    let sql = "SELECT r1.id, error('boom' || r1.id::VARCHAR) AS e FROM w1b_lat r1, LATERAL (SELECT r2.v FROM w1b_lat r2 WHERE r2.id = r1.id) l";
    assert!(rt.execute_sql_host_fallback(sql).is_err());

    let queries_before = query_single_i64(
        &conn,
        "SELECT value FROM bolt_metrics() WHERE metric = 'queries_total'",
    )?;
    let tier = rt.execution_tier(sql);
    match &tier {
        Ok(t) => assert!(["Gpu", "Host", "Hybrid"].contains(&t.as_str()), "{t}"),
        Err(e) => assert!(!e.to_string().contains("boom"), "query was executed: {e}"),
    }
    if !gpu_backend() {
        assert_eq!(tier?, "Host");
    }
    let queries_after = query_single_i64(
        &conn,
        "SELECT value FROM bolt_metrics() WHERE metric = 'queries_total'",
    )?;
    assert_eq!(queries_before, queries_after, "execution_tier ran a query");
    Ok(())
}

#[test]
fn metrics_include_phase_histograms() -> Result<()> {
    let _guard = lock();
    let (_env, conn) = setup_connection()?;
    assert_eq!(
        query_single_i64(
            &conn,
            "SELECT count(*) FROM bolt_metrics() WHERE category = 'histogram_ns'"
        )?,
        4
    );
    assert_eq!(
        query_single_i64(
            &conn,
            "SELECT count(*) FROM bolt_metrics() WHERE metric = 'parse_latency_count' AND category = 'histogram_count'"
        )?,
        1
    );
    assert!(
        query_single_i64(
            &conn,
            "SELECT count(*) FROM bolt_metrics() WHERE category = 'counter'"
        )? > 0
    );
    Ok(())
}

#[test]
fn worker_survives_injected_panic() -> Result<()> {
    let _guard = lock();
    let (_env, conn) = setup_connection()?;
    let rt = BoltRuntime::global();
    rt.register_table("w1b_panic", int_batch("p", &[3, 4]), RegisterMode::Replace)?;

    let err = rt
        .inject_worker_panic_for_testing()
        .expect_err("panic must surface as an error");
    assert!(err.to_string().contains("panicked"), "{err}");

    // The worker still serves requests, and the catalog survived.
    assert_eq!(rt.table_info("w1b_panic")?.unwrap().num_rows, 2);
    let out = rt.execute_sql("SELECT SUM(p) AS s FROM w1b_panic")?;
    assert_eq!(first_i64(&out.batch), 7);
    assert_eq!(
        query_single_i64(&conn, "SELECT count(*) FROM bolt.w1b_panic")?,
        2
    );
    Ok(())
}

#[test]
fn concurrent_config_updates_are_not_lost() -> Result<()> {
    let _guard = lock();
    let (_env, _conn) = setup_connection()?;
    let rt = BoltRuntime::global();
    for round in 0..25usize {
        rt.update_config(|cfg| *cfg = BoltConfig::default())?;
        let barrier = Arc::new(Barrier::new(2));
        let b1 = barrier.clone();
        let t1 = std::thread::spawn(move || {
            b1.wait();
            BoltRuntime::global().update_config(|cfg| cfg.memory_budget = Some(1 << 30))
        });
        let b2 = barrier.clone();
        let t2 = std::thread::spawn(move || {
            b2.wait();
            BoltRuntime::global().update_config(|cfg| cfg.enable_tracing = true)
        });
        t1.join().unwrap()?;
        t2.join().unwrap()?;
        let cfg = rt.config()?;
        assert_eq!(
            cfg.memory_budget,
            Some(1 << 30),
            "round {round}: lost update"
        );
        assert!(cfg.enable_tracing, "round {round}: lost update");
    }
    rt.update_config(|cfg| *cfg = BoltConfig::default())?;
    rt.reset_engine()?;
    Ok(())
}

#[test]
fn version_and_reset_do_not_misreport() -> Result<()> {
    let _guard = lock();
    let (_env, conn) = setup_connection()?;
    let version = query_single_string(&conn, "SELECT bolt_version()")?;
    assert!(
        version.contains(&format!("craton-bolt v{CRATON_BOLT_VERSION}")),
        "{version}"
    );
    // Reset succeeds on GPU-less (cuda-stub) builds and on a healthy GPU.
    conn.execute("PRAGMA bolt_reset", Parameters::None)?;
    let version = query_single_string(&conn, "SELECT bolt_version()")?;
    let backend = query_single_string(&conn, "SELECT backend FROM bolt_pool_stats()")?;
    assert!(version.contains(&format!("backend={backend}")), "{version}");

    // A cached engine failure does not stop the host fallback from answering.
    let rt = BoltRuntime::global();
    rt.register_table("w1b_ver", int_batch("z", &[1]), RegisterMode::Replace)?;
    let out = rt.execute_sql("SELECT SUM(z) AS s FROM w1b_ver")?;
    assert_eq!(first_i64(&out.batch), 1);
    Ok(())
}

#[test]
fn scratch_table_is_engine_only_and_reusable() -> Result<()> {
    use duckdb_bolt_core::runtime::SCRATCH_TABLE_NAME;
    let _guard = lock();
    let (_env, _conn) = setup_connection()?;
    let rt = BoltRuntime::global();
    assert!(
        rt.register_table(
            SCRATCH_TABLE_NAME,
            int_batch("a", &[1]),
            RegisterMode::Replace
        )
        .is_err()
    );

    let sql = format!("SELECT SUM(a) AS s FROM {SCRATCH_TABLE_NAME}");
    let first = rt.execute_sql_with_scratch_table(int_batch("a", &[1, 2, 3]), &sql);
    if gpu_backend() {
        assert_eq!(first_i64(&first?.batch), 6);
        // Second call with a different schema reuses the scratch slot.
        let sql2 = format!("SELECT SUM(b) AS s FROM {SCRATCH_TABLE_NAME}");
        let second = rt.execute_sql_with_scratch_table(int_batch("b", &[10, 20]), &sql2)?;
        assert_eq!(first_i64(&second.batch), 30);
    } else {
        assert!(first.is_err(), "scratch execution requires a GPU engine");
    }
    assert!(
        rt.list_tables()?
            .iter()
            .all(|t| t.name != SCRATCH_TABLE_NAME)
    );
    assert!(rt.table_info(SCRATCH_TABLE_NAME)?.is_none());
    Ok(())
}

// SPDX-License-Identifier: Apache-2.0

//! Wave 3 (W3-A) runtime regression tests: deep predicates (W2-B-001), coalesced engine uploads
//! (W2-O-001), panic recovery (W2-E-006), reserved / unaddressable names (W2-E-008, W2-E-015),
//! execution tier (W2-E-012, W2-A-002), memory budget (W2-E-011), PTX cache override
//! (W2-E-009), request timeouts (W2-E-005), RFC-008 append checks (W2-C-002), and snapshot
//! reads (W1-B-004, W1-B-008). They run on both backends and branch on the reported backend
//! where behavior legitimately differs.

use std::{
    sync::{Arc, Mutex, Once},
    time::{Duration, Instant},
};

use arrow_array::{ArrayRef, Int32Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use duckdb_bolt_core::runtime::{BoltRuntime, RegisterMode};
use duckdb_loadable_macros::duckdb_entrypoint_c_api_v2;
use duckdb_neo::{
    Parameters, Result,
    connection::Connection,
    environment::{Environment, StorageLocation},
    extension::{Extension, register_static_extension},
};

#[duckdb_entrypoint_c_api_v2(extension_name = "bolt", api_version = "2.0.0")]
fn bolt_w3a_test_entrypoint(extension: &Extension) -> Result<()> {
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

fn backend_is_gpu(conn: &Connection) -> Result<bool> {
    Ok(query_single_string(conn, "SELECT backend FROM bolt_pool_stats()")? == "cuda-gpu")
}

fn id_name_batch(start: i64, len: i64) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("s", DataType::Utf8, false),
    ]));
    let ids: ArrayRef = Arc::new(Int64Array::from_iter_values(start..start + len));
    let names: ArrayRef = Arc::new(StringArray::from_iter_values(
        (start..start + len).map(|i| format!("name_{i}")),
    ));
    RecordBatch::try_new(schema, vec![ids, names]).unwrap()
}

/// Register `rows` rows of `(id BIGINT, s VARCHAR)` as `name`, in chunks of `chunk` rows.
fn register_id_name(name: &str, rows: i64, chunk: i64) -> Result<()> {
    let batches: Vec<RecordBatch> = (0..rows)
        .step_by(chunk as usize)
        .map(|start| id_name_batch(start, chunk.min(rows - start)))
        .collect();
    let schema = batches[0].schema();
    BoltRuntime::global().register_table_batches(name, schema, batches, RegisterMode::Replace)?;
    Ok(())
}

// ------------------------------------------------------------------------------ W2-B-001

#[test]
fn deep_or_chains_do_not_crash_the_process() -> Result<()> {
    let _g = lock();
    let (_env, conn) = setup_connection()?;
    register_id_name("w3a_deep", 5_000, 2_048)?;
    // 300 terms killed the process with the default 2 MiB worker stack; 1,000 is far beyond.
    for terms in [300usize, 1_000] {
        let predicate = (0..terms)
            .map(|i| format!("id = {i}"))
            .collect::<Vec<_>>()
            .join(" OR ");
        let sql = format!("SELECT count(*) AS n FROM w3a_deep WHERE {predicate}");
        let n = query_single_i64(
            &conn,
            &format!(
                "SELECT n::BIGINT FROM bolt_query('{}')",
                sql.replace('\'', "''")
            ),
        )?;
        assert_eq!(n, terms as i64, "{terms}-term OR");
    }
    // Above the term budget the query is routed to the host fallback (or rejected cleanly),
    // never a stack overflow.
    let predicate = (0..5_000)
        .map(|i| format!("id = {i}"))
        .collect::<Vec<_>>()
        .join(" OR ");
    let sql = format!("SELECT count(*) AS n FROM w3a_deep WHERE {predicate}");
    match BoltRuntime::global().execute_sql(&sql) {
        Ok(out) => {
            assert_eq!(out.batch.num_rows(), 1);
            assert_eq!(out.execution_tier, "Host");
        }
        Err(e) => println!("5,000-term OR rejected cleanly: {e}"),
    }
    Ok(())
}

#[test]
fn high_cardinality_string_predicates_do_not_crash_the_process() -> Result<()> {
    let _g = lock();
    let (_env, conn) = setup_connection()?;
    // The W2-B-001 repro: 5,000 distinct names, `s > 'name_4'` matched ~4,400 dictionary
    // entries and overflowed the worker stack.
    register_id_name("w3a_card", 5_000, 2_048)?;
    exec(
        &conn,
        "CREATE TABLE w3a_card_ref AS SELECT i AS id, 'name_' || i::VARCHAR AS s FROM range(0, 5000) t(i)",
    )?;
    for predicate in [
        "s > 'name_4'",
        "s LIKE 'name_1%'",
        "s <= 'name_2'",
        "'name_3' < s",
    ] {
        let bolt = query_single_i64(
            &conn,
            &format!(
                "SELECT n::BIGINT FROM bolt_query('SELECT count(*) AS n FROM w3a_card WHERE {}')",
                predicate.replace('\'', "''")
            ),
        )?;
        let duck = query_single_i64(
            &conn,
            &format!("SELECT count(*)::BIGINT FROM w3a_card_ref WHERE {predicate}"),
        )?;
        assert_eq!(bolt, duck, "{predicate}");
    }
    Ok(())
}

// ------------------------------------------------------------------------------ W2-O-001

#[test]
fn multi_batch_registration_is_coalesced_for_the_engine() -> Result<()> {
    let _g = lock();
    let (_env, conn) = setup_connection()?;
    // 300,000 rows in 147 batches took 91 s on an RTX 2060 with one `register_batch` per
    // batch (W2-O-001).
    let started = Instant::now();
    register_id_name("w3a_many", 300_000, 2_048)?;
    let elapsed = started.elapsed();
    println!(
        "register_table_batches(300,000 rows / 147 batches) on {}: {elapsed:?}",
        query_single_string(&conn, "SELECT backend FROM bolt_pool_stats()")?
    );
    let (_, stored) = BoltRuntime::global().get_table_batches_strict("w3a_many")?;
    assert_eq!(
        stored.len(),
        147,
        "the shadow catalog keeps the original batches"
    );
    assert_eq!(
        query_single_i64(
            &conn,
            "SELECT n::BIGINT FROM bolt_query('SELECT count(*) AS n FROM w3a_many WHERE id >= 150000')"
        )?,
        150_000
    );
    // Append: one coalesced engine call for the new batches.
    let extra: Vec<RecordBatch> = (0..10)
        .map(|i| id_name_batch(300_000 + i * 100, 100))
        .collect();
    let schema = extra[0].schema();
    BoltRuntime::global().register_table_batches(
        "w3a_many",
        schema,
        extra,
        RegisterMode::Append,
    )?;
    assert_eq!(
        query_single_i64(
            &conn,
            "SELECT n::BIGINT FROM bolt_query('SELECT count(*) AS n FROM w3a_many WHERE id >= 300000')"
        )?,
        1_000
    );
    assert!(
        elapsed < Duration::from_secs(30),
        "multi-batch registration took {elapsed:?}"
    );
    Ok(())
}

// ------------------------------------------------------------------------------ W2-E-006

#[test]
fn a_panic_during_panic_recovery_keeps_the_worker_alive() -> Result<()> {
    let _g = lock();
    let (_env, conn) = setup_connection()?;
    let rt = BoltRuntime::global();
    let err = rt
        .inject_worker_panic_in_recovery_for_testing()
        .expect_err("the injected panic must surface as an error");
    assert!(err.message.contains("panicked"), "{err}");
    // The worker survived both panics: requests that need it still work.
    assert!(rt.version_info()?.contains("duckdb-bolt-extension"));
    exec(
        &conn,
        "COPY (SELECT 41 AS v) TO 'w3a_after_panic' (FORMAT bolt, USE_TMP_FILE FALSE)",
    )?;
    assert_eq!(
        query_single_i64(
            &conn,
            "SELECT (v + 1)::BIGINT FROM bolt_query('SELECT v FROM w3a_after_panic')"
        )?,
        42
    );
    Ok(())
}

// --------------------------------------------------------------------- W2-E-008, W2-E-015

#[test]
fn reserved_and_unaddressable_names_are_rejected() -> Result<()> {
    let _g = lock();
    let (_env, conn) = setup_connection()?;
    let rt = BoltRuntime::global();
    for name in [
        "__bolt_mirror_source",
        "__BOLT_scratch",
        "__bolt_scan_filter_tmp_1",
    ] {
        let batch = id_name_batch(0, 3);
        let err = rt
            .register_table(name, batch, RegisterMode::Replace)
            .expect_err("reserved name");
        assert!(err.message.contains("reserved"), "{name}: {err}");
    }
    let err = conn
        .execute(
            "COPY (SELECT 1 AS a) TO '__bolt_mirror_source' (FORMAT bolt, USE_TMP_FILE FALSE)",
            Parameters::None,
        )
        .expect_err("COPY to a reserved name");
    assert!(err.message.contains("reserved"), "{err}");

    // Duplicate and empty column names cannot be mirrored by the host
    // fallback or addressed in SQL: rejected on every backend.
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Int32, true),
        Field::new("x", DataType::Int32, true),
    ]));
    let cols: Vec<ArrayRef> = vec![
        Arc::new(Int32Array::from(vec![1])),
        Arc::new(Int32Array::from(vec![2])),
    ];
    let err = rt
        .register_table(
            "w3a_dup",
            RecordBatch::try_new(schema, cols).unwrap(),
            RegisterMode::Replace,
        )
        .expect_err("duplicate column names");
    assert!(err.message.contains("duplicate column name"), "{err}");
    // DuckDB itself renames duplicate SELECT names before COPY sees them; whatever it
    // registers must stay usable on the host fallback.
    if conn
        .execute(
            "COPY (SELECT 1 AS x, 2 AS X) TO 'w3a_dup' (FORMAT bolt, USE_TMP_FILE FALSE)",
            Parameters::None,
        )
        .is_ok()
    {
        assert_eq!(
            query_single_i64(
                &conn,
                "SELECT count(*)::BIGINT FROM bolt_query('SELECT * FROM w3a_dup')"
            )?,
            1
        );
    }
    let schema = Arc::new(Schema::new(vec![Field::new("", DataType::Int32, true)]));
    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(Int32Array::from(vec![1])) as ArrayRef],
    )
    .unwrap();
    let err = rt
        .register_table("w3a_empty_col", batch, RegisterMode::Replace)
        .expect_err("empty column name");
    assert!(err.message.contains("must not be empty"), "{err}");

    // A one-letter table name no longer matches every query that contains the letter, and
    // the other queries keep working.
    exec(
        &conn,
        "COPY (SELECT 7 AS a) TO 't' (FORMAT bolt, USE_TMP_FILE FALSE)",
    )?;
    exec(
        &conn,
        "COPY (SELECT 8 AS a) TO 'w3a_other' (FORMAT bolt, USE_TMP_FILE FALSE)",
    )?;
    assert_eq!(
        query_single_i64(
            &conn,
            "SELECT a::BIGINT FROM bolt_query('SELECT a FROM w3a_other')"
        )?,
        8
    );
    assert_eq!(
        query_single_i64(&conn, "SELECT a::BIGINT FROM bolt_query('SELECT a FROM t')")?,
        7
    );
    exec(&conn, "SELECT bolt_drop_table('t')")?;
    Ok(())
}

// --------------------------------------------------------------------- W2-E-012, W2-A-002

#[test]
fn execution_tier_reports_where_the_query_runs() -> Result<()> {
    let _g = lock();
    let (_env, conn) = setup_connection()?;
    exec(
        &conn,
        "COPY (SELECT i::INTEGER AS id, CASE WHEN i % 7 = 0 THEN NULL ELSE i::DOUBLE END AS d \
         FROM range(0, 100) t(i)) TO 'w3a_tier_nulls' (FORMAT bolt, USE_TMP_FILE FALSE)",
    )?;
    exec(
        &conn,
        "COPY (SELECT i::INTEGER AS id FROM range(0, 100) t(i)) TO 'w3a_tier_clean' \
         (FORMAT bolt, USE_TMP_FILE FALSE)",
    )?;
    let tier = |sql: &str| {
        query_single_string(
            &conn,
            &format!("SELECT bolt_execution_tier('{}')", sql.replace('\'', "''")),
        )
    };
    let guarded = "SELECT count(*) FROM w3a_tier_nulls WHERE d > 0";
    let clean = "SELECT id FROM w3a_tier_clean WHERE id > 10";
    let out = BoltRuntime::global().execute_sql(guarded)?;
    if backend_is_gpu(&conn)? {
        // The crash guard (W1-O-002) sends the filtered aggregate over NULLs to the host.
        assert_eq!(tier(guarded)?, "Host");
        assert_eq!(out.execution_tier, "Host");
        assert_ne!(tier(clean)?, "Host");
        let explain = query_single_string(
            &conn,
            &format!("SELECT bolt_explain('{}')", guarded.replace('\'', "''")),
        )?;
        assert!(explain.contains("host fallback"), "{explain}");
    } else {
        // Without a GPU engine every query runs on the embedded DuckDB.
        assert_eq!(tier(guarded)?, "Host");
        assert_eq!(tier(clean)?, "Host");
        assert_eq!(out.execution_tier, "Host");
        // W1-B-007: shapes Bolt's planner rejects are explained by DuckDB.
        let explain = query_single_string(
            &conn,
            "SELECT bolt_explain('SELECT r1.id FROM w3a_tier_clean r1, LATERAL (SELECT r2.id AS x FROM w3a_tier_clean r2 WHERE r2.id = r1.id) l')",
        )?;
        assert!(explain.contains("Host fallback (DuckDB) plan"), "{explain}");
    }
    Ok(())
}

// ------------------------------------------------------------------------------ W2-E-011

#[test]
fn memory_budget_keeps_oversized_tables_on_the_host() -> Result<()> {
    let _g = lock();
    let (_env, conn) = setup_connection()?;
    exec(&conn, "SET bolt_memory_budget = 4096")?;
    let result = (|| -> Result<()> {
        register_id_name("w3a_budget", 20_000, 2_048)?;
        // Correct results on every backend: the table is served by the host fallback.
        assert_eq!(
            query_single_i64(
                &conn,
                "SELECT n::BIGINT FROM bolt_query('SELECT count(*) AS n FROM w3a_budget WHERE id < 100')"
            )?,
            100
        );
        if backend_is_gpu(&conn)? {
            let reason = BoltRuntime::global().telemetry()?.fallback_reason;
            assert!(reason.contains("bolt_memory_budget"), "{reason}");
            assert_eq!(
                BoltRuntime::global().execution_tier("SELECT id FROM w3a_budget WHERE id < 100")?,
                "Host"
            );
        }
        Ok(())
    })();
    exec(&conn, "RESET bolt_memory_budget")?;
    result
}

// ------------------------------------------------------------------------------ W2-E-009

#[test]
fn empty_ptx_cache_dir_clears_the_process_override() -> Result<()> {
    let _g = lock();
    let (_env, conn) = setup_connection()?;
    let dir = std::env::temp_dir().join(format!("bolt_w3a_ptx_{}", std::process::id()));
    // A directory installed earlier (by a previous `SET bolt_ptx_cache_dir`, or anything else
    // in the process) must not survive an engine rebuild with an empty setting.
    craton_bolt::jit::set_disk_ptx_cache_dir(Some(dir.clone()));
    exec(&conn, "RESET bolt_ptx_cache_dir")?;
    // The rebuild may fail on GPU-less hosts; the override is cleared before building.
    let _ = conn.execute("PRAGMA bolt_reset", Parameters::None);
    assert_eq!(craton_bolt::jit::current_disk_ptx_cache_dir(), None);
    let _ = std::fs::remove_dir_all(&dir);
    Ok(())
}

// ------------------------------------------------------------------------------ W2-E-005

#[test]
fn request_timeout_interrupts_host_fallback_queries() -> Result<()> {
    let _g = lock();
    let (_env, conn) = setup_connection()?;
    exec(
        &conn,
        "COPY (SELECT range::INTEGER AS i FROM range(0, 100000)) TO 'w3a_slow' (FORMAT bolt, USE_TMP_FILE FALSE)",
    )?;
    let rt = BoltRuntime::global();
    exec(&conn, "PRAGMA bolt_request_timeout(300)")?;
    assert_eq!(rt.request_timeout(), Some(Duration::from_millis(300)));
    let started = Instant::now();
    // A 10^10-row join: minutes without the timeout.
    let res = rt.execute_sql_host_fallback(
        "SELECT count(*) FROM w3a_slow a, w3a_slow b WHERE (a.i * 7 + b.i) % 1000003 = -5",
    );
    exec(&conn, "PRAGMA bolt_request_timeout(0)")?;
    let err = res.expect_err("the slow query must time out");
    assert!(err.message.contains("timed out"), "{err}");
    assert!(started.elapsed() < Duration::from_secs(20));
    assert_eq!(rt.request_timeout(), None);
    // The interrupted query releases the worker: the next request completes.
    assert!(rt.version_info()?.contains("craton-bolt"));
    assert!(
        conn.execute("PRAGMA bolt_request_timeout(-1)", Parameters::None)
            .is_err()
    );
    Ok(())
}

// ------------------------------------------------------------------------------ W2-C-002

#[test]
fn append_honors_the_recorded_original_type() -> Result<()> {
    let _g = lock();
    let (_env, conn) = setup_connection()?;
    exec(
        &conn,
        "COPY (SELECT 1::SMALLINT AS v) TO 'w3a_app' (FORMAT bolt, USE_TMP_FILE FALSE)",
    )?;
    let err = conn
        .execute(
            "COPY (SELECT 70000 AS v) TO 'w3a_app' (FORMAT bolt, MODE 'append', USE_TMP_FILE FALSE)",
            Parameters::None,
        )
        .expect_err("70000 does not fit the SMALLINT the column was copied from");
    assert!(err.message.contains("SMALLINT"), "{err}");
    exec(
        &conn,
        "COPY (SELECT 2 AS v) TO 'w3a_app' (FORMAT bolt, MODE 'append', USE_TMP_FILE FALSE)",
    )?;
    assert_eq!(
        query_single_i64(&conn, "SELECT count(*)::BIGINT FROM bolt_scan('w3a_app')")?,
        2
    );
    Ok(())
}

// --------------------------------------------------------------------- W1-B-004, W1-B-008

#[test]
fn table_batches_are_served_from_the_snapshot() -> Result<()> {
    let _g = lock();
    let (_env, _conn) = setup_connection()?;
    let rt = BoltRuntime::global();
    rt.register_table("w3a_snap", id_name_batch(0, 10), RegisterMode::Replace)?;
    let (schema, batches) = rt.get_table_batches_strict("w3a_snap")?;
    assert_eq!(schema.fields().len(), 2);
    assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 10);
    // COPY paths are cleaned; plain references are not (W1-B-008).
    assert!(
        rt.get_table_batches_for_copy("dir/tmp_w3a_snap.bolt")
            .is_ok()
    );
    assert!(rt.get_table_batches_strict("dir/w3a_snap").is_err());
    rt.register_table("w3a_snap", id_name_batch(10, 5), RegisterMode::Append)?;
    let (_, batches) = rt.get_table_batches_strict("W3A_SNAP")?;
    assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 15);
    assert!(rt.drop_table("w3a_snap")?);
    assert!(rt.get_table_batches_strict("w3a_snap").is_err());
    // Metrics and devices no longer need the worker.
    let (rows, _) = rt.metrics()?;
    assert!(!rows.is_empty());
    let _ = rt.devices()?;
    Ok(())
}

// ------------------------------------------------------------------------------ W3-E-002

#[test]
fn query_generated_nulls_and_escaped_names_are_guarded() -> Result<()> {
    let _g = lock();
    let (_env, conn) = setup_connection()?;
    // No table stores a NULL; the outer join produces them.
    exec(
        &conn,
        "COPY (SELECT range::INTEGER AS k FROM range(4)) TO 'w3a_gl' (FORMAT bolt, USE_TMP_FILE FALSE)",
    )?;
    exec(
        &conn,
        "COPY (SELECT 1::INTEGER AS k, 5.0::DOUBLE AS v) TO 'w3a_gr' (FORMAT bolt, USE_TMP_FILE FALSE)",
    )?;
    let one = |sql: &str| {
        query_single_i64(
            &conn,
            &format!(
                "SELECT n::BIGINT FROM bolt_query('{}')",
                sql.replace('\'', "''")
            ),
        )
    };
    assert_eq!(
        one("SELECT count(*) AS n FROM w3a_gl l LEFT JOIN w3a_gr r ON l.k = r.k WHERE r.v < 1")?,
        0
    );
    assert_eq!(
        one(
            "SELECT count(w) AS n FROM (SELECT r.v + 1 AS w FROM w3a_gl l LEFT JOIN w3a_gr r ON l.k = r.k) s"
        )?,
        1
    );
    // A table whose name needs escaping in SQL is still recognized by the guard.
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("d", DataType::Float64, true),
    ]));
    let cols: Vec<ArrayRef> = vec![
        Arc::new(Int32Array::from(vec![1, 2])),
        Arc::new(arrow_array::Float64Array::from(vec![None, Some(3.0)])),
    ];
    BoltRuntime::global().register_table(
        "w3a_a\"b",
        RecordBatch::try_new(schema, cols).unwrap(),
        RegisterMode::Replace,
    )?;
    assert_eq!(
        one("SELECT count(e) AS n FROM (SELECT d + 1 AS e FROM \"w3a_a\"\"b\") s")?,
        1
    );
    Ok(())
}

// ------------------------------------------------------------------------------ W3-E-004

#[test]
fn bolt_calls_inside_host_fallback_queries_fail_fast() -> Result<()> {
    let _g = lock();
    let (_env, conn) = setup_connection()?;
    exec(
        &conn,
        "COPY (SELECT 1 AS id) TO 'w3a_re' (FORMAT bolt, USE_TMP_FILE FALSE)",
    )?;
    // Backstop: never hang the test binary.
    exec(&conn, "PRAGMA bolt_request_timeout(60000)")?;
    let started = Instant::now();
    let results = [
        conn.execute(
            "SELECT * FROM bolt_query('SELECT bolt_version()')",
            Parameters::None,
        ),
        conn.execute(
            "SELECT * FROM bolt_query('SELECT * FROM bolt.w3a_re')",
            Parameters::None,
        ),
        conn.execute(
            "SELECT * FROM bolt_query('SELECT * FROM bolt_scan(''w3a_re'')')",
            Parameters::None,
        ),
    ];
    exec(&conn, "PRAGMA bolt_request_timeout(0)")?;
    assert!(started.elapsed() < Duration::from_secs(30));
    for r in results {
        if let Err(e) = r {
            assert!(!e.message.contains("timed out"), "{e}");
        }
    }
    // The worker is still responsive.
    assert!(
        BoltRuntime::global()
            .version_info()?
            .contains("craton-bolt")
    );
    Ok(())
}

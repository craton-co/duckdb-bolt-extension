// SPDX-License-Identifier: Apache-2.0

//! Wave-2 runtime regressions: the engine crash guard (W1-O-002), type normalization on
//! registration (W1-O-001), multi-batch registration (W1-C-001), name-matched checked append
//! coercion (W1-E-001), host retry of engine errors (W1-B-002), `backend_is_gpu` (W1-A-005),
//! case-insensitive `bolt.` prefixes (W1-A-007), and strict bare-name replacement scans
//! (W1-E-005). They run on both backends (`cuda-stub` and a real GPU).

use std::sync::{Arc, Mutex, Once};

use arrow_array::{ArrayRef, Int16Array, Int32Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use duckdb_bolt_core::runtime::{BoltRuntime, RegisterMode, clean_table_identifier};
use duckdb_loadable_macros::duckdb_entrypoint_c_api_v2;
use duckdb_neo::{
    Parameters, Result,
    connection::Connection,
    environment::{Environment, StorageLocation},
    extension::{Extension, register_static_extension},
};

#[duckdb_entrypoint_c_api_v2(extension_name = "bolt", api_version = "2.0.0")]
fn bolt_w2a_test_entrypoint(extension: &Extension) -> Result<()> {
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

/// `1` when `bolt_query(bolt_sql)` returns the same single value as `duck_sql` on DuckDB.
fn same_as_duckdb(conn: &Connection, bolt_sql: &str, duck_sql: &str) -> Result<i64> {
    let escaped = bolt_sql.replace('\'', "''");
    query_single_i64(
        conn,
        &format!(
            "SELECT CASE WHEN (SELECT * FROM bolt_query('{escaped}')) IS NOT DISTINCT FROM \
             ({duck_sql}) THEN 1 ELSE 0 END::BIGINT"
        ),
    )
}

/// The W1-O-002 repro: a 100-row single nullable column `d`, every 7th row NULL.
const CRASH_QUERIES: [&str; 7] = [
    "SELECT COUNT(*) FROM {t} WHERE d > 0",
    "SELECT SUM(d) FROM {t} WHERE d IS NULL",
    "SELECT COUNT(*) FROM {t} WHERE d IS NOT NULL AND d > 0",
    "SELECT COUNT(*) FROM {t} WHERE d IS NULL",
    "SELECT SUM(d) FROM {t} WHERE d > 10",
    "SELECT COUNT(*) FROM (SELECT d FROM {t} WHERE d > 0) s",
    "SELECT MAX(d) FROM {t} WHERE d < 50",
];

#[test]
fn filtered_aggregates_over_nulls_do_not_crash_and_match_duckdb() -> Result<()> {
    let _guard = lock();
    let (_env, conn) = setup_connection()?;
    for ty in ["DOUBLE", "FLOAT", "INTEGER", "BIGINT"] {
        let src = format!("w2a_src_{}", ty.to_ascii_lowercase());
        let gpu = format!("w2a_gpu_{}", ty.to_ascii_lowercase());
        exec(
            &conn,
            &format!(
                "CREATE OR REPLACE TABLE {src} AS SELECT CASE WHEN i % 7 = 0 THEN NULL \
                 ELSE (i - 20)::{ty} END AS d FROM range(100) r(i)"
            ),
        )?;
        exec(
            &conn,
            &format!("COPY {src} TO '{gpu}' (FORMAT bolt, USE_TMP_FILE FALSE)"),
        )?;
        for q in CRASH_QUERIES {
            let bolt_sql = q.replace("{t}", &gpu);
            let duck_sql = q.replace("{t}", &src);
            assert_eq!(
                same_as_duckdb(&conn, &bolt_sql, &duck_sql)?,
                1,
                "{ty}: {bolt_sql}"
            );
        }
        // Shapes that are safe on the GPU still work (and agree with DuckDB).
        for q in [
            "SELECT COUNT(*) FROM {t}",
            "SELECT SUM(d) FROM {t}",
            "SELECT COUNT(d) FROM {t}",
        ] {
            assert_eq!(
                same_as_duckdb(&conn, &q.replace("{t}", &gpu), &q.replace("{t}", &src))?,
                1,
                "{ty}: {q}"
            );
        }
        assert_eq!(
            query_single_i64(
                &conn,
                &format!("SELECT count(*) FROM bolt_query('SELECT d FROM {gpu} WHERE d > 0')")
            )?,
            query_single_i64(&conn, &format!("SELECT count(*) FROM {src} WHERE d > 0"))?
        );
        // craton-bolt 0.9.0 drops validity in filters and computed columns on the GPU;
        // the guard must keep these correct too.
        for (bolt_q, duck_q) in [
            (
                format!(
                    "SELECT count(*) FROM bolt_query('SELECT d FROM {gpu} WHERE d IS NULL') WHERE d IS NULL"
                ),
                format!("SELECT count(*) FROM {src} WHERE d IS NULL"),
            ),
            (
                format!(
                    "SELECT count(*) FROM bolt_query('SELECT d + 1 AS e FROM {gpu}') WHERE e IS NULL"
                ),
                format!("SELECT count(*) FROM {src} WHERE d IS NULL"),
            ),
            (
                format!("SELECT count(*) FROM bolt_scan('{gpu}') WHERE d > 0"),
                format!("SELECT count(*) FROM {src} WHERE d > 0"),
            ),
            (
                format!(
                    "SELECT count(*) FROM bolt_query('SELECT CASE WHEN d > 0 THEN 1 ELSE 0 END AS c FROM {gpu}') WHERE c = 1"
                ),
                format!("SELECT count(*) FROM {src} WHERE d > 0"),
            ),
        ] {
            assert_eq!(
                query_single_i64(&conn, &bolt_q)?,
                query_single_i64(&conn, &duck_q)?,
                "{ty}: {bolt_q}"
            );
        }
    }
    Ok(())
}

#[test]
fn null_free_tables_and_appended_nulls_are_tracked() -> Result<()> {
    let _guard = lock();
    let (_env, conn) = setup_connection()?;
    exec(
        &conn,
        "COPY (SELECT (i - 20)::DOUBLE AS d FROM range(100) r(i)) TO 'w2a_nonull' \
         (FORMAT bolt, USE_TMP_FILE FALSE)",
    )?;
    assert_eq!(
        query_single_i64(
            &conn,
            "SELECT * FROM bolt_query('SELECT COUNT(*) FROM w2a_nonull WHERE d > 0')"
        )?,
        79
    );
    // An append that brings NULLs must re-arm the guard for the same table.
    exec(
        &conn,
        "COPY (SELECT NULL::DOUBLE AS d FROM range(5)) TO 'w2a_nonull' \
         (FORMAT bolt, MODE 'append', USE_TMP_FILE FALSE)",
    )?;
    assert_eq!(
        query_single_i64(
            &conn,
            "SELECT * FROM bolt_query('SELECT COUNT(*) FROM w2a_nonull WHERE d > 0')"
        )?,
        79
    );
    assert_eq!(
        query_single_i64(
            &conn,
            "SELECT * FROM bolt_query('SELECT COUNT(*) FROM w2a_nonull WHERE d IS NULL')"
        )?,
        5
    );
    Ok(())
}

fn schema(fields: Vec<(&str, DataType)>) -> Arc<Schema> {
    Arc::new(Schema::new(
        fields
            .into_iter()
            .map(|(n, t)| Field::new(n, t, true))
            .collect::<Vec<_>>(),
    ))
}

#[test]
fn register_paths_normalize_types_on_every_backend() -> Result<()> {
    let _guard = lock();
    let (_env, conn) = setup_connection()?;
    let rt = BoltRuntime::global();
    let s = schema(vec![("a", DataType::Int16)]);
    let batch = RecordBatch::try_new(
        s.clone(),
        vec![Arc::new(Int16Array::from(vec![1, 2, 3])) as ArrayRef],
    )
    .unwrap();
    let summary = rt.register_table("w2a_i16", batch.clone(), RegisterMode::Replace)?;
    assert_eq!(summary.schema.field(0).data_type(), &DataType::Int32);
    assert_eq!(
        query_single_i64(
            &conn,
            "SELECT columns(*)::BIGINT FROM bolt_query('SELECT SUM(a) FROM w2a_i16')"
        )?,
        6
    );

    let summary = rt.register_table_batches(
        "w2a_i16b",
        s,
        vec![batch.clone(), batch],
        RegisterMode::Create,
    )?;
    assert_eq!(summary.schema.field(0).data_type(), &DataType::Int32);
    assert_eq!(summary.num_rows, 6);

    // Unsupported types fail the same way everywhere.
    let s = schema(vec![(
        "l",
        DataType::List(Arc::new(Field::new("item", DataType::Int32, true))),
    )]);
    assert!(
        rt.register_table_batches("w2a_list", s, vec![], RegisterMode::Replace)
            .is_err()
    );
    Ok(())
}

#[test]
fn register_table_batches_keeps_batches_and_modes() -> Result<()> {
    let _guard = lock();
    let (_env, conn) = setup_connection()?;
    let rt = BoltRuntime::global();
    let s = schema(vec![("id", DataType::Int64), ("v", DataType::Utf8)]);
    let mk = |ids: Vec<i64>, vs: Vec<&str>| {
        RecordBatch::try_new(
            s.clone(),
            vec![
                Arc::new(Int64Array::from(ids)) as ArrayRef,
                Arc::new(StringArray::from(vs)) as ArrayRef,
            ],
        )
        .unwrap()
    };
    let b1 = mk(vec![1, 2], vec!["a", "b"]);
    let b2 = mk(vec![3], vec!["c"]);
    let empty = RecordBatch::new_empty(s.clone());

    let summary = rt.register_table_batches(
        "w2a_multi",
        s.clone(),
        vec![b1.clone(), empty.clone(), b2.clone()],
        RegisterMode::Create,
    )?;
    assert_eq!(summary.num_rows, 3);
    let (_, stored) = rt.get_table_batches("w2a_multi")?;
    assert_eq!(stored.len(), 2, "batches are stored without concatenation");
    assert_eq!(
        query_single_i64(
            &conn,
            "SELECT columns(*)::BIGINT FROM bolt_query('SELECT SUM(id) FROM w2a_multi')"
        )?,
        6
    );
    assert_eq!(
        query_single_i64(&conn, "SELECT count(*) FROM w2a_multi WHERE v = 'c'")?,
        1
    );

    // Create on an existing name fails and leaves the table unchanged.
    assert!(
        rt.register_table_batches(
            "W2A_MULTI",
            s.clone(),
            vec![b2.clone()],
            RegisterMode::Create
        )
        .is_err()
    );
    assert_eq!(rt.table_info("w2a_multi")?.unwrap().num_rows, 3);

    // Append several batches at once.
    let summary = rt.register_table_batches(
        "w2a_multi",
        s.clone(),
        vec![b2.clone(), b1.clone()],
        RegisterMode::Append,
    )?;
    assert_eq!(summary.num_rows, 6);
    assert_eq!(
        query_single_i64(
            &conn,
            "SELECT columns(*)::BIGINT FROM bolt_query('SELECT SUM(id) FROM w2a_multi')"
        )?,
        12
    );

    // Replace with an empty batch list: an empty table with the schema.
    let summary =
        rt.register_table_batches("w2a_multi", s.clone(), vec![], RegisterMode::Replace)?;
    assert_eq!(summary.num_rows, 0);
    assert_eq!(summary.schema.fields().len(), 2);
    assert_eq!(
        query_single_i64(&conn, "SELECT count(*) FROM bolt_scan('w2a_multi')")?,
        0
    );
    // Appending to the empty table drops its placeholder batch.
    rt.register_table_batches("w2a_multi", s.clone(), vec![b1], RegisterMode::Append)?;
    let (_, stored) = rt.get_table_batches("w2a_multi")?;
    assert_eq!(stored.len(), 1);
    assert_eq!(
        query_single_i64(
            &conn,
            "SELECT * FROM bolt_query('SELECT COUNT(*) FROM w2a_multi')"
        )?,
        2
    );

    // Column count mismatch is rejected before anything is registered.
    let narrow = schema(vec![("id", DataType::Int64)]);
    assert!(
        rt.register_table_batches("w2a_bad", narrow, vec![b2], RegisterMode::Create)
            .is_err()
    );
    assert!(rt.table_info("w2a_bad")?.is_none());
    Ok(())
}

#[test]
fn append_matches_columns_by_name_and_rejects_lossy_casts() -> Result<()> {
    let _guard = lock();
    let (_env, conn) = setup_connection()?;
    exec(
        &conn,
        "COPY (SELECT 1::INTEGER AS a, 'x' AS b) TO 'w2a_app' (FORMAT bolt, USE_TMP_FILE FALSE)",
    )?;
    // Reordered (and differently cased) columns are matched by name.
    exec(
        &conn,
        "COPY (SELECT 'y' AS B, 2 AS a) TO 'w2a_app' (FORMAT bolt, MODE 'append', USE_TMP_FILE FALSE)",
    )?;
    assert_eq!(
        query_single_i64(
            &conn,
            "SELECT count(*) FROM bolt_scan('w2a_app') WHERE (a = 1 AND b = 'x') OR (a = 2 AND b = 'y')"
        )?,
        2
    );
    // Out-of-range values are an error, not NULL.
    let err = conn
        .execute(
            "COPY (SELECT 5000000000::BIGINT AS a, 'z' AS b) TO 'w2a_app' \
             (FORMAT bolt, MODE 'append', USE_TMP_FILE FALSE)",
            Parameters::None,
        )
        .expect_err("overflowing append must fail");
    assert!(err.to_string().contains("Schema mismatch"), "{err}");
    // Unconvertible strings too.
    assert!(
        conn.execute(
            "COPY (SELECT 'abc' AS a, 'z' AS b) TO 'w2a_app' \
             (FORMAT bolt, MODE 'append', USE_TMP_FILE FALSE)",
            Parameters::None,
        )
        .is_err()
    );
    // Missing and extra columns.
    assert!(
        conn.execute(
            "COPY (SELECT 3 AS a) TO 'w2a_app' (FORMAT bolt, MODE 'append', USE_TMP_FILE FALSE)",
            Parameters::None,
        )
        .is_err()
    );
    assert!(
        conn.execute(
            "COPY (SELECT 3 AS a, 'q' AS b, 1 AS c) TO 'w2a_app' \
             (FORMAT bolt, MODE 'append', USE_TMP_FILE FALSE)",
            Parameters::None,
        )
        .is_err()
    );
    assert_eq!(
        query_single_i64(&conn, "SELECT count(*) FROM bolt_scan('w2a_app')")?,
        2,
        "failed appends leave the table unchanged"
    );
    Ok(())
}

#[test]
fn engine_errors_are_retried_on_the_host() -> Result<()> {
    let _guard = lock();
    let (_env, conn) = setup_connection()?;
    exec(
        &conn,
        "COPY (SELECT i::INTEGER AS id FROM range(1, 4) r(i)) TO 'w2a_retry' \
         (FORMAT bolt, USE_TMP_FILE FALSE)",
    )?;
    // md5 is a DuckDB function the Bolt engine does not implement: same answer on every backend.
    assert_eq!(
        query_single_i64(
            &conn,
            "SELECT count(*) FROM bolt_query('SELECT md5(CAST(id AS VARCHAR)) AS h FROM w2a_retry') \
             WHERE h = md5('2')"
        )?,
        1
    );
    // Invalid SQL still fails, with a clear message.
    let err = BoltRuntime::global()
        .execute_sql("SELECT nope FROM w2a_retry")
        .expect_err("unknown column");
    assert!(
        err.to_string().to_ascii_lowercase().contains("nope"),
        "{err}"
    );
    Ok(())
}

#[test]
fn backend_is_gpu_agrees_with_telemetry() -> Result<()> {
    let _guard = lock();
    let _conn = setup_connection()?;
    let rt = BoltRuntime::global();
    let gpu = rt.backend_is_gpu()?;
    let telemetry = rt.telemetry()?;
    assert_eq!(gpu, telemetry.backend == "cuda-gpu");
    if cfg!(feature = "cuda-stub") {
        assert!(!gpu);
        assert!(!telemetry.fallback_reason.is_empty());
    }
    if gpu {
        assert!(
            telemetry.fallback_reason.is_empty(),
            "{}",
            telemetry.fallback_reason
        );
    }
    Ok(())
}

#[test]
fn bolt_prefix_is_case_insensitive_and_bare_names_are_strict() -> Result<()> {
    let _guard = lock();
    let (_env, conn) = setup_connection()?;
    assert_eq!(clean_table_identifier("Bolt.GPU_D"), "GPU_D");
    assert_eq!(clean_table_identifier("BOLT:x"), "x");
    assert_eq!(clean_table_identifier("bolt.MAIN.x"), "x");

    let rt = BoltRuntime::global();
    let s = schema(vec![("id", DataType::Int32)]);
    let b = RecordBatch::try_new(s, vec![Arc::new(Int32Array::from(vec![7])) as ArrayRef]).unwrap();
    rt.register_table("w2a_orders", b, RegisterMode::Replace)?;

    assert_eq!(
        query_single_i64(
            &conn,
            "SELECT * FROM bolt_aggregate('Bolt.W2A_ORDERS', '', 'count(*) AS c')"
        )?,
        1
    );
    // Bare names: exact or case-insensitive matches only.
    assert_eq!(
        query_single_i64(&conn, "SELECT id::BIGINT FROM w2a_orders")?,
        7
    );
    assert_eq!(
        query_single_i64(&conn, "SELECT id::BIGINT FROM W2A_Orders")?,
        7
    );
    assert!(
        conn.query("SELECT * FROM tmp_w2a_orders", Parameters::None)
            .is_err()
    );
    assert!(
        conn.query("SELECT * FROM 'some/dir/w2a_orders'", Parameters::None)
            .is_err()
    );
    assert!(
        conn.query("SELECT * FROM w2a_orders.bolt", Parameters::None)
            .is_err()
    );
    // Explicit namespaces still resolve.
    assert_eq!(
        query_single_i64(&conn, "SELECT id::BIGINT FROM bolt.w2a_orders")?,
        7
    );
    assert_eq!(
        query_single_i64(&conn, "SELECT id::BIGINT FROM 'bolt:w2a_orders'")?,
        7
    );
    Ok(())
}

#[test]
fn field_metadata_survives_registration_and_append() -> Result<()> {
    let _guard = lock();
    let (_env, conn) = setup_connection()?;
    let rt = BoltRuntime::global();
    let meta = |ty: &str| {
        std::collections::HashMap::from([("duckdb.logical_type".to_string(), ty.to_string())])
    };
    // Int16 is normalized to Int32; the metadata must be kept.
    let s = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int16, true).with_metadata(meta("SMALLINT")),
        Field::new("b", DataType::Utf8, true).with_metadata(meta("VARCHAR")),
    ]));
    let mk = |a: Vec<i16>, b: Vec<&str>| {
        RecordBatch::try_new(
            s.clone(),
            vec![
                Arc::new(Int16Array::from(a)) as ArrayRef,
                Arc::new(StringArray::from(b)) as ArrayRef,
            ],
        )
        .unwrap()
    };
    rt.register_table_batches(
        "w2a_meta",
        s.clone(),
        vec![mk(vec![1, 2], vec!["x", "y"]), mk(vec![3], vec!["z"])],
        RegisterMode::Replace,
    )?;
    let check = |rt: &BoltRuntime| -> Result<()> {
        let info = rt.table_info("w2a_meta")?.expect("registered");
        assert_eq!(info.schema.field(0).data_type(), &DataType::Int32);
        assert_eq!(
            info.schema.field(0).metadata().get("duckdb.logical_type"),
            Some(&"SMALLINT".to_string())
        );
        assert_eq!(
            info.schema.field(1).metadata().get("duckdb.logical_type"),
            Some(&"VARCHAR".to_string())
        );
        Ok(())
    };
    check(rt)?;

    // Append a batch without metadata, columns reordered: stored schema (and metadata) win.
    let plain = Arc::new(Schema::new(vec![
        Field::new("b", DataType::Utf8, true),
        Field::new("a", DataType::Int64, true),
    ]));
    let appended = RecordBatch::try_new(
        plain.clone(),
        vec![
            Arc::new(StringArray::from(vec!["w"])) as ArrayRef,
            Arc::new(Int64Array::from(vec![4])) as ArrayRef,
        ],
    )
    .unwrap();
    let summary =
        rt.register_table_batches("w2a_meta", plain, vec![appended], RegisterMode::Append)?;
    assert_eq!(summary.num_rows, 4);
    check(rt)?;
    let (_, stored) = rt.get_table_batches("w2a_meta")?;
    assert!(stored.iter().all(|b| b.schema() == stored[0].schema()));
    assert_eq!(
        query_single_i64(
            &conn,
            "SELECT columns(*)::BIGINT FROM bolt_query('SELECT SUM(a) FROM w2a_meta')"
        )?,
        10
    );
    assert_eq!(
        query_single_i64(&conn, "SELECT count(*) FROM w2a_meta WHERE b = 'w'")?,
        1
    );
    // On a GPU host the metadata must not keep the table off the device.
    let reason = rt.telemetry()?.fallback_reason;
    assert!(!reason.contains("w2a_meta"), "{reason}");
    Ok(())
}

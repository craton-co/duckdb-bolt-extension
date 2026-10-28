// SPDX-License-Identifier: Apache-2.0

//! Wave 5 (W5-B) regression tests:
//!
//! * `CALL` / table-function forms of the catalog mutations act exactly once per execution
//!   (W4-C-004): `bolt_clear_tables()`, `bolt_drop_table(name)`,
//!   `bolt_register_sql(name, sql [, mode])`, `bolt_load_parquet(name, path [, mode])` and
//!   `bolt_load_arrow_ipc(name, path [, mode])`, including prepared statements, `EXPLAIN`,
//!   named `mode`, the file sandbox and coexistence with the scalar forms;
//! * an unfiltered `bolt_scan` (and, without a GPU, every `bolt_scan`, whose filters DuckDB then
//!   evaluates) is served from the shadow catalog and never from the host-fallback mirror
//!   (W4-A-001).
//!
//! Valid for both GPU and `cuda-stub` builds.

use std::{
    path::PathBuf,
    sync::{Arc, Mutex, Once},
};

use arrow_array::{Int32Array, RecordBatch};
use arrow_ipc::writer::FileWriter;
use arrow_schema::{DataType, Field, Schema};

use duckdb_bolt_core::table_functions::last_bolt_scan_trace_for_testing;
use duckdb_loadable_macros::duckdb_entrypoint_c_api_v2;
use duckdb_neo::{
    Parameters, Result,
    connection::Connection,
    environment::{Environment, StorageLocation},
    extension::{Extension, register_static_extension},
    logical_type::LogicalTypeID,
    vector::Unknown,
};

#[duckdb_entrypoint_c_api_v2(extension_name = "bolt", api_version = "2.0.0")]
fn bolt_w5b_test_entrypoint(extension: &Extension) -> Result<()> {
    duckdb_bolt_core::register(extension)
}

static REGISTER_ONCE: Once = Once::new();
static TEST_MUTEX: Mutex<()> = Mutex::new(());

fn lock() -> std::sync::MutexGuard<'static, ()> {
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

fn error_of(conn: &Connection, sql: &str) -> String {
    match conn.execute(sql, Parameters::None) {
        Ok(_) => panic!("expected an error from {sql}"),
        Err(e) => e.message,
    }
}

/// All rows of a one-column query, rendered by DuckDB's `VARCHAR` cast. `CALL` and `EXECUTE`
/// cannot be wrapped in a subquery; their (BIGINT or BOOLEAN) column is read natively.
fn column(conn: &Connection, sql: &str) -> Result<Vec<Option<String>>> {
    let upper = sql.trim_start().to_ascii_uppercase();
    let wrapped = if upper.starts_with("CALL") || upper.starts_with("EXECUTE") {
        sql.to_string()
    } else {
        format!("SELECT (COLUMNS(*))::VARCHAR FROM ({sql}) AS q")
    };
    let mut result = conn.query(wrapped.as_str(), Parameters::None)?;
    let mut out = Vec::new();
    while let Some(chunk) = result.next_chunk()? {
        let rows = chunk.row_count()?;
        let type_id = chunk.get_vector_at::<Unknown>(0)?.logical_type().id()?;
        match type_id {
            LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_BIGINT => {
                let vector = chunk.get_vector_at::<i64>(0)?;
                let reader = vector.reader()?;
                for row in 0..rows {
                    out.push(reader.get(row)?.map(|v| v.to_string()));
                }
            }
            LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_BOOLEAN => {
                let vector = chunk.get_vector_at::<bool>(0)?;
                let reader = vector.reader()?;
                for row in 0..rows {
                    out.push(reader.get(row)?.map(|v| v.to_string()));
                }
            }
            _ => {
                let vector = chunk.get_vector_at::<String>(0)?;
                let reader = vector.reader()?;
                for row in 0..rows {
                    out.push(reader.get(row)?.map(|s| s.to_string()));
                }
            }
        }
    }
    Ok(out)
}

fn one(conn: &Connection, sql: &str) -> Result<String> {
    let rows = column(conn, sql)?;
    assert_eq!(rows.len(), 1, "{sql}: {rows:?}");
    Ok(rows[0].clone().unwrap_or_else(|| "NULL".to_string()))
}

fn table_rows(conn: &Connection, table: &str) -> Result<String> {
    one(
        conn,
        &format!("SELECT num_rows FROM bolt_tables() WHERE table_name = '{table}'"),
    )
}

fn table_count(conn: &Connection) -> Result<String> {
    one(conn, "SELECT count(*) FROM bolt_tables()")
}

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("w5b_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("temp dir");
        Self(dir)
    }

    fn file(&self, name: &str) -> String {
        self.0.join(name).to_string_lossy().replace('\\', "/")
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn write_parquet(conn: &Connection, path: &str, rows: usize) -> Result<()> {
    exec(
        conn,
        &format!(
            "COPY (SELECT range::BIGINT AS v FROM range({rows})) TO '{path}' (FORMAT parquet)"
        ),
    )
}

fn write_arrow_ipc(path: &str, rows: i32) {
    let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int32, true)]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from_iter_values(0..rows))],
    )
    .expect("batch");
    let file = std::fs::File::create(path).expect("create ipc file");
    let mut writer = FileWriter::try_new(file, &schema).expect("ipc writer");
    writer.write(&batch).expect("write batch");
    writer.finish().expect("finish ipc");
}

// ------------------------------------------------------------------------------ W4-C-004

#[test]
fn call_clear_tables_clears_once_and_coexists_with_the_scalar() -> Result<()> {
    let _g = lock();
    let (_env, conn) = setup_connection()?;
    for t in ["w5b_c1", "w5b_c2", "w5b_c3"] {
        exec(
            &conn,
            &format!("COPY (SELECT 1 AS a) TO '{t}' (FORMAT bolt, USE_TMP_FILE FALSE)"),
        )?;
    }
    exec(&conn, "SET threads = 8")?;
    // Table form: one execution, one clear, one row.
    assert_eq!(
        column(&conn, "SELECT * FROM bolt_clear_tables()")?,
        [Some("3".into())]
    );
    assert_eq!(table_count(&conn)?, "0");

    exec(
        &conn,
        "COPY (SELECT 1 AS a) TO 'w5b_c1' (FORMAT bolt, USE_TMP_FILE FALSE)",
    )?;
    assert_eq!(one(&conn, "CALL bolt_clear_tables()")?, "1");
    assert_eq!(one(&conn, "CALL bolt_clear_tables()")?, "0");
    // The scalar and pragma forms are still there.
    assert_eq!(one(&conn, "SELECT bolt_clear_tables()")?, "0");
    exec(&conn, "PRAGMA bolt_clear_tables")?;

    // EXPLAIN binds but does not execute: no side effect.
    exec(
        &conn,
        "COPY (SELECT 1 AS a) TO 'w5b_c1' (FORMAT bolt, USE_TMP_FILE FALSE)",
    )?;
    exec(&conn, "EXPLAIN CALL bolt_clear_tables()")?;
    exec(&conn, "EXPLAIN SELECT * FROM bolt_drop_table('w5b_c1')")?;
    assert_eq!(table_count(&conn)?, "1");

    // Prepared: every EXECUTE clears again.
    exec(
        &conn,
        "PREPARE clear_p AS SELECT * FROM bolt_clear_tables()",
    )?;
    assert_eq!(table_count(&conn)?, "1");
    assert_eq!(one(&conn, "EXECUTE clear_p")?, "1");
    exec(
        &conn,
        "COPY (SELECT 1 AS a) TO 'w5b_c1' (FORMAT bolt, USE_TMP_FILE FALSE)",
    )?;
    assert_eq!(one(&conn, "EXECUTE clear_p")?, "1");
    exec(&conn, "RESET threads")?;
    Ok(())
}

#[test]
fn call_drop_table_reports_whether_it_dropped() -> Result<()> {
    let _g = lock();
    let (_env, conn) = setup_connection()?;
    exec(
        &conn,
        "COPY (SELECT 1 AS a) TO 'w5b_d' (FORMAT bolt, USE_TMP_FILE FALSE)",
    )?;
    assert_eq!(one(&conn, "CALL bolt_drop_table('w5b_d')")?, "true");
    assert_eq!(one(&conn, "CALL bolt_drop_table('w5b_d')")?, "false");

    // A prepared parameter is bound at EXECUTE time.
    exec(
        &conn,
        "COPY (SELECT 1 AS a) TO 'w5b_d' (FORMAT bolt, USE_TMP_FILE FALSE)",
    )?;
    exec(&conn, "PREPARE drop_p AS SELECT * FROM bolt_drop_table($1)")?;
    assert_eq!(one(&conn, "EXECUTE drop_p('w5b_nope')")?, "false");
    assert_eq!(one(&conn, "EXECUTE drop_p('w5b_d')")?, "true");

    let err = error_of(&conn, "CALL bolt_drop_table(NULL)");
    assert!(err.contains("cannot be NULL"), "{err}");
    Ok(())
}

#[test]
fn call_register_sql_validates_mode_before_running() -> Result<()> {
    let _g = lock();
    let (_env, conn) = setup_connection()?;
    exec(
        &conn,
        "COPY (SELECT range::INTEGER AS id FROM range(10)) TO 'w5b_src' (FORMAT bolt, USE_TMP_FILE FALSE)",
    )?;
    assert_eq!(
        one(
            &conn,
            "CALL bolt_register_sql('w5b_r', 'SELECT id FROM w5b_src WHERE id < 4')"
        )?,
        "4"
    );
    assert_eq!(
        one(
            &conn,
            "CALL bolt_register_sql('w5b_r', 'SELECT id FROM w5b_src WHERE id < 4', mode := 'append')"
        )?,
        "8"
    );
    assert_eq!(table_rows(&conn, "w5b_r")?, "8");
    let err = error_of(
        &conn,
        "CALL bolt_register_sql('w5b_r2', 'SELECT id FROM w5b_src', 'sideways')",
    );
    assert!(err.contains("bolt_register_sql"), "{err}");
    assert_eq!(
        one(
            &conn,
            "SELECT count(*) FROM bolt_tables() WHERE table_name = 'w5b_r2'"
        )?,
        "0"
    );
    let err = error_of(
        &conn,
        "CALL bolt_register_sql('w5b_r', 'SELECT id FROM w5b_src', 'create')",
    );
    assert!(!err.is_empty());
    assert_eq!(table_rows(&conn, "w5b_r")?, "8");
    exec(&conn, "PRAGMA bolt_clear_tables")?;
    Ok(())
}

#[test]
fn call_load_parquet_and_arrow_ipc_load_once_per_execution() -> Result<()> {
    let _g = lock();
    let (_env, conn) = setup_connection()?;
    let dir = TempDir::new("call_load");
    let parquet = dir.file("t.parquet");
    write_parquet(&conn, &parquet, 100)?;

    exec(&conn, "SET threads = 8")?;
    assert_eq!(
        one(
            &conn,
            &format!("CALL bolt_load_parquet('w5b_pq', '{parquet}')")
        )?,
        "100"
    );
    // Prepared append: one load per EXECUTE.
    exec(
        &conn,
        &format!(
            "PREPARE load_p AS SELECT * FROM bolt_load_parquet('w5b_pq', '{parquet}', 'append')"
        ),
    )?;
    assert_eq!(table_rows(&conn, "w5b_pq")?, "100");
    assert_eq!(one(&conn, "EXECUTE load_p")?, "200");
    assert_eq!(one(&conn, "EXECUTE load_p")?, "300");
    assert_eq!(
        one(
            &conn,
            &format!("SELECT * FROM bolt_load_parquet('w5b_pq', '{parquet}', mode := 'append')")
        )?,
        "400"
    );
    exec(
        &conn,
        &format!("EXPLAIN CALL bolt_load_parquet('w5b_pq', '{parquet}', 'append')"),
    )?;
    assert_eq!(table_rows(&conn, "w5b_pq")?, "400");
    assert_eq!(
        one(&conn, "SELECT sum(v) FROM bolt_scan('w5b_pq')")?,
        "19800"
    );

    // Arrow IPC file written with arrow-ipc.
    let ipc = dir.file("t.arrow");
    write_arrow_ipc(&ipc, 25);
    assert_eq!(
        one(
            &conn,
            &format!("CALL bolt_load_arrow_ipc('w5b_ipc', '{ipc}')")
        )?,
        "25"
    );
    assert_eq!(
        one(
            &conn,
            &format!("CALL bolt_load_arrow_ipc('w5b_ipc', '{ipc}', 'append')")
        )?,
        "50"
    );
    let err = error_of(
        &conn,
        &format!(
            "CALL bolt_load_arrow_ipc('w5b_ipc', '{}')",
            dir.file("missing.arrow")
        ),
    );
    assert!(err.contains("bolt_load_arrow_ipc('w5b_ipc'"), "{err}");
    assert_eq!(table_rows(&conn, "w5b_ipc")?, "50");
    let err = error_of(
        &conn,
        &format!("CALL bolt_load_parquet('w5b_pq', '{parquet}', 'bogus')"),
    );
    assert!(err.contains("bolt_load_parquet"), "{err}");
    assert_eq!(table_rows(&conn, "w5b_pq")?, "400");
    exec(&conn, "RESET threads")?;
    exec(&conn, "PRAGMA bolt_clear_tables")?;
    Ok(())
}

#[test]
fn call_load_parquet_honors_the_file_sandbox() -> Result<()> {
    let _g = lock();
    let (_env, conn) = setup_connection()?;
    let dir = TempDir::new("call_sandbox");
    let parquet = dir.file("t.parquet");
    write_parquet(&conn, &parquet, 10)?;
    exec(&conn, "SET enable_external_access = false")?;
    let err = error_of(
        &conn,
        &format!("CALL bolt_load_parquet('w5b_sbx', '{parquet}')"),
    );
    assert!(err.contains("enable_external_access"), "{err}");
    assert_eq!(table_count(&conn)?, "0");
    Ok(())
}

// ------------------------------------------------------------------------------ W4-A-001

#[test]
fn unfiltered_and_stub_scans_are_served_from_the_shadow_catalog() -> Result<()> {
    let _g = lock();
    let (_env, conn) = setup_connection()?;
    exec(
        &conn,
        "COPY (SELECT range::INTEGER AS id, (range % 7)::BIGINT AS k FROM range(3000)) \
         TO 'w5b_scan' (FORMAT bolt, USE_TMP_FILE FALSE)",
    )?;
    let gpu = one(&conn, "SELECT backend FROM bolt_pool_stats()")? == "cuda-gpu";

    for sql in [
        "SELECT count(*) FROM bolt_scan('w5b_scan')",
        "SELECT sum(id) FROM bolt.w5b_scan",
        "SELECT max(k) FROM 'bolt:w5b_scan'",
    ] {
        column(&conn, sql)?;
        let trace = last_bolt_scan_trace_for_testing().expect("trace");
        assert_eq!(trace.path, "unfiltered", "{sql}");
    }

    assert_eq!(
        one(
            &conn,
            "SELECT count(*) FROM bolt_scan('w5b_scan') WHERE k = 3"
        )?,
        "429"
    );
    let trace = last_bolt_scan_trace_for_testing().expect("trace");
    if gpu {
        assert_ne!(trace.path, "unfiltered");
    } else {
        // Without a GPU no filter is pushed: DuckDB filters the stored batches itself, and
        // the embedded fallback DuckDB (and its mirror) is not involved.
        assert_eq!(trace.path, "unfiltered");
        assert!(trace.filters.is_empty());
    }
    exec(&conn, "PRAGMA bolt_clear_tables")?;
    Ok(())
}

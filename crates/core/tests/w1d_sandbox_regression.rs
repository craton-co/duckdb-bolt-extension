// SPDX-License-Identifier: Apache-2.0

//! Regression tests for W1-D: DuckDB sandboxing (`enable_external_access`,
//! `allowed_directories`, `allowed_paths`) for `bolt_load_*` and `bolt_ptx_cache_dir`,
//! zero-argument scalar row counts, `bolt_clear_tables()` semantics, and NULL handling
//! of the side-effecting scalars.
//!
//! Valid for both GPU and `cuda-stub` builds: nothing here depends on the backend.

use std::{
    fs::File,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, Once},
};

use arrow_array::{Int16Array, RecordBatch, StringArray};
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
fn bolt_test_entrypoint(extension: &Extension) -> Result<()> {
    duckdb_bolt_core::register(extension)
}

static REGISTER_ONCE: Once = Once::new();
// Serialize tests because Craton Bolt's runtime catalog and config are process-wide.
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

/// Run `sql` to completion and return its error message, or `None` if it succeeded.
/// Scalar execution errors surface lazily, so every chunk is drained.
fn query_error(conn: &Connection, sql: &str) -> Option<String> {
    match conn.query(sql, Parameters::None) {
        Err(e) => Some(e.message),
        Ok(mut r) => loop {
            match r.next_chunk() {
                Err(e) => break Some(e.message),
                Ok(None) => break None,
                Ok(Some(_)) => {}
            }
        },
    }
}

fn expect_error(conn: &Connection, sql: &str, needles: &[&str]) {
    let msg = query_error(conn, sql).unwrap_or_else(|| panic!("expected failure: {sql}"));
    for needle in needles {
        assert!(
            msg.contains(needle),
            "error for `{sql}` should contain `{needle}`, got: {msg}"
        );
    }
}

fn i64_column(conn: &Connection, sql: &str) -> Result<Vec<Option<i64>>> {
    let mut out = Vec::new();
    let mut result = conn.query(sql, Parameters::None)?;
    while let Some(chunk) = result.next_chunk()? {
        let vec = chunk.get_vector_at::<i64>(0)?;
        let reader = vec.reader()?;
        for row in 0..chunk.row_count()? {
            out.push(reader.get(row)?.copied());
        }
    }
    Ok(out)
}

fn string_column(conn: &Connection, sql: &str) -> Result<Vec<Option<String>>> {
    let mut out = Vec::new();
    let mut result = conn.query(sql, Parameters::None)?;
    while let Some(chunk) = result.next_chunk()? {
        let vec = chunk.get_vector_at::<String>(0)?;
        let reader = vec.reader()?;
        for row in 0..chunk.row_count()? {
            out.push(reader.get(row)?.map(str::to_string));
        }
    }
    Ok(out)
}

fn bool_column(conn: &Connection, sql: &str) -> Result<Vec<Option<bool>>> {
    let mut out = Vec::new();
    let mut result = conn.query(sql, Parameters::None)?;
    while let Some(chunk) = result.next_chunk()? {
        let vec = chunk.get_vector_at::<bool>(0)?;
        let reader = vec.reader()?;
        for row in 0..chunk.row_count()? {
            out.push(reader.get(row)?.copied());
        }
    }
    Ok(out)
}

/// A scratch directory removed on drop.
struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("bolt_w1d_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }

    fn path(&self, rel: &str) -> PathBuf {
        self.0.join(rel)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Quote a path for use inside a SQL string literal.
fn lit(p: &Path) -> String {
    p.to_string_lossy().replace('\'', "''")
}

fn sample_batch() -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int16, false),
        Field::new("label", DataType::Utf8, true),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int16Array::from(vec![1, 2, 3, 4])),
            Arc::new(StringArray::from(vec![
                Some("a"),
                None,
                Some("c"),
                Some("d"),
            ])),
        ],
    )
    .unwrap()
}

fn write_parquet(path: &Path) {
    let batch = sample_batch();
    let mut w =
        parquet::arrow::ArrowWriter::try_new(File::create(path).unwrap(), batch.schema(), None)
            .unwrap();
    w.write(&batch).unwrap();
    w.close().unwrap();
}

fn write_ipc(path: &Path) {
    let batch = sample_batch();
    let mut w =
        arrow_ipc::writer::FileWriter::try_new(File::create(path).unwrap(), &batch.schema())
            .unwrap();
    w.write(&batch).unwrap();
    w.finish().unwrap();
}

fn is_registered(name: &str) -> Result<bool> {
    Ok(BoltRuntime::global().table_info(name)?.is_some())
}

#[test]
fn test_load_refused_when_external_access_disabled() -> Result<()> {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let scratch = Scratch::new("denied");
    let pq = scratch.path("t.parquet");
    let ipc = scratch.path("t.arrow");
    write_parquet(&pq);
    write_ipc(&ipc);

    let (_env, conn) = setup_connection()?;
    conn.execute("SET enable_external_access = false", Parameters::None)?;

    for sql in [
        format!("SELECT bolt_load_parquet('w1d_denied_pq', '{}')", lit(&pq)),
        format!(
            "SELECT bolt_load_parquet('w1d_denied_pq', '{}', 'append')",
            lit(&pq)
        ),
        format!(
            "SELECT bolt_load_arrow_ipc('w1d_denied_ipc', '{}')",
            lit(&ipc)
        ),
        // A missing file gets the same permission error (no existence oracle).
        format!(
            "SELECT bolt_load_parquet('w1d_denied_pq', '{}')",
            lit(&scratch.path("missing.parquet"))
        ),
    ] {
        expect_error(&conn, &sql, &["enable_external_access", "w1d_denied"]);
    }
    assert!(!is_registered("w1d_denied_pq")?);
    assert!(!is_registered("w1d_denied_ipc")?);

    // The PTX cache directory cannot be pointed anywhere while sandboxed...
    let ptx = scratch.path("ptx");
    std::fs::create_dir_all(&ptx).unwrap();
    assert!(
        conn.execute(
            format!("SET bolt_ptx_cache_dir = '{}'", lit(&ptx)),
            Parameters::None
        )
        .is_err_and(|e| e.message.contains("enable_external_access"))
    );
    assert_eq!(BoltRuntime::global().config()?.ptx_cache_dir, None);
    // ...but disabling it is always allowed.
    conn.execute("SET bolt_ptx_cache_dir = ''", Parameters::None)?;
    conn.execute("RESET bolt_ptx_cache_dir", Parameters::None)?;
    Ok(())
}

#[test]
fn test_allowed_directories_and_paths_are_honored() -> Result<()> {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let scratch = Scratch::new("allow");
    std::fs::create_dir_all(scratch.path("allowed")).unwrap();
    std::fs::create_dir_all(scratch.path("allowed_sibling")).unwrap();
    std::fs::create_dir_all(scratch.path("other")).unwrap();
    let inside = scratch.path("allowed").join("in.parquet");
    let sibling = scratch.path("allowed_sibling").join("sib.parquet");
    let outside = scratch.path("other").join("out.parquet");
    let single = scratch.path("other").join("single.arrow");
    write_parquet(&inside);
    write_parquet(&sibling);
    write_parquet(&outside);
    write_ipc(&single);

    let (_env, conn) = setup_connection()?;
    // Allow-lists must be configured before external access is switched off.
    conn.execute(
        format!(
            "SET allowed_directories = ['{}']",
            lit(&scratch.path("allowed"))
        ),
        Parameters::None,
    )?;
    conn.execute(
        format!("SET allowed_paths = ['{}']", lit(&single)),
        Parameters::None,
    )?;
    conn.execute("SET enable_external_access = false", Parameters::None)?;

    assert_eq!(
        i64_column(
            &conn,
            &format!("SELECT bolt_load_parquet('w1d_in', '{}')", lit(&inside))
        )?,
        vec![Some(4)]
    );
    assert_eq!(
        i64_column(
            &conn,
            &format!(
                "SELECT bolt_load_arrow_ipc('w1d_single', '{}')",
                lit(&single)
            )
        )?,
        vec![Some(4)]
    );
    expect_error(
        &conn,
        &format!("SELECT bolt_load_parquet('w1d_out', '{}')", lit(&outside)),
        &["enable_external_access"],
    );
    // A directory sharing a string prefix with the allowed one is not inside it.
    expect_error(
        &conn,
        &format!("SELECT bolt_load_parquet('w1d_out', '{}')", lit(&sibling)),
        &["enable_external_access"],
    );
    // `..` cannot escape the allowed directory.
    let traversal = scratch
        .path("allowed")
        .join("..")
        .join("other")
        .join("out.parquet");
    expect_error(
        &conn,
        &format!("SELECT bolt_load_parquet('w1d_out', '{}')", lit(&traversal)),
        &["enable_external_access"],
    );
    assert!(is_registered("w1d_in")?);
    assert!(is_registered("w1d_single")?);
    assert!(!is_registered("w1d_out")?);

    BoltRuntime::global().drop_table("w1d_in")?;
    BoltRuntime::global().drop_table("w1d_single")?;
    Ok(())
}

#[test]
fn test_load_and_ptx_cache_dir_work_with_external_access_enabled() -> Result<()> {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let scratch = Scratch::new("enabled");
    let pq = scratch.path("t.parquet");
    let ipc = scratch.path("t.arrow");
    write_parquet(&pq);
    write_ipc(&ipc);

    let (_env, conn) = setup_connection()?;
    assert_eq!(
        i64_column(
            &conn,
            &format!("SELECT bolt_load_parquet('w1d_ok', '{}')", lit(&pq))
        )?,
        vec![Some(4)]
    );
    assert_eq!(
        i64_column(
            &conn,
            &format!(
                "SELECT bolt_load_parquet('w1d_ok', '{}', 'append')",
                lit(&pq)
            )
        )?,
        vec![Some(8)]
    );
    assert_eq!(
        i64_column(
            &conn,
            &format!("SELECT bolt_load_arrow_ipc('w1d_ok_ipc', '{}')", lit(&ipc))
        )?,
        vec![Some(4)]
    );
    assert_eq!(
        i64_column(&conn, "SELECT SUM(id)::BIGINT FROM bolt_scan('w1d_ok')")?,
        vec![Some(20)]
    );

    // PTX cache dir validation: absolute existing directory is accepted (and not created).
    let ptx = scratch.path("ptx");
    std::fs::create_dir_all(&ptx).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&ptx, std::fs::Permissions::from_mode(0o700)).unwrap();
    }
    conn.execute(
        format!("SET bolt_ptx_cache_dir = '{}'", lit(&ptx)),
        Parameters::None,
    )?;
    assert_eq!(
        BoltRuntime::global().config()?.ptx_cache_dir,
        Some(ptx.clone())
    );
    // Relative paths and non-directories are rejected and leave the setting unchanged.
    assert!(
        conn.execute("SET bolt_ptx_cache_dir = 'relative/ptx'", Parameters::None)
            .is_err()
    );
    assert!(
        conn.execute(
            format!("SET bolt_ptx_cache_dir = '{}'", lit(&pq)),
            Parameters::None
        )
        .is_err()
    );
    assert_eq!(
        BoltRuntime::global().config()?.ptx_cache_dir,
        Some(ptx.clone())
    );
    // NULL is rejected; RESET restores the default (disabled).
    assert!(
        conn.execute("SET bolt_ptx_cache_dir = NULL", Parameters::None)
            .is_err()
    );
    conn.execute("RESET bolt_ptx_cache_dir", Parameters::None)?;
    assert_eq!(BoltRuntime::global().config()?.ptx_cache_dir, None);

    // Option NULL handling: SET NULL is an error, RESET restores the default.
    for opt in ["bolt_device", "bolt_memory_budget", "bolt_enable_tracing"] {
        assert!(
            conn.execute(format!("SET {opt} = NULL"), Parameters::None)
                .is_err(),
            "SET {opt} = NULL must be rejected"
        );
        conn.execute(format!("RESET {opt}"), Parameters::None)?;
    }
    assert_eq!(BoltRuntime::global().config()?.memory_budget, None);
    // The runtime is process-wide, so session-scoped writes are refused.
    assert!(
        conn.execute("SET SESSION bolt_enable_tracing = true", Parameters::None)
            .is_err()
    );

    BoltRuntime::global().drop_table("w1d_ok")?;
    BoltRuntime::global().drop_table("w1d_ok_ipc")?;
    Ok(())
}

#[test]
fn test_zero_arg_scalars_write_exactly_row_count_rows() -> Result<()> {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let (_env, conn) = setup_connection()?;

    let one = string_column(&conn, "SELECT bolt_version()")?;
    assert_eq!(one.len(), 1);
    assert!(
        one[0]
            .as_deref()
            .is_some_and(|v| v.starts_with("duckdb-bolt-extension"))
    );

    let many = string_column(&conn, "SELECT bolt_version() FROM range(5000)")?;
    assert_eq!(many.len(), 5000);
    assert!(many.iter().all(Option::is_some));

    assert!(string_column(&conn, "SELECT bolt_version() FROM range(0)")?.is_empty());

    let prom = string_column(&conn, "SELECT bolt_prometheus_metrics() FROM range(3000)")?;
    assert_eq!(prom.len(), 3000);
    assert!(prom.iter().all(Option::is_some));
    assert_eq!(
        string_column(&conn, "SELECT bolt_metrics_prometheus()")?.len(),
        1
    );

    assert_eq!(
        i64_column(&conn, "SELECT bolt_clear_tables()")?,
        vec![Some(0)]
    );
    Ok(())
}

#[test]
fn test_clear_tables_clears_once_per_query() -> Result<()> {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let (_env, conn) = setup_connection()?;
    // One worker thread: the documented semantics are "once per worker per query".
    conn.execute("SET threads = 1", Parameters::None)?;

    for name in ["w1d_clear_a", "w1d_clear_b"] {
        BoltRuntime::global().register_table(
            name,
            duckdb_bolt_core::ingest::normalize_batch_for_bolt(&sample_batch())?,
            RegisterMode::Replace,
        )?;
    }
    // 5000 rows span three DuckDB chunks; every row must report the same, single clear.
    let mut result = conn.query(
        "SELECT min(c), max(c), count(*) FROM (SELECT bolt_clear_tables() AS c FROM range(5000))",
        Parameters::None,
    )?;
    let chunk = result.next_chunk()?.expect("one result row");
    let get = |i: usize| -> Result<Option<i64>> {
        Ok(chunk.get_vector_at::<i64>(i)?.reader()?.get(0)?.copied())
    };
    assert_eq!(get(0)?, Some(2));
    assert_eq!(get(1)?, Some(2));
    assert_eq!(get(2)?, Some(5000));
    drop(result);

    assert!(!is_registered("w1d_clear_a")?);
    assert!(!is_registered("w1d_clear_b")?);
    // A new query clears again.
    BoltRuntime::global().register_table(
        "w1d_clear_a",
        duckdb_bolt_core::ingest::normalize_batch_for_bolt(&sample_batch())?,
        RegisterMode::Replace,
    )?;
    assert_eq!(
        i64_column(&conn, "SELECT bolt_clear_tables()")?,
        vec![Some(1)]
    );
    Ok(())
}

#[test]
fn test_null_arguments_and_per_row_side_effects() -> Result<()> {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let scratch = Scratch::new("nulls");
    let pq = scratch.path("t.parquet");
    write_parquet(&pq);
    let pq_s = lit(&pq);
    let (_env, conn) = setup_connection()?;

    // Constant NULLs.
    for sql in [
        "SELECT bolt_load_parquet('w1d_null', NULL)".to_string(),
        format!("SELECT bolt_load_parquet(NULL, '{pq_s}')"),
        format!("SELECT bolt_load_parquet('w1d_null', '{pq_s}', NULL)"),
        "SELECT bolt_load_arrow_ipc('w1d_null', NULL, 'replace')".to_string(),
        "SELECT bolt_register_sql(NULL, 'SELECT 1 AS x')".to_string(),
        "SELECT bolt_register_sql('w1d_null', NULL)".to_string(),
    ] {
        assert_eq!(i64_column(&conn, &sql)?, vec![None], "{sql}");
    }
    assert_eq!(
        bool_column(&conn, "SELECT bolt_drop_table(NULL)")?,
        vec![None]
    );

    // Non-constant NULLs behave the same (NULL mode is NULL, not 'replace').
    assert_eq!(
        i64_column(
            &conn,
            &format!(
                "SELECT bolt_load_parquet(t, p, m) FROM (VALUES \
                 ('w1d_null', '{pq_s}', NULL::VARCHAR), \
                 ('w1d_null', NULL, 'replace'), \
                 (NULL, '{pq_s}', 'replace')) v(t, p, m)"
            )
        )?,
        vec![None, None, None]
    );
    assert!(!is_registered("w1d_null")?);

    // One side effect per non-NULL row.
    assert_eq!(
        i64_column(
            &conn,
            &format!(
                "SELECT bolt_load_parquet(t, '{pq_s}') FROM (VALUES ('w1d_row_a'), ('w1d_row_b')) v(t)"
            )
        )?,
        vec![Some(4), Some(4)]
    );
    assert!(is_registered("w1d_row_a")?);
    assert!(is_registered("w1d_row_b")?);
    assert_eq!(
        bool_column(
            &conn,
            "SELECT bolt_drop_table(t) FROM (VALUES ('w1d_row_a'), ('w1d_row_b'), ('w1d_row_c')) v(t)"
        )?,
        vec![Some(true), Some(true), Some(false)]
    );

    // Errors name the function and the row's table.
    expect_error(
        &conn,
        &format!(
            "SELECT bolt_load_parquet('w1d_missing_tbl', '{}')",
            lit(&scratch.path("missing.parquet"))
        ),
        &["bolt_load_parquet", "w1d_missing_tbl"],
    );
    expect_error(
        &conn,
        &format!("SELECT bolt_load_parquet('w1d_bad_mode', '{pq_s}', 'bogus')"),
        &["w1d_bad_mode"],
    );
    assert!(!is_registered("w1d_missing_tbl")?);
    assert!(!is_registered("w1d_bad_mode")?);
    Ok(())
}

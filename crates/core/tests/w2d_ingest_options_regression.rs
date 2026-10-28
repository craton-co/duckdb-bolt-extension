// SPDX-License-Identifier: Apache-2.0

//! Regression tests for W2-D:
//!
//! * ingestion registers decoded batches without concatenating them (W1-D-006), makes
//!   every column nullable and strips time zones to naive UTC (W1-E-010), and rejects
//!   `Date64` values that are not whole days (W1-D-007);
//! * loads go through DuckDB's file system as well: `disabled_filesystems` and `file:`
//!   URLs (W1-D-001, W1-D-002);
//! * `mode` is parsed the same way for `bolt_register_sql`, `bolt_load_parquet` and
//!   `bolt_load_arrow_ipc` (W1-D-005, W1-E-007);
//! * `current_setting('bolt_*')` follows the process-wide runtime across databases, and
//!   `RESET` restores the documented defaults (W1-E-006, W1-D-003).
//!
//! Valid for both GPU and `cuda-stub` builds: nothing here depends on the backend.

use std::{
    fs::File,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, Once},
};

use arrow_array::{Date64Array, Int16Array, RecordBatch, StringArray, TimestampMicrosecondArray};
use arrow_schema::{DataType, Field, Schema, TimeUnit};
use duckdb_bolt_core::runtime::BoltRuntime;
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

fn register_extension() {
    REGISTER_ONCE.call_once(|| {
        // SAFETY: `duckdb_extension_bolt_describe` is emitted by `#[duckdb_entrypoint_c_api_v2]`, which
        // fills the descriptor as `duckdb_static_extension.h` specifies (the contract of
        // `register_static_extension`); `Once` makes this a single call.
        unsafe { register_static_extension(duckdb_extension_bolt_describe) }
            .expect("Failed to register static bolt extension");
    });
}

/// A fresh in-memory database (its own `Environment`), with the extension loaded.
fn setup_connection() -> Result<(Environment, Connection)> {
    register_extension();
    let env = Environment::new()?;
    let instance = env.open(StorageLocation::InMemory)?;
    let conn = instance.connect()?;
    conn.execute("PRAGMA bolt_clear_tables", Parameters::None)?;
    Ok((env, conn))
}

/// Run `sql` to completion and return its error message, or `None` if it succeeded.
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

fn single_i64(conn: &Connection, sql: &str) -> Result<i64> {
    let values = i64_column(conn, sql)?;
    assert_eq!(values.len(), 1, "`{sql}` should return one row");
    Ok(values[0].unwrap_or_else(|| panic!("`{sql}` returned NULL")))
}

fn single_string(conn: &Connection, sql: &str) -> Result<Option<String>> {
    let mut result = conn.query(sql, Parameters::None)?;
    let chunk = result.next_chunk()?.expect("one result row");
    let vec = chunk.get_vector_at::<String>(0)?;
    let value = vec.reader()?.get(0)?.map(str::to_string);
    Ok(value)
}

/// A scratch directory removed on drop.
struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("bolt_w2d_{tag}_{}", std::process::id()));
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

/// `file:///...` URL for an absolute local path (forward slashes on every platform).
fn file_url(p: &Path) -> String {
    let text = p.to_string_lossy().replace('\\', "/");
    format!("file:///{}", text.trim_start_matches('/')).replace('\'', "''")
}

fn is_registered(name: &str) -> Result<bool> {
    Ok(BoltRuntime::global().table_info(name)?.is_some())
}

/// `id SMALLINT NOT NULL`, `ts TIMESTAMP(us, 'America/New_York')`, `label VARCHAR`.
fn zoned_batch(first_id: i16) -> RecordBatch {
    let tz: Arc<str> = Arc::from("America/New_York");
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int16, false),
        Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, Some(tz.clone())),
            true,
        ),
        Field::new("label", DataType::Utf8, true),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int16Array::from(vec![first_id, first_id + 1, first_id + 2])),
            Arc::new(
                TimestampMicrosecondArray::from(vec![Some(1_000_000), None, Some(3_000_000)])
                    .with_timezone(tz),
            ),
            Arc::new(StringArray::from(vec![Some("a"), None, Some("c")])),
        ],
    )
    .unwrap()
}

/// Write `row_groups` row groups of [`zoned_batch`] to a Parquet file.
fn write_parquet(path: &Path, row_groups: i16) {
    let schema = zoned_batch(0).schema();
    let mut w =
        parquet::arrow::ArrowWriter::try_new(File::create(path).unwrap(), schema, None).unwrap();
    for g in 0..row_groups {
        w.write(&zoned_batch(g * 3)).unwrap();
        w.flush().unwrap();
    }
    w.close().unwrap();
}

/// Write `batches` record batches of [`zoned_batch`] to an Arrow IPC file.
fn write_ipc(path: &Path, batches: i16) {
    let schema = zoned_batch(0).schema();
    let mut w =
        arrow_ipc::writer::FileWriter::try_new(File::create(path).unwrap(), &schema).unwrap();
    for b in 0..batches {
        w.write(&zoned_batch(b * 3)).unwrap();
    }
    w.finish().unwrap();
}

#[test]
fn test_load_streams_batches_nullable_and_naive_utc() -> Result<()> {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let scratch = Scratch::new("stream");
    let pq = scratch.path("t.parquet");
    let ipc = scratch.path("t.arrow");
    write_parquet(&pq, 3);
    write_ipc(&ipc, 4);
    let (_env, conn) = setup_connection()?;

    assert_eq!(
        single_i64(
            &conn,
            &format!("SELECT bolt_load_parquet('w2d_pq', '{}')", lit(&pq))
        )?,
        9
    );
    assert_eq!(
        single_i64(
            &conn,
            &format!("SELECT bolt_load_arrow_ipc('w2d_ipc', '{}')", lit(&ipc))
        )?,
        12
    );

    // The batches were registered as they were decoded, not concatenated.
    let (schema, batches) = BoltRuntime::global().get_table_batches("w2d_ipc")?;
    assert!(
        batches.len() > 1,
        "expected several batches, got {}",
        batches.len()
    );
    assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 12);
    // Every field is nullable, and the zone is gone (naive UTC, values unchanged).
    assert!(
        schema.fields().iter().all(|f| f.is_nullable()),
        "{schema:?}"
    );
    assert_eq!(
        schema.field_with_name("ts").unwrap().data_type(),
        &DataType::Timestamp(TimeUnit::Microsecond, None)
    );
    // Stored as naive UTC; the column records its original type TIMESTAMPTZ (RFC-008,
    // W2-C-002), which bolt_scan shows once the RFC-008 read side is wired (W2-C-001).
    let shown = single_string(&conn, "SELECT typeof(ts) FROM bolt_scan('w2d_pq') LIMIT 1")?;
    assert!(
        matches!(
            shown.as_deref(),
            Some("TIMESTAMP") | Some("TIMESTAMP WITH TIME ZONE")
        ),
        "{shown:?}"
    );
    assert_eq!(
        single_i64(
            &conn,
            "SELECT epoch_us(ts) FROM bolt_scan('w2d_pq') WHERE id = 0"
        )?,
        1_000_000
    );
    assert_eq!(
        single_i64(&conn, "SELECT SUM(id)::BIGINT FROM bolt_scan('w2d_pq')")?,
        (0..9).sum::<i64>()
    );

    // W1-E-010 repro: appending a NULL id into a table loaded from a NOT NULL column.
    conn.execute(
        "COPY (SELECT NULL::INTEGER AS id, NULL::TIMESTAMP AS ts, NULL::VARCHAR AS label) \
         TO 'w2d_pq' (FORMAT bolt, MODE 'append', USE_TMP_FILE FALSE)",
        Parameters::None,
    )?;
    assert_eq!(
        single_i64(
            &conn,
            "SELECT count(*) FROM bolt_scan('w2d_pq') WHERE id IS NULL"
        )?,
        1
    );

    BoltRuntime::global().drop_table("w2d_pq")?;
    BoltRuntime::global().drop_table("w2d_ipc")?;
    Ok(())
}

#[test]
fn test_date64_with_time_of_day_is_rejected() -> Result<()> {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let scratch = Scratch::new("date64");
    let ipc = scratch.path("d.arrow");
    let schema = Arc::new(Schema::new(vec![Field::new("d", DataType::Date64, true)]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Date64Array::from(vec![Some(0), Some(43_200_000)]))],
    )
    .unwrap();
    let mut w =
        arrow_ipc::writer::FileWriter::try_new(File::create(&ipc).unwrap(), &schema).unwrap();
    w.write(&batch).unwrap();
    w.finish().unwrap();
    drop(w);

    let (_env, conn) = setup_connection()?;
    expect_error(
        &conn,
        &format!("SELECT bolt_load_arrow_ipc('w2d_date64', '{}')", lit(&ipc)),
        &["bolt_load_arrow_ipc", "w2d_date64", "whole number of days"],
    );
    assert!(!is_registered("w2d_date64")?);
    Ok(())
}

#[test]
fn test_file_urls_are_read_through_duckdb() -> Result<()> {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let scratch = Scratch::new("url");
    let pq = scratch.path("u.parquet");
    let ipc = scratch.path("u.arrow");
    write_parquet(&pq, 2);
    write_ipc(&ipc, 2);
    let (_env, conn) = setup_connection()?;

    // `file:` URLs are resolved by DuckDB's file system, not by std::fs.
    assert_eq!(
        single_i64(
            &conn,
            &format!(
                "SELECT bolt_load_parquet('w2d_url_pq', '{}')",
                file_url(&pq)
            )
        )?,
        6
    );
    assert_eq!(
        single_i64(
            &conn,
            &format!(
                "SELECT bolt_load_arrow_ipc('w2d_url_ipc', '{}')",
                file_url(&ipc)
            )
        )?,
        6
    );
    assert_eq!(
        single_i64(
            &conn,
            "SELECT SUM(id)::BIGINT FROM bolt_scan('w2d_url_ipc')"
        )?,
        (0..6).sum::<i64>()
    );

    // A URL nothing can resolve fails and registers nothing.
    expect_error(
        &conn,
        "SELECT bolt_load_parquet('w2d_url_bad', 'nosuchscheme://bucket/x.parquet')",
        &["bolt_load_parquet", "w2d_url_bad"],
    );
    assert!(!is_registered("w2d_url_bad")?);

    // With external access disabled DuckDB itself refuses the URL.
    conn.execute("SET enable_external_access = false", Parameters::None)?;
    expect_error(
        &conn,
        &format!(
            "SELECT bolt_load_parquet('w2d_url_denied', '{}')",
            file_url(&pq)
        ),
        &["bolt_load_parquet", "w2d_url_denied"],
    );
    assert!(!is_registered("w2d_url_denied")?);

    BoltRuntime::global().drop_table("w2d_url_pq")?;
    BoltRuntime::global().drop_table("w2d_url_ipc")?;
    Ok(())
}

#[test]
fn test_disabled_filesystems_is_honored_for_local_loads() -> Result<()> {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let scratch = Scratch::new("disabled_fs");
    let pq = scratch.path("t.parquet");
    write_parquet(&pq, 1);
    // A fresh database: a disabled file system cannot be re-enabled.
    let (_env, conn) = setup_connection()?;
    conn.execute(
        "SET disabled_filesystems = 'LocalFileSystem'",
        Parameters::None,
    )?;
    expect_error(
        &conn,
        &format!("SELECT bolt_load_parquet('w2d_nofs', '{}')", lit(&pq)),
        &["bolt_load_parquet", "w2d_nofs"],
    );
    assert!(!is_registered("w2d_nofs")?);
    Ok(())
}

#[test]
fn test_mode_parsing_is_consistent_across_functions() -> Result<()> {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let scratch = Scratch::new("modes");
    let pq = scratch.path("m.parquet");
    let ipc = scratch.path("m.arrow");
    write_parquet(&pq, 1);
    write_ipc(&ipc, 1);
    let pq_s = lit(&pq);
    let ipc_s = lit(&ipc);
    let (_env, conn) = setup_connection()?;
    BoltRuntime::global().register_table(
        "w2d_src",
        duckdb_bolt_core::ingest::normalize_batch_for_bolt(&zoned_batch(0))?,
        duckdb_bolt_core::runtime::RegisterMode::Replace,
    )?;
    let src = "SELECT id FROM w2d_src";

    // Accepted spellings: case-insensitive, trimmed, with aliases.
    for (sql, rows) in [
        (format!("SELECT bolt_register_sql('w2d_rs', '{src}')"), 3),
        (
            format!("SELECT bolt_register_sql('w2d_rs', '{src}', ' Append ')"),
            6,
        ),
        (
            format!("SELECT bolt_register_sql('w2d_rs', '{src}', 'OVERWRITE')"),
            3,
        ),
        (
            format!("SELECT bolt_register_sql('w2d_rs', '{src}', 'replace')"),
            3,
        ),
        (
            format!("SELECT bolt_load_parquet('w2d_lp', '{pq_s}', 'REPLACE')"),
            3,
        ),
        (
            format!("SELECT bolt_load_parquet('w2d_lp', '{pq_s}', 'append ')"),
            6,
        ),
        (
            format!("SELECT bolt_load_arrow_ipc('w2d_li', '{ipc_s}', 'Create')"),
            3,
        ),
        (
            format!("SELECT bolt_load_arrow_ipc('w2d_li2', '{ipc_s}', 'error')"),
            3,
        ),
    ] {
        assert_eq!(single_i64(&conn, &sql)?, rows, "{sql}");
    }

    // 'create' / 'error' refuse an existing table, for every function.
    for sql in [
        format!("SELECT bolt_register_sql('w2d_rs', '{src}', 'create')"),
        format!("SELECT bolt_load_parquet('w2d_lp', '{pq_s}', 'ERROR')"),
        format!("SELECT bolt_load_arrow_ipc('w2d_li', '{ipc_s}', 'create')"),
    ] {
        assert!(query_error(&conn, &sql).is_some(), "{sql} must fail");
    }

    // One message for an invalid mode, naming the function and the table, and nothing
    // is registered (the mode is checked before the file is read or the query runs).
    for (sql, function) in [
        (
            format!("SELECT bolt_register_sql('w2d_bad_rs', '{src}', 'bogus')"),
            "bolt_register_sql",
        ),
        (
            format!("SELECT bolt_load_parquet('w2d_bad_lp', '{pq_s}', 'bogus')"),
            "bolt_load_parquet",
        ),
        (
            format!("SELECT bolt_load_arrow_ipc('w2d_bad_li', '{ipc_s}', 'bogus')"),
            "bolt_load_arrow_ipc",
        ),
    ] {
        expect_error(
            &conn,
            &sql,
            &[
                function,
                "w2d_bad_",
                "Invalid Bolt",
                "bogus",
                "'replace', 'append', or 'create'",
            ],
        );
    }
    for name in ["w2d_bad_rs", "w2d_bad_lp", "w2d_bad_li"] {
        assert!(!is_registered(name)?, "{name}");
    }

    // A NULL mode is NULL with no side effect, like every other NULL argument.
    for sql in [
        format!("SELECT bolt_register_sql('w2d_null_rs', '{src}', NULL)"),
        format!("SELECT bolt_load_parquet('w2d_null_lp', '{pq_s}', NULL)"),
    ] {
        assert_eq!(i64_column(&conn, &sql)?, vec![None], "{sql}");
    }
    assert!(!is_registered("w2d_null_rs")?);
    assert!(!is_registered("w2d_null_lp")?);

    for name in ["w2d_src", "w2d_rs", "w2d_lp", "w2d_li", "w2d_li2"] {
        BoltRuntime::global().drop_table(name)?;
    }
    Ok(())
}

#[test]
fn test_settings_follow_the_process_wide_runtime() -> Result<()> {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let scratch = Scratch::new("settings");
    let ptx = scratch.path("ptx");
    std::fs::create_dir_all(&ptx).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&ptx, std::fs::Permissions::from_mode(0o700)).unwrap();
    }

    let (_env_a, a) = setup_connection()?;
    a.execute("SET bolt_memory_budget = 1048576", Parameters::None)?;
    a.execute("SET bolt_enable_tracing = true", Parameters::None)?;
    assert_eq!(
        single_i64(&a, "SELECT memory_budget_bytes FROM bolt_pool_stats()")?,
        1_048_576
    );

    // A second database in the same process shows what the shared runtime uses.
    let (_env_b, b) = setup_connection()?;
    assert_eq!(
        single_i64(&b, "SELECT current_setting('bolt_memory_budget')::BIGINT")?,
        1_048_576
    );
    assert_eq!(
        single_string(&b, "SELECT current_setting('bolt_enable_tracing')::VARCHAR")?.as_deref(),
        Some("true")
    );

    // RESET restores the documented default, not the load-time default, in both the
    // setting and the runtime.
    b.execute("RESET bolt_memory_budget", Parameters::None)?;
    b.execute("RESET bolt_enable_tracing", Parameters::None)?;
    assert_eq!(
        single_i64(&b, "SELECT current_setting('bolt_memory_budget')::BIGINT")?,
        0
    );
    assert_eq!(
        single_i64(&b, "SELECT memory_budget_bytes FROM bolt_pool_stats()")?,
        0
    );
    assert_eq!(
        single_string(&b, "SELECT current_setting('bolt_enable_tracing')::VARCHAR")?.as_deref(),
        Some("false")
    );
    assert!(!BoltRuntime::global().config()?.enable_tracing);

    // The PTX cache directory is shown exactly as the runtime stores it (trimmed).
    b.execute(
        format!("SET bolt_ptx_cache_dir = '  {}  '", lit(&ptx)),
        Parameters::None,
    )?;
    assert_eq!(
        single_string(&b, "SELECT current_setting('bolt_ptx_cache_dir')")?,
        Some(ptx.to_string_lossy().into_owned())
    );
    assert_eq!(
        BoltRuntime::global().config()?.ptx_cache_dir,
        Some(ptx.clone())
    );
    // A database loaded now shows the directory too; RESET disables the cache.
    let (_env_c, c) = setup_connection()?;
    assert_eq!(
        single_string(&c, "SELECT current_setting('bolt_ptx_cache_dir')")?,
        Some(ptx.to_string_lossy().into_owned())
    );
    c.execute("RESET bolt_ptx_cache_dir", Parameters::None)?;
    assert_eq!(
        single_string(&c, "SELECT current_setting('bolt_ptx_cache_dir')")?,
        Some(String::new())
    );
    assert_eq!(BoltRuntime::global().config()?.ptx_cache_dir, None);

    // A refused value leaves both the setting and the runtime unchanged.
    assert!(
        c.execute("SET bolt_memory_budget = -5", Parameters::None)
            .is_err()
    );
    assert_eq!(
        single_i64(&c, "SELECT current_setting('bolt_memory_budget')::BIGINT")?,
        0
    );
    assert_eq!(BoltRuntime::global().config()?.memory_budget, None);

    // Session scope is refused; the descriptions say the options are process-wide.
    assert!(
        c.execute("SET SESSION bolt_device = 0", Parameters::None)
            .is_err()
    );
    let description = single_string(
        &c,
        "SELECT description FROM duckdb_settings() WHERE name = 'bolt_device'",
    )?
    .unwrap_or_default();
    assert!(description.contains("Process-wide"), "{description}");

    // Leave the shared runtime at its defaults for other tests.
    for opt in [
        "bolt_device",
        "bolt_memory_budget",
        "bolt_ptx_cache_dir",
        "bolt_enable_tracing",
    ] {
        c.execute(format!("RESET {opt}"), Parameters::None)?;
    }
    Ok(())
}

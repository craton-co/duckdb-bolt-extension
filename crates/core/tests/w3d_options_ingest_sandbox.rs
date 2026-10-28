// SPDX-License-Identifier: Apache-2.0

//! Regression tests for W3-D:
//!
//! * `bolt_*` settings are reconciled with the process-wide runtime at the start of every Bolt
//!   function call: a setting another database made stale is refreshed (W1-D-003), and a
//!   startup option is either refused by DuckDB or applied through the update callback
//!   (W2-D-004);
//! * `SET bolt_ptx_cache_dir = ''` and `SET bolt_enable_tracing = false` reach the runtime
//!   configuration (W2-E-009, option side);
//! * the active PTX cache directory is re-validated when it is used (W2-D-003 item 2);
//! * loaded files record RFC-008 original DuckDB types (W2-C-002), and keep field / schema
//!   metadata (W1-D-007 item 3);
//! * `file:` URLs get the extension's canonical sandbox check (W2-D-001 item 2);
//! * corrupt Arrow IPC footers are errors (W3-D-002), and rows loading the same path in one
//!   chunk each have their side effect (W1-D-005 item 4).
//!
//! Valid for both GPU and `cuda-stub` builds: nothing here depends on the backend.

use std::{
    fs::File,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, Once},
};

use arrow_array::{Int16Array, RecordBatch, StringArray, TimestampMillisecondArray, UInt32Array};
use arrow_schema::{DataType, Field, Schema, TimeUnit};
use duckdb_bolt_core::{arrow_bridge::DUCKDB_TYPE_METADATA_KEY, runtime::BoltRuntime};
use duckdb_loadable_macros::duckdb_entrypoint_c_api_v2;
use duckdb_neo::{
    Parameters, Result,
    connection::Connection,
    environment::{Environment, StorageLocation},
    extension::{Extension, register_static_extension},
    instance::{AttachBuilder, InstanceBuilder},
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

fn expect_error(conn: &Connection, sql: &str, needles: &[&str]) -> String {
    let msg = query_error(conn, sql).unwrap_or_else(|| panic!("expected failure: {sql}"));
    for needle in needles {
        assert!(
            msg.contains(needle),
            "error for `{sql}` should contain `{needle}`, got: {msg}"
        );
    }
    msg
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

/// Any Bolt function call (runs the settings sync).
fn touch_bolt(conn: &Connection) -> Result<()> {
    single_string(conn, "SELECT bolt_version()")?;
    Ok(())
}

/// A scratch directory removed on drop.
struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("bolt_w3d_{tag}_{}", std::process::id()));
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

#[cfg(unix)]
fn chmod(path: &Path, mode: u32) {
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode)).unwrap();
}

/// A directory that passes `bolt_ptx_cache_dir` validation on every platform.
fn private_dir(path: &Path) {
    std::fs::create_dir_all(path).unwrap();
    #[cfg(unix)]
    chmod(path, 0o700);
}

fn reset_all(conn: &Connection) -> Result<()> {
    for opt in [
        "bolt_device",
        "bolt_memory_budget",
        "bolt_ptx_cache_dir",
        "bolt_enable_tracing",
    ] {
        conn.execute(format!("RESET {opt}"), Parameters::None)?;
    }
    Ok(())
}

/// `id SMALLINT NOT NULL`, `u UINTEGER`, `ts TIMESTAMP_MS WITH TIME ZONE`, `label VARCHAR`;
/// field `id` carries extra metadata and the schema carries `app.schema`.
fn typed_batch() -> RecordBatch {
    let tz: Arc<str> = Arc::from("UTC");
    let id = Field::new("id", DataType::Int16, false)
        .with_metadata([("app.field".to_string(), "kept".to_string())].into());
    let schema = Arc::new(Schema::new_with_metadata(
        vec![
            id,
            Field::new("u", DataType::UInt32, true),
            Field::new(
                "ts",
                DataType::Timestamp(TimeUnit::Millisecond, Some(tz.clone())),
                true,
            ),
            Field::new("label", DataType::Utf8, true),
        ],
        [("app.schema".to_string(), "kept".to_string())].into(),
    ));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int16Array::from(vec![1, 2, 3])),
            Arc::new(UInt32Array::from(vec![Some(4_000_000_000), None, Some(7)])),
            Arc::new(
                TimestampMillisecondArray::from(vec![Some(1_000), None, Some(3_000)])
                    .with_timezone(tz),
            ),
            Arc::new(StringArray::from(vec![Some("a"), None, Some("c")])),
        ],
    )
    .unwrap()
}

fn write_parquet(path: &Path) {
    let batch = typed_batch();
    let mut w =
        parquet::arrow::ArrowWriter::try_new(File::create(path).unwrap(), batch.schema(), None)
            .unwrap();
    w.write(&batch).unwrap();
    w.close().unwrap();
}

fn write_ipc(path: &Path) {
    let batch = typed_batch();
    let mut w =
        arrow_ipc::writer::FileWriter::try_new(File::create(path).unwrap(), &batch.schema())
            .unwrap();
    w.write(&batch).unwrap();
    w.finish().unwrap();
}

fn original_type(schema: &Schema, column: &str) -> Option<String> {
    schema
        .field_with_name(column)
        .unwrap()
        .metadata()
        .get(DUCKDB_TYPE_METADATA_KEY)
        .cloned()
}

#[test]
fn test_stale_setting_is_refreshed_by_a_bolt_call() -> Result<()> {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let (_env_a, a) = setup_connection()?;
    let (_env_b, b) = setup_connection()?;
    reset_all(&b)?;
    a.execute("SET bolt_memory_budget = 0", Parameters::None)?;

    // Another database changes the shared runtime.
    b.execute("SET bolt_memory_budget = 3145728", Parameters::None)?;
    assert_eq!(
        BoltRuntime::global().config()?.memory_budget,
        Some(3_145_728)
    );
    // The first database still shows its own last value until it calls Bolt (W1-D-003)...
    assert_eq!(
        single_i64(&a, "SELECT current_setting('bolt_memory_budget')::BIGINT")?,
        0
    );
    touch_bolt(&a)?;
    // ...then it shows what the runtime uses, and the runtime was not changed back.
    assert_eq!(
        single_i64(&a, "SELECT current_setting('bolt_memory_budget')::BIGINT")?,
        3_145_728
    );
    assert_eq!(
        BoltRuntime::global().config()?.memory_budget,
        Some(3_145_728)
    );
    reset_all(&b)?;
    Ok(())
}

#[test]
fn test_startup_option_is_refused_or_applied() -> Result<()> {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    register_extension();
    let (_env0, c0) = setup_connection()?;
    reset_all(&c0)?;

    let env = Environment::new()?;
    let built = InstanceBuilder::new()
        .option("bolt_memory_budget", "2097152")
        .build(&env);
    match built {
        // DuckDB 2.0 checks unrecognized startup options before statically linked
        // extensions are loaded, so the open itself fails and nothing is bypassed.
        Err(e) => assert!(e.message.contains("bolt_memory_budget"), "{e}"),
        // A DuckDB that copies the value into the setting without the callback: the first
        // Bolt call applies it through the callback (W2-D-004).
        Ok(instance) => {
            instance.attach(AttachBuilder::new(StorageLocation::InMemory).make_default())?;
            let conn = instance.connect()?;
            touch_bolt(&conn)?;
            assert_eq!(
                BoltRuntime::global().config()?.memory_budget,
                Some(2_097_152)
            );
            assert_eq!(
                single_i64(
                    &conn,
                    "SELECT current_setting('bolt_memory_budget')::BIGINT"
                )?,
                2_097_152
            );
            reset_all(&conn)?;
        }
    }

    // An invalid startup value is never silently accepted.
    let env = Environment::new()?;
    if let Ok(instance) = InstanceBuilder::new()
        .option("bolt_memory_budget", "-5")
        .build(&env)
    {
        instance.attach(AttachBuilder::new(StorageLocation::InMemory).make_default())?;
        let conn = instance.connect()?;
        expect_error(&conn, "SELECT bolt_version()", &["bolt_memory_budget"]);
        assert_eq!(BoltRuntime::global().config()?.memory_budget, None);
        reset_all(&conn)?;
        touch_bolt(&conn)?;
    }
    Ok(())
}

#[test]
fn test_disabling_ptx_cache_and_tracing_reaches_the_runtime() -> Result<()> {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let scratch = Scratch::new("ptx_off");
    let ptx = scratch.path("ptx");
    private_dir(&ptx);
    let (_env, conn) = setup_connection()?;
    reset_all(&conn)?;

    conn.execute(
        format!("SET bolt_ptx_cache_dir = '{}'", lit(&ptx)),
        Parameters::None,
    )?;
    conn.execute("SET bolt_enable_tracing = true", Parameters::None)?;
    let cfg = BoltRuntime::global().config()?;
    assert_eq!(cfg.ptx_cache_dir, Some(ptx.clone()));
    assert!(cfg.enable_tracing);
    touch_bolt(&conn)?;

    // The empty value and `false` are applied (the runtime rebuilds the engine without them).
    conn.execute("SET bolt_ptx_cache_dir = ''", Parameters::None)?;
    conn.execute("SET bolt_enable_tracing = false", Parameters::None)?;
    let cfg = BoltRuntime::global().config()?;
    assert_eq!(cfg.ptx_cache_dir, None);
    assert!(!cfg.enable_tracing);
    assert_eq!(
        single_string(&conn, "SELECT current_setting('bolt_ptx_cache_dir')")?.as_deref(),
        Some("")
    );
    assert_eq!(
        single_string(
            &conn,
            "SELECT current_setting('bolt_enable_tracing')::VARCHAR"
        )?
        .as_deref(),
        Some("false")
    );
    // Blank is the same as empty.
    conn.execute(
        format!("SET bolt_ptx_cache_dir = '{}'", lit(&ptx)),
        Parameters::None,
    )?;
    conn.execute("SET bolt_ptx_cache_dir = '   '", Parameters::None)?;
    assert_eq!(BoltRuntime::global().config()?.ptx_cache_dir, None);
    touch_bolt(&conn)?;
    reset_all(&conn)?;
    Ok(())
}

#[test]
fn test_ptx_cache_dir_is_rechecked_when_used() -> Result<()> {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let scratch = Scratch::new("ptx_recheck");
    let ptx = scratch.path("ptx");
    private_dir(&ptx);
    let (_env, conn) = setup_connection()?;
    reset_all(&conn)?;
    conn.execute(
        format!("SET bolt_ptx_cache_dir = '{}'", lit(&ptx)),
        Parameters::None,
    )?;
    touch_bolt(&conn)?;

    // Replace the directory with a regular file after it was validated.
    std::fs::remove_dir_all(&ptx).unwrap();
    std::fs::write(&ptx, b"not a directory").unwrap();
    expect_error(
        &conn,
        "SELECT bolt_version()",
        &[
            "bolt_version",
            "persistent PTX cache was disabled",
            "not a directory",
        ],
    );
    // The cache is off in the runtime and in the setting; the next call succeeds.
    assert_eq!(BoltRuntime::global().config()?.ptx_cache_dir, None);
    assert_eq!(
        single_string(&conn, "SELECT current_setting('bolt_ptx_cache_dir')")?.as_deref(),
        Some("")
    );
    touch_bolt(&conn)?;

    // Unix: a directory that became group/world-writable is refused the same way.
    #[cfg(unix)]
    {
        let ptx2 = scratch.path("ptx2");
        private_dir(&ptx2);
        conn.execute(
            format!("SET bolt_ptx_cache_dir = '{}'", lit(&ptx2)),
            Parameters::None,
        )?;
        chmod(&ptx2, 0o777);
        expect_error(&conn, "SELECT bolt_version()", &["world-writable"]);
        assert_eq!(BoltRuntime::global().config()?.ptx_cache_dir, None);
        chmod(&ptx2, 0o700);
    }
    reset_all(&conn)?;
    Ok(())
}

#[test]
fn test_loaded_files_record_original_duckdb_types() -> Result<()> {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let scratch = Scratch::new("rfc008");
    let pq = scratch.path("t.parquet");
    let ipc = scratch.path("t.arrow");
    write_parquet(&pq);
    write_ipc(&ipc);
    let (_env, conn) = setup_connection()?;

    assert_eq!(
        single_i64(
            &conn,
            &format!("SELECT bolt_load_parquet('w3d_pq', '{}')", lit(&pq))
        )?,
        3
    );
    assert_eq!(
        single_i64(
            &conn,
            &format!("SELECT bolt_load_arrow_ipc('w3d_ipc', '{}')", lit(&ipc))
        )?,
        3
    );
    for table in ["w3d_pq", "w3d_ipc"] {
        let (schema, batches) = BoltRuntime::global().get_table_batches(table)?;
        // Widened storage types...
        assert_eq!(
            schema.field_with_name("id").unwrap().data_type(),
            &DataType::Int32
        );
        assert_eq!(
            schema.field_with_name("u").unwrap().data_type(),
            &DataType::Int64
        );
        assert_eq!(
            schema.field_with_name("ts").unwrap().data_type(),
            &DataType::Timestamp(TimeUnit::Microsecond, None)
        );
        // ...with the original DuckDB types recorded (RFC-008).
        assert_eq!(
            original_type(&schema, "id").as_deref(),
            Some("SMALLINT"),
            "{table}"
        );
        assert_eq!(
            original_type(&schema, "u").as_deref(),
            Some("UINTEGER"),
            "{table}"
        );
        assert_eq!(
            original_type(&schema, "ts").as_deref(),
            Some("TIMESTAMPTZ"),
            "{table}"
        );
        assert_eq!(
            original_type(&schema, "label").as_deref(),
            Some("VARCHAR"),
            "{table}"
        );
        // Values are unchanged (ms -> us, checked; UINTEGER above i32::MAX kept).
        assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 3);
    }
    // IPC keeps field and schema metadata as written (W1-D-007 item 3).
    let (schema, _) = BoltRuntime::global().get_table_batches("w3d_ipc")?;
    assert_eq!(
        schema
            .field_with_name("id")
            .unwrap()
            .metadata()
            .get("app.field")
            .map(String::as_str),
        Some("kept")
    );
    assert_eq!(
        schema.metadata().get("app.schema").map(String::as_str),
        Some("kept")
    );
    assert_eq!(
        single_i64(
            &conn,
            "SELECT epoch_ms(ts)::BIGINT FROM bolt_scan('w3d_pq') WHERE id = 1"
        )?,
        1_000
    );
    assert_eq!(
        single_i64(
            &conn,
            "SELECT u::BIGINT FROM bolt_scan('w3d_ipc') WHERE id = 1"
        )?,
        4_000_000_000
    );
    BoltRuntime::global().drop_table("w3d_pq")?;
    BoltRuntime::global().drop_table("w3d_ipc")?;
    Ok(())
}

#[test]
fn test_rows_loading_the_same_path_each_take_effect() -> Result<()> {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let scratch = Scratch::new("same_path");
    let pq = scratch.path("t.parquet");
    write_parquet(&pq);
    let (_env, conn) = setup_connection()?;
    conn.execute("SET threads = 1", Parameters::None)?;
    single_i64(
        &conn,
        &format!("SELECT bolt_load_parquet('w3d_rep', '{}')", lit(&pq)),
    )?;
    // Three rows in one chunk append the (once decoded) file three times.
    assert_eq!(
        i64_column(
            &conn,
            &format!(
                "SELECT bolt_load_parquet('w3d_rep', '{}', 'append') FROM range(3)",
                lit(&pq)
            )
        )?,
        vec![Some(6), Some(9), Some(12)]
    );
    assert_eq!(
        single_i64(&conn, "SELECT count(*) FROM bolt_scan('w3d_rep')")?,
        12
    );
    BoltRuntime::global().drop_table("w3d_rep")?;
    Ok(())
}

#[test]
fn test_corrupt_arrow_ipc_footer_is_an_error() -> Result<()> {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let scratch = Scratch::new("bad_ipc");
    let ipc = scratch.path("bad.arrow");
    // Valid magic at both ends, but the footer length points far outside the file.
    let mut data = b"ARROW1\0\0".to_vec();
    data.extend_from_slice(&[0u8; 32]);
    data.extend_from_slice(&i32::MAX.to_le_bytes());
    data.extend_from_slice(b"ARROW1");
    std::fs::write(&ipc, &data).unwrap();
    let (_env, conn) = setup_connection()?;
    expect_error(
        &conn,
        &format!("SELECT bolt_load_arrow_ipc('w3d_bad', '{}')", lit(&ipc)),
        &["bolt_load_arrow_ipc", "w3d_bad", "footer"],
    );
    assert!(BoltRuntime::global().table_info("w3d_bad")?.is_none());
    Ok(())
}

#[cfg(unix)]
#[test]
fn test_file_url_cannot_escape_allowed_directories() -> Result<()> {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let scratch = Scratch::new("file_url");
    std::fs::create_dir_all(scratch.path("allowed")).unwrap();
    std::fs::create_dir_all(scratch.path("other")).unwrap();
    let outside = scratch.path("other").join("secret.parquet");
    let inside = scratch.path("allowed").join("ok.parquet");
    write_parquet(&outside);
    write_parquet(&inside);
    let link = scratch.path("allowed").join("link.parquet");
    std::os::unix::fs::symlink(&outside, &link).unwrap();

    let (_env, conn) = setup_connection()?;
    conn.execute(
        format!(
            "SET allowed_directories = ['{}']",
            lit(&scratch.path("allowed"))
        ),
        Parameters::None,
    )?;
    conn.execute("SET enable_external_access = false", Parameters::None)?;

    // W2-D-001 item 2: the URL form is refused like the plain path.
    for path in [lit(&link), format!("file://{}", lit(&link))] {
        expect_error(
            &conn,
            &format!("SELECT bolt_load_parquet('w3d_escape', '{path}')"),
            &["enable_external_access", "w3d_escape"],
        );
    }
    assert!(BoltRuntime::global().table_info("w3d_escape")?.is_none());
    // A file inside the allowed directory still loads through a file: URL.
    assert_eq!(
        single_i64(
            &conn,
            &format!(
                "SELECT bolt_load_parquet('w3d_inside', 'file://{}')",
                lit(&inside)
            )
        )?,
        3
    );
    BoltRuntime::global().drop_table("w3d_inside")?;
    Ok(())
}

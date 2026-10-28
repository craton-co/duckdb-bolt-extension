// SPDX-License-Identifier: Apache-2.0

//! Regression tests for W4-C:
//!
//! * Parquet files compressed with every codec DuckDB writes (Snappy, Gzip, ZSTD, LZ4_RAW,
//!   Brotli) and Arrow IPC files compressed with LZ4_FRAME / ZSTD load through
//!   `bolt_load_parquet` / `bolt_load_arrow_ipc` (W3-D-001, W2-D-002 item 1);
//! * a compressed Arrow IPC buffer that declares a huge uncompressed length is an error, not
//!   a process abort, and the declared size is bounded by DuckDB's `memory_limit` (W4-C-001);
//! * the settings sync that runs in every Bolt scalar call still refreshes stale settings and
//!   re-checks a replaced PTX cache directory after many fast-path calls (W4-C-002).
//!
//! Valid for both GPU and `cuda-stub` builds: nothing here depends on the backend.

use std::{
    fs::File,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, Once},
};

use arrow_array::{Int32Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use duckdb_bolt_core::{
    ingest::read_arrow_ipc_limited, runtime::BoltRuntime, sandbox::FileAccessPolicy,
};
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

/// A scratch directory removed on drop.
struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("bolt_w4c_{tag}_{}", std::process::id()));
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

/// A directory that passes `bolt_ptx_cache_dir` validation on every platform (on Windows
/// the temporary directory is inside `%LOCALAPPDATA%`).
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

const ROWS: usize = 10_000;

/// `ROWS` rows that compress well, so every codec really compresses.
fn compressible_batch() -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("z", DataType::Int32, true),
        Field::new("s", DataType::Utf8, true),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from_iter_values(
                (0..ROWS as i32).map(|i| i % 4),
            )),
            Arc::new(StringArray::from(vec!["bolt"; ROWS])),
        ],
    )
    .unwrap()
}

fn write_parquet(path: &Path, codec: parquet::basic::Compression) {
    let batch = compressible_batch();
    let props = parquet::file::properties::WriterProperties::builder()
        .set_compression(codec)
        .build();
    let mut w = parquet::arrow::ArrowWriter::try_new(
        File::create(path).unwrap(),
        batch.schema(),
        Some(props),
    )
    .unwrap();
    w.write(&batch).unwrap();
    w.close().unwrap();
}

fn write_ipc(path: &Path, codec: Option<arrow_ipc::CompressionType>) {
    let batch = compressible_batch();
    let options = arrow_ipc::writer::IpcWriteOptions::default()
        .try_with_compression(codec)
        .unwrap();
    let mut w = arrow_ipc::writer::FileWriter::try_new_with_options(
        File::create(path).unwrap(),
        &batch.schema(),
        options,
    )
    .unwrap();
    w.write(&batch).unwrap();
    w.finish().unwrap();
}

/// Absolute offset of the first really compressed buffer's 8-byte length prefix in the first
/// record batch of an IPC file.
fn first_compressed_prefix(file: &[u8]) -> usize {
    let n = file.len();
    let footer_len =
        i32::from_le_bytes([file[n - 10], file[n - 9], file[n - 8], file[n - 7]]) as usize;
    let footer = arrow_ipc::root_as_footer(&file[n - 10 - footer_len..n - 10]).unwrap();
    let block = footer.recordBatches().unwrap().get(0);
    let (offset, meta) = (block.offset() as usize, block.metaDataLength() as usize);
    // Writers of format 0.15+ start each message with the 0xFFFFFFFF continuation marker.
    let message = arrow_ipc::root_as_message(&file[offset + 8..offset + meta]).unwrap();
    let batch = message.header_as_record_batch().unwrap();
    batch
        .buffers()
        .unwrap()
        .iter()
        .filter(|b| b.length() >= 8)
        .map(|b| offset + meta + b.offset() as usize)
        .find(|&at| i64::from_le_bytes(file[at..at + 8].try_into().unwrap()) > 0)
        .expect("the writer compressed at least one buffer")
}

fn sum_of_z(conn: &Connection, table: &str) -> Result<i64> {
    single_i64(
        conn,
        &format!("SELECT sum(z)::BIGINT FROM bolt_scan('{table}')"),
    )
}

/// Sum of `i % 4` over `0..ROWS`.
const EXPECTED_SUM: i64 = (ROWS as i64 / 4) * 6;

#[test]
fn test_every_parquet_codec_loads() -> Result<()> {
    use parquet::basic::{BrotliLevel, Compression, GzipLevel, ZstdLevel};
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let scratch = Scratch::new("pq_codecs");
    let (_env, conn) = setup_connection()?;
    for (name, codec) in [
        ("uncompressed", Compression::UNCOMPRESSED),
        ("snappy", Compression::SNAPPY),
        ("gzip", Compression::GZIP(GzipLevel::default())),
        ("zstd", Compression::ZSTD(ZstdLevel::default())),
        ("lz4_raw", Compression::LZ4_RAW),
        ("lz4_hadoop", Compression::LZ4),
        ("brotli", Compression::BROTLI(BrotliLevel::default())),
    ] {
        let path = scratch.path(&format!("{name}.parquet"));
        write_parquet(&path, codec);
        let table = format!("w4c_pq_{name}");
        assert_eq!(
            single_i64(
                &conn,
                &format!("SELECT bolt_load_parquet('{table}', '{}')", lit(&path))
            )?,
            ROWS as i64,
            "{name}"
        );
        assert_eq!(sum_of_z(&conn, &table)?, EXPECTED_SUM, "{name}");
        BoltRuntime::global().drop_table(&table)?;
    }
    Ok(())
}

#[test]
fn test_parquet_files_written_by_duckdb_load_with_every_codec() -> Result<()> {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let scratch = Scratch::new("duckdb_codecs");
    let (_env, conn) = setup_connection()?;
    // DuckDB's `lz4` is LZ4_RAW.
    for codec in [
        "uncompressed",
        "snappy",
        "gzip",
        "zstd",
        "lz4",
        "lz4_raw",
        "brotli",
    ] {
        let path = scratch.path(&format!("{codec}.parquet"));
        let written = conn.execute(
            format!(
                "COPY (SELECT (range % 4)::INTEGER AS z FROM range({ROWS})) TO '{}' \
                 (FORMAT parquet, COMPRESSION {codec})",
                lit(&path)
            ),
            Parameters::None,
        );
        if written.is_err() {
            // This DuckDB build cannot write the codec; nothing to load.
            continue;
        }
        let table = format!("w4c_duck_{codec}");
        assert_eq!(
            single_i64(
                &conn,
                &format!("SELECT bolt_load_parquet('{table}', '{}')", lit(&path))
            )?,
            ROWS as i64,
            "{codec}"
        );
        assert_eq!(sum_of_z(&conn, &table)?, EXPECTED_SUM, "{codec}");
        BoltRuntime::global().drop_table(&table)?;
    }
    Ok(())
}

#[test]
fn test_lz4_and_zstd_arrow_ipc_files_load() -> Result<()> {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let scratch = Scratch::new("ipc_codecs");
    let (_env, conn) = setup_connection()?;
    for (name, codec) in [
        ("plain", None),
        ("lz4", Some(arrow_ipc::CompressionType::LZ4_FRAME)),
        ("zstd", Some(arrow_ipc::CompressionType::ZSTD)),
    ] {
        let path = scratch.path(&format!("{name}.arrow"));
        write_ipc(&path, codec);
        let table = format!("w4c_ipc_{name}");
        assert_eq!(
            single_i64(
                &conn,
                &format!("SELECT bolt_load_arrow_ipc('{table}', '{}')", lit(&path))
            )?,
            ROWS as i64,
            "{name}"
        );
        assert_eq!(sum_of_z(&conn, &table)?, EXPECTED_SUM, "{name}");
        // The same file through a file: URL (the local branch, W2-D-001 item 2).
        #[cfg(unix)]
        {
            let url_table = format!("{table}_url");
            assert_eq!(
                single_i64(
                    &conn,
                    &format!(
                        "SELECT bolt_load_arrow_ipc('{url_table}', 'file://{}')",
                        lit(&path)
                    )
                )?,
                ROWS as i64,
                "{name}"
            );
            BoltRuntime::global().drop_table(&url_table)?;
        }
        BoltRuntime::global().drop_table(&table)?;
    }
    Ok(())
}

#[test]
fn test_hostile_compressed_ipc_buffer_is_an_error_not_an_abort() -> Result<()> {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let scratch = Scratch::new("ipc_bomb");
    let good = scratch.path("good.arrow");
    write_ipc(&good, Some(arrow_ipc::CompressionType::ZSTD));
    let data = std::fs::read(&good).unwrap();
    let at = first_compressed_prefix(&data);

    // Without the check, arrow-ipc would call `Vec::with_capacity(i64::MAX as usize)`.
    let mut bomb = data.clone();
    bomb[at..at + 8].copy_from_slice(&i64::MAX.to_le_bytes());
    let bad = scratch.path("bomb.arrow");
    std::fs::write(&bad, &bomb).unwrap();

    let (_env, conn) = setup_connection()?;
    expect_error(
        &conn,
        &format!("SELECT bolt_load_arrow_ipc('w4c_bomb', '{}')", lit(&bad)),
        &["bolt_load_arrow_ipc", "w4c_bomb", "declares"],
    );
    assert!(BoltRuntime::global().table_info("w4c_bomb")?.is_none());

    // A plausible declared size above the limit (what `bolt_load_arrow_ipc` passes as
    // DuckDB's memory_limit) is refused before decompression...
    let policy = FileAccessPolicy::unrestricted();
    let good_path = good.to_string_lossy().into_owned();
    let err = read_arrow_ipc_limited(&good_path, &policy, None, Some(1024)).unwrap_err();
    assert!(err.message.contains("memory_limit"), "{err}");
    let loaded = read_arrow_ipc_limited(&good_path, &policy, None, Some(1 << 30)).unwrap();
    assert_eq!(loaded.num_rows(), ROWS);
    // ...and the default memory_limit allows it through SQL.
    assert_eq!(
        single_i64(
            &conn,
            &format!("SELECT bolt_load_arrow_ipc('w4c_limit', '{}')", lit(&good))
        )?,
        ROWS as i64
    );
    BoltRuntime::global().drop_table("w4c_limit")?;
    Ok(())
}

#[test]
fn test_settings_sync_still_refreshes_after_many_fast_path_calls() -> Result<()> {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let (_env_a, a) = setup_connection()?;
    let (_env_b, b) = setup_connection()?;
    reset_all(&b)?;
    a.execute("SET bolt_memory_budget = 0", Parameters::None)?;
    // Many chunks of fast-path syncs (every setting matches the runtime).
    assert_eq!(
        single_i64(
            &a,
            "SELECT count(DISTINCT bolt_version())::BIGINT FROM range(100000)"
        )?,
        1
    );
    // Another database changes the runtime: the next call of the first one refreshes.
    b.execute("SET bolt_memory_budget = 4194304", Parameters::None)?;
    single_string(&a, "SELECT bolt_version()")?;
    assert_eq!(
        single_i64(&a, "SELECT current_setting('bolt_memory_budget')::BIGINT")?,
        4_194_304
    );
    assert_eq!(
        BoltRuntime::global().config()?.memory_budget,
        Some(4_194_304)
    );
    reset_all(&b)?;
    Ok(())
}

#[test]
fn test_replaced_ptx_cache_dir_is_caught_after_fast_path_calls() -> Result<()> {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let scratch = Scratch::new("ptx_fast");
    let ptx = scratch.path("ptx");
    private_dir(&ptx);
    let (_env, conn) = setup_connection()?;
    reset_all(&conn)?;
    conn.execute(
        format!("SET bolt_ptx_cache_dir = '{}'", lit(&ptx)),
        Parameters::None,
    )?;
    // Several chunks: the first call runs the full check, the others only `stat`.
    single_i64(
        &conn,
        "SELECT count(bolt_version())::BIGINT FROM range(10000)",
    )?;
    // Replace the directory right away (well inside the full-recheck interval).
    std::fs::remove_dir_all(&ptx).unwrap();
    std::fs::write(&ptx, b"not a directory").unwrap();
    expect_error(
        &conn,
        "SELECT bolt_version()",
        &["persistent PTX cache was disabled", "not a directory"],
    );
    assert_eq!(BoltRuntime::global().config()?.ptx_cache_dir, None);
    single_string(&conn, "SELECT bolt_version()")?;

    // Unix: permissions changed after the fast path are caught the same way.
    #[cfg(unix)]
    {
        let ptx2 = scratch.path("ptx2");
        private_dir(&ptx2);
        conn.execute(
            format!("SET bolt_ptx_cache_dir = '{}'", lit(&ptx2)),
            Parameters::None,
        )?;
        single_string(&conn, "SELECT bolt_version()")?;
        chmod(&ptx2, 0o777);
        expect_error(&conn, "SELECT bolt_version()", &["world-writable"]);
        assert_eq!(BoltRuntime::global().config()?.ptx_cache_dir, None);
        chmod(&ptx2, 0o700);
    }
    reset_all(&conn)?;
    Ok(())
}

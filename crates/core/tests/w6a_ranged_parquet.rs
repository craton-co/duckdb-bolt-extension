// SPDX-License-Identifier: Apache-2.0

//! Regression tests for W6-A: Parquet files read through DuckDB's virtual file system use
//! ranged reads (a `ChunkReader` over the DuckDB file handle) instead of being buffered whole
//! (W2-D-002 item 2):
//!
//! * a local file read through DuckDB's file system decodes to the same data as the direct
//!   local read, including a file with several row groups;
//! * a column projection reads only the footer and that column's chunks;
//! * each column chunk is fetched with one request, however many pages it has (W6-A-001),
//!   and a chunk larger than the limit is streamed instead;
//! * truncated or corrupt footers and out-of-range column chunks are errors, not panics or
//!   huge allocations, and a range larger than the buffer limit is refused;
//! * DuckDB's sandbox (`enable_external_access`, `allowed_directories`) still refuses files
//!   outside the allow-list, through the VFS reader and through `bolt_load_parquet`.
//!
//! No `https://` test (W2-D-002 item 3): it needs the httpfs extension, which is not built
//! into the test DuckDB and would have to be downloaded, and the tests must run offline.
//!
//! Valid for both GPU and `cuda-stub` builds: nothing here depends on the backend.

use std::{
    fs::File,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, Once},
};

use arrow_array::{Array, Int64Array, RecordBatch, StringArray, cast::AsArray, types::Int64Type};
use arrow_schema::{DataType, Field, Schema};
use duckdb_bolt_core::{
    ingest::{IngestedTable, read_parquet, read_parquet_through_vfs},
    runtime::BoltRuntime,
    sandbox::FileAccessPolicy,
};
use duckdb_loadable_macros::duckdb_entrypoint_c_api_v2;
use duckdb_neo::{
    Parameters, Result,
    connection::Connection,
    environment::{Environment, StorageLocation},
    extension::{Extension, register_static_extension},
    file::FileSystem,
};
use parquet::{
    basic::Compression,
    file::{
        properties::WriterProperties,
        reader::{FileReader, SerializedFileReader},
    },
};

#[duckdb_entrypoint_c_api_v2(extension_name = "bolt", api_version = "2.0.0")]
fn bolt_test_entrypoint(extension: &Extension) -> Result<()> {
    duckdb_bolt_core::register(extension)
}

static REGISTER_ONCE: Once = Once::new();
// Serialize tests because Craton Bolt's runtime catalog and config are process-wide.
static TEST_MUTEX: Mutex<()> = Mutex::new(());

/// Rows in the test file, and rows per row group (four row groups).
const ROWS: i64 = 100_000;
const ROW_GROUP_ROWS: usize = 25_000;

fn register_extension() {
    REGISTER_ONCE.call_once(|| {
        // SAFETY: `duckdb_extension_bolt_describe` is emitted by `#[duckdb_entrypoint_c_api_v2]`,
        // which fills the descriptor as `duckdb_static_extension.h` specifies (the contract of
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

fn single_i64(conn: &Connection, sql: &str) -> Result<i64> {
    let mut result = conn.query(sql, Parameters::None)?;
    let chunk = result.next_chunk()?.expect("one result row");
    let vec = chunk.get_vector_at::<i64>(0)?;
    let value = vec.reader()?.get(0)?.copied();
    Ok(value.unwrap_or_else(|| panic!("`{sql}` returned NULL")))
}

/// A scratch directory removed on drop.
struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("bolt_w6a_{tag}_{}", std::process::id()));
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

fn s(p: &Path) -> &str {
    p.to_str().expect("UTF-8 scratch path")
}

/// Three columns of similar size (`a`, `b`: Int64; `c`: 12-byte strings), uncompressed and
/// without dictionaries, so each column chunk's size is predictable, in four row groups.
fn write_multi_row_group_parquet(path: &Path) {
    write_parquet_with_page_limit(path, None);
}

/// [`write_multi_row_group_parquet`], with data pages of at most about `page_bytes` bytes.
fn write_parquet_with_page_limit(path: &Path, page_bytes: Option<usize>) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int64, false),
        Field::new("b", DataType::Int64, false),
        Field::new("c", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from_iter_values(0..ROWS)),
            Arc::new(Int64Array::from_iter_values((0..ROWS).map(|i| i * 3))),
            Arc::new(StringArray::from_iter_values(
                (0..ROWS).map(|i| format!("row-{i:08}")),
            )),
        ],
    )
    .unwrap();
    let mut props = WriterProperties::builder()
        .set_compression(Compression::UNCOMPRESSED)
        .set_dictionary_enabled(false)
        .set_max_row_group_size(ROW_GROUP_ROWS);
    if let Some(page_bytes) = page_bytes {
        // The page size is checked after each write batch: keep batches small too.
        props = props
            .set_data_page_size_limit(page_bytes)
            .set_write_batch_size(64);
    }
    let props = props.build();
    let mut w =
        parquet::arrow::ArrowWriter::try_new(File::create(path).unwrap(), schema, Some(props))
            .unwrap();
    w.write(&batch).unwrap();
    w.close().unwrap();
}

/// Compressed size of column `index` summed over all row groups, and the row group count.
fn column_chunk_bytes(path: &Path, index: usize) -> (u64, usize) {
    let reader = SerializedFileReader::new(File::open(path).unwrap()).unwrap();
    let meta = reader.metadata();
    let bytes = meta
        .row_groups()
        .iter()
        .map(|rg| rg.column(index).compressed_size() as u64)
        .sum();
    (bytes, meta.num_row_groups())
}

fn i64_sum(table: &IngestedTable, column: usize) -> i64 {
    table
        .batches
        .iter()
        .map(|b| {
            b.column(column)
                .as_primitive::<Int64Type>()
                .iter()
                .map(|v| v.expect("no NULLs"))
                .sum::<i64>()
        })
        .sum()
}

#[test]
fn test_local_file_through_vfs_matches_direct_read() -> Result<()> {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let scratch = Scratch::new("match");
    let path = scratch.path("multi.parquet");
    write_multi_row_group_parquet(&path);
    let (_, row_groups) = column_chunk_bytes(&path, 0);
    assert_eq!(row_groups, 4, "the test file must have several row groups");

    let (_env, conn) = setup_connection()?;
    let fs = FileSystem::new(&conn)?;
    let (table, stats) = read_parquet_through_vfs(&fs, s(&path), None, None)?;
    let direct = read_parquet(s(&path), &FileAccessPolicy::unrestricted())?;

    assert_eq!(table.num_rows(), ROWS as usize);
    assert_eq!(table.schema, direct.schema);
    assert_eq!(table.batches, direct.batches);
    assert_eq!(i64_sum(&table, 0), (0..ROWS).sum::<i64>());
    assert_eq!(i64_sum(&table, 1), (0..ROWS).map(|i| i * 3).sum::<i64>());
    let last = table.batches.last().unwrap().column(2).as_string::<i32>();
    assert_eq!(last.value(last.len() - 1), format!("row-{:08}", ROWS - 1));

    // Ranged reads: several requests, and about the file once (each page reader may read
    // ahead up to 64 KiB past the end of its column chunk), never a single whole-file read.
    let file_len = std::fs::metadata(&path).unwrap().len();
    assert_eq!(stats.file_len, file_len);
    // W6-A-001: the footer (two reads), then one request per column chunk.
    assert!(stats.requests <= 2 + row_groups as u64 * 3, "{stats:?}");
    assert!(stats.requests >= row_groups as u64 * 3, "{stats:?}");
    assert!(
        stats.bytes <= file_len + 12 * (64 << 10),
        "{stats:?} (file is {file_len} bytes)"
    );
    Ok(())
}

#[test]
fn test_column_projection_reads_only_the_footer_and_that_column() -> Result<()> {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let scratch = Scratch::new("projection");
    let path = scratch.path("multi.parquet");
    write_multi_row_group_parquet(&path);
    let file_len = std::fs::metadata(&path).unwrap().len();
    let (b_bytes, row_groups) = column_chunk_bytes(&path, 1);

    let (_env, conn) = setup_connection()?;
    let fs = FileSystem::new(&conn)?;
    let (table, stats) = read_parquet_through_vfs(&fs, s(&path), Some(&["b"]), None)?;
    assert_eq!(table.schema.fields().len(), 1);
    assert_eq!(table.schema.field(0).name(), "b");
    assert_eq!(table.num_rows(), ROWS as usize);
    assert_eq!(i64_sum(&table, 0), (0..ROWS).map(|i| i * 3).sum::<i64>());

    // Column `b` is about a quarter of the file. Allow the footer and one read-ahead per
    // column chunk on top of its chunks.
    assert!(stats.bytes >= b_bytes, "{stats:?}, b = {b_bytes}");
    let footer_slack = 64 << 10;
    assert!(
        stats.bytes <= b_bytes + row_groups as u64 * (64 << 10) + footer_slack,
        "{stats:?}, b = {b_bytes}"
    );
    assert!(stats.bytes < file_len / 2, "{stats:?} (file is {file_len})");

    // Two columns, given out of order: decoded in file order.
    let (two, _) = read_parquet_through_vfs(&fs, s(&path), Some(&["c", "a"]), None)?;
    let names: Vec<_> = two
        .schema
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect();
    assert_eq!(names, ["a", "c"]);
    assert_eq!(two.num_rows(), ROWS as usize);

    // An unknown column is an error naming it.
    let err = read_parquet_through_vfs(&fs, s(&path), Some(&["nope"]), None).unwrap_err();
    assert!(err.message.contains("no column 'nope'"), "{err}");
    Ok(())
}

#[test]
fn test_corrupt_and_truncated_files_are_errors() -> Result<()> {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let scratch = Scratch::new("corrupt");
    let good = scratch.path("good.parquet");
    write_multi_row_group_parquet(&good);
    let data = std::fs::read(&good).unwrap();
    let n = data.len();

    // Footer length (the 4 bytes before the trailing `PAR1`).
    let mut huge_footer = data.clone();
    huge_footer[n - 8..n - 4].copy_from_slice(&0x7fff_ffffu32.to_le_bytes());
    // Footer metadata overwritten with garbage.
    let mut garbage_footer = data.clone();
    let footer_len = u32::from_le_bytes(data[n - 8..n - 4].try_into().unwrap()) as usize;
    for byte in &mut garbage_footer[n - 8 - footer_len..n - 8] {
        *byte = 0xa5;
    }
    // Data pages cut out: the footer points beyond the end of the file.
    let mut cut = data[..4].to_vec();
    cut.extend_from_slice(&data[n - 8 - footer_len..]);
    // Data pages overwritten.
    let mut bad_pages = data.clone();
    for byte in &mut bad_pages[4..n / 2] {
        *byte = 0xff;
    }
    let cases: [(&str, Vec<u8>); 7] = [
        ("empty", Vec::new()),
        ("tiny", b"PAR".to_vec()),
        ("truncated_end", data[..n - 1].to_vec()),
        ("huge_footer_len", huge_footer),
        ("garbage_footer", garbage_footer),
        ("pages_cut", cut),
        ("bad_pages", bad_pages),
    ];

    let (_env, conn) = setup_connection()?;
    let fs = FileSystem::new(&conn)?;
    for (name, bytes) in cases {
        let path = scratch.path(&format!("{name}.parquet"));
        std::fs::write(&path, bytes).unwrap();
        let err = read_parquet_through_vfs(&fs, s(&path), None, None)
            .err()
            .unwrap_or_else(|| panic!("{name}: expected an error"));
        assert!(err.message.contains(s(&path)), "{name}: {err}");
    }

    // A range larger than the buffer limit (DuckDB's memory_limit) is refused: the footer
    // metadata is the one range `parquet` asks for in one piece.
    let err = read_parquet_through_vfs(&fs, s(&good), None, Some(16)).unwrap_err();
    assert!(err.message.contains("memory_limit"), "{err}");
    // W7-B-002: a column chunk larger than the limit is refused before decoding, naming the
    // column...
    let limit = Some(footer_len as u64 + 1);
    let err = read_parquet_through_vfs(&fs, s(&good), None, limit).unwrap_err();
    assert!(err.message.contains("memory_limit"), "{err}");
    assert!(err.message.contains("column 'a' of row group 0"), "{err}");
    // ...unless the projection skips it: only the projected chunks are checked.
    let a_largest = largest_chunk_of(&good, Some(0));
    assert!(
        a_largest < largest_chunk(&good),
        "column c must be the widest"
    );
    let (only_a, _) = read_parquet_through_vfs(&fs, s(&good), Some(&["a"]), Some(a_largest))?;
    assert_eq!(only_a.num_rows(), ROWS as usize);
    let err = read_parquet_through_vfs(&fs, s(&good), None, Some(a_largest)).unwrap_err();
    assert!(err.message.contains("column 'c'"), "{err}");
    // ...and a limit that fits every chunk loads the file.
    let largest = largest_chunk(&good);
    let (table, _) = read_parquet_through_vfs(&fs, s(&good), None, Some(largest))?;
    assert_eq!(table.num_rows(), ROWS as usize);
    let err = read_parquet_through_vfs(&fs, s(&good), None, Some(largest - 1)).unwrap_err();
    assert!(err.message.contains("memory_limit"), "{err}");
    // A chunk declared outside the file is an invalid file, named before any page is read.
    let err = read_parquet_through_vfs(&fs, s(&scratch.path("pages_cut.parquet")), None, None)
        .unwrap_err();
    assert!(err.message.contains("lies outside the file"), "{err}");
    Ok(())
}

#[test]
fn test_sandbox_is_still_enforced() -> Result<()> {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let scratch = Scratch::new("sandbox");
    std::fs::create_dir_all(scratch.path("allowed")).unwrap();
    std::fs::create_dir_all(scratch.path("other")).unwrap();
    let inside = scratch.path("allowed").join("in.parquet");
    let outside = scratch.path("other").join("out.parquet");
    write_multi_row_group_parquet(&inside);
    write_multi_row_group_parquet(&outside);

    let (_env, conn) = setup_connection()?;
    conn.execute(
        format!(
            "SET allowed_directories = ['{}']",
            lit(&scratch.path("allowed"))
        ),
        Parameters::None,
    )?;
    conn.execute("SET enable_external_access = false", Parameters::None)?;

    // Through DuckDB's file system: DuckDB's own checks refuse the file before any read.
    let fs = FileSystem::new(&conn)?;
    let err = read_parquet_through_vfs(&fs, s(&outside), None, None).unwrap_err();
    assert!(err.message.contains("DuckDB refused to open"), "{err}");
    let (table, _) = read_parquet_through_vfs(&fs, s(&inside), None, None)?;
    assert_eq!(table.num_rows(), ROWS as usize);

    // Through SQL: the extension's canonical check and DuckDB's both still apply.
    for path in [
        lit(&outside),
        format!("file:///{}", lit(&outside).trim_start_matches('/')),
    ] {
        let msg = query_error(
            &conn,
            &format!("SELECT bolt_load_parquet('w6a_out', '{path}')"),
        )
        .unwrap_or_else(|| panic!("loading {path} should fail"));
        assert!(msg.contains("enable_external_access"), "{path}: {msg}");
    }
    assert!(BoltRuntime::global().table_info("w6a_out")?.is_none());
    assert_eq!(
        single_i64(
            &conn,
            &format!("SELECT bolt_load_parquet('w6a_in', '{}')", lit(&inside))
        )?,
        ROWS
    );
    BoltRuntime::global().drop_table("w6a_in")?;
    Ok(())
}

/// Size of the largest column chunk of the file.
fn largest_chunk(path: &Path) -> u64 {
    largest_chunk_of(path, None)
}

/// Size of the largest chunk of column `index` (of any column with `None`).
fn largest_chunk_of(path: &Path, index: Option<usize>) -> u64 {
    let reader = SerializedFileReader::new(File::open(path).unwrap()).unwrap();
    reader
        .metadata()
        .row_groups()
        .iter()
        .flat_map(|rg| {
            rg.columns()
                .iter()
                .enumerate()
                .filter(|(i, _)| index.is_none_or(|index| *i == index))
                .map(|(_, c)| c.byte_range().1)
        })
        .max()
        .unwrap()
}

/// Total data pages of column `index` over all row groups.
fn page_count(path: &Path, index: usize) -> usize {
    let reader = SerializedFileReader::new(File::open(path).unwrap()).unwrap();
    (0..reader.num_row_groups())
        .map(|rg| {
            reader
                .get_row_group(rg)
                .unwrap()
                .get_column_page_reader(index)
                .unwrap()
                .count()
        })
        .sum()
}

#[test]
fn test_column_chunks_are_fetched_with_one_request_each() -> Result<()> {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let scratch = Scratch::new("pages");
    let path = scratch.path("small_pages.parquet");
    write_parquet_with_page_limit(&path, Some(4096));
    let (b_bytes, row_groups) = column_chunk_bytes(&path, 1);
    let pages = page_count(&path, 1);
    assert!(
        pages >= row_groups * 20,
        "the test file must have many pages per chunk ({pages})"
    );

    let (_env, conn) = setup_connection()?;
    let fs = FileSystem::new(&conn)?;
    // Prefetched: footer (two reads) plus one request per chunk of column `b`.
    let (table, stats) = read_parquet_through_vfs(&fs, s(&path), Some(&["b"]), None)?;
    assert_eq!(i64_sum(&table, 0), (0..ROWS).map(|i| i * 3).sum::<i64>());
    assert!(stats.requests <= 2 + row_groups as u64, "{stats:?}");
    assert!(stats.bytes >= b_bytes, "{stats:?}");
    assert!(
        stats.bytes <= b_bytes + (64 << 10),
        "{stats:?}, b = {b_bytes}"
    );

    // A limit that fits one chunk but not a row group's chunks together: one chunk per row
    // group is prefetched, the others stream page by page. Still correct, with many more
    // requests.
    let limit = Some(largest_chunk(&path));
    let (streamed, streamed_stats) = read_parquet_through_vfs(&fs, s(&path), None, limit)?;
    let (prefetched, prefetched_stats) = read_parquet_through_vfs(&fs, s(&path), None, None)?;
    assert_eq!(streamed.batches, prefetched.batches);
    assert!(
        streamed_stats.requests >= pages as u64,
        "{streamed_stats:?}, {pages} pages per column"
    );
    assert!(
        streamed_stats.requests > prefetched_stats.requests * 5,
        "prefetch {prefetched_stats:?} vs streamed {streamed_stats:?}"
    );

    // The whole file, all columns: still one request per chunk, same data as the local read.
    let (all, all_stats) = read_parquet_through_vfs(&fs, s(&path), None, None)?;
    let direct = read_parquet(s(&path), &FileAccessPolicy::unrestricted())?;
    assert_eq!(all.batches, direct.batches);
    assert!(
        all_stats.requests <= 2 + 3 * row_groups as u64,
        "{all_stats:?}"
    );
    Ok(())
}

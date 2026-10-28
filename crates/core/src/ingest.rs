// SPDX-License-Identifier: Apache-2.0

//! Direct Parquet / Arrow IPC ingestion into the Craton Bolt catalog (RFC-004).
//!
//! Files are decoded on the host with the `parquet` and `arrow-ipc` crates, their column
//! types are normalized to the subset Craton Bolt supports, and the normalized batches
//! are handed to [`BoltRuntime::register_table_batches`] as they are, without being
//! concatenated into one batch.
//!
//! Where the bytes come from:
//!
//! * **Local paths** are opened with `std::fs` after the caller's DuckDB sandbox settings
//!   (`enable_external_access`, `allowed_directories`, `allowed_paths`) were checked by a
//!   [`FileAccessPolicy`] (which also re-checks the opened handle). When a DuckDB
//!   [`FileSystem`] is available (always, from SQL), a leading `~` is expanded by DuckDB
//!   and the file is additionally opened once through DuckDB's virtual file system, so
//!   DuckDB's own checks (including `disabled_filesystems`) apply as well.
//! * **`file:` URLs** that DuckDB maps to local paths (`file:///x`, `file:/x`,
//!   `file://localhost/x`) are reduced to that path and treated as local paths, so the
//!   canonical sandbox check applies to them too (W2-D-001 item 2).
//! * **Other URLs** (anything containing `://`: `s3://`, `https://`, ...) are read through
//!   DuckDB's virtual file system only, so httpfs, secrets and DuckDB's sandbox apply exactly
//!   as they do for `read_parquet`. Arrow IPC is streamed with positional reads through a
//!   1 MiB buffer. Parquet is read with ranged reads through a [`ChunkReader`] over the DuckDB
//!   file handle: the footer, then each decoded column chunk with one request (W6-A-001), so
//!   a file is never buffered whole (W2-D-002 item 2; see [`read_parquet_through_vfs`]).
//!
//! Arrow IPC files are validated before decoding (blocks must lie inside the file, buffers
//! inside their block, and the declared uncompressed size of LZ4 / ZSTD buffers is bounded),
//! so a corrupt or hostile file is an error instead of a panic or a huge allocation
//! (W3-D-002, W4-C-001).
//!
//! Codecs (W3-D-001): Parquet pages compressed with Snappy, Gzip, ZSTD, LZ4 (`LZ4_RAW` and
//! the deprecated Hadoop `LZ4`) or Brotli decode, and so do Arrow IPC buffers compressed with
//! LZ4_FRAME or ZSTD. LZO (Parquet) is not implemented by the `parquet` crate.
//!
//! Normalization (see [`normalized_schema`]):
//!
//! * integer, float, string, date and dictionary types are widened or unpacked to the types
//!   Craton Bolt stores; `Date64` values that are not whole days are an error;
//! * `Timestamp(unit, None)` is kept. `Timestamp(unit, tz)` becomes naive UTC: Arrow
//!   timestamps with a zone are UTC instants, so the values are unchanged, exactly as
//!   `COPY ... TO (FORMAT bolt)` stores a `TIMESTAMPTZ` column (`Timestamp(us, None)`;
//!   nanosecond zoned timestamps keep nanoseconds, second and millisecond ones are converted
//!   to microseconds, checked);
//! * every field is nullable, as in tables created by `COPY`, so `MODE 'append'` can mix
//!   both sources and appending NULLs never fails on a file's `NOT NULL` flag;
//! * **RFC-008**: every field records the DuckDB type the file's column has (what
//!   `read_parquet` / `read_arrow` would return: `SMALLINT` for `Int16`, `UINTEGER` for
//!   `UInt32`, `TIMESTAMPTZ` for a zoned timestamp, ...) under
//!   [`DUCKDB_TYPE_METADATA_KEY`], so `bolt_scan` can return the original types
//!   (W2-C-002). Any such key already in the file is replaced (the Arrow type is
//!   authoritative); other field metadata and the schema metadata are kept (W1-D-007 item 3).

use std::{
    io::{self, BufReader, Read, Seek, SeekFrom},
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
};

use arrow_array::{Array, ArrayRef, RecordBatch, cast::AsArray, types::Date64Type};
use arrow_cast::cast::CastOptions;

use crate::arrow_bridge::cast_with_options;
use arrow_schema::{DataType, Field, Schema, SchemaRef, TimeUnit};
use bytes::Bytes;
use duckdb_neo::{
    Error, FromValue, Result,
    connection::Context,
    file::{File as DuckDbFile, FileBuilder, FileSystem},
};
use parquet::{
    arrow::{ProjectionMask, arrow_reader::ParquetRecordBatchReaderBuilder},
    errors::ParquetError,
    file::{
        metadata::ParquetMetaData,
        reader::{ChunkReader, Length},
    },
};

use crate::arrow_bridge::{DUCKDB_TYPE_METADATA_KEY, OriginalDuckDbType};
use crate::runtime::{BoltRuntime, RegisterMode, TableSummary};
use crate::sandbox::{FileAccessPolicy, is_remote, local_path_of_file_url};

/// Upper bound on the rows decoded into one Parquet batch.
pub const MAX_PARQUET_BATCH_ROWS: usize = 1 << 20;

/// Lower bound on the rows decoded into one Parquet batch.
///
/// One row: a file whose rows are hundreds of MiB wide must still produce batches whose
/// `Utf8` offsets (`i32`) stay below 2 GiB, so no larger floor is safe.
const MIN_PARQUET_BATCH_ROWS: usize = 1;

/// Upper bound on the capacity reserved up front for the decoded batches of one file (the
/// row count comes from file metadata, which may be wrong or hostile).
const MAX_BATCH_VEC_RESERVE: usize = 1 << 12;

/// Read-ahead buffer for Arrow IPC files read through DuckDB's file system: small messages
/// (footer, batch metadata) would otherwise be one remote request each.
const REMOTE_IPC_BUFFER: usize = 1 << 20;

/// Target uncompressed size of one decoded Parquet batch.
///
/// Keeps each batch's `Utf8` offsets (`i32`) far from their 2 GiB limit, even for
/// string-heavy files, and bounds the size of each decode buffer.
const TARGET_PARQUET_BATCH_BYTES: usize = 256 << 20;

/// Read-ahead of the sequential readers a [`DuckDbChunkReader`] hands out (`parquet` reads
/// each page header byte by byte, then the page body). A read at least this large goes to
/// DuckDB directly, without the buffer.
const VFS_PARQUET_READ_AHEAD: usize = 64 << 10;

/// Largest column chunk a [`DuckDbChunkReader`] prefetches whole (also bounded by DuckDB's
/// `memory_limit`); larger chunks are streamed page by page.
const MAX_PREFETCHED_CHUNK_BYTES: u64 = 256 << 20;

/// Milliseconds per day, for `Date64` values.
const MILLIS_PER_DAY: i64 = 86_400_000;

/// Normalized batches of one ingested file, ready for [`BoltRuntime::register_table_batches`].
#[derive(Debug, Clone)]
pub struct IngestedTable {
    /// Normalized schema (every field nullable).
    pub schema: SchemaRef,
    /// Normalized batches, each with exactly `schema` and at least one row; empty for a
    /// file without rows.
    pub batches: Vec<RecordBatch>,
}

impl IngestedTable {
    /// Total number of rows over all batches.
    pub fn num_rows(&self) -> usize {
        self.batches.iter().map(RecordBatch::num_rows).sum()
    }

    /// Register the batches as `table_name` in the process-wide runtime.
    pub fn register(self, table_name: &str, mode: RegisterMode) -> Result<TableSummary> {
        BoltRuntime::global().register_table_batches(table_name, self.schema, self.batches, mode)
    }
}

/// The DuckDB type a column of Arrow type `dt` has when DuckDB itself reads the file
/// (`read_parquet`, `read_arrow`), for RFC-008 metadata. `None` when there is none Craton Bolt
/// can store.
pub fn original_duckdb_type_of(dt: &DataType) -> Option<OriginalDuckDbType> {
    use OriginalDuckDbType as O;
    Some(match dt {
        DataType::Boolean => O::Boolean,
        DataType::Int8 => O::TinyInt,
        DataType::Int16 => O::SmallInt,
        DataType::Int32 => O::Integer,
        DataType::Int64 => O::BigInt,
        DataType::UInt8 => O::UTinyInt,
        DataType::UInt16 => O::USmallInt,
        DataType::UInt32 => O::UInteger,
        DataType::UInt64 => O::UBigInt,
        // DuckDB has no half-precision type; it reads Float16 as FLOAT.
        DataType::Float16 | DataType::Float32 => O::Float,
        DataType::Float64 => O::Double,
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => O::Varchar,
        DataType::Date32 | DataType::Date64 => O::Date,
        DataType::Timestamp(TimeUnit::Second, None) => O::TimestampSec,
        DataType::Timestamp(TimeUnit::Millisecond, None) => O::TimestampMs,
        DataType::Timestamp(TimeUnit::Microsecond, None) => O::Timestamp,
        DataType::Timestamp(TimeUnit::Nanosecond, None) => O::TimestampNs,
        DataType::Timestamp(TimeUnit::Nanosecond, Some(_)) => O::TimestampTzNs,
        DataType::Timestamp(_, Some(_)) => O::TimestampTz,
        DataType::Decimal128(p, s) => {
            let scale = u8::try_from(*s).ok()?;
            if !(1..=38).contains(p) || scale > *p {
                return None;
            }
            O::Decimal { width: *p, scale }
        }
        DataType::Dictionary(_, value) => match value.as_ref() {
            DataType::Dictionary(_, _) => return None,
            other => return original_duckdb_type_of(other),
        },
        _ => return None,
    })
}

/// The normalized field for `field` with storage type `target`: nullable, its metadata kept,
/// and RFC-008 metadata recording the original DuckDB type (when it is consistent with
/// `target`; otherwise any stale key is removed).
fn normalized_field(field: &Field, target: DataType) -> Field {
    let mut metadata = field.metadata().clone();
    match original_duckdb_type_of(field.data_type()) {
        Some(original) if original.storage_arrow_type() == target => {
            metadata.insert(
                DUCKDB_TYPE_METADATA_KEY.to_string(),
                original.to_metadata_text(),
            );
        }
        _ => {
            metadata.remove(DUCKDB_TYPE_METADATA_KEY);
        }
    }
    Field::new(field.name(), target, true).with_metadata(metadata)
}

/// Compute the Craton Bolt schema for `schema`, failing on the first unsupported column.
///
/// Called before any data is decoded, so unsupported files fail fast. Every field of the
/// result is nullable and carries RFC-008 metadata; the schema metadata is kept (see the
/// module documentation).
pub fn normalized_schema(schema: &Schema) -> Result<SchemaRef> {
    if schema.fields().is_empty() {
        return Err(Error::invalid_input(
            "The file has no columns; Craton Bolt tables need at least one column",
        ));
    }
    let fields = schema
        .fields()
        .iter()
        .map(|field| {
            let target = crate::arrow_bridge::bolt_storage_arrow_type(field.data_type())
                .ok_or_else(|| {
                    Error::invalid_input(format!(
                        "Column '{}' has Arrow type {}, which Craton Bolt does not support; \
                     load the file with COPY (SELECT ... FROM read_parquet(...)) TO '<table>' \
                     (FORMAT bolt) and cast or drop that column",
                        field.name(),
                        field.data_type()
                    ))
                })?;
            Ok(normalized_field(field, target))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Arc::new(Schema::new_with_metadata(
        fields,
        schema.metadata().clone(),
    )))
}

/// Reject `Date64` values that are not whole days (the cast to `Date32` would truncate them).
fn check_date64_whole_days(name: &str, col: &ArrayRef) -> Result<()> {
    let values: &dyn Array = match col.data_type() {
        DataType::Date64 => col.as_ref(),
        DataType::Dictionary(_, value) if value.as_ref() == &DataType::Date64 => {
            col.as_any_dictionary().values().as_ref()
        }
        _ => return Ok(()),
    };
    let dates = values.as_primitive::<Date64Type>();
    if let Some(bad) = dates.iter().flatten().find(|ms| ms % MILLIS_PER_DAY != 0) {
        return Err(Error::invalid_input(format!(
            "Column '{name}' has a Date64 value of {bad} ms, which is not a whole number of \
             days; Craton Bolt stores dates as days and would truncate it"
        )));
    }
    Ok(())
}

/// Cast every column of `batch` to the matching field type of `target`.
fn normalize_batch_to(batch: &RecordBatch, target: &SchemaRef) -> Result<RecordBatch> {
    if batch.num_columns() != target.fields().len() {
        return Err(Error::api_error(format!(
            "Decoded batch has {} columns, expected {}",
            batch.num_columns(),
            target.fields().len()
        )));
    }
    let options = CastOptions {
        safe: false,
        ..Default::default()
    };
    let mut columns: Vec<ArrayRef> = Vec::with_capacity(batch.num_columns());
    for (dst_field, col) in target.fields().iter().zip(batch.columns()) {
        let wanted = dst_field.data_type();
        let out = if col.data_type() == wanted {
            col.clone()
        } else {
            check_date64_whole_days(dst_field.name(), col)?;
            cast_with_options(col, wanted, &options).map_err(|e| {
                Error::invalid_input(format!(
                    "Column '{}' cannot be converted from {} to {}: {e}",
                    dst_field.name(),
                    col.data_type(),
                    wanted
                ))
            })?
        };
        columns.push(out);
    }
    RecordBatch::try_new(target.clone(), columns)
        .map_err(|e| Error::api_error(format!("Failed to build normalized batch: {e}")))
}

/// Cast every column of `batch` to a Craton Bolt supported type.
///
/// Lossy narrowing (for example `UInt64` values above `i64::MAX`) is an error rather than
/// a silent NULL.
pub fn normalize_batch_for_bolt(batch: &RecordBatch) -> Result<RecordBatch> {
    let target = normalized_schema(batch.schema_ref())?;
    normalize_batch_to(batch, &target)
}

/// Rows per decoded Parquet batch: about [`TARGET_PARQUET_BATCH_BYTES`] of uncompressed data.
fn parquet_batch_rows(metadata: &ParquetMetaData, total_rows: usize) -> usize {
    // Saturating: the sizes come from file metadata, which may be wrong.
    let bytes: i64 = metadata
        .row_groups()
        .iter()
        .map(|rg| rg.total_byte_size().max(0))
        .fold(0i64, i64::saturating_add);
    let per_row = usize::try_from(bytes)
        .unwrap_or(usize::MAX)
        .checked_div(total_rows)
        .unwrap_or(0)
        .max(1);
    (TARGET_PARQUET_BATCH_BYTES / per_row)
        .clamp(MIN_PARQUET_BATCH_ROWS, MAX_PARQUET_BATCH_ROWS)
        .min(total_rows.max(1))
}

/// Decode a Parquet file batch by batch, normalizing each batch as soon as it is decoded.
///
/// Column chunks must lie inside the file and, with `limit` (DuckDB's `memory_limit`), be at
/// most that large ([`check_parquet_chunk_ranges`]).
fn decode_parquet<T: ChunkReader + 'static>(
    reader: T,
    path: &str,
    limit: Option<u64>,
) -> Result<IngestedTable> {
    let bounds = ChunkBounds {
        file_len: reader.len(),
        limit,
    };
    decode_parquet_projected(
        ParquetRecordBatchReaderBuilder::try_new(reader),
        path,
        None,
        bounds,
    )
}

/// What a Parquet file's column chunks are checked against before decoding (W7-B-002).
#[derive(Debug, Clone, Copy)]
struct ChunkBounds {
    file_len: u64,
    /// Largest column chunk accepted (DuckDB's `memory_limit`).
    limit: Option<u64>,
}

/// Check every column chunk the footer declares (only the leaves of `mask`, if given) before
/// decoding: it must lie inside the file and be at most `bounds.limit` bytes (W7-B-002).
///
/// `parquet` reads each chunk's pages with sizes from untrusted page headers; a chunk that
/// lies outside the file or is larger than the memory budget is refused up front, with the
/// column named, instead of being discovered (or allocated) during decoding.
fn check_parquet_chunk_ranges(
    metadata: &ParquetMetaData,
    mask: Option<&ProjectionMask>,
    bounds: ChunkBounds,
    path: &str,
) -> Result<()> {
    for (rg, row_group) in metadata.row_groups().iter().enumerate() {
        for (leaf, column) in row_group.columns().iter().enumerate() {
            if mask.is_some_and(|m| !m.leaf_included(leaf)) {
                continue;
            }
            let (start, len) = column.byte_range();
            if start
                .checked_add(len)
                .is_none_or(|end| end > bounds.file_len)
            {
                return Err(Error::invalid_input(format!(
                    "Invalid Parquet file '{path}': column '{}' of row group {rg} ({start}+{len}) \
                     lies outside the file ({} bytes)",
                    column.column_path().string(),
                    bounds.file_len
                )));
            }
            if let Some(limit) = bounds.limit
                && len > limit
            {
                return Err(Error::invalid_input(format!(
                    "Cannot load Parquet file '{path}': column '{}' of row group {rg} is {len} \
                     bytes, more than DuckDB's memory_limit ({limit} bytes)",
                    column.column_path().string()
                )));
            }
        }
    }
    Ok(())
}

/// [`decode_parquet`], decoding only the top-level `columns` (in file order) when given.
fn decode_parquet_projected<T: ChunkReader + 'static>(
    builder: std::result::Result<ParquetRecordBatchReaderBuilder<T>, ParquetError>,
    path: &str,
    columns: Option<&[&str]>,
    bounds: ChunkBounds,
) -> Result<IngestedTable> {
    let builder =
        builder.map_err(|e| Error::invalid_input(format!("Invalid Parquet file '{path}': {e}")))?;
    // The schema comes from the builder, not the reader: the reader's schema has no schema
    // metadata (W7-B-001).
    let (arrow_schema, mask) = match columns {
        None => (builder.schema().clone(), None),
        Some([]) => {
            return Err(Error::invalid_input(format!(
                "No columns requested from Parquet file '{path}'"
            )));
        }
        Some(names) => {
            let mut roots = names
                .iter()
                .map(|name| {
                    builder.schema().index_of(name).map_err(|_| {
                        Error::invalid_input(format!(
                            "Parquet file '{path}' has no column '{name}'"
                        ))
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            // The reader returns the projected roots in file order, without duplicates.
            roots.sort_unstable();
            roots.dedup();
            let schema = Arc::new(builder.schema().project(&roots).map_err(|e| {
                Error::invalid_input(format!("Cannot project Parquet file '{path}': {e}"))
            })?);
            // Top-level Arrow fields of a Parquet file are its root columns, in order.
            let mask = ProjectionMask::roots(builder.parquet_schema(), roots);
            (schema, Some(mask))
        }
    };
    check_parquet_chunk_ranges(builder.metadata(), mask.as_ref(), bounds, path)?;
    let builder = match mask {
        Some(mask) => builder.with_projection(mask),
        None => builder,
    };
    let total_rows = usize::try_from(builder.metadata().file_metadata().num_rows())
        .map_err(|_| Error::invalid_input(format!("Invalid row count in Parquet file '{path}'")))?;
    let batch_rows = parquet_batch_rows(builder.metadata(), total_rows);
    let reader = builder
        .with_batch_size(batch_rows)
        .build()
        .map_err(|e| Error::invalid_input(format!("Cannot read Parquet file '{path}': {e}")))?;
    let schema = normalized_schema(&arrow_schema)?;
    if total_rows == 0 {
        return Ok(IngestedTable {
            schema,
            batches: Vec::new(),
        });
    }
    // `total_rows` comes from the footer: bound the up-front reservation.
    let mut batches =
        Vec::with_capacity(total_rows.div_ceil(batch_rows).min(MAX_BATCH_VEC_RESERVE));
    for batch in reader {
        let batch = batch
            .map_err(|e| Error::invalid_input(format!("Cannot read Parquet file '{path}': {e}")))?;
        if batch.num_rows() > 0 {
            batches.push(normalize_batch_to(&batch, &schema)?);
        }
    }
    Ok(IngestedTable { schema, batches })
}

/// Arrow IPC file magic (at the start, padded to 8 bytes, and at the end).
const IPC_MAGIC: &[u8; 6] = b"ARROW1";

/// Encapsulated IPC message prefix that precedes the metadata length (format 0.15+).
const IPC_CONTINUATION: [u8; 4] = [0xff; 4];

/// Largest uncompressed size an LZ4 frame or a ZSTD frame can plausibly produce per
/// compressed byte. ZSTD's densest encoding (an RLE block: 4 bytes for up to 128 KiB) is about
/// 32768:1; LZ4 frames stay below 256:1. Twice the larger ratio leaves room for frame overhead.
const MAX_IPC_COMPRESSION_RATIO: u64 = 1 << 16;

/// Constant slack on top of [`MAX_IPC_COMPRESSION_RATIO`], for tiny buffers whose frame
/// header is a large share of their compressed size.
const IPC_DECOMPRESSED_SLACK: u64 = 1 << 20;

/// One record batch or dictionary block from an Arrow IPC footer (validated to lie inside
/// the file).
#[derive(Debug, Clone, Copy)]
struct IpcBlock {
    kind: &'static str,
    offset: u64,
    meta: u64,
    body: u64,
}

/// A `BufReader` with a tracked logical position, so that moves to a nearby offset reuse the
/// buffer ([`BufReader::seek_relative`]) instead of discarding it: each [`DuckDbFileReader`]
/// read that misses the buffer is one remote request.
struct TrackedReader<'r, R> {
    inner: &'r mut BufReader<R>,
    pos: u64,
}

impl<R: Read + Seek> TrackedReader<'_, R> {
    fn read_exact_at(&mut self, at: u64, buf: &mut [u8]) -> io::Result<()> {
        let delta = i64::try_from(i128::from(at) - i128::from(self.pos))
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "file offset too large"))?;
        self.inner.seek_relative(delta)?;
        self.inner.read_exact(buf)?;
        self.pos = at + buf.len() as u64;
        Ok(())
    }
}

/// Check an Arrow IPC file's footer and every block's message header before `arrow-ipc`
/// decodes it.
///
/// `arrow-ipc` 53 trusts the file:
///
/// * a negative block length panics (`to_usize().unwrap()`), a missing schema or metadata key
///   panics (`unwrap()`), and a huge block length allocates that much memory before any read
///   fails (an allocation failure aborts the process);
/// * a buffer that does not lie inside its block's body panics (`slice_with_length`);
/// * a compressed buffer (LZ4_FRAME / ZSTD) shorter than its 8-byte length prefix panics, and
///   the prefix (the *declared* uncompressed length, up to `i64::MAX`) is passed to
///   `Vec::with_capacity` before anything is decompressed: a 1 KiB file could request
///   exabytes and abort the process (W4-C-001).
///
/// This requires every record batch and dictionary block to lie inside the file (metadata
/// length at least 8 bytes), every buffer to lie inside its block's body, and every
/// compressed buffer to declare at most [`MAX_IPC_COMPRESSION_RATIO`] times its compressed
/// size (plus [`IPC_DECOMPRESSED_SLACK`]); with `decompressed_limit` (DuckDB's
/// `memory_limit`), the buffers of one block must also declare at most that much in total.
/// Leaves the reader at an unspecified position.
fn validate_ipc_file<R: Read + Seek>(
    reader: &mut BufReader<R>,
    path: &str,
    decompressed_limit: Option<u64>,
) -> Result<()> {
    let invalid =
        |detail: String| Error::invalid_input(format!("Invalid Arrow IPC file '{path}': {detail}"));
    let io_err = |e: io::Error| invalid(e.to_string());
    let len = reader.seek(SeekFrom::End(0)).map_err(io_err)?;
    // Leading magic (8, padded) + footer length (4) + trailing magic (6).
    const MIN_LEN: u64 = 8 + 4 + 6;
    if len < MIN_LEN {
        return Err(invalid(format!("the file is only {len} bytes long")));
    }
    let mut tail = [0u8; 10];
    reader.seek(SeekFrom::End(-10)).map_err(io_err)?;
    reader.read_exact(&mut tail).map_err(io_err)?;
    if &tail[4..] != IPC_MAGIC {
        return Err(invalid(
            "the trailing ARROW1 magic is missing (an Arrow IPC *stream* is not a file; \
             write it with the IPC file format, also called Feather v2)"
                .into(),
        ));
    }
    let footer_len = i32::from_le_bytes([tail[0], tail[1], tail[2], tail[3]]);
    let footer_len = u64::try_from(footer_len)
        .ok()
        .filter(|&n| n <= len - MIN_LEN)
        .ok_or_else(|| {
            invalid(format!(
                "the footer length {footer_len} does not fit the file"
            ))
        })?;
    let data_end = len - 10 - footer_len;
    let footer_size =
        usize::try_from(footer_len).map_err(|_| invalid("the footer is too large".into()))?;
    let mut footer = vec![0u8; footer_size];
    reader.seek(SeekFrom::Start(data_end)).map_err(io_err)?;
    reader.read_exact(&mut footer).map_err(io_err)?;
    let mut blocks = parse_ipc_footer(&footer, data_end).map_err(invalid)?;
    // In file order, so neighbouring blocks share the read-ahead buffer.
    blocks.sort_by_key(|b| b.offset);
    let mut tracked = TrackedReader {
        inner: reader,
        pos: data_end + footer_len,
    };
    for block in &blocks {
        validate_ipc_block(&mut tracked, block, decompressed_limit).map_err(|detail| {
            invalid(format!(
                "the {} block at offset {}: {detail}",
                block.kind, block.offset
            ))
        })?;
    }
    Ok(())
}

/// Parse the footer flatbuffer and check that every block lies before `data_end` (the start
/// of the footer). The error is a detail text.
fn parse_ipc_footer(footer: &[u8], data_end: u64) -> std::result::Result<Vec<IpcBlock>, String> {
    let footer = arrow_ipc::root_as_footer(footer)
        .map_err(|e| format!("the footer cannot be parsed: {e}"))?;
    if footer.schema().is_none() {
        return Err("the footer has no schema".into());
    }
    if let Some(custom) = footer.custom_metadata()
        && custom
            .iter()
            .any(|kv| kv.key().is_none() || kv.value().is_none())
    {
        return Err("the footer has a metadata entry without key or value".into());
    }
    let Some(record_batches) = footer.recordBatches() else {
        return Err("the footer has no record batch list".into());
    };
    let dictionaries = footer.dictionaries();
    let raw = record_batches.iter().map(|b| ("record batch", b)).chain(
        dictionaries
            .iter()
            .flat_map(|d| d.iter())
            .map(|b| ("dictionary", b)),
    );
    let mut blocks = Vec::new();
    for (kind, block) in raw {
        let (offset, meta, body) = (block.offset(), block.metaDataLength(), block.bodyLength());
        let parsed = u64::try_from(offset)
            .ok()
            .zip(u64::try_from(meta).ok().filter(|&m| m >= 8))
            .zip(u64::try_from(body).ok())
            .filter(|((o, m), b)| {
                o.checked_add(*m)
                    .and_then(|end| end.checked_add(*b))
                    .is_some_and(|end| end <= data_end)
            });
        let Some(((offset, meta), body)) = parsed else {
            return Err(format!(
                "a {kind} block (offset {offset}, metadata {meta} bytes, body {body} bytes) does \
                 not lie inside the file"
            ));
        };
        blocks.push(IpcBlock {
            kind,
            offset,
            meta,
            body,
        });
    }
    Ok(blocks)
}

/// Check one block's message header (see [`validate_ipc_file`]). The error is a detail text.
fn validate_ipc_block<R: Read + Seek>(
    reader: &mut TrackedReader<'_, R>,
    block: &IpcBlock,
    decompressed_limit: Option<u64>,
) -> std::result::Result<(), String> {
    let meta_len =
        usize::try_from(block.meta).map_err(|_| "the metadata is too large".to_string())?;
    let mut meta = vec![0u8; meta_len];
    reader
        .read_exact_at(block.offset, &mut meta)
        .map_err(|e| e.to_string())?;
    // `<0xFFFFFFFF> <length: i32> <flatbuffer>`, or the pre-0.15 form `<length: i32>
    // <flatbuffer>`, exactly as `arrow-ipc` reads it (`meta >= 8` was checked).
    let flatbuffer = if meta[..4] == IPC_CONTINUATION {
        &meta[8..]
    } else {
        &meta[4..]
    };
    let message = arrow_ipc::root_as_message(flatbuffer)
        .map_err(|e| format!("the message header cannot be parsed: {e}"))?;
    let batch = if block.kind == "dictionary" {
        message
            .header_as_dictionary_batch()
            .and_then(|dictionary| dictionary.data())
    } else {
        message.header_as_record_batch()
    };
    // A message of another type is an error `arrow-ipc` reports itself, without panicking.
    let Some(batch) = batch else {
        return Ok(());
    };
    let Some(buffers) = batch.buffers() else {
        return Ok(());
    };
    let compressed = batch.compression().is_some();
    let body_start = block.offset + block.meta;
    let mut declared_total: u64 = 0;
    for (i, buffer) in buffers.iter().enumerate() {
        let (offset, length) = (buffer.offset(), buffer.length());
        let range = u64::try_from(offset)
            .ok()
            .zip(u64::try_from(length).ok())
            .filter(|(o, l)| o.checked_add(*l).is_some_and(|end| end <= block.body));
        let Some((offset, length)) = range else {
            return Err(format!(
                "buffer {i} (offset {offset}, length {length}) does not lie inside the {}-byte \
                 body",
                block.body
            ));
        };
        if !compressed || length == 0 {
            continue;
        }
        if length < 8 {
            return Err(format!(
                "compressed buffer {i} is {length} bytes, shorter than its 8-byte length prefix"
            ));
        }
        let mut prefix = [0u8; 8];
        reader
            .read_exact_at(body_start + offset, &mut prefix)
            .map_err(|e| e.to_string())?;
        let declared = i64::from_le_bytes(prefix);
        // -1: stored uncompressed; 0: empty.
        if declared == -1 || declared == 0 {
            continue;
        }
        let payload = length - 8;
        let bound = payload
            .saturating_mul(MAX_IPC_COMPRESSION_RATIO)
            .saturating_add(IPC_DECOMPRESSED_SLACK);
        let Some(declared) = u64::try_from(declared).ok().filter(|&d| d <= bound) else {
            return Err(format!(
                "compressed buffer {i} declares an uncompressed length of {declared} bytes, \
                 which {payload} compressed bytes cannot hold"
            ));
        };
        declared_total = declared_total.saturating_add(declared);
        if let Some(limit) = decompressed_limit
            && declared_total > limit
        {
            return Err(format!(
                "its compressed buffers declare more than DuckDB's memory_limit ({limit} bytes) \
                 of uncompressed data"
            ));
        }
    }
    Ok(())
}

/// Decode an Arrow IPC file batch by batch, normalizing each batch as soon as it is decoded.
///
/// `buffer` is the read-ahead buffer size (see [`REMOTE_IPC_BUFFER`]); `decompressed_limit`
/// bounds the declared uncompressed size of each block (see [`validate_ipc_file`]).
fn decode_arrow_ipc<R: Read + Seek>(
    reader: R,
    buffer: usize,
    path: &str,
    decompressed_limit: Option<u64>,
) -> Result<IngestedTable> {
    let mut reader = BufReader::with_capacity(buffer, reader);
    validate_ipc_file(&mut reader, path, decompressed_limit)?;
    let reader = arrow_ipc::reader::FileReader::try_new(reader, None)
        .map_err(|e| Error::invalid_input(format!("Invalid Arrow IPC file '{path}': {e}")))?;
    let schema = normalized_schema(&reader.schema())?;
    let mut batches = Vec::with_capacity(reader.num_batches().min(MAX_BATCH_VEC_RESERVE));
    for batch in reader {
        let batch = batch.map_err(|e| {
            Error::invalid_input(format!("Cannot read Arrow IPC file '{path}': {e}"))
        })?;
        if batch.num_rows() > 0 {
            batches.push(normalize_batch_to(&batch, &schema)?);
        }
    }
    Ok(IngestedTable { schema, batches })
}

/// `Read + Seek` over a file opened through DuckDB's virtual file system.
///
/// Uses positional reads, so it works for remote file systems (httpfs range requests).
struct DuckDbFileReader {
    file: DuckDbFile,
    pos: u64,
    len: u64,
}

impl DuckDbFileReader {
    fn new(file: DuckDbFile) -> Result<Self> {
        let len = file.size()?;
        Ok(Self { file, pos: 0, len })
    }
}

/// Reads issued by a [`DuckDbChunkReader`] while one Parquet file was decoded.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VfsReadStats {
    /// Size of the file, as DuckDB's file system reported it.
    pub file_len: u64,
    /// Positional reads sent to DuckDB's file system (one range request each for httpfs).
    pub requests: u64,
    /// Bytes read in total.
    pub bytes: u64,
}

/// The DuckDB file handle shared by a [`DuckDbChunkReader`] and the readers it hands out.
struct VfsFile {
    /// `duckdb_neo::file::File` is `Send` but not `Sync` (concurrent reads are only safe for
    /// files opened with parallel access), and a `ChunkReader` must be `Sync`.
    file: Mutex<DuckDbFile>,
    len: u64,
    path: String,
    /// Largest range [`ChunkReader::get_bytes`] may allocate, and largest total of prefetched
    /// column chunks held at once (DuckDB's `memory_limit`).
    limit: Option<u64>,
    /// Column chunk ranges from the footer and the chunks prefetched so far (W6-A-001).
    chunks: Mutex<ChunkCache>,
    /// The first error DuckDB's file system returned, with its code: reported instead of the
    /// decoder's wrapper, which would call it an invalid file (W7-B-003).
    read_error: Mutex<Option<Error>>,
    requests: AtomicU64,
    bytes: AtomicU64,
}

/// One column chunk's byte range, as the footer declares it.
#[derive(Debug, Clone, Copy)]
struct ChunkRange {
    start: u64,
    len: u64,
    row_group: usize,
}

impl ChunkRange {
    fn contains(&self, pos: u64) -> bool {
        pos >= self.start && pos - self.start < self.len
    }
}

/// Column chunks a [`DuckDbChunkReader`] fetches whole, with one request each, the first time
/// `parquet` reads inside them.
///
/// `parquet`'s synchronous reader asks for each page separately (`get_read` at the page
/// offset); serving the pages from one prefetched buffer per column chunk turns one or two
/// range requests per page into one per chunk. The reader decodes the projected columns of a
/// row group in lockstep, so chunks of row groups before the previous one are dropped; the
/// buffers held at once stay within the limit (a chunk that does not fit is streamed instead).
#[derive(Default)]
struct ChunkCache {
    /// Every column chunk of the file, sorted by start offset.
    ranges: Vec<ChunkRange>,
    /// Prefetched chunks.
    cached: Vec<(ChunkRange, Bytes)>,
    cached_bytes: u64,
}

impl VfsFile {
    /// Fill `buf` from offset `at`; all of it must lie inside the file.
    fn read_exact_at(&self, at: u64, buf: &mut [u8]) -> std::result::Result<(), String> {
        if at
            .checked_add(buf.len() as u64)
            .is_none_or(|end| end > self.len)
        {
            return Err(format!(
                "range {at}+{} lies beyond the end of '{}' ({} bytes)",
                buf.len(),
                self.path,
                self.len
            ));
        }
        let at = usize::try_from(at).map_err(|_| "file offset too large".to_string())?;
        let file = self.file.lock().unwrap_or_else(|e| e.into_inner());
        if let Err(e) = file.read_exact_at(at, buf) {
            let err = annotate_vfs_read(e, &self.path);
            let message = err.message.clone();
            self.read_error
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .get_or_insert(err);
            return Err(message);
        }
        self.requests.fetch_add(1, Ordering::Relaxed);
        self.bytes.fetch_add(buf.len() as u64, Ordering::Relaxed);
        Ok(())
    }

    /// Read `len` bytes at `at` into a new buffer (an allocation failure is an error).
    fn read_range(&self, at: u64, len: usize) -> std::result::Result<Bytes, String> {
        let mut data = Vec::new();
        data.try_reserve_exact(len)
            .map_err(|_| format!("cannot allocate {len} bytes for a range of '{}'", self.path))?;
        data.resize(len, 0);
        self.read_exact_at(at, &mut data)?;
        Ok(Bytes::from(data))
    }

    /// Record the column chunk ranges the footer declares (hostile values are harmless: a
    /// range is only used once it lies inside the file and fits the limits).
    fn set_chunk_ranges(&self, metadata: &ParquetMetaData) {
        let mut ranges: Vec<ChunkRange> = metadata
            .row_groups()
            .iter()
            .enumerate()
            .flat_map(|(row_group, rg)| {
                rg.columns().iter().map(move |column| {
                    let (start, len) = column.byte_range();
                    ChunkRange {
                        start,
                        len,
                        row_group,
                    }
                })
            })
            .filter(|r| r.len > 0)
            .collect();
        ranges.sort_by_key(|r| r.start);
        let mut cache = self.chunks.lock().unwrap_or_else(|e| e.into_inner());
        *cache = ChunkCache {
            ranges,
            ..ChunkCache::default()
        };
    }

    /// The prefetched column chunk containing `pos`, fetching it with one request if it is
    /// not cached yet. `None` when `pos` lies in no usable chunk (the caller streams).
    fn chunk_containing(
        &self,
        pos: u64,
    ) -> std::result::Result<Option<(ChunkRange, Bytes)>, String> {
        let mut cache = self.chunks.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((range, data)) = cache.cached.iter().find(|(r, _)| r.contains(pos)) {
            return Ok(Some((*range, data.clone())));
        }
        let after = cache.ranges.partition_point(|r| r.start <= pos);
        let Some(range) = after
            .checked_sub(1)
            .map(|i| cache.ranges[i])
            .filter(|r| r.contains(pos))
        else {
            return Ok(None);
        };
        let limit = self
            .limit
            .unwrap_or(MAX_PREFETCHED_CHUNK_BYTES)
            .min(MAX_PREFETCHED_CHUNK_BYTES);
        let in_file = range
            .start
            .checked_add(range.len)
            .is_some_and(|end| end <= self.len);
        let Ok(len) = usize::try_from(range.len) else {
            return Ok(None);
        };
        if !in_file || range.len > limit {
            return Ok(None);
        }
        // Chunks of row groups before the previous one are no longer read; under memory
        // pressure, the previous row group's are dropped too (a column still reading one
        // fetches it again). Chunks of this row group are read in lockstep: if they do not fit
        // together, this one is streamed.
        let keep_from = range.row_group.saturating_sub(1);
        cache.cached.retain(|(r, _)| r.row_group >= keep_from);
        cache.cached_bytes = cache.cached.iter().map(|(r, _)| r.len).sum();
        if cache.cached_bytes + range.len > limit {
            cache.cached.retain(|(r, _)| r.row_group >= range.row_group);
            cache.cached_bytes = cache.cached.iter().map(|(r, _)| r.len).sum();
        }
        if cache.cached_bytes + range.len > limit {
            return Ok(None);
        }
        let data = self.read_range(range.start, len)?;
        cache.cached_bytes += range.len;
        cache.cached.push((range, data.clone()));
        Ok(Some((range, data)))
    }

    fn stats(&self) -> VfsReadStats {
        VfsReadStats {
            file_len: self.len,
            requests: self.requests.load(Ordering::Relaxed),
            bytes: self.bytes.load(Ordering::Relaxed),
        }
    }
}

/// [`ChunkReader`] over a file opened through DuckDB's virtual file system: the footer is
/// read with two positional reads, then each column chunk `parquet` decodes is fetched whole
/// with one read ([`ChunkCache`]), so only the footer and the projected column chunks are
/// read and the file is never buffered whole (W2-D-002 item 2, W6-A-001).
struct DuckDbChunkReader(Arc<VfsFile>);

impl DuckDbChunkReader {
    fn new(file: DuckDbFile, path: &str, limit: Option<u64>) -> Result<Self> {
        let len = file.size().map_err(|e| annotate_vfs_read(e, path))?;
        Ok(Self(Arc::new(VfsFile {
            file: Mutex::new(file),
            len,
            path: path.to_string(),
            limit,
            chunks: Mutex::new(ChunkCache::default()),
            read_error: Mutex::new(None),
            requests: AtomicU64::new(0),
            bytes: AtomicU64::new(0),
        })))
    }
}

impl Length for DuckDbChunkReader {
    fn len(&self) -> u64 {
        self.0.len
    }
}

impl ChunkReader for DuckDbChunkReader {
    type T = VfsRangeReader;

    fn get_read(&self, start: u64) -> parquet::errors::Result<VfsRangeReader> {
        if start > self.0.len {
            return Err(ParquetError::EOF(format!(
                "offset {start} lies beyond the end of '{}' ({} bytes)",
                self.0.path, self.0.len
            )));
        }
        let (pos, buf) = match self
            .0
            .chunk_containing(start)
            .map_err(ParquetError::General)?
        {
            // Serve from the prefetched chunk; reads past its end stream from the file.
            Some((range, data)) => {
                let skip = usize::try_from(start - range.start).unwrap_or(data.len());
                (range.start + range.len, data.slice(skip.min(data.len())..))
            }
            None => (start, Bytes::new()),
        };
        Ok(VfsRangeReader {
            file: Arc::clone(&self.0),
            pos,
            buf,
        })
    }

    fn get_bytes(&self, start: u64, length: usize) -> parquet::errors::Result<Bytes> {
        let len = length as u64;
        // Inside a prefetched chunk (pages read through the page index): no request.
        if let Some((range, data)) = self
            .0
            .chunk_containing(start)
            .map_err(ParquetError::General)?
            && start + len <= range.start + range.len
        {
            let skip = usize::try_from(start - range.start).unwrap_or(usize::MAX);
            if let Some(end) = skip.checked_add(length).filter(|&end| end <= data.len()) {
                return Ok(data.slice(skip..end));
            }
        }
        if let Some(limit) = self.0.limit
            && len > limit
        {
            return Err(ParquetError::General(format!(
                "a {len}-byte range of '{}' is more than DuckDB's memory_limit ({limit} bytes)",
                self.0.path
            )));
        }
        // The length comes from the file's metadata: check it before allocating.
        if start.checked_add(len).is_none_or(|end| end > self.0.len) {
            return Err(ParquetError::EOF(format!(
                "range {start}+{length} lies beyond the end of '{}' ({} bytes)",
                self.0.path, self.0.len
            )));
        }
        self.0
            .read_range(start, length)
            .map_err(ParquetError::General)
    }
}

/// Sequential reader from an offset of a [`DuckDbChunkReader`]'s file: serves a prefetched
/// column chunk first, then reads on with a [`VFS_PARQUET_READ_AHEAD`] buffer for small reads.
struct VfsRangeReader {
    file: Arc<VfsFile>,
    /// File offset of the end of `buf`.
    pos: u64,
    /// Bytes not returned yet.
    buf: Bytes,
}

impl Read for VfsRangeReader {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        if out.is_empty() {
            return Ok(0);
        }
        if self.buf.is_empty() {
            let remaining = self.file.len.saturating_sub(self.pos);
            if remaining == 0 {
                return Ok(0);
            }
            let cap = |n: usize| usize::try_from(remaining).map_or(n, |r| r.min(n));
            if out.len() >= VFS_PARQUET_READ_AHEAD {
                // Large reads (page bodies) go straight into the caller's buffer.
                let n = cap(out.len());
                self.file
                    .read_exact_at(self.pos, &mut out[..n])
                    .map_err(io::Error::other)?;
                self.pos += n as u64;
                return Ok(n);
            }
            let n = cap(VFS_PARQUET_READ_AHEAD);
            self.buf = self
                .file
                .read_range(self.pos, n)
                .map_err(io::Error::other)?;
            self.pos += n as u64;
        }
        let n = out.len().min(self.buf.len());
        out[..n].copy_from_slice(&self.buf[..n]);
        bytes::Buf::advance(&mut self.buf, n);
        Ok(n)
    }
}

/// Decode a Parquet file opened through DuckDB's file system, with ranged reads.
fn decode_parquet_vfs(
    file: DuckDbFile,
    path: &str,
    columns: Option<&[&str]>,
    buffer_limit: Option<u64>,
) -> Result<(IngestedTable, VfsReadStats)> {
    let reader = DuckDbChunkReader::new(file, path, buffer_limit)?;
    decode_parquet_chunk_reader(reader, path, columns, buffer_limit)
}

/// [`decode_parquet_vfs`] over an open [`DuckDbChunkReader`].
///
/// A failure of DuckDB's file system is reported as DuckDB returned it (code and message),
/// not as an invalid file (W7-B-003).
fn decode_parquet_chunk_reader(
    reader: DuckDbChunkReader,
    path: &str,
    columns: Option<&[&str]>,
    buffer_limit: Option<u64>,
) -> Result<(IngestedTable, VfsReadStats)> {
    let shared = Arc::clone(&reader.0);
    let bounds = ChunkBounds {
        file_len: shared.len,
        limit: buffer_limit,
    };
    let table = contain_decoder_panic("Parquet", path, || {
        let builder = ParquetRecordBatchReaderBuilder::try_new(reader);
        if let Ok(builder) = &builder {
            shared.set_chunk_ranges(builder.metadata());
        }
        decode_parquet_projected(builder, path, columns, bounds)
    });
    match table {
        Ok(table) => Ok((table, shared.stats())),
        Err(decoder) => {
            let first = shared
                .read_error
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .take();
            Err(first.unwrap_or(decoder))
        }
    }
}

/// Read a Parquet file through DuckDB's virtual file system only, with ranged reads, and
/// report the reads that took.
///
/// This is how [`read_parquet_limited`] reads URLs. Only DuckDB's own checks apply
/// (`enable_external_access`, `allowed_directories` / `allowed_paths` lexically,
/// `disabled_filesystems`): a local path is not checked by a [`FileAccessPolicy`], so SQL
/// entry points use [`read_parquet_limited`]. With `columns`, only those top-level columns
/// are decoded (and read), in file order. A footer, or a column chunk to decode, larger than
/// `buffer_limit` (pass [`memory_limit_bytes`]) is refused, as is a chunk outside the file.
pub fn read_parquet_through_vfs(
    fs: &FileSystem<'_>,
    path: &str,
    columns: Option<&[&str]>,
    buffer_limit: Option<u64>,
) -> Result<(IngestedTable, VfsReadStats)> {
    if path.trim().is_empty() {
        return Err(Error::invalid_input("File path cannot be empty"));
    }
    let file = open_through_duckdb(fs, path)?;
    decode_parquet_vfs(file, path, columns, buffer_limit)
}

impl Read for DuckDbFileReader {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let remaining = self.len.saturating_sub(self.pos);
        let n = usize::try_from(remaining).map_or(buf.len(), |r| r.min(buf.len()));
        if n == 0 {
            return Ok(0);
        }
        let pos = usize::try_from(self.pos)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "file offset too large"))?;
        self.file
            .read_exact_at(pos, &mut buf[..n])
            .map_err(|e| io::Error::other(e.message))?;
        self.pos += n as u64;
        Ok(n)
    }
}

impl Seek for DuckDbFileReader {
    fn seek(&mut self, target: SeekFrom) -> io::Result<u64> {
        let new_pos = match target {
            SeekFrom::Start(offset) => Some(offset),
            SeekFrom::End(delta) => self.len.checked_add_signed(delta),
            SeekFrom::Current(delta) => self.pos.checked_add_signed(delta),
        };
        let new_pos = new_pos.ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "seek before the start of the file",
            )
        })?;
        self.pos = new_pos;
        Ok(new_pos)
    }
}

fn annotate_vfs_read(err: Error, path: &str) -> Error {
    Error {
        code: err.code,
        message: format!(
            "Cannot read '{path}' through DuckDB's file system: {}",
            err.message
        ),
    }
}

/// Open `path` for reading through DuckDB's virtual file system (DuckDB checks access).
fn open_through_duckdb(fs: &FileSystem<'_>, path: &str) -> Result<DuckDbFile> {
    FileBuilder::new(fs, path)?
        .read()
        .open()
        .map_err(|e| Error {
            code: e.code,
            message: format!("DuckDB refused to open '{path}': {}", e.message),
        })
}

/// Whether `path` is a URL that only DuckDB's file system can resolve.
///
/// Call it after [`local_path_of_file_url`]: `file:` URLs DuckDB maps to local paths are
/// local paths. Other `file:` forms (`file:x`, `file://host/x`) are left to DuckDB.
fn reads_through_duckdb(path: &str) -> bool {
    is_remote(path) || path.starts_with("file:")
}

/// DuckDB's `memory_limit` in bytes, from the calling connection's settings, if readable.
pub fn memory_limit_bytes(context: &Context) -> Option<u64> {
    let config = context.config().ok()?;
    let (value, _scope) = config.try_get_value("memory_limit").ok()??;
    parse_byte_size(&String::from_value(&value).ok()??)
}

/// Parse a DuckDB byte size such as `12.4 GiB`, `500MB`, `1000 bytes` or `2048`.
/// `None` for anything else (negative, NaN, unknown unit).
pub fn parse_byte_size(text: &str) -> Option<u64> {
    let text = text.trim();
    let split = text
        .find(|c: char| !(c.is_ascii_digit() || c == '.'))
        .unwrap_or(text.len());
    let (number, unit) = text.split_at(split);
    let number: f64 = number.parse().ok()?;
    let multiplier: f64 = match unit.trim().to_ascii_lowercase().as_str() {
        "" | "b" | "byte" | "bytes" => 1.0,
        "k" | "kb" => 1e3,
        "kib" => 1024.0,
        "m" | "mb" => 1e6,
        "mib" => 1024.0 * 1024.0,
        "g" | "gb" => 1e9,
        "gib" => 1024.0 * 1024.0 * 1024.0,
        "t" | "tb" => 1e12,
        "tib" => 1024.0f64.powi(4),
        "p" | "pb" => 1e15,
        "pib" => 1024.0f64.powi(5),
        _ => return None,
    };
    let bytes = number * multiplier;
    // `as` saturates at u64::MAX for huge values (DuckDB shows "unlimited" as ~16 EiB).
    (bytes.is_finite() && bytes >= 0.0).then_some(bytes as u64)
}

/// Where a load reads its bytes from; see the module documentation.
enum Source {
    Local(std::fs::File),
    DuckDb(DuckDbFileReader),
}

/// Open the source for `path`.
///
/// Local paths go through `policy` (and, with `vfs`, through DuckDB's checks as well);
/// URLs need `vfs`.
fn open_source(
    path: &str,
    policy: &FileAccessPolicy,
    vfs: Option<&FileSystem<'_>>,
) -> Result<Source> {
    if path.trim().is_empty() {
        return Err(Error::invalid_input("File path cannot be empty"));
    }
    if let Some(local) = local_path_of_file_url(path) {
        // A local file: the canonical sandbox check applies, as for the plain path.
        let file = policy.open_regular_file(local)?;
        if let Some(fs) = vfs {
            drop(open_through_duckdb(fs, local)?);
        }
        return Ok(Source::Local(file));
    }
    if reads_through_duckdb(path) {
        let fs = vfs.ok_or_else(|| {
            Error::invalid_input(format!(
                "'{path}' is a URL; URLs are read through DuckDB's file system, which is only \
                 available when the load runs inside a DuckDB query"
            ))
        })?;
        let file = open_through_duckdb(fs, path)?;
        return Ok(Source::DuckDb(DuckDbFileReader::new(file)?));
    }
    let expanded = match vfs {
        Some(fs) if path.starts_with('~') => fs.expand(path)?,
        _ => path.to_string(),
    };
    // Our policy first: its denial does not reveal whether the file exists.
    let file = policy.open_regular_file(&expanded)?;
    if let Some(fs) = vfs {
        // DuckDB's own checks: enable_external_access / allowed_* (lexically, as DuckDB
        // does) and disabled_filesystems, which cannot be read through the config API.
        drop(open_through_duckdb(fs, &expanded)?);
    }
    Ok(Source::Local(file))
}

/// Read a Parquet file into normalized batches (no buffering limit besides allocation).
///
/// See [`read_parquet_limited`].
pub fn read_parquet_with(
    path: &str,
    policy: &FileAccessPolicy,
    vfs: Option<&FileSystem<'_>>,
) -> Result<IngestedTable> {
    read_parquet_limited(path, policy, vfs, None)
}

/// Read a Parquet file into normalized batches.
///
/// Local files are read with ranged reads from the handle the [`FileAccessPolicy`] verified.
/// URLs are read with ranged reads through DuckDB's file system (the footer, then one request
/// per decoded column chunk; see [`read_parquet_through_vfs`]). For both, a column chunk
/// that lies outside the file, or is larger than `buffer_limit` (pass
/// [`memory_limit_bytes`]), is refused before decoding (W7-B-002); over a URL the footer is
/// bounded the same way, and chunks of one row group that do not fit the limit together are
/// streamed page by page.
pub fn read_parquet_limited(
    path: &str,
    policy: &FileAccessPolicy,
    vfs: Option<&FileSystem<'_>>,
    buffer_limit: Option<u64>,
) -> Result<IngestedTable> {
    match open_source(path, policy, vfs)? {
        Source::Local(file) => {
            contain_decoder_panic("Parquet", path, || decode_parquet(file, path, buffer_limit))
        }
        Source::DuckDb(reader) => {
            decode_parquet_vfs(reader.file, path, None, buffer_limit).map(|(table, _)| table)
        }
    }
}

/// Read an Arrow IPC *file* (Feather v2) into normalized batches, without a total limit on
/// the declared uncompressed size of a block (the compression-ratio bound still applies).
///
/// See [`read_arrow_ipc_limited`].
pub fn read_arrow_ipc_with(
    path: &str,
    policy: &FileAccessPolicy,
    vfs: Option<&FileSystem<'_>>,
) -> Result<IngestedTable> {
    read_arrow_ipc_limited(path, policy, vfs, None)
}

/// Read an Arrow IPC *file* (Feather v2; uncompressed, LZ4_FRAME or ZSTD) into normalized
/// batches.
///
/// A block whose compressed buffers declare more than `decompressed_limit` bytes in total
/// (pass [`memory_limit_bytes`]) is refused before anything is allocated for it.
pub fn read_arrow_ipc_limited(
    path: &str,
    policy: &FileAccessPolicy,
    vfs: Option<&FileSystem<'_>>,
    decompressed_limit: Option<u64>,
) -> Result<IngestedTable> {
    match open_source(path, policy, vfs)? {
        // `std::io::BufReader`'s default capacity: local reads are cheap.
        Source::Local(file) => contain_decoder_panic("Arrow IPC", path, || {
            decode_arrow_ipc(file, 8 << 10, path, decompressed_limit)
        }),
        Source::DuckDb(reader) => contain_decoder_panic("Arrow IPC", path, || {
            decode_arrow_ipc(reader, REMOTE_IPC_BUFFER, path, decompressed_limit)
        }),
    }
}

/// Run a decoder, turning a panic inside `parquet` / `arrow-ipc` into an ordinary error.
///
/// Both crates still panic on some malformed inputs the validation here does not model (for
/// example a negative Parquet `uncompressed_page_size`, which becomes a `Vec::with_capacity`
/// capacity overflow). duckdb-neo reports a panic that escapes a callback as an INTERNAL
/// error, which invalidates the whole DuckDB instance; a corrupt file must only fail the call
/// (W4-C-001). Everything the decoder owned is dropped while unwinding, so nothing
/// half-built is kept or registered.
fn contain_decoder_panic<T>(
    format: &str,
    path: &str,
    decode: impl FnOnce() -> Result<T>,
) -> Result<T> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(decode)).unwrap_or_else(|panic| {
        let detail = panic
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| panic.downcast_ref::<&str>().copied())
            .unwrap_or("unknown panic");
        Err(Error::invalid_input(format!(
            "Invalid {format} file '{path}': the decoder rejected it ({detail})"
        )))
    })
}

/// Read a local Parquet file into normalized batches (no DuckDB file system involved).
pub fn read_parquet(path: &str, policy: &FileAccessPolicy) -> Result<IngestedTable> {
    read_parquet_with(path, policy, None)
}

/// Read a local Arrow IPC file into normalized batches (no DuckDB file system involved).
pub fn read_arrow_ipc(path: &str, policy: &FileAccessPolicy) -> Result<IngestedTable> {
    read_arrow_ipc_with(path, policy, None)
}

/// Load a Parquet file into the Craton Bolt catalog as `table_name`.
///
/// Nothing is registered when the path is refused or the file is invalid.
pub fn load_parquet(
    table_name: &str,
    path: &str,
    mode: RegisterMode,
    policy: &FileAccessPolicy,
    vfs: Option<&FileSystem<'_>>,
) -> Result<TableSummary> {
    read_parquet_with(path, policy, vfs)?.register(table_name, mode)
}

/// Load an Arrow IPC file into the Craton Bolt catalog as `table_name`.
///
/// Nothing is registered when the path is refused or the file is invalid.
pub fn load_arrow_ipc(
    table_name: &str,
    path: &str,
    mode: RegisterMode,
    policy: &FileAccessPolicy,
    vfs: Option<&FileSystem<'_>>,
) -> Result<TableSummary> {
    read_arrow_ipc_with(path, policy, vfs)?.register(table_name, mode)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{
        Date64Array, DictionaryArray, Float32Array, Int8Array, Int32Array, LargeStringArray,
        StringArray, StringViewArray, TimestampMicrosecondArray, UInt64Array, types::Int32Type,
    };
    use arrow_schema::TimeUnit;
    use std::fs::File;

    fn sample_batch() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int8, false),
            Field::new("s", DataType::LargeUtf8, true),
            Field::new("t", DataType::Timestamp(TimeUnit::Microsecond, None), true),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int8Array::from(vec![1, 2, 3])),
                Arc::new(LargeStringArray::from(vec![Some("x"), None, Some("z")])),
                Arc::new(TimestampMicrosecondArray::from(vec![
                    Some(1),
                    None,
                    Some(3),
                ])),
            ],
        )
        .unwrap()
    }

    fn open() -> FileAccessPolicy {
        FileAccessPolicy::unrestricted()
    }

    fn parquet_bytes(batch: &RecordBatch) -> Vec<u8> {
        let mut data = Vec::new();
        let mut w = parquet::arrow::ArrowWriter::try_new(&mut data, batch.schema(), None).unwrap();
        w.write(batch).unwrap();
        w.close().unwrap();
        data
    }

    #[test]
    fn normalize_casts_to_supported_types() {
        let out = normalize_batch_for_bolt(&sample_batch()).unwrap();
        assert_eq!(out.schema().field(0).data_type(), &DataType::Int32);
        assert_eq!(out.schema().field(1).data_type(), &DataType::Utf8);
        assert!(
            out.column(1)
                .as_any()
                .downcast_ref::<StringArray>()
                .is_some()
        );
        assert_eq!(out.num_rows(), 3);
        // Every normalized field is nullable, whatever the source said (W1-E-010).
        assert!(out.schema().fields().iter().all(|f| f.is_nullable()));
    }

    #[test]
    fn normalize_handles_view_float16_date64_and_dictionaries() {
        let dict: DictionaryArray<Int32Type> =
            vec![Some("a"), None, Some("a")].into_iter().collect();
        let int_dict = DictionaryArray::<Int32Type>::try_new(
            Int32Array::from(vec![0, 1, 0]),
            Arc::new(Int8Array::from(vec![7, 9])),
        )
        .unwrap();
        let half = arrow_cast::cast::cast(
            &Float32Array::from(vec![Some(1.5), None, Some(-2.0)]),
            &DataType::Float16,
        )
        .unwrap();
        let schema = Arc::new(Schema::new(vec![
            Field::new("v", DataType::Utf8View, true),
            Field::new("h", DataType::Float16, true),
            Field::new("d", DataType::Date64, true),
            Field::new("ds", dict.data_type().clone(), true),
            Field::new("di", int_dict.data_type().clone(), true),
        ]));
        let b = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringViewArray::from(vec![Some("p"), None, Some("q")])),
                half,
                Arc::new(Date64Array::from(vec![Some(86_400_000), None, Some(0)])),
                Arc::new(dict),
                Arc::new(int_dict),
            ],
        )
        .unwrap();
        let out = normalize_batch_for_bolt(&b).unwrap();
        let types: Vec<_> = out
            .schema()
            .fields()
            .iter()
            .map(|f| f.data_type().clone())
            .collect();
        assert_eq!(
            types,
            vec![
                DataType::Utf8,
                DataType::Float32,
                DataType::Date32,
                DataType::Utf8,
                DataType::Int32,
            ]
        );
        assert_eq!(out.column(1).null_count(), 1);
        assert_eq!(out.column(3).null_count(), 1);
        let f = out
            .column(1)
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap();
        assert_eq!(f.value(0), 1.5);
    }

    #[test]
    fn date64_with_time_of_day_is_rejected() {
        let schema = Arc::new(Schema::new(vec![Field::new("d", DataType::Date64, true)]));
        let b = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Date64Array::from(vec![
                Some(86_400_000),
                Some(86_400_001),
            ]))],
        )
        .unwrap();
        let err = normalize_batch_for_bolt(&b).unwrap_err();
        assert!(err.message.contains("whole number of days"), "{err}");
        // Negative whole days and NULLs are fine.
        let b = RecordBatch::try_new(
            schema,
            vec![Arc::new(Date64Array::from(vec![Some(-86_400_000), None]))],
        )
        .unwrap();
        assert!(normalize_batch_for_bolt(&b).is_ok());
    }

    #[test]
    fn zoned_timestamps_become_naive_utc_with_unchanged_values() {
        let tz: Arc<str> = Arc::from("America/New_York");
        let col =
            TimestampMicrosecondArray::from(vec![Some(1_000_000), None]).with_timezone(tz.clone());
        let schema = Arc::new(Schema::new(vec![Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, Some(tz)),
            true,
        )]));
        let b = RecordBatch::try_new(schema, vec![Arc::new(col)]).unwrap();
        let out = normalize_batch_for_bolt(&b).unwrap();
        assert_eq!(
            out.schema().field(0).data_type(),
            &DataType::Timestamp(TimeUnit::Microsecond, None)
        );
        let ts = out
            .column(0)
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .unwrap();
        assert_eq!(ts.value(0), 1_000_000);
        assert!(ts.is_null(1));
    }

    #[test]
    fn normalize_rejects_overflow_and_unsupported() {
        let schema = Arc::new(Schema::new(vec![Field::new("u", DataType::UInt64, false)]));
        let b = RecordBatch::try_new(schema, vec![Arc::new(UInt64Array::from(vec![u64::MAX]))])
            .unwrap();
        assert!(normalize_batch_for_bolt(&b).is_err());

        let schema = Arc::new(Schema::new(vec![Field::new("b", DataType::Binary, false)]));
        let b = RecordBatch::try_new(
            schema,
            vec![Arc::new(arrow_array::BinaryArray::from(vec![
                b"a".as_ref(),
            ]))],
        )
        .unwrap();
        let err = normalize_batch_for_bolt(&b).unwrap_err();
        assert!(err.message.contains("FORMAT bolt"), "{err}");

        assert!(normalized_schema(&Schema::empty()).is_err());
        let negative_scale = Schema::new(vec![Field::new("d", DataType::Decimal128(10, -2), true)]);
        assert!(normalized_schema(&negative_scale).is_err());
        let ok_scale = Schema::new(vec![Field::new("d", DataType::Decimal128(10, 2), true)]);
        assert!(normalized_schema(&ok_scale).is_ok());
    }

    #[test]
    fn file_helpers_reject_bad_paths() {
        assert!(read_parquet("", &open()).is_err());
        assert!(read_parquet("definitely/not/here.parquet", &open()).is_err());
        assert!(read_arrow_ipc("definitely/not/here.arrow", &open()).is_err());
        // URLs need DuckDB's file system.
        let err = read_parquet("s3://bucket/x.parquet", &open()).unwrap_err();
        assert!(err.message.contains("DuckDB's file system"), "{err}");
        assert!(read_arrow_ipc("https://example.invalid/x.arrow", &open()).is_err());
    }

    #[test]
    fn parquet_and_ipc_roundtrip_keep_batches_separate() {
        let dir = std::env::temp_dir();
        let pq = dir.join(format!("bolt_ingest_{}.parquet", std::process::id()));
        let ipc = dir.join(format!("bolt_ingest_{}.arrow", std::process::id()));
        let batch = sample_batch();

        let f = File::create(&pq).unwrap();
        let mut w = parquet::arrow::ArrowWriter::try_new(f, batch.schema(), None).unwrap();
        w.write(&batch).unwrap();
        // A second row group exercises reading across row groups.
        w.flush().unwrap();
        w.write(&batch).unwrap();
        w.close().unwrap();
        let back = read_parquet(pq.to_str().unwrap(), &open()).unwrap();
        assert_eq!(back.num_rows(), 6);
        assert_eq!(back.schema.field(0).data_type(), &DataType::Int32);
        assert!(back.batches.iter().all(|b| b.schema() == back.schema));

        let f = File::create(&ipc).unwrap();
        let mut w = arrow_ipc::writer::FileWriter::try_new(f, &batch.schema()).unwrap();
        w.write(&batch).unwrap();
        w.write(&batch).unwrap();
        w.finish().unwrap();
        drop(w);
        let back = read_arrow_ipc(ipc.to_str().unwrap(), &open()).unwrap();
        assert_eq!(back.num_rows(), 6);
        // Not concatenated: one normalized batch per IPC record batch.
        assert_eq!(back.batches.len(), 2);
        assert_eq!(back.schema.field(1).data_type(), &DataType::Utf8);

        // A restricted policy refuses both files before they are opened.
        let denied = FileAccessPolicy::restricted(vec![], vec![]);
        assert!(read_parquet(pq.to_str().unwrap(), &denied).is_err());
        assert!(read_arrow_ipc(ipc.to_str().unwrap(), &denied).is_err());

        // Garbage content is rejected rather than panicking.
        std::fs::write(&pq, b"not parquet").unwrap();
        assert!(read_parquet(pq.to_str().unwrap(), &open()).is_err());
        std::fs::write(&ipc, b"not arrow").unwrap();
        assert!(read_arrow_ipc(ipc.to_str().unwrap(), &open()).is_err());

        let _ = std::fs::remove_file(pq);
        let _ = std::fs::remove_file(ipc);
    }

    #[test]
    fn in_memory_parquet_reader_reads_the_same_data() {
        let data = parquet_bytes(&sample_batch());
        let back = decode_parquet(Bytes::from(data), "<memory>", None).unwrap();
        assert_eq!(back.num_rows(), 3);
        assert_eq!(back.schema.field(0).data_type(), &DataType::Int32);
    }

    #[test]
    fn empty_parquet_file_yields_schema_and_no_batches() {
        let pq =
            std::env::temp_dir().join(format!("bolt_ingest_empty_{}.parquet", std::process::id()));
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int16, true)]));
        let f = File::create(&pq).unwrap();
        let w = parquet::arrow::ArrowWriter::try_new(f, schema, None).unwrap();
        w.close().unwrap();
        let back = read_parquet(pq.to_str().unwrap(), &open()).unwrap();
        assert_eq!(back.num_rows(), 0);
        assert!(back.batches.is_empty());
        assert_eq!(back.schema.field(0).data_type(), &DataType::Int32);
        let _ = std::fs::remove_file(pq);
    }

    #[test]
    fn normalized_fields_record_the_original_duckdb_type() {
        use crate::arrow_bridge::original_duckdb_type;
        let tz: Arc<str> = Arc::from("Europe/Berlin");
        let mut keep = std::collections::HashMap::new();
        keep.insert("app.note".to_string(), "kept".to_string());
        // A stale key in the file is replaced by what the Arrow type says.
        keep.insert(DUCKDB_TYPE_METADATA_KEY.to_string(), "VARCHAR".to_string());
        let schema = Schema::new_with_metadata(
            vec![
                Field::new("i16", DataType::Int16, false).with_metadata(keep),
                Field::new("u32", DataType::UInt32, true),
                Field::new("u64", DataType::UInt64, true),
                Field::new("h", DataType::Float16, true),
                Field::new("s", DataType::Utf8View, true),
                Field::new("d64", DataType::Date64, true),
                Field::new("ts_s", DataType::Timestamp(TimeUnit::Second, None), true),
                Field::new(
                    "tz_ms",
                    DataType::Timestamp(TimeUnit::Millisecond, Some(tz.clone())),
                    true,
                ),
                Field::new(
                    "tz_ns",
                    DataType::Timestamp(TimeUnit::Nanosecond, Some(tz)),
                    true,
                ),
                Field::new("dec", DataType::Decimal128(9, 2), true),
                Field::new(
                    "dict",
                    DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Int8)),
                    true,
                ),
            ],
            [("pandas".to_string(), "{}".to_string())].into(),
        );
        let out = normalized_schema(&schema).unwrap();
        let expect = [
            ("i16", DataType::Int32, "SMALLINT"),
            ("u32", DataType::Int64, "UINTEGER"),
            ("u64", DataType::Int64, "UBIGINT"),
            ("h", DataType::Float32, "FLOAT"),
            ("s", DataType::Utf8, "VARCHAR"),
            ("d64", DataType::Date32, "DATE"),
            (
                "ts_s",
                DataType::Timestamp(TimeUnit::Second, None),
                "TIMESTAMP_S",
            ),
            (
                "tz_ms",
                DataType::Timestamp(TimeUnit::Microsecond, None),
                "TIMESTAMPTZ",
            ),
            (
                "tz_ns",
                DataType::Timestamp(TimeUnit::Nanosecond, None),
                "TIMESTAMP_TZ_NS",
            ),
            ("dec", DataType::Decimal128(9, 2), "DECIMAL(9,2)"),
            ("dict", DataType::Int32, "TINYINT"),
        ];
        for (name, storage, original) in expect {
            let f = out.field_with_name(name).unwrap();
            assert_eq!(f.data_type(), &storage, "{name}");
            assert!(f.is_nullable(), "{name}");
            assert_eq!(
                f.metadata()
                    .get(DUCKDB_TYPE_METADATA_KEY)
                    .map(String::as_str),
                Some(original),
                "{name}"
            );
            // The arrow_bridge read side accepts it (consistent with the storage type).
            assert_eq!(
                original_duckdb_type(f)
                    .map(|o| o.to_metadata_text())
                    .as_deref(),
                Some(original),
                "{name}"
            );
        }
        assert_eq!(
            out.field_with_name("i16")
                .unwrap()
                .metadata()
                .get("app.note")
                .map(String::as_str),
            Some("kept")
        );
        assert_eq!(out.metadata().get("pandas").map(String::as_str), Some("{}"));
    }

    #[test]
    fn zoned_millisecond_timestamps_are_stored_as_utc_micros() {
        let tz: Arc<str> = Arc::from("+02:00");
        let col = arrow_array::TimestampMillisecondArray::from(vec![Some(1_500), None])
            .with_timezone(tz.clone());
        let schema = Arc::new(Schema::new(vec![Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Millisecond, Some(tz)),
            true,
        )]));
        let b = RecordBatch::try_new(schema, vec![Arc::new(col)]).unwrap();
        let out = normalize_batch_for_bolt(&b).unwrap();
        let ts = out
            .column(0)
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .unwrap();
        assert_eq!(ts.value(0), 1_500_000);
        assert!(ts.is_null(1));
    }

    fn ipc_bytes(batch: &RecordBatch) -> Vec<u8> {
        let mut data = Vec::new();
        let mut w = arrow_ipc::writer::FileWriter::try_new(&mut data, &batch.schema()).unwrap();
        w.write(batch).unwrap();
        w.finish().unwrap();
        drop(w);
        data
    }

    #[test]
    fn corrupt_ipc_footers_are_errors_not_panics() {
        let good = ipc_bytes(&sample_batch());
        let decode =
            |data: Vec<u8>| decode_arrow_ipc(io::Cursor::new(data), 8 << 10, "<mem>", None);
        assert_eq!(decode(good.clone()).unwrap().num_rows(), 3);

        // Too short, and an IPC stream (no trailing magic).
        assert!(decode(b"ARROW1".to_vec()).is_err());
        let mut stream = Vec::new();
        {
            let batch = sample_batch();
            let mut w =
                arrow_ipc::writer::StreamWriter::try_new(&mut stream, &batch.schema()).unwrap();
            w.write(&batch).unwrap();
            w.finish().unwrap();
        }
        let err = decode(stream).unwrap_err();
        assert!(err.message.contains("stream"), "{err}");

        // A footer length larger than the file.
        let mut bad = good.clone();
        let n = bad.len();
        bad[n - 10..n - 6].copy_from_slice(&i32::MAX.to_le_bytes());
        assert!(decode(bad).is_err());
        let mut bad = good.clone();
        bad[n - 10..n - 6].copy_from_slice(&(-1i32).to_le_bytes());
        assert!(decode(bad).is_err());

        // Block lengths that arrow-ipc would `unwrap` (negative) or allocate (huge) are
        // refused. The `Block` struct is stored inline in the footer: offset (i64),
        // metaDataLength (i32), 4 bytes padding, bodyLength (i64).
        let footer_len =
            i32::from_le_bytes([good[n - 10], good[n - 9], good[n - 8], good[n - 7]]) as usize;
        let start = n - 10 - footer_len;
        let (offset, meta) = {
            let footer = arrow_ipc::root_as_footer(&good[start..n - 10]).unwrap();
            let block = footer.recordBatches().unwrap().get(0);
            (block.offset(), block.metaDataLength())
        };
        let mut pattern = offset.to_le_bytes().to_vec();
        pattern.extend_from_slice(&meta.to_le_bytes());
        let at = good[start..n - 10]
            .windows(pattern.len())
            .position(|w| w == pattern.as_slice())
            .map(|i| start + i)
            .expect("the record batch block is stored in the footer");
        for body in [-1i64, i64::MAX / 2] {
            let mut bad = good.clone();
            bad[at + 16..at + 24].copy_from_slice(&body.to_le_bytes());
            let err = decode(bad).unwrap_err();
            assert!(err.message.contains("block"), "{err}");
        }
        let mut bad = good.clone();
        bad[at + 8..at + 12].copy_from_slice(&(-5i32).to_le_bytes());
        assert!(decode(bad).is_err());
    }

    /// 10 000 rows that compress well (a constant `Int32` column and a repeated string).
    fn compressible_batch() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("z", DataType::Int32, true),
            Field::new("s", DataType::Utf8, true),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![7; 10_000])),
                Arc::new(StringArray::from(vec!["bolt"; 10_000])),
            ],
        )
        .unwrap()
    }

    fn compressed_ipc_bytes(batch: &RecordBatch, codec: arrow_ipc::CompressionType) -> Vec<u8> {
        let options = arrow_ipc::writer::IpcWriteOptions::default()
            .try_with_compression(Some(codec))
            .unwrap();
        let mut data = Vec::new();
        let mut w = arrow_ipc::writer::FileWriter::try_new_with_options(
            &mut data,
            &batch.schema(),
            options,
        )
        .unwrap();
        w.write(batch).unwrap();
        w.finish().unwrap();
        drop(w);
        data
    }

    /// Absolute offsets of every compressed buffer's 8-byte length prefix whose declared
    /// length is positive (really compressed), in the file's first record batch.
    fn compressed_prefix_offsets(file: &[u8]) -> Vec<usize> {
        let n = file.len();
        let footer_len =
            i32::from_le_bytes([file[n - 10], file[n - 9], file[n - 8], file[n - 7]]) as usize;
        let footer = arrow_ipc::root_as_footer(&file[n - 10 - footer_len..n - 10]).unwrap();
        let block = footer.recordBatches().unwrap().get(0);
        let (offset, meta) = (block.offset() as usize, block.metaDataLength() as usize);
        let header = &file[offset..offset + meta];
        let fb = if header[..4] == IPC_CONTINUATION {
            &header[8..]
        } else {
            &header[4..]
        };
        let message = arrow_ipc::root_as_message(fb).unwrap();
        let batch = message.header_as_record_batch().unwrap();
        assert!(batch.compression().is_some());
        batch
            .buffers()
            .unwrap()
            .iter()
            .filter(|b| b.length() >= 8)
            .map(|b| offset + meta + b.offset() as usize)
            .filter(|&at| i64::from_le_bytes(file[at..at + 8].try_into().unwrap()) > 0)
            .collect()
    }

    #[test]
    fn lz4_and_zstd_compressed_ipc_files_decode() {
        // W3-D-001: needs the `lz4` / `zstd` features of arrow-ipc.
        for codec in [
            arrow_ipc::CompressionType::LZ4_FRAME,
            arrow_ipc::CompressionType::ZSTD,
        ] {
            let data = compressed_ipc_bytes(&compressible_batch(), codec);
            assert!(
                !compressed_prefix_offsets(&data).is_empty(),
                "{codec:?}: the writer compressed nothing"
            );
            let back = decode_arrow_ipc(io::Cursor::new(data), 8 << 10, "<mem>", None)
                .unwrap_or_else(|e| panic!("{codec:?}: {e}"));
            assert_eq!(back.num_rows(), 10_000, "{codec:?}");
            let z = back.batches[0]
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap();
            assert!(z.iter().all(|v| v == Some(7)), "{codec:?}");
        }
    }

    #[test]
    fn hostile_declared_uncompressed_lengths_are_refused_before_allocation() {
        // W4-C-001: arrow-ipc passes the 8-byte prefix to `Vec::with_capacity`.
        for codec in [
            arrow_ipc::CompressionType::LZ4_FRAME,
            arrow_ipc::CompressionType::ZSTD,
        ] {
            let good = compressed_ipc_bytes(&compressible_batch(), codec);
            let at = compressed_prefix_offsets(&good)[0];
            for declared in [i64::MAX, 1 << 50, -7] {
                let mut bad = good.clone();
                bad[at..at + 8].copy_from_slice(&declared.to_le_bytes());
                let err =
                    decode_arrow_ipc(io::Cursor::new(bad), 8 << 10, "<mem>", None).unwrap_err();
                assert!(
                    err.message.contains("declares"),
                    "{codec:?} {declared}: {err}"
                );
            }
            // A plausible declared size is still bounded by memory_limit.
            let err = decode_arrow_ipc(io::Cursor::new(good.clone()), 8 << 10, "<mem>", Some(1024))
                .unwrap_err();
            assert!(err.message.contains("memory_limit"), "{codec:?}: {err}");
            // ...and accepted under a sufficient limit.
            assert!(
                decode_arrow_ipc(io::Cursor::new(good), 8 << 10, "<mem>", Some(1 << 30)).is_ok()
            );
        }
    }

    #[test]
    fn ipc_buffers_outside_their_body_are_refused() {
        // A buffer's (offset, length) is stored inline in the message header (16 bytes);
        // make the first non-empty buffer's length run past the body.
        let good = ipc_bytes(&sample_batch());
        let n = good.len();
        let footer_len =
            i32::from_le_bytes([good[n - 10], good[n - 9], good[n - 8], good[n - 7]]) as usize;
        let (offset, meta, body) = {
            let footer = arrow_ipc::root_as_footer(&good[n - 10 - footer_len..n - 10]).unwrap();
            let block = footer.recordBatches().unwrap().get(0);
            (
                block.offset() as usize,
                block.metaDataLength() as usize,
                block.bodyLength(),
            )
        };
        let (buf_offset, buf_len) = {
            let header = &good[offset..offset + meta];
            let message = arrow_ipc::root_as_message(&header[8..]).unwrap();
            let batch = message.header_as_record_batch().unwrap();
            let b = batch
                .buffers()
                .unwrap()
                .iter()
                .find(|b| b.length() > 0)
                .copied()
                .unwrap();
            (b.offset(), b.length())
        };
        let mut pattern = buf_offset.to_le_bytes().to_vec();
        pattern.extend_from_slice(&buf_len.to_le_bytes());
        let at = good[offset..offset + meta]
            .windows(16)
            .position(|w| w == pattern.as_slice())
            .map(|i| offset + i)
            .expect("the buffer is stored inline in the message header");
        let mut bad = good.clone();
        bad[at + 8..at + 16].copy_from_slice(&(body + 1).to_le_bytes());
        let err = decode_arrow_ipc(io::Cursor::new(bad), 8 << 10, "<mem>", None).unwrap_err();
        assert!(err.message.contains("inside the"), "{err}");
    }

    #[test]
    fn every_parquet_codec_decodes() {
        // W3-D-001: needs the `zstd`, `lz4`, `brotli` (and `snap`, `flate2`) features.
        use parquet::basic::{BrotliLevel, Compression, GzipLevel, ZstdLevel};
        use parquet::file::properties::WriterProperties;
        for codec in [
            Compression::UNCOMPRESSED,
            Compression::SNAPPY,
            Compression::GZIP(GzipLevel::default()),
            Compression::ZSTD(ZstdLevel::default()),
            Compression::LZ4_RAW,
            Compression::LZ4,
            Compression::BROTLI(BrotliLevel::default()),
        ] {
            let batch = compressible_batch();
            let props = WriterProperties::builder().set_compression(codec).build();
            let mut data = Vec::new();
            let mut w =
                parquet::arrow::ArrowWriter::try_new(&mut data, batch.schema(), Some(props))
                    .unwrap();
            w.write(&batch).unwrap();
            w.close().unwrap();
            let back = decode_parquet(Bytes::from(data), "<memory>", None)
                .unwrap_or_else(|e| panic!("{codec:?}: {e}"));
            assert_eq!(back.num_rows(), 10_000, "{codec:?}");
        }
    }

    #[test]
    fn decoder_panics_become_errors() {
        // W4-C-001: a panic must not reach duckdb-neo (which would invalidate the instance).
        let err =
            contain_decoder_panic::<()>("Parquet", "x.parquet", || panic!("capacity overflow"))
                .unwrap_err();
        assert!(err.message.contains("x.parquet"), "{err}");
        assert!(err.message.contains("capacity overflow"), "{err}");
        let err = contain_decoder_panic::<()>("Arrow IPC", "y.arrow", || {
            std::panic::panic_any(format!("index {} out of range", 9))
        })
        .unwrap_err();
        assert!(err.message.contains("index 9 out of range"), "{err}");
        // Errors and values pass through unchanged.
        assert_eq!(contain_decoder_panic("Parquet", "z", || Ok(5)).unwrap(), 5);
        let err = contain_decoder_panic::<()>("Parquet", "z", || {
            Err(Error::invalid_input("plain error"))
        })
        .unwrap_err();
        assert_eq!(err.message, "plain error");
    }

    #[test]
    fn byte_sizes_parse_like_duckdb_shows_them() {
        assert_eq!(parse_byte_size("2048"), Some(2048));
        assert_eq!(parse_byte_size("1000 bytes"), Some(1000));
        assert_eq!(parse_byte_size("1.5 KiB"), Some(1536));
        assert_eq!(parse_byte_size("500MB"), Some(500_000_000));
        assert_eq!(
            parse_byte_size(" 12.4 GiB "),
            Some((12.4 * 1024f64.powi(3)) as u64)
        );
        assert_eq!(
            parse_byte_size("16383.9 PiB"),
            Some((16383.9 * 1024f64.powi(5)) as u64)
        );
        assert_eq!(parse_byte_size("-1"), None);
        assert_eq!(parse_byte_size("lots"), None);
        assert_eq!(parse_byte_size("3 parsecs"), None);
    }

    #[test]
    fn parquet_batch_size_is_bounded() {
        let data = parquet_bytes(&sample_batch());
        let builder = ParquetRecordBatchReaderBuilder::try_new(Bytes::from(data)).unwrap();
        // Never more rows than the file has...
        assert_eq!(parquet_batch_rows(builder.metadata(), 3), 3);
        // ...and never more than the cap, however narrow the rows are.
        assert_eq!(
            parquet_batch_rows(builder.metadata(), 100_000_000),
            MAX_PARQUET_BATCH_ROWS
        );
    }

    /// A three-column file (`a`, `b`, `c`) with schema metadata `{"app": "x"}`.
    fn parquet_with_schema_metadata() -> Vec<u8> {
        let schema = Arc::new(
            Schema::new(vec![
                Field::new("a", DataType::Int32, false),
                Field::new("b", DataType::Utf8, true),
                Field::new("c", DataType::Int32, true),
            ])
            .with_metadata([("app".to_string(), "x".to_string())].into()),
        );
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                Arc::new(StringArray::from(vec![Some("p"), None, Some("q")])),
                Arc::new(Int32Array::from(vec![7, 8, 9])),
            ],
        )
        .unwrap();
        parquet_bytes(&batch)
    }

    fn unbounded(data: &[u8]) -> ChunkBounds {
        ChunkBounds {
            file_len: data.len() as u64,
            limit: None,
        }
    }

    #[test]
    fn parquet_schema_metadata_is_kept_with_and_without_projection() {
        // W7-B-001: the reader's schema has no schema metadata; the builder's has.
        let data = parquet_with_schema_metadata();
        let table = decode_parquet(Bytes::from(data.clone()), "<memory>", None).unwrap();
        assert_eq!(
            table.schema.metadata().get("app").map(String::as_str),
            Some("x")
        );

        let projected = decode_parquet_projected(
            ParquetRecordBatchReaderBuilder::try_new(Bytes::from(data.clone())),
            "<memory>",
            Some(&["c", "a", "c"]),
            unbounded(&data),
        )
        .unwrap();
        assert_eq!(
            projected.schema.metadata().get("app").map(String::as_str),
            Some("x")
        );
        // File order, without duplicates, exactly as the batches come back.
        let names: Vec<_> = projected
            .schema
            .fields()
            .iter()
            .map(|f| f.name().as_str())
            .collect();
        assert_eq!(names, ["a", "c"]);
        for batch in &projected.batches {
            assert_eq!(batch.schema(), projected.schema);
        }
        assert_eq!(projected.num_rows(), 3);
    }

    #[test]
    fn empty_projection_is_a_request_error() {
        // W7-B-003: not "the file has no columns".
        let data = parquet_with_schema_metadata();
        let err = decode_parquet_projected(
            ParquetRecordBatchReaderBuilder::try_new(Bytes::from(data.clone())),
            "<memory>",
            Some(&[]),
            unbounded(&data),
        )
        .unwrap_err();
        assert!(err.message.contains("No columns requested"), "{err}");
    }

    #[test]
    fn column_chunks_are_checked_against_the_file_and_the_limit() {
        // W7-B-002, local / in-memory reads too.
        let data = parquet_with_schema_metadata();
        let builder = ParquetRecordBatchReaderBuilder::try_new(Bytes::from(data.clone())).unwrap();
        let largest = builder
            .metadata()
            .row_groups()
            .iter()
            .flat_map(|rg| rg.columns().iter().map(|c| c.byte_range().1))
            .max()
            .unwrap();
        drop(builder);
        assert!(decode_parquet(Bytes::from(data.clone()), "<memory>", Some(largest)).is_ok());
        let err =
            decode_parquet(Bytes::from(data.clone()), "<memory>", Some(largest - 1)).unwrap_err();
        assert!(err.message.contains("memory_limit"), "{err}");
        assert!(err.message.contains("column '"), "{err}");

        // A file length shorter than the footer claims: the chunks lie outside it.
        let err = decode_parquet_projected(
            ParquetRecordBatchReaderBuilder::try_new(Bytes::from(data.clone())),
            "<memory>",
            None,
            ChunkBounds {
                file_len: 16,
                limit: None,
            },
        )
        .unwrap_err();
        assert!(err.message.contains("lies outside the file"), "{err}");
    }

    #[test]
    fn vfs_read_failures_keep_duckdb_error() {
        // W7-B-003: a failing DuckDB read is reported as DuckDB's error, not an invalid file.
        use duckdb_neo::environment::{Environment, StorageLocation};
        let env = Environment::new().unwrap();
        let instance = env.open(StorageLocation::InMemory).unwrap();
        let conn = instance.connect().unwrap();
        let fs = FileSystem::new(&conn).unwrap();
        let path =
            std::env::temp_dir().join(format!("bolt_w7a_vfs_err_{}.parquet", std::process::id()));
        std::fs::write(&path, parquet_with_schema_metadata()).unwrap();
        let path_str = path.to_str().unwrap();
        let file = open_through_duckdb(&fs, path_str).unwrap();
        let mut reader = DuckDbChunkReader::new(file, path_str, None).unwrap();
        // Claim the file is longer than it is: DuckDB fails the footer read past the end.
        Arc::get_mut(&mut reader.0).unwrap().len += 4096;
        let err = decode_parquet_chunk_reader(reader, path_str, None, None).unwrap_err();
        let _ = std::fs::remove_file(&path);
        assert!(
            err.message.starts_with(&format!(
                "Cannot read '{path_str}' through DuckDB's file system"
            )),
            "{err}"
        );
        assert!(!err.message.contains("Invalid Parquet file"), "{err}");
        assert_ne!(
            err.code,
            Error::invalid_input("").code,
            "DuckDB's error code is kept: {err}"
        );
    }
}

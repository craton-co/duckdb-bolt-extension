// SPDX-License-Identifier: Apache-2.0

//! Bidirectional bridge between Arrow `RecordBatch`es and DuckDB `DataChunk`s / `LogicalType`s,
//! featuring zero-copy Arrow C Data Interface (`FFI_ArrowArray` / `FFI_ArrowSchema` /
//! `FFI_ArrowArrayStream`) streaming via `duckdb_neo::arrow`.

use std::{
    collections::HashMap,
    mem::transmute,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::Arc,
};

use arrow_array::{
    Array, ArrayRef, BooleanArray, Date32Array, Decimal128Array, Float32Array, Float64Array,
    Int8Array, Int16Array, Int32Array, Int64Array, IntervalMonthDayNanoArray, LargeStringArray,
    RecordBatch, RecordBatchReader, StringArray, StringViewArray, StructArray,
    TimestampMicrosecondArray, TimestampMillisecondArray, TimestampNanosecondArray,
    TimestampSecondArray, UInt8Array, UInt16Array, UInt32Array, UInt64Array,
    builder::{
        BooleanBuilder, Date32Builder, Decimal128Builder, Float32Builder, Float64Builder,
        Int32Builder, Int64Builder, StringBuilder, TimestampMicrosecondBuilder,
        TimestampMillisecondBuilder, TimestampNanosecondBuilder, TimestampSecondBuilder,
    },
    cast::AsArray,
    ffi::{FFI_ArrowArray, FFI_ArrowSchema, from_ffi, to_ffi},
    ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream},
    make_array, new_empty_array,
};
use arrow_buffer::ScalarBuffer;
use arrow_cast::CastOptions;
use arrow_schema::ArrowError;
use arrow_schema::{DataType, Field, FieldRef, IntervalUnit, Schema, SchemaRef, TimeUnit};
use arrow_select::concat::concat_batches;
use duckdb_neo::{
    DuckDBType, Error, Parameters, Result,
    arrow::{
        ArrowArray as DuckArrowArray, ArrowExporter, ArrowImporter, ArrowSchema as DuckArrowSchema,
    },
    column_data_collection::ColumnDataCollection,
    connection::{Connection, Context, FFILink},
    data_chunk::{DataChunk, DataChunkRef},
    ffi as duck_ffi,
    logical_type::{LogicalType, LogicalTypeID},
    query_result::QueryResult,
    types::{
        DateValue, DecimalSignature, HugeDecimal, IntervalValue, LongDecimal, ShortDecimal,
        TimestampMsValue, TimestampNsValue, TimestampSecValue, TimestampTzNsValue,
        TimestampTzValue, TimestampValue, WordDecimal,
    },
    vector::{Unknown, Vector},
};

// Compile-time verification that arrow-rs and DuckDB C API v2 Arrow C Data Interface
// structs have identical size and alignment.
const _: () = {
    assert!(std::mem::size_of::<FFI_ArrowSchema>() == std::mem::size_of::<duck_ffi::ArrowSchema>());
    assert!(
        std::mem::align_of::<FFI_ArrowSchema>() == std::mem::align_of::<duck_ffi::ArrowSchema>()
    );
    assert!(std::mem::size_of::<FFI_ArrowArray>() == std::mem::size_of::<duck_ffi::ArrowArray>());
    assert!(std::mem::align_of::<FFI_ArrowArray>() == std::mem::align_of::<duck_ffi::ArrowArray>());
    assert!(
        std::mem::size_of::<FFI_ArrowArrayStream>()
            == std::mem::size_of::<duck_ffi::ArrowArrayStream>()
    );
    assert!(
        std::mem::align_of::<FFI_ArrowArrayStream>()
            == std::mem::align_of::<duck_ffi::ArrowArrayStream>()
    );
};

/// Maximum number of rows emitted per DuckDB `DataChunk`.
pub const STANDARD_CHUNK_CAPACITY: usize = 2048;

/// Maximum `DECIMAL` width supported by DuckDB (and by Arrow `Decimal128`).
const MAX_DECIMAL_WIDTH: u8 = 38;

/// Arrow cast options that return an error instead of silently producing `NULL` when a value
/// does not fit the target type (e.g. `70000` cast to `Int16`).
fn checked_cast_options() -> CastOptions<'static> {
    CastOptions {
        safe: false,
        ..CastOptions::default()
    }
}

/// Validate an Arrow `Decimal128(precision, scale)` and convert it into a DuckDB
/// [`DecimalSignature`].
///
/// Negative scales, `precision == 0`, `precision > 38`, and `scale > precision` are rejected
/// instead of clamped: clamping a negative scale to `0` would silently divide every value by
/// `10^|scale|`.
fn checked_decimal_signature(precision: u8, scale: i8) -> Result<DecimalSignature> {
    if scale < 0 {
        return Err(Error::invalid_input(format!(
            "Arrow Decimal128({precision}, {scale}) has a negative scale, which DuckDB DECIMAL cannot represent"
        )));
    }
    if !(1..=MAX_DECIMAL_WIDTH).contains(&precision) {
        return Err(Error::invalid_input(format!(
            "Arrow Decimal128({precision}, {scale}) precision must be between 1 and {MAX_DECIMAL_WIDTH}"
        )));
    }
    // `scale >= 0` was checked above, so the conversion cannot fail.
    let scale_u8 = u8::try_from(scale)
        .map_err(|_| Error::invalid_input(format!("Invalid Arrow Decimal128 scale {scale}")))?;
    if scale_u8 > precision {
        return Err(Error::invalid_input(format!(
            "Arrow Decimal128({precision}, {scale}) scale exceeds its precision"
        )));
    }
    Ok(DecimalSignature {
        width: precision,
        scale: scale_u8,
    })
}

/// Return an error when `value` has more than `width` decimal digits (DuckDB trusts the physical
/// storage of a `DECIMAL(width, _)` vector, so out-of-range values must never be written).
fn check_decimal_digits(value: i128, width: u8) -> Result<()> {
    // `width <= 38`, and `10^38 < u128::MAX`, so `pow` cannot overflow.
    if value.unsigned_abs() >= 10u128.pow(u32::from(width.min(MAX_DECIMAL_WIDTH))) {
        return Err(Error::invalid_input(format!(
            "Decimal value {value} (unscaled) does not fit DECIMAL width {width}"
        )));
    }
    Ok(())
}

/// Validate a `Decimal128` column before handing it to DuckDB (zero-copy or fallback path):
/// the type signature must be representable and every non-NULL value must fit the precision.
fn validate_decimal_column_for_duckdb(col: &ArrayRef) -> Result<()> {
    if let DataType::Decimal128(precision, scale) = col.data_type() {
        checked_decimal_signature(*precision, *scale)?;
        let arr = col
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .ok_or_else(|| Error::api_error("Failed to downcast Decimal128Array"))?;
        arr.validate_decimal_precision(*precision).map_err(|e| {
            Error::invalid_input(format!(
                "Decimal128({precision}, {scale}) column holds an out-of-range value: {e}"
            ))
        })?;
    }
    Ok(())
}

/// Read a 128-bit signed integer (`HUGEINT` or `DECIMAL(19..38, s)`) from a DuckDB vector row
/// using `ptr::read_unaligned`.
///
/// Why `read_unaligned`? DuckDB's C++ `hugeint_t` (`{ uint64_t lower; int64_t upper; }`) has
/// 8-byte alignment, whereas Rust's `i128` (`HugeDecimal(i128)`) has 16-byte alignment on
/// x86_64-pc-windows-msvc and modern rustc. Calling `reader.get(row)` or `reader.iter()` on
/// `Vector<i128>` or `Vector<HugeDecimal>` creates a `&i128` reference into an 8-byte-aligned
/// DuckDB vector buffer, triggering a misaligned pointer dereference abort in debug/coverage
/// builds.
fn read_unaligned_i128_from_reader<T: duckdb_neo::vector::VectorElement>(
    reader: &duckdb_neo::vector::VectorReader<'_, T>,
    row: usize,
) -> Result<Option<i128>> {
    // The pointer arithmetic below assumes 16-byte slots; `T` must be a 128-bit element
    // (`i128` / `HugeDecimal`). Checked at run time (folded by the compiler) so a wrong
    // instantiation is an error, never an out-of-bounds read (W3-C audit).
    if std::mem::size_of::<T::Internal>() != 16 {
        return Err(Error::api_error(
            "read_unaligned_i128_from_reader: vector element is not 16 bytes wide",
        ));
    }
    // `is_null` bounds-checks `row` against `reader.len()` and returns an error when it is out
    // of range, so after this line `row < reader.len()` holds.
    if reader.is_null(row)? {
        return Ok(None);
    }
    let physical = reader.selection().index(row);
    let base = reader.as_ptr() as *const u8;
    if base.is_null() {
        return Err(Error::api_error(
            "DuckDB returned a 128-bit vector without a data buffer",
        ));
    }
    // SAFETY:
    // * In bounds: `row < reader.len()` (checked by `is_null` above). `physical` is the physical
    //   slot of that row as reported by the reader's own selection (identity for flat storage,
    //   0 for constant storage, the dictionary selection otherwise), and the reader's data
    //   pointer covers every selected physical slot (`VectorReader::as_slice` contract), each
    //   `size_of::<T::Internal>() == 16` bytes wide (checked above).
    // * Initialized: the row is not NULL, so DuckDB wrote a `hugeint_t`
    //   (`{ lower: u64, upper: i64 }`, little-endian, i.e. the `i128` bit pattern) into the slot.
    // * Alignment: DuckDB only guarantees 8-byte alignment for `hugeint_t`, while Rust's `i128`
    //   is 16-byte aligned; `read_unaligned` makes no alignment assumption and never creates a
    //   `&i128`.
    // * Lifetime: `reader` borrows the vector, which keeps the buffer alive for this read.
    let raw = unsafe { std::ptr::read_unaligned(base.add(physical * 16) as *const i128) };
    Ok(Some(raw))
}

/// Write 128-bit signed integers (`DECIMAL(19..38, s)` via `T = HugeDecimal`, or `HUGEINT` via
/// `T = i128`) into a DuckDB flat vector using `ptr::write_unaligned` to avoid 16-byte alignment
/// assumptions on DuckDB's 8-byte-aligned `hugeint_t` vector buffer.
///
/// When `max_digits` is `Some(w)`, every value is checked to have at most `w` decimal digits.
fn write_unaligned_i128_column<T: duckdb_neo::vector::VectorElement>(
    arr: &Decimal128Array,
    row_offset: usize,
    count: usize,
    vec: Vector<'_, Unknown>,
    max_digits: Option<u8>,
) -> Result<()> {
    if std::mem::size_of::<T::Internal>() != 16 {
        return Err(Error::api_error(
            "write_unaligned_i128_column: vector element is not 16 bytes wide",
        ));
    }
    // `arr.value(src)` panics (inside a DuckDB callback: INTERNAL error, invalidated database)
    // when `src` is out of range, so check the source range up front.
    if row_offset
        .checked_add(count)
        .is_none_or(|end| end > arr.len())
    {
        return Err(Error::api_error(format!(
            "write_unaligned_i128_column: rows {row_offset}..+{count} exceed the source length {}",
            arr.len()
        )));
    }
    let mut typed = vec.cast::<T>()?;
    // `duckdb_v2_vector_set_size` reserves storage for `count` rows before setting the size.
    typed.set_size(count)?;
    let mut writer = typed.writer()?;
    // Belt and braces for the raw writes below: the writer must expose at least `count` rows.
    if writer.len() < count {
        return Err(Error::api_error(format!(
            "write_unaligned_i128_column: output vector has {} rows, expected {count}",
            writer.len()
        )));
    }
    let base = writer.as_mut_ptr() as *mut u8;
    if base.is_null() {
        return Err(Error::api_error(
            "DuckDB returned a 128-bit output vector without a data buffer",
        ));
    }
    for i in 0..count {
        let src = row_offset + i;
        if arr.is_null(src) {
            // SAFETY: the row is marked NULL, so its (uninitialized) data slot is never read;
            // `set_row_validity` bounds-checks `i` itself, and a HUGEINT / DECIMAL row has no
            // descendants that would also need NULL markers.
            unsafe { writer.set_row_validity(i, false)? };
        } else {
            let val = arr.value(src);
            if let Some(width) = max_digits {
                check_decimal_digits(val, width)?;
            }
            // SAFETY:
            // * `set_row_validity` runs first and bounds-checks `i < writer.len()`; together with
            //   `writer.len() >= count` (checked above) the 16-byte slot at `base + i * 16` lies
            //   inside the writer's flat storage, which `writer` borrows exclusively. (Before
            //   W3-C the write happened first and the bounds check second.)
            // * `write_unaligned` makes no alignment assumption (DuckDB's `hugeint_t` storage is
            //   only 8-byte aligned, Rust's `i128` is 16-byte aligned) and never creates a
            //   `&mut i128`.
            // * The row is marked valid just before its slot is written; nothing reads the
            //   vector until this function returns, when the slot holds an initialized value.
            unsafe {
                writer.set_row_validity(i, true)?;
                std::ptr::write_unaligned(base.add(i * 16) as *mut i128, val);
            }
        }
    }
    Ok(())
}

/// Write a `Decimal128` column into a narrow DuckDB decimal vector (`ShortDecimal` / `WordDecimal`
/// / `LongDecimal`) with checked narrowing: values that exceed the destination width return an
/// error instead of wrapping.
macro_rules! write_narrow_decimal_to_vector {
    ($arr:expr, $row_offset:expr, $count:expr, $vec:expr, $width:expr, $duck_ty:ident, $int_ty:ty) => {{
        let arr: &Decimal128Array = $arr;
        let mut typed = $vec.cast::<$duck_ty>()?;
        typed.set_size($count)?;
        let mut writer = typed.writer()?;
        for i in 0..$count {
            let src = $row_offset + i;
            if arr.is_null(src) {
                writer.set(i, None)?;
                continue;
            }
            let v = arr.value(src);
            check_decimal_digits(v, $width)?;
            let narrowed = <$int_ty>::try_from(v).map_err(|_| {
                Error::invalid_input(format!(
                    "Decimal value {v} (unscaled) does not fit DuckDB {} storage",
                    stringify!($int_ty)
                ))
            })?;
            writer.set(i, Some($duck_ty(narrowed)))?;
        }
        Ok(())
    }};
}

/// Write all rows of `vec` as SQL `NULL`.
///
/// Used for output slots that do not map to a physical Arrow column (virtual / row-id
/// projection columns). Leaving such a vector untouched would let DuckDB read uninitialized
/// memory; writing `NULL` is always type-correct, so it is the safe choice (an error would break
/// e.g. `SELECT count(*)`, for which DuckDB may project only the row-id column).
fn write_all_null_vector(mut vec: Vector<'_, Unknown>, count: usize) -> Result<()> {
    vec.set_size(count)?;
    let nested = matches!(
        vec.logical_type().id()?,
        LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_STRUCT
            | LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_LIST
            | LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_ARRAY
            | LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_MAP
            | LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_UNION
            | LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_VARIANT
    );
    if !nested && let Ok(mut writer) = vec.writer() {
        // SAFETY: every row is marked invalid (NULL), so no data slot is ever read, and the
        // type has no descendants that would also need NULL markers.
        unsafe { writer.fill_validity(false)? };
        return Ok(());
    }
    // Slow path: DuckDB's generic NULL setter also propagates NULLs into nested children.
    for row in 0..count {
        vec.set_null_slow(row)?;
    }
    Ok(())
}

macro_rules! write_primitive_arrow_to_vector {
    ($col:expr, $row_offset:expr, $count:expr, $vec:expr, $arr_ty:ty, $duck_ty:ty, $map_val:expr) => {{
        let arr = $col
            .as_any()
            .downcast_ref::<$arr_ty>()
            .ok_or_else(|| Error::api_error(concat!("Failed to downcast ", stringify!($arr_ty))))?;
        let mut typed = $vec.cast::<$duck_ty>()?;
        typed.set_size($count)?;
        let mut writer = typed.writer()?;
        let map_fn = $map_val;
        for i in 0..$count {
            let src = $row_offset + i;
            writer.set(i, (!arr.is_null(src)).then(|| map_fn(arr.value(src))))?;
        }
        Ok(())
    }};
}

/// `arrow_cast::cast_with_options`, except for `Decimal128 -> Decimal128`, which is done here.
///
/// arrow-cast 53.4.1 corrupts a decimal cast that keeps the scale but lowers the precision: it
/// adds one unit in the last place (`Decimal128(12, 2) 21.25 -> Decimal128(7, 2) 21.26`,
/// `Decimal128(38, 0) 12345 -> Decimal128(18, 0) 12346`; W12-A-001). The replacement rescales
/// with exact integer arithmetic, rounds half away from zero when the scale decreases (as
/// DuckDB does), and checks the target precision: an overflow is an error, or NULL when
/// `options.safe`. Every other cast is delegated unchanged; use this function, never
/// `arrow_cast::cast_with_options` directly.
pub fn cast_with_options(
    col: &dyn Array,
    target: &DataType,
    options: &CastOptions,
) -> std::result::Result<ArrayRef, ArrowError> {
    match (col.data_type(), target) {
        (DataType::Decimal128(_, from_scale), DataType::Decimal128(precision, scale)) => {
            let arr = col
                .as_any()
                .downcast_ref::<Decimal128Array>()
                .ok_or_else(|| ArrowError::CastError("expected a Decimal128 array".into()))?;
            rescale_decimal128(arr, *from_scale, *precision, *scale, options.safe)
                .map(|a| Arc::new(a) as ArrayRef)
        }
        _ => arrow_cast::cast_with_options(col, target, options),
    }
}

/// `arr` (scale `from_scale`) as `Decimal128(precision, scale)`; see [`cast_with_options`].
fn rescale_decimal128(
    arr: &Decimal128Array,
    from_scale: i8,
    precision: u8,
    scale: i8,
    safe: bool,
) -> std::result::Result<Decimal128Array, ArrowError> {
    let invalid = |what: String| ArrowError::CastError(what);
    if precision == 0 || precision > MAX_DECIMAL_WIDTH {
        return Err(invalid(format!("invalid DECIMAL precision {precision}")));
    }
    let bound = 10i128.pow(u32::from(precision));
    let shift = i32::from(scale) - i32::from(from_scale);
    let factor = 10i128
        .checked_pow(shift.unsigned_abs())
        .ok_or_else(|| invalid(format!("cannot rescale DECIMAL by 10^{shift}")))?;
    let convert = |v: i128| -> Option<i128> {
        let out = if shift >= 0 {
            v.checked_mul(factor)?
        } else {
            let (q, r) = (v / factor, v % factor);
            // Half away from zero.
            if r.unsigned_abs() * 2 >= factor.unsigned_abs() {
                q + v.signum()
            } else {
                q
            }
        };
        (out.unsigned_abs() < bound.unsigned_abs()).then_some(out)
    };
    let mut values = Vec::with_capacity(arr.len());
    for i in 0..arr.len() {
        if arr.is_null(i) {
            values.push(None);
            continue;
        }
        let v = arr.value(i);
        match convert(v) {
            Some(out) => values.push(Some(out)),
            None if safe => values.push(None),
            None => {
                return Err(invalid(format!(
                    "value {v} (scale {from_scale}) does not fit Decimal128({precision}, {scale})"
                )));
            }
        }
    }
    Decimal128Array::from(values).with_precision_and_scale(precision, scale)
}

/// Convert an Arrow [`DataType`] into the corresponding DuckDB [`LogicalType`].
pub fn arrow_type_to_duckdb_type<C: FFILink + ?Sized>(
    link: &C,
    data_type: &DataType,
) -> Result<LogicalType> {
    match data_type {
        DataType::Boolean => bool::logical_type(link),
        DataType::Int8 => i8::logical_type(link),
        DataType::Int16 => i16::logical_type(link),
        DataType::Int32 => i32::logical_type(link),
        DataType::Int64 => i64::logical_type(link),
        DataType::UInt8 => u8::logical_type(link),
        DataType::UInt16 => u16::logical_type(link),
        DataType::UInt32 => u32::logical_type(link),
        DataType::UInt64 => u64::logical_type(link),
        DataType::Float32 => f32::logical_type(link),
        DataType::Float64 => f64::logical_type(link),
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => String::logical_type(link),
        // A dictionary-encoded column has the logical type of its values (the fallback writer
        // unpacks it to the value type, and DuckDB's Arrow importer does the same).
        DataType::Dictionary(_, value_type) => arrow_type_to_duckdb_type(link, value_type),
        DataType::Date32 => DateValue::logical_type(link),
        DataType::Timestamp(TimeUnit::Second, _) => TimestampSecValue::logical_type(link),
        DataType::Timestamp(TimeUnit::Millisecond, _) => TimestampMsValue::logical_type(link),
        DataType::Timestamp(TimeUnit::Microsecond, _) => TimestampValue::logical_type(link),
        DataType::Timestamp(TimeUnit::Nanosecond, _) => TimestampNsValue::logical_type(link),
        DataType::Decimal128(precision, scale) => {
            checked_decimal_signature(*precision, *scale)?.logical_type(link)
        }
        // `bolt_query` results only (W9-A-002): Craton Bolt cannot store intervals.
        DataType::Interval(IntervalUnit::MonthDayNano) => IntervalValue::logical_type(link),
        other => Err(Error::invalid_input(format!(
            "Unsupported Arrow data type for DuckDB conversion: {other:?}"
        ))),
    }
}

/// Convert a DuckDB [`LogicalType`] into a Craton Bolt-compatible Arrow [`DataType`].
pub fn duckdb_type_to_bolt_arrow_type(logical_type: &LogicalType) -> Result<DataType> {
    match OriginalDuckDbType::from_logical_type(logical_type)? {
        Some(original) => Ok(original.storage_arrow_type()),
        None => Err(Error::invalid_input(format!(
            "Unsupported DuckDB logical type for Craton Bolt ingestion: {:?}",
            logical_type.id()?
        ))),
    }
}

/// Convert a DuckDB [`LogicalType`] into the Arrow [`DataType`] that represents it **exactly**
/// (no storage widening), i.e. the type DuckDB's Arrow importer maps back to the same logical
/// type.
///
/// Unlike [`duckdb_type_to_bolt_arrow_type`] (which normalizes values for Craton Bolt storage,
/// e.g. `SMALLINT -> Int32`), this is used to convert Bolt data back into a DuckDB destination,
/// e.g. by `COPY <table> FROM '<bolt_table>' (FORMAT bolt)`:
///
/// | DuckDB | Arrow |
/// | :--- | :--- |
/// | `TINYINT` / `SMALLINT` / `INTEGER` / `BIGINT` | `Int8` / `Int16` / `Int32` / `Int64` |
/// | `UTINYINT` / `USMALLINT` / `UINTEGER` / `UBIGINT` | `UInt8` / `UInt16` / `UInt32` / `UInt64` |
/// | `HUGEINT` | `Decimal128(38, 0)` |
/// | `TIMESTAMP_S` / `_MS` / `TIMESTAMP` / `_NS` | `Timestamp(s / ms / us / ns, None)` |
/// | `TIMESTAMPTZ` / `TIMESTAMP_TZ_NS` | `Timestamp(us / ns, Some("UTC"))` |
/// | `DECIMAL(w, s)` | `Decimal128(w, s)` |
///
/// Types Craton Bolt cannot represent (BLOB, LIST, STRUCT, INTERVAL, UUID, ...) return an error.
pub fn duckdb_type_to_exact_arrow_type(logical_type: &LogicalType) -> Result<DataType> {
    match OriginalDuckDbType::from_logical_type(logical_type)? {
        Some(original) => Ok(original.exact_arrow_type()),
        None => Err(Error::invalid_input(format!(
            "DuckDB logical type {:?} cannot be represented by Craton Bolt",
            logical_type.id()?
        ))),
    }
}

// ------------------------------------------------------------------------------------------
// RFC-008: original DuckDB types in Arrow field metadata
// ------------------------------------------------------------------------------------------

/// Arrow **field** metadata key under which RFC-008 records the DuckDB logical type a column had
/// before it was widened for Craton Bolt storage (see `docs/issues/W1-C-004-*.md`).
///
/// The value is the canonical text of an [`OriginalDuckDbType`] (`SMALLINT`, `UINTEGER`,
/// `TIMESTAMPTZ`, `DECIMAL(9,2)`, ...). It is written for **every** column that enters Bolt
/// through `COPY ... TO (FORMAT bolt)`, including columns whose storage type is not widened, so
/// "metadata present" means "this column came from DuckDB with exactly this type".
pub const DUCKDB_TYPE_METADATA_KEY: &str = "duckdb.logical_type";

/// The DuckDB logical types Craton Bolt can store, with their storage and exact Arrow types.
///
/// | DuckDB | metadata text | storage Arrow type | exact Arrow type |
/// | :--- | :--- | :--- | :--- |
/// | `BOOLEAN` | `BOOLEAN` | `Boolean` | `Boolean` |
/// | `TINYINT` / `SMALLINT` / `INTEGER` | same | `Int32` | `Int8` / `Int16` / `Int32` |
/// | `UTINYINT` / `USMALLINT` | same | `Int32` | `UInt8` / `UInt16` |
/// | `BIGINT` / `UINTEGER` / `UBIGINT` | same | `Int64` | `Int64` / `UInt32` / `UInt64` |
/// | `HUGEINT` | `HUGEINT` | `Int64` | `Decimal128(38, 0)` |
/// | `FLOAT` / `DOUBLE` | same | `Float32` / `Float64` | same |
/// | `VARCHAR` | `VARCHAR` | `Utf8` | `Utf8` |
/// | `DATE` | `DATE` | `Date32` | `Date32` |
/// | `TIMESTAMP_S` / `_MS` / `TIMESTAMP` / `_NS` | same | `Timestamp(unit, None)` | same |
/// | `TIMESTAMPTZ` | `TIMESTAMPTZ` | `Timestamp(us, None)` (UTC micros) | `Timestamp(us, "UTC")` |
/// | `TIMESTAMP_TZ_NS` | `TIMESTAMP_TZ_NS` | `Timestamp(ns, None)` (UTC nanos) | `Timestamp(ns, "UTC")` |
/// | `DECIMAL(w, s)` | `DECIMAL(w,s)` | `Decimal128(w, s)` | `Decimal128(w, s)` |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OriginalDuckDbType {
    Boolean,
    TinyInt,
    SmallInt,
    Integer,
    BigInt,
    UTinyInt,
    USmallInt,
    UInteger,
    UBigInt,
    HugeInt,
    Float,
    Double,
    Varchar,
    Date,
    TimestampSec,
    TimestampMs,
    Timestamp,
    TimestampNs,
    TimestampTz,
    TimestampTzNs,
    Decimal { width: u8, scale: u8 },
}

impl OriginalDuckDbType {
    /// Classify a DuckDB logical type. `Ok(None)` means Craton Bolt cannot store it.
    pub fn from_logical_type(logical_type: &LogicalType) -> Result<Option<Self>> {
        Ok(Some(match logical_type.id()? {
            LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_BOOLEAN => Self::Boolean,
            LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_TINYINT => Self::TinyInt,
            LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_SMALLINT => Self::SmallInt,
            LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_INTEGER => Self::Integer,
            LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_BIGINT => Self::BigInt,
            LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_UTINYINT => Self::UTinyInt,
            LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_USMALLINT => Self::USmallInt,
            LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_UINTEGER => Self::UInteger,
            LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_UBIGINT => Self::UBigInt,
            LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_HUGEINT => Self::HugeInt,
            LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_FLOAT => Self::Float,
            LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_DOUBLE => Self::Double,
            LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_VARCHAR => Self::Varchar,
            LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_DATE => Self::Date,
            LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_TIMESTAMP_SEC => Self::TimestampSec,
            LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_TIMESTAMP_MS => Self::TimestampMs,
            LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_TIMESTAMP => Self::Timestamp,
            LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_TIMESTAMP_NS => Self::TimestampNs,
            LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_TIMESTAMP_TZ => Self::TimestampTz,
            LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_TIMESTAMP_TZ_NS => Self::TimestampTzNs,
            LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_DECIMAL => {
                let sig = DecimalSignature::from_logical_type(logical_type)?;
                sig.check()?;
                Self::Decimal {
                    width: sig.width,
                    scale: sig.scale,
                }
            }
            _ => return Ok(None),
        }))
    }

    /// Canonical metadata text (the value stored under [`DUCKDB_TYPE_METADATA_KEY`]).
    pub fn to_metadata_text(&self) -> String {
        match self {
            Self::Boolean => "BOOLEAN".into(),
            Self::TinyInt => "TINYINT".into(),
            Self::SmallInt => "SMALLINT".into(),
            Self::Integer => "INTEGER".into(),
            Self::BigInt => "BIGINT".into(),
            Self::UTinyInt => "UTINYINT".into(),
            Self::USmallInt => "USMALLINT".into(),
            Self::UInteger => "UINTEGER".into(),
            Self::UBigInt => "UBIGINT".into(),
            Self::HugeInt => "HUGEINT".into(),
            Self::Float => "FLOAT".into(),
            Self::Double => "DOUBLE".into(),
            Self::Varchar => "VARCHAR".into(),
            Self::Date => "DATE".into(),
            Self::TimestampSec => "TIMESTAMP_S".into(),
            Self::TimestampMs => "TIMESTAMP_MS".into(),
            Self::Timestamp => "TIMESTAMP".into(),
            Self::TimestampNs => "TIMESTAMP_NS".into(),
            Self::TimestampTz => "TIMESTAMPTZ".into(),
            Self::TimestampTzNs => "TIMESTAMP_TZ_NS".into(),
            Self::Decimal { width, scale } => format!("DECIMAL({width},{scale})"),
        }
    }

    /// Parse metadata text. Case-insensitive; accepts the canonical spellings, common DuckDB
    /// aliases (`INT`, `INT2`, `TEXT`, `TIMESTAMP WITH TIME ZONE`, `NUMERIC(w, s)`, ...) and
    /// whitespace inside `DECIMAL(w, s)`. Returns `None` for anything else, including invalid
    /// decimal signatures.
    pub fn parse_metadata_text(text: &str) -> Option<Self> {
        let normalized = text
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .to_ascii_uppercase();
        let simple = match normalized.as_str() {
            "BOOLEAN" | "BOOL" | "LOGICAL" => Some(Self::Boolean),
            "TINYINT" | "INT1" => Some(Self::TinyInt),
            "SMALLINT" | "INT2" | "SHORT" => Some(Self::SmallInt),
            "INTEGER" | "INT" | "INT4" | "SIGNED" => Some(Self::Integer),
            "BIGINT" | "INT8" | "LONG" => Some(Self::BigInt),
            "UTINYINT" | "UINT8" => Some(Self::UTinyInt),
            "USMALLINT" | "UINT16" => Some(Self::USmallInt),
            "UINTEGER" | "UINT32" => Some(Self::UInteger),
            "UBIGINT" | "UINT64" => Some(Self::UBigInt),
            "HUGEINT" | "INT128" => Some(Self::HugeInt),
            "FLOAT" | "REAL" | "FLOAT4" => Some(Self::Float),
            "DOUBLE" | "FLOAT8" => Some(Self::Double),
            "VARCHAR" | "TEXT" | "STRING" => Some(Self::Varchar),
            "DATE" => Some(Self::Date),
            "TIMESTAMP_S" => Some(Self::TimestampSec),
            "TIMESTAMP_MS" => Some(Self::TimestampMs),
            "TIMESTAMP" | "DATETIME" | "TIMESTAMP_US" => Some(Self::Timestamp),
            "TIMESTAMP_NS" => Some(Self::TimestampNs),
            "TIMESTAMPTZ" | "TIMESTAMP_TZ" | "TIMESTAMP WITH TIME ZONE" => Some(Self::TimestampTz),
            "TIMESTAMP_TZ_NS" | "TIMESTAMPTZ_NS" | "TIMESTAMP_NS WITH TIME ZONE" => {
                Some(Self::TimestampTzNs)
            }
            _ => None,
        };
        if simple.is_some() {
            return simple;
        }
        // DECIMAL(w, s) / NUMERIC(w, s) / DECIMAL(w).
        let compact: String = normalized.chars().filter(|c| !c.is_whitespace()).collect();
        let args = compact
            .strip_prefix("DECIMAL(")
            .or_else(|| compact.strip_prefix("NUMERIC("))?
            .strip_suffix(')')?;
        let (width, scale) = match args.split_once(',') {
            Some((w, s)) => (w.parse::<u8>().ok()?, s.parse::<u8>().ok()?),
            None => (args.parse::<u8>().ok()?, 0),
        };
        if !(1..=MAX_DECIMAL_WIDTH).contains(&width) || scale > width {
            return None;
        }
        Some(Self::Decimal { width, scale })
    }

    /// The Arrow type Craton Bolt stores this column as (see [`duckdb_type_to_bolt_arrow_type`]).
    pub fn storage_arrow_type(&self) -> DataType {
        match self {
            Self::Boolean => DataType::Boolean,
            Self::TinyInt | Self::SmallInt | Self::Integer | Self::UTinyInt | Self::USmallInt => {
                DataType::Int32
            }
            Self::BigInt | Self::UInteger | Self::UBigInt | Self::HugeInt => DataType::Int64,
            Self::Float => DataType::Float32,
            Self::Double => DataType::Float64,
            Self::Varchar => DataType::Utf8,
            Self::Date => DataType::Date32,
            Self::TimestampSec => DataType::Timestamp(TimeUnit::Second, None),
            Self::TimestampMs => DataType::Timestamp(TimeUnit::Millisecond, None),
            Self::Timestamp | Self::TimestampTz => DataType::Timestamp(TimeUnit::Microsecond, None),
            Self::TimestampNs | Self::TimestampTzNs => {
                DataType::Timestamp(TimeUnit::Nanosecond, None)
            }
            // `scale <= width <= 38`, so the conversion is lossless.
            Self::Decimal { width, scale } => DataType::Decimal128(*width, *scale as i8),
        }
    }

    /// The Arrow type that represents this DuckDB type exactly (see
    /// [`duckdb_type_to_exact_arrow_type`]); DuckDB's Arrow importer maps it back to the same
    /// logical type (except `HUGEINT`, which imports as `DECIMAL(38, 0)`).
    pub fn exact_arrow_type(&self) -> DataType {
        let utc = || Some(Arc::<str>::from("UTC"));
        match self {
            Self::TinyInt => DataType::Int8,
            Self::SmallInt => DataType::Int16,
            Self::UTinyInt => DataType::UInt8,
            Self::USmallInt => DataType::UInt16,
            Self::UInteger => DataType::UInt32,
            Self::UBigInt => DataType::UInt64,
            Self::HugeInt => DataType::Decimal128(MAX_DECIMAL_WIDTH, 0),
            Self::TimestampTz => DataType::Timestamp(TimeUnit::Microsecond, utc()),
            Self::TimestampTzNs => DataType::Timestamp(TimeUnit::Nanosecond, utc()),
            other => other.storage_arrow_type(),
        }
    }

    /// Build the DuckDB [`LogicalType`].
    pub fn to_logical_type<C: FFILink + ?Sized>(&self, link: &C) -> Result<LogicalType> {
        match self {
            Self::Boolean => bool::logical_type(link),
            Self::TinyInt => i8::logical_type(link),
            Self::SmallInt => i16::logical_type(link),
            Self::Integer => i32::logical_type(link),
            Self::BigInt => i64::logical_type(link),
            Self::UTinyInt => u8::logical_type(link),
            Self::USmallInt => u16::logical_type(link),
            Self::UInteger => u32::logical_type(link),
            Self::UBigInt => u64::logical_type(link),
            Self::HugeInt => i128::logical_type(link),
            Self::Float => f32::logical_type(link),
            Self::Double => f64::logical_type(link),
            Self::Varchar => String::logical_type(link),
            Self::Date => DateValue::logical_type(link),
            Self::TimestampSec => TimestampSecValue::logical_type(link),
            Self::TimestampMs => TimestampMsValue::logical_type(link),
            Self::Timestamp => TimestampValue::logical_type(link),
            Self::TimestampNs => TimestampNsValue::logical_type(link),
            Self::TimestampTz => TimestampTzValue::logical_type(link),
            Self::TimestampTzNs => TimestampTzNsValue::logical_type(link),
            Self::Decimal { width, scale } => DecimalSignature {
                width: *width,
                scale: *scale,
            }
            .logical_type(link),
        }
    }

    /// `true` when `data_type` is a type this column can legitimately have in Bolt: its storage
    /// type or its exact type (after [`restore_original_types`]). Metadata on a field of any
    /// other type is stale (e.g. the column was re-typed without updating the metadata) and is
    /// ignored.
    fn accepts_arrow_type(&self, data_type: &DataType) -> bool {
        data_type == &self.storage_arrow_type() || data_type == &self.exact_arrow_type()
    }
}

/// Map an Arrow type to the closest type Craton Bolt stores, or `None` if unsupported.
///
/// The single mapping shared by every registration path: the runtime's
/// `normalize_schema_for_bolt` (`register_table`, COPY, every backend) and
/// `ingest::normalized_schema` (`bolt_load_*`), so the two cannot drift (W5-C-001).
/// Every mapping is a cast that `arrow-cast` implements (`Float16 -> Float32`,
/// `Date64 -> Date32`, `Utf8View` / `LargeUtf8 -> Utf8`, time-zone removal, dictionary
/// unpacking).
pub fn bolt_storage_arrow_type(dt: &DataType) -> Option<DataType> {
    Some(match dt {
        DataType::Boolean => DataType::Boolean,
        DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::UInt8 | DataType::UInt16 => {
            DataType::Int32
        }
        DataType::Int64 | DataType::UInt32 | DataType::UInt64 => DataType::Int64,
        DataType::Float16 | DataType::Float32 => DataType::Float32,
        DataType::Float64 => DataType::Float64,
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => DataType::Utf8,
        // Dictionaries are unpacked to their value type (any supported, non-nested one).
        DataType::Dictionary(_, value) => match value.as_ref() {
            DataType::Dictionary(_, _) => return None,
            other => bolt_storage_arrow_type(other)?,
        },
        // `Date64` is milliseconds since the epoch; the ingest path rejects values that are not
        // whole days instead of truncating them.
        DataType::Date32 | DataType::Date64 => DataType::Date32,
        DataType::Timestamp(unit, None) => DataType::Timestamp(*unit, None),
        // Zoned timestamps are UTC instants: dropping the zone keeps every value (arrow-cast
        // does not shift `Timestamp(_, Some(_))` -> `Timestamp(_, None)`) and gives the
        // naive-UTC representation COPY stores `TIMESTAMPTZ` (microseconds) and
        // `TIMESTAMP_TZ_NS` as.
        DataType::Timestamp(TimeUnit::Nanosecond, Some(_)) => {
            DataType::Timestamp(TimeUnit::Nanosecond, None)
        }
        DataType::Timestamp(_, Some(_)) => DataType::Timestamp(TimeUnit::Microsecond, None),
        // DuckDB has no negative-scale decimals; the bridge would clamp the scale to 0 and
        // silently multiply every value by 10^|scale|, so refuse them.
        DataType::Decimal128(p, s) if *s >= 0 => DataType::Decimal128(*p, *s),
        _ => return None,
    })
}

/// Field metadata recording `original` under [`DUCKDB_TYPE_METADATA_KEY`].
fn original_type_metadata(original: OriginalDuckDbType) -> HashMap<String, String> {
    HashMap::from([(
        DUCKDB_TYPE_METADATA_KEY.to_string(),
        original.to_metadata_text(),
    )])
}

/// The nullable field that holds a DuckDB `original` column **exactly**
/// ([`OriginalDuckDbType::exact_arrow_type`]), with RFC-008 metadata, so
/// [`arrow_field_to_duckdb_type`] exposes `original` (`HUGEINT` too, whose exact Arrow type is
/// `Decimal128(38, 0)`). `bolt_query` binds its DuckDB-typed result columns this way (W9-A-001).
pub fn exact_field(name: &str, original: OriginalDuckDbType) -> Field {
    Field::new(name, original.exact_arrow_type(), true)
        .with_metadata(original_type_metadata(original))
}

/// The Arrow type a `bolt_query` result column of DuckDB type `logical_type` is returned as:
/// [`duckdb_type_to_exact_arrow_type`], plus `INTERVAL` as `Interval(MonthDayNano)`
/// (W9-A-001, W9-A-002).
pub fn duckdb_type_to_query_arrow_type(logical_type: &LogicalType) -> Result<DataType> {
    if matches!(
        logical_type.id()?,
        LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_INTERVAL
    ) {
        return Ok(DataType::Interval(IntervalUnit::MonthDayNano));
    }
    duckdb_type_to_exact_arrow_type(logical_type)
}

/// Build the nullable Bolt **storage** field for a DuckDB column: the widened storage type
/// ([`duckdb_type_to_bolt_arrow_type`]) plus RFC-008 metadata recording `original`.
pub fn bolt_storage_field(name: &str, original: &LogicalType) -> Result<Field> {
    let original_type = OriginalDuckDbType::from_logical_type(original)?.ok_or_else(|| {
        let id = original
            .id()
            .map(|id| format!("{id:?}"))
            .unwrap_or_else(|_| "?".into());
        Error::invalid_input(format!(
            "Unsupported DuckDB logical type for Craton Bolt ingestion: {id}"
        ))
    })?;
    Ok(Field::new(name, original_type.storage_arrow_type(), true)
        .with_metadata(original_type_metadata(original_type)))
}

/// The original DuckDB type recorded in `field`'s RFC-008 metadata, if present, parseable, and
/// consistent with the field's Arrow type (stale metadata is ignored).
pub fn original_duckdb_type(field: &Field) -> Option<OriginalDuckDbType> {
    let text = field.metadata().get(DUCKDB_TYPE_METADATA_KEY)?;
    let original = OriginalDuckDbType::parse_metadata_text(text)?;
    original
        .accepts_arrow_type(field.data_type())
        .then_some(original)
}

/// The DuckDB type to **expose** for a stored Bolt field (RFC-008 read side).
///
/// * With valid metadata: the original type (`SMALLINT`, `UINTEGER`, `TIMESTAMPTZ`, ...).
/// * Without metadata, a time-zone-annotated `Timestamp(_, Some(tz))` is an instant and maps to
///   `TIMESTAMPTZ` (`TIMESTAMP_TZ_NS` for nanoseconds); the values are UTC by Arrow definition.
/// * Everything else falls back to [`arrow_type_to_duckdb_type`].
///
/// Pair it with [`restore_column_to_original_type`] / [`original_types_schema`] so the emitted
/// vectors carry exactly the declared type (the slow writer would also convert, with checks).
pub fn arrow_field_to_duckdb_type<C: FFILink + ?Sized>(
    link: &C,
    field: &Field,
) -> Result<LogicalType> {
    if let Some(original) = original_duckdb_type(field) {
        return original.to_logical_type(link);
    }
    match field.data_type() {
        DataType::Timestamp(TimeUnit::Nanosecond, Some(_)) => {
            TimestampTzNsValue::logical_type(link)
        }
        DataType::Timestamp(_, Some(_)) => TimestampTzValue::logical_type(link),
        other => arrow_type_to_duckdb_type(link, other),
    }
}

/// The exact Arrow type a reader should emit for `field`: the exact type of the recorded
/// original DuckDB type, or the field's own type when there is no (valid) metadata.
pub fn original_exact_arrow_type(field: &Field) -> DataType {
    original_duckdb_type(field)
        .map(|original| original.exact_arrow_type())
        .unwrap_or_else(|| field.data_type().clone())
}

/// `schema` with every field that carries RFC-008 metadata re-typed to its original exact Arrow
/// type (metadata, nullability and schema metadata are kept). Use it as the target of
/// [`cast_batch_to_schema_checked`] when the batch itself has no metadata (engine results).
pub fn original_types_schema(schema: &Schema) -> SchemaRef {
    let fields = schema
        .fields()
        .iter()
        .map(|field| -> FieldRef {
            let exact = original_exact_arrow_type(field);
            if &exact == field.data_type() {
                field.clone()
            } else {
                Arc::new(field.as_ref().clone().with_data_type(exact))
            }
        })
        .collect::<Vec<_>>();
    Arc::new(Schema::new_with_metadata(fields, schema.metadata().clone()))
}

/// Convert one stored column to the exact Arrow type of `field`'s recorded original DuckDB type
/// (checked: a value that does not fit returns an error). Without metadata the column is
/// returned unchanged (zero-copy), as it is when the types already agree.
///
/// `TIMESTAMPTZ` storage (`Timestamp(us, None)` holding UTC micros) becomes
/// `Timestamp(us, "UTC")` by re-annotation only, so the instant is unchanged.
pub fn restore_column_to_original_type(col: &ArrayRef, field: &Field) -> Result<ArrayRef> {
    let target = original_exact_arrow_type(field);
    if col.data_type() == &target {
        return Ok(col.clone());
    }
    cast_array_to_exact_type(col, &target).map_err(|e| {
        Error::invalid_input(format!(
            "Column '{}' cannot be restored to its original DuckDB type: {}",
            field.name(),
            e.message
        ))
    })
}

/// Convert every column of `batch` that carries RFC-008 metadata in `batch`'s own schema to the
/// exact Arrow type of its original DuckDB type (see [`restore_column_to_original_type`]).
///
/// The result's schema is [`original_types_schema`] of the input (metadata kept, so restoring
/// twice is a no-op). A batch without metadata is returned as a cheap clone.
pub fn restore_original_types(batch: &RecordBatch) -> Result<RecordBatch> {
    let schema = batch.schema();
    if schema
        .fields()
        .iter()
        .all(|f| original_exact_arrow_type(f) == *f.data_type())
    {
        return Ok(batch.clone());
    }
    let target = original_types_schema(&schema);
    let columns = batch
        .columns()
        .iter()
        .zip(schema.fields())
        .map(|(col, field)| restore_column_to_original_type(col, field))
        .collect::<Result<Vec<_>>>()?;
    RecordBatch::try_new_with_options(
        target,
        columns,
        &arrow_array::RecordBatchOptions::new().with_row_count(Some(batch.num_rows())),
    )
    .map_err(|e| Error::api_error(format!("Failed to build restored RecordBatch: {e}")))
}

/// Reinterpret a timestamp array as another timestamp type with the **same unit** (only the
/// time-zone annotation changes). Zero-copy: the value and validity buffers are shared.
fn reinterpret_timestamp(col: &ArrayRef, target: &DataType) -> Result<ArrayRef> {
    let data = col
        .to_data()
        .into_builder()
        .data_type(target.clone())
        .build()
        .map_err(|e| {
            Error::api_error(format!(
                "Failed to reinterpret {:?} as {target:?}: {e}",
                col.data_type()
            ))
        })?;
    Ok(make_array(data))
}

/// Convert `col` to exactly `target` with **checked** semantics.
///
/// * Identical types are returned as-is (zero-copy).
/// * Timestamps with the same unit only change their time-zone annotation (zero-copy). Craton
///   Bolt stores `TIMESTAMPTZ` as naive UTC microseconds, so no wall-clock adjustment is applied;
///   unit changes are checked for overflow.
/// * Everything else goes through `arrow_cast::cast_with_options` with `safe = false`, so values
///   that do not fit (e.g. `70000` into `Int16`, `-1` into `UInt64`) return an error instead of
///   silently becoming `NULL`.
/// * `Decimal128` results are validated against their precision.
pub fn cast_array_to_exact_type(col: &ArrayRef, target: &DataType) -> Result<ArrayRef> {
    let src = col.data_type();
    let out = if src == target {
        col.clone()
    } else if let (DataType::Timestamp(src_unit, _), DataType::Timestamp(dst_unit, _)) =
        (src, target)
    {
        if src_unit == dst_unit {
            reinterpret_timestamp(col, target)?
        } else {
            let naive = reinterpret_timestamp(col, &DataType::Timestamp(*src_unit, None))?;
            let converted = cast_with_options(
                naive.as_ref(),
                &DataType::Timestamp(*dst_unit, None),
                &checked_cast_options(),
            )
            .map_err(|e| {
                Error::invalid_input(format!(
                    "Failed to convert {src:?} column to {target:?}: {e}"
                ))
            })?;
            reinterpret_timestamp(&converted, target)?
        }
    } else {
        cast_with_options(col.as_ref(), target, &checked_cast_options()).map_err(|e| {
            Error::invalid_input(format!(
                "Failed to convert {src:?} column to {target:?}: {e}"
            ))
        })?
    };
    validate_decimal_column_for_duckdb(&out)?;
    Ok(out)
}

/// Convert every column of `batch` to the exact type of the corresponding field in `target`
/// (same column order and count) using [`cast_array_to_exact_type`].
///
/// The result carries `target` as its schema, so field names come from the destination.
pub fn cast_batch_to_schema_checked(
    batch: &RecordBatch,
    target: &SchemaRef,
) -> Result<RecordBatch> {
    if batch.num_columns() != target.fields().len() {
        return Err(Error::invalid_input(format!(
            "Column count mismatch: source has {} columns, destination has {}",
            batch.num_columns(),
            target.fields().len()
        )));
    }
    let columns = batch
        .columns()
        .iter()
        .zip(target.fields().iter())
        .map(|(col, field)| {
            cast_array_to_exact_type(col, field.data_type()).map_err(|e| {
                Error::invalid_input(format!("Column '{}': {}", field.name(), e.message))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    // Explicit row count: a zero-column target must keep the source row count.
    RecordBatch::try_new_with_options(
        target.clone(),
        columns,
        &arrow_array::RecordBatchOptions::new().with_row_count(Some(batch.num_rows())),
    )
    .map_err(|e| Error::api_error(format!("Failed to build converted RecordBatch: {e}")))
}

/// Prepare one sliced column for export through the Arrow C Data Interface to DuckDB's importer.
///
/// * A dictionary whose **values** contain NULLs is unpacked to its value type first. Arrow
///   defines such an entry as NULL even when its key is valid; whether DuckDB's importer honors
///   the validity of dictionary values (not only of the keys) is not documented, so a NULL
///   dictionary value could surface as a zeroed value instead of NULL.
/// * A dictionary of `Decimal128` values is unpacked so its precision can be validated below
///   (DuckDB trusts decimal storage).
/// * `Decimal128` columns are validated (see [`validate_decimal_column_for_duckdb`]).
fn prepare_column_for_ffi(col: ArrayRef) -> Result<ArrayRef> {
    let unpack_to: Option<DataType> = match col.data_type() {
        DataType::Dictionary(_, value_type) => {
            let values_have_nulls = col
                .as_any_dictionary_opt()
                .is_some_and(|dict| dict.values().null_count() > 0);
            (values_have_nulls || matches!(value_type.as_ref(), DataType::Decimal128(_, _)))
                .then(|| value_type.as_ref().clone())
        }
        _ => None,
    };
    let col = match unpack_to {
        Some(value_type) => cast_array_to_exact_type(&col, &value_type)?,
        None => col,
    };
    validate_decimal_column_for_duckdb(&col)?;
    Ok(col)
}

/// Write a slice of rows `[row_offset .. row_offset + row_count)` from `batch` into `output`
/// using the **Arrow C Data Interface (`FFI_ArrowArray` / `FFI_ArrowSchema`)** and
/// [`ArrowImporter::append_consumed`] + [`Vector::reference`] for zero-copy buffer sharing.
///
/// For any column whose imported DuckDB logical type differs from the bound output vector
/// type (e.g. `Dictionary(Int32, Utf8)` bound as flat `VARCHAR`, or `Decimal128` widths mapped
/// to different physical decimal representations), that column falls back to
/// [`write_arrow_column_to_vector`].
pub fn write_record_batch_slice_to_chunk_zero_copy(
    context: &Context,
    batch: &RecordBatch,
    row_offset: usize,
    row_count: usize,
    projected_columns: Option<&[usize]>,
    output: &mut DataChunkRef<'_>,
) -> Result<usize> {
    if row_count == 0 || row_offset >= batch.num_rows() {
        let mut first = output.get_vector_at_mut::<Unknown>(0)?;
        first.set_size(0)?;
        return Ok(0);
    }

    let actual_count = row_count
        .min(batch.num_rows() - row_offset)
        .min(STANDARD_CHUNK_CAPACITY);

    // Collect `(out_idx, col_idx)` pairs for physical columns to export via FFI.
    let mut ffi_fields = Vec::new();
    let mut ffi_arrays = Vec::new();
    let mut ffi_mapping = Vec::new();

    if let Some(proj) = projected_columns {
        if proj.is_empty() {
            let mut first = output.get_vector_at_mut::<Unknown>(0)?;
            first.set_size(actual_count)?;
            return Ok(actual_count);
        }
        for (out_idx, &col_idx) in proj.iter().enumerate() {
            if col_idx >= batch.num_columns() {
                // Virtual / row-id projection column: there is no physical Arrow column to
                // export. Write all-NULL rather than leaving the vector uninitialized (DuckDB
                // would otherwise read garbage) or erroring (DuckDB may request only the row-id
                // column, e.g. for `count(*)`).
                let vec = output.get_vector_at_mut::<Unknown>(out_idx)?;
                write_all_null_vector(vec, actual_count)?;
                continue;
            }
            let sliced_col =
                prepare_column_for_ffi(batch.column(col_idx).slice(row_offset, actual_count))?;
            ffi_fields.push(Arc::new(Field::new(
                format!("c{out_idx}"),
                sliced_col.data_type().clone(),
                true,
            )));
            ffi_arrays.push(sliced_col);
            ffi_mapping.push((out_idx, col_idx));
        }
    } else {
        let num_cols = batch.num_columns();
        if num_cols == 0 {
            let mut first = output.get_vector_at_mut::<Unknown>(0)?;
            first.set_size(actual_count)?;
            return Ok(actual_count);
        }
        for col_idx in 0..num_cols {
            let sliced_col =
                prepare_column_for_ffi(batch.column(col_idx).slice(row_offset, actual_count))?;
            ffi_fields.push(Arc::new(Field::new(
                format!("c{col_idx}"),
                sliced_col.data_type().clone(),
                true,
            )));
            ffi_arrays.push(sliced_col);
            ffi_mapping.push((col_idx, col_idx));
        }
    }

    if ffi_arrays.is_empty() {
        return Ok(actual_count);
    }

    let struct_array = StructArray::try_new(ffi_fields.into(), ffi_arrays, None)
        .map_err(|e| Error::api_error(format!("Failed to build FFI StructArray: {e}")))?;
    let (ffi_array, ffi_schema) = to_ffi(&struct_array.to_data())
        .map_err(|e| Error::api_error(format!("arrow::ffi::to_ffi failed: {e}")))?;

    // SAFETY:
    // * Layout: `FFI_ArrowSchema` / `duck_ffi::ArrowSchema` and `FFI_ArrowArray` /
    //   `duck_ffi::ArrowArray` are both `#[repr(C)]` renderings of the Arrow C Data Interface
    //   structs (same fields, same order); size and alignment are asserted at compile time at the
    //   top of this module.
    // * Ownership / no double release: `transmute` moves the value without running
    //   `FFI_ArrowSchema` / `FFI_ArrowArray`'s `Drop`, so arrow-rs never calls `release`. The
    //   DuckDB wrappers (`from_raw`) become the single owners: `ArrowImporter::new` releases the
    //   schema once resolved (or on drop, on error), and `append_consumed` hands the array to
    //   DuckDB on success, or releases it on drop on error. Either way `release` runs exactly
    //   once, from whichever thread drops the last owner (arrow-rs's release callback only drops
    //   an `Arc`'d `ArrayData`, which is thread-safe).
    // * Validity: both structs were just produced by `arrow::ffi::to_ffi` from a valid
    //   `StructArray`, so `release` is non-null and the buffers are live: the exported private
    //   data holds clones of the batch's `Arc` buffers, which are immutable.
    let duck_schema = unsafe {
        DuckArrowSchema::from_raw(transmute::<FFI_ArrowSchema, duck_ffi::ArrowSchema>(
            ffi_schema,
        ))
    };
    let duck_array = unsafe {
        DuckArrowArray::from_raw(transmute::<FFI_ArrowArray, duck_ffi::ArrowArray>(ffi_array))
    };

    match ArrowImporter::new(context, duck_schema, Some(actual_count)).and_then(|mut importer| {
        importer.append_consumed(duck_array, true)?;
        importer.next_chunk()
    }) {
        // Only reference the imported vectors when DuckDB produced the whole slice in one chunk;
        // otherwise the output would claim rows the referenced vectors do not have.
        Ok(Some(imported_chunk))
            if imported_chunk
                .row_count()
                .is_ok_and(|rows| rows == actual_count) =>
        {
            for (import_idx, &(out_idx, col_idx)) in ffi_mapping.iter().enumerate() {
                let src_vec = imported_chunk.get_vector_at::<Unknown>(import_idx)?;
                let mut out_vec = output.get_vector_at_mut::<Unknown>(out_idx)?;
                if src_vec.logical_type() == out_vec.logical_type() {
                    out_vec.set_size(actual_count)?;
                    // SAFETY (`Vector::reference`: source storage must outlive every read of the
                    // destination and stay unchanged):
                    // * Lifetime: `duckdb_v2_vector_reference` is C++ `Vector::Reference`, which
                    //   copies the source's `buffer` / `auxiliary` `shared_ptr`s (and the validity
                    //   buffer) into `out_vec`. For a zero-copy Arrow import the auxiliary data is
                    //   DuckDB's `ArrowAuxiliaryData`, which owns the imported `ArrowArray` and
                    //   therefore its `release` callback. Dropping `imported_chunk` at the end of
                    //   this function only drops one reference; the Arrow buffers stay alive until
                    //   DuckDB resets or destroys `output`'s vector, whatever thread that happens on.
                    // * Immutability: the referenced memory is the Arrow buffers of `batch`
                    //   (immutable `Arc` buffers; the FFI export holds its own clones, so no
                    //   `Buffer::into_mutable` can succeed while they are shared) or DuckDB-owned
                    //   copies; neither is written again. `reference` returns a read-only vector, so
                    //   this code cannot write through `out_vec` either.
                    // * Types: checked equal just above (DuckDB re-checks and errors otherwise),
                    //   and the row count of the imported chunk equals `actual_count`.
                    unsafe {
                        out_vec.reference(src_vec)?;
                    }
                } else {
                    write_arrow_column_to_vector(
                        batch.column(col_idx),
                        row_offset,
                        actual_count,
                        out_vec,
                    )?;
                }
            }
            Ok(actual_count)
        }
        _ => {
            for (out_idx, col_idx) in ffi_mapping {
                let out_vec = output.get_vector_at_mut::<Unknown>(out_idx)?;
                write_arrow_column_to_vector(
                    batch.column(col_idx),
                    row_offset,
                    actual_count,
                    out_vec,
                )?;
            }
            Ok(actual_count)
        }
    }
}

impl OriginalDuckDbType {
    /// `true` for the DuckDB types that denote an **instant** (`TIMESTAMPTZ`,
    /// `TIMESTAMP_TZ_NS`) rather than a local wall-clock value or a non-temporal value.
    pub fn is_instant(&self) -> bool {
        matches!(self, Self::TimestampTz | Self::TimestampTzNs)
    }
}

/// `true` when DuckDB's own cast from `source` to `dest` depends on the session `TimeZone`
/// setting: exactly one side is an instant (`TIMESTAMPTZ` / `TIMESTAMP_TZ_NS`), e.g.
/// `TIMESTAMPTZ -> TIMESTAMP` (wall clock in the session zone), `TIMESTAMPTZ -> DATE` / `VARCHAR`,
/// or `TIMESTAMP` / `DATE` / `VARCHAR -> TIMESTAMPTZ` (the local value is interpreted in the
/// session zone).
///
/// Arrow casts know nothing about the session zone (they re-annotate, i.e. use UTC), so such a
/// conversion must go through [`write_column_with_duckdb_cast`] to match
/// `INSERT INTO t SELECT CAST(col AS <dest>)` (W2-C-003).
pub fn conversion_depends_on_time_zone(
    source: OriginalDuckDbType,
    dest: OriginalDuckDbType,
) -> bool {
    source.is_instant() != dest.is_instant()
}

/// Write the first `count` rows of `col` (a column holding values of the DuckDB type `source`, in
/// its storage or exact Arrow type) into `out` by letting **DuckDB** cast them to `out`'s logical
/// type, exactly like SQL `CAST`.
///
/// The cast runs under `context`, so it uses the session's settings (notably `TimeZone`) and the
/// cast functions of the loaded extensions (ICU's time-zone-aware timestamp casts). Values are
/// first staged in a temporary vector of type `source`; DuckDB's cast writes a fresh result and
/// references it into `out` (no stale validity can leak). A value DuckDB cannot convert is an
/// error, as it would be for the equivalent `INSERT ... SELECT CAST(...)`.
pub fn write_column_with_duckdb_cast(
    context: &Context,
    col: &ArrayRef,
    source: OriginalDuckDbType,
    count: usize,
    out: Vector<'_, Unknown>,
) -> Result<()> {
    if col.len() < count {
        return Err(Error::api_error(format!(
            "write_column_with_duckdb_cast: column has {} rows, expected at least {count}",
            col.len()
        )));
    }
    let source_exact = source.exact_arrow_type();
    let staged_col = if col.data_type() == &source_exact {
        col.clone()
    } else {
        cast_array_to_exact_type(&col.slice(0, count), &source_exact)?
    };
    let mut staging = DataChunk::create(context, [source.to_logical_type(context)?], true)?;
    write_arrow_column_to_vector(
        &staged_col,
        0,
        count,
        staging.get_vector_at_mut::<Unknown>(0)?,
    )?;
    let staged = staging.get_vector_at_mut::<Unknown>(0)?;
    // `duckdb_v2_vector_cast` casts `staged.size()` rows into a fresh vector under the client
    // context and references it into `out`, which keeps the result buffers alive (shared
    // ownership) after `staging` is dropped. No unsafe code is needed.
    let cast = out.cast_rows(&staged, context)?;
    if cast.len() != count {
        return Err(Error::api_error(format!(
            "DuckDB cast produced {} rows, expected {count}",
            cast.len()
        )));
    }
    Ok(())
}

/// Return `true` when the exact writer can store a `src` Arrow column directly into a vector whose
/// exact Arrow type is `dest` (see [`duckdb_type_to_exact_arrow_type`]) without a prior cast.
fn directly_writable(src: &DataType, dest: &DataType) -> bool {
    match (src, dest) {
        (a, b) if a == b => true,
        (DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View, DataType::Utf8) => true,
        // Only the time-zone annotation differs; the writer dispatches on the destination type.
        (DataType::Timestamp(src_unit, _), DataType::Timestamp(dst_unit, _)) => {
            src_unit == dst_unit
        }
        // The decimal writer rescales / range-checks against the destination signature itself
        // (this includes `HUGEINT`, whose exact Arrow type is `Decimal128(38, 0)`).
        (DataType::Decimal128(_, _), DataType::Decimal128(_, _)) => true,
        _ => false,
    }
}

/// Slow-path writer: copy `count` rows starting at `row_offset` of `col` into the flat output
/// vector `vec`.
///
/// When the Arrow type does not match the vector's logical type exactly (e.g. `Int32` into a
/// `SMALLINT` vector, or a dictionary column), the slice is first converted with a **checked**
/// cast ([`cast_array_to_exact_type`]): values that do not fit return an error instead of being
/// silently wrapped or turned into `NULL`.
fn write_arrow_column_to_vector(
    col: &ArrayRef,
    row_offset: usize,
    count: usize,
    vec: Vector<'_, Unknown>,
) -> Result<()> {
    // Destination types Bolt cannot represent fall through to the exact writer, which reports a
    // clear type-mismatch error.
    if let Ok(dest) = duckdb_type_to_exact_arrow_type(vec.logical_type())
        && !directly_writable(col.data_type(), &dest)
    {
        let sliced = col.slice(row_offset, count);
        let converted = cast_array_to_exact_type(&sliced, &dest)?;
        return write_arrow_column_to_vector_exact(&converted, 0, count, vec);
    }
    write_arrow_column_to_vector_exact(col, row_offset, count, vec)
}

fn write_arrow_column_to_vector_exact(
    col: &ArrayRef,
    row_offset: usize,
    count: usize,
    vec: Vector<'_, Unknown>,
) -> Result<()> {
    let dest_id = vec.logical_type().id()?;
    match col.data_type() {
        DataType::Boolean => {
            write_primitive_arrow_to_vector!(col, row_offset, count, vec, BooleanArray, bool, |v| v)
        }
        DataType::Int8 => {
            write_primitive_arrow_to_vector!(col, row_offset, count, vec, Int8Array, i8, |v| v)
        }
        DataType::Int16 => {
            write_primitive_arrow_to_vector!(col, row_offset, count, vec, Int16Array, i16, |v| v)
        }
        DataType::Int32 => {
            write_primitive_arrow_to_vector!(col, row_offset, count, vec, Int32Array, i32, |v| v)
        }
        DataType::Int64 => {
            write_primitive_arrow_to_vector!(col, row_offset, count, vec, Int64Array, i64, |v| v)
        }
        DataType::UInt8 => {
            write_primitive_arrow_to_vector!(col, row_offset, count, vec, UInt8Array, u8, |v| v)
        }
        DataType::UInt16 => {
            write_primitive_arrow_to_vector!(col, row_offset, count, vec, UInt16Array, u16, |v| v)
        }
        DataType::UInt32 => {
            write_primitive_arrow_to_vector!(col, row_offset, count, vec, UInt32Array, u32, |v| v)
        }
        DataType::UInt64 => {
            write_primitive_arrow_to_vector!(col, row_offset, count, vec, UInt64Array, u64, |v| v)
        }
        DataType::Float32 => {
            write_primitive_arrow_to_vector!(col, row_offset, count, vec, Float32Array, f32, |v| v)
        }
        // `bolt_query` results only (W9-A-002). DuckDB keeps microseconds.
        DataType::Interval(IntervalUnit::MonthDayNano) => write_primitive_arrow_to_vector!(
            col,
            row_offset,
            count,
            vec,
            IntervalMonthDayNanoArray,
            IntervalValue,
            |v: arrow_buffer::IntervalMonthDayNano| IntervalValue {
                months: v.months,
                days: v.days,
                micros: v.nanoseconds.div_euclid(1000),
            }
        ),
        DataType::Float64 => {
            write_primitive_arrow_to_vector!(col, row_offset, count, vec, Float64Array, f64, |v| v)
        }
        DataType::Utf8 => {
            write_primitive_arrow_to_vector!(
                col,
                row_offset,
                count,
                vec,
                StringArray,
                String,
                |v| v
            )
        }
        DataType::LargeUtf8 => {
            write_primitive_arrow_to_vector!(
                col,
                row_offset,
                count,
                vec,
                LargeStringArray,
                String,
                |v| v
            )
        }
        DataType::Utf8View => {
            write_primitive_arrow_to_vector!(
                col,
                row_offset,
                count,
                vec,
                StringViewArray,
                String,
                |v| v
            )
        }
        DataType::Dictionary(_, value_type) => {
            // Unpack to the dictionary's value type (not unconditionally to Utf8, which would turn
            // e.g. a dictionary of Int64 into strings), then write through the full writer.
            let sliced = col.slice(row_offset, count);
            let unpacked = cast_array_to_exact_type(&sliced, value_type)?;
            write_arrow_column_to_vector(&unpacked, 0, count, vec)
        }
        DataType::Date32 => {
            write_primitive_arrow_to_vector!(
                col,
                row_offset,
                count,
                vec,
                Date32Array,
                DateValue,
                DateValue
            )
        }
        DataType::Timestamp(TimeUnit::Second, _) => {
            write_primitive_arrow_to_vector!(
                col,
                row_offset,
                count,
                vec,
                TimestampSecondArray,
                TimestampSecValue,
                TimestampSecValue
            )
        }
        DataType::Timestamp(TimeUnit::Millisecond, _) => {
            write_primitive_arrow_to_vector!(
                col,
                row_offset,
                count,
                vec,
                TimestampMillisecondArray,
                TimestampMsValue,
                TimestampMsValue
            )
        }
        // Dispatch on the destination vector type: a `TIMESTAMPTZ` vector stores the same UTC
        // microseconds as `TIMESTAMP`, but `cast::<TimestampValue>()` rejects it.
        DataType::Timestamp(TimeUnit::Microsecond, _) => match dest_id {
            LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_TIMESTAMP_TZ => {
                write_primitive_arrow_to_vector!(
                    col,
                    row_offset,
                    count,
                    vec,
                    TimestampMicrosecondArray,
                    TimestampTzValue,
                    TimestampTzValue
                )
            }
            _ => write_primitive_arrow_to_vector!(
                col,
                row_offset,
                count,
                vec,
                TimestampMicrosecondArray,
                TimestampValue,
                TimestampValue
            ),
        },
        DataType::Timestamp(TimeUnit::Nanosecond, _) => match dest_id {
            LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_TIMESTAMP_TZ_NS => {
                write_primitive_arrow_to_vector!(
                    col,
                    row_offset,
                    count,
                    vec,
                    TimestampNanosecondArray,
                    TimestampTzNsValue,
                    TimestampTzNsValue
                )
            }
            _ => write_primitive_arrow_to_vector!(
                col,
                row_offset,
                count,
                vec,
                TimestampNanosecondArray,
                TimestampNsValue,
                TimestampNsValue
            ),
        },
        DataType::Decimal128(precision, scale) => {
            write_decimal128_column(col, *precision, *scale, row_offset, count, vec)
        }
        other => Err(Error::invalid_input(format!(
            "Unsupported Arrow array type when writing to DuckDB vector: {other:?}"
        ))),
    }
}

/// Write a `Decimal128(precision, scale)` column into a `DECIMAL(w, s)` or `HUGEINT` vector.
///
/// The physical storage is chosen from the **destination** width (not the Arrow precision), a
/// differing scale is converted with a checked cast, and every value is range-checked before it
/// is narrowed, so an engine result exceeding the declared precision errors instead of wrapping.
fn write_decimal128_column(
    col: &ArrayRef,
    precision: u8,
    scale: i8,
    row_offset: usize,
    count: usize,
    vec: Vector<'_, Unknown>,
) -> Result<()> {
    checked_decimal_signature(precision, scale)?;
    let (dest_width, dest_scale, is_hugeint) = match vec.logical_type().id()? {
        LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_DECIMAL => {
            let sig = DecimalSignature::from_logical_type(vec.logical_type())?;
            sig.check()?;
            (sig.width, sig.scale, false)
        }
        LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_HUGEINT => (MAX_DECIMAL_WIDTH, 0, true),
        other => {
            return Err(Error::invalid_input(format!(
                "Cannot write Arrow Decimal128({precision}, {scale}) into a DuckDB {other:?} vector"
            )));
        }
    };
    let dest_scale_i8 = i8::try_from(dest_scale)
        .map_err(|_| Error::invalid_input(format!("Invalid DECIMAL scale {dest_scale}")))?;

    if dest_scale_i8 != scale || dest_width < precision {
        // Rescale / narrow with a checked cast so values are never silently corrupted.
        let sliced = col.slice(row_offset, count);
        let converted =
            cast_array_to_exact_type(&sliced, &DataType::Decimal128(dest_width, dest_scale_i8))?;
        return write_decimal128_column(&converted, dest_width, dest_scale_i8, 0, count, vec);
    }

    let arr = col
        .as_any()
        .downcast_ref::<Decimal128Array>()
        .ok_or_else(|| Error::api_error("Failed to downcast Decimal128Array"))?;
    if is_hugeint {
        return write_unaligned_i128_column::<i128>(arr, row_offset, count, vec, None);
    }
    match dest_width {
        1..=4 => write_narrow_decimal_to_vector!(
            arr,
            row_offset,
            count,
            vec,
            dest_width,
            ShortDecimal,
            i16
        ),
        5..=9 => write_narrow_decimal_to_vector!(
            arr,
            row_offset,
            count,
            vec,
            dest_width,
            WordDecimal,
            i32
        ),
        10..=18 => write_narrow_decimal_to_vector!(
            arr,
            row_offset,
            count,
            vec,
            dest_width,
            LongDecimal,
            i64
        ),
        _ => write_unaligned_i128_column::<HugeDecimal>(
            arr,
            row_offset,
            count,
            vec,
            Some(dest_width),
        ),
    }
}

enum ColumnAccumulator {
    Boolean(BooleanBuilder),
    Int32(Int32Builder),
    Int64(Int64Builder),
    Float32(Float32Builder),
    Float64(Float64Builder),
    Utf8(StringBuilder),
    Date32(Date32Builder),
    TimestampSec(TimestampSecondBuilder),
    TimestampMs(TimestampMillisecondBuilder),
    TimestampUs(TimestampMicrosecondBuilder),
    TimestampNs(TimestampNanosecondBuilder),
    Decimal128 {
        builder: Decimal128Builder,
        precision: u8,
        scale: i8,
    },
}

impl ColumnAccumulator {
    fn from_duckdb_type(logical_type: &LogicalType, capacity: usize) -> Result<(Self, DataType)> {
        let dt = duckdb_type_to_bolt_arrow_type(logical_type)?;
        let acc = match &dt {
            DataType::Boolean => Self::Boolean(BooleanBuilder::with_capacity(capacity)),
            DataType::Int32 => Self::Int32(Int32Builder::with_capacity(capacity)),
            DataType::Int64 => Self::Int64(Int64Builder::with_capacity(capacity)),
            DataType::Float32 => Self::Float32(Float32Builder::with_capacity(capacity)),
            DataType::Float64 => Self::Float64(Float64Builder::with_capacity(capacity)),
            DataType::Utf8 => Self::Utf8(StringBuilder::with_capacity(capacity, capacity * 16)),
            DataType::Date32 => Self::Date32(Date32Builder::with_capacity(capacity)),
            DataType::Timestamp(TimeUnit::Second, _) => {
                Self::TimestampSec(TimestampSecondBuilder::with_capacity(capacity))
            }
            DataType::Timestamp(TimeUnit::Millisecond, _) => {
                Self::TimestampMs(TimestampMillisecondBuilder::with_capacity(capacity))
            }
            DataType::Timestamp(TimeUnit::Microsecond, _) => {
                Self::TimestampUs(TimestampMicrosecondBuilder::with_capacity(capacity))
            }
            DataType::Timestamp(TimeUnit::Nanosecond, _) => {
                Self::TimestampNs(TimestampNanosecondBuilder::with_capacity(capacity))
            }
            DataType::Decimal128(precision, scale) => Self::Decimal128 {
                builder: Decimal128Builder::with_capacity(capacity),
                precision: *precision,
                scale: *scale,
            },
            other => {
                return Err(Error::invalid_input(format!(
                    "Unsupported Arrow builder type: {other:?}"
                )));
            }
        };
        Ok((acc, dt))
    }

    fn append_from_chunk(
        &mut self,
        chunk: &DataChunkRef<'_>,
        col_idx: usize,
        duckdb_type: &LogicalType,
    ) -> Result<()> {
        let type_id = duckdb_type.id()?;
        match self {
            Self::Boolean(b) => {
                let vec = chunk.get_vector_at::<bool>(col_idx)?;
                for item in vec.reader()?.iter() {
                    b.append_option(item.copied());
                }
            }
            Self::Int32(b) => match type_id {
                LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_TINYINT => {
                    let vec = chunk.get_vector_at::<i8>(col_idx)?;
                    for item in vec.reader()?.iter() {
                        b.append_option(item.map(|&v| i32::from(v)));
                    }
                }
                LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_SMALLINT => {
                    let vec = chunk.get_vector_at::<i16>(col_idx)?;
                    for item in vec.reader()?.iter() {
                        b.append_option(item.map(|&v| i32::from(v)));
                    }
                }
                LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_UTINYINT => {
                    let vec = chunk.get_vector_at::<u8>(col_idx)?;
                    for item in vec.reader()?.iter() {
                        b.append_option(item.map(|&v| i32::from(v)));
                    }
                }
                LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_USMALLINT => {
                    let vec = chunk.get_vector_at::<u16>(col_idx)?;
                    for item in vec.reader()?.iter() {
                        b.append_option(item.map(|&v| i32::from(v)));
                    }
                }
                _ => {
                    let vec = chunk.get_vector_at::<i32>(col_idx)?;
                    for item in vec.reader()?.iter() {
                        b.append_option(item.copied());
                    }
                }
            },
            Self::Int64(b) => match type_id {
                LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_UINTEGER => {
                    let vec = chunk.get_vector_at::<u32>(col_idx)?;
                    for item in vec.reader()?.iter() {
                        b.append_option(item.map(|&v| i64::from(v)));
                    }
                }
                LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_UBIGINT => {
                    let vec = chunk.get_vector_at::<u64>(col_idx)?;
                    for item in vec.reader()?.iter() {
                        match item {
                            Some(&v) => {
                                let signed = i64::try_from(v).map_err(|_| {
                                    Error::invalid_input(format!(
                                        "UBIGINT value {v} exceeds i64::MAX supported by Craton Bolt"
                                    ))
                                })?;
                                b.append_value(signed);
                            }
                            None => b.append_null(),
                        }
                    }
                }
                LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_HUGEINT => {
                    let vec = chunk.get_vector_at::<i128>(col_idx)?;
                    let reader = vec.reader()?;
                    for row in 0..reader.len() {
                        match read_unaligned_i128_from_reader(&reader, row)? {
                            Some(v) => {
                                let signed = i64::try_from(v).map_err(|_| {
                                    Error::invalid_input(format!(
                                        "HUGEINT value {v} exceeds i64 range supported by Craton Bolt"
                                    ))
                                })?;
                                b.append_value(signed);
                            }
                            None => b.append_null(),
                        }
                    }
                }
                _ => {
                    let vec = chunk.get_vector_at::<i64>(col_idx)?;
                    for item in vec.reader()?.iter() {
                        b.append_option(item.copied());
                    }
                }
            },
            Self::Float32(b) => {
                let vec = chunk.get_vector_at::<f32>(col_idx)?;
                for item in vec.reader()?.iter() {
                    b.append_option(item.copied());
                }
            }
            Self::Float64(b) => {
                let vec = chunk.get_vector_at::<f64>(col_idx)?;
                for item in vec.reader()?.iter() {
                    b.append_option(item.copied());
                }
            }
            Self::Utf8(b) => {
                let vec = chunk.get_vector_at::<String>(col_idx)?;
                for item in vec.reader()?.iter() {
                    b.append_option(item);
                }
            }
            Self::Date32(b) => {
                let vec = chunk.get_vector_at::<DateValue>(col_idx)?;
                for item in vec.reader()?.iter() {
                    b.append_option(item.map(|v| v.0));
                }
            }
            Self::TimestampSec(b) => {
                let vec = chunk.get_vector_at::<TimestampSecValue>(col_idx)?;
                for item in vec.reader()?.iter() {
                    b.append_option(item.map(|v| v.0));
                }
            }
            Self::TimestampMs(b) => {
                let vec = chunk.get_vector_at::<TimestampMsValue>(col_idx)?;
                for item in vec.reader()?.iter() {
                    b.append_option(item.map(|v| v.0));
                }
            }
            Self::TimestampUs(b) => match type_id {
                LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_TIMESTAMP_TZ => {
                    let vec = chunk.get_vector_at::<TimestampTzValue>(col_idx)?;
                    for item in vec.reader()?.iter() {
                        b.append_option(item.map(|v| v.0));
                    }
                }
                _ => {
                    let vec = chunk.get_vector_at::<TimestampValue>(col_idx)?;
                    for item in vec.reader()?.iter() {
                        b.append_option(item.map(|v| v.0));
                    }
                }
            },
            Self::TimestampNs(b) => match type_id {
                LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_TIMESTAMP_TZ_NS => {
                    let vec = chunk.get_vector_at::<TimestampTzNsValue>(col_idx)?;
                    for item in vec.reader()?.iter() {
                        b.append_option(item.map(|v| v.0));
                    }
                }
                _ => {
                    let vec = chunk.get_vector_at::<TimestampNsValue>(col_idx)?;
                    for item in vec.reader()?.iter() {
                        b.append_option(item.map(|v| v.0));
                    }
                }
            },
            Self::Decimal128 {
                builder, precision, ..
            } => match *precision {
                1..=4 => {
                    let vec = chunk.get_vector_at::<ShortDecimal>(col_idx)?;
                    for item in vec.reader()?.iter() {
                        builder.append_option(item.map(|v| i128::from(v.0)));
                    }
                }
                5..=9 => {
                    let vec = chunk.get_vector_at::<WordDecimal>(col_idx)?;
                    for item in vec.reader()?.iter() {
                        builder.append_option(item.map(|v| i128::from(v.0)));
                    }
                }
                10..=18 => {
                    let vec = chunk.get_vector_at::<LongDecimal>(col_idx)?;
                    for item in vec.reader()?.iter() {
                        builder.append_option(item.map(|v| i128::from(v.0)));
                    }
                }
                _ => {
                    let vec = chunk.get_vector_at::<HugeDecimal>(col_idx)?;
                    let reader = vec.reader()?;
                    for row in 0..reader.len() {
                        builder.append_option(read_unaligned_i128_from_reader(&reader, row)?);
                    }
                }
            },
        }
        Ok(())
    }

    fn finish(self) -> Result<ArrayRef> {
        match self {
            Self::Boolean(mut b) => Ok(Arc::new(b.finish())),
            Self::Int32(mut b) => Ok(Arc::new(b.finish())),
            Self::Int64(mut b) => Ok(Arc::new(b.finish())),
            Self::Float32(mut b) => Ok(Arc::new(b.finish())),
            Self::Float64(mut b) => Ok(Arc::new(b.finish())),
            Self::Utf8(mut b) => Ok(Arc::new(b.finish())),
            Self::Date32(mut b) => Ok(Arc::new(b.finish())),
            Self::TimestampSec(mut b) => Ok(Arc::new(b.finish())),
            Self::TimestampMs(mut b) => Ok(Arc::new(b.finish())),
            Self::TimestampUs(mut b) => Ok(Arc::new(b.finish())),
            Self::TimestampNs(mut b) => Ok(Arc::new(b.finish())),
            Self::Decimal128 {
                mut builder,
                precision,
                scale,
            } => {
                let arr = builder
                    .finish()
                    .with_precision_and_scale(precision, scale)
                    .map_err(|e| {
                        Error::invalid_input(format!("Invalid DECIMAL({precision}, {scale}): {e}"))
                    })?;
                Ok(Arc::new(arr))
            }
        }
    }
}

/// Normalize a column imported from DuckDB's Arrow C Data Interface (`ArrowExporter` / `ArrowStream`)
/// to match the target Craton Bolt physical Arrow [`DataType`].
///
/// When `col.data_type() == target_dt` (true for `BOOLEAN`, `INTEGER`, `BIGINT`, `FLOAT`, `DOUBLE`,
/// `VARCHAR`/`Utf8`, `DATE`, `TIMESTAMP`, and `DECIMAL`), `col.clone()` is returned with **zero copy**.
fn normalize_exported_column_for_bolt(
    col: &ArrayRef,
    logical_type: &LogicalType,
    target_dt: &DataType,
) -> Result<ArrayRef> {
    if col.data_type() == target_dt {
        return Ok(col.clone());
    }

    match logical_type.id()? {
        LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_UBIGINT => {
            if let Some(u64_arr) = col.as_any().downcast_ref::<UInt64Array>() {
                for i in 0..u64_arr.len() {
                    if !u64_arr.is_null(i) {
                        let v = u64_arr.value(i);
                        if v > i64::MAX as u64 {
                            return Err(Error::invalid_input(format!(
                                "UBIGINT value {v} exceeds i64::MAX supported by Craton Bolt"
                            )));
                        }
                    }
                }
                // Zero-copy reinterpret of the 64-bit value buffer from UInt64 to Int64:
                let scalar_buf = ScalarBuffer::<i64>::new(
                    u64_arr.values().inner().clone(),
                    u64_arr.offset(),
                    u64_arr.len(),
                );
                return Ok(Arc::new(Int64Array::new(
                    scalar_buf,
                    u64_arr.nulls().cloned(),
                )));
            }
        }
        LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_HUGEINT => {
            if let Some(dec_arr) = col.as_any().downcast_ref::<Decimal128Array>() {
                let mut builder = Int64Builder::with_capacity(dec_arr.len());
                for i in 0..dec_arr.len() {
                    if dec_arr.is_null(i) {
                        builder.append_null();
                    } else {
                        let v = dec_arr.value(i);
                        let signed = i64::try_from(v).map_err(|_| {
                            Error::invalid_input(format!(
                                "HUGEINT value {v} exceeds i64 range supported by Craton Bolt"
                            ))
                        })?;
                        builder.append_value(signed);
                    }
                }
                return Ok(Arc::new(builder.finish()));
            }
        }
        LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_TIMESTAMP_TZ => {
            if let Some(ts_arr) = col.as_any().downcast_ref::<TimestampMicrosecondArray>() {
                // Zero-copy reuse of underlying microsecond buffer without timezone metadata:
                return Ok(Arc::new(TimestampMicrosecondArray::new(
                    ts_arr.values().clone(),
                    ts_arr.nulls().cloned(),
                )));
            }
        }
        LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_TIMESTAMP_TZ_NS => {
            if let Some(ts_arr) = col.as_any().downcast_ref::<TimestampNanosecondArray>() {
                // Zero-copy reuse of underlying nanosecond buffer without timezone metadata:
                return Ok(Arc::new(TimestampNanosecondArray::new(
                    ts_arr.values().clone(),
                    ts_arr.nulls().cloned(),
                )));
            }
        }
        _ => {}
    }

    // Checked cast: a value that does not fit the Bolt storage type must error, never become NULL.
    cast_with_options(col.as_ref(), target_dt, &checked_cast_options()).map_err(|e| {
        Error::invalid_input(format!(
            "Failed to cast exported Arrow column from {:?} to {:?}: {e}",
            col.data_type(),
            target_dt
        ))
    })
}

/// Convert a DuckDB [`ColumnDataCollection`] into an Arrow [`RecordBatch`] ready for Craton Bolt
/// using DuckDB's **Arrow C Data Interface (`ArrowExporter` -> `FFI_ArrowArray` / `FFI_ArrowSchema`)**.
///
/// Concatenates the batches of [`collection_to_record_batches`] (prefer that function: it avoids
/// the copy and the 2 GiB `Utf8` offset limit of a single concatenated string column).
pub fn collection_to_record_batch(
    context: &Context,
    columns: &[(String, LogicalType)],
    collection: &ColumnDataCollection<'_>,
) -> Result<RecordBatch> {
    let (schema, mut batches) = collection_to_record_batches(context, columns, collection)?;
    match batches.len() {
        0 => Ok(RecordBatch::new_empty(schema)),
        1 => Ok(batches.pop().expect("exactly one batch")),
        _ => concat_batches(&schema, &batches).map_err(|e| {
            Error::api_error(format!("Failed to concatenate exported FFI batches: {e}"))
        }),
    }
}

/// Convert a DuckDB [`ColumnDataCollection`] into Arrow [`RecordBatch`]es in Craton Bolt storage
/// types, one per array produced by DuckDB's **Arrow C Data Interface** exporter (no
/// concatenation).
///
/// The returned schema is built with [`bolt_storage_field`], so every field records its
/// original DuckDB type under [`DUCKDB_TYPE_METADATA_KEY`] (RFC-008); every returned batch has
/// exactly this schema. An empty collection returns no batches.
///
/// Primitive and string columns whose exported Arrow format matches Craton Bolt (`BOOLEAN`,
/// `INTEGER`, `BIGINT`, `FLOAT`, `DOUBLE`, `VARCHAR`, `DATE`, `TIMESTAMP`, `DECIMAL`) share their
/// Arrow buffers directly with zero element-by-element copying.
pub fn collection_to_record_batches(
    context: &Context,
    columns: &[(String, LogicalType)],
    collection: &ColumnDataCollection<'_>,
) -> Result<(SchemaRef, Vec<RecordBatch>)> {
    let mut fields = Vec::with_capacity(columns.len());
    let mut exporter_cols = Vec::with_capacity(columns.len());

    for (name, logical_type) in columns {
        fields.push(bolt_storage_field(name, logical_type)?);
        exporter_cols.push((name.as_str(), logical_type.clone()));
    }
    let schema = Arc::new(Schema::new(fields));

    // Propagate a failing row-count call instead of treating it as an empty collection (which
    // would silently drop the whole batch).
    if collection.len()? == 0 {
        return Ok((schema, Vec::new()));
    }

    // Export through DuckDB's Arrow C Data Interface (`ArrowExporter`).
    let mut exporter = ArrowExporter::new(context, &exporter_cols, None)?;
    let mut scanner = collection.to_scan(context)?;
    while let Some(chunk) = scanner.next_chunk()? {
        exporter.append(&chunk, false)?;
    }
    exporter.flush()?;

    // SAFETY: `duck_ffi::ArrowSchema` and `FFI_ArrowSchema` share the exact Arrow C Data Interface
    // ABI (compile-time size/alignment assertions above). `exporter.schema()` returns a fresh,
    // owned schema (`duckdb_v2_arrow_exporter_get_schema`), and `into_raw` moves its release
    // duty out (the wrapper is left with a null `release`, so its `Drop` is a no-op). The
    // transmute moves the struct without running any `Drop`, so `FFI_ArrowSchema`'s `Drop` is
    // the single owner and calls `release` exactly once, on every return path below.
    let ffi_schema = unsafe {
        transmute::<duck_ffi::ArrowSchema, FFI_ArrowSchema>(exporter.schema()?.into_raw())
    };

    let mut exported_batches = Vec::new();
    while let Some(duck_arr) = exporter.next_array()? {
        // SAFETY: same ABI argument as for the schema. `next_array` only returns arrays with a
        // non-null `release`, `into_raw` moves the release duty out of the DuckDB wrapper, and
        // `from_ffi` takes ownership (`Arc<FFI_ArrowArray>` inside the returned `ArrayData`,
        // released when the last buffer is dropped, also on its error path). The array was
        // produced by DuckDB for `ffi_schema`, which `from_ffi` requires.
        let ffi_arr =
            unsafe { transmute::<duck_ffi::ArrowArray, FFI_ArrowArray>(duck_arr.into_raw()) };
        let mut array_data = unsafe { from_ffi(ffi_arr, &ffi_schema) }.map_err(|e| {
            Error::api_error(format!("Arrow C Data Interface from_ffi failed: {e}"))
        })?;
        // `from_ffi` builds the `ArrayData` with `new_unchecked`. Typed arrays built from it
        // (`StructArray::from`, `make_array`) *panic* on a misaligned fixed-width buffer (e.g. a
        // `Decimal128` buffer that is only 8-byte aligned, which the C Data Interface allows),
        // and a panic inside a DuckDB callback becomes an INTERNAL error that invalidates the
        // database. Copy misaligned buffers instead, and run the cheap structural validation
        // (buffer sizes, child lengths) so a malformed export is an error, not UB or a panic.
        array_data.align_buffers();
        array_data
            .validate()
            .map_err(|e| Error::api_error(format!("DuckDB Arrow export failed validation: {e}")))?;
        if !matches!(array_data.data_type(), DataType::Struct(_)) {
            return Err(Error::api_error(format!(
                "DuckDB Arrow export produced {:?}, expected a struct array",
                array_data.data_type()
            )));
        }
        let struct_arr = StructArray::from(array_data);
        if struct_arr.num_columns() != columns.len() {
            return Err(Error::api_error(format!(
                "DuckDB Arrow export produced {} columns, expected {}",
                struct_arr.num_columns(),
                columns.len()
            )));
        }
        if struct_arr.is_empty() {
            continue;
        }
        let mut normalized_cols = Vec::with_capacity(columns.len());
        for (col_idx, ((_, logical_type), field)) in
            columns.iter().zip(schema.fields().iter()).enumerate()
        {
            let raw_col = struct_arr.column(col_idx);
            normalized_cols.push(normalize_exported_column_for_bolt(
                raw_col,
                logical_type,
                field.data_type(),
            )?);
        }
        let batch = RecordBatch::try_new(schema.clone(), normalized_cols)
            .map_err(|e| Error::api_error(format!("Failed to build RecordBatch from FFI: {e}")))?;
        exported_batches.push(batch);
    }

    Ok((schema, exported_batches))
}

/// Execute `sql` on `conn` and stream the result zero-copy through DuckDB's
/// **Arrow C Stream Interface (`ArrowQueryResult::to_arrow_stream` -> `FFI_ArrowArrayStream`)**,
/// concatenated into one batch.
///
/// Prefer [`arrow_query_to_record_batches`], which keeps the stream's batches and so does not
/// hold the result twice while concatenating (W4-A-002).
pub fn arrow_query_to_record_batch(conn: &Connection, sql: &str) -> Result<RecordBatch> {
    let (stream_schema, mut batches) = arrow_query_to_record_batches(conn, sql)?;
    if batches.is_empty() {
        let empty_cols = stream_schema
            .fields()
            .iter()
            .map(|f| new_empty_array(f.data_type()))
            .collect::<Vec<_>>();
        RecordBatch::try_new(stream_schema, empty_cols)
            .map_err(|e| Error::api_error(format!("Failed to build empty stream RecordBatch: {e}")))
    } else if batches.len() == 1 {
        Ok(batches.pop().unwrap())
    } else {
        concat_batches(&stream_schema, &batches)
            .map_err(|e| Error::api_error(format!("Failed to concat Arrow stream batches: {e}")))
    }
}

/// Execute `sql` on `conn` and return the stream's schema and its batches as DuckDB produced
/// them (no concatenation; zero-row batches are dropped, so an empty result has no batches).
/// See [`arrow_query_to_record_batch`] for the stream handling.
pub fn arrow_query_to_record_batches(
    conn: &Connection,
    sql: &str,
) -> Result<(SchemaRef, Vec<RecordBatch>)> {
    let arrow_result = conn.arrow_query(sql, Parameters::None, None)?;
    let arrow_stream = arrow_result.to_arrow_stream()?;

    // SAFETY (`ArrowStream::into_raw`: the caller must release the stream before `conn` is used
    // again or dropped, and keep the returned collections alive until then):
    // * `conn` is borrowed for this whole function and not used again in it; `_collections` is
    //   declared before `reader`, so on every path (including early `?` returns and unwinding)
    //   `reader` (which owns and releases the stream) is dropped first. The explicit inner scope
    //   below makes that order visible instead of relying on declaration order.
    // * The transmute moves the struct without running a `Drop`: `ArrowArrayStreamReader` (via
    //   `FFI_ArrowArrayStream`'s `Drop`) is the single owner and calls `release` exactly once;
    //   `try_new` drops (releases) it itself when it fails.
    // * Arrays returned by `get_next` are independent of the stream (C Stream Interface rule)
    //   and own DuckDB-allocated copies of the result, so the returned batches stay valid after
    //   the stream and `_collections` are gone.
    let (raw_stream, _collections) = unsafe { arrow_stream.into_raw() };
    let ffi_stream =
        unsafe { transmute::<duck_ffi::ArrowArrayStream, FFI_ArrowArrayStream>(raw_stream) };
    let (stream_schema, batches) = {
        let mut reader = ArrowArrayStreamReader::try_new(ffi_stream).map_err(|e| {
            Error::api_error(format!("ArrowArrayStreamReader::try_new failed: {e}"))
        })?;
        let stream_schema = reader.schema();
        // arrow-rs's reader panics on a misaligned imported buffer (it builds typed arrays
        // directly from `new_unchecked` data) and on a failing `get_next` without an error
        // message (`get_last_error` returning NULL is unwrapped). A panic here would surface as
        // an INTERNAL error that invalidates the database, so turn it into an ordinary error.
        // The reader is borrowed by the closure and still dropped (released) normally.
        let collected = catch_unwind(AssertUnwindSafe(|| {
            let mut batches = Vec::new();
            for next in &mut reader {
                let batch =
                    next.map_err(|e| Error::api_error(format!("Arrow stream step failed: {e}")))?;
                if batch.num_rows() > 0 {
                    batches.push(batch);
                }
            }
            Ok::<_, Error>(batches)
        }))
        .map_err(|_| {
            Error::api_error(
                "Arrow stream import panicked (malformed or misaligned Arrow data from DuckDB)",
            )
        })??;
        drop(reader);
        (stream_schema, collected)
    };
    drop(_collections);
    Ok((stream_schema, batches))
}

/// Populate a named DuckDB [`ColumnDataCollection`] from Arrow [`RecordBatch`]es.
pub fn record_batches_to_collection<'a>(
    conn: &'a Connection,
    schema: &Arc<Schema>,
    batches: &[RecordBatch],
) -> Result<ColumnDataCollection<'a>> {
    let mut named_cols = Vec::with_capacity(schema.fields().len());
    let mut col_types = Vec::with_capacity(schema.fields().len());
    for field in schema.fields() {
        let lt = arrow_type_to_duckdb_type(conn, field.data_type())?;
        col_types.push(lt.clone());
        named_cols.push((lt, field.name().clone()));
    }
    let collection = ColumnDataCollection::new(conn, named_cols)?;
    let mut appender = collection.to_append()?;
    for batch in batches {
        // Every vector of each chunk must be written: a batch with fewer columns than `schema`
        // would leave the remaining vectors uninitialized (DuckDB would read garbage).
        if batch.num_columns() != col_types.len() {
            return Err(Error::api_error(format!(
                "record_batches_to_collection: batch has {} columns, schema has {}",
                batch.num_columns(),
                col_types.len()
            )));
        }
        let mut offset = 0;
        while offset < batch.num_rows() {
            let count = (batch.num_rows() - offset).min(STANDARD_CHUNK_CAPACITY);
            let mut chunk = DataChunk::create(conn, &col_types, true)?;
            for col_idx in 0..batch.num_columns() {
                let vec = chunk.get_vector_at_mut::<Unknown>(col_idx)?;
                write_arrow_column_to_vector(batch.column(col_idx), offset, count, vec)?;
            }
            appender.append(&mut chunk)?;
            offset += count;
        }
    }
    Ok(appender.to_normal())
}

/// Drain a DuckDB [`QueryResult`] into an Arrow [`RecordBatch`].
pub fn query_result_to_record_batch(result: &mut QueryResult<'_>) -> Result<RecordBatch> {
    let columns = result.schema()?.get_all()?;
    let mut accumulators = Vec::with_capacity(columns.len());
    let mut fields = Vec::with_capacity(columns.len());

    for (name, logical_type) in &columns {
        let (acc, dt) = ColumnAccumulator::from_duckdb_type(logical_type, 64)?;
        accumulators.push(acc);
        fields.push(Field::new(name.clone(), dt, true));
    }

    while let Some(chunk) = result.next_chunk()? {
        for (col_idx, ((_, logical_type), acc)) in
            columns.iter().zip(accumulators.iter_mut()).enumerate()
        {
            acc.append_from_chunk(&chunk, col_idx, logical_type)?;
        }
    }

    let schema = Arc::new(Schema::new(fields));
    let arrays = accumulators
        .into_iter()
        .map(ColumnAccumulator::finish)
        .collect::<Result<Vec<_>>>()?;

    RecordBatch::try_new(schema, arrays)
        .map_err(|e| Error::api_error(format!("Failed to build Arrow RecordBatch: {e}")))
}

#[cfg(test)]
mod tests {
    //! RFC-008 helper and FFI-preparation unit tests. Pure Arrow: no DuckDB instance, no GPU.

    use arrow_array::{
        DictionaryArray, RecordBatchOptions,
        types::{Decimal128Type, Int16Type, Int32Type, Int64Type, UInt32Type},
    };

    use super::*;

    const ALL_TYPES: [OriginalDuckDbType; 21] = [
        OriginalDuckDbType::Boolean,
        OriginalDuckDbType::TinyInt,
        OriginalDuckDbType::SmallInt,
        OriginalDuckDbType::Integer,
        OriginalDuckDbType::BigInt,
        OriginalDuckDbType::UTinyInt,
        OriginalDuckDbType::USmallInt,
        OriginalDuckDbType::UInteger,
        OriginalDuckDbType::UBigInt,
        OriginalDuckDbType::HugeInt,
        OriginalDuckDbType::Float,
        OriginalDuckDbType::Double,
        OriginalDuckDbType::Varchar,
        OriginalDuckDbType::Date,
        OriginalDuckDbType::TimestampSec,
        OriginalDuckDbType::TimestampMs,
        OriginalDuckDbType::Timestamp,
        OriginalDuckDbType::TimestampNs,
        OriginalDuckDbType::TimestampTz,
        OriginalDuckDbType::TimestampTzNs,
        OriginalDuckDbType::Decimal {
            width: 38,
            scale: 6,
        },
    ];

    fn meta_field(name: &str, data_type: DataType, text: &str) -> Field {
        Field::new(name, data_type, true).with_metadata(HashMap::from([(
            DUCKDB_TYPE_METADATA_KEY.to_string(),
            text.to_string(),
        )]))
    }

    fn parse(text: &str) -> Option<OriginalDuckDbType> {
        OriginalDuckDbType::parse_metadata_text(text)
    }

    #[test]
    fn metadata_text_round_trips_for_every_type() {
        for t in ALL_TYPES {
            assert_eq!(parse(&t.to_metadata_text()), Some(t), "{t:?}");
        }
        assert_eq!(
            OriginalDuckDbType::Decimal { width: 9, scale: 2 }.to_metadata_text(),
            "DECIMAL(9,2)"
        );
        assert_eq!(
            OriginalDuckDbType::TimestampTz.to_metadata_text(),
            "TIMESTAMPTZ"
        );
    }

    #[test]
    fn metadata_text_accepts_aliases_and_rejects_garbage() {
        use OriginalDuckDbType as T;
        assert_eq!(parse(" timestamp  with time   zone "), Some(T::TimestampTz));
        assert_eq!(parse("TIMESTAMP_TZ"), Some(T::TimestampTz));
        assert_eq!(parse("int2"), Some(T::SmallInt));
        assert_eq!(parse("Text"), Some(T::Varchar));
        assert_eq!(
            parse("Decimal( 9 , 2 )"),
            Some(T::Decimal { width: 9, scale: 2 })
        );
        assert_eq!(
            parse("NUMERIC(18,4)"),
            Some(T::Decimal {
                width: 18,
                scale: 4
            })
        );
        assert_eq!(parse("DECIMAL(5)"), Some(T::Decimal { width: 5, scale: 0 }));
        for bad in [
            "",
            "BLOB",
            "INTERVAL",
            "DECIMAL(39,0)",
            "DECIMAL(0,0)",
            "DECIMAL(4,5)",
            "DECIMAL(4,-1)",
            "DECIMAL(9,2",
            "DECIMAL()",
            "INTEGER[]",
            "STRUCT(a INTEGER)",
        ] {
            assert_eq!(parse(bad), None, "{bad:?} must not parse");
        }
    }

    #[test]
    fn storage_and_exact_arrow_types() {
        use OriginalDuckDbType as T;
        let utc = Some(Arc::<str>::from("UTC"));
        let cases = [
            (T::SmallInt, DataType::Int32, DataType::Int16),
            (T::TinyInt, DataType::Int32, DataType::Int8),
            (T::UTinyInt, DataType::Int32, DataType::UInt8),
            (T::USmallInt, DataType::Int32, DataType::UInt16),
            (T::Integer, DataType::Int32, DataType::Int32),
            (T::UInteger, DataType::Int64, DataType::UInt32),
            (T::UBigInt, DataType::Int64, DataType::UInt64),
            (T::HugeInt, DataType::Int64, DataType::Decimal128(38, 0)),
            (
                T::TimestampTz,
                DataType::Timestamp(TimeUnit::Microsecond, None),
                DataType::Timestamp(TimeUnit::Microsecond, utc.clone()),
            ),
            (
                T::TimestampTzNs,
                DataType::Timestamp(TimeUnit::Nanosecond, None),
                DataType::Timestamp(TimeUnit::Nanosecond, utc),
            ),
            (
                T::Decimal { width: 9, scale: 2 },
                DataType::Decimal128(9, 2),
                DataType::Decimal128(9, 2),
            ),
        ];
        for (t, storage, exact) in cases {
            assert_eq!(t.storage_arrow_type(), storage, "{t:?} storage");
            assert_eq!(t.exact_arrow_type(), exact, "{t:?} exact");
        }
        // Every storage type converts to its exact type (timestamps by re-annotation).
        for t in ALL_TYPES {
            let (storage, exact) = (t.storage_arrow_type(), t.exact_arrow_type());
            let timestamps = matches!(
                (&storage, &exact),
                (DataType::Timestamp(_, _), DataType::Timestamp(_, _))
            );
            assert!(
                timestamps || arrow_cast::can_cast_types(&storage, &exact),
                "{t:?}: {storage:?} -> {exact:?}"
            );
        }
    }

    #[test]
    fn restore_original_types_converts_to_exact_types() {
        let tz_micros = 1_772_326_923_456_789_i64;
        let schema = Arc::new(Schema::new(vec![
            meta_field("s", DataType::Int32, "SMALLINT"),
            meta_field("u", DataType::Int64, "UINTEGER"),
            meta_field("h", DataType::Int64, "HUGEINT"),
            meta_field(
                "tz",
                DataType::Timestamp(TimeUnit::Microsecond, None),
                "TIMESTAMPTZ",
            ),
            meta_field("d", DataType::Decimal128(9, 2), "DECIMAL(9,2)"),
            meta_field("i", DataType::Int32, "INTEGER"),
            Field::new("plain", DataType::Utf8, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![Some(-32768), None, Some(32767)])),
                Arc::new(Int64Array::from(vec![Some(0), None, Some(4_294_967_295)])),
                Arc::new(Int64Array::from(vec![Some(i64::MIN), None, Some(7)])),
                Arc::new(TimestampMicrosecondArray::from(vec![
                    Some(0),
                    None,
                    Some(tz_micros),
                ])),
                Arc::new(
                    Decimal128Array::from(vec![Some(-999_999_999), None, Some(12_345)])
                        .with_precision_and_scale(9, 2)
                        .unwrap(),
                ),
                Arc::new(Int32Array::from(vec![Some(1), None, Some(3)])),
                Arc::new(StringArray::from(vec![Some("a"), None, Some("c")])),
            ],
        )
        .unwrap();

        let restored = restore_original_types(&batch).unwrap();
        let types = restored
            .schema()
            .fields()
            .iter()
            .map(|f| f.data_type().clone())
            .collect::<Vec<_>>();
        assert_eq!(
            types,
            vec![
                DataType::Int16,
                DataType::UInt32,
                DataType::Decimal128(38, 0),
                DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                DataType::Decimal128(9, 2),
                DataType::Int32,
                DataType::Utf8,
            ]
        );
        assert_eq!(
            restored
                .column(0)
                .as_primitive::<Int16Type>()
                .iter()
                .collect::<Vec<_>>(),
            vec![Some(-32768), None, Some(32767)]
        );
        assert_eq!(
            restored
                .column(1)
                .as_primitive::<UInt32Type>()
                .iter()
                .collect::<Vec<_>>(),
            vec![Some(0), None, Some(4_294_967_295)]
        );
        assert_eq!(
            restored
                .column(2)
                .as_primitive::<Decimal128Type>()
                .iter()
                .collect::<Vec<_>>(),
            vec![Some(i128::from(i64::MIN)), None, Some(7)]
        );
        // TIMESTAMPTZ: same UTC instants, only the annotation changes.
        let tz = restored
            .column(3)
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .unwrap();
        assert_eq!(tz.timezone(), Some("UTC"));
        assert_eq!(
            tz.iter().collect::<Vec<_>>(),
            vec![Some(0), None, Some(tz_micros)]
        );
        // Unchanged columns are shared, not copied.
        assert!(Arc::ptr_eq(restored.column(4), batch.column(4)));
        assert!(Arc::ptr_eq(restored.column(5), batch.column(5)));
        assert!(Arc::ptr_eq(restored.column(6), batch.column(6)));
        // Metadata survives, so restoring twice is a no-op.
        assert_eq!(
            original_duckdb_type(restored.schema().field(0)),
            Some(OriginalDuckDbType::SmallInt)
        );
        assert_eq!(
            original_duckdb_type(restored.schema().field(3)),
            Some(OriginalDuckDbType::TimestampTz)
        );
        let twice = restore_original_types(&restored).unwrap();
        assert_eq!(twice.schema(), restored.schema());
        for i in 0..twice.num_columns() {
            assert!(Arc::ptr_eq(twice.column(i), restored.column(i)));
        }
    }

    #[test]
    fn restore_rejects_values_outside_the_original_type() {
        let small = RecordBatch::try_new(
            Arc::new(Schema::new(vec![meta_field(
                "v",
                DataType::Int32,
                "SMALLINT",
            )])),
            vec![Arc::new(Int32Array::from(vec![1, 70_000]))],
        )
        .unwrap();
        let err = restore_original_types(&small).unwrap_err();
        assert!(err.message.contains("'v'"), "{}", err.message);

        let unsigned = RecordBatch::try_new(
            Arc::new(Schema::new(vec![meta_field(
                "n",
                DataType::Int64,
                "UINTEGER",
            )])),
            vec![Arc::new(Int64Array::from(vec![-1]))],
        )
        .unwrap();
        assert!(restore_original_types(&unsigned).is_err());
    }

    #[test]
    fn stale_or_unknown_metadata_is_ignored() {
        // Metadata that does not match the storage type (re-typed column) or does not parse.
        assert_eq!(
            original_duckdb_type(&meta_field("a", DataType::Utf8, "SMALLINT")),
            None
        );
        assert_eq!(
            original_duckdb_type(&meta_field("a", DataType::Int64, "SMALLINT")),
            None
        );
        assert_eq!(
            original_duckdb_type(&meta_field("a", DataType::Int32, "BLOB")),
            None
        );
        assert_eq!(
            original_duckdb_type(&Field::new("a", DataType::Int32, true)),
            None
        );
        // The exact type is accepted as well (already restored).
        assert_eq!(
            original_duckdb_type(&meta_field("a", DataType::Int16, "smallint")),
            Some(OriginalDuckDbType::SmallInt)
        );

        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![meta_field(
                "a",
                DataType::Utf8,
                "SMALLINT",
            )])),
            vec![Arc::new(StringArray::from(vec!["x"]))],
        )
        .unwrap();
        let restored = restore_original_types(&batch).unwrap();
        assert!(Arc::ptr_eq(restored.column(0), batch.column(0)));
        assert_eq!(restored.schema(), batch.schema());
    }

    #[test]
    fn original_types_schema_retypes_only_annotated_fields() {
        let schema = Schema::new_with_metadata(
            vec![
                meta_field("s", DataType::Int32, "USMALLINT"),
                Field::new("plain", DataType::Int32, false),
            ],
            HashMap::from([("k".to_string(), "v".to_string())]),
        );
        let target = original_types_schema(&schema);
        assert_eq!(target.field(0).data_type(), &DataType::UInt16);
        assert_eq!(target.field(0).metadata(), schema.field(0).metadata());
        assert_eq!(target.field(1), schema.field(1));
        assert_eq!(target.metadata(), schema.metadata());
    }

    #[test]
    fn prepare_column_for_ffi_unpacks_dictionaries_with_null_values() {
        // A NULL dictionary value referenced by a valid key is NULL.
        let values = Int64Array::from(vec![Some(10), None, Some(30)]);
        let keys = Int32Array::from(vec![Some(0), Some(1), None, Some(2)]);
        let dict = DictionaryArray::<Int32Type>::try_new(keys, Arc::new(values)).unwrap();
        let out = prepare_column_for_ffi(Arc::new(dict)).unwrap();
        assert_eq!(out.data_type(), &DataType::Int64);
        assert_eq!(
            out.as_primitive::<Int64Type>().iter().collect::<Vec<_>>(),
            vec![Some(10), None, None, Some(30)]
        );

        // Sliced dictionary (non-zero offset) with NULL values.
        let values = Int64Array::from(vec![Some(1), None]);
        let keys = Int32Array::from(vec![0, 1, 0, 1, 0]);
        let dict: ArrayRef =
            Arc::new(DictionaryArray::<Int32Type>::try_new(keys, Arc::new(values)).unwrap());
        let out = prepare_column_for_ffi(dict.slice(1, 3)).unwrap();
        assert_eq!(
            out.as_primitive::<Int64Type>().iter().collect::<Vec<_>>(),
            vec![None, Some(1), None]
        );

        // Without NULL values the column stays dictionary-encoded (zero-copy export).
        let keys = Int32Array::from(vec![0, 1, 0]);
        let dict = DictionaryArray::<Int32Type>::try_new(
            keys,
            Arc::new(StringArray::from(vec!["a", "b"])),
        )
        .unwrap();
        let out = prepare_column_for_ffi(Arc::new(dict)).unwrap();
        assert!(matches!(out.data_type(), DataType::Dictionary(_, _)));
    }

    #[test]
    fn prepare_column_for_ffi_validates_decimal_dictionaries() {
        // 12345 does not fit DECIMAL(4, 0): must error, not reach DuckDB.
        let values = Decimal128Array::from(vec![12_345_i128])
            .with_precision_and_scale(4, 0)
            .unwrap();
        let keys = Int32Array::from(vec![0, 0]);
        let dict = DictionaryArray::<Int32Type>::try_new(keys, Arc::new(values)).unwrap();
        assert!(prepare_column_for_ffi(Arc::new(dict)).is_err());

        let plain: ArrayRef = Arc::new(
            Decimal128Array::from(vec![12_345_i128])
                .with_precision_and_scale(4, 0)
                .unwrap(),
        );
        assert!(prepare_column_for_ffi(plain).is_err());
    }

    #[test]
    fn cast_batch_to_schema_checked_keeps_row_count_for_zero_columns() {
        let empty_schema = Arc::new(Schema::empty());
        let batch = RecordBatch::try_new_with_options(
            empty_schema.clone(),
            vec![],
            &RecordBatchOptions::new().with_row_count(Some(5)),
        )
        .unwrap();
        let out = cast_batch_to_schema_checked(&batch, &empty_schema).unwrap();
        assert_eq!(out.num_rows(), 5);
        assert_eq!(restore_original_types(&batch).unwrap().num_rows(), 5);
    }
}

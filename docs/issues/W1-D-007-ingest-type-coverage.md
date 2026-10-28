# W1-D-007: Ingestion type coverage and lossy conversions

- **Severity:** low
- **Status:** blocked-upstream (items 2, 3, 4 fixed; item 1 needs Craton Bolt types)
- **Files:** `crates/core/src/ingest.rs` (`bolt_target_type`)

## Description

Checked against the arrow-cast 53.4.1 support matrix: every mapping in `bolt_target_type` is a cast
arrow implements (`Float16 -> Float32`, `Date64 -> Date32`, `Utf8View`/`LargeUtf8 -> Utf8`, and dictionary
unpacking via `unpack_dictionary`). W1-D made two changes:

- dictionaries are accepted with any supported value type, not only strings;
- negative-scale `Decimal128` is now rejected. The DuckDB bridge would clamp the scale to 0 and change
  every value by 10^|scale| (review 1.1.9).

Remaining gaps:

1. **Unsupported types are rejected with an error:** `Time32/Time64`, `Duration`, `Interval`,
   `Binary`/`LargeBinary`/`BinaryView`/`FixedSizeBinary` (including UUID), `Decimal256`, `Null`, and
   nested types (`List`, `Struct`, `Map`). DuckDB itself supports all of them, so the workaround is
   `COPY (SELECT * FROM read_parquet(...)) TO 't' (FORMAT bolt)`.
2. **`Date64 -> Date32` silently truncates** non-midnight milliseconds (arrow divides by 86,400,000).
   The Arrow spec requires whole days, but some writers violate it.
3. **Field metadata and Parquet key/value metadata are dropped** (for example pandas or GeoParquet
   metadata).
4. **`Timestamp(unit, tz)` is passed through unchanged.** With a non-UTC zone, the round-trip through
   the bridge depends on review 1.1.2.

## Suggested fix

Map `Time*` to `Int64` microseconds, or add Bolt support. Reject `Date64` values with a remainder
(check `value % 86_400_000 == 0` before casting). Keep schema metadata on the registered table.

## Update (W2-D)

- Item 2 fixed: `Date64` values (plain or dictionary-encoded) that are not whole days are rejected
  with a column-level error instead of being truncated (`check_date64_whole_days`).
- Item 4 fixed: `Timestamp(unit, tz)` is normalized to `Timestamp(unit, None)`; values are UTC
  instants and stay unchanged (W1-E-010).
- Every ingested field is now nullable (W1-E-010).
- Item 1: still rejected, but the error now names the Arrow type in display form and points to the
  `COPY (SELECT ... FROM read_parquet(...)) TO '<table>' (FORMAT bolt)` workaround. Mapping `Time*`
  to `Int64` would change the logical type users see, so it is left for Bolt-side support.
- Item 3 unchanged (metadata dropped; documented in the `ingest.rs` module docs).
- New gap found: the `parquet` dependency is built with only the `snap` and `flate2` codecs, so ZSTD,
  LZ4 and Brotli compressed files cannot be loaded (W2-D-002).

## Update (W3-D, commit 1fec37a)

- Item 3 fixed: field metadata and schema metadata (for example pandas or GeoParquet keys) are
  kept on the registered table, and every field also records its original DuckDB type (RFC-008,
  W2-C-002). Test: `test_loaded_files_record_original_duckdb_types`.
- Zoned `Timestamp(s | ms, tz)` columns are now stored as `Timestamp(us, None)` (checked
  conversion) so they match `TIMESTAMPTZ` storage.
- Item 1 unchanged.

## Resolution (W4-C)

Item 1 is blocked upstream. Craton Bolt 0.9 stores only the types listed in `bolt_target_type`,
and the RFC-008 metadata that lets `bolt_scan` restore a narrower DuckDB type
(`OriginalDuckDbType`, `arrow_bridge.rs`) has no `TIME`, `INTERVAL`, `BLOB`, `UUID` or nested
entries. Mapping `Time*` to `Int64` here would make `bolt_scan` return `BIGINT` for a `TIME`
column, the lossy change RFC-008 exists to avoid. When craton-bolt gains the types (W1-O-003, the
0.10 upgrade), add them to `bolt_target_type`, `original_duckdb_type_of` and `OriginalDuckDbType`
together. The error already names the Arrow type and the `COPY ... (FORMAT bolt)` workaround.

Also new since W3-D: every Parquet codec except LZO, and LZ4 / ZSTD IPC compression, now decode
(W3-D-001).

## Update (W5-B)

No change: item 1 remains blocked on Craton Bolt types (W1-O-003). The `CALL` forms of
`bolt_load_parquet` / `bolt_load_arrow_ipc` (W4-C-004) use the same `bolt_target_type` mapping
and errors.

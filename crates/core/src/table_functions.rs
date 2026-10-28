// SPDX-License-Identifier: Apache-2.0

//! DuckDB table functions (`bolt_query`, `bolt_sql`, `bolt_scan`, `bolt_aggregate`, `bolt_tables`,
//! `bolt_pool_stats`, `bolt_metrics`, `bolt_devices`), and the `CALL` forms of the catalog
//! mutations (`bolt_clear_tables`, `bolt_drop_table`, `bolt_register_sql`, `bolt_load_parquet`,
//! `bolt_load_arrow_ipc`), which act once per execution.

use std::{
    cell::{Cell, OnceCell, RefCell},
    collections::{BTreeMap, HashMap, HashSet},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};

use arrow_array::{
    Array, ArrayRef, Int64Array, RecordBatch, RecordBatchOptions, UInt64Array, cast::AsArray,
    types::Int64Type,
};
use arrow_schema::{DataType, Field, FieldRef, Schema, SchemaRef};
use arrow_select::{concat::concat, take::take};
use duckdb_neo::{
    DuckDBType, Error, FromValue, Result,
    bind_arguments::BindArgument,
    connection::Context,
    data_chunk::DataChunkRef,
    expression::{Expression, ExpressionType},
    extension::Extension,
    logical_type::{LogicalType, LogicalTypeID},
    signature::{FunctionDocumentationBuilder, FunctionParameter, SignatureBuilder},
    table_function::{
        BindFunctionHandle, ExecColumnInfo, InitColumnData, OrderPreservation, PartitionData,
        PushdownData, TableFilterPushdownCallbacks, TableFunctionBuilder, TableFunctionCallbacks,
        TablePartitioningCallbacks,
    },
    types::{
        DateValue, DecimalSignature, DecimalValue, TimestampMsValue, TimestampNsValue,
        TimestampSecValue, TimestampTzValue, TimestampValue,
    },
    value::Value,
    vector::Unknown,
};

use crate::{
    arrow_bridge::{
        OriginalDuckDbType, STANDARD_CHUNK_CAPACITY, arrow_field_to_duckdb_type,
        cast_array_to_exact_type, original_duckdb_type, original_exact_arrow_type,
        write_record_batch_slice_to_chunk_zero_copy,
    },
    runtime::{
        BoltConfig, BoltRuntime, BoltTelemetrySnapshot, DeviceInfo, MetricRow, RegisterMode,
        SCRATCH_TABLE_NAME, TableSummary,
    },
};

// ---------------------------------------------------------------------------
// 0. Shared chunked result state (bolt_query, bolt_sql, bolt_aggregate, bolt_scan)
// ---------------------------------------------------------------------------

/// Minimum number of output chunks per DuckDB worker thread before a scan asks for more
/// threads (2048 rows per chunk, so 16 chunks is ~32k rows).
const CHUNKS_PER_THREAD: usize = 16;
/// Upper bound on the threads a single scan asks DuckDB for.
const MAX_SCAN_THREADS: usize = 64;

/// One DuckDB output chunk: rows `offset .. offset + len` of `batches[batch]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ChunkRef {
    batch: usize,
    offset: usize,
    len: usize,
    /// Rows in all chunks before this one (for progress reporting).
    rows_before: usize,
}

/// A materialized result exposed to DuckDB chunk by chunk, without concatenating batches.
///
/// Every `exec` call claims the next chunk with one atomic increment, so several DuckDB
/// threads can scan it at once. The chunk index doubles as the batch index reported through
/// [`TablePartitioningCallbacks::partition_data`], which lets DuckDB keep insertion order
/// while scanning in parallel.
///
/// The chunk plan is built on the first `exec` from the output chunk's capacity (clamped to
/// [`STANDARD_CHUNK_CAPACITY`]), so a DuckDB built with a smaller `STANDARD_VECTOR_SIZE` never
/// receives more rows than its chunks hold (docs/issues/W3-C-002).
struct ChunkedResult {
    batches: Vec<RecordBatch>,
    /// Column of `batches` written to each DuckDB output vector (`usize::MAX` or any other
    /// out-of-range index: an all-NULL virtual column).
    projected_columns: Vec<usize>,
    chunks: std::sync::OnceLock<Vec<ChunkRef>>,
    total_rows: usize,
    next_chunk: AtomicUsize,
}

/// Per-thread scan state: the index of the chunk this thread produced last.
#[derive(Default)]
struct ChunkLocalState {
    batch_index: usize,
}

/// Rows a table-function `exec` may write into `output`: its capacity, clamped to
/// `1 ..= STANDARD_CHUNK_CAPACITY` (docs/issues/W3-C-002).
fn output_rows(output: &DataChunkRef<'_>) -> Result<usize> {
    Ok(output.capacity()?.clamp(1, STANDARD_CHUNK_CAPACITY))
}

/// Split `batches` into chunks of at most `chunk_rows` rows.
fn plan_chunks(batches: &[RecordBatch], chunk_rows: usize) -> Vec<ChunkRef> {
    let chunk_rows = chunk_rows.max(1);
    let mut chunks = Vec::new();
    let mut rows_before = 0usize;
    for (batch_idx, batch) in batches.iter().enumerate() {
        let rows = batch.num_rows();
        let mut offset = 0usize;
        while offset < rows {
            let len = (rows - offset).min(chunk_rows);
            chunks.push(ChunkRef {
                batch: batch_idx,
                offset,
                len,
                rows_before,
            });
            offset += len;
            rows_before += len;
        }
    }
    chunks
}

impl ChunkedResult {
    fn new(batches: Vec<RecordBatch>, projected_columns: Vec<usize>) -> Self {
        let total_rows = batches.iter().map(RecordBatch::num_rows).sum();
        Self {
            batches,
            projected_columns,
            chunks: std::sync::OnceLock::new(),
            total_rows,
            next_chunk: AtomicUsize::new(0),
        }
    }

    #[cfg(test)]
    fn single(batch: RecordBatch, projected_columns: Vec<usize>) -> Self {
        Self::new(vec![batch], projected_columns)
    }

    /// Thread limit to report to DuckDB (estimated with standard-size chunks).
    fn max_threads(&self) -> usize {
        let chunks: usize = self
            .batches
            .iter()
            .map(|b| b.num_rows().div_ceil(STANDARD_CHUNK_CAPACITY))
            .sum();
        (chunks / CHUNKS_PER_THREAD).clamp(1, MAX_SCAN_THREADS)
    }

    fn progress(&self) -> f64 {
        if self.total_rows == 0 {
            return 1.0;
        }
        let Some(chunks) = self.chunks.get() else {
            return 0.0;
        };
        let claimed = self.next_chunk.load(Ordering::Relaxed);
        let done = chunks
            .get(claimed)
            .map_or(self.total_rows, |c| c.rows_before);
        done as f64 / self.total_rows as f64
    }

    fn write_next(
        &self,
        context: &Context,
        local: Option<&mut ChunkLocalState>,
        output: &mut DataChunkRef<'_>,
    ) -> Result<()> {
        let capacity = output_rows(output)?;
        let chunks = self
            .chunks
            .get_or_init(|| plan_chunks(&self.batches, capacity));
        let idx = self.next_chunk.fetch_add(1, Ordering::Relaxed);
        let Some(chunk) = chunks.get(idx) else {
            output.get_vector_at_mut::<Unknown>(0)?.set_size(0)?;
            return Ok(());
        };
        if chunk.len > capacity {
            // Every chunk of one query has the same capacity; this only guards the invariant.
            return Err(Error::api_error(format!(
                "Bolt result chunk of {} rows does not fit a DuckDB output chunk of {capacity} rows",
                chunk.len
            )));
        }
        if let Some(local) = local {
            local.batch_index = idx;
        }
        write_record_batch_slice_to_chunk_zero_copy(
            context,
            &self.batches[chunk.batch],
            chunk.offset,
            chunk.len,
            Some(&self.projected_columns),
            output,
        )?;
        Ok(())
    }
}

/// Reconcile this database's `bolt_*` settings with the process-wide runtime (stale values,
/// startup options, PTX cache use-time check) at the start of every table-function bind
/// (docs/issues/W3-D-004).
fn sync_options(context: &Context) -> Result<()> {
    crate::options::sync_with_runtime(context).map_err(|e| Error {
        code: e.code,
        message: format!("Bolt settings: {}", e.message),
    })
}

// ---------------------------------------------------------------------------
// 1. bolt_query(sql VARCHAR) / bolt_sql(sql VARCHAR)
// ---------------------------------------------------------------------------

struct BoltQueryTableFunction;

struct BoltQueryBindData {
    sql: String,
    /// The Bolt catalog when the query was bound, to tell a result that no longer fits the
    /// declared schema because the tables changed from one that never fit (W9-A-001).
    bound_catalog: CatalogFingerprint,
    /// Schema declared to DuckDB at bind time (inferred by planning, or taken from
    /// an eager execution when inference was not possible).
    schema: SchemaRef,
    /// Pre-executed result, present only when schema inference failed and the query
    /// had to be executed at bind time (RFC-002 fallback). Used by at most one execution, and
    /// only while the Bolt catalog is unchanged since bind: DuckDB reuses the bind data of a
    /// prepared statement for every `EXECUTE` (docs/issues/W3-E-001).
    eager: Mutex<Option<EagerResult>>,
}

/// A bind-time result (in the batches it was produced in, W4-A-002) and the catalog state it
/// was computed from.
struct EagerResult {
    batches: Vec<RecordBatch>,
    catalog: CatalogFingerprint,
}

/// Cheap identity of the Bolt catalog: every table's name, row count, byte size and schema
/// allocation (a replace installs a new schema, an append changes the row count).
///
/// The schemas themselves are kept (and compared with [`Arc::ptr_eq`]), not their addresses:
/// holding the `Arc` keeps its allocation alive, so a later replace can never be handed the same
/// address and look unchanged (docs/issues/W4-D-005).
#[derive(Debug)]
struct CatalogFingerprint(Vec<(String, usize, usize, SchemaRef)>);

impl PartialEq for CatalogFingerprint {
    fn eq(&self, other: &Self) -> bool {
        self.0.len() == other.0.len()
            && self
                .0
                .iter()
                .zip(&other.0)
                .all(|(a, b)| a.0 == b.0 && a.1 == b.1 && a.2 == b.2 && Arc::ptr_eq(&a.3, &b.3))
    }
}

impl CatalogFingerprint {
    fn current() -> Result<Self> {
        let mut tables: Vec<_> = BoltRuntime::global()
            .list_tables()?
            .into_iter()
            .map(|t| (t.name, t.num_rows, t.bytes_in_memory, t.schema))
            .collect();
        tables.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(Self(tables))
    }
}

/// Cast an executed batch so it matches the schema that was declared at bind time.
fn conform_batch_to_schema(batch: RecordBatch, schema: &SchemaRef) -> Result<RecordBatch> {
    if batch.num_columns() != schema.fields().len() {
        return Err(Error::api_error(format!(
            "bolt_query result has {} columns but {} were declared at bind time",
            batch.num_columns(),
            schema.fields().len()
        )));
    }
    // Fast path: the types already match, so only the schema metadata (names, nullability)
    // is swapped; no column is touched.
    let types_match = batch
        .columns()
        .iter()
        .zip(schema.fields())
        .all(|(col, field)| col.data_type() == field.data_type());
    if types_match {
        if batch.schema() == *schema {
            return Ok(batch);
        }
        return RecordBatch::try_new_with_options(
            schema.clone(),
            batch.columns().to_vec(),
            &RecordBatchOptions::new().with_row_count(Some(batch.num_rows())),
        )
        .map_err(|e| Error::api_error(format!("bolt_query schema conformance failed: {e}")));
    }
    let mut columns = Vec::with_capacity(batch.num_columns());
    for (col, field) in batch.columns().iter().zip(schema.fields()) {
        if col.data_type() == field.data_type() {
            columns.push(col.clone());
        } else {
            // Checked, so values that cannot be represented in the declared type raise an
            // error instead of being silently replaced by NULL; zoned timestamps are only
            // re-annotated.
            let cast = cast_array_to_exact_type(col, field.data_type());
            columns.push(cast.map_err(|e| {
                Error::invalid_input(format!(
                    "bolt_query column '{}' cannot be cast from {:?} to {:?}: {}",
                    field.name(),
                    col.data_type(),
                    field.data_type(),
                    e.message
                ))
            })?);
        }
    }
    RecordBatch::try_new(schema.clone(), columns)
        .map_err(|e| Error::api_error(format!("bolt_query schema conformance failed: {e}")))
}

/// Result-column names for `fields`: empty names become `col_<i>` and names that collide
/// case-insensitively (DuckDB column names are case-insensitive) get a `_<n>` suffix.
fn unique_column_names(fields: &arrow_schema::Fields) -> Vec<String> {
    let mut used = HashSet::with_capacity(fields.len());
    fields
        .iter()
        .enumerate()
        .map(|(idx, field)| {
            let base = if field.name().is_empty() {
                format!("col_{idx}")
            } else {
                field.name().clone()
            };
            let mut name = base.clone();
            let mut suffix = 1usize;
            while !used.insert(name.to_ascii_lowercase()) {
                name = format!("{base}_{suffix}");
                suffix += 1;
            }
            name
        })
        .collect()
}

fn bind_bolt_sql_query(
    context: &Context,
    sql: String,
    bind_handle: &BindFunctionHandle<'_>,
) -> Result<BoltQueryBindData> {
    if sql.trim().is_empty() {
        return Err(Error::invalid_input(
            "bolt_query SQL argument cannot be empty",
        ));
    }

    // RFC-002: plan-only schema inference so EXPLAIN / PREPARE / LIMIT 0 do not
    // execute the query. Fall back to eager execution when the planner cannot
    // statically resolve the query.
    let (schema, eager) = match BoltRuntime::global().infer_sql_schema(&sql) {
        Ok(inferred) => {
            // Nullable, keeping the RFC-008 metadata that names the DuckDB type (W9-A-001).
            let fields: Vec<Field> = inferred
                .fields()
                .iter()
                .map(|f| f.as_ref().clone().with_nullable(true))
                .collect();
            (Arc::new(Schema::new(fields)), None)
        }
        Err(_) => {
            let catalog = CatalogFingerprint::current()?;
            let out = BoltRuntime::global().execute_query_batches(&sql)?;
            (
                out.schema,
                Some(EagerResult {
                    batches: out.batches,
                    catalog,
                }),
            )
        }
    };

    // The inferred schema has DuckDB's result types (RFC-008 metadata on each field, W9-A-001);
    // the engine's storage-typed results are cast to them by `conform_batch_to_schema`.
    for (field, col_name) in schema
        .fields()
        .iter()
        .zip(unique_column_names(schema.fields()))
    {
        let duckdb_ty = arrow_field_to_duckdb_type(context, field)?;
        bind_handle.add_result_column(&col_name, &duckdb_ty)?;
    }

    match &eager {
        Some(eager) => bind_handle
            .set_cardinality(eager.batches.iter().map(RecordBatch::num_rows).sum(), true)?,
        None => bind_handle.set_cardinality(1000, false)?,
    }
    bind_handle.set_order(OrderPreservation::Insertion)?;

    Ok(BoltQueryBindData {
        sql,
        bound_catalog: CatalogFingerprint::current()?,
        schema,
        eager: Mutex::new(eager),
    })
}

impl TableFunctionCallbacks for BoltQueryTableFunction {
    type BindData = BoltQueryBindData;
    type GlobalState = ChunkedResult;
    type LocalState = ChunkLocalState;

    fn bind(
        &self,
        context: &Context,
        arguments: Vec<BindArgument>,
        bind_handle: BindFunctionHandle<'_>,
    ) -> Result<Self::BindData> {
        sync_options(context)?;
        let sql_arg = arguments.first().ok_or_else(|| {
            Error::invalid_input("bolt_query requires a SQL query string argument")
        })?;
        let sql_val = sql_arg.value.as_ref().ok_or_else(|| {
            Error::invalid_input("bolt_query SQL argument must be a constant string at bind time")
        })?;
        let sql = String::from_value(sql_val)?
            .ok_or_else(|| Error::invalid_input("bolt_query SQL argument cannot be NULL"))?;

        bind_bolt_sql_query(context, sql, &bind_handle)
    }
    fn init_global_state(
        &self,
        bind_data: &Self::BindData,
        _context: &Context,
        column_data: InitColumnData<'_>,
    ) -> Result<(Option<Self::GlobalState>, Option<usize>)> {
        let projected_columns = (0..column_data.count()?)
            .map(|i| column_data.get_column_index(i))
            .collect::<Result<Vec<_>>>()?;

        // The bind-time result serves the first execution if the catalog has not changed
        // since; any other execution (a prepared statement's later `EXECUTE`, or a catalog
        // change between PREPARE and EXECUTE) runs the query again (docs/issues/W3-E-001).
        let eager = bind_data
            .eager
            .lock()
            .map_err(|_| Error::api_error("bolt_query eager result mutex poisoned"))?
            .take();
        let batches = match eager {
            Some(eager) if eager.catalog == CatalogFingerprint::current()? => eager.batches,
            _ => BoltRuntime::global()
                .execute_query_batches(&bind_data.sql)?
                .batches
                .into_iter()
                .map(|executed| {
                    conform_batch_to_schema(executed, &bind_data.schema).map_err(|e| {
                        // Only blame the catalog when it did change (W9-A-001).
                        let changed = CatalogFingerprint::current()
                            .is_ok_and(|now| now != bind_data.bound_catalog);
                        if changed {
                            Error::invalid_input(format!(
                                "{}; the Bolt tables it reads changed after the query was bound, re-run the query",
                                e.message
                            ))
                        } else {
                            e
                        }
                    })
                })
                .collect::<Result<Vec<_>>>()?,
        };
        let state = ChunkedResult::new(batches, projected_columns);
        let threads = state.max_threads();
        Ok((Some(state), Some(threads)))
    }

    fn init_local_state(
        &self,
        _bind_data: &Self::BindData,
        _global_state: Option<&Self::GlobalState>,
        _context: &Context,
        _column_data: InitColumnData<'_>,
    ) -> Result<Option<Self::LocalState>> {
        Ok(Some(ChunkLocalState::default()))
    }

    fn progress(
        &self,
        _bind_data: &Self::BindData,
        global_state: Option<&Self::GlobalState>,
        _context: &Context,
    ) -> Result<Option<f64>> {
        Ok(global_state.map(ChunkedResult::progress))
    }

    fn exec(
        &self,
        _bind_data: &Self::BindData,
        global_state: Option<&Self::GlobalState>,
        local_state: Option<&mut Self::LocalState>,
        context: &Context,
        mut output: DataChunkRef<'_>,
        _column_info: ExecColumnInfo<'_>,
    ) -> Result<()> {
        global_state
            .ok_or_else(|| Error::api_error("Missing bolt_query global state"))?
            .write_next(context, local_state, &mut output)
    }
}

impl TablePartitioningCallbacks for BoltQueryTableFunction {
    fn partition_data(
        &self,
        _bind_data: &Self::BindData,
        _global_state: Option<&Self::GlobalState>,
        local_state: Option<&mut Self::LocalState>,
        _context: &Context,
        _partition_data: PartitionData<'_>,
    ) -> Result<usize> {
        Ok(local_state.map_or(0, |l| l.batch_index))
    }
}

// ---------------------------------------------------------------------------
// 2. bolt_scan(table_name VARCHAR) — with projection & filter pushdown
// ---------------------------------------------------------------------------

/// Utf8 predicates that the engine lowers through its string dictionary (`LIKE`, `<` / `>`
/// against a literal, and the `LIKE '%'` NULL guard) become a left-deep OR of one integer
/// equality per matching dictionary entry, which the engine walks recursively: about 250
/// entries overflow its worker-thread stack and abort the process
/// (docs/issues/W2-B-001-engine-stack-overflow-on-deep-predicates.md). Such predicates are
/// only pushed for columns with at most this many distinct values.
const UTF8_DICTIONARY_PUSHDOWN_LIMIT: usize = 64;
/// Upper bound on the predicate terms (comparisons, IN items, dictionary entries) pushed for
/// one scan; the engine's SQL frontend also recurses once per AND / OR term (300 terms
/// overflow its stack).
const PUSHDOWN_TERM_BUDGET: usize = 128;
/// The engine rejects longer IN lists ("IN with > 64 values not yet supported").
const MAX_IN_LIST_ITEMS: usize = 64;

struct BoltScanTableFunction;

/// A filter `bolt_scan` evaluates on the engine.
#[derive(Clone, Debug)]
struct PushedFilter {
    sql: String,
    /// Columns (bind-schema indices) the predicate references.
    columns: Vec<usize>,
    /// Utf8 columns whose dictionary must stay within [`UTF8_DICTIONARY_PUSHDOWN_LIMIT`]
    /// for the predicate to be safe to send to the engine.
    dictionary_columns: Vec<usize>,
    /// `true`: the filter was accepted, DuckDB no longer evaluates it. `false`: a
    /// pre-filter that is TRUE for every row the DuckDB filter keeps (DuckDB still applies
    /// the exact filter on top), e.g. an inclusive translation of a `BETWEEN`.
    exact: bool,
}

struct BoltScanBindData {
    table_name: String,
    schema: SchemaRef,
    pushed_filters: Mutex<Vec<PushedFilter>>,
}

/// How the last `bolt_scan` produced its rows (test hook).
#[doc(hidden)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BoltScanTrace {
    /// `"unfiltered"`, `"gpu"`, `"filter-take"` or `"host-fallback"`.
    pub path: &'static str,
    /// Pushed predicates and whether each one was accepted (exact) or is a pre-filter.
    pub filters: Vec<(String, bool)>,
}

static LAST_SCAN_TRACE: Mutex<Option<BoltScanTrace>> = Mutex::new(None);

fn record_scan_trace(path: &'static str, filters: &[PushedFilter]) {
    if let Ok(mut guard) = LAST_SCAN_TRACE.lock() {
        *guard = Some(BoltScanTrace {
            path,
            filters: filters.iter().map(|f| (f.sql.clone(), f.exact)).collect(),
        });
    }
}

/// The path and pushed predicates of the most recent `bolt_scan` initialization in this
/// process. Test hook; not a stable API.
#[doc(hidden)]
pub fn last_bolt_scan_trace_for_testing() -> Option<BoltScanTrace> {
    LAST_SCAN_TRACE.lock().ok().and_then(|g| g.clone())
}

impl TableFunctionCallbacks for BoltScanTableFunction {
    type BindData = BoltScanBindData;
    type GlobalState = ChunkedResult;
    type LocalState = ChunkLocalState;

    fn bind(
        &self,
        context: &Context,
        arguments: Vec<BindArgument>,
        bind_handle: BindFunctionHandle<'_>,
    ) -> Result<Self::BindData> {
        sync_options(context)?;
        let arg = arguments
            .first()
            .ok_or_else(|| Error::invalid_input("bolt_scan requires a table_name argument"))?;
        let val = arg.value.as_ref().ok_or_else(|| {
            Error::invalid_input("bolt_scan table_name argument must be a constant string")
        })?;
        let raw_name = String::from_value(val)?
            .ok_or_else(|| Error::invalid_input("bolt_scan table_name cannot be NULL"))?;

        let summary = BoltRuntime::global()
            .table_info(&raw_name)?
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "Table '{raw_name}' is not registered in Craton Bolt"
                ))
            })?;

        // Arrow allows duplicate (or case-only different) column names; DuckDB does not.
        // RFC-008: columns are declared with their original DuckDB type (`SMALLINT`,
        // `UINTEGER`, `TIMESTAMPTZ`, ...) when the field metadata records one; the bind schema
        // keeps the storage types, which pushed-down SQL refers to.
        let names = unique_column_names(summary.schema.fields());
        for (field, name) in summary.schema.fields().iter().zip(names) {
            let duckdb_ty = arrow_field_to_duckdb_type(context, field)?;
            bind_handle.add_result_column(&name, &duckdb_ty)?;
        }

        bind_handle.set_cardinality(summary.num_rows, false)?;
        bind_handle.set_order(OrderPreservation::Insertion)?;

        Ok(BoltScanBindData {
            table_name: summary.name,
            schema: summary.schema,
            pushed_filters: Mutex::new(Vec::new()),
        })
    }

    fn init_global_state(
        &self,
        bind_data: &Self::BindData,
        _context: &Context,
        column_data: InitColumnData<'_>,
    ) -> Result<(Option<Self::GlobalState>, Option<usize>)> {
        let projected_columns = (0..column_data.count()?)
            .map(|i| column_data.get_column_index(i))
            .collect::<Result<Vec<_>>>()?;

        // The table may have been replaced (`MODE 'replace'`, `bolt_register_sql`, ...)
        // between bind and init. The projection indices, pushed filters and declared
        // result types all refer to the bind-time schema, so refuse to continue.
        let current = BoltRuntime::global().table_info(&bind_data.table_name)?;
        match &current {
            Some(summary) if schemas_compatible(&summary.schema, &bind_data.schema) => {}
            _ => return Err(schema_drift_error(&bind_data.table_name)),
        }
        let (stored_schema, batches) =
            BoltRuntime::global().get_table_batches_strict(&bind_data.table_name)?;
        if !schemas_compatible(&stored_schema, &bind_data.schema) {
            return Err(schema_drift_error(&bind_data.table_name));
        }

        let filters = bind_data
            .pushed_filters
            .lock()
            .map_err(|_| Error::api_error("pushed_filters mutex poisoned"))?
            .clone();

        let projection = ScanProjection::new(&projected_columns, bind_data.schema.fields().len());
        let state = if filters.is_empty() {
            // Nothing to evaluate: stream the stored batches (only the projected columns,
            // converted to their declared types). A GPU round trip would only download the
            // same data again.
            record_scan_trace("unfiltered", &filters);
            let fields = projection.fields(&bind_data.schema);
            let out = batches
                .iter()
                .map(|b| {
                    let columns = projection.unique.iter().map(|&c| b.column(c).clone());
                    declared_stored_batch(columns.collect(), b.num_rows(), &fields)
                })
                .collect::<Result<Vec<_>>>()?;
            ChunkedResult::new(out, projection.positions)
        } else {
            filtered_scan(bind_data, &filters, batches, projection)?
        };
        let threads = state.max_threads();
        Ok((Some(state), Some(threads)))
    }

    fn init_local_state(
        &self,
        _bind_data: &Self::BindData,
        _global_state: Option<&Self::GlobalState>,
        _context: &Context,
        _column_data: InitColumnData<'_>,
    ) -> Result<Option<Self::LocalState>> {
        Ok(Some(ChunkLocalState::default()))
    }

    fn progress(
        &self,
        _bind_data: &Self::BindData,
        global_state: Option<&Self::GlobalState>,
        _context: &Context,
    ) -> Result<Option<f64>> {
        Ok(global_state.map(ChunkedResult::progress))
    }

    fn exec(
        &self,
        _bind_data: &Self::BindData,
        global_state: Option<&Self::GlobalState>,
        local_state: Option<&mut Self::LocalState>,
        context: &Context,
        mut output: DataChunkRef<'_>,
        _column_info: ExecColumnInfo<'_>,
    ) -> Result<()> {
        global_state
            .ok_or_else(|| Error::api_error("Missing bolt_scan global state"))?
            .write_next(context, local_state, &mut output)
    }
}

impl TablePartitioningCallbacks for BoltScanTableFunction {
    fn partition_data(
        &self,
        _bind_data: &Self::BindData,
        _global_state: Option<&Self::GlobalState>,
        local_state: Option<&mut Self::LocalState>,
        _context: &Context,
        _partition_data: PartitionData<'_>,
    ) -> Result<usize> {
        Ok(local_state.map_or(0, |l| l.batch_index))
    }
}

fn is_utf8_type(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View
    )
}

/// Number of distinct non-NULL values of Utf8 column `col` across `batches`, or `None` when
/// it exceeds `limit` (or the column is not a string column).
fn utf8_distinct_within(batches: &[RecordBatch], col: usize, limit: usize) -> Option<usize> {
    fn add<'a>(
        seen: &mut HashSet<&'a str>,
        values: impl Iterator<Item = Option<&'a str>>,
        limit: usize,
    ) -> bool {
        for value in values.flatten() {
            if seen.insert(value) && seen.len() > limit {
                return false;
            }
        }
        true
    }
    let mut seen: HashSet<&str> = HashSet::new();
    for batch in batches {
        let array = batch.columns().get(col)?;
        let within = if let Some(a) = array.as_string_opt::<i32>() {
            add(&mut seen, a.iter(), limit)
        } else if let Some(a) = array.as_string_opt::<i64>() {
            add(&mut seen, a.iter(), limit)
        } else if let Some(a) = array.as_string_view_opt() {
            add(&mut seen, a.iter(), limit)
        } else {
            false
        };
        if !within {
            return None;
        }
    }
    Some(seen.len())
}

/// The bind-schema columns a scan produces and where each DuckDB output slot finds its column.
#[derive(Clone, Debug, PartialEq, Eq)]
struct ScanProjection {
    /// Bind-schema indices, each once, in first-use order: the columns of every batch the
    /// scan hands to [`ChunkedResult`] (so the engine never sees duplicate output names).
    unique: Vec<usize>,
    /// Per DuckDB output slot: its position in `unique`, or `usize::MAX` for a virtual
    /// (row-id) column, which the writer emits as NULL.
    positions: Vec<usize>,
}

impl ScanProjection {
    fn new(projected_columns: &[usize], num_fields: usize) -> Self {
        let mut unique: Vec<usize> = Vec::with_capacity(projected_columns.len());
        let mut positions: Vec<usize> = Vec::with_capacity(projected_columns.len());
        for &col_idx in projected_columns {
            if col_idx >= num_fields {
                positions.push(usize::MAX);
            } else if let Some(pos) = unique.iter().position(|&u| u == col_idx) {
                positions.push(pos);
            } else {
                positions.push(unique.len());
                unique.push(col_idx);
            }
        }
        Self { unique, positions }
    }

    /// Bind-schema fields (with RFC-008 metadata) of [`Self::unique`].
    fn fields(&self, schema: &Schema) -> Vec<FieldRef> {
        self.unique
            .iter()
            .map(|&c| schema.fields()[c].clone())
            .collect()
    }
}

/// An output batch made of `columns` (positionally the bind-schema `fields`), each converted
/// to the exact Arrow type of its declared DuckDB type (RFC-008: e.g. Int32 storage back to
/// Int16 for a `SMALLINT` column, naive UTC micros to `Timestamp(us, "UTC")` for
/// `TIMESTAMPTZ`). Conversions are checked: a value that does not fit is an error, never NULL.
fn declared_batch(
    columns: Vec<ArrayRef>,
    num_rows: usize,
    fields: &[FieldRef],
) -> Result<RecordBatch> {
    if columns.len() != fields.len() {
        return Err(Error::api_error(format!(
            "bolt_scan produced {} columns but {} were expected",
            columns.len(),
            fields.len()
        )));
    }
    let mut out_fields = Vec::with_capacity(fields.len());
    let mut out_columns = Vec::with_capacity(fields.len());
    for (col, field) in columns.into_iter().zip(fields) {
        let target = original_exact_arrow_type(field);
        let converted = cast_array_to_exact_type(&col, &target).map_err(|e| {
            Error::invalid_input(format!(
                "bolt_scan column '{}' cannot be converted to its declared type: {}",
                field.name(),
                e.message
            ))
        })?;
        out_fields.push(Field::new(field.name(), target, true));
        out_columns.push(converted);
    }
    RecordBatch::try_new_with_options(
        Arc::new(Schema::new(out_fields)),
        out_columns,
        &RecordBatchOptions::new().with_row_count(Some(num_rows)),
    )
    .map_err(|e| Error::api_error(format!("Failed to build the bolt_scan batch: {e}")))
}

/// Restored (RFC-008 declared-type) copies of stored columns, so repeated unfiltered scans of
/// a narrow-typed table convert each stored column once instead of on every scan
/// (docs/issues/W3-B-003).
///
/// Keyed by the address of the stored column's `Arc` plus the target type. Each entry keeps a
/// clone of the stored column, so the address cannot be reused by another array while the
/// entry exists; an entry whose stored column is referenced by nothing else (the table was
/// dropped, replaced or rebuilt) is evicted on the next miss. Replace / append install new
/// column arrays, so a cached copy is never served for changed data.
struct RestoredColumnCache {
    entries: HashMap<(usize, DataType), (ArrayRef, ArrayRef)>,
    bytes: usize,
    hits: u64,
    misses: u64,
}

/// Upper bound on the memory held by [`RestoredColumnCache`]; beyond it columns are converted
/// per scan as before.
const RESTORED_CACHE_BUDGET_BYTES: usize = 512 << 20;

static RESTORED_CACHE: Mutex<Option<RestoredColumnCache>> = Mutex::new(None);

/// `(hits, misses, cached columns)` of the restored-column cache. Test hook; not a stable API.
#[doc(hidden)]
pub fn restored_column_cache_stats_for_testing() -> (u64, u64, usize) {
    RESTORED_CACHE
        .lock()
        .ok()
        .and_then(|g| g.as_ref().map(|c| (c.hits, c.misses, c.entries.len())))
        .unwrap_or((0, 0, 0))
}

/// Whether converting `src` to `target` copies data (a same-unit timestamp re-annotation and
/// an identical type are zero-copy).
fn restoration_copies(src: &DataType, target: &DataType) -> bool {
    match (src, target) {
        (DataType::Timestamp(a, _), DataType::Timestamp(b, _)) => a != b,
        _ => src != target,
    }
}

/// [`cast_array_to_exact_type`] for a column of a *stored* batch, through
/// [`RestoredColumnCache`].
fn restore_stored_column(col: &ArrayRef, target: &DataType) -> Result<ArrayRef> {
    if !restoration_copies(col.data_type(), target) {
        return cast_array_to_exact_type(col, target);
    }
    let key = (Arc::as_ptr(col) as *const () as usize, target.clone());
    if let Ok(mut guard) = RESTORED_CACHE.lock() {
        let cache = guard.get_or_insert_with(|| RestoredColumnCache {
            entries: HashMap::new(),
            bytes: 0,
            hits: 0,
            misses: 0,
        });
        if let Some((source, restored)) = cache.entries.get(&key)
            && Arc::ptr_eq(source, col)
        {
            let restored = restored.clone();
            cache.hits += 1;
            return Ok(restored);
        }
        cache.misses += 1;
        // Evict entries whose stored column is gone (only the cache still holds it).
        let mut freed = 0usize;
        cache.entries.retain(|_, (source, restored)| {
            let alive = Arc::strong_count(source) > 1;
            if !alive {
                freed += restored.get_array_memory_size();
            }
            alive
        });
        cache.bytes = cache.bytes.saturating_sub(freed);
    }
    // Convert outside the lock: it can take a while for a large column.
    let restored = cast_array_to_exact_type(col, target)?;
    let size = restored.get_array_memory_size();
    if let Ok(mut guard) = RESTORED_CACHE.lock()
        && let Some(cache) = guard.as_mut()
        && cache.bytes.saturating_add(size) <= RESTORED_CACHE_BUDGET_BYTES
        && !cache.entries.contains_key(&key)
    {
        cache.bytes += size;
        cache.entries.insert(key, (col.clone(), restored.clone()));
    }
    Ok(restored)
}

/// [`declared_batch`] for columns taken unchanged from stored batches: conversions are served
/// from [`RestoredColumnCache`].
fn declared_stored_batch(
    columns: Vec<ArrayRef>,
    num_rows: usize,
    fields: &[FieldRef],
) -> Result<RecordBatch> {
    if columns.len() != fields.len() {
        return Err(Error::api_error(format!(
            "bolt_scan produced {} columns but {} were expected",
            columns.len(),
            fields.len()
        )));
    }
    let mut restored = Vec::with_capacity(columns.len());
    for (col, field) in columns.iter().zip(fields) {
        let target = original_exact_arrow_type(field);
        restored.push(restore_stored_column(col, &target).map_err(|e| {
            Error::invalid_input(format!(
                "bolt_scan column '{}' cannot be converted to its declared type: {}",
                field.name(),
                e.message
            ))
        })?);
    }
    // The columns now have their declared types: `declared_batch` only builds the batch.
    declared_batch(restored, num_rows, fields)
}

/// Byte width of a fixed-width column type the engine can return cheaply, `None` otherwise.
fn fixed_width_bytes(data_type: &DataType) -> Option<usize> {
    Some(match data_type {
        DataType::Boolean | DataType::Int8 | DataType::UInt8 => 1,
        DataType::Int16 | DataType::UInt16 => 2,
        DataType::Int32 | DataType::UInt32 | DataType::Float32 | DataType::Date32 => 4,
        DataType::Int64
        | DataType::UInt64
        | DataType::Float64
        | DataType::Date64
        | DataType::Timestamp(..) => 8,
        _ => return None,
    })
}

/// The column to select when DuckDB needs only the row count of a filtered scan: the
/// narrowest fixed-width column, preferring one the filters reference. `None` when the table
/// has no fixed-width column (Utf8 and DECIMAL results are slow or unsupported on the engine,
/// docs/issues/W2-E-018).
fn row_count_column(schema: &Schema, filters: &[PushedFilter]) -> Option<usize> {
    schema
        .fields()
        .iter()
        .enumerate()
        .filter_map(|(idx, field)| {
            let referenced = filters.iter().any(|f| f.columns.contains(&idx));
            fixed_width_bytes(field.data_type()).map(|width| (width, !referenced, idx))
        })
        .min()
        .map(|(_, _, idx)| idx)
}

/// Whether an engine result has exactly the requested columns, in order (names compared ASCII
/// case-insensitively). Columns are mapped by position, so a missing, extra or reordered
/// column must not be accepted (docs/issues/W2-E-017).
fn result_shape_matches(batch: &RecordBatch, names: &[&str]) -> bool {
    batch.num_columns() == names.len()
        && batch
            .schema()
            .fields()
            .iter()
            .zip(names)
            .all(|(field, name)| field.name().eq_ignore_ascii_case(name))
}

/// Evaluate the pushed filters for a scan and return the matching rows.
///
/// The pushed SQL may run on the GPU engine or, when the runtime's crash guard routes the
/// query there (a referenced table holds NULLs, docs/issues/W2-A-001), on the embedded host
/// DuckDB. Every translation is therefore written so both dialects read it identically
/// (docs/issues/W2-O-002): floats are exponent literals (DOUBLE in both), integers are
/// integer literals, and no DECIMAL literal is ever emitted.
///
/// 1. When a dictionary-lowered Utf8 predicate is no longer safe for the current data (the
///    table grew past [`UTF8_DICTIONARY_PUSHDOWN_LIMIT`] distinct values after bind), the
///    filter runs on the host.
/// 2. Otherwise, when no projected column holds NULLs, the projected columns are selected on
///    the engine's resident table. Filtered engine plans lose the validity of projected
///    columns (NULLs come back as values, docs/issues/W2-A-001) and cannot return a Utf8
///    column with NULLs at all ("Utf8 not supported in PTX codegen"), so such scans skip
///    straight to step 3.
/// 3. Filter-take: only the referenced columns plus a row-id column are sent to the engine's
///    scratch table, and the matching row ids are gathered from the stored batches.
/// 4. If the engine rejects the filter, the embedded host DuckDB evaluates it.
///
/// Every result is converted to the declared (RFC-008 original) column types.
fn filtered_scan(
    bind: &BoltScanBindData,
    filters: &[PushedFilter],
    batches: Vec<RecordBatch>,
    projection: ScanProjection,
) -> Result<ChunkedResult> {
    let schema = &bind.schema;
    let where_clause = format!(
        " WHERE {}",
        filters
            .iter()
            .map(|f| f.sql.as_str())
            .collect::<Vec<_>>()
            .join(" AND ")
    );
    let fields = projection.fields(schema);

    let dictionaries_ok = filters
        .iter()
        .flat_map(|f| f.dictionary_columns.iter())
        .all(|&col| utf8_distinct_within(&batches, col, UTF8_DICTIONARY_PUSHDOWN_LIMIT).is_some());
    if !dictionaries_ok {
        return host_filtered_scan(bind, filters, &where_clause, projection);
    }

    // Filtered engine plans drop the validity of projected columns (docs/issues/W2-A-001):
    // when a projected column holds NULLs, only row ids are taken from the engine.
    let nulls_projected = projection
        .unique
        .iter()
        .any(|&col| batches.iter().any(|b| b.column(col).null_count() > 0));
    // Row count only: one cheap column (none: filter-take, which transfers only row ids).
    let select_columns = if projection.unique.is_empty() {
        row_count_column(schema, filters).map(|c| vec![c])
    } else {
        Some(projection.unique.clone())
    };
    if !nulls_projected && let Some(select_columns) = select_columns {
        let names: Vec<&str> = select_columns
            .iter()
            .map(|&c| schema.field(c).name().as_str())
            .collect();
        let gpu_sql = format!(
            "SELECT {} FROM {}{where_clause}",
            names
                .iter()
                .map(|n| quote_ident(n))
                .collect::<Vec<_>>()
                .join(", "),
            quote_ident(&bind.table_name)
        );
        // Batch by batch: a result the guard routed to the host keeps DuckDB's Arrow stream
        // batches (W4-A-002).
        if let Ok(out) = BoltRuntime::global().execute_sql_batches(&gpu_sql)
            && result_shape_matches(&RecordBatch::new_empty(out.schema.clone()), &names)
        {
            let declared = out
                .batches
                .iter()
                .map(|batch| {
                    let columns = if projection.unique.is_empty() {
                        Vec::new()
                    } else {
                        batch.columns().to_vec()
                    };
                    declared_batch(columns, batch.num_rows(), &fields)
                })
                .collect::<Result<Vec<_>>>();
            // A result the declared types cannot hold falls through to the stored data.
            if let Ok(declared) = declared {
                record_scan_trace("gpu", filters);
                return Ok(ChunkedResult::new(declared, projection.positions));
            }
        }
    }

    match filter_take(&batches, filters, &where_clause, &projection.unique) {
        Ok(taken) => {
            let out = taken
                .into_iter()
                .map(|b| declared_batch(b.columns().to_vec(), b.num_rows(), &fields))
                .collect::<Result<Vec<_>>>()?;
            record_scan_trace("filter-take", filters);
            Ok(ChunkedResult::new(out, projection.positions))
        }
        Err(_) => host_filtered_scan(bind, filters, &where_clause, projection),
    }
}

/// Evaluate the pushed filters on the embedded host DuckDB.
fn host_filtered_scan(
    bind: &BoltScanBindData,
    filters: &[PushedFilter],
    where_clause: &str,
    projection: ScanProjection,
) -> Result<ChunkedResult> {
    let select_list = if projection.unique.is_empty() {
        "1".to_string()
    } else {
        projection
            .unique
            .iter()
            .map(|&c| quote_ident(bind.schema.field(c).name()))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let fallback_sql = format!(
        "SELECT {select_list} FROM {}{where_clause}",
        quote_ident(&bind.table_name)
    );
    let host = BoltRuntime::global().execute_sql_host_fallback_batches(&fallback_sql)?;
    // The host engine reports its own (storage or original) types; convert positionally to
    // the declared ones, batch by batch (no concatenation, W4-A-002).
    let fields = projection.fields(&bind.schema);
    let batches = host
        .batches
        .into_iter()
        .map(|host| {
            let columns = if projection.unique.is_empty() {
                Vec::new()
            } else {
                host.columns().to_vec()
            };
            declared_batch(columns, host.num_rows(), &fields)
        })
        .collect::<Result<Vec<_>>>()?;
    record_scan_trace("host-fallback", filters);
    Ok(ChunkedResult::new(batches, projection.positions))
}

/// Run the filters on the engine against a scratch table made of the referenced columns and a
/// synthetic row-id column, then gather `output_columns` of the matching rows from `batches`
/// (in insertion order, one output batch per stored batch that has matches).
///
/// The scratch table lives only in the engine (`SCRATCH_TABLE_NAME`): it is never visible in
/// the Bolt catalog, cannot collide with a user table, and costs one worker round trip.
fn filter_take(
    batches: &[RecordBatch],
    filters: &[PushedFilter],
    where_clause: &str,
    output_columns: &[usize],
) -> Result<Vec<RecordBatch>> {
    let total_rows: usize = batches.iter().map(RecordBatch::num_rows).sum();
    let Some(first) = batches.first() else {
        return Ok(Vec::new());
    };
    if total_rows == 0 {
        return Ok(Vec::new());
    }
    let stored_schema = first.schema();

    let mut referenced: Vec<usize> = filters
        .iter()
        .flat_map(|f| f.columns.iter().copied())
        .filter(|&c| c < stored_schema.fields().len())
        .collect();
    referenced.sort_unstable();
    referenced.dedup();

    // A row-id column name that cannot collide with a referenced column.
    let mut row_id_col = "__bolt_row_id".to_string();
    while referenced.iter().any(|&c| {
        stored_schema
            .field(c)
            .name()
            .eq_ignore_ascii_case(&row_id_col)
    }) {
        row_id_col.push('_');
    }

    let mut fields: Vec<Field> = Vec::with_capacity(referenced.len() + 1);
    let mut columns: Vec<ArrayRef> = Vec::with_capacity(referenced.len() + 1);
    for &col in &referenced {
        let field = stored_schema.field(col);
        fields.push(Field::new(field.name(), field.data_type().clone(), true));
        columns.push(concat_column(batches, col)?);
    }
    fields.push(Field::new(&row_id_col, DataType::Int64, false));
    columns.push(Arc::new(Int64Array::from_iter_values(0..total_rows as i64)));
    let scratch = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).map_err(|e| {
        Error::api_error(format!("Failed to build the bolt_scan filter batch: {e}"))
    })?;

    let filter_sql = format!(
        "SELECT {} FROM {}{where_clause}",
        quote_ident(&row_id_col),
        quote_ident(SCRATCH_TABLE_NAME)
    );
    let out = BoltRuntime::global().execute_sql_with_scratch_table(scratch, &filter_sql)?;
    let mut row_ids: Vec<usize> = Vec::with_capacity(out.batch.num_rows());
    if out.batch.num_rows() > 0 {
        let ids = out
            .batch
            .columns()
            .first()
            .and_then(|c| c.as_primitive_opt::<Int64Type>())
            .ok_or_else(|| Error::api_error("bolt_scan filter returned no Int64 row ids"))?;
        for id in ids.iter() {
            let id = id
                .and_then(|v| usize::try_from(v).ok())
                .filter(|&v| v < total_rows)
                .ok_or_else(|| Error::api_error("bolt_scan filter returned an invalid row id"))?;
            row_ids.push(id);
        }
    }
    // Insertion order, whatever order the engine returned.
    if !row_ids.is_sorted() {
        row_ids.sort_unstable();
    }
    row_ids.dedup();
    gather_rows(batches, &row_ids, output_columns)
}

/// Column `col` of all `batches` as one array.
fn concat_column(batches: &[RecordBatch], col: usize) -> Result<ArrayRef> {
    if let [only] = batches {
        return Ok(only.column(col).clone());
    }
    let arrays: Vec<&dyn Array> = batches.iter().map(|b| b.column(col).as_ref()).collect();
    concat(&arrays).map_err(|e| Error::api_error(format!("Failed to concatenate column: {e}")))
}

/// `output_columns` of the rows `row_ids` (sorted, global over `batches`), one output batch
/// per stored batch with at least one selected row.
fn gather_rows(
    batches: &[RecordBatch],
    row_ids: &[usize],
    output_columns: &[usize],
) -> Result<Vec<RecordBatch>> {
    let mut out = Vec::new();
    let mut batch_start = 0usize;
    let mut pos = 0usize;
    for batch in batches {
        let batch_end = batch_start + batch.num_rows();
        let begin = pos;
        while pos < row_ids.len() && row_ids[pos] < batch_end {
            pos += 1;
        }
        if pos > begin {
            let local: UInt64Array = row_ids[begin..pos]
                .iter()
                .map(|&id| (id - batch_start) as u64)
                .collect();
            let schema = batch.schema();
            let mut fields = Vec::with_capacity(output_columns.len());
            let mut columns = Vec::with_capacity(output_columns.len());
            for &col in output_columns {
                let field = schema.field(col);
                fields.push(Field::new(field.name(), field.data_type().clone(), true));
                columns.push(take(batch.column(col).as_ref(), &local, None).map_err(|e| {
                    Error::api_error(format!("Failed to gather GPU-filtered rows: {e}"))
                })?);
            }
            out.push(
                RecordBatch::try_new_with_options(
                    Arc::new(Schema::new(fields)),
                    columns,
                    &RecordBatchOptions::new().with_row_count(Some(pos - begin)),
                )
                .map_err(|e| Error::api_error(format!("Failed to build filtered batch: {e}")))?,
            );
        }
        batch_start = batch_end;
    }
    Ok(out)
}

impl TableFilterPushdownCallbacks for BoltScanTableFunction {
    fn pushdown_filter(
        &self,
        bind_data: &Self::BindData,
        _context: &Context,
        pushdown_data: PushdownData<'_>,
    ) -> Result<()> {
        // Only accept filter pushdown when the GPU execution backend is available.
        if !gpu_backend_available()? {
            return Ok(());
        }

        let ctx = PushdownCtx::new(&pushdown_data, &bind_data.schema, &bind_data.table_name);
        let filter_count = pushdown_data.filter_count()?;
        let mut budget = PUSHDOWN_TERM_BUDGET;
        let mut candidates: Vec<(usize, PushedFilter)> = Vec::with_capacity(filter_count);
        for i in 0..filter_count {
            let expr = pushdown_data.filter(i)?;
            // An API error while walking a filter only means it is not pushed.
            if let Ok(Some((filter, cost))) = translate_filter(&expr, &ctx)
                && cost <= budget
            {
                budget -= cost;
                candidates.push((i, filter));
            }
        }
        if candidates.is_empty() {
            return Ok(());
        }

        let table = quote_ident(&bind_data.table_name);
        let plans = |preds: &[&str]| {
            let sql = format!("SELECT * FROM {table} WHERE {}", preds.join(" AND "));
            BoltRuntime::global().explain_sql(&sql).is_ok()
        };
        // Validate all candidates with a single plan first; only fall back to one plan
        // per predicate when the combined filter is rejected.
        let all: Vec<&str> = candidates.iter().map(|(_, f)| f.sql.as_str()).collect();
        let accepted: Vec<(usize, PushedFilter)> = if plans(&all) {
            candidates
        } else {
            candidates
                .into_iter()
                .filter(|(_, f)| plans(&[f.sql.as_str()]))
                .collect()
        };

        let mut pushed = Vec::with_capacity(accepted.len());
        for (i, filter) in accepted {
            if filter.exact {
                pushdown_data.accept_pushdown(i)?;
            }
            pushed.push(filter);
        }

        if !pushed.is_empty() {
            bind_data
                .pushed_filters
                .lock()
                .map_err(|_| Error::api_error("pushed_filters mutex poisoned"))?
                .extend(pushed);
        }

        Ok(())
    }
}

/// Whether the Craton Bolt GPU backend is active, for deciding filter pushdown.
///
/// `telemetry()` initializes the engine and enumerates CUDA devices, which is too costly to
/// run for every `bolt_scan` bind (and on a GPU-less host re-attempts CUDA initialization each
/// time). The answer is cached per [`BoltConfig`]: changing any `bolt_*` setting re-probes.
/// A stale answer only affects performance, never results: with a cached `true` and no
/// working engine, `bolt_scan` evaluates pushed filters on its host fallback path.
fn gpu_backend_available() -> Result<bool> {
    static CACHE: Mutex<Option<(BoltConfig, bool)>> = Mutex::new(None);
    let config = BoltRuntime::global().config()?;
    if let Ok(guard) = CACHE.lock()
        && let Some((cached_config, available)) = guard.as_ref()
        && *cached_config == config
    {
        return Ok(*available);
    }
    let available = BoltRuntime::global().telemetry()?.backend == "cuda-gpu";
    if let Ok(mut guard) = CACHE.lock() {
        *guard = Some((config, available));
    }
    Ok(available)
}

/// Same column names, storage types and RFC-008 original types, in the same order
/// (nullability is ignored). The original type matters because it is the declared type: a
/// replace that turns a `SMALLINT` into an `INTEGER` keeps the Int32 storage type.
fn schemas_compatible(current: &Schema, bound: &Schema) -> bool {
    current.fields().len() == bound.fields().len()
        && current.fields().iter().zip(bound.fields()).all(|(a, b)| {
            a.name() == b.name()
                && a.data_type() == b.data_type()
                && original_duckdb_type(a) == original_duckdb_type(b)
        })
}

fn schema_drift_error(table: &str) -> Error {
    Error::invalid_input(format!(
        "bolt_scan: table '{table}' was dropped or replaced after the query was bound; re-run the query"
    ))
}

/// Quote an identifier for the engine's SQL frontend (which accepts `""` escapes).
fn quote_ident(name: &str) -> String {
    format!("\"{}\"", name.replace('"', "\"\""))
}

/// State shared while translating the filters of one `bolt_scan` bind.
struct PushdownCtx<'a, 'b> {
    pushdown_data: &'a PushdownData<'b>,
    schema: &'a SchemaRef,
    table: &'a str,
    /// Stored batches, fetched on first use (only Utf8 dictionary checks need them).
    batches: OnceCell<Option<Vec<RecordBatch>>>,
    /// Column -> distinct non-NULL value count, `None` when above the limit.
    distinct: RefCell<HashMap<usize, Option<usize>>>,
    /// Terms of the filter being translated.
    cost: Cell<usize>,
    columns: RefCell<Vec<usize>>,
    dictionary_columns: RefCell<Vec<usize>>,
}

impl<'a, 'b> PushdownCtx<'a, 'b> {
    fn new(pushdown_data: &'a PushdownData<'b>, schema: &'a SchemaRef, table: &'a str) -> Self {
        Self {
            pushdown_data,
            schema,
            table,
            batches: OnceCell::new(),
            distinct: RefCell::new(HashMap::new()),
            cost: Cell::new(0),
            columns: RefCell::new(Vec::new()),
            dictionary_columns: RefCell::new(Vec::new()),
        }
    }

    fn begin(&self) {
        self.cost.set(0);
        self.columns.borrow_mut().clear();
        self.dictionary_columns.borrow_mut().clear();
    }

    fn finish(&self, sql: String, exact: bool) -> (PushedFilter, usize) {
        let mut columns = self.columns.borrow().clone();
        columns.sort_unstable();
        columns.dedup();
        let mut dictionary_columns = self.dictionary_columns.borrow().clone();
        dictionary_columns.sort_unstable();
        dictionary_columns.dedup();
        (
            PushedFilter {
                sql,
                columns,
                dictionary_columns,
                exact,
            },
            self.cost.get(),
        )
    }

    fn add_cost(&self, terms: usize) {
        self.cost.set(self.cost.get().saturating_add(terms));
    }

    /// Bind-schema index of a column reference (`None` for virtual columns).
    fn column_index(&self, expr: &Expression<'_>) -> Result<Option<usize>> {
        if expr.expression_type()? != ExpressionType::BoundColumnRef {
            return Ok(None);
        }
        let col_idx = self
            .pushdown_data
            .get_column_index(expr.reference_index()?)?;
        Ok((col_idx < self.schema.fields().len()).then_some(col_idx))
    }

    /// The Utf8 column `expr` refers to (directly or through a VARCHAR -> VARCHAR cast).
    fn utf8_column(&self, expr: &Expression<'_>) -> Result<Option<usize>> {
        match expr.expression_type()? {
            ExpressionType::BoundColumnRef => Ok(self
                .column_index(expr)?
                .filter(|&c| is_utf8_type(self.schema.field(c).data_type()))),
            ExpressionType::OperatorCast if expr.child_count()? == 1 => {
                let inner = expr.child(0)?;
                if type_class(&inner.return_type()?)? == TypeClass::Varchar
                    && type_class(&expr.return_type()?)? == TypeClass::Varchar
                {
                    self.utf8_column(&inner)
                } else {
                    Ok(None)
                }
            }
            _ => Ok(None),
        }
    }

    /// Whether a dictionary-lowered predicate over Utf8 column `col` is safe to push (see
    /// [`UTF8_DICTIONARY_PUSHDOWN_LIMIT`]). Records the requirement and its cost.
    fn small_dictionary(&self, col: usize) -> bool {
        let cached = self.distinct.borrow().get(&col).copied();
        let distinct = cached.unwrap_or_else(|| {
            let batches = self.batches.get_or_init(|| {
                BoltRuntime::global()
                    .get_table_batches_strict(self.table)
                    .ok()
                    .map(|(_, b)| b)
            });
            let distinct = batches
                .as_deref()
                .and_then(|b| utf8_distinct_within(b, col, UTF8_DICTIONARY_PUSHDOWN_LIMIT));
            self.distinct.borrow_mut().insert(col, distinct);
            distinct
        });
        match distinct {
            Some(n) => {
                self.add_cost(n.max(1));
                self.dictionary_columns.borrow_mut().push(col);
                true
            }
            None => false,
        }
    }
}

/// Translate one DuckDB filter: exactly when possible (the filter is then accepted), else as a
/// pre-filter that keeps a superset of the rows (DuckDB re-applies the filter on top).
/// Returns the filter and its term count.
fn translate_filter(
    expr: &Expression<'_>,
    ctx: &PushdownCtx<'_, '_>,
) -> Result<Option<(PushedFilter, usize)>> {
    ctx.begin();
    if let Some(sql) = translate_predicate(expr, ctx)? {
        return Ok(Some(ctx.finish(sql, true)));
    }
    ctx.begin();
    if let Some(sql) = translate_superset(expr, ctx)? {
        return Ok(Some(ctx.finish(sql, false)));
    }
    Ok(None)
}

/// Exact translation of a boolean-valued filter expression; a bare (or cast) BOOLEAN column
/// is guarded against NULL.
fn translate_predicate(expr: &Expression<'_>, ctx: &PushdownCtx<'_, '_>) -> Result<Option<String>> {
    let Some(sql) = translate(expr, ctx)? else {
        return Ok(None);
    };
    if matches!(
        expr.expression_type()?,
        ExpressionType::BoundColumnRef | ExpressionType::OperatorCast
    ) {
        return guard_nullable_columns(expr, ctx, sql, false);
    }
    Ok(Some(sql))
}

/// A predicate that is TRUE (on the engine) for every row where `expr` is TRUE (in DuckDB).
///
/// - `BETWEEN`: DuckDB binds both `x BETWEEN a AND b` and `x > a AND x < b` to a BETWEEN node
///   whose bound inclusivity the C API does not report
///   (docs/issues/W1-A-001-between-inclusivity.md), so the inclusive form is pushed.
/// - `AND`: the conjunction of the children that can be translated.
/// - `OR`: the disjunction, when every child can be translated.
///
/// Never used below `NOT`, where a superset would turn into a subset.
fn translate_superset(expr: &Expression<'_>, ctx: &PushdownCtx<'_, '_>) -> Result<Option<String>> {
    match expr.expression_type()? {
        ExpressionType::CompareBetween => {
            if expr.child_count()? != 3 {
                return Ok(None);
            }
            let (input, lower, upper) = (expr.child(0)?, expr.child(1)?, expr.child(2)?);
            let Some(lo) = translate_comparison(
                expr,
                ExpressionType::CompareGreaterThanOrEqualTo,
                &input,
                &lower,
                ctx,
            )?
            else {
                return Ok(None);
            };
            let Some(hi) = translate_comparison(
                expr,
                ExpressionType::CompareLessThanOrEqualTo,
                &input,
                &upper,
                ctx,
            )?
            else {
                return Ok(None);
            };
            Ok(Some(format!("({lo} AND {hi})")))
        }
        op @ (ExpressionType::ConjunctionAnd | ExpressionType::ConjunctionOr) => {
            let count = expr.child_count()?;
            let mut parts = Vec::with_capacity(count);
            for idx in 0..count {
                let child = expr.child(idx)?;
                let part = match translate_predicate(&child, ctx)? {
                    Some(sql) => Some(sql),
                    None => translate_superset(&child, ctx)?,
                };
                match part {
                    Some(sql) => parts.push(sql),
                    // Dropping a conjunct only widens an AND; an OR needs every branch.
                    None if op == ExpressionType::ConjunctionAnd => {}
                    None => return Ok(None),
                }
            }
            if parts.is_empty() {
                return Ok(None);
            }
            let glue = if op == ExpressionType::ConjunctionAnd {
                " AND "
            } else {
                " OR "
            };
            Ok(Some(format!("({})", parts.join(glue))))
        }
        _ => translate_predicate(expr, ctx),
    }
}

fn collect_nullable_columns(
    expr: &Expression<'_>,
    ctx: &PushdownCtx<'_, '_>,
    out: &mut Vec<usize>,
) -> Result<()> {
    match expr.expression_type()? {
        ExpressionType::BoundColumnRef => {
            if let Some(col_idx) = ctx.column_index(expr)?
                && ctx.schema.field(col_idx).is_nullable()
                && !out.contains(&col_idx)
            {
                out.push(col_idx);
            }
        }
        ExpressionType::OperatorIsNull | ExpressionType::OperatorIsNotNull => {}
        _ => {
            let count = expr.child_count()?;
            for idx in 0..count {
                collect_nullable_columns(&expr.child(idx)?, ctx, out)?;
            }
        }
    }
    Ok(())
}

/// Prefix `sql` with `IS NOT NULL` tests for the nullable columns of `expr`, so rows where any
/// of them is NULL never pass (the engine evaluates some predicates on the value slots of NULL
/// rows).
///
/// The engine cannot evaluate `IS [NOT] NULL` on Utf8 columns ("Utf8 not supported in PTX
/// codegen"); their guard is `col LIKE '%'`, which matches exactly the non-NULL values but is
/// lowered through the string dictionary. `utf8_null_safe` skips the Utf8 guard for positive
/// predicates (`=`, ordering against a literal, `LIKE`, `IN`), which never match the NULL
/// dictionary slot. Returns `None` when a needed guard cannot be pushed.
fn guard_nullable_columns(
    expr: &Expression<'_>,
    ctx: &PushdownCtx<'_, '_>,
    sql: String,
    utf8_null_safe: bool,
) -> Result<Option<String>> {
    let mut nullable_cols = Vec::new();
    collect_nullable_columns(expr, ctx, &mut nullable_cols)?;
    let mut parts = Vec::with_capacity(nullable_cols.len() + 1);
    for col in nullable_cols {
        let field = ctx.schema.field(col);
        let quoted = quote_ident(field.name());
        if is_utf8_type(field.data_type()) {
            if utf8_null_safe {
                continue;
            }
            if !ctx.small_dictionary(col) {
                return Ok(None);
            }
            parts.push(format!("({quoted} LIKE '%')"));
        } else {
            ctx.add_cost(1);
            parts.push(format!("({quoted} IS NOT NULL)"));
        }
    }
    if parts.is_empty() {
        return Ok(Some(sql));
    }
    parts.push(sql);
    Ok(Some(format!("({})", parts.join(" AND "))))
}

/// Coarse classification of a DuckDB logical type, used to decide which expressions can be
/// evaluated by the Craton Bolt engine with exactly DuckDB's semantics.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum TypeClass {
    Int { signed: bool, bits: u8 },
    Float32,
    Float64,
    Decimal { width: u8, scale: u8 },
    Varchar,
    Boolean,
    Date,
    Timestamp,
    TimestampNs,
    TimestampTz,
    Other,
}

impl TypeClass {
    fn is_float(self) -> bool {
        matches!(self, TypeClass::Float32 | TypeClass::Float64)
    }
}

fn type_class(ty: &LogicalType) -> Result<TypeClass> {
    let int = |signed, bits| TypeClass::Int { signed, bits };
    Ok(match ty.id()? {
        LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_TINYINT => int(true, 8),
        LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_SMALLINT => int(true, 16),
        LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_INTEGER => int(true, 32),
        LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_BIGINT => int(true, 64),
        LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_HUGEINT => int(true, 128),
        LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_UTINYINT => int(false, 8),
        LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_USMALLINT => int(false, 16),
        LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_UINTEGER => int(false, 32),
        LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_UBIGINT => int(false, 64),
        LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_FLOAT => TypeClass::Float32,
        LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_DOUBLE => TypeClass::Float64,
        LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_DECIMAL => {
            let sig = DecimalSignature::from_logical_type(ty)?;
            TypeClass::Decimal {
                width: sig.width,
                scale: sig.scale,
            }
        }
        LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_VARCHAR => TypeClass::Varchar,
        LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_BOOLEAN => TypeClass::Boolean,
        LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_DATE => TypeClass::Date,
        LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_TIMESTAMP => TypeClass::Timestamp,
        LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_TIMESTAMP_NS => TypeClass::TimestampNs,
        LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_TIMESTAMP_TZ => TypeClass::TimestampTz,
        _ => TypeClass::Other,
    })
}

/// Number of decimal digits needed to represent every value of an integer type.
fn integer_decimal_digits(signed: bool, bits: u8) -> u8 {
    match (bits, signed) {
        (8, _) => 3,
        (16, _) => 5,
        (32, _) => 10,
        (64, true) => 19,
        (64, false) => 20,
        _ => 39,
    }
}

/// Whether a `CAST(from AS to)` can be dropped without changing the result of a comparison:
/// every value of `from` maps to a distinct, exactly representable and order-preserving value
/// of `to`. Everything else (narrowing, float -> int, numeric -> VARCHAR, time-zone dependent
/// casts such as TIMESTAMP -> TIMESTAMPTZ, ...) must be evaluated by DuckDB.
fn is_lossless_widening(from: TypeClass, to: TypeClass) -> bool {
    use TypeClass::*;
    match (from, to) {
        (Other, _) | (_, Other) => false,
        (a, b) if a == b => true,
        (
            Int {
                signed: s1,
                bits: b1,
            },
            Int {
                signed: s2,
                bits: b2,
            },
        ) => {
            if s1 == s2 {
                b2 >= b1
            } else {
                // unsigned -> strictly wider signed; signed -> unsigned is never lossless.
                !s1 && s2 && b2 > b1
            }
        }
        // Integers up to 32 bits (signed or unsigned) are exact in an f64 mantissa,
        // up to 16 bits in an f32 mantissa.
        (Int { bits, .. }, Float64) => bits <= 32,
        (Int { bits, .. }, Float32) => bits <= 16,
        (Float32, Float64) => true,
        (Int { signed, bits }, Decimal { width, scale }) => {
            width.saturating_sub(scale) >= integer_decimal_digits(signed, bits)
        }
        (
            Decimal {
                width: w1,
                scale: s1,
            },
            Decimal {
                width: w2,
                scale: s2,
            },
        ) => s2 >= s1 && w2.saturating_sub(s2) >= w1.saturating_sub(s1),
        _ => false,
    }
}

fn is_constant(expr: &Expression<'_>) -> Result<bool> {
    Ok(expr.expression_type()? == ExpressionType::ValueConstant)
}

fn is_float_expr(expr: &Expression<'_>) -> Result<bool> {
    Ok(type_class(&expr.return_type()?)?.is_float())
}

/// `a op b` == `b flip(op) a`.
fn flip_comparison(op: ExpressionType) -> ExpressionType {
    match op {
        ExpressionType::CompareLessThan => ExpressionType::CompareGreaterThan,
        ExpressionType::CompareGreaterThan => ExpressionType::CompareLessThan,
        ExpressionType::CompareLessThanOrEqualTo => ExpressionType::CompareGreaterThanOrEqualTo,
        ExpressionType::CompareGreaterThanOrEqualTo => ExpressionType::CompareLessThanOrEqualTo,
        other => other,
    }
}

/// `TIMESTAMPTZ` constants are rendered as UTC wall-clock literals. That is only meaningful
/// when they are compared directly against a column whose stored values are UTC instants:
/// `Timestamp(us, Some("UTC"))`, or the naive `Timestamp(us, None)` storage RFC-008 records
/// as `TIMESTAMPTZ`. Both the engine and the host fallback (whose mirror uses the storage
/// types) compare the naive UTC micros with the naive UTC literal. Against anything else (a
/// naive `TIMESTAMP` column, an expression) the comparison is declined so DuckDB evaluates it
/// with the session time zone.
fn tz_constants_safe(children: &[&Expression<'_>], ctx: &PushdownCtx<'_, '_>) -> Result<bool> {
    let mut has_tz_constant = false;
    for child in children {
        if is_constant(child)? && type_class(&child.return_type()?)? == TypeClass::TimestampTz {
            has_tz_constant = true;
        }
    }
    if !has_tz_constant {
        return Ok(true);
    }
    for child in children {
        if is_constant(child)? {
            continue;
        }
        let Some(col_idx) = ctx.column_index(child)? else {
            return Ok(false);
        };
        let field = ctx.schema.field(col_idx);
        let utc_column = match field.data_type() {
            DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, Some(tz)) => {
                matches!(tz.as_ref(), "UTC" | "utc" | "Etc/UTC" | "+00:00" | "Z")
            }
            DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, None) => {
                original_duckdb_type(field) == Some(OriginalDuckDbType::TimestampTz)
            }
            _ => false,
        };
        if !utc_column {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Whether `expr` evaluates to NULL whenever any of the nullable columns it references (as
/// collected by [`collect_nullable_columns`]) is NULL. For such expressions
/// `NOT expr` can only be TRUE when all of those columns are non-NULL, so the `IS NOT NULL`
/// guard added around a pushed `NOT` is exact. `AND` / `OR` / `IS [NOT] NULL` are not strict
/// (`NULL AND FALSE` is FALSE), so `NOT` over them is not pushed.
fn is_null_strict(expr: &Expression<'_>) -> Result<bool> {
    Ok(match expr.expression_type()? {
        ExpressionType::BoundColumnRef
        | ExpressionType::ValueConstant
        | ExpressionType::CompareEqual
        | ExpressionType::CompareNotEqual
        | ExpressionType::CompareLessThan
        | ExpressionType::CompareGreaterThan
        | ExpressionType::CompareLessThanOrEqualTo
        | ExpressionType::CompareGreaterThanOrEqualTo
        | ExpressionType::CompareIn
        | ExpressionType::CompareNotIn
        | ExpressionType::BoundFunction => {
            // Every pushed comparison / IN / LIKE / arithmetic is NULL-propagating in each
            // operand; IN lists and LIKE patterns are restricted to non-NULL constants.
            let mut strict = true;
            for idx in 0..expr.child_count()? {
                strict &= is_null_strict(&expr.child(idx)?)?;
            }
            strict
        }
        ExpressionType::OperatorCast | ExpressionType::OperatorNot => {
            expr.child_count()? == 1 && is_null_strict(&expr.child(0)?)?
        }
        _ => false,
    })
}

/// Whether `expr` may be the operand of a pushed `IS [NOT] NULL`: a column or constant,
/// optionally under casts, or pushed DOUBLE arithmetic over such operands. Their translation
/// adds no NULL guards, so the engine sees the same NULLs as DuckDB. A predicate operand
/// (comparison, `IN`, `LIKE`, `AND` / `OR` / `NOT`) is translated with `IS NOT NULL` guards
/// that turn its NULL into FALSE, so `(a = 1) IS NULL` would never be TRUE
/// (docs/issues/W2-E-002).
fn is_unguarded_value(expr: &Expression<'_>) -> Result<bool> {
    Ok(match expr.expression_type()? {
        ExpressionType::BoundColumnRef | ExpressionType::ValueConstant => true,
        ExpressionType::OperatorCast => {
            expr.child_count()? == 1 && is_unguarded_value(&expr.child(0)?)?
        }
        ExpressionType::BoundFunction => {
            matches!(expr.function_name()?.as_str(), "+" | "-" | "*")
                && expr.child_count()? == 2
                && is_unguarded_value(&expr.child(0)?)?
                && is_unguarded_value(&expr.child(1)?)?
        }
        _ => false,
    })
}

fn comparison_sql_op(op: ExpressionType) -> Option<&'static str> {
    Some(match op {
        ExpressionType::CompareEqual => "=",
        ExpressionType::CompareNotEqual => "!=",
        ExpressionType::CompareLessThan => "<",
        ExpressionType::CompareGreaterThan => ">",
        ExpressionType::CompareLessThanOrEqualTo => "<=",
        ExpressionType::CompareGreaterThanOrEqualTo => ">=",
        _ => return None,
    })
}

/// Exact translation of `expr` into Craton Bolt SQL, or `None` when the engine might not
/// evaluate it exactly like DuckDB.
fn translate(expr: &Expression<'_>, ctx: &PushdownCtx<'_, '_>) -> Result<Option<String>> {
    match expr.expression_type()? {
        ExpressionType::BoundColumnRef => {
            let Some(col_idx) = ctx.column_index(expr)? else {
                return Ok(None);
            };
            let field = ctx.schema.field(col_idx);
            // The engine cannot compare Decimal128 columns with numeric literals and
            // evaluates `IS NULL` on them incorrectly (docs/issues/W1-A-002-decimal-gpu-predicates.md), so any
            // predicate touching a DECIMAL column stays with DuckDB.
            if matches!(
                field.data_type(),
                DataType::Decimal128(..) | DataType::Decimal256(..)
            ) {
                return Ok(None);
            }
            ctx.columns.borrow_mut().push(col_idx);
            Ok(Some(quote_ident(field.name())))
        }
        ExpressionType::ValueConstant => {
            let val = expr.get_constant_value()?;
            value_to_bolt_sql_literal(&val)
        }
        ExpressionType::OperatorCast => {
            if expr.child_count()? != 1 {
                return Ok(None);
            }
            let inner = expr.child(0)?;
            // Only drop casts that cannot change the outcome of the enclosing predicate;
            // e.g. `CAST(x AS INTEGER) = 3` on a DOUBLE column must not become `x = 3`.
            let from = type_class(&inner.return_type()?)?;
            let to = type_class(&expr.return_type()?)?;
            if !is_lossless_widening(from, to) {
                return Ok(None);
            }
            translate(&inner, ctx)
        }
        ExpressionType::OperatorNot => {
            if expr.child_count()? != 1 {
                return Ok(None);
            }
            let child_expr = expr.child(0)?;
            let child_type = child_expr.expression_type()?;
            let is_null_test = matches!(
                child_type,
                ExpressionType::OperatorIsNull | ExpressionType::OperatorIsNotNull
            );
            if is_null_test {
                // A Utf8 `IS NOT NULL` is pushed as `LIKE '%'`, whose negation is not
                // `IS NULL` on every engine path.
                if child_expr.child_count()? != 1
                    || type_class(&child_expr.child(0)?.return_type()?)? == TypeClass::Varchar
                {
                    return Ok(None);
                }
            } else if !is_null_strict(&child_expr)? {
                return Ok(None);
            }
            let Some(child) = translate(&child_expr, ctx)? else {
                return Ok(None);
            };
            let raw = format!("(NOT ({child}))");
            guard_nullable_columns(expr, ctx, raw, false)
        }
        ExpressionType::OperatorIsNull => {
            if expr.child_count()? != 1 {
                return Ok(None);
            }
            let child_expr = expr.child(0)?;
            // The engine cannot evaluate IS NULL on Utf8 and has no exact substitute.
            if type_class(&child_expr.return_type()?)? == TypeClass::Varchar
                || !is_unguarded_value(&child_expr)?
            {
                return Ok(None);
            }
            let Some(child) = translate(&child_expr, ctx)? else {
                return Ok(None);
            };
            ctx.add_cost(1);
            Ok(Some(format!("({child} IS NULL)")))
        }
        ExpressionType::OperatorIsNotNull => {
            if expr.child_count()? != 1 {
                return Ok(None);
            }
            let child_expr = expr.child(0)?;
            if type_class(&child_expr.return_type()?)? == TypeClass::Varchar {
                // `col LIKE '%'` is TRUE exactly for the non-NULL values.
                let Some(col) = ctx.utf8_column(&child_expr)? else {
                    return Ok(None);
                };
                if !ctx.small_dictionary(col) {
                    return Ok(None);
                }
                let Some(child) = translate(&child_expr, ctx)? else {
                    return Ok(None);
                };
                return Ok(Some(format!("({child} LIKE '%')")));
            }
            if !is_unguarded_value(&child_expr)? {
                return Ok(None);
            }
            let Some(child) = translate(&child_expr, ctx)? else {
                return Ok(None);
            };
            ctx.add_cost(1);
            Ok(Some(format!("({child} IS NOT NULL)")))
        }
        op @ (ExpressionType::CompareEqual
        | ExpressionType::CompareNotEqual
        | ExpressionType::CompareLessThan
        | ExpressionType::CompareGreaterThan
        | ExpressionType::CompareLessThanOrEqualTo
        | ExpressionType::CompareGreaterThanOrEqualTo) => {
            if expr.child_count()? != 2 {
                return Ok(None);
            }
            translate_comparison(expr, op, &expr.child(0)?, &expr.child(1)?, ctx)
        }
        // DuckDB also binds range conjunctions such as `x > a AND x < b` to this node, with
        // exclusive bounds, but the C API reports every BETWEEN as `__between` without its
        // inclusivity. Translating it as `>= .. AND <= ..` returned extra rows, so BETWEEN is
        // never accepted; `translate_superset` pushes the inclusive form as a pre-filter
        // (see docs/issues/W1-A-001-between-inclusivity.md).
        ExpressionType::CompareBetween => Ok(None),
        op @ (ExpressionType::CompareIn | ExpressionType::CompareNotIn) => {
            translate_in_list(expr, op, ctx)
        }
        op @ (ExpressionType::ConjunctionAnd | ExpressionType::ConjunctionOr) => {
            let count = expr.child_count()?;
            if count == 0 {
                return Ok(None);
            }
            let mut parts = Vec::with_capacity(count);
            for idx in 0..count {
                let child = expr.child(idx)?;
                let Some(part) = translate_predicate(&child, ctx)? else {
                    return Ok(None);
                };
                parts.push(part);
            }
            let glue = if op == ExpressionType::ConjunctionAnd {
                " AND "
            } else {
                " OR "
            };
            Ok(Some(format!("({})", parts.join(glue))))
        }
        ExpressionType::BoundFunction => translate_bound_function(expr, ctx),
        _ => Ok(None),
    }
}

/// `left op right`; `guard_expr` is the expression whose nullable columns guard the result
/// (the comparison itself, or the enclosing BETWEEN).
fn translate_comparison(
    guard_expr: &Expression<'_>,
    op: ExpressionType,
    left: &Expression<'_>,
    right: &Expression<'_>,
    ctx: &PushdownCtx<'_, '_>,
) -> Result<Option<String>> {
    let (mut left, mut right) = (left, right);
    let mut op = op;
    if is_constant(left)? && !is_constant(right)? {
        std::mem::swap(&mut left, &mut right);
        op = flip_comparison(op);
    }
    if is_constant(right)?
        && matches!(
            type_class(&right.return_type()?)?,
            TypeClass::Decimal { .. }
        )
    {
        return translate_integer_vs_decimal(guard_expr, op, left, right, ctx);
    }
    let utf8 = type_class(&left.return_type()?)? == TypeClass::Varchar
        || type_class(&right.return_type()?)? == TypeClass::Varchar;
    let mut utf8_null_safe = false;
    if utf8 {
        // Only `column op 'literal'`: the engine compares two Utf8 columns by dictionary
        // index / rank, which is not NULL-safe for `=`.
        if !is_constant(right)? {
            return Ok(None);
        }
        let Some(col) = ctx.utf8_column(left)? else {
            return Ok(None);
        };
        match op {
            // Folded to one dictionary-index equality; never matches the NULL slot.
            ExpressionType::CompareEqual => utf8_null_safe = true,
            // `idx <> k` is TRUE for the NULL slot: needs the guard.
            ExpressionType::CompareNotEqual => {}
            // Ordering is lowered to an OR over the matching dictionary entries.
            _ => {
                if !ctx.small_dictionary(col) {
                    return Ok(None);
                }
                utf8_null_safe = true;
            }
        }
    }
    let float = is_float_expr(left)? || is_float_expr(right)?;
    // DuckDB treats NaN = NaN as TRUE; IEEE comparisons on the GPU do not. With a
    // finite constant on one side the difference can be patched up below.
    if float && !is_constant(right)? {
        return Ok(None);
    }
    if !tz_constants_safe(&[left, right], ctx)? {
        return Ok(None);
    }
    let Some(lhs) = translate(left, ctx)? else {
        return Ok(None);
    };
    let Some(rhs) = translate(right, ctx)? else {
        return Ok(None);
    };
    let Some(sql_op) = comparison_sql_op(op) else {
        return Ok(None);
    };
    let raw = if float
        && matches!(
            op,
            ExpressionType::CompareGreaterThan
                | ExpressionType::CompareGreaterThanOrEqualTo
                | ExpressionType::CompareNotEqual
        ) {
        // DuckDB orders NaN above every other value (so `NaN > c`, `NaN >= c` and
        // `NaN != c` are TRUE); the engine's IEEE comparisons return FALSE for NaN.
        // `NOT (e = e)` is TRUE exactly for NaN.
        format!("(({lhs} {sql_op} {rhs}) OR (NOT ({lhs} = {lhs})))")
    } else {
        format!("({lhs} {sql_op} {rhs})")
    };
    ctx.add_cost(1);
    guard_nullable_columns(guard_expr, ctx, raw, utf8_null_safe)
}

/// `CAST(<integer> AS DECIMAL(w, s)) op <decimal constant>`, which is how DuckDB binds an
/// integer compared with a decimal literal, pushed as an exact integer comparison. The engine
/// has no DECIMAL literal (it reads `2.5` as a DOUBLE and would compare in binary floating
/// point, docs/issues/W2-E-003), while the host fallback would read it as an exact DECIMAL:
/// integer bounds mean the same in both. For an integer `x` and a constant `c` with a
/// fractional part, `x < c` is `x < ceil(c)`, `x <= c` is `x <= floor(c)`, `x > c` is
/// `x > floor(c)` and `x >= c` is `x >= ceil(c)`; `=` / `!=` are left to DuckDB.
fn translate_integer_vs_decimal(
    guard_expr: &Expression<'_>,
    op: ExpressionType,
    left: &Expression<'_>,
    right: &Expression<'_>,
    ctx: &PushdownCtx<'_, '_>,
) -> Result<Option<String>> {
    if left.expression_type()? != ExpressionType::OperatorCast
        || left.child_count()? != 1
        || !matches!(
            type_class(&left.child(0)?.return_type()?)?,
            TypeClass::Int { .. }
        )
    {
        return Ok(None);
    }
    let Some(dec) = DecimalValue::from_value(&right.get_constant_value()?)? else {
        return Ok(None);
    };
    let Some(unit) = 10i128.checked_pow(u32::from(dec.scale)) else {
        return Ok(None);
    };
    let floor = dec.value.div_euclid(unit);
    let fractional = dec.value.rem_euclid(unit) != 0;
    let ceil = floor + i128::from(fractional);
    let (op, bound) = match op {
        ExpressionType::CompareEqual | ExpressionType::CompareNotEqual if fractional => {
            return Ok(None);
        }
        ExpressionType::CompareEqual | ExpressionType::CompareNotEqual => (op, floor),
        ExpressionType::CompareLessThan | ExpressionType::CompareGreaterThanOrEqualTo => (op, ceil),
        ExpressionType::CompareLessThanOrEqualTo | ExpressionType::CompareGreaterThan => {
            (op, floor)
        }
        _ => return Ok(None),
    };
    let (Ok(bound), Some(sql_op)) = (i64::try_from(bound), comparison_sql_op(op)) else {
        return Ok(None);
    };
    // `translate` strips the cast only when it is a lossless widening.
    let Some(lhs) = translate(left, ctx)? else {
        return Ok(None);
    };
    ctx.add_cost(1);
    guard_nullable_columns(guard_expr, ctx, format!("({lhs} {sql_op} {bound})"), false)
}

/// `target [NOT] IN (constants...)`.
fn translate_in_list(
    expr: &Expression<'_>,
    op: ExpressionType,
    ctx: &PushdownCtx<'_, '_>,
) -> Result<Option<String>> {
    let count = expr.child_count()?;
    if count < 2 || count - 1 > MAX_IN_LIST_ITEMS {
        return Ok(None);
    }
    let target_expr = expr.child(0)?;
    let mut item_exprs = Vec::with_capacity(count - 1);
    for idx in 1..count {
        let item = expr.child(idx)?;
        // Only constant lists: a NULL-able column in the list would make the
        // `IS NOT NULL` guard wrong (`3 IN (NULL, 3)` is TRUE).
        if !is_constant(&item)? {
            return Ok(None);
        }
        item_exprs.push(item);
    }
    let mut all: Vec<&Expression<'_>> = vec![&target_expr];
    all.extend(item_exprs.iter());
    if !tz_constants_safe(&all, ctx)? {
        return Ok(None);
    }
    let utf8 = type_class(&target_expr.return_type()?)? == TypeClass::Varchar;
    if utf8 && ctx.utf8_column(&target_expr)?.is_none() {
        return Ok(None);
    }
    let Some(target) = translate(&target_expr, ctx)? else {
        return Ok(None);
    };
    let mut items = Vec::with_capacity(item_exprs.len());
    for item in &item_exprs {
        // NULL constants are not translated, so `x NOT IN (1, NULL)` (never TRUE)
        // stays with DuckDB.
        let Some(item) = translate(item, ctx)? else {
            return Ok(None);
        };
        items.push(item);
    }
    ctx.add_cost(items.len());
    let membership = if utf8 {
        // The engine cannot run IN on Utf8 ("__idx_<col> not found in evaluator env"); an OR
        // of dictionary-index equalities is equivalent.
        let terms: Vec<String> = items.iter().map(|i| format!("({target} = {i})")).collect();
        format!("({})", terms.join(" OR "))
    } else {
        format!("({target} IN ({}))", items.join(", "))
    };
    let raw = if op == ExpressionType::CompareIn {
        membership
    } else {
        // Spelled as NOT (.. IN ..) so a NaN target is TRUE as in DuckDB.
        format!("(NOT {membership})")
    };
    // `IN` never matches a NULL Utf8 value; `NOT IN` needs the guard.
    guard_nullable_columns(expr, ctx, raw, op == ExpressionType::CompareIn)
}

/// RFC-003: translate a subset of DuckDB scalar functions (arithmetic and LIKE-family
/// string predicates) into Craton Bolt SQL. Anything with uncertain semantics returns
/// `Ok(None)` so DuckDB keeps evaluating the filter itself.
fn translate_bound_function(
    expr: &Expression<'_>,
    ctx: &PushdownCtx<'_, '_>,
) -> Result<Option<String>> {
    let name = expr.function_name()?;
    let count = expr.child_count()?;
    match (name.as_str(), count) {
        ("+" | "-" | "*", 2) => {
            // Only DOUBLE arithmetic: DuckDB raises an Out of Range error on integer
            // overflow while the GPU wraps silently, DECIMAL promotion rules differ, and
            // FLOAT arithmetic may be evaluated at a different precision.
            if type_class(&expr.return_type()?)? != TypeClass::Float64 {
                return Ok(None);
            }
            let Some(lhs) = translate(&expr.child(0)?, ctx)? else {
                return Ok(None);
            };
            let Some(rhs) = translate(&expr.child(1)?, ctx)? else {
                return Ok(None);
            };
            ctx.add_cost(1);
            Ok(Some(format!("({lhs} {name} {rhs})")))
        }
        ("~~" | "!~~", 2) => {
            let Some(pattern) = constant_like_text(&expr.child(1)?)? else {
                return Ok(None);
            };
            if pattern.contains('\\') {
                return Ok(None);
            }
            like_predicate(expr, ctx, &pattern, name == "!~~")
        }
        ("prefix" | "suffix" | "contains", 2) => {
            let Some(text) = constant_like_text(&expr.child(1)?)? else {
                return Ok(None);
            };
            if text.contains(['%', '_', '\\']) {
                return Ok(None);
            }
            let pattern = match name.as_str() {
                "prefix" => format!("{text}%"),
                "suffix" => format!("%{text}"),
                _ => format!("%{text}%"),
            };
            like_predicate(expr, ctx, &pattern, false)
        }
        _ => Ok(None),
    }
}

/// Constant, non-NULL VARCHAR child text, if the expression is one.
fn constant_like_text(expr: &Expression<'_>) -> Result<Option<String>> {
    if expr.expression_type()? != ExpressionType::ValueConstant {
        return Ok(None);
    }
    let val = expr.get_constant_value()?;
    if val.is_null()?
        || val.logical_type()?.id()? != LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_VARCHAR
    {
        return Ok(None);
    }
    String::from_value(&val)
}

fn like_predicate(
    expr: &Expression<'_>,
    ctx: &PushdownCtx<'_, '_>,
    pattern: &str,
    negated: bool,
) -> Result<Option<String>> {
    let target = expr.child(0)?;
    // Only Utf8 column references (optionally cast); the engine's LIKE works on Utf8 and is
    // lowered through the column's string dictionary.
    if type_class(&target.return_type()?)? != TypeClass::Varchar {
        return Ok(None);
    }
    let Some(col) = ctx.utf8_column(&target)? else {
        return Ok(None);
    };
    if !ctx.small_dictionary(col) {
        return Ok(None);
    }
    let Some(col_sql) = translate(&target, ctx)? else {
        return Ok(None);
    };
    let kw = if negated { "NOT LIKE" } else { "LIKE" };
    let raw = format!("({col_sql} {kw} {})", sql_string_literal(pattern));
    // A positive LIKE never matches the NULL dictionary slot; NOT LIKE needs the guard.
    guard_nullable_columns(expr, ctx, raw, !negated)
}

fn sql_string_literal(text: &str) -> String {
    format!("'{}'", text.replace('\'', "''"))
}

/// Render a finite float as a SQL numeric literal that parses back to exactly `value`
/// (shortest round-trip representation), always in scientific notation (`2e-1`, `3e0`).
///
/// Both dialects that may evaluate pushed SQL read an exponent literal as a DOUBLE: the
/// engine parses every non-integer literal as Float64, and DuckDB (the host fallback) types
/// `2e-1` as DOUBLE. A plain `0.2` would be a DECIMAL in DuckDB, so `f = 0.2` on a FLOAT
/// column would compare in FLOAT there and in DOUBLE on the engine
/// (docs/issues/W2-O-002-pushdown-sql-dialect-on-host-fallback.md).
fn float_literal(value: f64) -> Option<String> {
    value.is_finite().then(|| format!("{value:e}"))
}

/// Civil date (proleptic Gregorian) for a day count relative to 1970-01-01.
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let month = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let year = yoe + era * 400 + i64::from(month <= 2);
    (year, month, day)
}

fn format_date_days(days: i64) -> Option<String> {
    let (year, month, day) = civil_from_days(days);
    if !(1..=9999).contains(&year) {
        return None;
    }
    Some(format!("{year:04}-{month:02}-{day:02}"))
}

/// Render an instant given in nanoseconds since the epoch as `YYYY-MM-DD HH:MM:SS[.ffffff]`.
/// Values with sub-microsecond precision are declined: the host fallback path parses
/// `TIMESTAMP` literals with microsecond precision only.
fn format_timestamp_nanos(nanos: i128) -> Option<String> {
    const NANOS_PER_DAY: i128 = 86_400 * 1_000_000_000;
    if nanos % 1_000 != 0 {
        return None;
    }
    let days = i64::try_from(nanos.div_euclid(NANOS_PER_DAY)).ok()?;
    let in_day = nanos.rem_euclid(NANOS_PER_DAY);
    let date = format_date_days(days)?;
    let secs = in_day / 1_000_000_000;
    let micros = (in_day % 1_000_000_000) / 1_000;
    let (h, m, s) = (secs / 3600, (secs / 60) % 60, secs % 60);
    Some(if micros == 0 {
        format!("{date} {h:02}:{m:02}:{s:02}")
    } else {
        format!("{date} {h:02}:{m:02}:{s:02}.{micros:06}")
    })
}

fn timestamp_literal(nanos: Option<i128>) -> Option<String> {
    nanos
        .and_then(format_timestamp_nanos)
        .map(|text| format!("TIMESTAMP '{text}'"))
}

fn value_to_bolt_sql_literal(val: &Value) -> Result<Option<String>> {
    if val.is_null()? {
        return Ok(None);
    }
    let micros = |v: i64| Some(i128::from(v) * 1_000);
    match val.logical_type()?.id()? {
        LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_BOOLEAN => {
            Ok(bool::from_value(val)?.map(|b| if b { "TRUE" } else { "FALSE" }.to_string()))
        }
        LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_TINYINT => {
            Ok(i8::from_value(val)?.map(|v| v.to_string()))
        }
        LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_SMALLINT => {
            Ok(i16::from_value(val)?.map(|v| v.to_string()))
        }
        LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_INTEGER => {
            Ok(i32::from_value(val)?.map(|v| v.to_string()))
        }
        LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_BIGINT => {
            Ok(i64::from_value(val)?.map(|v| v.to_string()))
        }
        LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_UTINYINT => {
            Ok(u8::from_value(val)?.map(|v| v.to_string()))
        }
        LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_USMALLINT => {
            Ok(u16::from_value(val)?.map(|v| v.to_string()))
        }
        LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_UINTEGER => {
            Ok(u32::from_value(val)?.map(|v| v.to_string()))
        }
        LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_UBIGINT => Ok(u64::from_value(val)?
            .filter(|&v| v <= i64::MAX as u64)
            .map(|v| v.to_string())),
        LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_HUGEINT => {
            let text = val.dbg_string()?;
            Ok(text
                .trim()
                .parse::<i128>()
                .ok()
                .and_then(|v| i64::try_from(v).ok())
                .map(|v| v.to_string()))
        }
        // An f32 is rendered via its exact f64 value: the engine compares FLOAT columns
        // after widening them, so `0.3f32` must be written `0.30000001192092896`, not `0.3`.
        LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_FLOAT => {
            Ok(f32::from_value(val)?.and_then(|f| float_literal(f64::from(f))))
        }
        LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_DOUBLE => {
            Ok(f64::from_value(val)?.and_then(float_literal))
        }
        LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_VARCHAR => {
            Ok(String::from_value(val)?.map(|s| sql_string_literal(&s)))
        }
        // Only integral DECIMAL constants, as integer literals: the engine reads `2.5` as a
        // DOUBLE and DuckDB as an exact DECIMAL (docs/issues/W2-E-003). Comparisons of
        // integer columns with fractional constants go through
        // `translate_integer_vs_decimal`.
        LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_DECIMAL => Ok(DecimalValue::from_value(val)?
            .and_then(|dec| {
                let unit = 10i128.checked_pow(u32::from(dec.scale))?;
                (dec.value % unit == 0)
                    .then(|| i64::try_from(dec.value / unit).ok())
                    .flatten()
                    .map(|v| v.to_string())
            })),
        LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_DATE => Ok(DateValue::from_value(val)?
            .and_then(|d| format_date_days(i64::from(d.0)))
            .map(|text| format!("DATE '{text}'"))),
        // Timestamps are rendered from the raw epoch value, never from the display string
        // (which depends on the session `TimeZone` for TIMESTAMPTZ).
        LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_TIMESTAMP => Ok(timestamp_literal(
            TimestampValue::from_value(val)?.and_then(|v| micros(v.0)),
        )),
        LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_TIMESTAMP_SEC => Ok(timestamp_literal(
            TimestampSecValue::from_value(val)?.map(|v| i128::from(v.0) * 1_000_000_000),
        )),
        LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_TIMESTAMP_MS => Ok(timestamp_literal(
            TimestampMsValue::from_value(val)?.map(|v| i128::from(v.0) * 1_000_000),
        )),
        LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_TIMESTAMP_NS => Ok(timestamp_literal(
            TimestampNsValue::from_value(val)?.map(|v| i128::from(v.0)),
        )),
        // UTC instant rendered as UTC wall-clock time; only used against UTC-typed columns
        // (see `tz_constants_safe`). TIMESTAMP_TZ_NS constants are not pushed.
        LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_TIMESTAMP_TZ => Ok(timestamp_literal(
            TimestampTzValue::from_value(val)?.and_then(|v| micros(v.0)),
        )),
        _ => Ok(None),
    }
}

// ---------------------------------------------------------------------------
// 2b. bolt_aggregate(table_name VARCHAR, group_by VARCHAR, aggregates VARCHAR)
// ---------------------------------------------------------------------------

struct BoltAggregateTableFunction;

struct BoltAggregateBindData {
    query_bind: BoltQueryBindData,
}

/// Reject `group_by` / `aggregates` fragments that would escape the generated
/// `SELECT <group_by>, <aggregates> FROM <table> GROUP BY <group_by>`: statement separators,
/// comments (which would swallow the `FROM` clause), unbalanced parentheses and unterminated
/// quotes. Quoted text is skipped, so `'a;b'` and `"odd--name"` are fine.
///
/// Quoting this scan does not model is refused outright (W9-D-002): a backtick identifier
/// (`` `a'` `` in the engine's dialect) or a dollar-quoted string (`$$'$$`) can hold a `'`,
/// which would make the scan skip a real `--` or `;` after it. So is a backslash in a string
/// opened right after an `E` / `e`: in an escape string (`E'\''`) it escapes the quote in both
/// dialects, while after a longer word (`name'...'`) it does not, and the two cannot be told
/// apart reliably here.
fn validate_sql_fragment(text: &str, what: &str) -> Result<()> {
    let invalid = |reason: &str| {
        Err(Error::invalid_input(format!(
            "bolt_aggregate {what} {reason}: {text}"
        )))
    };
    let mut chars = text.chars().peekable();
    let mut depth = 0usize;
    // The previous character outside quotes, and whether the string that just closed (when
    // `prev` is its closing `'`) was a possible escape string: `E'a''\''` reopens it.
    let mut prev: Option<char> = None;
    let mut last_string_escaped = false;
    while let Some(c) = chars.next() {
        match c {
            '\'' | '"' => {
                let maybe_escape_string = c == '\''
                    && (matches!(prev, Some('e' | 'E'))
                        || (prev == Some('\'') && last_string_escaped));
                last_string_escaped = maybe_escape_string;
                // `''` / `""` escapes simply close and reopen the quote.
                let mut closed = false;
                for n in chars.by_ref() {
                    if maybe_escape_string && n == '\\' {
                        return invalid("must not contain backslashes in E'...' strings");
                    }
                    if n == c {
                        closed = true;
                        break;
                    }
                }
                if !closed {
                    return invalid("has an unterminated quote");
                }
            }
            '`' => return invalid("must not contain backticks (quote names with \"...\")"),
            '$' => return invalid("must not contain '$'"),
            ';' => return invalid("must not contain ';'"),
            '-' if chars.peek() == Some(&'-') => return invalid("must not contain comments"),
            '/' if chars.peek() == Some(&'*') => return invalid("must not contain comments"),
            '(' => depth += 1,
            ')' => {
                if depth == 0 {
                    return invalid("has unbalanced parentheses");
                }
                depth -= 1;
            }
            _ => {}
        }
        prev = Some(c);
    }
    if depth != 0 {
        return invalid("has unbalanced parentheses");
    }
    Ok(())
}

impl TableFunctionCallbacks for BoltAggregateTableFunction {
    type BindData = BoltAggregateBindData;
    type GlobalState = ChunkedResult;
    type LocalState = ChunkLocalState;

    fn bind(
        &self,
        context: &Context,
        arguments: Vec<BindArgument>,
        bind_handle: BindFunctionHandle<'_>,
    ) -> Result<Self::BindData> {
        sync_options(context)?;
        if arguments.len() < 3 {
            return Err(Error::invalid_input(
                "bolt_aggregate requires three string arguments: table_name, group_by, aggregates",
            ));
        }
        let tbl_arg = arguments[0].value.as_ref().ok_or_else(|| {
            Error::invalid_input("bolt_aggregate table_name must be a constant string at bind time")
        })?;
        let grp_arg = arguments[1].value.as_ref().ok_or_else(|| {
            Error::invalid_input("bolt_aggregate group_by must be a constant string at bind time")
        })?;
        let agg_arg = arguments[2].value.as_ref().ok_or_else(|| {
            Error::invalid_input("bolt_aggregate aggregates must be a constant string at bind time")
        })?;

        let tbl = String::from_value(tbl_arg)?
            .ok_or_else(|| Error::invalid_input("bolt_aggregate table_name cannot be NULL"))?;
        let grp = String::from_value(grp_arg)?
            .ok_or_else(|| Error::invalid_input("bolt_aggregate group_by cannot be NULL"))?;
        let agg = String::from_value(agg_arg)?
            .ok_or_else(|| Error::invalid_input("bolt_aggregate aggregates cannot be NULL"))?;

        let tbl = tbl.trim();
        let grp = grp.trim();
        let agg = agg.trim();

        if tbl.is_empty() {
            return Err(Error::invalid_input(
                "bolt_aggregate table_name cannot be empty",
            ));
        }
        if agg.is_empty() {
            return Err(Error::invalid_input(
                "bolt_aggregate aggregates cannot be empty",
            ));
        }

        validate_sql_fragment(grp, "group_by")?;
        validate_sql_fragment(agg, "aggregates")?;

        // Validate the table exists and use its resolved catalog name: `table_info` strips a
        // `bolt.` prefix and matches case-insensitively, the engine's SQL frontend does not.
        let summary = BoltRuntime::global().table_info(tbl)?.ok_or_else(|| {
            Error::invalid_input(format!("Table '{tbl}' is not registered in Craton Bolt"))
        })?;
        let table = quote_ident(&summary.name);

        let sql = if grp.is_empty() {
            format!("SELECT {agg} FROM {table}")
        } else {
            format!("SELECT {grp}, {agg} FROM {table} GROUP BY {grp}")
        };

        let query_bind = bind_bolt_sql_query(context, sql, &bind_handle)?;

        Ok(BoltAggregateBindData { query_bind })
    }

    fn init_global_state(
        &self,
        bind_data: &Self::BindData,
        context: &Context,
        column_data: InitColumnData<'_>,
    ) -> Result<(Option<Self::GlobalState>, Option<usize>)> {
        BoltQueryTableFunction.init_global_state(&bind_data.query_bind, context, column_data)
    }

    fn init_local_state(
        &self,
        _bind_data: &Self::BindData,
        _global_state: Option<&Self::GlobalState>,
        _context: &Context,
        _column_data: InitColumnData<'_>,
    ) -> Result<Option<Self::LocalState>> {
        Ok(Some(ChunkLocalState::default()))
    }

    fn progress(
        &self,
        bind_data: &Self::BindData,
        global_state: Option<&Self::GlobalState>,
        context: &Context,
    ) -> Result<Option<f64>> {
        BoltQueryTableFunction.progress(&bind_data.query_bind, global_state, context)
    }

    fn exec(
        &self,
        bind_data: &Self::BindData,
        global_state: Option<&Self::GlobalState>,
        local_state: Option<&mut Self::LocalState>,
        context: &Context,
        output: DataChunkRef<'_>,
        column_info: ExecColumnInfo<'_>,
    ) -> Result<()> {
        BoltQueryTableFunction.exec(
            &bind_data.query_bind,
            global_state,
            local_state,
            context,
            output,
            column_info,
        )
    }
}

impl TablePartitioningCallbacks for BoltAggregateTableFunction {
    fn partition_data(
        &self,
        _bind_data: &Self::BindData,
        _global_state: Option<&Self::GlobalState>,
        local_state: Option<&mut Self::LocalState>,
        _context: &Context,
        _partition_data: PartitionData<'_>,
    ) -> Result<usize> {
        Ok(local_state.map_or(0, |l| l.batch_index))
    }
}

// ---------------------------------------------------------------------------
// 3. bolt_tables()
// ---------------------------------------------------------------------------

struct BoltTablesFunction;

struct BoltTablesGlobalState {
    rows: Vec<TableSummary>,
    /// Whether the backend is the GPU engine (`None`: not known yet), see
    /// `BoltRuntime::table_gpu_status`.
    gpu: Option<bool>,
    /// Catalog key -> why the engine does not hold that table (W5-C-002).
    notes: BTreeMap<String, String>,
    cursor: AtomicUsize,
}

impl TableFunctionCallbacks for BoltTablesFunction {
    type BindData = ();
    type GlobalState = BoltTablesGlobalState;
    type LocalState = ();

    fn bind(
        &self,
        context: &Context,
        _arguments: Vec<BindArgument>,
        bind_handle: BindFunctionHandle<'_>,
    ) -> Result<Self::BindData> {
        sync_options(context)?;
        bind_handle.add_result_column("table_name", &String::logical_type(context)?)?;
        bind_handle.add_result_column("num_rows", &i64::logical_type(context)?)?;
        bind_handle.add_result_column("num_columns", &i32::logical_type(context)?)?;
        bind_handle.add_result_column("bytes_in_memory", &i64::logical_type(context)?)?;
        bind_handle.add_result_column("schema", &String::logical_type(context)?)?;
        // Appended last so positional consumers keep their column indices (W5-C-002).
        bind_handle.add_result_column("on_gpu", &bool::logical_type(context)?)?;
        bind_handle.add_result_column("gpu_note", &String::logical_type(context)?)?;
        bind_handle.set_order(OrderPreservation::Insertion)?;
        Ok(())
    }

    fn init_global_state(
        &self,
        _bind_data: &Self::BindData,
        _context: &Context,
        _column_data: InitColumnData<'_>,
    ) -> Result<(Option<Self::GlobalState>, Option<usize>)> {
        let rows = BoltRuntime::global().list_tables()?;
        let (gpu, notes) = BoltRuntime::global().table_gpu_status();
        Ok((
            Some(BoltTablesGlobalState {
                rows,
                gpu,
                notes,
                cursor: AtomicUsize::new(0),
            }),
            Some(1),
        ))
    }

    fn exec(
        &self,
        _bind_data: &Self::BindData,
        global_state: Option<&Self::GlobalState>,
        _local_state: Option<&mut Self::LocalState>,
        _context: &Context,
        mut output: DataChunkRef<'_>,
        _column_info: ExecColumnInfo<'_>,
    ) -> Result<()> {
        let state =
            global_state.ok_or_else(|| Error::api_error("Missing bolt_tables global state"))?;
        let chunk_rows = output_rows(&output)?;
        let start = state.cursor.fetch_add(chunk_rows, Ordering::SeqCst);
        let remaining = state.rows.len().saturating_sub(start);
        let count = remaining.min(chunk_rows);

        if count == 0 {
            output.get_vector_at_mut::<Unknown>(0)?.set_size(0)?;
            return Ok(());
        }

        let [v_name, v_rows, v_cols, v_bytes, v_schema, v_on_gpu, v_note] =
            output.get_vector_array_mut([0, 1, 2, 3, 4, 5, 6])?;
        let mut v_name = v_name.cast::<String>()?;
        let mut v_rows = v_rows.cast::<i64>()?;
        let mut v_cols = v_cols.cast::<i32>()?;
        let mut v_bytes = v_bytes.cast::<i64>()?;
        let mut v_schema = v_schema.cast::<String>()?;
        let mut v_on_gpu = v_on_gpu.cast::<bool>()?;
        let mut v_note = v_note.cast::<String>()?;

        v_name.set_size(count)?;
        v_rows.set_size(count)?;
        v_cols.set_size(count)?;
        v_bytes.set_size(count)?;
        v_schema.set_size(count)?;
        v_on_gpu.set_size(count)?;
        v_note.set_size(count)?;

        let mut w_name = v_name.writer()?;
        let mut w_rows = v_rows.writer()?;
        let mut w_cols = v_cols.writer()?;
        let mut w_bytes = v_bytes.writer()?;
        let mut w_schema = v_schema.writer()?;
        let mut w_on_gpu = v_on_gpu.writer()?;
        let mut w_note = v_note.writer()?;

        for i in 0..count {
            let item = &state.rows[start + i];
            let schema_str = item.schema_string();
            let note = state.notes.get(&item.name);
            w_name.set(i, Some(item.name.as_str()))?;
            w_rows.set(i, Some(item.num_rows as i64))?;
            w_cols.set(i, Some(item.schema.fields().len() as i32))?;
            w_bytes.set(i, Some(item.bytes_in_memory as i64))?;
            w_schema.set(i, Some(schema_str.as_str()))?;
            // NULL while the backend is not known yet; `false` on the host fallback.
            w_on_gpu.set(i, state.gpu.map(|gpu| gpu && note.is_none()))?;
            w_note.set(i, note.map(String::as_str))?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// 4. bolt_pool_stats()
// ---------------------------------------------------------------------------

struct BoltPoolStatsFunction;

struct BoltPoolStatsGlobalState {
    snapshot: BoltTelemetrySnapshot,
    emitted: AtomicBool,
}

impl TableFunctionCallbacks for BoltPoolStatsFunction {
    type BindData = ();
    type GlobalState = BoltPoolStatsGlobalState;
    type LocalState = ();

    fn bind(
        &self,
        context: &Context,
        _arguments: Vec<BindArgument>,
        bind_handle: BindFunctionHandle<'_>,
    ) -> Result<Self::BindData> {
        sync_options(context)?;
        let str_ty = String::logical_type(context)?;
        let i32_ty = i32::logical_type(context)?;
        let i64_ty = i64::logical_type(context)?;

        bind_handle.add_result_column("backend", &str_ty)?;
        bind_handle.add_result_column("device_id", &i32_ty)?;
        bind_handle.add_result_column("device_name", &str_ty)?;
        bind_handle.add_result_column("memory_budget_bytes", &i64_ty)?;
        bind_handle.add_result_column("registered_tables", &i64_ty)?;
        bind_handle.add_result_column("total_pooled_bytes", &i64_ty)?;
        bind_handle.add_result_column("bucket_count", &i64_ty)?;
        bind_handle.add_result_column("oom_recovery_count", &i64_ty)?;
        bind_handle.add_result_column("proactive_eviction_count", &i64_ty)?;
        bind_handle.add_result_column("ptx_cache_hits", &i64_ty)?;
        bind_handle.add_result_column("ptx_cache_misses", &i64_ty)?;
        bind_handle.add_result_column("ptx_cache_evictions", &i64_ty)?;
        bind_handle.add_result_column("plan_cache_hits", &i64_ty)?;
        bind_handle.add_result_column("plan_cache_misses", &i64_ty)?;
        bind_handle.add_result_column("plan_cache_evictions", &i64_ty)?;
        // Appended last so `SELECT *` consumers keep their column positions (W5-C-002).
        bind_handle.add_result_column("fallback_reason", &str_ty)?;
        bind_handle.set_cardinality(1, true)?;
        Ok(())
    }

    fn init_global_state(
        &self,
        _bind_data: &Self::BindData,
        _context: &Context,
        _column_data: InitColumnData<'_>,
    ) -> Result<(Option<Self::GlobalState>, Option<usize>)> {
        let snapshot = BoltRuntime::global().telemetry()?;
        Ok((
            Some(BoltPoolStatsGlobalState {
                snapshot,
                emitted: AtomicBool::new(false),
            }),
            Some(1),
        ))
    }

    fn exec(
        &self,
        _bind_data: &Self::BindData,
        global_state: Option<&Self::GlobalState>,
        _local_state: Option<&mut Self::LocalState>,
        _context: &Context,
        mut output: DataChunkRef<'_>,
        _column_info: ExecColumnInfo<'_>,
    ) -> Result<()> {
        let state =
            global_state.ok_or_else(|| Error::api_error("Missing bolt_pool_stats global state"))?;
        if state.emitted.swap(true, Ordering::SeqCst) {
            output.get_vector_at_mut::<Unknown>(0)?.set_size(0)?;
            return Ok(());
        }

        let s = &state.snapshot;

        let mut c0 = output.get_vector_at_mut::<String>(0)?;
        c0.set_size(1)?;
        c0.writer()?.set(0, Some(s.backend.as_str()))?;

        let mut c1 = output.get_vector_at_mut::<i32>(1)?;
        c1.set_size(1)?;
        c1.writer()?.set(0, Some(s.device_id))?;

        let mut c2 = output.get_vector_at_mut::<String>(2)?;
        c2.set_size(1)?;
        c2.writer()?.set(0, Some(s.device_name.as_str()))?;

        let i64_cols = [
            (3, s.memory_budget_bytes),
            (4, s.registered_tables),
            (5, s.total_pooled_bytes),
            (6, s.bucket_count),
            (7, s.oom_recovery_count),
            (8, s.proactive_eviction_count),
            (9, s.ptx_cache_hits),
            (10, s.ptx_cache_misses),
            (11, s.ptx_cache_evictions),
            (12, s.plan_cache_hits),
            (13, s.plan_cache_misses),
            (14, s.plan_cache_evictions),
        ];

        for (col_idx, value) in i64_cols {
            let mut vec = output.get_vector_at_mut::<i64>(col_idx)?;
            vec.set_size(1)?;
            vec.writer()?.set(0, Some(value))?;
        }

        // NULL (not '') when everything runs on the GPU (W5-C-002).
        let mut c15 = output.get_vector_at_mut::<String>(15)?;
        c15.set_size(1)?;
        let reason = (!s.fallback_reason.is_empty()).then_some(s.fallback_reason.as_str());
        c15.writer()?.set(0, reason)?;

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// 5. bolt_metrics()
// ---------------------------------------------------------------------------

struct BoltMetricsFunction;

struct BoltMetricsGlobalState {
    rows: Vec<MetricRow>,
    cursor: AtomicUsize,
}

impl TableFunctionCallbacks for BoltMetricsFunction {
    type BindData = ();
    type GlobalState = BoltMetricsGlobalState;
    type LocalState = ();

    fn bind(
        &self,
        context: &Context,
        _arguments: Vec<BindArgument>,
        bind_handle: BindFunctionHandle<'_>,
    ) -> Result<Self::BindData> {
        sync_options(context)?;
        bind_handle.add_result_column("metric", &String::logical_type(context)?)?;
        bind_handle.add_result_column("category", &String::logical_type(context)?)?;
        bind_handle.add_result_column("value", &i64::logical_type(context)?)?;
        bind_handle.set_order(OrderPreservation::Insertion)?;
        Ok(())
    }

    fn init_global_state(
        &self,
        _bind_data: &Self::BindData,
        _context: &Context,
        _column_data: InitColumnData<'_>,
    ) -> Result<(Option<Self::GlobalState>, Option<usize>)> {
        let (rows, _prom) = BoltRuntime::global().metrics()?;
        Ok((
            Some(BoltMetricsGlobalState {
                rows,
                cursor: AtomicUsize::new(0),
            }),
            Some(1),
        ))
    }

    fn exec(
        &self,
        _bind_data: &Self::BindData,
        global_state: Option<&Self::GlobalState>,
        _local_state: Option<&mut Self::LocalState>,
        _context: &Context,
        mut output: DataChunkRef<'_>,
        _column_info: ExecColumnInfo<'_>,
    ) -> Result<()> {
        let state =
            global_state.ok_or_else(|| Error::api_error("Missing bolt_metrics global state"))?;
        let chunk_rows = output_rows(&output)?;
        let start = state.cursor.fetch_add(chunk_rows, Ordering::SeqCst);
        let remaining = state.rows.len().saturating_sub(start);
        let count = remaining.min(chunk_rows);

        if count == 0 {
            output.get_vector_at_mut::<Unknown>(0)?.set_size(0)?;
            return Ok(());
        }

        let [v_metric, v_cat, v_val] = output.get_vector_array_mut([0, 1, 2])?;
        let mut v_metric = v_metric.cast::<String>()?;
        let mut v_cat = v_cat.cast::<String>()?;
        let mut v_val = v_val.cast::<i64>()?;

        v_metric.set_size(count)?;
        v_cat.set_size(count)?;
        v_val.set_size(count)?;

        let mut w_metric = v_metric.writer()?;
        let mut w_cat = v_cat.writer()?;
        let mut w_val = v_val.writer()?;

        for i in 0..count {
            let row = &state.rows[start + i];
            w_metric.set(i, Some(row.metric.as_str()))?;
            w_cat.set(i, Some(row.category.as_str()))?;
            w_val.set(i, Some(row.value))?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// 6. bolt_devices()
// ---------------------------------------------------------------------------

struct BoltDevicesFunction;

struct BoltDevicesGlobalState {
    rows: Vec<DeviceInfo>,
    cursor: AtomicUsize,
}

impl TableFunctionCallbacks for BoltDevicesFunction {
    type BindData = ();
    type GlobalState = BoltDevicesGlobalState;
    type LocalState = ();

    fn bind(
        &self,
        context: &Context,
        _arguments: Vec<BindArgument>,
        bind_handle: BindFunctionHandle<'_>,
    ) -> Result<Self::BindData> {
        sync_options(context)?;
        bind_handle.add_result_column("device_id", &i32::logical_type(context)?)?;
        bind_handle.add_result_column("name", &String::logical_type(context)?)?;
        bind_handle.add_result_column("total_memory_bytes", &i64::logical_type(context)?)?;
        bind_handle.add_result_column("is_active", &bool::logical_type(context)?)?;
        bind_handle.set_order(OrderPreservation::Insertion)?;
        Ok(())
    }

    fn init_global_state(
        &self,
        _bind_data: &Self::BindData,
        _context: &Context,
        _column_data: InitColumnData<'_>,
    ) -> Result<(Option<Self::GlobalState>, Option<usize>)> {
        let rows = BoltRuntime::global().devices()?;
        Ok((
            Some(BoltDevicesGlobalState {
                rows,
                cursor: AtomicUsize::new(0),
            }),
            Some(1),
        ))
    }

    fn exec(
        &self,
        _bind_data: &Self::BindData,
        global_state: Option<&Self::GlobalState>,
        _local_state: Option<&mut Self::LocalState>,
        _context: &Context,
        mut output: DataChunkRef<'_>,
        _column_info: ExecColumnInfo<'_>,
    ) -> Result<()> {
        let state =
            global_state.ok_or_else(|| Error::api_error("Missing bolt_devices global state"))?;
        let chunk_rows = output_rows(&output)?;
        let start = state.cursor.fetch_add(chunk_rows, Ordering::SeqCst);
        let remaining = state.rows.len().saturating_sub(start);
        let count = remaining.min(chunk_rows);

        if count == 0 {
            output.get_vector_at_mut::<Unknown>(0)?.set_size(0)?;
            return Ok(());
        }

        let [v_id, v_name, v_mem, v_active] = output.get_vector_array_mut([0, 1, 2, 3])?;
        let mut v_id = v_id.cast::<i32>()?;
        let mut v_name = v_name.cast::<String>()?;
        let mut v_mem = v_mem.cast::<i64>()?;
        let mut v_active = v_active.cast::<bool>()?;

        v_id.set_size(count)?;
        v_name.set_size(count)?;
        v_mem.set_size(count)?;
        v_active.set_size(count)?;

        let mut w_id = v_id.writer()?;
        let mut w_name = v_name.writer()?;
        let mut w_mem = v_mem.writer()?;
        let mut w_active = v_active.writer()?;

        for i in 0..count {
            let dev = &state.rows[start + i];
            w_id.set(i, Some(dev.device_id))?;
            w_name.set(i, Some(dev.name.as_str()))?;
            w_mem.set(i, Some(dev.total_memory_bytes))?;
            w_active.set(i, Some(dev.is_active))?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// 7. CALL forms of catalog mutation (docs/issues/W4-C-004)
// ---------------------------------------------------------------------------

/// A catalog mutation offered as a table function, so that `CALL bolt_clear_tables()` (or
/// `SELECT * FROM bolt_clear_tables()`) acts exactly once per execution. The scalar forms of
/// the same names act once per evaluated row, which the optimizer decides (see
/// `scalar_functions.rs`); DuckDB keeps scalar and table functions in separate catalogs, so the
/// names are shared.
///
/// The side effect runs in the global init, which DuckDB calls once per execution: not at bind
/// (`EXPLAIN` and prepared-statement re-binds bind again without executing), and again for every
/// `EXECUTE` of a prepared `CALL`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CatalogCall {
    ClearTables,
    DropTable,
    RegisterSql,
    #[cfg(not(target_arch = "wasm32"))]
    LoadFile(crate::scalar_functions::IngestFormat),
}

impl CatalogCall {
    fn function_name(self) -> &'static str {
        match self {
            Self::ClearTables => "bolt_clear_tables",
            Self::DropTable => "bolt_drop_table",
            Self::RegisterSql => "bolt_register_sql",
            #[cfg(not(target_arch = "wasm32"))]
            Self::LoadFile(format) => format.function_name(),
        }
    }

    /// Names of the required arguments; calls with two also take an optional `mode`.
    fn required_args(self) -> &'static [&'static str] {
        match self {
            Self::ClearTables => &[],
            Self::DropTable => &["table_name"],
            Self::RegisterSql => &["table_name", "sql"],
            #[cfg(not(target_arch = "wasm32"))]
            Self::LoadFile(_) => &["table_name", "path"],
        }
    }

    fn takes_mode(self) -> bool {
        self.required_args().len() == 2
    }
}

struct CatalogCallFunction(CatalogCall);

struct CatalogCallBindData {
    /// The required arguments, in signature order.
    args: Vec<String>,
    mode: RegisterMode,
}

/// The one result row: a count (`removed` / `rows`) or `dropped`.
#[derive(Clone, Copy)]
enum CatalogCallValue {
    Count(i64),
    Flag(bool),
}

struct CatalogCallGlobalState {
    value: CatalogCallValue,
    emitted: AtomicBool,
}

fn saturating_i64(n: usize) -> i64 {
    i64::try_from(n).unwrap_or(i64::MAX)
}

fn prefixed(err: Error, prefix: impl std::fmt::Display) -> Error {
    Error {
        code: err.code,
        message: format!("{prefix}: {}", err.message),
    }
}

impl TableFunctionCallbacks for CatalogCallFunction {
    type BindData = CatalogCallBindData;
    type GlobalState = CatalogCallGlobalState;
    type LocalState = ();

    fn bind(
        &self,
        context: &Context,
        arguments: Vec<BindArgument>,
        bind_handle: BindFunctionHandle<'_>,
    ) -> Result<Self::BindData> {
        let name = self.0.function_name();
        sync_options(context)?;
        let expected = self.0.required_args().len() + usize::from(self.0.takes_mode());
        if arguments.len() != expected {
            return Err(Error::invalid_input(format!(
                "{name} expects {expected} argument(s), got {}",
                arguments.len()
            )));
        }
        let mut texts = Vec::with_capacity(expected);
        for (i, arg) in arguments.iter().enumerate() {
            let Some(value) = arg.value.as_ref() else {
                // A prepared-statement parameter that is not bound yet: DuckDB binds the call
                // again at EXECUTE time, with the value.
                return Err(Error {
                    code:
                        duckdb_neo::error::DuckDBError::DUCKDB_V2_ERROR_QUERY_PARAMETER_NOT_RESOLVED,
                    message: format!("{name} argument {} must be a constant", i + 1),
                });
            };
            let text = String::from_value(value)?.ok_or_else(|| {
                Error::invalid_input(format!("{name} argument {} cannot be NULL", i + 1))
            })?;
            texts.push(text);
        }
        // Validate the mode at bind time (no side effect yet), as the scalar forms do before
        // reading a file or running a query.
        let mode = if self.0.takes_mode() {
            let text = texts.pop().expect("mode argument is present");
            RegisterMode::parse(&text).map_err(|e| prefixed(e, name))?
        } else {
            RegisterMode::Replace
        };

        let (column, ty) = match self.0 {
            CatalogCall::ClearTables => ("removed", i64::logical_type(context)?),
            CatalogCall::DropTable => ("dropped", bool::logical_type(context)?),
            _ => ("rows", i64::logical_type(context)?),
        };
        bind_handle.add_result_column(column, &ty)?;
        bind_handle.set_cardinality(1, true)?;
        Ok(CatalogCallBindData { args: texts, mode })
    }

    fn init_global_state(
        &self,
        bind_data: &Self::BindData,
        context: &Context,
        _column_data: InitColumnData<'_>,
    ) -> Result<(Option<Self::GlobalState>, Option<usize>)> {
        let name = self.0.function_name();
        // A prepared CALL may execute long after it was bound.
        sync_options(context)?;
        let args = &bind_data.args;
        let value = match self.0 {
            CatalogCall::ClearTables => CatalogCallValue::Count(saturating_i64(
                BoltRuntime::global()
                    .clear_tables()
                    .map_err(|e| prefixed(e, format!("{name}()")))?,
            )),
            CatalogCall::DropTable => CatalogCallValue::Flag(
                BoltRuntime::global()
                    .drop_table(&args[0])
                    .map_err(|e| prefixed(e, format!("{name}('{}')", args[0])))?,
            ),
            CatalogCall::RegisterSql => CatalogCallValue::Count(saturating_i64(
                crate::scalar_functions::register_sql_once(&args[0], &args[1], bind_data.mode)?,
            )),
            #[cfg(not(target_arch = "wasm32"))]
            CatalogCall::LoadFile(format) => {
                CatalogCallValue::Count(saturating_i64(crate::scalar_functions::load_file_once(
                    context,
                    format,
                    &args[0],
                    &args[1],
                    bind_data.mode,
                )?))
            }
        };
        Ok((
            Some(CatalogCallGlobalState {
                value,
                emitted: AtomicBool::new(false),
            }),
            Some(1),
        ))
    }

    fn exec(
        &self,
        _bind_data: &Self::BindData,
        global_state: Option<&Self::GlobalState>,
        _local_state: Option<&mut Self::LocalState>,
        _context: &Context,
        mut output: DataChunkRef<'_>,
        _column_info: ExecColumnInfo<'_>,
    ) -> Result<()> {
        let state = global_state.ok_or_else(|| {
            Error::api_error(format!("Missing {} global state", self.0.function_name()))
        })?;
        if state.emitted.swap(true, Ordering::SeqCst) {
            output.get_vector_at_mut::<Unknown>(0)?.set_size(0)?;
            return Ok(());
        }
        match state.value {
            CatalogCallValue::Count(n) => {
                let mut v = output.get_vector_at_mut::<i64>(0)?;
                v.set_size(1)?;
                v.writer()?.set(0, Some(n))?;
            }
            CatalogCallValue::Flag(b) => {
                let mut v = output.get_vector_at_mut::<bool>(0)?;
                v.set_size(1)?;
                v.writer()?.set(0, Some(b))?;
            }
        }
        Ok(())
    }
}

/// Register the `CALL` forms (W4-C-004): `bolt_clear_tables()`, `bolt_drop_table(name)`,
/// `bolt_register_sql(name, sql [, mode])`, `bolt_load_parquet(name, path [, mode])` and
/// `bolt_load_arrow_ipc(name, path [, mode])`.
fn register_catalog_calls(extension: &Extension) -> Result<()> {
    #[allow(unused_mut)] // wasm32 has no file loads
    let mut calls = vec![
        (
            CatalogCall::ClearTables,
            "Removes every table from the Craton Bolt catalog, once per execution",
        ),
        (
            CatalogCall::DropTable,
            "Removes one table from the Craton Bolt catalog, once per execution",
        ),
        (
            CatalogCall::RegisterSql,
            "Runs a SQL query on Craton Bolt and registers the result, once per execution",
        ),
    ];
    #[cfg(not(target_arch = "wasm32"))]
    {
        use crate::scalar_functions::IngestFormat;
        calls.push((
            CatalogCall::LoadFile(IngestFormat::Parquet),
            "Loads a Parquet file into a Craton Bolt table, once per execution",
        ));
        calls.push((
            CatalogCall::LoadFile(IngestFormat::ArrowIpc),
            "Loads an Arrow IPC file into a Craton Bolt table, once per execution",
        ));
    }
    for (call, description) in calls {
        let mut signature = SignatureBuilder::new();
        for name in call.required_args() {
            signature = signature.param(FunctionParameter::new::<String>(*name));
        }
        if call.takes_mode() {
            signature = signature.param(FunctionParameter::with_default::<String>(
                "mode",
                "replace".to_string(),
            ));
        }
        TableFunctionBuilder::new(
            call.function_name(),
            signature.documentation(FunctionDocumentationBuilder::new().description(description)),
            CatalogCallFunction(call),
        )
        .register(extension)?;
    }
    Ok(())
}

/// Register all Craton Bolt table functions on `extension`.
pub fn register_table_functions(extension: &Extension) -> Result<()> {
    register_catalog_calls(extension)?;

    for fn_name in ["bolt_query", "bolt_sql"] {
        TableFunctionBuilder::new(
            fn_name,
            SignatureBuilder::new()
                .param(FunctionParameter::new::<String>("sql"))
                .documentation(FunctionDocumentationBuilder::new().description(
                    "Executes a SQL query on the Craton Bolt GPU SQL engine and streams the result",
                )),
            BoltQueryTableFunction,
        )
        .with_projection_pushdown(true)
        .with_progress()
        .with_partitioning()
        .register(extension)?;
    }

    TableFunctionBuilder::new(
        "bolt_scan",
        SignatureBuilder::new()
            .param(FunctionParameter::new::<String>("table_name"))
            .documentation(FunctionDocumentationBuilder::new().description(
                "Scans a table registered in Craton Bolt GPU memory with projection and filter pushdown",
            )),
        BoltScanTableFunction,
    )
    .with_projection_pushdown(true)
    .with_filter_pushdown()
    .with_progress()
    .with_partitioning()
    .register(extension)?;

    TableFunctionBuilder::new(
        "bolt_aggregate",
        SignatureBuilder::new()
            .param(FunctionParameter::new::<String>("table_name"))
            .param(FunctionParameter::new::<String>("group_by"))
            .param(FunctionParameter::new::<String>("aggregates"))
            .documentation(FunctionDocumentationBuilder::new().description(
                "Pushes down aggregation and group by expressions directly to Craton Bolt GPU SQL engine",
            )),
        BoltAggregateTableFunction,
    )
    .with_projection_pushdown(true)
    .with_progress()
    .with_partitioning()
    .register(extension)?;

    TableFunctionBuilder::new(
        "bolt_tables",
        SignatureBuilder::new().documentation(
            FunctionDocumentationBuilder::new().description(
                "Lists all tables currently registered in the Craton Bolt GPU catalog",
            ),
        ),
        BoltTablesFunction,
    )
    .register(extension)?;

    TableFunctionBuilder::new(
        "bolt_pool_stats",
        SignatureBuilder::new().documentation(
            FunctionDocumentationBuilder::new().description(
                "Returns GPU memory pool, PTX JIT cache, SQL plan cache, and device telemetry for Craton Bolt",
            ),
        ),
        BoltPoolStatsFunction,
    )
    .register(extension)?;

    TableFunctionBuilder::new(
        "bolt_metrics",
        SignatureBuilder::new().documentation(
            FunctionDocumentationBuilder::new()
                .description("Returns Craton Bolt execution phase counters and telemetry metrics"),
        ),
        BoltMetricsFunction,
    )
    .register(extension)?;

    TableFunctionBuilder::new(
        "bolt_devices",
        SignatureBuilder::new().documentation(
            FunctionDocumentationBuilder::new()
                .description("Lists available CUDA GPU devices detected by Craton Bolt"),
        ),
        BoltDevicesFunction,
    )
    .register(extension)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::StringArray;

    #[test]
    fn test_conform_batch_to_schema() {
        use arrow_array::Int32Array;
        let bound = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, true)]));

        // Same type passes through; different type is cast to the declared one.
        let same = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, true)])),
            vec![Arc::new(Int64Array::from(vec![1, 2]))],
        )
        .unwrap();
        assert_eq!(conform_batch_to_schema(same, &bound).unwrap().num_rows(), 2);
        let narrow = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("v", DataType::Int32, false)])),
            vec![Arc::new(Int32Array::from(vec![7]))],
        )
        .unwrap();
        let out = conform_batch_to_schema(narrow, &bound).unwrap();
        assert_eq!(out.column(0).data_type(), &DataType::Int64);

        // Column-count mismatch and uncastable types are errors.
        let two = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("a", DataType::Int64, true),
                Field::new("b", DataType::Int64, true),
            ])),
            vec![
                Arc::new(Int64Array::from(vec![1])),
                Arc::new(Int64Array::from(vec![1])),
            ],
        )
        .unwrap();
        assert!(conform_batch_to_schema(two, &bound).is_err());
        let text = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("v", DataType::Utf8, true)])),
            vec![Arc::new(StringArray::from(vec!["abc"]))],
        )
        .unwrap();
        // Unparseable values are an error, not silently NULL.
        assert!(conform_batch_to_schema(text, &bound).is_err());
        let numeric_text = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("v", DataType::Utf8, true)])),
            vec![Arc::new(StringArray::from(vec![Some("42"), None]))],
        )
        .unwrap();
        let out = conform_batch_to_schema(numeric_text, &bound).unwrap();
        let values = out.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(values.value(0), 42);
        assert!(values.is_null(1));
    }
    fn id_label_batch(ids: Vec<i64>, labels: Vec<Option<&str>>) -> RecordBatch {
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("id", DataType::Int64, false),
                Field::new("label", DataType::Utf8, true),
            ])),
            vec![
                Arc::new(Int64Array::from(ids)),
                Arc::new(StringArray::from(labels)),
            ],
        )
        .expect("batch")
    }

    fn filter_on(sql: &str, columns: Vec<usize>) -> PushedFilter {
        PushedFilter {
            sql: sql.to_string(),
            columns,
            dictionary_columns: Vec::new(),
            exact: true,
        }
    }

    #[test]
    fn test_quote_ident_and_chunked_result() {
        assert_eq!(quote_ident("simple"), "\"simple\"");
        assert_eq!(quote_ident("has\"quote"), "\"has\"\"quote\"");
        assert_eq!(
            quote_ident("col\"with\"\"quotes"),
            "\"col\"\"with\"\"\"\"quotes\""
        );

        let empty = ChunkedResult::new(Vec::new(), vec![0]);
        assert_eq!(empty.progress(), 1.0);
        assert_eq!(empty.max_threads(), 1);

        // Two stored batches: 3000 rows (two chunks) and 10 rows (one chunk); no concat.
        let big = id_label_batch((0..3000).collect(), vec![Some("x"); 3000]);
        let small = id_label_batch((0..10).collect(), vec![None; 10]);
        let state = ChunkedResult::new(
            vec![big, RecordBatch::new_empty(small.schema()), small],
            vec![1, 0],
        );
        assert_eq!(state.progress(), 0.0);
        state
            .chunks
            .set(plan_chunks(&state.batches, STANDARD_CHUNK_CAPACITY))
            .unwrap();
        assert_eq!(
            state.chunks.get().unwrap(),
            &vec![
                ChunkRef {
                    batch: 0,
                    offset: 0,
                    len: STANDARD_CHUNK_CAPACITY,
                    rows_before: 0
                },
                ChunkRef {
                    batch: 0,
                    offset: STANDARD_CHUNK_CAPACITY,
                    len: 3000 - STANDARD_CHUNK_CAPACITY,
                    rows_before: STANDARD_CHUNK_CAPACITY,
                },
                ChunkRef {
                    batch: 2,
                    offset: 0,
                    len: 10,
                    rows_before: 3000
                },
            ]
        );
        assert_eq!(state.total_rows, 3010);
        assert_eq!(state.progress(), 0.0);
        state.next_chunk.store(2, Ordering::Relaxed);
        assert!((state.progress() - 3000.0 / 3010.0).abs() < 1e-12);
        state.next_chunk.store(9, Ordering::Relaxed);
        assert_eq!(state.progress(), 1.0);

        let many = ChunkedResult::single(
            id_label_batch((0..200_000).collect(), vec![None; 200_000]),
            vec![0],
        );
        assert_eq!(
            plan_chunks(&many.batches, STANDARD_CHUNK_CAPACITY).len(),
            98
        );
        // A smaller output capacity (a DuckDB with a smaller STANDARD_VECTOR_SIZE) gives
        // proportionally more, smaller chunks (docs/issues/W3-C-002).
        let small_plan = plan_chunks(&many.batches, 1024);
        assert_eq!(small_plan.len(), 196);
        assert!(small_plan.iter().all(|c| c.len <= 1024));
        assert_eq!(many.max_threads(), 6);
    }

    #[test]
    fn test_restored_column_cache_hits_and_evicts() {
        use arrow_array::Int32Array;
        let live: ArrayRef = Arc::new(Int32Array::from(vec![1, 2, 3]));
        let (h0, m0, _) = restored_column_cache_stats_for_testing();
        let a = restore_stored_column(&live, &DataType::Int16).unwrap();
        let b = restore_stored_column(&live, &DataType::Int16).unwrap();
        assert_eq!(a.data_type(), &DataType::Int16);
        assert!(
            Arc::ptr_eq(&a, &b),
            "the second conversion is served from the cache"
        );
        let (h1, m1, _) = restored_column_cache_stats_for_testing();
        assert_eq!((h1 - h0, m1 - m0), (1, 1));
        // Same type: zero-copy, never cached.
        let same = restore_stored_column(&live, &DataType::Int32).unwrap();
        assert!(Arc::ptr_eq(&same, &live));

        let key = (Arc::as_ptr(&live) as *const () as usize, DataType::Int16);
        let cached = |key: &(usize, DataType)| {
            RESTORED_CACHE
                .lock()
                .unwrap()
                .as_ref()
                .is_some_and(|c| c.entries.contains_key(key))
        };
        assert!(cached(&key));
        // Once nothing but the cache holds the stored column, the next miss evicts it.
        drop((live, a, b, same));
        let other: ArrayRef = Arc::new(Int32Array::from(vec![4]));
        restore_stored_column(&other, &DataType::Int16).unwrap();
        assert!(!cached(&key));
        // Out-of-range values are still an error, never cached.
        let wide: ArrayRef = Arc::new(Int32Array::from(vec![70_000]));
        assert!(restore_stored_column(&wide, &DataType::Int16).is_err());
    }

    #[test]
    fn test_gather_rows_across_batches() {
        let a = id_label_batch(vec![10, 11, 12], vec![Some("a"), None, Some("c")]);
        let b = id_label_batch(vec![20, 21], vec![Some("d"), Some("e")]);
        let out = gather_rows(&[a.clone(), b.clone()], &[0, 2, 4], &[1, 0]).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].num_rows(), 2);
        assert_eq!(out[1].num_rows(), 1);
        let labels = out[0]
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(labels.value(0), "a");
        assert_eq!(labels.value(1), "c");
        let ids = out[1]
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(ids.values(), &[21]);
        // No output columns: row counts only.
        let counts = gather_rows(&[a, b], &[1, 3, 4], &[]).unwrap();
        assert_eq!(
            counts.iter().map(RecordBatch::num_rows).collect::<Vec<_>>(),
            vec![1, 2]
        );
    }

    #[test]
    fn test_filter_take_uses_scratch_table() {
        let gpu = BoltRuntime::global().telemetry().unwrap().backend == "cuda-gpu";
        let tables_before = BoltRuntime::global().list_tables().unwrap().len();
        let empty = RecordBatch::new_empty(id_label_batch(vec![], vec![]).schema());
        assert!(
            filter_take(
                &[empty],
                &[filter_on("(\"id\" = 1)", vec![0])],
                " WHERE (\"id\" = 1)",
                &[0, 1]
            )
            .unwrap()
            .is_empty()
        );

        let first = id_label_batch(vec![10, 20], vec![Some("a"), Some("b")]);
        let second = id_label_batch(vec![30, 40], vec![None, Some("d")]);
        let filters = [filter_on("(\"id\" >= 20) AND (\"id\" <= 30)", vec![0])];
        let taken = filter_take(
            &[first.clone(), second.clone()],
            &filters,
            " WHERE (\"id\" >= 20) AND (\"id\" <= 30)",
            &[0, 1],
        );
        if !gpu {
            // The scratch table needs the GPU engine; bolt_scan then uses the host path.
            assert!(taken.is_err());
            return;
        }
        let taken = taken.expect("filter take");
        assert_eq!(taken.len(), 2);
        let ids = taken[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(ids.values(), &[20]);
        let labels = taken[1]
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert!(labels.is_null(0));

        // A user column named like the synthetic row-id column must not collide with it.
        let clash = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                "__bolt_row_id",
                DataType::Int64,
                false,
            )])),
            vec![Arc::new(Int64Array::from(vec![7, 8, 9]))],
        )
        .unwrap();
        let taken = filter_take(
            &[clash],
            &[filter_on("(\"__bolt_row_id\" > 7)", vec![0])],
            " WHERE (\"__bolt_row_id\" > 7)",
            &[0],
        )
        .expect("filter take with clashing column name");
        let vals = taken[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(vals.values(), &[8, 9]);

        // Nothing matched; and the scratch table never shows up in the catalog.
        let none = filter_take(
            &[first],
            &[filter_on("(\"id\" = 999)", vec![0])],
            " WHERE (\"id\" = 999)",
            &[0],
        )
        .unwrap();
        assert!(none.is_empty());
        assert_eq!(
            BoltRuntime::global().list_tables().unwrap().len(),
            tables_before
        );
    }

    #[test]
    fn test_utf8_distinct_and_sql_fragments() {
        let batch = id_label_batch(
            vec![1, 2, 3, 4],
            vec![Some("a"), None, Some("b"), Some("a")],
        );
        assert_eq!(
            utf8_distinct_within(std::slice::from_ref(&batch), 1, 2),
            Some(2)
        );
        assert_eq!(
            utf8_distinct_within(std::slice::from_ref(&batch), 1, 1),
            None
        );
        // Not a string column / out of range.
        assert_eq!(
            utf8_distinct_within(std::slice::from_ref(&batch), 0, 10),
            None
        );
        assert_eq!(utf8_distinct_within(&[batch], 5, 10), None);

        for ok in [
            "sum(x)",
            "count(*), avg(\"a;b\")",
            "max(s) FILTER (WHERE s <> 'x--y;')",
            "\"odd\"\"name\"",
        ] {
            assert!(validate_sql_fragment(ok, "aggregates").is_ok(), "{ok}");
        }
        for bad in [
            "sum(x); DROP TABLE t",
            "sum(x) -- c",
            "sum(x) /* c */",
            "sum(x))",
            "sum((x)",
            "'open",
        ] {
            assert!(validate_sql_fragment(bad, "aggregates").is_err(), "{bad}");
        }

        let fields = arrow_schema::Fields::from(vec![
            Field::new("a", DataType::Int64, true),
            Field::new("A", DataType::Int64, true),
            Field::new("", DataType::Int64, true),
            Field::new("a_1", DataType::Int64, true),
        ]);
        assert_eq!(
            unique_column_names(&fields),
            vec!["a", "A_1", "col_2", "a_1_1"]
        );
    }
    #[test]
    fn test_lossless_widening_rules() {
        use TypeClass::*;
        let int = |signed, bits| Int { signed, bits };
        // Widening integers, exact int -> float, float -> double, decimal widening.
        assert!(is_lossless_widening(int(true, 32), int(true, 64)));
        assert!(is_lossless_widening(int(false, 8), int(true, 16)));
        assert!(is_lossless_widening(int(false, 32), Float64));
        assert!(is_lossless_widening(int(true, 16), Float32));
        assert!(is_lossless_widening(Float32, Float64));
        assert!(is_lossless_widening(
            int(true, 32),
            Decimal {
                width: 12,
                scale: 2
            }
        ));
        assert!(is_lossless_widening(
            Decimal { width: 5, scale: 1 },
            Decimal { width: 7, scale: 2 }
        ));
        assert!(is_lossless_widening(Varchar, Varchar));
        // Narrowing, lossy, sign-changing and type-changing casts are not.
        assert!(!is_lossless_widening(Float64, int(true, 32)));
        assert!(!is_lossless_widening(int(true, 64), int(true, 32)));
        assert!(!is_lossless_widening(int(true, 8), int(false, 16)));
        assert!(!is_lossless_widening(int(false, 16), int(true, 16)));
        assert!(!is_lossless_widening(int(true, 64), Float64));
        assert!(!is_lossless_widening(int(true, 32), Float32));
        assert!(!is_lossless_widening(Float64, Float32));
        assert!(!is_lossless_widening(int(true, 32), Varchar));
        assert!(!is_lossless_widening(Date, Timestamp));
        assert!(!is_lossless_widening(Timestamp, TimestampTz));
        assert!(!is_lossless_widening(TimestampNs, Timestamp));
        assert!(!is_lossless_widening(Other, Other));
        assert!(!is_lossless_widening(
            int(true, 32),
            Decimal { width: 9, scale: 0 }
        ));
        assert!(!is_lossless_widening(
            Decimal { width: 5, scale: 2 },
            Decimal { width: 5, scale: 1 }
        ));
        assert!(!is_lossless_widening(
            Decimal { width: 5, scale: 1 },
            Decimal { width: 5, scale: 2 }
        ));
    }

    #[test]
    fn test_literal_formatting() {
        // Always scientific notation: DOUBLE in DuckDB and on the engine (W2-O-002).
        assert_eq!(float_literal(3.0).as_deref(), Some("3e0"));
        assert_eq!(float_literal(-2.5).as_deref(), Some("-2.5e0"));
        assert_eq!(float_literal(0.0).as_deref(), Some("0e0"));
        assert_eq!(float_literal(0.2).as_deref(), Some("2e-1"));
        assert_eq!(float_literal(1e300).as_deref(), Some("1e300"));
        assert_eq!(float_literal(-1.5e-7).as_deref(), Some("-1.5e-7"));
        assert_eq!(float_literal(f64::NAN), None);
        assert_eq!(float_literal(f64::INFINITY), None);
        // FLOAT constants are rendered through their exact f64 value.
        assert_eq!(
            float_literal(f64::from(0.3f32)).as_deref(),
            Some("3.0000001192092896e-1")
        );
        for v in [
            0.1,
            1.0 / 3.0,
            123456.789,
            9.999e15,
            1e16,
            2.2250738585072014e-308,
        ] {
            let text = float_literal(v).unwrap();
            assert_eq!(text.parse::<f64>().unwrap(), v, "{text}");
        }

        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(19_723), (2024, 1, 1));
        assert_eq!(civil_from_days(19_782), (2024, 2, 29));
        assert_eq!(civil_from_days(-1), (1969, 12, 31));
        assert_eq!(format_date_days(-719_162).as_deref(), Some("0001-01-01"));
        assert_eq!(format_date_days(-719_163), None);
        assert_eq!(format_date_days(i64::from(i32::MAX)), None);

        let micros = 1_704_067_200_000_000i128; // 2024-01-01 00:00:00 UTC
        assert_eq!(
            format_timestamp_nanos(micros * 1_000).as_deref(),
            Some("2024-01-01 00:00:00")
        );
        assert_eq!(
            format_timestamp_nanos((micros + 3_723_000_042) * 1_000).as_deref(),
            Some("2024-01-01 01:02:03.000042")
        );
        assert_eq!(
            format_timestamp_nanos(-1_000).as_deref(),
            Some("1969-12-31 23:59:59.999999")
        );
        // Sub-microsecond and out-of-range values are declined.
        assert_eq!(format_timestamp_nanos(micros * 1_000 + 1), None);
        assert_eq!(format_timestamp_nanos(i128::from(i64::MAX) * 1_000), None);
        assert_eq!(
            timestamp_literal(Some(micros * 1_000)).as_deref(),
            Some("TIMESTAMP '2024-01-01 00:00:00'")
        );
        assert_eq!(timestamp_literal(None), None);
        assert_eq!(sql_string_literal("O'Brien"), "'O''Brien'");
    }

    #[test]
    fn test_schema_compatibility_and_comparison_flip() {
        let a = Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("s", DataType::Utf8, true),
        ]);
        let a_nullable = Schema::new(vec![
            Field::new("id", DataType::Int32, true),
            Field::new("s", DataType::Utf8, true),
        ]);
        let retyped = Schema::new(vec![
            Field::new("id", DataType::Float64, false),
            Field::new("s", DataType::Utf8, true),
        ]);
        let renamed = Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("t", DataType::Utf8, true),
        ]);
        let shorter = Schema::new(vec![Field::new("id", DataType::Int32, false)]);
        assert!(schemas_compatible(&a, &a));
        assert!(schemas_compatible(&a_nullable, &a));
        assert!(!schemas_compatible(&retyped, &a));
        assert!(!schemas_compatible(&renamed, &a));
        assert!(!schemas_compatible(&shorter, &a));
        assert!(
            schema_drift_error("t")
                .to_string()
                .contains("replaced after the query was bound")
        );

        assert_eq!(
            flip_comparison(ExpressionType::CompareLessThan),
            ExpressionType::CompareGreaterThan
        );
        assert_eq!(
            flip_comparison(ExpressionType::CompareGreaterThanOrEqualTo),
            ExpressionType::CompareLessThanOrEqualTo
        );
        assert_eq!(
            flip_comparison(ExpressionType::CompareEqual),
            ExpressionType::CompareEqual
        );
        assert_eq!(
            comparison_sql_op(ExpressionType::CompareNotEqual),
            Some("!=")
        );
        assert_eq!(comparison_sql_op(ExpressionType::CompareIn), None);
    }

    #[test]
    fn test_conform_rejects_out_of_range_values() {
        // W1-E-001: a value that does not fit the declared type is an error, never NULL.
        let bound = Arc::new(Schema::new(vec![Field::new("v", DataType::Int32, true)]));
        let wide = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, true)])),
            vec![Arc::new(Int64Array::from(vec![1, 5_000_000_000]))],
        )
        .unwrap();
        assert!(conform_batch_to_schema(wide, &bound).is_err());
    }

    #[test]
    fn test_scan_projection_and_declared_types() {
        use crate::arrow_bridge::DUCKDB_TYPE_METADATA_KEY;
        let meta = |t: &str| HashMap::from([(DUCKDB_TYPE_METADATA_KEY.to_string(), t.to_string())]);
        let schema = Schema::new(vec![
            Field::new("s", DataType::Utf8, true),
            Field::new("small", DataType::Int32, true).with_metadata(meta("SMALLINT")),
            Field::new("big", DataType::Int64, true),
            Field::new("flag", DataType::Boolean, true),
        ]);
        let p = ScanProjection::new(&[2, 1, 2, 99], 4);
        assert_eq!(p.unique, vec![2, 1]);
        assert_eq!(p.positions, vec![0, 1, 0, usize::MAX]);
        let fields = p.fields(&schema);
        let batch = declared_batch(
            vec![
                Arc::new(Int64Array::from(vec![7, 8])) as ArrayRef,
                Arc::new(arrow_array::Int32Array::from(vec![-3, 300])),
            ],
            2,
            &fields,
        )
        .unwrap();
        assert_eq!(batch.column(0).data_type(), &DataType::Int64);
        assert_eq!(batch.column(1).data_type(), &DataType::Int16);
        // Checked: a stored value outside SMALLINT is an error.
        assert!(
            declared_batch(
                vec![Arc::new(arrow_array::Int32Array::from(vec![70_000])) as ArrayRef],
                1,
                &fields[1..],
            )
            .is_err()
        );
        assert!(declared_batch(Vec::new(), 1, &fields).is_err());
        assert_eq!(declared_batch(Vec::new(), 5, &[]).unwrap().num_rows(), 5);

        // Row-count scans pick the narrowest fixed-width column, never Utf8 (W2-E-018).
        assert_eq!(row_count_column(&schema, &[]), Some(3));
        let utf8_only = Schema::new(vec![Field::new("s", DataType::Utf8, true)]);
        assert_eq!(row_count_column(&utf8_only, &[]), None);
        let ints = Schema::new(vec![
            Field::new("a", DataType::Int64, true),
            Field::new("b", DataType::Int64, true),
        ]);
        assert_eq!(row_count_column(&ints, &[filter_on("x", vec![1])]), Some(1));

        // Engine results must have exactly the requested columns (W2-E-017).
        let two = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("ID", DataType::Int64, true),
                Field::new("v", DataType::Int64, true),
            ])),
            vec![
                Arc::new(Int64Array::from(vec![1])) as ArrayRef,
                Arc::new(Int64Array::from(vec![2])),
            ],
        )
        .unwrap();
        assert!(result_shape_matches(&two, &["id", "v"]));
        assert!(!result_shape_matches(&two, &["id"]));
        assert!(!result_shape_matches(&two, &["v", "id"]));
        assert!(!result_shape_matches(&two, &["id", "v", "w"]));

        // A replace that keeps the storage type but changes the declared type is drift.
        let as_integer = Schema::new(vec![
            Field::new("s", DataType::Utf8, true),
            Field::new("small", DataType::Int32, true).with_metadata(meta("INTEGER")),
            Field::new("big", DataType::Int64, true),
            Field::new("flag", DataType::Boolean, true),
        ]);
        assert!(schemas_compatible(&schema, &schema));
        assert!(!schemas_compatible(&as_integer, &schema));
    }

    #[test]
    fn test_gpu_backend_check_is_cached_per_config() {
        let expected = BoltRuntime::global().telemetry().unwrap().backend == "cuda-gpu";
        assert_eq!(gpu_backend_available().unwrap(), expected);
        // Second call is served from the cache and agrees.
        assert_eq!(gpu_backend_available().unwrap(), expected);
        #[cfg(feature = "cuda-stub")]
        assert!(!expected, "cuda-stub builds never report the GPU backend");
    }
}

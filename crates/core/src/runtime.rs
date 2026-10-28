// SPDX-License-Identifier: Apache-2.0

//! Thread-safe Craton Bolt runtime and dedicated CUDA worker thread.
//!
//! Why a dedicated worker thread?
//! 1. `craton_bolt::Engine` enforces a single active engine per process (`ACTIVE_ENGINE`) and
//!    contains `!Sync` internal caches (`RefCell`).
//! 2. CUDA Driver API contexts (`CUcontext`) are thread-bound; DuckDB executes callbacks across
//!    its worker thread pool. Running all `craton_bolt::Engine` operations on a single dedicated
//!    OS thread guarantees that the CUDA context created during `EngineBuilder::build()` is
//!    always current for every upload, kernel launch, and device-to-host transfer.
//!
//! Metadata reads (`config`, `table_info`, `list_tables`) are answered from a snapshot that the
//! worker publishes after every mutation, so they never queue behind a long-running query.

use std::{
    collections::{BTreeMap, HashMap, HashSet},
    panic::{AssertUnwindSafe, catch_unwind},
    path::PathBuf,
    sync::{
        Arc, Mutex, OnceLock, RwLock, RwLockReadGuard,
        atomic::{AtomicU64, Ordering},
        mpsc,
    },
    time::Duration,
};
#[cfg(not(target_arch = "wasm32"))]
use std::{
    sync::{atomic::AtomicU8, mpsc::Sender},
    thread,
};

use arrow_array::{Array, ArrayRef, RecordBatch};
use arrow_cast::cast::CastOptions;

use crate::arrow_bridge::cast_with_options;
#[cfg(not(target_arch = "wasm32"))]
use arrow_schema::TimeUnit as ArrowTimeUnit;
use arrow_schema::{DataType as ArrowDataType, Schema, SchemaRef};
use duckdb_neo::{Error, Result};

#[cfg(not(target_arch = "wasm32"))]
mod engine_support;
#[cfg(not(target_arch = "wasm32"))]
mod fallback;
#[cfg(not(target_arch = "wasm32"))]
mod plan_facts;
#[cfg(not(target_arch = "wasm32"))]
mod sql_text;
#[cfg(not(target_arch = "wasm32"))]
mod type_binder;
#[cfg(not(target_arch = "wasm32"))]
use self::{engine_support::*, fallback::*, plan_facts::*, sql_text::*, type_binder::*};

/// Version of the embedded `craton-bolt` crate.
///
/// `craton-bolt` does not export a version constant, so the value lives here next to the
/// dependency; the unit test `craton_bolt_version_matches_cargo_lock` fails when it drifts from
/// the version resolved in `Cargo.lock`.
pub const CRATON_BOLT_VERSION: &str = "0.9.0";

/// Name of the engine-only scratch table used by
/// [`BoltRuntime::execute_sql_with_scratch_table`]. Reserved: user tables cannot use it.
pub const SCRATCH_TABLE_NAME: &str = "__bolt_scratch";

/// Prefix of every name Craton Bolt uses internally (scratch table, host-fallback mirror source,
/// hybrid-filter temporary tables, row-id columns). Catalog tables cannot use it (W2-E-015).
pub const RESERVED_NAME_PREFIX: &str = "__bolt_";

/// Stack sizes tried, in order, for the worker thread (W2-B-001).
///
/// craton-bolt 0.9 walks predicate trees recursively, and long `OR` chains (user-written, or
/// the engine's dictionary lowering of Utf8 `LIKE` / ordering predicates, one `OR` per matching
/// dictionary entry) are left-deep, so the depth grows with the query and the data. With the
/// default 2 MiB stack a 300-term `OR` killed the process (`STATUS_STACK_OVERFLOW`). The
/// reservation is virtual memory; pages are only committed when touched. Smaller sizes are
/// tried when the platform refuses a large reservation.
#[cfg(not(target_arch = "wasm32"))]
pub const WORKER_STACK_SIZES: [usize; 3] = [512 << 20, 64 << 20, 8 << 20];

/// Most `AND` / `OR` / `IN`-list terms a query may contain to be sent to the GPU engine; larger
/// queries run on the host fallback (W2-B-001). The worker stack ([`WORKER_STACK_SIZES`]) holds
/// much deeper trees, so this is a backstop against unbounded recursion, not a tuning knob.
#[cfg(not(target_arch = "wasm32"))]
pub const MAX_GPU_PREDICATE_TERMS: usize = 4096;

/// Most distinct values a Utf8 column may have for a dictionary-lowered predicate (`LIKE`, or an
/// ordering comparison with a string literal) over it to run on the GPU engine. craton-bolt
/// lowers such a predicate to one equality per matching dictionary entry, chained with `OR`, so
/// the tree depth is bounded by this count (W2-B-001).
#[cfg(not(target_arch = "wasm32"))]
pub const MAX_GPU_DICTIONARY_LOWERED_VALUES: usize = 4096;

/// Largest number of string bytes one Utf8 column of a table may hold to be uploaded to the
/// engine: craton-bolt 0.9 keeps (and concatenates) every table as one `Utf8` array with `i32`
/// offsets. Larger tables are served by the host fallback (W2-E-004).
#[cfg(not(target_arch = "wasm32"))]
pub const MAX_ENGINE_UTF8_BYTES: usize = i32::MAX as usize;

/// Number of dropped tables kept as empty placeholders in the engine before the engine is
/// rebuilt to purge them (see [`BoltWorkerState::engine_forget_table`]).
#[cfg(not(target_arch = "wasm32"))]
const MAX_ENGINE_TOMBSTONES: usize = 128;

/// How a table registration should behave when the table name already exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegisterMode {
    /// Fail if a table with the same name already exists.
    Create,
    /// Replace any existing table with the same name.
    Replace,
    /// Append batches to an existing table, or create it if it does not exist.
    Append,
}

impl RegisterMode {
    /// Parse a registration mode (`COPY ... (FORMAT bolt, MODE '...')`, the `mode` argument of
    /// `bolt_load_*` / `bolt_register_sql`). ASCII case-insensitive, surrounding whitespace is
    /// ignored.
    ///
    /// The error is the single canonical message for every caller (W1-E-007): it names no
    /// particular entry point and echoes the value as the user wrote it, see
    /// [`RegisterMode::invalid_mode_message`].
    pub fn parse(mode: &str) -> Result<Self> {
        match mode.trim().to_ascii_lowercase().as_str() {
            "replace" | "overwrite" => Ok(Self::Replace),
            "append" => Ok(Self::Append),
            "create" | "error" => Ok(Self::Create),
            _ => Err(Error::invalid_input(Self::invalid_mode_message(mode))),
        }
    }

    /// The canonical "invalid mode" message (W1-E-007), `mode` echoed as written:
    /// `Invalid Bolt registration mode 'x': expected 'replace', 'append', or 'create' (aliases:
    /// 'overwrite' = 'replace', 'error' = 'create')`. Callers may add where the mode came from.
    pub fn invalid_mode_message(mode: &str) -> String {
        format!(
            "Invalid Bolt registration mode '{mode}': expected 'replace', 'append', or 'create' \
             (aliases: 'overwrite' = 'replace', 'error' = 'create')"
        )
    }
}

/// Runtime configuration options controlled via DuckDB `SET bolt_*` options.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct BoltConfig {
    /// Target CUDA device ordinal (default `0`).
    pub device: i32,
    /// Optional GPU memory pool cap in bytes (`None` or `Some(0)` = uncapped).
    pub memory_budget: Option<usize>,
    /// Optional directory for persistent on-disk PTX cache.
    pub ptx_cache_dir: Option<PathBuf>,
    /// Whether `craton-bolt` tracing is enabled.
    pub enable_tracing: bool,
}

/// Summary metadata for a table registered in the Bolt catalog.
#[derive(Debug, Clone)]
pub struct TableSummary {
    /// Table name in the Bolt catalog.
    pub name: String,
    /// Arrow schema of the table.
    pub schema: SchemaRef,
    /// Total number of rows across all batches.
    pub num_rows: usize,
    /// Approximate in-memory Arrow byte footprint.
    pub bytes_in_memory: usize,
}

impl TableSummary {
    /// Human-readable schema summary string (`col: TYPE, ...`), as shown by `bolt_tables()`.
    ///
    /// A column that records its original DuckDB type (RFC-008 `duckdb.logical_type`
    /// metadata) shows that type (`v: SMALLINT`), the same type `bolt_scan` returns
    /// (W3-O-001); other columns show their Arrow storage type (`v: Int32`).
    pub fn schema_string(&self) -> String {
        self.schema
            .fields()
            .iter()
            .map(|f| match crate::arrow_bridge::original_duckdb_type(f) {
                Some(original) => format!("{}: {}", f.name(), original.to_metadata_text()),
                None => format!("{}: {}", f.name(), f.data_type()),
            })
            .collect::<Vec<_>>()
            .join(", ")
    }
}

/// Information about an available CUDA device.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// CUDA device ordinal (`0..count`).
    pub device_id: i32,
    /// Device name reported by `cuDeviceGetName`.
    pub name: String,
    /// Total device VRAM in bytes reported by `cuDeviceTotalMem_v2`.
    pub total_memory_bytes: i64,
    /// Whether this device is the currently configured Bolt device.
    pub is_active: bool,
}

/// Telemetry snapshot combining pool, PTX cache, plan cache, and runtime state.
#[derive(Debug, Clone)]
pub struct BoltTelemetrySnapshot {
    /// Active execution backend (`"cuda-gpu"` or `"host-planner-fallback"`).
    pub backend: String,
    /// Active CUDA device ordinal.
    pub device_id: i32,
    /// Active CUDA device name (or fallback reason).
    pub device_name: String,
    /// Configured memory budget in bytes (`0` if uncapped).
    pub memory_budget_bytes: i64,
    /// Number of tables currently registered in Bolt.
    pub registered_tables: i64,
    /// Total pooled GPU bytes in `craton_bolt::pool_stats()`.
    pub total_pooled_bytes: i64,
    /// Number of active size-class buckets in the GPU memory pool.
    pub bucket_count: i64,
    /// Number of OOM recovery evictions performed by the GPU memory pool.
    pub oom_recovery_count: i64,
    /// Number of proactive low-watermark evictions performed by the pool watcher.
    pub proactive_eviction_count: i64,
    /// PTX JIT cache hits.
    pub ptx_cache_hits: i64,
    /// PTX JIT cache misses.
    pub ptx_cache_misses: i64,
    /// PTX JIT cache evictions.
    pub ptx_cache_evictions: i64,
    /// SQL frontend plan cache hits.
    pub plan_cache_hits: i64,
    /// SQL frontend plan cache misses.
    pub plan_cache_misses: i64,
    /// SQL frontend plan cache evictions.
    pub plan_cache_evictions: i64,
    /// Why queries (or some tables) run on the host fallback instead of the GPU: the cached
    /// engine-initialization error, and/or one `table 'x' not on GPU: <reason>` entry per
    /// catalog table the engine could not upload (W1-E-003). Empty when everything is on the
    /// GPU.
    pub fallback_reason: String,
}

/// A single metric row from `craton_bolt::metrics_snapshot()`.
#[derive(Debug, Clone)]
pub struct MetricRow {
    /// Metric name.
    pub metric: String,
    /// Metric category: `"counter"` for counters; for each pipeline phase histogram,
    /// `"histogram_ns"` (`<phase>_latency_sum`, total latency in nanoseconds) and
    /// `"histogram_count"` (`<phase>_latency_count`, number of observations).
    pub category: String,
    /// Metric value.
    pub value: i64,
}

/// Result of executing a SQL query on Craton Bolt.
#[derive(Debug, Clone)]
pub struct SqlExecutionOutput {
    /// Result rows as an Arrow `RecordBatch`.
    pub batch: RecordBatch,
    /// Planned execution tier (`"Gpu"`, `"Host"`, or `"Hybrid"`).
    pub execution_tier: String,
}

/// Result of executing a SQL query on Craton Bolt, in the batches it was produced in
/// (W4-A-002): a host-fallback result keeps DuckDB's Arrow stream batches instead of being
/// concatenated, so a large result is not held twice. See [`BoltRuntime::execute_sql_batches`].
#[derive(Debug, Clone)]
pub struct SqlExecutionBatches {
    /// Schema of every batch.
    pub schema: SchemaRef,
    /// Result rows; empty when the result has no rows.
    pub batches: Vec<RecordBatch>,
    /// Planned execution tier (`"Gpu"`, `"Host"`, or `"Hybrid"`).
    pub execution_tier: String,
}

impl SqlExecutionBatches {
    /// Wrap a single result batch.
    pub fn single(batch: RecordBatch, execution_tier: String) -> Self {
        Self {
            schema: batch.schema(),
            batches: vec![batch],
            execution_tier,
        }
    }

    /// Total number of result rows.
    pub fn num_rows(&self) -> usize {
        self.batches.iter().map(RecordBatch::num_rows).sum()
    }

    /// Concatenate the batches into one [`SqlExecutionOutput`] (a single batch is moved, not
    /// copied).
    pub fn into_output(mut self) -> Result<SqlExecutionOutput> {
        let batch =
            match self.batches.len() {
                0 => RecordBatch::new_empty(self.schema),
                1 => self.batches.pop().expect("one batch"),
                _ => arrow_select::concat::concat_batches(&self.schema, &self.batches).map_err(
                    |e| Error::api_error(format!("Failed to concatenate Bolt result batches: {e}")),
                )?,
            };
        Ok(SqlExecutionOutput {
            batch,
            execution_tier: self.execution_tier,
        })
    }
}

struct RegisteredTable {
    schema: SchemaRef,
    batches: Vec<RecordBatch>,
    num_rows: usize,
    bytes_in_memory: usize,
    /// Total number of NULL values across all columns and batches. Used by the engine crash
    /// guard (W1-O-002, see [`BoltWorkerState::gpu_crash_guard`]).
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    null_count: usize,
    /// NULL values per column (same order as `schema`), for the column-level checks of the
    /// crash guard (W4-B-001, W4-B-003).
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    column_nulls: Vec<usize>,
    /// Monotonic version stamp, bumped on every mutation; used to detect stale mirrors in the
    /// embedded host-fallback DuckDB.
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    generation: u64,
    /// Cached [`RegisteredTable::max_utf8_distinct`]; reset on append.
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    utf8_distinct: OnceLock<usize>,
    /// Cached [`RegisteredTable::int_abs_max`]; reset on append.
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    int_abs_max: OnceLock<Vec<Option<u64>>>,
}

fn batch_null_count(batch: &RecordBatch) -> usize {
    batch.columns().iter().map(|c| c.null_count()).sum()
}

/// NULL values per column of `batch`.
fn batch_column_nulls(batch: &RecordBatch) -> Vec<usize> {
    batch.columns().iter().map(|c| c.null_count()).collect()
}

/// Add `batch`'s per-column NULL counts to `totals`.
fn add_column_nulls(totals: &mut Vec<usize>, batch: &RecordBatch) {
    if totals.len() < batch.num_columns() {
        totals.resize(batch.num_columns(), 0);
    }
    for (total, col) in totals.iter_mut().zip(batch.columns()) {
        *total += col.null_count();
    }
}

impl RegisteredTable {
    /// Build an entry from already-normalized batches that all share `schema`.
    /// `batches` must not be empty (an empty table is one zero-row batch).
    fn from_batches(schema: SchemaRef, batches: Vec<RecordBatch>) -> Self {
        let num_rows = batches.iter().map(RecordBatch::num_rows).sum();
        let bytes_in_memory = batches.iter().map(RecordBatch::get_array_memory_size).sum();
        let null_count = batches.iter().map(batch_null_count).sum();
        let mut column_nulls = vec![0; schema.fields().len()];
        for batch in &batches {
            add_column_nulls(&mut column_nulls, batch);
        }
        Self {
            schema,
            batches,
            num_rows,
            bytes_in_memory,
            null_count,
            column_nulls,
            generation: 0,
            utf8_distinct: OnceLock::new(),
            int_abs_max: OnceLock::new(),
        }
    }

    /// Largest absolute value per column for Int32 / Int64 columns (`None` for other
    /// columns), for the SUM overflow check of the crash guard (W8-B-007). Computed on first
    /// use from the stored values (NULL slots included, which can only raise the bound).
    #[cfg(not(target_arch = "wasm32"))]
    fn int_abs_max(&self) -> &[Option<u64>] {
        self.int_abs_max
            .get_or_init(|| int_abs_max_per_column(self.schema.fields().len(), &self.batches))
    }

    /// Upper bound for the distinct non-NULL values of the table's Utf8 columns (the largest
    /// column), saturating just above [`MAX_GPU_DICTIONARY_LOWERED_VALUES`] (W2-B-001). Tables
    /// with at most that many rows answer with their row count without scanning.
    #[cfg(not(target_arch = "wasm32"))]
    fn max_utf8_distinct(&self) -> usize {
        use arrow_array::cast::AsArray;
        let limit = MAX_GPU_DICTIONARY_LOWERED_VALUES;
        if self.num_rows <= limit {
            return self.num_rows;
        }
        *self.utf8_distinct.get_or_init(|| {
            let mut max = 0;
            for (idx, field) in self.schema.fields().iter().enumerate() {
                if field.data_type() != &ArrowDataType::Utf8 {
                    continue;
                }
                let mut seen: HashSet<&str> = HashSet::new();
                'batches: for batch in &self.batches {
                    for value in batch.column(idx).as_string::<i32>().iter().flatten() {
                        seen.insert(value);
                        if seen.len() > limit {
                            break 'batches;
                        }
                    }
                }
                max = max.max(seen.len());
                if max > limit {
                    break;
                }
            }
            max
        })
    }

    fn append_batch(&mut self, batch: RecordBatch) {
        if batch.num_rows() == 0 {
            return;
        }
        // Drop the zero-row placeholder an empty table is stored as.
        if self.num_rows == 0 {
            self.batches.clear();
            self.bytes_in_memory = 0;
        }
        self.num_rows += batch.num_rows();
        self.bytes_in_memory += batch.get_array_memory_size();
        self.null_count += batch_null_count(&batch);
        add_column_nulls(&mut self.column_nulls, &batch);
        self.batches.push(batch);
        self.utf8_distinct = OnceLock::new();
        self.int_abs_max = OnceLock::new();
    }

    fn summary(&self, name: &str) -> TableSummary {
        TableSummary {
            name: name.to_string(),
            schema: self.schema.clone(),
            num_rows: self.num_rows,
            bytes_in_memory: self.bytes_in_memory,
        }
    }
}

/// Read-only metadata published by the worker after every mutation.
#[derive(Debug, Default)]
struct MetaSnapshot {
    config: BoltConfig,
    tables: BTreeMap<String, TableSummary>,
    /// ASCII-lowercased name -> catalog key.
    lower_index: HashMap<String, String>,
    /// Catalog key -> stored batches (`Arc`-backed clones of the catalog's), so readers never
    /// wait for the worker (W1-B-004).
    batches: HashMap<String, Vec<RecordBatch>>,
    /// Engine state, published before every reply (W1-B-004).
    engine: EngineStatus,
}

/// What the worker knows about the GPU engine, published so `telemetry` and `backend_is_gpu`
/// can answer without queueing behind a long query (W1-B-004).
#[derive(Debug, Default, Clone)]
struct EngineStatus {
    /// `Some(true)`: an engine is active, or was built for the current configuration and is
    /// only waiting for a lazy rebuild. `Some(false)`: the build failure is cached. `None`:
    /// not known yet (no build attempted since the last configuration change); the answer
    /// then needs the worker, which tries to build the engine.
    gpu: Option<bool>,
    /// The cached engine-initialization error, if any.
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    last_error: Option<String>,
    /// See [`BoltTelemetrySnapshot::fallback_reason`].
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    fallback_reason: String,
    /// Catalog tables the live engine could not upload (catalog key -> reason), see
    /// [`BoltRuntime::table_gpu_status`] (W5-C-002).
    table_notes: BTreeMap<String, String>,
}

impl MetaSnapshot {
    fn resolve(&self, name: &str) -> Option<&TableSummary> {
        self.tables.get(name).or_else(|| {
            self.lower_index
                .get(&name.to_ascii_lowercase())
                .and_then(|key| self.tables.get(key))
        })
    }

    fn stored_batches(
        &self,
        key: Option<&str>,
        requested: &str,
    ) -> Result<(SchemaRef, Vec<RecordBatch>)> {
        key.and_then(|key| Some((self.tables.get(key)?, self.batches.get(key)?)))
            .map(|(summary, batches)| (summary.schema.clone(), batches.clone()))
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "Table '{requested}' is not registered in Craton Bolt"
                ))
            })
    }
}

enum WorkerCommand {
    UpdateConfig(BoltConfig, mpsc::SyncSender<Result<()>>),
    ResetEngine(mpsc::SyncSender<Result<()>>),
    RegisterTable {
        name: String,
        /// Normalized schema shared by every batch.
        schema: SchemaRef,
        /// Normalized, non-empty list of batches (an empty table is one zero-row batch).
        batches: Vec<RecordBatch>,
        mode: RegisterMode,
        reply: mpsc::SyncSender<Result<TableSummary>>,
    },
    DropTable {
        name: String,
        reply: mpsc::SyncSender<Result<bool>>,
    },
    ClearTables(mpsc::SyncSender<Result<usize>>),
    ExecuteSql {
        sql: String,
        /// Host-fallback results in DuckDB's exact types (`bolt_query`, W9-A-001).
        exact: bool,
        reply: mpsc::SyncSender<Result<SqlExecutionBatches>>,
    },
    ExecuteSqlHostFallback {
        sql: String,
        reply: mpsc::SyncSender<Result<SqlExecutionBatches>>,
    },
    ExplainSql {
        sql: String,
        reply: mpsc::SyncSender<Result<String>>,
    },
    InferSqlSchema {
        sql: String,
        reply: mpsc::SyncSender<Result<SchemaRef>>,
    },
    ExecutionTier {
        sql: String,
        reply: mpsc::SyncSender<Result<String>>,
    },
    Telemetry(mpsc::SyncSender<BoltTelemetrySnapshot>),
    BackendIsGpu(mpsc::SyncSender<bool>),
    VersionInfo(mpsc::SyncSender<String>),
    ExecuteWithScratch {
        batch: RecordBatch,
        sql: String,
        reply: mpsc::SyncSender<Result<SqlExecutionOutput>>,
    },
    InjectPanic {
        /// Also make the panic recovery panic (W2-E-006 test hook).
        in_recovery: bool,
        reply: mpsc::SyncSender<()>,
    },
}

#[cfg(not(target_arch = "wasm32"))]
impl WorkerCommand {
    /// Whether the command changes the catalog, the configuration or the engine. A mutation
    /// that has started is always waited for, whatever `bolt_request_timeout` says, so a
    /// statement never reports a failure for work that still takes effect (W4-D-003).
    fn is_mutation(&self) -> bool {
        matches!(
            self,
            Self::UpdateConfig(..)
                | Self::ResetEngine(_)
                | Self::RegisterTable { .. }
                | Self::DropTable { .. }
                | Self::ClearTables(_)
        )
    }
}

/// Lifecycle of one request, shared by the caller and the worker (W4-D-003).
#[cfg(not(target_arch = "wasm32"))]
mod request_state {
    /// Sent, not started yet.
    pub const QUEUED: u8 = 0;
    /// The worker is handling it.
    pub const RUNNING: u8 = 1;
    /// The caller gave up before it started: the worker skips it.
    pub const ABANDONED: u8 = 2;
    /// Handled.
    pub const DONE: u8 = 3;
}

/// A command plus its [`request_state`], as queued for the worker thread.
#[cfg(not(target_arch = "wasm32"))]
struct Envelope {
    state: Arc<AtomicU8>,
    cmd: WorkerCommand,
}

/// The one message for a Bolt entry point reached from inside a host-fallback query
/// (re-entrancy refusal, W3-E-004 / W4-D-004).
#[cfg(not(target_arch = "wasm32"))]
const REENTRANT_CALL_MESSAGE: &str = "Bolt functions, the bolt namespace and 'bolt:' table names \
     cannot be used inside a query that runs on the Craton Bolt host fallback";

/// How commands reach the [`BoltWorkerState`].
enum Transport {
    /// A dedicated OS thread owns the state (native builds; keeps the CUDA context current).
    #[cfg(not(target_arch = "wasm32"))]
    Thread(Mutex<Sender<Envelope>>),
    /// The worker thread could not be started; every request fails with this message instead
    /// of panicking (W2-E-001: a panic in a DuckDB callback invalidates the instance).
    #[cfg(not(target_arch = "wasm32"))]
    Unavailable(String),
    /// wasm32 has no threads (and no GPU): commands run on the caller's thread, serialized by
    /// the mutex (W2-E-001).
    #[cfg(target_arch = "wasm32")]
    Inline(Mutex<BoltWorkerState>),
}

/// Process-wide thread-safe handle to the Craton Bolt worker thread.
pub struct BoltRuntime {
    transport: Transport,
    meta: Arc<RwLock<MetaSnapshot>>,
    /// Serializes read-modify-write cycles of [`BoltRuntime::update_config`].
    config_update_lock: Mutex<()>,
    /// Per-request wait limit in milliseconds (`0` = wait forever), see
    /// [`BoltRuntime::set_request_timeout`].
    request_timeout_ms: AtomicU64,
    /// Interrupts the host-fallback DuckDB query in flight (W2-E-005).
    #[cfg(not(target_arch = "wasm32"))]
    fallback_interrupt: FallbackInterrupt,
}

/// Interrupt state of the host-fallback connection, shared by callers and the worker
/// (W2-E-005, W4-A-004).
///
/// A caller whose request timed out interrupts the fallback connection only while
/// `in_flight` is its own request, under this mutex. The worker sets `in_flight` before it runs
/// a request and clears it (under the same mutex) before the next one, so an interrupt can no
/// longer hit the next caller's query. Whether DuckDB keeps an interrupt issued after the
/// query finished for the connection's next query is not documented, so the worker absorbs it
/// (see [`BoltWorkerState::absorb_fallback_interrupt`]) when `interrupted` is set.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Default)]
struct InterruptSlot {
    /// Interrupt handle of the current fallback connection.
    handle: Option<duckdb_neo::connection::InterruptHandle>,
    /// [`request_state`] of the request the worker is running.
    in_flight: Option<Arc<AtomicU8>>,
    /// A caller interrupted the connection while `in_flight` ran.
    interrupted: bool,
}

/// Shared [`InterruptSlot`].
#[cfg(not(target_arch = "wasm32"))]
type FallbackInterrupt = Arc<Mutex<InterruptSlot>>;

/// Lock the interrupt slot (plain data: a poisoned lock still holds a consistent value).
#[cfg(not(target_arch = "wasm32"))]
fn lock_slot(slot: &FallbackInterrupt) -> std::sync::MutexGuard<'_, InterruptSlot> {
    slot.lock().unwrap_or_else(|p| p.into_inner())
}

static GLOBAL_RUNTIME: OnceLock<BoltRuntime> = OnceLock::new();

#[cfg(not(target_arch = "wasm32"))]
thread_local! {
    /// Set on the worker thread: a request from it would wait for itself (W3-E-004).
    static ON_BOLT_WORKER: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Run one command, keeping the worker usable whatever panics (W2-E-006).
///
/// A panic inside craton-bolt (or this crate) must not kill the worker: the reply channel of the
/// failing command is dropped during unwinding, so the caller receives an error. The recovery
/// itself drops the engine and the fallback database, whose `Drop` may panic too after a broken
/// operation, so it runs under its own `catch_unwind`, with a minimal second attempt.
fn run_worker_command(worker: &mut BoltWorkerState, cmd: WorkerCommand) {
    if catch_unwind(AssertUnwindSafe(|| worker.handle(cmd))).is_ok() {
        return;
    }
    if catch_unwind(AssertUnwindSafe(|| worker.recover_after_panic(true))).is_ok() {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| worker.recover_after_panic(false)));
}

impl BoltRuntime {
    /// Obtain the process-wide [`BoltRuntime`] singleton.
    ///
    /// Never panics: if the worker thread cannot be started, the runtime is still returned and
    /// every worker request fails with an ordinary error (W2-E-001).
    pub fn global() -> &'static Self {
        GLOBAL_RUNTIME.get_or_init(Self::spawn)
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn spawn() -> Self {
        let meta = Arc::new(RwLock::new(MetaSnapshot::default()));
        let fallback_interrupt: FallbackInterrupt = Arc::default();
        let mut last_error = String::from("no stack size tried");
        let mut transport = None;
        for stack_size in WORKER_STACK_SIZES {
            let (tx, rx) = mpsc::channel::<Envelope>();
            let worker_meta = meta.clone();
            let worker_interrupt = fallback_interrupt.clone();
            let spawned = thread::Builder::new()
                .name("duckdb-bolt-worker".to_string())
                .stack_size(stack_size)
                .spawn(move || {
                    ON_BOLT_WORKER.with(|flag| flag.set(true));
                    let mut worker = BoltWorkerState::new(worker_meta);
                    worker.fallback_interrupt = Some(worker_interrupt.clone());
                    while let Ok(Envelope { state, cmd }) = rx.recv() {
                        // A request its caller abandoned before it started is skipped (its
                        // reply channel is dropped with it).
                        if state
                            .compare_exchange(
                                request_state::QUEUED,
                                request_state::RUNNING,
                                Ordering::AcqRel,
                                Ordering::Acquire,
                            )
                            .is_ok()
                        {
                            {
                                let mut slot = lock_slot(&worker_interrupt);
                                slot.in_flight = Some(state.clone());
                                slot.interrupted = false;
                            }
                            run_worker_command(&mut worker, cmd);
                            // From here on no caller can interrupt this request (W4-A-004).
                            let interrupted = {
                                let mut slot = lock_slot(&worker_interrupt);
                                slot.in_flight = None;
                                std::mem::take(&mut slot.interrupted)
                            };
                            if interrupted {
                                let _ = catch_unwind(AssertUnwindSafe(|| {
                                    worker.absorb_fallback_interrupt()
                                }));
                            }
                            state.store(request_state::DONE, Ordering::Release);
                        }
                    }
                });
            match spawned {
                Ok(_) => {
                    transport = Some(Transport::Thread(Mutex::new(tx)));
                    break;
                }
                Err(e) => {
                    last_error = format!("{e} (stack size {} MiB)", stack_size >> 20);
                }
            }
        }
        let transport = transport.unwrap_or_else(|| {
            Transport::Unavailable(format!(
                "Craton Bolt worker thread could not be started: {last_error}"
            ))
        });
        Self {
            transport,
            meta,
            config_update_lock: Mutex::new(()),
            request_timeout_ms: AtomicU64::new(0),
            fallback_interrupt,
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn spawn() -> Self {
        let meta = Arc::new(RwLock::new(MetaSnapshot::default()));
        let state = BoltWorkerState::new(meta.clone());
        Self {
            transport: Transport::Inline(Mutex::new(state)),
            meta,
            config_update_lock: Mutex::new(()),
            request_timeout_ms: AtomicU64::new(0),
        }
    }

    /// Limit how long a caller waits for one worker request (W2-E-005); `None` (the default)
    /// waits until the request completes. The limit is process-wide (like every `bolt_*`
    /// setting) and is read when a request is sent.
    ///
    /// On timeout (W4-D-003):
    /// - a request that has not started yet (queued behind other callers' work) is abandoned:
    ///   the worker skips it, so it has no effect, and the caller gets an error saying so;
    /// - a running query (`bolt_query`, `bolt_scan` paths, `EXPLAIN`, ...) fails; if it is on the
    ///   host fallback, *its* DuckDB query is interrupted. A GPU query cannot be cancelled
    ///   (craton-bolt 0.9 has no cancellation hook) and finishes in the background;
    /// - a running mutation (register / drop / clear / configuration / reset) is waited for,
    ///   so a statement never fails while its change still takes effect.
    ///
    /// Other callers' requests are never interrupted. On wasm32 requests run inline and the
    /// limit is ignored.
    pub fn set_request_timeout(&self, timeout: Option<Duration>) {
        let ms = timeout.map_or(0, |t| {
            u64::try_from(t.as_millis()).unwrap_or(u64::MAX).max(1)
        });
        self.request_timeout_ms.store(ms, Ordering::Relaxed);
    }

    /// The current per-request wait limit (see [`BoltRuntime::set_request_timeout`]).
    pub fn request_timeout(&self) -> Option<Duration> {
        match self.request_timeout_ms.load(Ordering::Relaxed) {
            0 => None,
            ms => Some(Duration::from_millis(ms)),
        }
    }

    fn request<T>(&self, build: impl FnOnce(mpsc::SyncSender<T>) -> WorkerCommand) -> Result<T> {
        let (reply_tx, reply_rx) = mpsc::sync_channel(1);
        let aborted = || {
            Error::api_error(
                "Craton Bolt worker panicked while handling the request; the request was aborted \
                 and the GPU engine will be re-initialized on next use",
            )
        };
        match &self.transport {
            #[cfg(not(target_arch = "wasm32"))]
            Transport::Thread(sender) => {
                if ON_BOLT_WORKER.with(std::cell::Cell::get) {
                    // A Bolt entry point reached from inside a host-fallback query (static
                    // builds load the extension into the fallback instance too).
                    return Err(Error::invalid_input(REENTRANT_CALL_MESSAGE));
                }
                let cmd = build(reply_tx);
                let mutation = cmd.is_mutation();
                let timeout = self.request_timeout();
                let state = Arc::new(AtomicU8::new(request_state::QUEUED));
                sender
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .send(Envelope {
                        state: state.clone(),
                        cmd,
                    })
                    .map_err(|_| {
                        Error::api_error("BoltRuntime worker thread terminated unexpectedly")
                    })?;
                let Some(timeout) = timeout else {
                    return reply_rx.recv().map_err(|_| aborted());
                };
                match reply_rx.recv_timeout(timeout) {
                    Ok(value) => Ok(value),
                    Err(mpsc::RecvTimeoutError::Disconnected) => Err(aborted()),
                    Err(mpsc::RecvTimeoutError::Timeout) => {
                        let ms = timeout.as_millis();
                        if state
                            .compare_exchange(
                                request_state::QUEUED,
                                request_state::ABANDONED,
                                Ordering::AcqRel,
                                Ordering::Acquire,
                            )
                            .is_ok()
                        {
                            return Err(Error::api_error(format!(
                                "Craton Bolt request timed out after {ms} ms (PRAGMA \
                                 bolt_request_timeout) while waiting for earlier requests; it \
                                 was not started and had no effect"
                            )));
                        }
                        if mutation {
                            // Started: its change takes effect, so report its real outcome.
                            return reply_rx.recv().map_err(|_| aborted());
                        }
                        {
                            // Interrupt only while this very request runs (W4-A-004): the
                            // worker cannot move on to the next request while the slot is
                            // locked here.
                            let mut slot = lock_slot(&self.fallback_interrupt);
                            let ours = slot
                                .in_flight
                                .as_ref()
                                .is_some_and(|running| Arc::ptr_eq(running, &state));
                            if ours && let Some(handle) = slot.handle.as_ref() {
                                let _ = handle.interrupt();
                                slot.interrupted = true;
                            }
                        }
                        Err(Error::api_error(format!(
                            "Craton Bolt request timed out after {ms} ms (PRAGMA \
                             bolt_request_timeout); a host-fallback query was interrupted, a \
                             GPU query cannot be cancelled and finishes in the background"
                        )))
                    }
                }
            }
            #[cfg(not(target_arch = "wasm32"))]
            Transport::Unavailable(message) => {
                drop(build(reply_tx));
                Err(Error::api_error(message.clone()))
            }
            #[cfg(target_arch = "wasm32")]
            Transport::Inline(state) => {
                let cmd = build(reply_tx);
                let mut worker = state.lock().unwrap_or_else(|p| p.into_inner());
                run_worker_command(&mut worker, cmd);
                drop(worker);
                reply_rx.try_recv().map_err(|_| aborted())
            }
        }
    }

    fn read_meta(&self) -> RwLockReadGuard<'_, MetaSnapshot> {
        // The snapshot is plain data that is always replaced wholesale per entry, so a poisoned
        // lock still holds a consistent value.
        self.meta.read().unwrap_or_else(|p| p.into_inner())
    }

    /// Read the current [`BoltConfig`] (answered without a worker round trip).
    pub fn config(&self) -> Result<BoltConfig> {
        Ok(self.read_meta().config.clone())
    }

    /// Update the [`BoltConfig`] and mark the engine for reconfiguration.
    ///
    /// Concurrent updates are serialized, so two `SET bolt_*` statements issued from different
    /// connections at the same time cannot lose each other's change.
    pub fn update_config(&self, update_fn: impl FnOnce(&mut BoltConfig)) -> Result<()> {
        let _guard = self
            .config_update_lock
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let mut cfg = self.config()?;
        update_fn(&mut cfg);
        self.request(|reply| WorkerCommand::UpdateConfig(cfg, reply))?
    }

    /// Reset/rebuild the Craton Bolt engine using the current configuration.
    ///
    /// Also clears a cached engine-initialization failure, so the next GPU operation retries.
    pub fn reset_engine(&self) -> Result<()> {
        self.request(WorkerCommand::ResetEngine)?
    }

    /// Register, replace, or append a table in the Craton Bolt catalog.
    ///
    /// The name is normalized with [`normalize_table_reference`] (quotes and `bolt.` / `bolt:`
    /// prefixes only); `tmp_x` stays `tmp_x`.
    ///
    /// Column types are normalized to the subset Craton Bolt stores (see
    /// [`normalize_schema_for_bolt`]) on every backend, so a table registers identically with
    /// and without a GPU; values that do not fit the storage type are an error (W1-O-001).
    pub fn register_table(
        &self,
        name: impl Into<String>,
        batch: RecordBatch,
        mode: RegisterMode,
    ) -> Result<TableSummary> {
        let schema = batch.schema();
        self.register_table_batches(name, schema, vec![batch], mode)
    }

    /// Register a table from several batches without concatenating them.
    ///
    /// Every batch must have the columns of `schema` (by position); types are normalized like
    /// [`BoltRuntime::register_table`]. Zero-row batches are skipped, and an empty `batches`
    /// registers an empty table with `schema`. The registration is one worker command, so it is
    /// atomic with respect to other catalog operations: a failure leaves the catalog unchanged.
    pub fn register_table_batches(
        &self,
        name: impl Into<String>,
        schema: SchemaRef,
        batches: Vec<RecordBatch>,
        mode: RegisterMode,
    ) -> Result<TableSummary> {
        let name = normalize_table_name(&name.into())?;
        check_table_name_not_reserved(&name)?;
        // Normalize on the calling thread so the worker is not blocked by the casts.
        let target = normalize_schema_for_bolt(&schema)?;
        check_column_names(&name, &target)?;
        let mut normalized = Vec::with_capacity(batches.len().max(1));
        for batch in batches {
            if batch.num_columns() != target.fields().len() {
                return Err(Error::invalid_input(format!(
                    "Cannot register Bolt table '{name}': batch has {} columns, schema has {}",
                    batch.num_columns(),
                    target.fields().len()
                )));
            }
            if batch.num_rows() == 0 {
                continue;
            }
            normalized.push(normalize_batch_to(&batch, &target)?);
        }
        if normalized.is_empty() {
            normalized.push(RecordBatch::new_empty(target.clone()));
        }
        self.request(|reply| WorkerCommand::RegisterTable {
            name,
            schema: target,
            batches: normalized,
            mode,
            reply,
        })?
    }

    /// Cheap check whether the GPU engine is active.
    ///
    /// No device enumeration and no pool/cache statistics. If no engine exists yet and no
    /// initialization failure is cached, this builds the engine once (like the first GPU
    /// query would); a cached failure is honored, so GPU-less hosts answer immediately.
    pub fn backend_is_gpu(&self) -> Result<bool> {
        if let Some(gpu) = self.read_meta().engine.gpu {
            return Ok(gpu);
        }
        self.request(WorkerCommand::BackendIsGpu)
    }

    /// Drop a registered table from the Craton Bolt catalog.
    pub fn drop_table(&self, name: &str) -> Result<bool> {
        let name = normalize_table_name(name)?;
        self.request(|reply| WorkerCommand::DropTable { name, reply })?
    }

    /// Clear all registered tables from the Craton Bolt catalog.
    pub fn clear_tables(&self) -> Result<usize> {
        self.request(WorkerCommand::ClearTables)?
    }

    /// Look up summary metadata for a registered table (case-insensitive fallback).
    ///
    /// Answered from the published metadata snapshot (no worker round trip).
    pub fn table_info(&self, name: &str) -> Result<Option<TableSummary>> {
        let name = normalize_table_reference(name);
        if name.is_empty() {
            return Ok(None);
        }
        Ok(self.read_meta().resolve(&name).cloned())
    }

    /// Look up a table by its bare name: exact match, else ASCII case-insensitive match.
    ///
    /// Unlike [`BoltRuntime::table_info`] nothing is stripped (no quotes, no `bolt.` prefix),
    /// so it is safe for resolving unqualified identifiers in the replacement scan (W1-E-005).
    pub fn table_info_exact(&self, name: &str) -> Result<Option<TableSummary>> {
        if name.is_empty() {
            return Ok(None);
        }
        Ok(self.read_meta().resolve(name).cloned())
    }

    /// List all tables currently registered in the Craton Bolt catalog.
    pub fn list_tables(&self) -> Result<Vec<TableSummary>> {
        Ok(self.read_meta().tables.values().cloned().collect())
    }

    /// Retrieve the stored Arrow batches for a registered table.
    ///
    /// `name` is resolved as a table reference first; if no such table exists it is cleaned as a
    /// `COPY` path with [`clean_table_identifier`] (so `COPY t FROM 'dir/x.bolt' (FORMAT bolt)`
    /// keeps working). Prefer [`BoltRuntime::get_table_batches_strict`] for table references and
    /// [`BoltRuntime::get_table_batches_for_copy`] for COPY paths (W1-B-008).
    ///
    /// Answered from the published snapshot (the batches are `Arc`-backed), so it never queues
    /// behind a long GPU query (W1-B-004).
    pub fn get_table_batches(&self, name: &str) -> Result<(SchemaRef, Vec<RecordBatch>)> {
        self.get_table_batches_for_copy(name)
    }

    /// [`BoltRuntime::get_table_batches`] for a `COPY ... FROM '<path>'` source: a table
    /// reference first, then the path cleaned with [`clean_table_identifier`].
    pub fn get_table_batches_for_copy(&self, path: &str) -> Result<(SchemaRef, Vec<RecordBatch>)> {
        let meta = self.read_meta();
        let reference = normalize_table_reference(path);
        let key = meta
            .resolve(&reference)
            .or_else(|| meta.resolve(&clean_table_identifier(path)))
            .map(|s| s.name.clone());
        meta.stored_batches(key.as_deref(), &reference)
    }

    /// Retrieve the stored batches of the table `name` refers to ([`normalize_table_reference`]
    /// only: `a/x` never resolves to table `x`, W1-B-008).
    pub fn get_table_batches_strict(&self, name: &str) -> Result<(SchemaRef, Vec<RecordBatch>)> {
        let meta = self.read_meta();
        let reference = normalize_table_reference(name);
        let key = meta.resolve(&reference).map(|s| s.name.clone());
        meta.stored_batches(key.as_deref(), &reference)
    }

    /// Execute a SQL query on the Craton Bolt GPU engine.
    ///
    /// Without a usable GPU the query runs on the embedded host-fallback DuckDB (see
    /// [`BoltRuntime::execute_sql_host_fallback`]).
    ///
    /// The result is one batch; [`BoltRuntime::execute_sql_batches`] avoids concatenating a
    /// multi-batch (host-fallback) result.
    pub fn execute_sql(&self, sql: &str) -> Result<SqlExecutionOutput> {
        self.execute_sql_batches(sql)?.into_output()
    }

    /// Like [`BoltRuntime::execute_sql`], but returns the result in the batches it was produced
    /// in (W4-A-002).
    pub fn execute_sql_batches(&self, sql: &str) -> Result<SqlExecutionBatches> {
        self.request(|reply| WorkerCommand::ExecuteSql {
            sql: sql.to_string(),
            exact: false,
            reply,
        })?
    }

    /// Like [`BoltRuntime::execute_sql_batches`], but a result computed by the host fallback
    /// keeps DuckDB's exact result types (`HUGEINT` as `Decimal128(38, 0)`, unsigned integers
    /// as unsigned, `INTERVAL` as `Interval(MonthDayNano)`) instead of Bolt storage types. GPU
    /// results are unchanged (storage types). Used by `bolt_query`, which binds DuckDB's
    /// result types ([`BoltRuntime::infer_sql_schema`], W9-A-001).
    pub fn execute_query_batches(&self, sql: &str) -> Result<SqlExecutionBatches> {
        self.request(|reply| WorkerCommand::ExecuteSql {
            sql: sql.to_string(),
            exact: true,
            reply,
        })?
    }

    /// Execute a SQL query on the embedded host-fallback DuckDB.
    ///
    /// The fallback instance runs with `enable_external_access = false` and extension
    /// auto-loading disabled, accepts a single `SELECT` statement only, and sees the Bolt
    /// catalog tables referenced by the query. Queries Craton Bolt's planner cannot plan are
    /// still executed (reported tier: `"Host"`).
    pub fn execute_sql_host_fallback(&self, sql: &str) -> Result<SqlExecutionOutput> {
        self.execute_sql_host_fallback_batches(sql)?.into_output()
    }

    /// Like [`BoltRuntime::execute_sql_host_fallback`], but returns the result in DuckDB's
    /// Arrow stream batches (W4-A-002).
    pub fn execute_sql_host_fallback_batches(&self, sql: &str) -> Result<SqlExecutionBatches> {
        self.request(|reply| WorkerCommand::ExecuteSqlHostFallback {
            sql: sql.to_string(),
            reply,
        })?
    }

    /// Explain a SQL query using Craton Bolt's logical and physical planner.
    pub fn explain_sql(&self, sql: &str) -> Result<String> {
        self.request(|reply| WorkerCommand::ExplainSql {
            sql: sql.to_string(),
            reply,
        })?
    }

    /// Infer the output Arrow schema of a SQL query by planning it only (no execution).
    ///
    /// Returns an error when the planner cannot statically resolve the query (for
    /// example host-side special paths); callers should then fall back to executing it.
    pub fn infer_sql_schema(&self, sql: &str) -> Result<SchemaRef> {
        self.request(|reply| WorkerCommand::InferSqlSchema {
            sql: sql.to_string(),
            reply,
        })?
    }

    /// Determine the planned execution tier (`"Gpu"`, `"Host"`, or `"Hybrid"`) for a SQL query.
    ///
    /// Never executes the query.
    pub fn execution_tier(&self, sql: &str) -> Result<String> {
        self.request(|reply| WorkerCommand::ExecutionTier {
            sql: sql.to_string(),
            reply,
        })?
    }

    /// Collect pool, cache, and runtime telemetry.
    ///
    /// Answered on the calling thread from the published engine status once the backend is
    /// known (W1-B-004), so `bolt_scan` filter pushdown and `bolt_pool_stats()` do not queue
    /// behind a long GPU query. Only the first call after a configuration change (or
    /// `PRAGMA bolt_reset`) goes through the worker, which may initialize CUDA to find out.
    pub fn telemetry(&self) -> Result<BoltTelemetrySnapshot> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let meta = self.read_meta();
            if let Some(gpu) = meta.engine.gpu {
                return Ok(native_telemetry(
                    &meta.config,
                    meta.tables.len(),
                    gpu,
                    meta.engine.last_error.as_deref(),
                    meta.engine.fallback_reason.clone(),
                ));
            }
        }
        self.request(WorkerCommand::Telemetry)
    }

    /// Which catalog tables the GPU engine holds, from the published engine status (never
    /// queues behind the worker, never initializes CUDA; W5-C-002).
    ///
    /// Returns whether the backend is the GPU engine (`None` while unknown: no engine build
    /// was attempted since the last configuration change or `PRAGMA bolt_reset`) and, per
    /// catalog key, why the live engine does not hold that table (over `bolt_memory_budget`,
    /// an upload error). The notes describe the last engine build and the uploads since; they
    /// are kept while an engine dropped for a lazy rebuild waits to be rebuilt (W8-A-001).
    pub fn table_gpu_status(&self) -> (Option<bool>, BTreeMap<String, String>) {
        let meta = self.read_meta();
        (meta.engine.gpu, meta.engine.table_notes.clone())
    }

    /// Collect execution metrics rows and Prometheus exposition text.
    ///
    /// Reads craton-bolt's process-global counters on the calling thread (W1-B-004).
    pub fn metrics(&self) -> Result<(Vec<MetricRow>, String)> {
        Ok(collect_metrics())
    }

    /// Enumerate available CUDA devices (driver queries on the calling thread, W1-B-004).
    pub fn devices(&self) -> Result<Vec<DeviceInfo>> {
        Ok(cuda_devices(self.read_meta().config.device))
    }

    /// Version and backend status string. Does not initialize the GPU engine.
    pub fn version_info(&self) -> Result<String> {
        self.request(WorkerCommand::VersionInfo)
    }

    /// Run `sql` on the GPU engine with `batch` available as table [`SCRATCH_TABLE_NAME`].
    ///
    /// The scratch table lives only in the engine: it is not added to the Bolt catalog (so it
    /// never shows up in `bolt_tables()` or the host fallback), it is uploaded and released in
    /// one worker round trip, and reusing the fixed name avoids registering/dropping a fresh
    /// table per call (the hybrid filter-scan pattern, review 1.5.1). Errors when no GPU
    /// engine is available; callers should then use a host path.
    pub fn execute_sql_with_scratch_table(
        &self,
        batch: RecordBatch,
        sql: &str,
    ) -> Result<SqlExecutionOutput> {
        self.request(|reply| WorkerCommand::ExecuteWithScratch {
            batch,
            sql: sql.to_string(),
            reply,
        })?
    }

    /// Test hook: make the worker panic while handling a command. Returns the error the caller
    /// observes; the worker must keep serving later requests.
    #[doc(hidden)]
    pub fn inject_worker_panic_for_testing(&self) -> Result<()> {
        self.request(|reply| WorkerCommand::InjectPanic {
            in_recovery: false,
            reply,
        })
    }

    /// Test hook: like [`BoltRuntime::inject_worker_panic_for_testing`], and the panic recovery
    /// panics too (W2-E-006). The worker must still keep serving later requests.
    #[doc(hidden)]
    pub fn inject_worker_panic_in_recovery_for_testing(&self) -> Result<()> {
        self.request(|reply| WorkerCommand::InjectPanic {
            in_recovery: true,
            reply,
        })
    }
}

/// Reject names in the reserved [`RESERVED_NAME_PREFIX`] namespace (W2-E-015).
fn check_table_name_not_reserved(name: &str) -> Result<()> {
    let reserved = name
        .get(..RESERVED_NAME_PREFIX.len())
        .is_some_and(|head| head.eq_ignore_ascii_case(RESERVED_NAME_PREFIX));
    if reserved {
        return Err(Error::invalid_input(format!(
            "Table name '{name}' is reserved by Craton Bolt (names starting with \
             '{RESERVED_NAME_PREFIX}' are internal)"
        )));
    }
    Ok(())
}

/// Reject empty and duplicate column names: SQL cannot address them, and the host fallback
/// (DuckDB) cannot even create such a table, so it would only work on some backends (W2-E-008).
/// Names that differ only in case stay allowed (`bolt_scan` exposes them as `ID_1`, W2-B).
fn check_column_names(table: &str, schema: &Schema) -> Result<()> {
    let mut seen = HashSet::with_capacity(schema.fields().len());
    for field in schema.fields() {
        let name = field.name();
        if name.trim().is_empty() {
            return Err(Error::invalid_input(format!(
                "Cannot register Bolt table '{table}': column names must not be empty"
            )));
        }
        if !seen.insert(name.as_str()) {
            return Err(Error::invalid_input(format!(
                "Cannot register Bolt table '{table}': duplicate column name '{name}' (rename \
                 the columns, e.g. with AS)"
            )));
        }
    }
    Ok(())
}

fn strip_quotes(raw: &str) -> &str {
    raw.trim().trim_matches('"').trim_matches('\'').trim()
}

fn strip_bolt_prefix(raw: &str) -> &str {
    for prefix in ["bolt://", "bolt:", "bolt.main.", "bolt."] {
        if let Some(head) = raw.get(..prefix.len())
            && head.eq_ignore_ascii_case(prefix)
        {
            return &raw[prefix.len()..];
        }
    }
    raw
}

/// Normalize a table reference (`bolt.orders`, `'bolt:orders'`, `"orders"`) into a catalog name.
///
/// Strips surrounding quotes and a leading `bolt://`, `bolt:`, `bolt.main.`, or `bolt.` prefix
/// (ASCII case-insensitive) only. Unlike [`clean_table_identifier`] it keeps `tmp_` prefixes,
/// `.bolt` suffixes, and `/` characters, so `tmp_x` and `x` are distinct tables.
pub fn normalize_table_reference(raw: &str) -> String {
    strip_quotes(strip_bolt_prefix(strip_quotes(raw))).to_string()
}

/// Clean a `COPY` target path (`bolt.orders`, `dir/tmp_orders.bolt`, etc.) into a bare name.
///
/// Strips quotes, `bolt.`/`bolt:` prefixes, directory parts, the `tmp_` prefix DuckDB adds for
/// `USE_TMP_FILE`, and a `.bolt` suffix. Use [`normalize_table_reference`] for table names.
pub fn clean_table_identifier(raw: &str) -> String {
    let trimmed = raw.trim().trim_matches('"').trim_matches('\'');
    // ASCII case-insensitive, like table lookup (W1-A-007).
    let without_prefix = strip_bolt_prefix(trimmed);
    let base = without_prefix
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or(without_prefix);
    let base = base.strip_prefix("tmp_").unwrap_or(base);
    let base = base.strip_suffix(".bolt").unwrap_or(base);
    base.trim().to_string()
}

fn normalize_table_name(raw: &str) -> Result<String> {
    let cleaned = normalize_table_reference(raw);
    if cleaned.is_empty() {
        return Err(Error::invalid_input("Bolt table name cannot be empty"));
    }
    Ok(cleaned)
}

// ------------------------------------------------------------------- type normalization
//
// The single source of truth for the Arrow types the Bolt catalog stores. Every registration
// path (`register_table`, `register_table_batches`, COPY, `bolt_load_*`) goes through
// `normalize_schema_for_bolt` / `normalize_batch_to`, on every backend (W1-O-001). The type
// mapping is [`crate::arrow_bridge::bolt_storage_arrow_type`], shared with
// `ingest::normalized_schema` (W5-C-001): zoned timestamps are stored as naive UTC instants
// (microseconds, or nanoseconds for `ns`), like COPY stores `TIMESTAMPTZ`, and
// [`normalize_schema_for_bolt`] records `TIMESTAMPTZ` / `TIMESTAMP_TZ_NS` in the RFC-008
// metadata when the field has none.

/// The DuckDB type of a zoned Arrow timestamp (`TIMESTAMP_TZ_NS` for nanoseconds,
/// `TIMESTAMPTZ` otherwise), or `None` for any other type.
fn zoned_timestamp_original(dt: &ArrowDataType) -> Option<crate::arrow_bridge::OriginalDuckDbType> {
    use crate::arrow_bridge::OriginalDuckDbType as O;
    match dt {
        ArrowDataType::Timestamp(arrow_schema::TimeUnit::Nanosecond, Some(_)) => {
            Some(O::TimestampTzNs)
        }
        ArrowDataType::Timestamp(_, Some(_)) => Some(O::TimestampTz),
        ArrowDataType::Dictionary(_, value) => zoned_timestamp_original(value),
        _ => None,
    }
}

/// Compute the Bolt storage schema for `schema` (field names, nullability, and metadata are
/// kept), failing on the first unsupported column or an empty schema.
pub fn normalize_schema_for_bolt(schema: &Schema) -> Result<SchemaRef> {
    if schema.fields().is_empty() {
        return Err(Error::invalid_input(
            "Craton Bolt tables need at least one column",
        ));
    }
    let mut changed = false;
    let fields = schema
        .fields()
        .iter()
        .map(|field| {
            let target = crate::arrow_bridge::bolt_storage_arrow_type(field.data_type())
                .ok_or_else(|| {
                    Error::invalid_input(format!(
                        "Column '{}' has Arrow type {:?}, which Craton Bolt does not support",
                        field.name(),
                        field.data_type()
                    ))
                })?;
            if &target == field.data_type() {
                Ok(field.as_ref().clone())
            } else {
                changed = true;
                // Keep name, nullability, and metadata (e.g. `duckdb.logical_type`, RFC-008).
                let zoned = zoned_timestamp_original(field.data_type());
                let mut field = field.as_ref().clone().with_data_type(target);
                // A zoned timestamp loses its zone in storage; record the DuckDB type so
                // `bolt_scan` and the planner still see `TIMESTAMPTZ` (W5-C-001). A key that is
                // already present (COPY, ingest) is kept.
                if let Some(original) = zoned
                    && !field
                        .metadata()
                        .contains_key(crate::arrow_bridge::DUCKDB_TYPE_METADATA_KEY)
                    && &original.storage_arrow_type() == field.data_type()
                {
                    let mut metadata = field.metadata().clone();
                    metadata.insert(
                        crate::arrow_bridge::DUCKDB_TYPE_METADATA_KEY.to_string(),
                        original.to_metadata_text(),
                    );
                    field = field.with_metadata(metadata);
                }
                Ok(field)
            }
        })
        .collect::<Result<Vec<_>>>()?;
    if !changed {
        return Ok(Arc::new(schema.clone()));
    }
    Ok(Arc::new(
        Schema::new(fields).with_metadata(schema.metadata().clone()),
    ))
}

/// Normalize every column of `batch` to the Bolt storage type.
///
/// Lossy narrowing (for example `UInt64` values above `i64::MAX`) is an error, never NULL.
pub fn normalize_batch_for_bolt(batch: &RecordBatch) -> Result<RecordBatch> {
    let target = normalize_schema_for_bolt(batch.schema_ref())?;
    normalize_batch_to(batch, &target)
}

fn checked_cast_options() -> CastOptions<'static> {
    CastOptions {
        safe: false,
        ..Default::default()
    }
}

/// Cast `batch` (by position) to `target`; values that do not fit are an error.
fn normalize_batch_to(batch: &RecordBatch, target: &SchemaRef) -> Result<RecordBatch> {
    if batch.num_columns() != target.fields().len() {
        return Err(Error::invalid_input(format!(
            "Batch has {} columns, expected {}",
            batch.num_columns(),
            target.fields().len()
        )));
    }
    if batch.schema_ref() == target {
        return Ok(batch.clone());
    }
    let options = checked_cast_options();
    let mut columns: Vec<ArrayRef> = Vec::with_capacity(batch.num_columns());
    for (dst_field, col) in target.fields().iter().zip(batch.columns()) {
        let wanted = dst_field.data_type();
        let out = if col.data_type() == wanted {
            col.clone()
        } else {
            cast_with_options(col, wanted, &options).map_err(|e| {
                Error::invalid_input(format!(
                    "Column '{}' cannot be converted from {:?} to {:?}: {e}",
                    dst_field.name(),
                    col.data_type(),
                    wanted
                ))
            })?
        };
        columns.push(out);
    }
    RecordBatch::try_new(target.clone(), columns)
        .map_err(|e| Error::invalid_input(format!("Failed to build normalized batch: {e}")))
}

struct BoltWorkerState {
    config: BoltConfig,
    meta: Arc<RwLock<MetaSnapshot>>,
    catalog: BTreeMap<String, RegisteredTable>,
    /// ASCII-lowercased name -> catalog key (O(1) case-insensitive lookup).
    lower_index: HashMap<String, String>,
    next_generation: u64,
    #[cfg(not(target_arch = "wasm32"))]
    engine: Option<craton_bolt::Engine>,
    #[cfg(not(target_arch = "wasm32"))]
    last_engine_error: Option<String>,
    /// Set when building the engine failed; cleared on config change or `ResetEngine`, so a
    /// GPU-less host does not retry CUDA initialization on every call.
    #[cfg(not(target_arch = "wasm32"))]
    engine_init_failed: bool,
    /// Tables dropped from the catalog that are still registered in the engine as empty
    /// placeholders (craton-bolt 0.9 has no per-table unregister API).
    #[cfg(not(target_arch = "wasm32"))]
    engine_tombstones: std::collections::HashSet<String>,
    /// Whether [`SCRATCH_TABLE_NAME`] is registered in the live engine.
    #[cfg(not(target_arch = "wasm32"))]
    engine_has_scratch: bool,
    /// Catalog tables the live engine could not upload (catalog key -> reason). Queries that
    /// may reference one run on the host fallback; the reasons are reported in
    /// `BoltTelemetrySnapshot::fallback_reason` (W1-E-003). Kept across a lazy engine drop
    /// (W8-A-001), recomputed by every engine build, cleared by `invalidate_engine`.
    #[cfg(not(target_arch = "wasm32"))]
    engine_upload_errors: BTreeMap<String, String>,
    #[cfg(not(target_arch = "wasm32"))]
    fallback_db: Option<FallbackDb>,
    /// Binds `bolt_query` SQL in DuckDB for its result types (W9-A-001); opened lazily.
    #[cfg(not(target_arch = "wasm32"))]
    type_binder: Option<TypeBinder>,
    /// Where the interrupt handle of the fallback connection is published (W2-E-005).
    #[cfg(not(target_arch = "wasm32"))]
    fallback_interrupt: Option<FallbackInterrupt>,
    /// CUDA ordinal of the first successful engine build. craton-bolt 0.9 latches the process
    /// to that device (`cudarc_backend::ensure_device`), so a later build on another ordinal
    /// would silently keep using it (W2-E-010).
    #[cfg(not(target_arch = "wasm32"))]
    latched_device: Option<i32>,
    /// Whether an engine was built successfully for the current configuration (kept when the
    /// engine is only dropped for a lazy rebuild; cleared on configuration change / reset).
    #[cfg(not(target_arch = "wasm32"))]
    engine_built_ok: bool,
    /// Why the configured `bolt_ptx_cache_dir` is not used by the current engine (it failed
    /// the use-time check, W3-D-004); shown by `bolt_version()`.
    #[cfg(not(target_arch = "wasm32"))]
    ptx_cache_note: Option<String>,
    /// `log::max_level()` before an engine was built with tracing; restored when
    /// `bolt_enable_tracing` is turned off again (W2-E-009).
    #[cfg(not(target_arch = "wasm32"))]
    log_level_before_tracing: Option<log::LevelFilter>,
    /// Test hook: make the next panic recovery panic (W2-E-006).
    panic_in_recovery: bool,
}

impl BoltWorkerState {
    fn new(meta: Arc<RwLock<MetaSnapshot>>) -> Self {
        Self {
            config: BoltConfig::default(),
            meta,
            catalog: BTreeMap::new(),
            lower_index: HashMap::new(),
            next_generation: 1,
            #[cfg(not(target_arch = "wasm32"))]
            engine: None,
            #[cfg(not(target_arch = "wasm32"))]
            last_engine_error: None,
            #[cfg(not(target_arch = "wasm32"))]
            engine_init_failed: false,
            #[cfg(not(target_arch = "wasm32"))]
            engine_tombstones: std::collections::HashSet::new(),
            #[cfg(not(target_arch = "wasm32"))]
            engine_has_scratch: false,
            #[cfg(not(target_arch = "wasm32"))]
            engine_upload_errors: BTreeMap::new(),
            #[cfg(not(target_arch = "wasm32"))]
            fallback_db: None,
            #[cfg(not(target_arch = "wasm32"))]
            type_binder: None,
            #[cfg(not(target_arch = "wasm32"))]
            fallback_interrupt: None,
            #[cfg(not(target_arch = "wasm32"))]
            latched_device: None,
            #[cfg(not(target_arch = "wasm32"))]
            engine_built_ok: false,
            #[cfg(not(target_arch = "wasm32"))]
            ptx_cache_note: None,
            #[cfg(not(target_arch = "wasm32"))]
            log_level_before_tracing: None,
            panic_in_recovery: false,
        }
    }

    // ---------------------------------------------------------------- catalog & metadata

    fn publish(&self, update: impl FnOnce(&mut MetaSnapshot)) {
        let mut meta = self.meta.write().unwrap_or_else(|p| p.into_inner());
        update(&mut meta);
    }

    /// Publish the engine status (W1-B-004). Called before every reply, so a caller that got
    /// its answer never reads an older status.
    fn publish_engine_status(&self) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let status = EngineStatus {
                // A failed rebuild after a lazy drop reports `false`, like a failed first
                // build: queries run on the host until a configuration change or reset.
                gpu: if self.engine.is_some() {
                    Some(true)
                } else if self.engine_init_failed {
                    Some(false)
                } else if self.engine_built_ok {
                    Some(true)
                } else {
                    None
                },
                last_error: self.last_engine_error.clone(),
                fallback_reason: self.fallback_reason(),
                table_notes: self.engine_upload_errors.clone(),
            };
            self.publish(|m| m.engine = status);
        }
    }

    /// Send a command's reply after publishing the engine status.
    fn reply<T>(&self, reply: mpsc::SyncSender<T>, value: T) {
        self.publish_engine_status();
        let _ = reply.send(value);
    }

    /// Share the engine's coalesced copy of `key` with the catalog (W3-A-001).
    ///
    /// craton-bolt keeps the one batch it was given per upload on the host. When that batch
    /// was concatenated from several catalog batches, the catalog's batches covering the same
    /// rows (the last ones, whose row counts add up to `combined`) are replaced by zero-copy
    /// slices of it, so host memory holds one copy instead of two. Batch boundaries, data,
    /// `bytes_in_memory` and the generation are unchanged.
    #[cfg(not(target_arch = "wasm32"))]
    fn share_engine_copy(&mut self, key: &str, combined: RecordBatch) {
        let Some(entry) = self.catalog.get_mut(key) else {
            return;
        };
        let rows = combined.num_rows();
        let mut start = entry.batches.len();
        let mut covered = 0usize;
        while start > 0 && covered < rows {
            start -= 1;
            covered += entry.batches[start].num_rows();
        }
        if covered != rows || rows == 0 || *combined.schema() != *entry.schema {
            return;
        }
        let mut offset = 0;
        for batch in &mut entry.batches[start..] {
            let len = batch.num_rows();
            *batch = combined.slice(offset, len);
            offset += len;
        }
        let batches = entry.batches.clone();
        self.publish(|m| {
            m.batches.insert(key.to_string(), batches);
        });
    }

    fn resolve_table_key(&self, name: &str) -> Option<String> {
        if self.catalog.contains_key(name) {
            return Some(name.to_string());
        }
        self.lower_index.get(&name.to_ascii_lowercase()).cloned()
    }

    fn bump_generation(&mut self) -> u64 {
        let generation = self.next_generation;
        self.next_generation += 1;
        generation
    }

    fn catalog_insert(&mut self, key: String, mut entry: RegisteredTable) -> TableSummary {
        entry.generation = self.bump_generation();
        let summary = entry.summary(&key);
        let lower = key.to_ascii_lowercase();
        self.lower_index.insert(lower.clone(), key.clone());
        let batches = entry.batches.clone();
        self.catalog.insert(key.clone(), entry);
        self.publish(|m| {
            m.lower_index.insert(lower, key.clone());
            m.batches.insert(key.clone(), batches);
            m.tables.insert(key, summary.clone());
        });
        summary
    }

    fn catalog_append(&mut self, key: &str, batches: Vec<RecordBatch>) -> TableSummary {
        let generation = self.bump_generation();
        let entry = self.catalog.get_mut(key).expect("append target exists");
        for batch in batches {
            entry.append_batch(batch);
        }
        entry.generation = generation;
        let summary = entry.summary(key);
        let batches = entry.batches.clone();
        self.publish(|m| {
            m.batches.insert(key.to_string(), batches);
            m.tables.insert(key.to_string(), summary.clone());
        });
        summary
    }

    fn catalog_remove(&mut self, key: &str) -> Option<RegisteredTable> {
        let entry = self.catalog.remove(key)?;
        let lower = key.to_ascii_lowercase();
        self.lower_index.remove(&lower);
        self.publish(|m| {
            m.tables.remove(key);
            m.batches.remove(key);
            m.lower_index.remove(&lower);
        });
        Some(entry)
    }

    // ---------------------------------------------------------------- engine lifecycle

    /// Build an engine from the current configuration. On success also returns why the
    /// configured `bolt_ptx_cache_dir` is not used, if it failed its use-time check.
    #[cfg(not(target_arch = "wasm32"))]
    fn build_engine_from_config(
        &self,
    ) -> std::result::Result<(craton_bolt::Engine, Option<String>), String> {
        let device = self.config.device;
        let devices = cuda_devices(device);
        // W1-B-001: an ordinal that does not exist is a misconfiguration; say so clearly
        // instead of surfacing the driver's error. (No device at all, e.g. `cuda-stub`, is
        // left to the build, which reports why CUDA is unavailable.)
        if !devices.is_empty() && !devices.iter().any(|d| d.device_id == device) {
            return Err(format!(
                "bolt_device = {device} does not exist: {} CUDA device(s) visible (ordinals \
                 0..={}); SET bolt_device to one of them",
                devices.len(),
                devices.len() - 1
            ));
        }
        if let Some(latched) = self.latched_device
            && latched != device
        {
            return Err(format!(
                "bolt_device = {device} cannot take effect: craton-bolt 0.9 binds the process to \
                 the first CUDA device it used (device {latched}) and cannot switch GPUs; SET \
                 bolt_device = {latched} or restart the process"
            ));
        }
        let mut builder = craton_bolt::Engine::builder().device(device);
        if let Some(budget) = self.config.memory_budget.filter(|&b| b > 0) {
            builder = builder.memory_budget(budget);
        }
        // W3-D-004: re-check the directory at use time (it may have become unsafe since
        // `SET`); if it fails, build without it and report why in `bolt_version()`.
        let mut ptx_note = None;
        let cache_dir = self
            .config
            .ptx_cache_dir
            .as_ref()
            .filter(|dir| !dir.as_os_str().is_empty())
            .filter(
                |dir| match crate::options::check_ptx_cache_dir_in_use(dir) {
                    Ok(()) => true,
                    Err(e) => {
                        ptx_note = Some(format!(
                            "bolt_ptx_cache_dir '{}' is not used: {}",
                            dir.display(),
                            e.message
                        ));
                        false
                    }
                },
            );
        match cache_dir {
            Some(cache_dir) => builder = builder.persistent_cache(cache_dir.clone()),
            None => {
                // craton-bolt only ever *installs* its process-wide cache override, so a
                // directory set earlier would stay active after `SET bolt_ptx_cache_dir = ''`
                // (W2-E-009): clear it explicitly.
                craton_bolt::jit::set_disk_ptx_cache_dir(None);
                // Without an override craton-bolt falls back to `BOLT_PTX_CACHE_DIR`. Apply
                // the checks `SET bolt_ptx_cache_dir` applies (absolute, private directory).
                if let Ok(raw) = std::env::var(craton_bolt::jit::disk_cache::DISK_PTX_CACHE_ENV)
                    && !raw.trim().is_empty()
                {
                    crate::options::validate_ptx_cache_dir(&raw).map_err(|e| {
                        format!(
                            "the {} environment variable is rejected: {} (unset it, or SET \
                             bolt_ptx_cache_dir to an allowed directory)",
                            craton_bolt::jit::disk_cache::DISK_PTX_CACHE_ENV,
                            e.message
                        )
                    })?;
                }
            }
        }
        if self.config.enable_tracing {
            builder = builder.enable_tracing();
        }
        builder
            .build()
            .map(|engine| (engine, ptx_note))
            .map_err(|e| e.to_string())
    }

    /// Keep `log::max_level()` in line with `bolt_enable_tracing` (W2-E-009).
    ///
    /// craton-bolt's `enable_tracing` raises the process-global `log` level to `Info` and
    /// never lowers it. The level seen before the first tracing build is remembered here and
    /// restored once tracing is turned off, so `SET bolt_enable_tracing = false` takes
    /// effect. Called before every build and on every configuration change.
    #[cfg(not(target_arch = "wasm32"))]
    fn sync_log_level(&mut self) {
        if self.config.enable_tracing {
            self.log_level_before_tracing
                .get_or_insert_with(log::max_level);
        } else if let Some(previous) = self.log_level_before_tracing.take() {
            log::set_max_level(previous);
        }
    }

    /// `bolt_memory_budget` admission for engine uploads (W2-E-011).
    ///
    /// craton-bolt 0.9 does not cap its memory pool with the budget (it only uses it to split
    /// streaming scans), so the extension enforces it for the tables it uploads: a table whose
    /// upload would bring the engine-resident tables (Arrow bytes) above the budget is kept on
    /// the host and served by the fallback. Query intermediates are not capped.
    #[cfg(not(target_arch = "wasm32"))]
    fn check_engine_budget(&self, key: &str, bytes: usize) -> Result<()> {
        let Some(budget) = self.config.memory_budget.filter(|&b| b > 0) else {
            return Ok(());
        };
        let resident: usize = self
            .catalog
            .iter()
            .filter(|(name, _)| {
                name.as_str() != key && !self.engine_upload_errors.contains_key(*name)
            })
            .map(|(_, t)| t.bytes_in_memory)
            .sum();
        if resident.saturating_add(bytes) > budget {
            return Err(Error::invalid_input(format!(
                "Bolt table '{key}' ({bytes} bytes) is not uploaded to the GPU engine: \
                 bolt_memory_budget is {budget} bytes and {resident} bytes are already \
                 uploaded; the table is served by the host fallback"
            )));
        }
        Ok(())
    }

    /// Drop the engine (if any) together with the bookkeeping tied to it; the next GPU
    /// operation rebuilds it lazily (unless a build failure is cached).
    ///
    /// `engine_upload_errors` is kept (W8-A-001): it describes which catalog tables the engine
    /// keeps on the host for the current configuration (`bolt_memory_budget`, string capacity,
    /// failed uploads), which `bolt_tables()` and `bolt_pool_stats()` keep reporting while the
    /// engine waits for its lazy rebuild; [`Self::build_engine`] recomputes it, and
    /// [`Self::invalidate_engine`] clears it.
    #[cfg(not(target_arch = "wasm32"))]
    fn drop_engine(&mut self) {
        self.engine = None;
        self.engine_tombstones.clear();
        self.engine_has_scratch = false;
    }

    /// Drop the engine (if any) and forget a cached init failure and the per-table upload
    /// notes, so the next GPU operation builds a fresh engine. Used on config change,
    /// `ResetEngine`, and panic recovery.
    #[cfg(not(target_arch = "wasm32"))]
    fn invalidate_engine(&mut self) {
        self.drop_engine();
        self.engine_upload_errors.clear();
        self.engine_init_failed = false;
        self.engine_built_ok = false;
    }

    /// Build the engine from the current config and register every catalog table.
    ///
    /// Records (caches) the failure when the engine itself cannot be built. A table the engine
    /// cannot upload does not fail the build: it is recorded in `engine_upload_errors` and
    /// served by the host fallback (W1-E-003).
    #[cfg(not(target_arch = "wasm32"))]
    fn build_engine(&mut self) -> Result<()> {
        // Drop any existing engine first so its ACTIVE_ENGINE permit is released.
        self.drop_engine();
        self.sync_log_level();
        match self.build_engine_from_config() {
            Ok((mut eng, ptx_note)) => {
                self.latched_device.get_or_insert(self.config.device);
                let budget = self.config.memory_budget.filter(|&b| b > 0);
                let mut resident = 0usize;
                let mut upload_errors = BTreeMap::new();
                let mut shared = Vec::new();
                for (name, table) in &self.catalog {
                    if let Some(budget) = budget
                        && resident.saturating_add(table.bytes_in_memory) > budget
                    {
                        upload_errors.insert(
                            name.clone(),
                            format!(
                                "Bolt table '{name}' ({} bytes) is not uploaded to the GPU \
                                 engine: bolt_memory_budget is {budget} bytes and {resident} \
                                 bytes are already uploaded; the table is served by the host \
                                 fallback",
                                table.bytes_in_memory
                            ),
                        );
                        continue;
                    }
                    match engine_upload(&mut eng, name, &table.batches, false) {
                        Ok(copy) => {
                            resident = resident.saturating_add(table.bytes_in_memory);
                            shared.extend(copy.map(|batch| (name.clone(), batch)));
                        }
                        Err(e) => {
                            upload_errors.insert(name.clone(), e.to_string());
                        }
                    }
                }
                self.engine = Some(eng);
                self.engine_upload_errors = upload_errors;
                self.engine_init_failed = false;
                self.engine_built_ok = true;
                self.last_engine_error = None;
                self.ptx_cache_note = ptx_note;
                for (name, batch) in shared {
                    self.share_engine_copy(&name, batch);
                }
                Ok(())
            }
            Err(msg) => {
                let e = Error::api_error(format!(
                    "Craton Bolt GPU engine initialization failed: {msg}"
                ));
                self.engine_init_failed = true;
                self.last_engine_error = Some(e.to_string());
                // No table is on the GPU: the build error is the engine-wide reason, as for a
                // first build that fails (W8-A-001 keeps the notes only across a lazy drop).
                self.engine_upload_errors.clear();
                Err(e)
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn ensure_engine(&mut self) -> Result<&mut craton_bolt::Engine> {
        if self.engine.is_none() {
            if self.engine_init_failed {
                return Err(Error::api_error(format!(
                    "Craton Bolt GPU engine unavailable (cached failure; SET a bolt_* option or \
                     PRAGMA bolt_reset to retry): {}",
                    self.last_engine_error.as_deref().unwrap_or("unknown error")
                )));
            }
            self.build_engine()?;
        }
        Ok(self.engine.as_mut().expect("engine initialized above"))
    }

    /// Before running `sql` on the engine: if it may reference a dropped table that is still an
    /// empty placeholder in the engine, rebuild the engine so the query sees "unknown table"
    /// instead of an empty result.
    #[cfg(not(target_arch = "wasm32"))]
    fn purge_tombstones_referenced_by(&mut self, sql: &str) {
        if self.engine_tombstones.is_empty() {
            return;
        }
        let names = SqlNames::new(sql);
        if self
            .engine_tombstones
            .iter()
            .any(|name| names.may_reference(name))
        {
            self.drop_engine();
        }
    }

    /// Remove `key` from the live engine without rebuilding it: the table is replaced by an
    /// empty batch (releasing its device memory) and remembered as a tombstone.
    #[cfg(not(target_arch = "wasm32"))]
    fn engine_forget_table(&mut self, key: &str, schema: &SchemaRef) {
        let Some(eng) = self.engine.as_mut() else {
            return;
        };
        match eng.replace_table(key.to_string(), RecordBatch::new_empty(schema.clone())) {
            Ok(()) => {
                self.engine_tombstones.insert(key.to_string());
                if self.engine_tombstones.len() > MAX_ENGINE_TOMBSTONES {
                    self.drop_engine();
                }
            }
            Err(_) => {
                // Fall back to a lazy rebuild without the table.
                self.drop_engine();
            }
        }
    }

    /// Register a table that is new to the catalog in the live engine (if one can be built).
    ///
    /// An upload failure is recorded in `engine_upload_errors` (the table is then served by the
    /// host fallback) instead of failing the registration, so the catalog behaves the same on
    /// every backend (W1-E-003).
    ///
    /// Returns the engine's coalesced copy for [`Self::share_engine_copy`] (W3-A-001).
    #[cfg(not(target_arch = "wasm32"))]
    fn engine_register_new(&mut self, name: &str, batches: &[RecordBatch]) -> Option<RecordBatch> {
        if self.ensure_engine().is_err() {
            return None;
        }
        let bytes = batches.iter().map(RecordBatch::get_array_memory_size).sum();
        if let Err(e) = self.check_engine_budget(name, bytes) {
            // A tombstone of the same name keeps answering with 0 rows; the guard routes the
            // name to the host anyway, and the next rebuild drops it.
            self.engine_upload_errors
                .insert(name.to_string(), e.to_string());
            return None;
        }
        let exact_tombstone = self.engine_tombstones.remove(name);
        if !exact_tombstone {
            let lower = name.to_ascii_lowercase();
            if self
                .engine_tombstones
                .iter()
                .any(|t| t.to_ascii_lowercase() == lower)
            {
                // A placeholder differing only in case: rebuild lazily (catalog includes `name`).
                self.drop_engine();
                return None;
            }
        }
        let eng = self.engine.as_mut().expect("engine ensured above");
        match engine_upload(eng, name, batches, exact_tombstone) {
            Ok(copy) => {
                self.engine_upload_errors.remove(name);
                copy
            }
            Err(e) => {
                self.engine_upload_errors
                    .insert(name.to_string(), e.to_string());
                None
            }
        }
    }

    /// Replace an existing catalog table in the live engine (if one can be built). Failures
    /// are recorded like [`Self::engine_register_new`]; returns the engine's coalesced copy.
    #[cfg(not(target_arch = "wasm32"))]
    fn engine_replace_existing(
        &mut self,
        key: &str,
        batches: &[RecordBatch],
    ) -> Option<RecordBatch> {
        if self.ensure_engine().is_err() {
            return None;
        }
        let bytes = batches.iter().map(RecordBatch::get_array_memory_size).sum();
        if let Err(e) = self.check_engine_budget(key, bytes) {
            self.engine_upload_errors
                .insert(key.to_string(), e.to_string());
            self.engine_release_table(key, batches);
            return None;
        }
        let eng = self.engine.as_mut().expect("engine ensured above");
        // The engine copy may be missing or partial if an earlier upload failed: try
        // `replace_table` first, then a fresh `register_table`.
        let res = engine_upload(eng, key, batches, true)
            .or_else(|_| engine_upload(eng, key, batches, false));
        match res {
            Ok(copy) => {
                self.engine_upload_errors.remove(key);
                copy
            }
            Err(e) => {
                self.engine_upload_errors
                    .insert(key.to_string(), e.to_string());
                None
            }
        }
    }

    /// Replace the engine copy of `key` with an empty batch (releasing its device memory) after
    /// it was routed to the host; on failure the engine is rebuilt lazily.
    #[cfg(not(target_arch = "wasm32"))]
    fn engine_release_table(&mut self, key: &str, batches: &[RecordBatch]) {
        let Some(eng) = self.engine.as_mut() else {
            return;
        };
        let Some(schema) = batches.first().map(RecordBatch::schema) else {
            return;
        };
        if eng
            .replace_table(key.to_string(), RecordBatch::new_empty(schema))
            .is_err()
        {
            self.drop_engine();
        }
    }

    /// Append batches to an existing catalog table in the live engine (if any). A table whose
    /// upload already failed stays host-only until it is replaced or the engine is rebuilt.
    ///
    /// The new batches are coalesced into one `register_batch` call (W2-O-001): craton-bolt
    /// re-concatenates the whole table on every call. Returns the engine's coalesced copy of
    /// the appended rows for [`Self::share_engine_copy`] (W3-A-001).
    #[cfg(not(target_arch = "wasm32"))]
    fn engine_append_existing(
        &mut self,
        key: &str,
        batches: &[RecordBatch],
    ) -> Option<RecordBatch> {
        if self.engine_upload_errors.contains_key(key) || self.ensure_engine().is_err() {
            return None;
        }
        let existing = &self.catalog[key];
        let added: usize = batches.iter().map(RecordBatch::get_array_memory_size).sum();
        let admitted = self
            .check_engine_budget(key, existing.bytes_in_memory.saturating_add(added))
            .and_then(|()| {
                check_engine_utf8_capacity(
                    key,
                    &existing.schema,
                    existing.batches.iter().chain(batches),
                )
            })
            .and_then(|()| coalesce_for_engine(key, batches));
        let combined = match admitted {
            Ok(batch) => batch,
            Err(e) => {
                self.engine_upload_errors
                    .insert(key.to_string(), e.to_string());
                let schema_batch = [RecordBatch::new_empty(existing.schema.clone())];
                self.engine_release_table(key, &schema_batch);
                return None;
            }
        };
        let shared = (batches.len() > 1).then(|| combined.clone());
        let eng = self.engine.as_mut().expect("engine ensured above");
        match eng.register_batch(key, combined) {
            Ok(()) => shared,
            Err(e) => {
                // `register_batch` is not transactional: the engine copy may now be partial.
                // Route the table to the host until it is replaced or the engine is rebuilt.
                self.engine_upload_errors.insert(
                    key.to_string(),
                    format!("Bolt engine failed to append to table '{key}': {e}"),
                );
                None
            }
        }
    }

    /// Planner catalog for the shadow catalog, plus `extra` (the engine-only scratch table).
    #[cfg(not(target_arch = "wasm32"))]
    fn build_mem_table_provider(
        &self,
        extra: Option<(&str, &Schema)>,
    ) -> Result<craton_bolt::plan::MemTableProvider> {
        let mut provider = craton_bolt::plan::MemTableProvider::new();
        for (name, entry) in &self.catalog {
            provider.register(name.clone(), plan_schema_of(&entry.schema)?);
        }
        if let Some((name, schema)) = extra {
            provider.register(name.to_string(), plan_schema_of(schema)?);
        }
        Ok(provider)
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn plan_logical_without_gpu(&self, sql: &str) -> Result<craton_bolt::plan::LogicalPlan> {
        self.plan_logical_with(sql, None)
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn plan_logical_with(
        &self,
        sql: &str,
        extra: Option<(&str, &Schema)>,
    ) -> Result<craton_bolt::plan::LogicalPlan> {
        self.plan_logical_stages(sql, extra)
            .map(|(_, optimized)| optimized)
    }

    /// The plan of `sql` as parsed, and after the default optimizer passes.
    #[cfg(not(target_arch = "wasm32"))]
    fn plan_logical_stages(
        &self,
        sql: &str,
        extra: Option<(&str, &Schema)>,
    ) -> Result<(
        craton_bolt::plan::LogicalPlan,
        craton_bolt::plan::LogicalPlan,
    )> {
        let provider = self.build_mem_table_provider(extra)?;
        let parsed = craton_bolt::plan::parse_sql(sql, &provider)
            .map_err(|e| Error::invalid_input(format!("Bolt SQL error: {e}")))?;
        let mut logical = parsed.clone();
        for pass in craton_bolt::plan::default_passes() {
            logical = pass
                .rewrite(logical)
                .map_err(|e| Error::invalid_input(format!("Bolt optimizer error: {e}")))?;
        }
        Ok((parsed, logical))
    }

    /// Engine guard (W1-O-002, W1-E-003): returns why `sql` must not run on the GPU engine.
    ///
    /// craton-bolt 0.9.0 mishandles NULLs: an aggregate over a filtered input crashes the
    /// whole process (`STATUS_ACCESS_VIOLATION`, sometimes after CUDA error 700), and filters
    /// and computed expressions ignore validity bitmaps (wrong results); see
    /// [`plan_is_null_safe`]. The guard routes to the host fallback:
    ///
    /// 1. any query that may reference (case-insensitive substring) a table the engine could
    ///    not upload (W1-E-003);
    /// 2. any query Bolt's planner cannot plan statically (W8-B-003). The engine runs such
    ///    shapes (`WITH RECURSIVE`, `LATERAL`, correlated subqueries, `VALUES`,
    ///    `generate_series`, `DISTINCT ON`, ...) as host-orchestrated GPU sub-plans, which have
    ///    every hazard below, and the guard cannot see them: it fails closed;
    /// 3. any query that may reference a table containing a NULL, unless the plan is
    ///    null-safe.
    ///
    /// Tables without NULLs are unaffected. Expressions are not tracked to their column, so
    /// any expression over a table with a NULL anywhere is unsafe (conservative); the
    /// column-level exceptions of [`plan_is_null_safe`] (grouped `SUM` / `MIN` / `MAX`,
    /// `NULLS FIRST` sort keys) follow each column through the plan to the stored table it
    /// comes from ([`NodeFacts`], W8-B-004). `extra` is an engine-only table (the scratch
    /// table) with its data.
    ///
    /// Type hazards (W4-B-002, W4-B-004, W5-A-001, W5-A-002, W5-A-003, W8-B-001, W8-B-002,
    /// W8-B-006, W8-B-007, W8-B-008) go to the host whatever the NULLs, see
    /// [`plan_type_hazard`].
    ///
    /// Stack-depth guard (W2-B-001): queries with more than [`MAX_GPU_PREDICATE_TERMS`]
    /// `AND`/`OR` terms, and dictionary-lowered Utf8 predicates over tables with more than
    /// [`MAX_GPU_DICTIONARY_LOWERED_VALUES`] distinct strings in a column, go to the host too.
    #[cfg(not(target_arch = "wasm32"))]
    fn gpu_crash_guard(&self, sql: &str, extra: Option<(&str, &RecordBatch)>) -> Option<String> {
        let names = SqlNames::new(sql);
        if let Some((name, reason)) = self
            .engine_upload_errors
            .iter()
            .find(|(name, _)| names.may_reference(name))
        {
            return Some(format!("table '{name}' is not on the GPU: {reason}"));
        }
        let terms = names.predicate_terms();
        if terms > MAX_GPU_PREDICATE_TERMS {
            return Some(format!(
                "the query has {terms} AND/OR terms, more than the {MAX_GPU_PREDICATE_TERMS} \
                 craton-bolt 0.9 evaluates safely (deep predicate trees)"
            ));
        }
        // W8-B-003: what the guard cannot plan, it cannot vouch for.
        let (parsed, plan) =
            match self.plan_logical_stages(sql, extra.map(|(n, b)| (n, b.schema_ref().as_ref()))) {
                Ok(plans) => plans,
                Err(e) => {
                    return Some(format!(
                        "the guard's Bolt planner cannot check the query ({e}); craton-bolt 0.9 \
                     would run it as host-orchestrated GPU sub-plans with the same hazards \
                     (W8-B-003)"
                    ));
                }
            };
        let referenced: Vec<(&str, &RegisteredTable)> = self
            .catalog
            .iter()
            .filter(|(name, _)| names.may_reference(name))
            .map(|(name, t)| (name.as_str(), t))
            .collect();
        let wide: Vec<&str> = referenced
            .iter()
            .filter(|(_, t)| t.max_utf8_distinct() > MAX_GPU_DICTIONARY_LOWERED_VALUES)
            .map(|(name, _)| *name)
            .collect();
        if !wide.is_empty() && plan_has_dictionary_lowered_predicate(&plan) {
            return Some(format!(
                "the query has a LIKE or string-ordering predicate over table(s) with more than \
                 {MAX_GPU_DICTIONARY_LOWERED_VALUES} distinct strings in a column ({}), which \
                 craton-bolt 0.9 lowers to one OR term per value",
                wide.join(", ")
            ));
        }
        let tables = GuardTables::new(&self.catalog, extra);
        // Type hazards depend on expression types, so every query is checked, before and after
        // the optimizer: constant folding evaluates `7 / 2` or `0.1 + 0.2` with the engine's
        // semantics and leaves only the (wrong) literal (W8-B-001, W8-B-002).
        let fractional = has_fractional_literal(sql);
        if let Some(reason) = plan_type_hazard(&parsed, &tables, fractional, false)
            .or_else(|| plan_type_hazard(&plan, &tables, fractional, false))
        {
            return Some(reason.to_string());
        }
        let mut nullable: Vec<&str> = referenced
            .iter()
            .filter(|(_, t)| t.null_count > 0)
            .map(|(name, _)| *name)
            .collect();
        if let Some((name, _)) = extra
            && tables.scratch_has_nulls()
        {
            nullable.push(name);
        }
        if nullable.is_empty() {
            // No stored NULLs, but the query itself may produce them: outer-join padding,
            // `CASE` without `ELSE`, NULL literals (W3-E-002). Only checked when the text has a
            // keyword that can introduce one.
            let may_produce = ["left", "right", "full", "outer", "case", "null", "nullif"]
                .iter()
                .any(|k| names.words.contains(*k));
            if may_produce && plan_may_produce_nulls(&plan) && !plan_is_null_safe(&plan, &tables) {
                return Some(
                    "the query produces NULLs (outer join, CASE or NULL literal) that \
                     craton-bolt 0.9.0 mishandles on the GPU"
                        .to_string(),
                );
            }
            return None;
        }
        // Subquery expressions embed plans `plan_is_null_safe` does not descend into.
        if plan_is_null_safe(&plan, &tables) && !format!("{plan:?}").contains("Subquery") {
            return None;
        }
        Some(format!(
            "the query evaluates expressions over table(s) with NULLs ({}), which craton-bolt \
             0.9.0 mishandles on the GPU (NULL validity)",
            nullable.join(", ")
        ))
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn plan_without_gpu(
        &self,
        sql: &str,
    ) -> Result<(
        craton_bolt::plan::LogicalPlan,
        craton_bolt::plan::PhysicalPlan,
    )> {
        let logical = self.plan_logical_without_gpu(sql)?;
        let physical = craton_bolt::plan::lower_physical(&logical)
            .map_err(|e| Error::invalid_input(format!("Bolt physical lowering error: {e}")))?;
        Ok((logical, physical))
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn fallback_db(&mut self) -> Result<&mut FallbackDb> {
        if self.fallback_db.is_none() {
            let db = FallbackDb::open()?;
            if let Some(slot) = &self.fallback_interrupt {
                lock_slot(slot).handle = Some(db.conn.interrupt_handle());
            }
            self.fallback_db = Some(db);
        }
        Ok(self.fallback_db.as_mut().expect("fallback db opened above"))
    }

    /// Clear an interrupt a timed-out caller issued for the request that just finished
    /// (W4-A-004), so it cannot fail the next host-fallback query: run a trivial query until
    /// one succeeds (DuckDB either reset the flag already or consumes it on the first query).
    /// If the connection still fails, the fallback database is rebuilt lazily.
    #[cfg(not(target_arch = "wasm32"))]
    fn absorb_fallback_interrupt(&mut self) {
        let Some(db) = self.fallback_db.as_mut() else {
            return;
        };
        for _ in 0..2 {
            if db
                .conn
                .execute("SELECT 1", duckdb_neo::Parameters::None)
                .is_ok()
            {
                return;
            }
        }
        self.fallback_db = None;
        if let Some(slot) = &self.fallback_interrupt {
            lock_slot(slot).handle = None;
        }
    }

    /// Restore a usable state after a command panicked: the engine and the fallback database
    /// may be mid-mutation, so both are rebuilt lazily. The catalog (host-side Arrow data) is
    /// only mutated after fallible engine calls succeed, so it is kept.
    ///
    /// The bookkeeping is reset and published first; the engine and the fallback database are
    /// dropped last, each under its own `catch_unwind`, because their `Drop` is the code most
    /// likely to fail after a broken operation (W2-E-006). `honor_test_hook` lets the test hook
    /// make this call panic once.
    fn recover_after_panic(&mut self, honor_test_hook: bool) {
        if honor_test_hook && std::mem::take(&mut self.panic_in_recovery) {
            panic!("duckdb-bolt-worker: injected panic during recovery (test hook)");
        }
        self.panic_in_recovery = false;
        #[cfg(not(target_arch = "wasm32"))]
        let (engine, fallback_db) = (self.engine.take(), self.fallback_db.take());
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.invalidate_engine();
            if let Some(slot) = &self.fallback_interrupt {
                lock_slot(slot).handle = None;
            }
        }
        let config = self.config.clone();
        let tables: BTreeMap<String, TableSummary> = self
            .catalog
            .iter()
            .map(|(k, v)| (k.clone(), v.summary(k)))
            .collect();
        let batches: HashMap<String, Vec<RecordBatch>> = self
            .catalog
            .iter()
            .map(|(k, v)| (k.clone(), v.batches.clone()))
            .collect();
        let lower_index = self.lower_index.clone();
        self.publish(|m| {
            m.config = config;
            m.tables = tables;
            m.batches = batches;
            m.lower_index = lower_index;
        });
        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ = catch_unwind(AssertUnwindSafe(move || drop(fallback_db)));
            let _ = catch_unwind(AssertUnwindSafe(move || drop(engine)));
        }
    }

    // ---------------------------------------------------------------- command dispatch

    fn handle(&mut self, cmd: WorkerCommand) {
        match cmd {
            WorkerCommand::UpdateConfig(new_cfg, reply) => {
                let changed = self.config != new_cfg;
                self.config = new_cfg;
                if changed {
                    // Device, budget, PTX cache, or tracing changed: rebuild lazily on next use.
                    #[cfg(not(target_arch = "wasm32"))]
                    {
                        self.invalidate_engine();
                        self.sync_log_level();
                    }
                    let config = self.config.clone();
                    self.publish(|m| m.config = config);
                }
                self.reply(reply, Ok(()));
            }
            WorkerCommand::ResetEngine(reply) => {
                let result = self.handle_reset_engine();
                self.reply(reply, result);
            }
            WorkerCommand::RegisterTable {
                name,
                schema,
                batches,
                mode,
                reply,
            } => {
                let result = self.handle_register_table(name, schema, batches, mode);
                self.reply(reply, result);
            }
            WorkerCommand::DropTable { name, reply } => {
                let result = self.handle_drop_table(&name);
                self.reply(reply, result);
            }
            WorkerCommand::ClearTables(reply) => {
                let result = self.handle_clear_tables();
                self.reply(reply, result);
            }
            WorkerCommand::ExecuteSql { sql, exact, reply } => {
                let res = self.handle_execute_sql(&sql, exact);
                self.reply(reply, res);
            }
            WorkerCommand::ExecuteSqlHostFallback { sql, reply } => {
                #[cfg(not(target_arch = "wasm32"))]
                {
                    let res = self.execute_sql_without_gpu(&sql, false);
                    self.reply(reply, res);
                }
                #[cfg(target_arch = "wasm32")]
                {
                    let _ = sql;
                    self.reply(
                        reply,
                        Err(Error::invalid_input(
                            "Craton Bolt SQL execution is not supported on wasm32 targets",
                        )),
                    );
                }
            }
            WorkerCommand::ExplainSql { sql, reply } => {
                let res = self.handle_explain_sql(&sql);
                self.reply(reply, res);
            }
            WorkerCommand::InferSqlSchema { sql, reply } => {
                let res = self.handle_infer_sql_schema(&sql);
                self.reply(reply, res);
            }
            WorkerCommand::ExecutionTier { sql, reply } => {
                let res = self.handle_execution_tier(&sql);
                self.reply(reply, res);
            }
            WorkerCommand::BackendIsGpu(reply) => {
                #[cfg(not(target_arch = "wasm32"))]
                let gpu = self.ensure_engine().is_ok();
                #[cfg(target_arch = "wasm32")]
                let gpu = false;
                self.reply(reply, gpu);
            }
            WorkerCommand::Telemetry(reply) => {
                let snap = self.handle_telemetry();
                self.reply(reply, snap);
            }
            WorkerCommand::VersionInfo(reply) => {
                let info = self.handle_version_info();
                self.reply(reply, info);
            }
            WorkerCommand::ExecuteWithScratch { batch, sql, reply } => {
                let result = self.handle_execute_with_scratch(batch, &sql);
                self.reply(reply, result);
            }
            WorkerCommand::InjectPanic { in_recovery, reply } => {
                // Keep the reply sender alive until the panic so it is dropped by unwinding.
                let _reply = reply;
                self.panic_in_recovery = in_recovery;
                panic!("duckdb-bolt-worker: injected panic (test hook)");
            }
        }
    }

    /// `PRAGMA bolt_reset`: drop the engine, clear a cached init failure, and rebuild.
    ///
    /// Error policy (W1-B-001, review 1.1.13): the reset fails whenever the engine cannot be
    /// built **and at least one CUDA device is visible**: a driver or context failure, an
    /// unusable `BOLT_PTX_CACHE_DIR`, a `bolt_device` the process cannot switch to (W2-E-010),
    /// or a `bolt_device` ordinal that does not exist (a misconfiguration, reported with the
    /// visible ordinals). The host fallback stays active and the reason is cached for
    /// `bolt_pool_stats()`. The reset succeeds without an engine only when no CUDA device is
    /// visible at all, which is always the case in `cuda-stub` builds: there the host fallback
    /// is the only backend, not an error. A catalog table the engine cannot upload never fails
    /// the reset: it is served by the host fallback and reported in
    /// `BoltTelemetrySnapshot::fallback_reason` (W1-E-003).
    fn handle_reset_engine(&mut self) -> Result<()> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.invalidate_engine();
            match self.build_engine() {
                Ok(()) => Ok(()),
                Err(_) if cuda_devices(self.config.device).is_empty() => Ok(()),
                Err(e) => Err(e),
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            Ok(())
        }
    }

    fn handle_register_table(
        &mut self,
        name: String,
        schema: SchemaRef,
        batches: Vec<RecordBatch>,
        mode: RegisterMode,
    ) -> Result<TableSummary> {
        check_table_name_not_reserved(&name)?;
        check_column_names(&name, &schema)?;
        let existing_key = self.resolve_table_key(&name);
        match (mode, existing_key) {
            (RegisterMode::Create, Some(key)) => Err(Error::invalid_input(format!(
                "Table '{key}' is already registered in Craton Bolt"
            ))),
            (RegisterMode::Create | RegisterMode::Replace | RegisterMode::Append, None) => {
                #[cfg(not(target_arch = "wasm32"))]
                let copy = self.engine_register_new(&name, &batches);
                let summary = self
                    .catalog_insert(name.clone(), RegisteredTable::from_batches(schema, batches));
                #[cfg(not(target_arch = "wasm32"))]
                if let Some(copy) = copy {
                    self.share_engine_copy(&name, copy);
                }
                Ok(summary)
            }
            (RegisterMode::Replace, Some(key)) => {
                #[cfg(not(target_arch = "wasm32"))]
                let copy = self.engine_replace_existing(&key, &batches);
                let summary = self
                    .catalog_insert(key.clone(), RegisteredTable::from_batches(schema, batches));
                #[cfg(not(target_arch = "wasm32"))]
                if let Some(copy) = copy {
                    self.share_engine_copy(&key, copy);
                }
                Ok(summary)
            }
            (RegisterMode::Append, Some(key)) => {
                let existing_schema = self.catalog[&key].schema.clone();
                // Coerce every batch before touching the engine or the catalog, so a bad
                // batch leaves both unchanged.
                let batches = batches
                    .into_iter()
                    .filter(|b| b.num_rows() > 0)
                    .map(|b| coerce_batch_to_schema(&key, b, &existing_schema))
                    .collect::<Result<Vec<_>>>()?;
                #[cfg(not(target_arch = "wasm32"))]
                let copy = if batches.is_empty() {
                    None
                } else {
                    self.engine_append_existing(&key, &batches)
                };
                let summary = self.catalog_append(&key, batches);
                #[cfg(not(target_arch = "wasm32"))]
                if let Some(copy) = copy {
                    self.share_engine_copy(&key, copy);
                }
                Ok(summary)
            }
        }
    }

    fn handle_execute_with_scratch(
        &mut self,
        batch: RecordBatch,
        sql: &str,
    ) -> Result<SqlExecutionOutput> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.purge_tombstones_referenced_by(sql);
            self.ensure_engine()?;
            let batch = normalize_batch_for_bolt(&batch)?;
            let schema = batch.schema();
            if let Some(reason) = self.gpu_crash_guard(sql, Some((SCRATCH_TABLE_NAME, &batch))) {
                // Callers treat an error as "use a host path".
                craton_bolt::metrics().inc(craton_bolt::Counter::HostFallbacksTotal);
                return Err(Error::invalid_input(format!(
                    "Craton Bolt scratch query not run on the GPU: {reason}"
                )));
            }
            let had_scratch = self.engine_has_scratch;
            let eng = self.engine.as_mut().expect("engine ensured above");
            let installed = if had_scratch {
                eng.replace_table(SCRATCH_TABLE_NAME, batch)
            } else {
                eng.register_table(SCRATCH_TABLE_NAME, batch)
            };
            installed.map_err(|e| {
                Error::invalid_input(format!("Bolt engine failed to upload scratch table: {e}"))
            })?;
            self.engine_has_scratch = true;
            let mut device_error = false;
            let result = eng
                .sql(sql)
                .map(|handle| SqlExecutionOutput {
                    execution_tier: format!("{:?}", handle.planned_execution_tier()),
                    batch: handle.into_record_batch(),
                })
                .map_err(|e| {
                    device_error = matches!(
                        e,
                        craton_bolt::BoltError::Cuda(_)
                            | craton_bolt::BoltError::CudaWithCode { .. }
                    );
                    Error::invalid_input(format!("Craton Bolt SQL execution error: {e}"))
                });
            // Release the scratch data's device memory right away.
            if device_error
                || eng
                    .replace_table(SCRATCH_TABLE_NAME, RecordBatch::new_empty(schema))
                    .is_err()
            {
                self.drop_engine();
            }
            result
        }
        #[cfg(target_arch = "wasm32")]
        {
            let _ = (batch, sql);
            Err(Error::invalid_input(
                "Craton Bolt GPU SQL execution is not supported on wasm32 targets",
            ))
        }
    }

    fn handle_drop_table(&mut self, name: &str) -> Result<bool> {
        let Some(key) = self.resolve_table_key(name) else {
            return Ok(false);
        };
        let Some(_entry) = self.catalog_remove(&key) else {
            return Ok(false);
        };
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.engine_forget_table(&key, &_entry.schema);
            self.engine_upload_errors.remove(&key);
            if let Some(db) = self.fallback_db.as_mut()
                && db.drop_mirror(&key).is_err()
            {
                self.fallback_db = None;
            }
        }
        Ok(true)
    }

    fn handle_clear_tables(&mut self) -> Result<usize> {
        let keys: Vec<String> = self.catalog.keys().cloned().collect();
        let removed = keys.len();
        #[cfg(not(target_arch = "wasm32"))]
        if self.engine.is_some() && self.engine_tombstones.len() + removed > MAX_ENGINE_TOMBSTONES {
            self.drop_engine();
        }
        for key in keys {
            if let Some(_entry) = self.catalog.remove(&key) {
                #[cfg(not(target_arch = "wasm32"))]
                self.engine_forget_table(&key, &_entry.schema);
            }
        }
        self.lower_index.clear();
        #[cfg(not(target_arch = "wasm32"))]
        self.engine_upload_errors.clear();
        self.publish(|m| {
            m.tables.clear();
            m.batches.clear();
            m.lower_index.clear();
        });
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(db) = self.fallback_db.as_mut()
            && db.drop_all_mirrors().is_err()
        {
            self.fallback_db = None;
        }
        Ok(removed)
    }

    fn handle_execute_sql(&mut self, sql: &str, exact: bool) -> Result<SqlExecutionBatches> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.purge_tombstones_referenced_by(sql);
            if self.ensure_engine().is_err() {
                return self.execute_sql_without_gpu(sql, exact);
            }
            if let Some(reason) = self.gpu_crash_guard(sql, None) {
                craton_bolt::metrics().inc(craton_bolt::Counter::HostFallbacksTotal);
                return self.execute_sql_without_gpu(sql, exact).map_err(|e| {
                    Error::invalid_input(format!("{e} (routed to the host fallback: {reason})"))
                });
            }
            let eng = self.engine.as_mut().expect("engine ensured above");
            match eng.sql(sql) {
                Ok(handle) => {
                    let tier = format!("{:?}", handle.planned_execution_tier());
                    let batch = handle.into_record_batch();
                    Ok(SqlExecutionBatches::single(batch, tier))
                }
                Err(gpu_err) => {
                    // W1-B-002: whatever the engine rejected or failed on (planning gaps,
                    // unsupported functions, capacity, device errors) is retried on the host
                    // fallback, so a query behaves the same with and without a GPU. Invalid SQL
                    // fails on the host too, and then both errors are reported.
                    if matches!(
                        gpu_err,
                        craton_bolt::BoltError::Cuda(_)
                            | craton_bolt::BoltError::CudaWithCode { .. }
                    ) {
                        // A CUDA driver error (e.g. 700, illegal address) can leave the context
                        // unusable: rebuild lazily.
                        self.drop_engine();
                    }
                    craton_bolt::metrics().inc(craton_bolt::Counter::HostFallbacksTotal);
                    self.execute_sql_without_gpu(sql, exact)
                        .map_err(|host_err| {
                            Error::invalid_input(format!(
                                "Craton Bolt SQL execution error: {gpu_err}; {host_err}"
                            ))
                        })
                }
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            let _ = sql;
            Err(Error::invalid_input(
                "Craton Bolt GPU SQL execution is not supported on wasm32 targets",
            ))
        }
    }

    /// Whether the host fallback must mirror widened integer columns with their original
    /// DuckDB types for `sql` (W4-B-002): it references such a column and does arithmetic on
    /// it (by the Bolt plan; by the text when the planner rejects the query). Other queries
    /// keep the storage types (W3-B-002).
    #[cfg(not(target_arch = "wasm32"))]
    fn wants_original_integers(&self, sql: &str) -> bool {
        let names = SqlNames::new(sql);
        let any_narrow = self
            .catalog
            .iter()
            .filter(|(name, _)| names.may_reference(name))
            .flat_map(|(_, t)| t.schema.fields().iter())
            .any(|f| widened_integer_original(f).is_some());
        if !any_narrow {
            return false;
        }
        match self.plan_logical_without_gpu(sql) {
            Ok(plan) => plan_has_narrow_arithmetic(&plan, &GuardTables::new(&self.catalog, None)),
            // Shapes the Bolt planner rejects: any arithmetic operator in the text counts.
            Err(_) => sql.contains(['+', '-', '*', '/', '%']),
        }
    }

    /// Execute `sql` on the embedded host-fallback DuckDB. The reported tier is `"Host"`: that
    /// is where the query ran (W2-E-012, W2-A-002).
    #[cfg(not(target_arch = "wasm32"))]
    fn execute_sql_without_gpu(&mut self, sql: &str, exact: bool) -> Result<SqlExecutionBatches> {
        let tier = "Host".to_string();
        let original_integers = self.wants_original_integers(sql);
        self.fallback_db()?;
        let db = self.fallback_db.as_mut().expect("fallback db opened above");
        let (schema, batches) = db
            .query(&self.catalog, sql, original_integers, exact)
            .map_err(|e| Error::invalid_input(format!("Craton Bolt host fallback error: {e}")))?;
        Ok(SqlExecutionBatches {
            schema,
            batches,
            execution_tier: tier,
        })
    }

    /// `bolt_query`'s result schema, without executing anything (RFC-002).
    ///
    /// DuckDB's result types (W9-A-001): the query is bound by the [`TypeBinder`], whose
    /// tables have the catalog's original DuckDB types, so `7 / 2` is `DOUBLE`, `SUM(BIGINT)`
    /// `HUGEINT`, `SUM(FLOAT)` `DOUBLE`, a `TINYINT` column `TINYINT`, `ts - ts` `INTERVAL`.
    /// Column names are the Bolt planner's when it plans the query (unchanged for existing
    /// callers), else DuckDB's. When DuckDB cannot bind it, or a result type has no Arrow
    /// equivalent here, the Bolt planner's types are used, as before.
    fn handle_infer_sql_schema(&mut self, sql: &str) -> Result<SchemaRef> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let planned = self.bolt_planned_schema(sql);
            if self.type_binder.is_none() {
                self.type_binder = TypeBinder::open().ok();
            }
            let described = match self.type_binder.as_mut() {
                Some(binder) => binder.describe(&self.catalog, sql),
                None => Err(Error::api_error(
                    "the bolt_query type binder is unavailable",
                )),
            };
            let Ok(columns) = described else {
                return planned;
            };
            let names: Vec<String> = match &planned {
                Ok(schema) if schema.fields().len() == columns.len() => {
                    schema.fields().iter().map(|f| f.name().clone()).collect()
                }
                Ok(_) => return planned,
                Err(_) => columns.iter().map(|(name, _)| name.clone()).collect(),
            };
            let fields: Option<Vec<arrow_schema::Field>> = columns
                .iter()
                .zip(&names)
                .map(|((_, ty), name)| ty.field(name))
                .collect();
            match fields {
                Some(fields) if !fields.is_empty() => {
                    Ok(std::sync::Arc::new(arrow_schema::Schema::new(fields)))
                }
                _ => planned,
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            let _ = sql;
            Err(Error::invalid_input(
                "Craton Bolt schema inference is not supported on wasm32 targets",
            ))
        }
    }

    /// The Bolt planner's output schema of `sql` (the types the engine returns).
    #[cfg(not(target_arch = "wasm32"))]
    fn bolt_planned_schema(&self, sql: &str) -> Result<SchemaRef> {
        {
            use craton_bolt::plan::{DataType as P, TimeUnit as PU};

            let logical = self.plan_logical_without_gpu(sql)?;
            let plan_schema = logical
                .schema()
                .map_err(|e| Error::invalid_input(format!("Bolt schema error: {e}")))?;
            let fields: Vec<arrow_schema::Field> = plan_schema
                .fields
                .iter()
                .map(|f| {
                    let dt = match f.dtype {
                        P::Bool => ArrowDataType::Boolean,
                        P::Int32 => ArrowDataType::Int32,
                        P::Int64 => ArrowDataType::Int64,
                        P::Float32 => ArrowDataType::Float32,
                        P::Float64 => ArrowDataType::Float64,
                        P::Utf8 => ArrowDataType::Utf8,
                        P::Decimal128(p, s) => ArrowDataType::Decimal128(p, s),
                        P::Date32 => ArrowDataType::Date32,
                        P::Timestamp(u, tz) => ArrowDataType::Timestamp(
                            match u {
                                PU::Second => ArrowTimeUnit::Second,
                                PU::Millisecond => ArrowTimeUnit::Millisecond,
                                PU::Microsecond => ArrowTimeUnit::Microsecond,
                                PU::Nanosecond => ArrowTimeUnit::Nanosecond,
                            },
                            tz.map(std::sync::Arc::from),
                        ),
                    };
                    arrow_schema::Field::new(&f.name, dt, f.nullable)
                })
                .collect();
            if fields.is_empty() {
                return Err(Error::invalid_input(
                    "Bolt query has an empty output schema",
                ));
            }
            Ok(std::sync::Arc::new(arrow_schema::Schema::new(fields)))
        }
    }

    fn handle_explain_sql(&mut self, sql: &str) -> Result<String> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.purge_tombstones_referenced_by(sql);
            if self.ensure_engine().is_ok() {
                // W2-A-002: say so when the crash guard sends the query to the host.
                let routed = self.gpu_crash_guard(sql, None);
                let eng = self.engine.as_mut().expect("engine ensured above");
                let plan = match eng.explain_sql(sql) {
                    Ok(plan) => plan,
                    // A routed query the engine cannot explain (W8-B-003: shapes the planner
                    // rejects) is explained by the DuckDB that runs it.
                    Err(_) if routed.is_some() => {
                        self.fallback_db()?;
                        let db = self.fallback_db.as_mut().expect("fallback db opened above");
                        let plan = db.explain(&self.catalog, sql)?;
                        format!("=== Host fallback (DuckDB) plan ===\n{plan}")
                    }
                    Err(e) => {
                        return Err(Error::invalid_input(format!(
                            "Craton Bolt EXPLAIN error: {e}"
                        )));
                    }
                };
                return Ok(match routed {
                    Some(reason) => {
                        format!("=== Executed on the host fallback (DuckDB): {reason} ===\n{plan}")
                    }
                    None => plan,
                });
            }
            match self.plan_without_gpu(sql) {
                Ok((logical, physical)) => Ok(format!(
                    "=== No GPU engine: executed on the host fallback (DuckDB) ===\n\
                     === Logical Plan ===\n{}\n=== Physical Plan (tier: {:?}) ===\n{}",
                    craton_bolt::plan::format_logical(&logical),
                    physical.planned_execution_tier(),
                    craton_bolt::plan::format_physical(&physical),
                )),
                Err(plan_err) => {
                    // W1-B-007: shapes Bolt's planner rejects (LATERAL, ...) still run on the
                    // host fallback, so explain them with DuckDB.
                    self.fallback_db()?;
                    let db = self.fallback_db.as_mut().expect("fallback db opened above");
                    let plan = db.explain(&self.catalog, sql).map_err(|e| {
                        Error::invalid_input(format!(
                            "Query cannot be planned: {plan_err}; host fallback: {e}"
                        ))
                    })?;
                    Ok(format!("=== Host fallback (DuckDB) plan ===\n{plan}"))
                }
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            let _ = sql;
            Err(Error::invalid_input(
                "Craton Bolt EXPLAIN is not supported on wasm32 targets",
            ))
        }
    }

    /// Tier the query would run on, without executing anything (W2-E-012, W2-A-002):
    /// 1. No GPU engine (cuda-stub, no device, cached init failure), or the crash guard
    ///    routes the query to the host -> `"Host"` if the embedded DuckDB fallback binds it.
    /// 2. Else Craton Bolt's raw planner plans it -> its planned tier (`"Gpu"` / `"Hybrid"` /
    ///    `"Host"`).
    /// 3. Else (the logical plan exists but its physical lowering fails) the engine explains
    ///    it -> `"Hybrid"`. Host-orchestrated shapes the planner rejects (LATERAL, recursive
    ///    CTEs, correlated subqueries, ...) are routed by the guard since W8-B-003 (step 1).
    ///
    /// A query the engine fails on at run time is still retried on the host (W1-B-002); that
    /// cannot be predicted here.
    fn handle_execution_tier(&mut self, sql: &str) -> Result<String> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.purge_tombstones_referenced_by(sql);
            let engine_ok = self.ensure_engine().is_ok();
            let routed = engine_ok && self.gpu_crash_guard(sql, None).is_some();
            let plan_note = if engine_ok && !routed {
                match self.plan_without_gpu(sql) {
                    Ok((_logical, physical)) => {
                        return Ok(format!("{:?}", physical.planned_execution_tier()));
                    }
                    Err(plan_err) => {
                        let eng = self.engine.as_mut().expect("engine ensured above");
                        return eng
                            .explain_sql(sql)
                            .map(|_| "Hybrid".to_string())
                            .map_err(|e| {
                                Error::invalid_input(format!(
                                    "Craton Bolt cannot plan this query: {plan_err}; {e}"
                                ))
                            });
                    }
                }
            } else {
                match self.plan_without_gpu(sql) {
                    Ok(_) => String::new(),
                    Err(e) => format!("; Bolt planner: {e}"),
                }
            };
            self.fallback_db()?;
            let db = self.fallback_db.as_mut().expect("fallback db opened above");
            db.bind(&self.catalog, sql).map_err(|e| {
                Error::invalid_input(format!(
                    "Query cannot be run on the host fallback: {e}{plan_note}"
                ))
            })?;
            Ok("Host".to_string())
        }
        #[cfg(target_arch = "wasm32")]
        {
            let _ = sql;
            Err(Error::invalid_input(
                "Craton Bolt execution tier inspection is not supported on wasm32 targets",
            ))
        }
    }

    /// Telemetry. Determining `backend` requires knowing whether an engine can be built, so the
    /// first call may initialize CUDA; a failure is cached, so later calls are cheap (and are
    /// answered from the published status, see [`BoltRuntime::telemetry`]).
    fn handle_telemetry(&mut self) -> BoltTelemetrySnapshot {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let gpu_ok = self.ensure_engine().is_ok();
            native_telemetry(
                &self.config,
                self.catalog.len(),
                gpu_ok,
                self.last_engine_error.as_deref(),
                self.fallback_reason(),
            )
        }
        #[cfg(target_arch = "wasm32")]
        {
            BoltTelemetrySnapshot {
                backend: "wasm-stub".to_string(),
                device_id: self.config.device,
                device_name: "wasm32".to_string(),
                memory_budget_bytes: self.config.memory_budget.unwrap_or(0) as i64,
                registered_tables: self.catalog.len() as i64,
                total_pooled_bytes: 0,
                bucket_count: 0,
                oom_recovery_count: 0,
                proactive_eviction_count: 0,
                ptx_cache_hits: 0,
                ptx_cache_misses: 0,
                ptx_cache_evictions: 0,
                plan_cache_hits: 0,
                plan_cache_misses: 0,
                plan_cache_evictions: 0,
                fallback_reason: "wasm32 builds have no GPU engine".to_string(),
            }
        }
    }

    /// See [`BoltTelemetrySnapshot::fallback_reason`].
    #[cfg(not(target_arch = "wasm32"))]
    fn fallback_reason(&self) -> String {
        let mut parts = Vec::new();
        if self.engine.is_none()
            && let Some(err) = &self.last_engine_error
        {
            parts.push(err.clone());
        }
        for (name, reason) in &self.engine_upload_errors {
            parts.push(format!("table '{name}' not on GPU: {reason}"));
        }
        parts.join("; ")
    }

    /// Version string. Reports the backend state without initializing the GPU engine:
    /// `not-initialized` until a GPU operation (query, telemetry, reset) has run.
    fn handle_version_info(&mut self) -> String {
        #[cfg(not(target_arch = "wasm32"))]
        let (backend, device_name) = if self.engine.is_some() {
            let device = self.config.device;
            let name = cuda_devices(self.config.device)
                .into_iter()
                .find(|d| d.device_id == device)
                .map(|d| d.name)
                .unwrap_or_else(|| format!("CUDA Device {device}"));
            ("cuda-gpu", name)
        } else if self.engine_init_failed {
            (
                "host-planner-fallback",
                self.last_engine_error
                    .clone()
                    .unwrap_or_else(|| "unavailable".to_string()),
            )
        } else {
            ("not-initialized", "pending".to_string())
        };
        #[cfg(target_arch = "wasm32")]
        let (backend, device_name) = ("wasm-stub", "wasm32".to_string());
        #[cfg(not(target_arch = "wasm32"))]
        let note = self
            .ptx_cache_note
            .as_ref()
            .filter(|_| self.engine.is_some())
            .map(|note| format!("; {note}"))
            .unwrap_or_default();
        #[cfg(target_arch = "wasm32")]
        let note = String::new();
        format!(
            "duckdb-bolt-extension v{} (craton-bolt v{CRATON_BOLT_VERSION}, backend={backend}, device={}: {device_name}){note}",
            env!("CARGO_PKG_VERSION"),
            self.config.device,
        )
    }
}

/// Telemetry from the engine state (`gpu_ok`, the cached error, the fallback reason) plus
/// craton-bolt's process-global pool and cache counters. Runs on any thread: everything else
/// it reads is process-global or a driver query (W1-B-004).
#[cfg(not(target_arch = "wasm32"))]
fn native_telemetry(
    config: &BoltConfig,
    registered_tables: usize,
    gpu_ok: bool,
    last_engine_error: Option<&str>,
    fallback_reason: String,
) -> BoltTelemetrySnapshot {
    let device = config.device;
    let device_name = cuda_devices(device)
        .into_iter()
        .find(|d| d.device_id == device)
        .map(|d| d.name)
        .unwrap_or_else(|| last_engine_error.unwrap_or("unavailable").to_string());
    let pool = craton_bolt::pool_stats();
    let (ptx_hits, ptx_misses, ptx_evictions) = craton_bolt::jit::ptx_cache_stats();
    let (plan_hits, plan_misses, plan_evictions) =
        craton_bolt::plan::sql_frontend::plan_cache_stats();
    BoltTelemetrySnapshot {
        backend: if gpu_ok {
            "cuda-gpu".to_string()
        } else {
            "host-planner-fallback".to_string()
        },
        device_id: device,
        device_name,
        memory_budget_bytes: config.memory_budget.unwrap_or(0) as i64,
        registered_tables: registered_tables as i64,
        total_pooled_bytes: pool.total_pooled_bytes as i64,
        bucket_count: pool.bucket_count as i64,
        oom_recovery_count: pool.oom_recovery_count as i64,
        proactive_eviction_count: pool.proactive_eviction_count as i64,
        ptx_cache_hits: ptx_hits as i64,
        ptx_cache_misses: ptx_misses as i64,
        ptx_cache_evictions: ptx_evictions as i64,
        plan_cache_hits: plan_hits as i64,
        plan_cache_misses: plan_misses as i64,
        plan_cache_evictions: plan_evictions as i64,
        fallback_reason,
    }
}

/// Execution metrics rows and Prometheus text from craton-bolt's process-global counters.
fn collect_metrics() -> (Vec<MetricRow>, String) {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let snap = craton_bolt::metrics_snapshot();
        let prom = craton_bolt::render_prometheus(&snap);
        let mut rows = Vec::new();
        for (name, val) in snap.counters() {
            rows.push(MetricRow {
                metric: name.to_string(),
                category: "counter".to_string(),
                value: i64::try_from(val).unwrap_or(i64::MAX),
            });
        }
        for hist in snap.histograms() {
            let phase = hist.phase.as_str();
            rows.push(MetricRow {
                metric: format!("{phase}_latency_sum"),
                category: "histogram_ns".to_string(),
                value: i64::try_from(hist.sum_micros.saturating_mul(1_000)).unwrap_or(i64::MAX),
            });
            rows.push(MetricRow {
                metric: format!("{phase}_latency_count"),
                category: "histogram_count".to_string(),
                value: i64::try_from(hist.count).unwrap_or(i64::MAX),
            });
        }
        rows.sort_by(|a, b| a.metric.cmp(&b.metric));
        (rows, prom)
    }
    #[cfg(target_arch = "wasm32")]
    {
        (Vec::new(), String::new())
    }
}

/// Enumerate CUDA devices with the driver API; `active_device` sets `is_active`.
///
/// Uses craton-bolt's safe driver wrappers (`cuda_sys::{init, device_count, device_get,
/// device_name, device_total_mem}`), so this crate needs no `unsafe` for it (W3-C-001):
/// `init` latches a successful `cuInit` like the engine does, and the name buffer and its
/// termination are handled upstream. A device whose memory cannot be queried reports
/// `total_memory_bytes = 0`; a device whose name cannot be read is called `CUDA Device <n>`.
fn cuda_devices(active_device: i32) -> Vec<DeviceInfo> {
    #[cfg(all(not(target_arch = "wasm32"), not(feature = "cuda-stub")))]
    {
        use craton_bolt::cuda::cuda_sys;
        if cuda_sys::init().is_err() {
            return Vec::new();
        }
        let Ok(count) = cuda_sys::device_count() else {
            return Vec::new();
        };
        (0..count.max(0))
            .filter_map(|ordinal| {
                let dev = cuda_sys::device_get(ordinal).ok()?;
                Some(DeviceInfo {
                    device_id: ordinal,
                    name: cuda_sys::device_name(dev)
                        .unwrap_or_else(|_| format!("CUDA Device {ordinal}")),
                    total_memory_bytes: cuda_sys::device_total_mem(dev)
                        .map_or(0, |bytes| i64::try_from(bytes).unwrap_or(i64::MAX)),
                    is_active: ordinal == active_device,
                })
            })
            .collect()
    }
    #[cfg(any(target_arch = "wasm32", feature = "cuda-stub"))]
    {
        let _ = active_device;
        Vec::new()
    }
}

/// Coerce an appended batch to the stored schema of `table_name` (W1-E-001).
///
/// Columns are matched by name (exact first, then ASCII case-insensitive); a stored column
/// missing from the batch, or a batch column the table does not have, is an error. Types are
/// converted with checked casts: a value that does not fit is an error, never NULL.
///
/// When a stored column records its original DuckDB type (RFC-008 `duckdb.logical_type`
/// metadata), every appended value must also fit that type: appending `70000` to a column
/// copied from `SMALLINT` is an error, although both store `Int32` (W2-C-002).
fn coerce_batch_to_schema(
    table_name: &str,
    batch: RecordBatch,
    target_schema: &SchemaRef,
) -> Result<RecordBatch> {
    let coerced = coerce_batch_columns(table_name, batch, target_schema)?;
    for (field, col) in target_schema.fields().iter().zip(coerced.columns()) {
        if crate::arrow_bridge::original_duckdb_type(field).is_none() {
            continue;
        }
        crate::arrow_bridge::restore_column_to_original_type(col, field).map_err(|e| {
            Error::invalid_input(format!(
                "Cannot append to Bolt table '{table_name}': column '{}' was registered as {}: \
                 {}",
                field.name(),
                field
                    .metadata()
                    .get(crate::arrow_bridge::DUCKDB_TYPE_METADATA_KEY)
                    .map_or("its original type", String::as_str),
                e.message
            ))
        })?;
    }
    Ok(coerced)
}

fn coerce_batch_columns(
    table_name: &str,
    batch: RecordBatch,
    target_schema: &SchemaRef,
) -> Result<RecordBatch> {
    if batch.schema() == *target_schema {
        return Ok(batch);
    }
    let source = batch.schema();
    let describe = |schema: &Schema| {
        schema
            .fields()
            .iter()
            .map(|f| format!("{} {}", f.name(), f.data_type()))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let mismatch = |detail: String| {
        Error::invalid_input(format!(
            "Schema mismatch when appending to Bolt table '{table_name}': {detail} \
             (table: [{}]; appended: [{}])",
            describe(target_schema),
            describe(&source)
        ))
    };
    let mut used = vec![false; source.fields().len()];
    let mut cols = Vec::with_capacity(target_schema.fields().len());
    let options = checked_cast_options();
    for target_field in target_schema.fields() {
        let name = target_field.name();
        let idx = match source.fields().iter().position(|f| f.name() == name) {
            Some(idx) => idx,
            None => {
                let mut matches = source
                    .fields()
                    .iter()
                    .enumerate()
                    .filter(|(_, f)| f.name().eq_ignore_ascii_case(name));
                match (matches.next(), matches.next()) {
                    (Some((idx, _)), None) => idx,
                    (Some(_), Some(_)) => {
                        return Err(mismatch(format!(
                            "column '{name}' matches several appended columns"
                        )));
                    }
                    (None, _) => return Err(mismatch(format!("column '{name}' is missing"))),
                }
            }
        };
        if used[idx] {
            return Err(mismatch(format!(
                "appended column '{}' matches several table columns",
                source.field(idx).name()
            )));
        }
        used[idx] = true;
        let src_col = batch.column(idx);
        if src_col.data_type() == target_field.data_type() {
            cols.push(src_col.clone());
        } else {
            let casted =
                cast_with_options(src_col, target_field.data_type(), &options).map_err(|e| {
                    mismatch(format!(
                        "cannot convert column '{name}' from {:?} to {:?}: {e}",
                        src_col.data_type(),
                        target_field.data_type()
                    ))
                })?;
            cols.push(casted);
        }
    }
    if let Some(extra) = used.iter().position(|u| !u) {
        return Err(mismatch(format!(
            "the table has no column '{}'",
            source.field(extra).name()
        )));
    }
    RecordBatch::try_new(target_schema.clone(), cols).map_err(|e| {
        Error::invalid_input(format!(
            "Failed to build coerced RecordBatch for Bolt table '{table_name}': {e}"
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_strips_only_quotes_and_bolt_prefixes() {
        assert_eq!(normalize_table_reference("bolt.orders"), "orders");
        assert_eq!(normalize_table_reference("BOLT.main.orders"), "orders");
        assert_eq!(normalize_table_reference("'bolt:orders'"), "orders");
        assert_eq!(normalize_table_reference("bolt://orders"), "orders");
        assert_eq!(normalize_table_reference("bolt.\"My T\""), "My T");
        assert_eq!(normalize_table_reference("tmp_orders"), "tmp_orders");
        assert_eq!(normalize_table_reference("orders.bolt"), "orders.bolt");
        assert_eq!(normalize_table_reference("a/b"), "a/b");
        assert_eq!(normalize_table_reference("boltx"), "boltx");
    }

    #[test]
    fn clean_table_identifier_keeps_copy_path_semantics() {
        assert_eq!(clean_table_identifier("dir/tmp_orders.bolt"), "orders");
        assert_eq!(clean_table_identifier("bolt:orders"), "orders");
        assert_eq!(clean_table_identifier("tmp_x"), "x");
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn sql_names_match_whole_identifiers_only() {
        let names = SqlNames::new("SELECT \"My Tab\".x, s FROM __bolt_x_10 JOIN query_table('q')");
        assert!(names.may_reference("__bolt_x_10"));
        assert!(!names.may_reference("__bolt_x_1"));
        // `select` contains `t`, `from` contains `f`: whole words only (W2-E-007).
        assert!(!names.may_reference("t"));
        assert!(!names.may_reference("f"));
        assert!(names.may_reference("S"));
        assert!(names.may_reference("q"), "string literals still count");
        assert!(
            names.may_reference("my tab"),
            "names with separators use substrings"
        );
        assert_eq!(
            SqlNames::new("a = 1 OR b = 2 AND c = 3 or d").predicate_terms(),
            3
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn only_real_bolt_calls_count_as_reentrant() {
        let calls = |sql: &str| SqlNames::new(sql).may_call_bolt(|w| w == "bolt_tbl");
        // W4-D-002: mentions are not calls.
        assert!(!calls("SELECT bolt_flag FROM t WHERE bolt_flag > 3"));
        assert!(!calls("SELECT count(*) AS bolt_rows FROM t"));
        assert!(!calls(
            "SELECT * FROM t WHERE url LIKE '%bolt.com%' OR note = 'bolt: done'"
        ));
        assert!(!calls(
            "SELECT \"bolt_version()\" FROM t -- bolt_version()\n"
        ));
        assert!(!calls("SELECT x FROM t /* bolt.x bolt_scan( */"));
        assert!(
            !calls("SELECT * FROM bolt_tbl (a)"),
            "a catalog table, not a function"
        );
        // Real calls into the extension.
        assert!(calls("SELECT bolt_version()"));
        assert!(calls("SELECT * FROM BOLT_SCAN ('x')"));
        assert!(calls("SELECT * FROM bolt.t"));
        assert!(calls("SELECT * FROM \"bolt\".main.t"));
        assert!(calls("SELECT * FROM 'bolt:t'"));
        assert!(calls("SELECT * FROM a JOIN \"bolt:t\" ON true"));
        assert_eq!(
            sql_code_tokens("'it''s' \"a\"\"b\""),
            vec![
                SqlToken::Str("it's".to_string()),
                SqlToken::Quoted("a\"b".to_string())
            ]
        );
    }

    #[test]
    fn user_facing_messages_have_no_runs_of_spaces() {
        // W4-D-004: a string literal wrapped without a trailing `\` keeps the indentation.
        let sources = [
            ("runtime.rs", include_str!("runtime.rs")),
            (
                "runtime/engine_support.rs",
                include_str!("runtime/engine_support.rs"),
            ),
            ("runtime/fallback.rs", include_str!("runtime/fallback.rs")),
            ("runtime/sql_text.rs", include_str!("runtime/sql_text.rs")),
        ];
        for (file, no, line) in sources
            .iter()
            .flat_map(|(file, src)| src.lines().enumerate().map(move |(no, l)| (file, no, l)))
        {
            let code = line.trim_start();
            if code.starts_with("//") {
                continue;
            }
            let run = " ".repeat(3);
            let quoted = code.split('"').skip(1).step_by(2);
            for text in quoted {
                assert!(!text.contains(&run), "{file}:{}: {line}", no + 1);
            }
        }
        #[cfg(not(target_arch = "wasm32"))]
        assert!(!REENTRANT_CALL_MESSAGE.contains("  "));
        assert!(!RegisterMode::invalid_mode_message("x").contains("  "));
    }

    #[test]
    fn reserved_and_column_names_are_checked() {
        assert!(check_table_name_not_reserved("__BOLT_x").is_err());
        assert!(check_table_name_not_reserved("_bolt_x").is_ok());
        let dup = Schema::new(vec![
            arrow_schema::Field::new("a", ArrowDataType::Int32, true),
            arrow_schema::Field::new("a", ArrowDataType::Int32, true),
        ]);
        assert!(check_column_names("t", &dup).is_err());
        let case = Schema::new(vec![
            arrow_schema::Field::new("a", ArrowDataType::Int32, true),
            arrow_schema::Field::new("A", ArrowDataType::Int32, true),
        ]);
        assert!(check_column_names("t", &case).is_ok());
        let empty = Schema::new(vec![arrow_schema::Field::new(
            " ",
            ArrowDataType::Int32,
            true,
        )]);
        assert!(check_column_names("t", &empty).is_err());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn utf8_bytes_count_sliced_arrays_by_offsets() {
        let schema = Arc::new(Schema::new(vec![
            arrow_schema::Field::new("s", ArrowDataType::Utf8, true),
            arrow_schema::Field::new("i", ArrowDataType::Int32, true),
        ]));
        let strings: ArrayRef = Arc::new(arrow_array::StringArray::from(vec!["ab", "cde", "f"]));
        let ints: ArrayRef = Arc::new(arrow_array::Int32Array::from(vec![1, 2, 3]));
        let batch = RecordBatch::try_new(schema.clone(), vec![strings, ints]).unwrap();
        let sliced = batch.slice(1, 2);
        assert_eq!(utf8_bytes_per_column(2, [&batch, &sliced]), vec![6 + 4, 0]);
        assert!(check_engine_utf8_capacity("t", &schema, [&batch, &sliced]).is_ok());
        let combined = coalesce_for_engine("t", &[batch, sliced]).unwrap();
        assert_eq!(combined.num_rows(), 5);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn dictionary_columns_are_planned_as_their_value_type() {
        use craton_bolt::plan::DataType as P;
        let schema = Schema::new(vec![
            arrow_schema::Field::new(
                "k",
                ArrowDataType::Dictionary(
                    Box::new(ArrowDataType::Int32),
                    Box::new(ArrowDataType::Int64),
                ),
                true,
            ),
            arrow_schema::Field::new(
                "s",
                ArrowDataType::Dictionary(
                    Box::new(ArrowDataType::Int32),
                    Box::new(ArrowDataType::Utf8),
                ),
                true,
            ),
        ]);
        let plan = plan_schema_of(&schema).unwrap();
        assert_eq!(plan.fields[0].dtype, P::Int64);
        assert_eq!(plan.fields[1].dtype, P::Utf8);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn dictionary_lowered_predicates_are_detected() {
        let schema = Schema::new(vec![
            arrow_schema::Field::new("id", ArrowDataType::Int64, true),
            arrow_schema::Field::new("s", ArrowDataType::Utf8, true),
        ]);
        let mut provider = craton_bolt::plan::MemTableProvider::new();
        provider.register("t".to_string(), plan_schema_of(&schema).unwrap());
        let lowered = |sql: &str| {
            let plan = craton_bolt::plan::parse_sql(sql, &provider).unwrap();
            plan_has_dictionary_lowered_predicate(&plan)
        };
        assert!(lowered("SELECT id FROM t WHERE s LIKE 'a%'"));
        assert!(lowered("SELECT id FROM t WHERE s > 'k'"));
        assert!(lowered("SELECT id FROM t WHERE 'k' <= s"));
        assert!(lowered("SELECT id FROM t WHERE id = 1 OR s < 'it''s'"));
        assert!(!lowered("SELECT id FROM t WHERE s = 'k'"));
        assert!(!lowered("SELECT id FROM t WHERE id > 3"));
    }

    #[test]
    fn fractional_literals_are_found_in_code_only() {
        assert!(has_fractional_literal("SELECT id * 0.1 FROM t"));
        assert!(has_fractional_literal("SELECT id FROM t WHERE x > .5"));
        assert!(has_fractional_literal("SELECT (12.25)"));
        assert!(!has_fractional_literal("SELECT t.c, \"t\".c FROM t"));
        assert!(!has_fractional_literal("SELECT 1e3, 1.5e3, 2 FROM t"));
        assert!(!has_fractional_literal(
            "SELECT '0.1' -- 0.1\n FROM t /* .5 */"
        ));
    }

    #[test]
    fn column_facts_follow_renames_unions_and_null_literals() {
        use arrow_array::Int64Array;
        let schema = Arc::new(Schema::new(vec![
            arrow_schema::Field::new("v", ArrowDataType::Int64, true),
            arrow_schema::Field::new("w", ArrowDataType::Int64, true),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(vec![Some(1), Some(-7)])) as ArrayRef,
                Arc::new(Int64Array::from(vec![Some(3), None])) as ArrayRef,
            ],
        )
        .unwrap();
        let mut catalog = BTreeMap::new();
        catalog.insert(
            "t".to_string(),
            RegisteredTable::from_batches(schema.clone(), vec![batch]),
        );
        let tables = GuardTables::new(&catalog, None);
        let mut provider = craton_bolt::plan::MemTableProvider::new();
        provider.register("t".to_string(), plan_schema_of(&schema).unwrap());
        let facts = |sql: &str| {
            let plan = craton_bolt::plan::parse_sql(sql, &provider).unwrap();
            let schema = plan.schema().unwrap();
            let node = NodeFacts::of(&plan, &tables);
            schema
                .fields
                .iter()
                .map(|f| node.expr(&craton_bolt::plan::Expr::Column(f.name.clone())))
                .map(|f| (f.nullable, f.abs_max))
                .collect::<Vec<_>>()
        };
        assert_eq!(
            facts("SELECT v, w FROM t"),
            [(false, Some(7)), (true, Some(3))]
        );
        assert_eq!(facts("SELECT w AS v FROM t"), [(true, Some(3))]);
        assert_eq!(
            facts("SELECT CAST(NULL AS BIGINT) AS v FROM t"),
            [(true, Some(0))]
        );
        assert_eq!(
            facts("SELECT v FROM t UNION ALL SELECT w FROM t"),
            [(true, Some(7))]
        );
        assert_eq!(facts("SELECT v * 2 + 1 AS x FROM t"), [(true, Some(15))]);
        let plan = craton_bolt::plan::parse_sql("SELECT v FROM t LIMIT 0", &provider).unwrap();
        assert!(may_be_empty(&plan, &tables));
        let plan = craton_bolt::plan::parse_sql("SELECT v FROM t LIMIT 1", &provider).unwrap();
        assert!(!may_be_empty(&plan, &tables));
        assert_eq!(row_bound(&plan, &tables), Some(1));
    }

    #[test]
    fn craton_bolt_version_matches_cargo_lock() {
        let lock = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../Cargo.lock");
        let Ok(text) = std::fs::read_to_string(&lock) else {
            // Packaged crate (crates.io) has no workspace lock file.
            return;
        };
        let mut lines = text.lines();
        let mut found = None;
        while let Some(line) = lines.next() {
            if line.trim() == "name = \"craton-bolt\"" {
                found = lines
                    .next()
                    .and_then(|v| v.trim().strip_prefix("version = \""))
                    .and_then(|v| v.strip_suffix('"'))
                    .map(str::to_string);
                break;
            }
        }
        let locked = found.expect("craton-bolt entry in Cargo.lock");
        assert_eq!(
            locked, CRATON_BOLT_VERSION,
            "update CRATON_BOLT_VERSION in runtime.rs"
        );
    }

    #[test]
    fn normalize_zoned_timestamps_to_naive_utc() {
        use crate::arrow_bridge::DUCKDB_TYPE_METADATA_KEY as KEY;
        use arrow_array::{
            TimestampMicrosecondArray, TimestampMillisecondArray, TimestampNanosecondArray,
            cast::AsArray, types::TimestampMicrosecondType, types::TimestampNanosecondType,
        };
        use arrow_schema::{Field, TimeUnit};
        let kept: HashMap<String, String> = [(KEY.to_string(), "TIMESTAMP".to_string())]
            .into_iter()
            .collect();
        let schema = Arc::new(Schema::new(vec![
            Field::new(
                "ms",
                ArrowDataType::Timestamp(TimeUnit::Millisecond, Some("Europe/Berlin".into())),
                true,
            ),
            Field::new(
                "ns",
                ArrowDataType::Timestamp(TimeUnit::Nanosecond, Some("+02:00".into())),
                true,
            ),
            Field::new(
                "us_meta",
                ArrowDataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                true,
            )
            .with_metadata(kept),
            Field::new(
                "naive",
                ArrowDataType::Timestamp(TimeUnit::Millisecond, None),
                true,
            ),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(
                    TimestampMillisecondArray::from(vec![1_500]).with_timezone("Europe/Berlin"),
                ),
                Arc::new(TimestampNanosecondArray::from(vec![7]).with_timezone("+02:00")),
                Arc::new(TimestampMicrosecondArray::from(vec![9]).with_timezone("UTC")),
                Arc::new(TimestampMillisecondArray::from(vec![2])),
            ],
        )
        .expect("batch");
        let out = normalize_batch_for_bolt(&batch).expect("normalize");
        let s = out.schema();
        assert_eq!(
            s.field(0).data_type(),
            &ArrowDataType::Timestamp(TimeUnit::Microsecond, None)
        );
        assert_eq!(
            s.field(0).metadata().get(KEY).map(String::as_str),
            Some("TIMESTAMPTZ")
        );
        assert_eq!(
            out.column(0)
                .as_primitive::<TimestampMicrosecondType>()
                .value(0),
            1_500_000
        );
        assert_eq!(
            s.field(1).data_type(),
            &ArrowDataType::Timestamp(TimeUnit::Nanosecond, None)
        );
        assert_eq!(
            s.field(1).metadata().get(KEY).map(String::as_str),
            Some("TIMESTAMP_TZ_NS")
        );
        assert_eq!(
            out.column(1)
                .as_primitive::<TimestampNanosecondType>()
                .value(0),
            7
        );
        // An existing key is kept, even when it disagrees with the zone.
        assert_eq!(
            s.field(2).data_type(),
            &ArrowDataType::Timestamp(TimeUnit::Microsecond, None)
        );
        assert_eq!(
            s.field(2).metadata().get(KEY).map(String::as_str),
            Some("TIMESTAMP")
        );
        // Naive timestamps keep their unit and get no metadata (unchanged behavior).
        assert_eq!(
            s.field(3).data_type(),
            &ArrowDataType::Timestamp(TimeUnit::Millisecond, None)
        );
        assert!(!s.field(3).metadata().contains_key(KEY));
    }
}

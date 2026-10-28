// SPDX-License-Identifier: Apache-2.0

//! DuckDB scalar functions for Craton Bolt introspection and catalog management.
//!
//! Row semantics: every function writes exactly `input.row_count()` rows, as the
//! `ScalarCallbacks::exec` contract requires. All functions are `VOLATILE`, so DuckDB
//! passes the real row count even for zero-argument calls: `SELECT bolt_version()`
//! evaluates one row, `SELECT bolt_version() FROM range(n)` evaluates `n` rows.
//!
//! NULL semantics: any NULL argument yields a NULL result and no side effect. This
//! matches DuckDB's default NULL handling, which already short-circuits *constant*
//! NULL arguments before the callback runs; treating a NULL `mode` as `'replace'`
//! would therefore only apply to non-constant NULLs and be inconsistent. Omit the
//! `mode` argument to get the default (`'replace'`).
//!
//! `mode` (`bolt_register_sql`, `bolt_load_parquet`, `bolt_load_arrow_ipc`) is parsed by
//! [`RegisterMode::parse`], exactly like `COPY ... (FORMAT bolt, MODE ...)`: `'replace'`
//! (alias `'overwrite'`), `'append'`, or `'create'` (alias `'error'`), case-insensitive,
//! surrounding whitespace ignored. It is validated before any file is read or query run.
//!
//! Side-effecting functions (`bolt_register_sql`, `bolt_load_*`, `bolt_drop_table`)
//! act once per non-NULL input row. Errors name the function and the row's table.
//! See `docs/issues/W1-D-005-side-effecting-scalars.md` for the planner-dependent parts.
//! For exactly one side effect per execution, use the table-function forms of the same names:
//! `CALL bolt_load_parquet('t', 'f.parquet')`, `SELECT * FROM bolt_clear_tables()`, ...
//! (`table_functions.rs`, W4-C-004).

use std::fmt::Display;

use duckdb_neo::{
    DuckDBType, Error, Result,
    connection::Context,
    data_chunk::VectorCollection,
    enums::{FunctionProperty, FunctionPropertyStability},
    extension::Extension,
    scalar::{ScalarCallbacks, ScalarFunctionBuilder, ScalarInitCallbacks},
    signature::{FunctionDocumentationBuilder, FunctionParameter, SignatureBuilder},
    vector::{Unknown, Vector},
};

#[cfg(not(target_arch = "wasm32"))]
use duckdb_neo::file::FileSystem;

#[cfg(not(target_arch = "wasm32"))]
use crate::ingest;
use crate::runtime::{BoltRuntime, RegisterMode};
#[cfg(not(target_arch = "wasm32"))]
use crate::sandbox::FileAccessPolicy;

/// Prefix an error message with call-site context, keeping its DuckDB error code.
fn annotate(err: Error, context: impl Display) -> Error {
    Error {
        code: err.code,
        message: format!("{context}: {}", err.message),
    }
}

/// Reconcile the calling database's `bolt_*` settings with the runtime before a call that
/// evaluates rows (startup options, stale settings, PTX cache use-time check; see
/// [`crate::options::sync_with_runtime`]).
fn sync_options(context: &Context, rows: usize, function: &str) -> Result<()> {
    if rows == 0 {
        return Ok(());
    }
    crate::options::sync_with_runtime(context).map_err(|e| annotate(e, function))
}

/// Saturating `usize -> BIGINT` conversion for row and table counts.
fn to_i64(n: usize) -> i64 {
    i64::try_from(n).unwrap_or(i64::MAX)
}

struct BoltVersionScalar;

impl ScalarCallbacks for BoltVersionScalar {
    type BindData = ();
    type InitData = ();

    fn exec(
        &self,
        _bind_data: Option<&Self::BindData>,
        _init_data: Option<&mut Self::InitData>,
        context: &Context,
        input: &mut VectorCollection,
        output: Vector<'_, Unknown>,
    ) -> Result<()> {
        let rows = input.row_count()?;
        sync_options(context, rows, "bolt_version")?;
        let mut out = output.cast::<String>()?;
        out.set_size(rows)?;
        if rows == 0 {
            return Ok(());
        }
        let version = BoltRuntime::global().version_info()?;
        let mut writer = out.writer()?;
        for row in 0..rows {
            writer.set(row, Some(version.as_str()))?;
        }
        Ok(())
    }
}

struct BoltExplainScalar;

impl ScalarCallbacks for BoltExplainScalar {
    type BindData = ();
    type InitData = ();

    fn exec(
        &self,
        _bind_data: Option<&Self::BindData>,
        _init_data: Option<&mut Self::InitData>,
        context: &Context,
        input: &mut VectorCollection,
        output: Vector<'_, Unknown>,
    ) -> Result<()> {
        let rows = input.row_count()?;
        sync_options(context, rows, "bolt_explain")?;
        let sql_vec = input.get_vector_at::<String>(0)?;
        let reader = sql_vec.reader()?;
        let mut out = output.cast::<String>()?;
        out.set_size(rows)?;
        let mut writer = out.writer()?;
        for row in 0..rows {
            match reader.get(row)? {
                Some(sql) => {
                    let plan = BoltRuntime::global()
                        .explain_sql(sql)
                        .map_err(|e| annotate(e, "bolt_explain"))?;
                    writer.set(row, Some(plan.as_str()))?;
                }
                None => writer.set(row, None)?,
            }
        }
        Ok(())
    }
}

struct BoltExecutionTierScalar;

impl ScalarCallbacks for BoltExecutionTierScalar {
    type BindData = ();
    type InitData = ();

    fn exec(
        &self,
        _bind_data: Option<&Self::BindData>,
        _init_data: Option<&mut Self::InitData>,
        context: &Context,
        input: &mut VectorCollection,
        output: Vector<'_, Unknown>,
    ) -> Result<()> {
        let rows = input.row_count()?;
        sync_options(context, rows, "bolt_execution_tier")?;
        let sql_vec = input.get_vector_at::<String>(0)?;
        let reader = sql_vec.reader()?;
        let mut out = output.cast::<String>()?;
        out.set_size(rows)?;
        let mut writer = out.writer()?;
        for row in 0..rows {
            match reader.get(row)? {
                Some(sql) => {
                    let tier = BoltRuntime::global()
                        .execution_tier(sql)
                        .map_err(|e| annotate(e, "bolt_execution_tier"))?;
                    writer.set(row, Some(tier.as_str()))?;
                }
                None => writer.set(row, None)?,
            }
        }
        Ok(())
    }
}

/// `bolt_register_sql(table_name, sql [, mode])` -> BIGINT rows registered.
struct BoltRegisterSqlScalar {
    with_mode: bool,
}

impl ScalarCallbacks for BoltRegisterSqlScalar {
    type BindData = ();
    type InitData = ();

    fn exec(
        &self,
        _bind_data: Option<&Self::BindData>,
        _init_data: Option<&mut Self::InitData>,
        context: &Context,
        input: &mut VectorCollection,
        output: Vector<'_, Unknown>,
    ) -> Result<()> {
        let rows = input.row_count()?;
        sync_options(context, rows, "bolt_register_sql")?;
        let name_vec = input.get_vector_at::<String>(0)?;
        let sql_vec = input.get_vector_at::<String>(1)?;
        let name_reader = name_vec.reader()?;
        let sql_reader = sql_vec.reader()?;
        let mode_vec = if self.with_mode {
            Some(input.get_vector_at::<String>(2)?)
        } else {
            None
        };
        let mode_reader = match &mode_vec {
            Some(v) => Some(v.reader()?),
            None => None,
        };
        let mut out = output.cast::<i64>()?;
        out.set_size(rows)?;
        let mut writer = out.writer()?;
        for row in 0..rows {
            let mode_text = match &mode_reader {
                Some(r) => r.get(row)?,
                None => Some("replace"),
            };
            match (name_reader.get(row)?, sql_reader.get(row)?, mode_text) {
                (Some(table_name), Some(sql), Some(mode)) => {
                    let ctx = || format!("bolt_register_sql('{table_name}')");
                    // Validate the mode before running the query.
                    let mode = RegisterMode::parse(mode).map_err(|e| annotate(e, ctx()))?;
                    let rows = register_sql_once(table_name, sql, mode)?;
                    writer.set(row, Some(to_i64(rows)))?;
                }
                _ => writer.set(row, None)?,
            }
        }
        Ok(())
    }
}

/// File formats supported by `bolt_load_parquet` / `bolt_load_arrow_ipc` (RFC-004).
#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum IngestFormat {
    Parquet,
    ArrowIpc,
}

#[cfg(not(target_arch = "wasm32"))]
impl IngestFormat {
    pub(crate) fn function_name(self) -> &'static str {
        match self {
            Self::Parquet => "bolt_load_parquet",
            Self::ArrowIpc => "bolt_load_arrow_ipc",
        }
    }

    fn read(
        self,
        path: &str,
        policy: &FileAccessPolicy,
        vfs: Option<&FileSystem<'_>>,
        limit: Option<u64>,
    ) -> Result<ingest::IngestedTable> {
        match self {
            Self::Parquet => ingest::read_parquet_limited(path, policy, vfs, limit),
            // `limit` also bounds what compressed IPC buffers may declare as their
            // uncompressed size (W4-C-001).
            Self::ArrowIpc => ingest::read_arrow_ipc_limited(path, policy, vfs, limit),
        }
    }
}

/// Load one file into `table_name` with the calling connection's file-access settings, exactly
/// as one row of `bolt_load_parquet` / `bolt_load_arrow_ipc` does. Used by the `CALL` forms
/// (`table_functions.rs`, W4-C-004). Returns the registered table's row count.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn load_file_once(
    context: &Context,
    format: IngestFormat,
    table_name: &str,
    path: &str,
    mode: RegisterMode,
) -> Result<usize> {
    let ctx = || format!("{}('{table_name}', '{path}')", format.function_name());
    let policy = FileAccessPolicy::from_context(context).map_err(|e| annotate(e, ctx()))?;
    let vfs = FileSystem::new(context).map_err(|e| annotate(e, ctx()))?;
    let limit = ingest::memory_limit_bytes(context);
    let table = format
        .read(path, &policy, Some(&vfs), limit)
        .map_err(|e| annotate(e, ctx()))?;
    table
        .register(table_name, mode)
        .map(|summary| summary.num_rows)
        .map_err(|e| annotate(e, ctx()))
}

/// Run `sql` on Craton Bolt and register the result as `table_name`, exactly as one row of
/// `bolt_register_sql` does (also used by its `CALL` form, W4-C-004). Returns the registered
/// table's row count.
pub(crate) fn register_sql_once(table_name: &str, sql: &str, mode: RegisterMode) -> Result<usize> {
    let ctx = || format!("bolt_register_sql('{table_name}')");
    // The result's batches are registered as they are (no concatenation, W4-A-002).
    let exec_out = BoltRuntime::global()
        .execute_sql_batches(sql)
        .map_err(|e| annotate(e, ctx()))?;
    BoltRuntime::global()
        .register_table_batches(table_name, exec_out.schema, exec_out.batches, mode)
        .map(|summary| summary.num_rows)
        .map_err(|e| annotate(e, ctx()))
}

/// `bolt_load_parquet` / `bolt_load_arrow_ipc` (table_name, path [, mode]) -> BIGINT rows loaded.
///
/// Local paths are checked against DuckDB's `enable_external_access` / `allowed_directories` /
/// `allowed_paths` of the calling connection (a refused path is a `Permission Error` and
/// registers nothing) and are also opened once through DuckDB's file system, so
/// `disabled_filesystems` applies. URLs (`s3://`, `https://`, ...) are read through DuckDB's
/// file system, with DuckDB's own access checks, secrets and extensions (httpfs).
#[cfg(not(target_arch = "wasm32"))]
struct BoltLoadFileScalar {
    format: IngestFormat,
    with_mode: bool,
}

#[cfg(not(target_arch = "wasm32"))]
impl ScalarCallbacks for BoltLoadFileScalar {
    type BindData = ();
    type InitData = ();

    fn exec(
        &self,
        _bind_data: Option<&Self::BindData>,
        _init_data: Option<&mut Self::InitData>,
        context: &Context,
        input: &mut VectorCollection,
        output: Vector<'_, Unknown>,
    ) -> Result<()> {
        let rows = input.row_count()?;
        sync_options(context, rows, self.format.function_name())?;
        let name_vec = input.get_vector_at::<String>(0)?;
        let path_vec = input.get_vector_at::<String>(1)?;
        let name_reader = name_vec.reader()?;
        let path_reader = path_vec.reader()?;
        let mode_vec = if self.with_mode {
            Some(input.get_vector_at::<String>(2)?)
        } else {
            None
        };
        let mode_reader = match &mode_vec {
            Some(v) => Some(v.reader()?),
            None => None,
        };
        let mut out = output.cast::<i64>()?;
        out.set_size(rows)?;
        let mut writer = out.writer()?;
        // Settings and the file system are obtained once per chunk, from the calling
        // connection's context, and only when a row actually loads something.
        let mut policy: Option<FileAccessPolicy> = None;
        let mut vfs: Option<FileSystem<'_>> = None;
        let mut buffer_limit: Option<Option<u64>> = None;
        // Rows of one chunk that load the same path reuse the decoded file (W1-D-005 item
        // 4): every row still has its own side effect (register / append), but the file is
        // read and decoded once per chunk instead of once per row.
        let mut decoded: Option<(String, ingest::IngestedTable)> = None;
        let fname = self.format.function_name();
        for row in 0..rows {
            let mode_text = match &mode_reader {
                Some(r) => r.get(row)?,
                None => Some("replace"),
            };
            match (name_reader.get(row)?, path_reader.get(row)?, mode_text) {
                (Some(table_name), Some(path), Some(mode)) => {
                    let ctx = || format!("{fname}('{table_name}', '{path}')");
                    let mode = RegisterMode::parse(mode).map_err(|e| annotate(e, ctx()))?;
                    if policy.is_none() {
                        policy = Some(
                            FileAccessPolicy::from_context(context)
                                .map_err(|e| annotate(e, ctx()))?,
                        );
                    }
                    if vfs.is_none() {
                        vfs = Some(FileSystem::new(context).map_err(|e| annotate(e, ctx()))?);
                    }
                    let policy = policy.as_ref().expect("policy was just initialized");
                    let limit =
                        *buffer_limit.get_or_insert_with(|| ingest::memory_limit_bytes(context));
                    let table = match &decoded {
                        Some((cached_path, table)) if cached_path == path => table.clone(),
                        _ => {
                            let table = self
                                .format
                                .read(path, policy, vfs.as_ref(), limit)
                                .map_err(|e| annotate(e, ctx()))?;
                            decoded = Some((path.to_string(), table.clone()));
                            table
                        }
                    };
                    let summary = table
                        .register(table_name, mode)
                        .map_err(|e| annotate(e, ctx()))?;
                    writer.set(row, Some(to_i64(summary.num_rows)))?;
                }
                _ => writer.set(row, None)?,
            }
        }
        Ok(())
    }
}

struct BoltDropTableScalar;

impl ScalarCallbacks for BoltDropTableScalar {
    type BindData = ();
    type InitData = ();

    fn exec(
        &self,
        _bind_data: Option<&Self::BindData>,
        _init_data: Option<&mut Self::InitData>,
        context: &Context,
        input: &mut VectorCollection,
        output: Vector<'_, Unknown>,
    ) -> Result<()> {
        let rows = input.row_count()?;
        sync_options(context, rows, "bolt_drop_table")?;
        let name_vec = input.get_vector_at::<String>(0)?;
        let reader = name_vec.reader()?;
        let mut out = output.cast::<bool>()?;
        out.set_size(rows)?;
        let mut writer = out.writer()?;
        for row in 0..rows {
            match reader.get(row)? {
                Some(name) => {
                    let dropped = BoltRuntime::global()
                        .drop_table(name)
                        .map_err(|e| annotate(e, format!("bolt_drop_table('{name}')")))?;
                    writer.set(row, Some(dropped))?;
                }
                None => writer.set(row, None)?,
            }
        }
        Ok(())
    }
}

/// `bolt_clear_tables()`: clears the catalog **once per worker thread per query**.
///
/// DuckDB evaluates a scalar once per data chunk, so a per-chunk clear would make
/// `SELECT bolt_clear_tables() FROM range(5000)` clear three times and report `0` for
/// the later chunks. The count removed by the worker's single clear is cached in the
/// worker-local init data and reported for every row that worker evaluates. With
/// several threads, each worker clears once; workers after the first typically report
/// `0`. Use `SELECT bolt_clear_tables()` or `PRAGMA bolt_clear_tables` for a
/// well-defined single clear. A query that evaluates no rows clears nothing.
struct BoltClearTablesScalar;

impl ScalarCallbacks for BoltClearTablesScalar {
    type BindData = ();
    /// Tables removed by this worker's clear, once it has happened.
    type InitData = Option<i64>;

    fn exec(
        &self,
        _bind_data: Option<&Self::BindData>,
        init_data: Option<&mut Self::InitData>,
        context: &Context,
        input: &mut VectorCollection,
        output: Vector<'_, Unknown>,
    ) -> Result<()> {
        let rows = input.row_count()?;
        sync_options(context, rows, "bolt_clear_tables")?;
        let mut out = output.cast::<i64>()?;
        out.set_size(rows)?;
        if rows == 0 {
            return Ok(());
        }
        let clear = || -> Result<i64> {
            BoltRuntime::global()
                .clear_tables()
                .map(to_i64)
                .map_err(|e| annotate(e, "bolt_clear_tables()"))
        };
        let removed = match init_data {
            Some(cached) => {
                let previous: Option<i64> = *cached;
                match previous {
                    Some(n) => n,
                    None => {
                        let n = clear()?;
                        *cached = Some(n);
                        n
                    }
                }
            }
            // Only reachable if the init callback were not registered.
            None => clear()?,
        };
        let mut writer = out.writer()?;
        for row in 0..rows {
            writer.set(row, Some(removed))?;
        }
        Ok(())
    }
}

impl ScalarInitCallbacks for BoltClearTablesScalar {
    fn init(
        &self,
        _bind_data: Option<&Self::BindData>,
        _context: &Context,
    ) -> Result<Self::InitData> {
        Ok(None)
    }
}

struct BoltPrometheusMetricsScalar;

impl ScalarCallbacks for BoltPrometheusMetricsScalar {
    type BindData = ();
    type InitData = ();

    fn exec(
        &self,
        _bind_data: Option<&Self::BindData>,
        _init_data: Option<&mut Self::InitData>,
        context: &Context,
        input: &mut VectorCollection,
        output: Vector<'_, Unknown>,
    ) -> Result<()> {
        let rows = input.row_count()?;
        sync_options(context, rows, "bolt_prometheus_metrics")?;
        let mut out = output.cast::<String>()?;
        out.set_size(rows)?;
        if rows == 0 {
            return Ok(());
        }
        let (_rows, prom) = BoltRuntime::global().metrics()?;
        let mut writer = out.writer()?;
        for row in 0..rows {
            writer.set(row, Some(prom.as_str()))?;
        }
        Ok(())
    }
}

/// Shared description of the `mode` argument (keep in sync with `RegisterMode::parse`).
macro_rules! mode_doc {
    () => {
        "Optional mode: 'replace' (default; alias 'overwrite'), 'append', or 'create' (alias 'error'), \
         case-insensitive. NULL if any argument is NULL."
    };
}

const REGISTER_SQL_DESCRIPTION: &str = concat!(
    "Executes a SQL query on Craton Bolt and registers the result as a GPU table, returning the row count. ",
    mode_doc!()
);

#[cfg(not(target_arch = "wasm32"))]
const LOAD_PARQUET_DESCRIPTION: &str = concat!(
    "Loads a Parquet file (local path, or a URL read through DuckDB's file system) into the Craton Bolt GPU \
     catalog as table_name, returning rows loaded. ",
    mode_doc!(),
    " Honors enable_external_access / allowed_directories / allowed_paths / disabled_filesystems."
);

#[cfg(not(target_arch = "wasm32"))]
const LOAD_ARROW_IPC_DESCRIPTION: &str = concat!(
    "Loads an Arrow IPC file (local path, or a URL read through DuckDB's file system) into the Craton Bolt GPU \
     catalog as table_name, returning rows loaded. ",
    mode_doc!(),
    " Honors enable_external_access / allowed_directories / allowed_paths / disabled_filesystems."
);

/// Register all Craton Bolt scalar functions on `extension`.
pub fn register_scalar_functions(extension: &Extension) -> Result<()> {
    let volatile = FunctionProperty::Stability(FunctionPropertyStability::Volatile);

    ScalarFunctionBuilder::new(
        "bolt_version",
        SignatureBuilder::new()
            .return_type(String::logical_type(extension)?)
            .documentation(
                FunctionDocumentationBuilder::new().description(
                    "Returns the duckdb-bolt-extension version, craton-bolt version, and active GPU device",
                ),
            ),
        BoltVersionScalar,
    )
    .set_property(volatile)
    .register(extension)?;

    ScalarFunctionBuilder::new(
        "bolt_explain",
        SignatureBuilder::new()
            .param(FunctionParameter::new::<String>("sql"))
            .return_type(String::logical_type(extension)?)
            .documentation(
                FunctionDocumentationBuilder::new().description(
                    "Returns the Craton Bolt optimized logical and physical execution plan for a SQL query",
                ),
            ),
        BoltExplainScalar,
    )
    .set_property(volatile)
    .register(extension)?;

    ScalarFunctionBuilder::new(
        "bolt_execution_tier",
        SignatureBuilder::new()
            .param(FunctionParameter::new::<String>("sql"))
            .return_type(String::logical_type(extension)?)
            .documentation(
                FunctionDocumentationBuilder::new().description(
                    "Returns the planned Craton Bolt execution tier ('Gpu', 'Host', or 'Hybrid') for a SQL query",
                ),
            ),
        BoltExecutionTierScalar,
    )
    .set_property(volatile)
    .register(extension)?;

    for with_mode in [false, true] {
        let mut sig = SignatureBuilder::new()
            .param(FunctionParameter::new::<String>("table_name"))
            .param(FunctionParameter::new::<String>("sql"));
        if with_mode {
            sig = sig.param(FunctionParameter::new::<String>("mode"));
        }
        ScalarFunctionBuilder::new(
            "bolt_register_sql",
            sig.return_type(i64::logical_type(extension)?)
                .documentation(
                    FunctionDocumentationBuilder::new().description(REGISTER_SQL_DESCRIPTION),
                ),
            BoltRegisterSqlScalar { with_mode },
        )
        .set_property(volatile)
        .register(extension)?;
    }

    #[cfg(not(target_arch = "wasm32"))]
    for (format, desc) in [
        (IngestFormat::Parquet, LOAD_PARQUET_DESCRIPTION),
        (IngestFormat::ArrowIpc, LOAD_ARROW_IPC_DESCRIPTION),
    ] {
        for with_mode in [false, true] {
            let mut sig = SignatureBuilder::new()
                .param(FunctionParameter::new::<String>("table_name"))
                .param(FunctionParameter::new::<String>("path"));
            if with_mode {
                sig = sig.param(FunctionParameter::new::<String>("mode"));
            }
            ScalarFunctionBuilder::new(
                format.function_name(),
                sig.return_type(i64::logical_type(extension)?)
                    .documentation(FunctionDocumentationBuilder::new().description(desc)),
                BoltLoadFileScalar { format, with_mode },
            )
            .set_property(volatile)
            .register(extension)?;
        }
    }

    ScalarFunctionBuilder::new(
        "bolt_drop_table",
        SignatureBuilder::new()
            .param(FunctionParameter::new::<String>("table_name"))
            .return_type(bool::logical_type(extension)?)
            .documentation(
                FunctionDocumentationBuilder::new().description(
                    "Drops a registered table from the Craton Bolt GPU catalog, returning true if it existed",
                ),
            ),
        BoltDropTableScalar,
    )
    .set_property(volatile)
    .register(extension)?;

    ScalarFunctionBuilder::new(
        "bolt_clear_tables",
        SignatureBuilder::new()
            .return_type(i64::logical_type(extension)?)
            .documentation(
                FunctionDocumentationBuilder::new().description(
                    "Clears all registered tables from Craton Bolt GPU memory (once per worker thread per query) and returns the count of tables removed",
                ),
            ),
        BoltClearTablesScalar,
    )
    .with_init()
    .set_property(volatile)
    .register(extension)?;

    for fn_name in ["bolt_prometheus_metrics", "bolt_metrics_prometheus"] {
        ScalarFunctionBuilder::new(
            fn_name,
            SignatureBuilder::new()
                .return_type(String::logical_type(extension)?)
                .documentation(FunctionDocumentationBuilder::new().description(
                    "Returns Craton Bolt execution metrics rendered in Prometheus exposition format",
                )),
            BoltPrometheusMetricsScalar,
        )
        .set_property(volatile)
        .register(extension)?;
    }

    Ok(())
}

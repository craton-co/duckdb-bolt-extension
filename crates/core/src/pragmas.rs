// SPDX-License-Identifier: Apache-2.0

//! DuckDB `PRAGMA` functions for managing and inspecting the Craton Bolt runtime.

use duckdb_neo::{
    Error, FromValue, Result,
    bind_arguments::BindArguments,
    connection::Context,
    extension::Extension,
    pragma_function::{PragmaExecCallbacks, PragmaFunctionBuilder, PragmaQueryCallbacks},
    signature::{FunctionDocumentationBuilder, FunctionParameter, SignatureBuilder},
};

use std::time::Duration;

use crate::runtime::BoltRuntime;

/// Reconcile this database's `bolt_*` settings with the process-wide runtime before a pragma
/// runs (W3-D-004): refreshes a setting another database made stale and re-checks the active
/// `bolt_ptx_cache_dir`, like every Bolt scalar function does.
fn sync_options(context: &Context) -> Result<()> {
    crate::options::sync_with_runtime(context)
}

/// Refuse a pragma that changes process-wide runtime configuration when the calling database
/// runs with `lock_configuration = true` (W6-B-002). DuckDB enforces the lock for `SET` of the
/// `bolt_*` options itself; these pragmas bypass `SET`, so they check it here. An unreadable
/// value counts as locked.
fn refuse_if_configuration_locked(context: &Context, pragma: &str) -> Result<()> {
    let config = context.config()?;
    let locked = match config.try_get_value("lock_configuration")? {
        None => false,
        Some((value, _scope)) => bool::from_value(&value).ok().flatten().unwrap_or(true),
    };
    if locked {
        return Err(Error::invalid_input(format!(
            "PRAGMA {pragma}: cannot change the Craton Bolt runtime: the configuration is locked (lock_configuration = true)"
        )));
    }
    Ok(())
}

struct BoltResetPragma;

impl PragmaExecCallbacks for BoltResetPragma {
    fn exec(&self, context: &Context, _arguments: BindArguments<'_>) -> Result<()> {
        refuse_if_configuration_locked(context, "bolt_reset")?;
        sync_options(context)?;
        // Errors whenever the engine cannot be built while a CUDA device is visible (including
        // a `bolt_device` ordinal that does not exist); succeeds without an engine only on
        // hosts with no CUDA device at all (always in `cuda-stub` builds), leaving the host
        // fallback active. See `BoltWorkerState::handle_reset_engine` for the policy (W1-B-001).
        BoltRuntime::global().reset_engine()
    }
}

struct BoltClearTablesPragma;

impl PragmaExecCallbacks for BoltClearTablesPragma {
    fn exec(&self, context: &Context, _arguments: BindArguments<'_>) -> Result<()> {
        sync_options(context)?;
        let _ = BoltRuntime::global().clear_tables()?;
        Ok(())
    }
}

struct BoltRequestTimeoutPragma;

impl PragmaExecCallbacks for BoltRequestTimeoutPragma {
    fn exec(&self, context: &Context, arguments: BindArguments<'_>) -> Result<()> {
        refuse_if_configuration_locked(context, "bolt_request_timeout")?;
        sync_options(context)?;
        let millis = arguments
            .value(0)?
            .ok_or_else(|| Error::invalid_input("bolt_request_timeout needs a constant argument"))?
            .get::<i64>()?
            .ok_or_else(|| Error::invalid_input("bolt_request_timeout cannot be NULL"))?;
        let millis = u64::try_from(millis).map_err(|_| {
            Error::invalid_input("bolt_request_timeout must be >= 0 milliseconds (0 = no limit)")
        })?;
        // Process-wide, like every bolt_* setting (W1-D-003).
        BoltRuntime::global()
            .set_request_timeout((millis > 0).then(|| Duration::from_millis(millis)));
        Ok(())
    }
}

struct BoltInfoPragma;

impl PragmaQueryCallbacks for BoltInfoPragma {
    fn query(&self, context: &Context, _arguments: BindArguments<'_>) -> Result<String> {
        sync_options(context)?;
        Ok("SELECT * FROM bolt_pool_stats()".to_string())
    }
}

/// Register Craton Bolt `PRAGMA` statements on `extension`.
pub fn register_pragmas(extension: &Extension) -> Result<()> {
    PragmaFunctionBuilder::exec(
        "bolt_reset",
        SignatureBuilder::new().documentation(
            FunctionDocumentationBuilder::new()
                .description("Rebuilds the Craton Bolt GPU engine using current bolt_* settings and retries a previously failed GPU initialization"),
        ),
        BoltResetPragma,
    )
    .register(extension)?;

    PragmaFunctionBuilder::exec(
        "bolt_clear_tables",
        SignatureBuilder::new().documentation(FunctionDocumentationBuilder::new().description(
            "Removes all registered tables from Craton Bolt GPU memory and releases pooled buffers",
        )),
        BoltClearTablesPragma,
    )
    .register(extension)?;

    PragmaFunctionBuilder::exec(
        "bolt_request_timeout",
        SignatureBuilder::new()
            .param(FunctionParameter::new::<i64>("milliseconds"))
            .documentation(FunctionDocumentationBuilder::new().description(
                "Limits how long a statement waits for one Craton Bolt request (0 = no limit). On timeout the statement fails and a host-fallback query is interrupted; a GPU query keeps running in the background. Process-wide.",
            )),
        BoltRequestTimeoutPragma,
    )
    .register(extension)?;

    PragmaFunctionBuilder::query(
        "bolt_info",
        SignatureBuilder::new().documentation(FunctionDocumentationBuilder::new().description(
            "Returns Craton Bolt engine, GPU device, memory pool, and JIT cache telemetry",
        )),
        BoltInfoPragma,
    )
    .register(extension)?;

    Ok(())
}

// SPDX-License-Identifier: Apache-2.0

//! Core implementation of `duckdb-bolt-extension`: DuckDB v2.0 C API integration
//! for the Craton Bolt JIT-compiled GPU SQL engine (`craton-bolt`).

pub mod arrow_bridge;
pub mod copy_function;
#[cfg(not(target_arch = "wasm32"))]
pub mod ingest;
pub mod options;
pub mod pragmas;
pub mod replacement_scan;
pub mod runtime;
pub mod sandbox;
pub mod scalar_functions;
pub mod table_functions;

use duckdb_neo::{Result, extension::Extension};

/// Register all Craton Bolt options, pragmas, scalar functions, table functions,
/// copy formats, and replacement scans on the given DuckDB [`Extension`].
pub fn register(extension: &Extension) -> Result<()> {
    options::register_options(extension)?;
    pragmas::register_pragmas(extension)?;
    scalar_functions::register_scalar_functions(extension)?;
    table_functions::register_table_functions(extension)?;
    copy_function::register_copy_functions(extension)?;
    replacement_scan::register_replacement_scan(extension)?;
    Ok(())
}

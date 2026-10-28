// SPDX-License-Identifier: Apache-2.0

//! WebAssembly loadable DuckDB v2.0 C API extension entrypoint (`bolt.duckdb_extension.wasm`).

use duckdb_loadable_macros::duckdb_entrypoint_c_api_v2;
use duckdb_neo::{Result, extension::Extension};

#[duckdb_entrypoint_c_api_v2(extension_name = "bolt", api_version = "2.0.0")]
pub fn entrypoint(extension: &Extension) -> Result<()> {
    duckdb_bolt_core::register(extension)
}

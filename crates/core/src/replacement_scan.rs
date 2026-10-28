// SPDX-License-Identifier: Apache-2.0

//! DuckDB replacement scan for Craton Bolt GPU tables (`bolt.<table_name>`, `'bolt:<table_name>'`,
//! or bare `<table_name>` when registered in Bolt).

use duckdb_neo::{
    Result, ToValue,
    connection::Context,
    extension::Extension,
    qualified_name::QualifiedName,
    replacement_scan::{
        ReplacementHandle, ReplacementScanBuilder, ReplacementScanCallbacks, ReplacementType,
    },
};

use crate::runtime::{BoltRuntime, normalize_table_reference};

struct BoltReplacementScan;

fn has_bolt_scheme(table: &str) -> bool {
    table
        .get(..5)
        .is_some_and(|head| head.eq_ignore_ascii_case("bolt:"))
}

fn is_ascii_ci(part: Option<&str>, expected: &str) -> bool {
    part.is_some_and(|p| p.eq_ignore_ascii_case(expected))
}

/// Whether a table reference names the Bolt namespace explicitly, in one of the documented
/// forms (SQL_REFERENCE 2.1 / 4; W6-B-001): `bolt.<t>`, `bolt.main.<t>`, or an unqualified
/// `'bolt:<t>'` / `'bolt://<t>'`. Any other qualifier (`bolt.other.<t>`, `mydb.bolt.<t>`) is
/// left to DuckDB, so its own catalog error is reported.
fn is_explicit_bolt_reference(catalog: Option<&str>, schema: Option<&str>, table: &str) -> bool {
    match (catalog, schema) {
        // `bolt.t`: DuckDB binds a two-part name as schema-qualified (and may also report it
        // with the catalog set instead).
        (None, Some(_)) => is_ascii_ci(schema, "bolt"),
        (Some(_), None) => is_ascii_ci(catalog, "bolt"),
        // `bolt.main.t`.
        (Some(_), Some(_)) => is_ascii_ci(catalog, "bolt") && is_ascii_ci(schema, "main"),
        (None, None) => has_bolt_scheme(table),
    }
}

impl ReplacementScanCallbacks for BoltReplacementScan {
    fn scan(
        &self,
        context: &Context,
        name: &QualifiedName,
        replacement: ReplacementHandle<'_>,
    ) -> Result<()> {
        let view = name.get_view()?;
        let explicit_bolt_namespace = is_explicit_bolt_reference(
            view.catalog.as_deref(),
            view.schema.as_deref(),
            &view.table,
        );

        if explicit_bolt_namespace {
            // Prefix-only normalization: `bolt.tmp_x` must resolve to table `tmp_x`, not `x`.
            let candidate = normalize_table_reference(&view.table);
            if candidate.is_empty() {
                return Ok(());
            }
            replacement.set_reference(ReplacementType::Table("bolt_scan".try_into()?))?;
            replacement.add_parameter(candidate.value(context)?)?;
            return Ok(());
        }

        // For unqualified table references not found in DuckDB's local catalog, check if
        // the table is registered in Craton Bolt (metadata snapshot; does not block on the
        // worker). The bare name must match a Bolt table exactly or ASCII case-insensitively;
        // nothing is stripped, so `tmp_x`, `"bolt.x"`, or a path like `'dir/x'` never capture
        // table `x` (W1-E-005). Pass the resolved catalog name so case-insensitive matches
        // bind exactly.
        if view.catalog.is_none()
            && view.schema.is_none()
            && let Some(summary) = BoltRuntime::global().table_info_exact(&view.table)?
        {
            replacement.set_reference(ReplacementType::Table("bolt_scan".try_into()?))?;
            replacement.add_parameter(summary.name.value(context)?)?;
        }

        Ok(())
    }
}

/// Register the Craton Bolt replacement scan on `extension`.
pub fn register_replacement_scan(extension: &Extension) -> Result<()> {
    ReplacementScanBuilder::new(BoltReplacementScan).register(extension)
}

#[cfg(test)]
mod tests {
    use super::is_explicit_bolt_reference as explicit;

    #[test]
    fn only_documented_qualified_forms_are_explicit_bolt_references() {
        // W6-B-001: `bolt.t`, `bolt.main.t`, `'bolt:t'`, `'bolt://t'` (ASCII case-insensitive).
        assert!(explicit(None, Some("bolt"), "t"));
        assert!(explicit(None, Some("BOLT"), "t"));
        assert!(explicit(Some("bolt"), None, "t"));
        assert!(explicit(Some("bolt"), Some("main"), "t"));
        assert!(explicit(Some("Bolt"), Some("MAIN"), "t"));
        assert!(explicit(None, None, "bolt:t"));
        assert!(explicit(None, None, "BOLT://t"));
        // Everything else is left to DuckDB.
        assert!(!explicit(Some("bolt"), Some("other"), "t"));
        assert!(!explicit(Some("memory"), Some("bolt"), "t"));
        assert!(!explicit(Some("mydb"), Some("main"), "t"));
        assert!(!explicit(None, Some("main"), "t"));
        assert!(!explicit(None, Some("other"), "bolt:t"));
        assert!(!explicit(None, None, "t"));
        assert!(!explicit(None, None, "boltx"));
    }
}

// SPDX-License-Identifier: Apache-2.0

//! Runs every `test/sql/*.test` SQLLogicTest file against the statically linked extension, and
//! repeats the run against a packaged `build/*/bolt.duckdb_extension` when one exists.
//!
//! This is a small subset of DuckDB's `sqllogictest` runner, meant as a smoke test that needs no
//! DuckDB source checkout. Use `make test` (DuckDB's real `unittest` runner) for full fidelity.
//!
//! Supported records:
//! - `statement ok` / `statement error` / `statement maybe`; for `error` and `maybe`, the lines
//!   after `----` are an expected error-message substring (empty = any error).
//! - `query <types> [nosort|rowsort|valuesort] [label]` with tab-separated rows, or one value per
//!   line (DuckDB's alternative layout). Cells are formatted by casting every result column to
//!   `VARCHAR` in SQL (so they match DuckDB's own text rendering); `NULL` prints as `NULL` and the
//!   empty string as `(empty)`.
//! - `require <ext>`: `bolt` is always available; `notwindows` / `windows` are honored; anything
//!   else skips the whole file. `require-env VAR` skips the file when `VAR` is unset.
//! - `mode skip` / `mode unskip` (other modes are ignored), `halt`, and `#` comments.
//!
//! Anything else (`loop`, `foreach`, `load`, `restart`, `<REGEX>:` matching, ...) fails loudly.

use std::{
    panic::{self, AssertUnwindSafe},
    path::{Path, PathBuf},
    sync::{Mutex, Once, OnceLock},
};

use duckdb_loadable_macros::duckdb_entrypoint_c_api_v2;
use duckdb_neo::{
    Parameters, Result,
    connection::Connection,
    data_chunk::DataChunk,
    environment::{Environment, StorageLocation},
    extension::{Extension, register_static_extension},
    instance::{AttachBuilder, InstanceBuilder},
    logical_type::LogicalTypeID,
    vector::Unknown,
};

#[duckdb_entrypoint_c_api_v2(extension_name = "bolt", api_version = "2.0.0")]
fn bolt_slt_entrypoint(extension: &Extension) -> Result<()> {
    duckdb_bolt_core::register(extension)
}

static REGISTER_ONCE: Once = Once::new();
static TEST_MUTEX: Mutex<()> = Mutex::new(());
/// Outcome of the dynamic-load check, computed once and *before* the static extension is
/// registered: after `register_static_extension`, every new instance auto-loads the static copy
/// and `LOAD 'bolt.duckdb_extension'` would silently resolve to it instead of the packaged binary.
static DYNAMIC_LOAD_OUTCOME: OnceLock<std::result::Result<String, String>> = OnceLock::new();

fn lock_tests() -> std::sync::MutexGuard<'static, ()> {
    TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner())
}

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates dir")
        .parent()
        .expect("workspace root")
        .to_path_buf()
}

/// All `test/sql/*.test` files, sorted by name.
fn sql_test_files() -> Vec<PathBuf> {
    let dir = workspace_root().join("test").join("sql");
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()))
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|p| p.is_file() && p.extension().is_some_and(|ext| ext == "test"))
        .collect();
    files.sort();
    files
}

fn display_path(path: &Path) -> String {
    path.strip_prefix(workspace_root())
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

// ---------------------------------------------------------------------------
// Result formatting
// ---------------------------------------------------------------------------

fn render_text(value: Option<&str>) -> String {
    match value {
        None => "NULL".to_string(),
        Some("") => "(empty)".to_string(),
        Some(s) => s.to_string(),
    }
}

/// Format one cell natively (used only when the statement cannot be wrapped in a `VARCHAR` cast,
/// for example `PRAGMA` or `EXPLAIN`).
fn native_cell(chunk: &DataChunk<'_>, col: usize, row: usize) -> Result<String> {
    macro_rules! read_as {
        ($ty:ty) => {
            chunk
                .get_vector_at::<$ty>(col)?
                .reader()?
                .get(row)?
                .map(|v| v.to_string())
                .unwrap_or_else(|| "NULL".to_string())
        };
    }
    let type_id = chunk.get_vector_at::<Unknown>(col)?.logical_type().id()?;
    Ok(match type_id {
        LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_BOOLEAN => read_as!(bool),
        LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_TINYINT => read_as!(i8),
        LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_SMALLINT => read_as!(i16),
        LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_INTEGER => read_as!(i32),
        LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_BIGINT => read_as!(i64),
        LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_UTINYINT => read_as!(u8),
        LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_USMALLINT => read_as!(u16),
        LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_UINTEGER => read_as!(u32),
        LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_UBIGINT => read_as!(u64),
        LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_DOUBLE => chunk
            .get_vector_at::<f64>(col)?
            .reader()?
            .get(row)?
            .map(|v| {
                if v.is_finite() && v.fract() == 0.0 {
                    format!("{v:.1}")
                } else {
                    format!("{v}")
                }
            })
            .unwrap_or_else(|| "NULL".to_string()),
        LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_VARCHAR => {
            render_text(chunk.get_vector_at::<String>(col)?.reader()?.get(row)?)
        }
        // Everything else (HUGEINT, DECIMAL, DATE, TIMESTAMP*, nested types, ...) goes through
        // DuckDB's value-to-string conversion. This path is slow but only used for statements
        // that cannot be wrapped (see `varchar_wrapped`).
        _ => {
            let value = chunk.get_vector_at::<Unknown>(col)?.get_value_slow(row)?;
            if value.is_null()? {
                "NULL".to_string()
            } else {
                value.dbg_string()?
            }
        }
    })
}

/// Wrap a row-producing statement so every column is rendered by DuckDB's own `VARCHAR` cast.
/// Returns `None` for statements that cannot appear in a `FROM (...)` subquery.
fn varchar_wrapped(sql: &str) -> Option<String> {
    let trimmed = sql.trim().trim_end_matches(';').trim_end();
    let first_word = trimmed
        .split(|c: char| c.is_whitespace() || c == '(')
        .next()
        .unwrap_or("")
        .to_ascii_lowercase();
    let wrappable = trimmed.starts_with('(')
        || matches!(
            first_word.as_str(),
            "select" | "with" | "from" | "values" | "table" | "describe" | "show" | "summarize"
        );
    wrappable.then(|| format!("SELECT COLUMNS(*)::VARCHAR FROM (\n{trimmed}\n) AS __slt_result"))
}

/// Execute `sql` and return its rows as vectors of formatted cells.
fn query_rows(conn: &Connection, sql: &str) -> Result<Vec<Vec<String>>> {
    // If the wrapper does not bind (a top-level-only construct) or fails while running, fall
    // back to the statement as written so any reported error is the statement's own.
    if let Some(wrapped) = varchar_wrapped(sql)
        && let Ok(mut result) = conn.query(wrapped.as_str(), Parameters::None)
    {
        let mut rows = Vec::new();
        let mut complete = false;
        while let Ok(next) = result.next_chunk() {
            let Some(chunk) = next else {
                complete = true;
                break;
            };
            for row in 0..chunk.row_count()? {
                let mut cells = Vec::with_capacity(chunk.col_count());
                for col in 0..chunk.col_count() {
                    cells.push(render_text(
                        chunk.get_vector_at::<String>(col)?.reader()?.get(row)?,
                    ));
                }
                rows.push(cells);
            }
        }
        drop(result);
        if complete {
            return Ok(rows);
        }
    }

    let mut result = conn.query(sql, Parameters::None)?;
    let mut rows = Vec::new();
    while let Some(chunk) = result.next_chunk()? {
        for row in 0..chunk.row_count()? {
            let mut cells = Vec::with_capacity(chunk.col_count());
            for col in 0..chunk.col_count() {
                cells.push(native_cell(&chunk, col, row)?);
            }
            rows.push(cells);
        }
    }
    Ok(rows)
}

// ---------------------------------------------------------------------------
// File runner
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(clippy::enum_variant_names)] // names mirror the sqllogictest keywords
enum SortMode {
    NoSort,
    RowSort,
    ValueSort,
}

enum FileOutcome {
    Ran { records: usize },
    Skipped(String),
}

/// Read lines starting at `*i` until a blank line (or `----` when `stop_at_separator`).
fn read_block(lines: &[&str], i: &mut usize, stop_at_separator: bool) -> Vec<String> {
    let mut block = Vec::new();
    while *i < lines.len() {
        let trimmed = lines[*i].trim();
        if trimmed.is_empty() || (stop_at_separator && trimmed == "----") {
            break;
        }
        block.push(lines[*i].trim_end().to_string());
        *i += 1;
    }
    block
}

/// Skip a `----` separator at `*i`, returning whether one was present.
fn skip_separator(lines: &[&str], i: &mut usize) -> bool {
    if *i < lines.len() && lines[*i].trim() == "----" {
        *i += 1;
        true
    } else {
        false
    }
}

/// Decide whether a `require` line lets the file run (`None`) or skips it (`Some(reason)`).
fn require_skip_reason(args: &str) -> Option<String> {
    match args.trim() {
        "bolt" => None,
        "notwindows" if cfg!(windows) => Some("require notwindows".to_string()),
        "notwindows" => None,
        "windows" if !cfg!(windows) => Some("require windows".to_string()),
        "windows" => None,
        other => Some(format!(
            "require {other}: not available in the Rust SQLLogicTest runner"
        )),
    }
}

fn check_error_message(location: &str, sql: &str, expected: &[String], err: &duckdb_neo::Error) {
    let expected = expected.join("\n");
    let expected = expected.trim();
    if expected.is_empty() {
        return;
    }
    assert!(
        !expected.starts_with("<REGEX>:") && !expected.starts_with("<!REGEX>:"),
        "{location}: <REGEX> error matching is not supported by the Rust runner; use `make test`"
    );
    let actual = err.to_string();
    assert!(
        actual.contains(expected),
        "{location}: statement failed with an unexpected error.\nSQL:\n{sql}\nExpected error containing:\n{expected}\nActual error:\n{actual}"
    );
}

fn run_sqllogictest_file(conn: &Connection, path: &Path) -> FileOutcome {
    let name = display_path(path);
    let content =
        std::fs::read_to_string(path).unwrap_or_else(|e| panic!("failed to read {name}: {e}"));
    let lines: Vec<&str> = content.lines().collect();
    let mut i = 0;
    let mut skipping = false;
    let mut records = 0usize;

    while i < lines.len() {
        let line = lines[i].trim();
        let location = format!("{name}:{}", i + 1);
        if line.is_empty() || line.starts_with('#') {
            i += 1;
            continue;
        }
        let mut words = line.split_whitespace();
        let directive = words.next().unwrap_or("");
        let rest: Vec<&str> = words.collect();

        match directive {
            "require" => {
                if let Some(reason) = require_skip_reason(&rest.join(" ")) {
                    return FileOutcome::Skipped(reason);
                }
                i += 1;
            }
            "require-env" => {
                let var = rest.first().copied().unwrap_or("");
                if std::env::var_os(var).is_none() {
                    return FileOutcome::Skipped(format!("require-env {var}: not set"));
                }
                i += 1;
            }
            "mode" => {
                match rest.first().copied() {
                    Some("skip") => skipping = true,
                    Some("unskip") => skipping = false,
                    _ => {}
                }
                i += 1;
            }
            "halt" => {
                if !skipping {
                    break;
                }
                i += 1;
            }
            "statement" => {
                let kind = rest.first().copied().unwrap_or("");
                i += 1;
                let sql = read_block(&lines, &mut i, true).join("\n");
                let expected_error = if skip_separator(&lines, &mut i) {
                    read_block(&lines, &mut i, false)
                } else {
                    Vec::new()
                };
                if skipping {
                    continue;
                }
                assert!(!sql.trim().is_empty(), "{location}: empty statement");
                let outcome = conn.execute(sql.as_str(), Parameters::None);
                match (kind, outcome) {
                    ("ok", Ok(_)) => {}
                    ("ok", Err(e)) => {
                        panic!("{location}: statement ok failed.\nSQL:\n{sql}\nError: {e}")
                    }
                    ("error", Ok(_)) => {
                        panic!("{location}: statement error unexpectedly succeeded.\nSQL:\n{sql}")
                    }
                    ("error" | "maybe", Err(e)) => {
                        check_error_message(&location, &sql, &expected_error, &e)
                    }
                    ("maybe", Ok(_)) => {}
                    (other, _) => panic!("{location}: unsupported statement kind '{other}'"),
                }
                records += 1;
            }
            "query" => {
                let types = rest.first().copied().unwrap_or_else(|| {
                    panic!("{location}: query directive is missing its column types")
                });
                let sort_mode = match rest.get(1).copied() {
                    Some("rowsort") => SortMode::RowSort,
                    Some("valuesort") => SortMode::ValueSort,
                    Some("nosort") | None => SortMode::NoSort,
                    // Anything else is a label; labels are accepted but not cross-checked.
                    Some(_) => SortMode::NoSort,
                };
                i += 1;
                let sql = read_block(&lines, &mut i, true).join("\n");
                assert!(
                    skip_separator(&lines, &mut i),
                    "{location}: expected `----` after the query"
                );
                let expected_lines = read_block(&lines, &mut i, false);
                if skipping {
                    continue;
                }

                let rows = query_rows(conn, &sql).unwrap_or_else(|e| {
                    panic!("{location}: query failed.\nSQL:\n{sql}\nError: {e}")
                });
                let col_count = types.chars().count();
                if let Some(bad) = rows.iter().find(|r| r.len() != col_count) {
                    panic!(
                        "{location}: query declares {col_count} column(s) ('{types}') but returned {}.\nSQL:\n{sql}",
                        bad.len()
                    );
                }

                // DuckDB accepts either one row per line (tab-separated) or one value per line.
                let one_value_per_line = col_count > 1
                    && expected_lines.len() == rows.len() * col_count
                    && expected_lines.iter().all(|l| !l.contains('\t'));
                let (mut actual, mut expected): (Vec<String>, Vec<String>) = if sort_mode
                    == SortMode::ValueSort
                    || one_value_per_line
                {
                    (
                        rows.into_iter().flatten().collect(),
                        expected_lines
                            .iter()
                            .flat_map(|l| l.split('\t'))
                            .map(|v| v.trim().to_string())
                            .collect(),
                    )
                } else {
                    (
                        rows.iter().map(|r| r.join("\t")).collect(),
                        expected_lines
                            .iter()
                            .map(|l| l.split('\t').map(str::trim).collect::<Vec<_>>().join("\t"))
                            .collect(),
                    )
                };
                if sort_mode != SortMode::NoSort {
                    actual.sort();
                    expected.sort();
                }
                assert_eq!(
                    actual, expected,
                    "{location}: result mismatch for query:\n{sql}"
                );
                records += 1;
            }
            "loop" | "foreach" | "endloop" | "concurrentloop" | "load" | "restart"
            | "reconnect" | "unzip" | "set" | "reset" | "sleep" => {
                panic!(
                    "{location}: directive '{directive}' is not supported by the Rust SQLLogicTest runner; use `make test`"
                );
            }
            other => panic!("{location}: unrecognized SQLLogicTest directive '{other}'"),
        }
    }

    FileOutcome::Ran { records }
}

/// Run every test file on fresh in-memory databases produced by `open`.
/// Returns `(files_run, files_skipped)`.
fn run_all_files(open: &dyn Fn() -> Result<(Environment, Connection)>) -> Result<(usize, usize)> {
    let files = sql_test_files();
    assert!(!files.is_empty(), "no test/sql/*.test files found");
    let (mut ran, mut skipped) = (0, 0);
    for file in &files {
        let (_env, conn) = open()?;
        // The Bolt catalog is process-wide; start every file from an empty catalog, as DuckDB's
        // `unittest` runner would see it when the file is run on its own.
        conn.execute("PRAGMA bolt_clear_tables", Parameters::None)?;
        match run_sqllogictest_file(&conn, file) {
            FileOutcome::Ran { records } => {
                println!("[slt] {}: {records} record(s) passed", display_path(file));
                ran += 1;
            }
            FileOutcome::Skipped(reason) => {
                println!("[slt] {}: skipped ({reason})", display_path(file));
                skipped += 1;
            }
        }
        conn.execute("PRAGMA bolt_clear_tables", Parameters::None)?;
    }
    Ok((ran, skipped))
}

// ---------------------------------------------------------------------------
// Dynamic load of the packaged extension
// ---------------------------------------------------------------------------

/// `build/<dir>/bolt.duckdb_extension` and `build/<dir>/extension/bolt/bolt.duckdb_extension`.
fn packaged_extensions() -> Vec<PathBuf> {
    let build = workspace_root().join("build");
    let Ok(entries) = std::fs::read_dir(&build) else {
        return Vec::new();
    };
    let mut found: Vec<PathBuf> = entries
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_dir())
        .flat_map(|dir| {
            [
                dir.join("bolt.duckdb_extension"),
                dir.join("extension")
                    .join("bolt")
                    .join("bolt.duckdb_extension"),
            ]
        })
        .filter(|p| p.is_file())
        .collect();
    found.sort();
    // `extension/bolt/` holds a copy of the top-level file; keep one per build directory.
    found.dedup_by(|nested, top| {
        let top_dir = top.parent().map(Path::to_path_buf);
        let nested_dir = nested
            .parent()
            .and_then(Path::parent)
            .and_then(Path::parent)
            .map(Path::to_path_buf);
        nested_dir.is_some() && nested_dir == top_dir
    });
    found
}

fn run_dynamic_load_check() -> Result<String> {
    let candidates = packaged_extensions();
    if candidates.is_empty() {
        return Ok(format!(
            "skipped: no build/*/bolt.duckdb_extension under {} (run `make release` or \
             `cargo build --release -p duckdb-bolt-extension && python scripts/package_extension.py --profile release`)",
            workspace_root().join("build").display()
        ));
    }

    let mut summary = Vec::new();
    for ext_path in &candidates {
        let load_sql = format!(
            "LOAD '{}'",
            ext_path
                .to_string_lossy()
                .replace('\\', "/")
                .replace('\'', "''")
        );
        let open = || -> Result<(Environment, Connection)> {
            let env = Environment::new()?;
            let instance = InstanceBuilder::new()
                .option("allow_unsigned_extensions", "true")
                .build(&env)?;
            instance.attach(AttachBuilder::new(StorageLocation::InMemory).make_default())?;
            let conn = instance.connect()?;
            conn.execute(load_sql.as_str(), Parameters::None)?;
            Ok((env, conn))
        };
        let (ran, skipped) = run_all_files(&open)?;
        summary.push(format!(
            "{}: {ran} file(s) passed, {skipped} skipped",
            display_path(ext_path)
        ));
    }
    Ok(format!("loaded {}", summary.join("; ")))
}

fn dynamic_load_outcome() -> &'static std::result::Result<String, String> {
    DYNAMIC_LOAD_OUTCOME.get_or_init(|| {
        match panic::catch_unwind(AssertUnwindSafe(run_dynamic_load_check)) {
            Ok(Ok(summary)) => Ok(summary),
            Ok(Err(e)) => Err(format!("dynamic load check failed: {e}")),
            Err(payload) => Err(payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "dynamic load check panicked".to_string())),
        }
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn test_bolt_sqllogictest_suite() -> Result<()> {
    let _guard = lock_tests();
    // Must run before the static extension is registered (see DYNAMIC_LOAD_OUTCOME); its own
    // test reports the result.
    let _ = dynamic_load_outcome();
    REGISTER_ONCE.call_once(|| {
        // SAFETY: `duckdb_extension_bolt_describe` is emitted by `#[duckdb_entrypoint_c_api_v2]`, which
        // fills the descriptor as `duckdb_static_extension.h` specifies (the contract of
        // `register_static_extension`); `Once` makes this a single call.
        unsafe { register_static_extension(duckdb_extension_bolt_describe) }
            .expect("Failed to register static bolt extension");
    });

    let open = || -> Result<(Environment, Connection)> {
        let env = Environment::new()?;
        let instance = env.open(StorageLocation::InMemory)?;
        let conn = instance.connect()?;
        Ok((env, conn))
    };
    let (ran, skipped) = run_all_files(&open)?;
    assert!(
        ran > 0,
        "every SQLLogicTest file was skipped ({skipped} skipped)"
    );
    Ok(())
}

#[test]
fn test_dynamic_load_packaged_extension_if_present() {
    let _guard = lock_tests();
    match dynamic_load_outcome() {
        Ok(summary) => println!("test_dynamic_load_packaged_extension_if_present: {summary}"),
        Err(e) => panic!("{e}"),
    }
}

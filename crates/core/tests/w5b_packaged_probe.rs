// SPDX-License-Identifier: Apache-2.0

//! Wave 5 (W5-B-001) probe of the *packaged* loadable extension: `LOAD`s every
//! `build/*/bolt.duckdb_extension` (see `scripts/package_extension.py`) into a DuckDB instance of
//! this test process and exercises paths the SQLLogicTest files do not cover: loading twice,
//! several instances sharing the process-wide catalog, COPY round trips, Parquet ingest, the
//! host fallback, re-entrant Bolt calls from inside a fallback query (W3-E-004, whose static-build
//! mechanism differs from the dynamic one), metrics and option round trips.
//!
//! This binary never registers the extension statically, so `LOAD` cannot resolve to a copy
//! linked into the test executable. Without a packaged binary the test passes with a notice.

use std::{
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use duckdb_neo::{
    Parameters, Result,
    connection::Connection,
    environment::{Environment, StorageLocation},
    instance::{AttachBuilder, InstanceBuilder},
};

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root")
        .to_path_buf()
}

/// `build/<dir>/bolt.duckdb_extension`, one per build directory.
fn packaged_extensions() -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(workspace_root().join("build")) else {
        return Vec::new();
    };
    let mut found: Vec<PathBuf> = entries
        .filter_map(|e| e.ok().map(|e| e.path()))
        .map(|dir| dir.join("bolt.duckdb_extension"))
        .filter(|p| p.is_file())
        .collect();
    found.sort();
    found
}

fn sql_path(path: &Path) -> String {
    path.to_string_lossy()
        .replace('\\', "/")
        .replace('\'', "''")
}

fn open_loaded(ext: &Path) -> Result<(Environment, Connection)> {
    let env = Environment::new()?;
    let instance = InstanceBuilder::new()
        .option("allow_unsigned_extensions", "true")
        .build(&env)?;
    instance.attach(AttachBuilder::new(StorageLocation::InMemory).make_default())?;
    let conn = instance.connect()?;
    conn.execute(
        format!("LOAD '{}'", sql_path(ext)).as_str(),
        Parameters::None,
    )?;
    Ok((env, conn))
}

fn exec(conn: &Connection, sql: &str) -> Result<()> {
    conn.execute(sql, Parameters::None).map(|_| ())
}

/// First column of the first row, rendered by DuckDB's `VARCHAR` cast (`None` for NULL).
fn scalar(conn: &Connection, sql: &str) -> Result<Option<String>> {
    let wrapped = format!("SELECT (COLUMNS(*))::VARCHAR FROM ({sql}) AS q LIMIT 1");
    let mut result = conn.query(wrapped.as_str(), Parameters::None)?;
    let chunk = result.next_chunk()?.expect("expected one row");
    let vector = chunk.get_vector_at::<String>(0)?;
    Ok(vector.reader()?.get(0)?.map(|s| s.to_string()))
}

fn scalar_str(conn: &Connection, sql: &str) -> Result<String> {
    Ok(scalar(conn, sql)?.unwrap_or_else(|| panic!("NULL from {sql}")))
}

fn probe(ext: &Path) -> Result<String> {
    let (_env, conn) = open_loaded(ext)?;
    // Loading again is a no-op, not an error or a second registration.
    exec(&conn, format!("LOAD '{}'", sql_path(ext)).as_str())?;
    exec(&conn, "PRAGMA bolt_clear_tables")?;

    let version = scalar_str(&conn, "SELECT bolt_version()")?;
    assert!(version.contains("craton-bolt"), "{version}");
    let loaded = scalar_str(
        &conn,
        "SELECT loaded FROM duckdb_extensions() WHERE extension_name = 'bolt'",
    )?;
    assert_eq!(loaded, "true");
    let backend = scalar_str(&conn, "SELECT backend FROM bolt_pool_stats()")?;

    // COPY round trip and every way of naming a Bolt table.
    exec(
        &conn,
        "COPY (SELECT range::INTEGER AS id, (range % 7)::BIGINT AS k, 'n' || range AS s \
         FROM range(5000)) TO 'w5b_t' (FORMAT bolt, USE_TMP_FILE FALSE)",
    )?;
    for sql in [
        "SELECT count(*) FROM bolt_scan('w5b_t')",
        "SELECT count(*) FROM bolt.w5b_t",
        "SELECT count(*) FROM 'bolt:w5b_t'",
        "SELECT count(*) FROM w5b_t",
        "SELECT count(*) FROM bolt_query('SELECT id FROM w5b_t')",
    ] {
        assert_eq!(scalar_str(&conn, sql)?, "5000", "{sql}");
    }
    assert_eq!(
        scalar_str(
            &conn,
            "SELECT sum(c) FROM bolt_query('SELECT k, COUNT(*) AS c FROM w5b_t GROUP BY k')"
        )?,
        "5000"
    );
    assert_eq!(
        scalar_str(&conn, "SELECT count(*) FROM bolt_scan('w5b_t') WHERE k = 3")?,
        "714"
    );
    exec(
        &conn,
        "CREATE TABLE w5b_back (id INTEGER, k BIGINT, s VARCHAR)",
    )?;
    exec(&conn, "COPY w5b_back FROM 'w5b_t' (FORMAT bolt)")?;
    assert_eq!(
        scalar_str(&conn, "SELECT sum(id) FROM w5b_back")?,
        "12497500"
    );

    // A query the GPU planner may hand to the host fallback (window function); either way the
    // result must be right.
    assert_eq!(
        scalar_str(
            &conn,
            "SELECT max(rn) FROM bolt_query('SELECT id, row_number() OVER (ORDER BY id) AS rn \
             FROM w5b_t')"
        )?,
        "5000"
    );

    // Parquet ingest through the packaged binary's own Parquet reader.
    let dir = std::env::temp_dir().join(format!("w5b_probe_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let parquet = dir.join("t.parquet");
    exec(
        &conn,
        format!(
            "COPY (SELECT range::BIGINT AS v FROM range(100)) TO '{}' (FORMAT parquet)",
            sql_path(&parquet)
        )
        .as_str(),
    )?;
    exec(
        &conn,
        format!(
            "SELECT bolt_load_parquet('w5b_pq', '{}')",
            sql_path(&parquet)
        )
        .as_str(),
    )?;
    assert_eq!(
        scalar_str(&conn, "SELECT sum(v) FROM bolt_scan('w5b_pq')")?,
        "4950"
    );
    let _ = std::fs::remove_dir_all(&dir);

    // Re-entrant Bolt calls from inside a host-fallback query must fail fast (or succeed), never
    // wait for the worker they run on (W3-E-004).
    exec(&conn, "PRAGMA bolt_request_timeout(60000)")?;
    let started = Instant::now();
    for sql in [
        "SELECT * FROM bolt_query('SELECT bolt_version()')",
        "SELECT * FROM bolt_query('SELECT * FROM bolt.w5b_t')",
        "SELECT * FROM bolt_query('SELECT * FROM bolt_scan(''w5b_t'')')",
    ] {
        if let Err(e) = conn.execute(sql, Parameters::None) {
            assert!(!e.message.contains("timed out"), "{sql}: {e}");
        }
    }
    exec(&conn, "PRAGMA bolt_request_timeout(0)")?;
    assert!(started.elapsed() < Duration::from_secs(30));
    assert!(scalar_str(&conn, "SELECT bolt_version()")?.contains("craton-bolt"));

    // A second instance in the same process sees the process-wide catalog of this binary.
    {
        let (_env2, conn2) = open_loaded(ext)?;
        assert_eq!(
            scalar_str(&conn2, "SELECT count(*) FROM bolt_scan('w5b_t')")?,
            "5000"
        );
    }

    // Metrics, pragmas and options.
    exec(&conn, "PRAGMA bolt_info")?;
    assert!(
        scalar_str(&conn, "SELECT count(*) FROM bolt_metrics()")?
            .parse::<u64>()
            .is_ok()
    );
    assert!(
        scalar_str(&conn, "SELECT bolt_prometheus_metrics()")?.contains("craton_queries_total")
    );
    exec(&conn, "SET bolt_memory_budget = 1073741824")?;
    exec(&conn, "RESET bolt_memory_budget")?;
    exec(&conn, "SELECT bolt_drop_table('w5b_t')")?;
    assert!(
        conn.execute("SELECT * FROM bolt_scan('w5b_t')", Parameters::None)
            .is_err()
    );
    exec(&conn, "PRAGMA bolt_clear_tables")?;

    Ok(format!("{} ({backend})", ext.display()))
}

#[test]
fn packaged_extension_probe() -> Result<()> {
    let candidates = packaged_extensions();
    if candidates.is_empty() {
        println!("skipped: no build/*/bolt.duckdb_extension");
        return Ok(());
    }
    for ext in &candidates {
        println!("probed {}", probe(ext)?);
    }
    Ok(())
}

// SPDX-License-Identifier: Apache-2.0

//! Regression tests for W7-A:
//!
//! * the replacement scan takes only the documented qualified forms (`bolt.t`, `bolt.main.t`,
//!   `'bolt:t'`, `'bolt://t'`, and a bare registered name); any other qualifier is left to
//!   DuckDB, which reports its own catalog error (W6-B-001);
//! * `PRAGMA bolt_request_timeout` and `PRAGMA bolt_reset` are refused when the calling
//!   database runs with `lock_configuration = true` (W6-B-002);
//! * `record_batches_to_collection` rejects a batch whose column count differs from the
//!   schema instead of leaving DuckDB vectors uninitialized (W6-B change, now tested).
//!
//! Valid for both GPU and `cuda-stub` builds: nothing here depends on the backend.

use std::{
    sync::{Arc, Mutex, Once},
    time::Duration,
};

use arrow_array::{Int32Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use duckdb_bolt_core::{arrow_bridge::record_batches_to_collection, runtime::BoltRuntime};
use duckdb_loadable_macros::duckdb_entrypoint_c_api_v2;
use duckdb_neo::{
    Parameters, Result,
    connection::Connection,
    environment::{Environment, StorageLocation},
    extension::{Extension, register_static_extension},
};

#[duckdb_entrypoint_c_api_v2(extension_name = "bolt", api_version = "2.0.0")]
fn bolt_test_entrypoint(extension: &Extension) -> Result<()> {
    duckdb_bolt_core::register(extension)
}

static REGISTER_ONCE: Once = Once::new();
// Serialize tests because Craton Bolt's runtime catalog and config are process-wide.
static TEST_MUTEX: Mutex<()> = Mutex::new(());

fn register_extension() {
    REGISTER_ONCE.call_once(|| {
        // SAFETY: `duckdb_extension_bolt_describe` is emitted by `#[duckdb_entrypoint_c_api_v2]`,
        // which fills the descriptor as `duckdb_static_extension.h` specifies (the contract of
        // `register_static_extension`); `Once` makes this a single call.
        unsafe { register_static_extension(duckdb_extension_bolt_describe) }
            .expect("Failed to register static bolt extension");
    });
}

/// A fresh in-memory database (its own `Environment`), with the extension loaded.
fn setup_connection() -> Result<(Environment, Connection)> {
    register_extension();
    let env = Environment::new()?;
    let instance = env.open(StorageLocation::InMemory)?;
    let conn = instance.connect()?;
    conn.execute("PRAGMA bolt_clear_tables", Parameters::None)?;
    Ok((env, conn))
}

/// Run `sql` to completion and return its error message, or `None` if it succeeded.
fn query_error(conn: &Connection, sql: &str) -> Option<String> {
    match conn.query(sql, Parameters::None) {
        Err(e) => Some(e.message),
        Ok(mut r) => loop {
            match r.next_chunk() {
                Err(e) => break Some(e.message),
                Ok(None) => break None,
                Ok(Some(_)) => {}
            }
        },
    }
}

fn single_i64(conn: &Connection, sql: &str) -> Result<i64> {
    let mut result = conn.query(sql, Parameters::None)?;
    let chunk = result.next_chunk()?.expect("one result row");
    let vec = chunk.get_vector_at::<i64>(0)?;
    let value = vec.reader()?.get(0)?.copied();
    Ok(value.unwrap_or_else(|| panic!("`{sql}` returned NULL")))
}

#[test]
fn test_only_documented_qualifiers_reach_the_bolt_catalog() -> Result<()> {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let (_env, conn) = setup_connection()?;
    conn.execute(
        "COPY (SELECT range::INTEGER AS v FROM range(3)) TO 'w7a_q' (FORMAT bolt, USE_TMP_FILE FALSE)",
        Parameters::None,
    )?;

    // The documented forms scan the Bolt table.
    for from in [
        "w7a_q",
        "bolt.w7a_q",
        "BOLT.w7a_q",
        "bolt.main.w7a_q",
        "Bolt.MAIN.w7a_q",
        "'bolt:w7a_q'",
        "'bolt://w7a_q'",
    ] {
        assert_eq!(
            single_i64(&conn, &format!("SELECT count(*) FROM {from}"))?,
            3,
            "{from}"
        );
    }

    // Any other qualifier is DuckDB's: its catalog error, not a Bolt lookup (W6-B-001).
    for from in [
        "bolt.other.w7a_q",
        "memory.bolt.w7a_q",
        "mydb.main.w7a_q",
        "memory.bolt.missing_w7a",
    ] {
        let msg = query_error(&conn, &format!("SELECT count(*) FROM {from}"))
            .unwrap_or_else(|| panic!("`{from}` should not resolve"));
        assert!(
            msg.contains("Catalog Error"),
            "`{from}` should get DuckDB's catalog error, got: {msg}"
        );
        assert!(
            !msg.contains("Craton Bolt"),
            "`{from}` must not be answered by the Bolt catalog, got: {msg}"
        );
    }

    // A missing table under a documented form is still the Bolt catalog's error.
    let msg = query_error(&conn, "SELECT * FROM bolt.missing_w7a").expect("missing table");
    assert!(msg.contains("not registered in Craton Bolt"), "{msg}");

    BoltRuntime::global().drop_table("w7a_q")?;
    Ok(())
}

#[test]
fn test_runtime_pragmas_honor_lock_configuration() -> Result<()> {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let runtime = BoltRuntime::global();
    let before = runtime.request_timeout();

    // Unlocked: the pragma works (and is restored below).
    let (_env, conn) = setup_connection()?;
    conn.execute("PRAGMA bolt_request_timeout(123456)", Parameters::None)?;
    assert_eq!(
        runtime.request_timeout(),
        Some(Duration::from_millis(123_456))
    );
    let configured = runtime.request_timeout();

    conn.execute("SET lock_configuration = true", Parameters::None)?;
    for pragma in ["PRAGMA bolt_request_timeout(1)", "PRAGMA bolt_reset"] {
        let msg = query_error(&conn, pragma).unwrap_or_else(|| panic!("{pragma} should fail"));
        assert!(msg.contains("lock_configuration = true"), "{pragma}: {msg}");
    }
    assert_eq!(runtime.request_timeout(), configured, "timeout unchanged");
    // Catalog pragmas are not configuration and stay available.
    conn.execute("PRAGMA bolt_clear_tables", Parameters::None)?;

    // Another, unlocked database in the same process is not affected by the lock.
    let (_env2, other) = setup_connection()?;
    let restore = before.map_or(0, |d| d.as_millis() as i64);
    other.execute(
        format!("PRAGMA bolt_request_timeout({restore})"),
        Parameters::None,
    )?;
    assert_eq!(runtime.request_timeout(), before);
    Ok(())
}

#[test]
fn test_record_batches_to_collection_rejects_column_count_mismatch() -> Result<()> {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let (_env, conn) = setup_connection()?;
    let two = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int32, true),
        Field::new("b", DataType::Utf8, true),
    ]));
    let one = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, true)]));
    let narrow =
        RecordBatch::try_new(one.clone(), vec![Arc::new(Int32Array::from(vec![1, 2]))]).unwrap();
    let wide = RecordBatch::try_new(
        two.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2])),
            Arc::new(StringArray::from(vec!["x", "y"])),
        ],
    )
    .unwrap();

    // Fewer columns than the schema: refused (would leave vectors uninitialized).
    let Err(err) = record_batches_to_collection(&conn, &two, std::slice::from_ref(&narrow)) else {
        panic!("narrow batch must be refused");
    };
    assert!(
        err.message.contains("batch has 1 columns, schema has 2"),
        "{err}"
    );
    // More columns than the schema: refused too.
    let Err(err) = record_batches_to_collection(&conn, &one, std::slice::from_ref(&wide)) else {
        panic!("wide batch must be refused");
    };
    assert!(
        err.message.contains("batch has 2 columns, schema has 1"),
        "{err}"
    );
    // A mismatch after a good batch is still found.
    let Err(err) = record_batches_to_collection(&conn, &two, &[wide.clone(), narrow]) else {
        panic!("second batch must be refused");
    };
    assert!(err.message.contains("schema has 2"), "{err}");
    // Matching batches still work.
    record_batches_to_collection(&conn, &two, &[wide.clone(), wide])?;
    Ok(())
}

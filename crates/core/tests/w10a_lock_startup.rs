// SPDX-License-Identifier: Apache-2.0

//! Wave 10 (W10-A) regression tests: Bolt functions keep working in a database that runs with
//! `lock_configuration = true` while its `bolt_*` settings disagree with the process-wide
//! runtime (W10-A-001):
//!
//! * a `bolt_*` startup option together with `lock_configuration = true` at open;
//! * a `bolt_*` startup option, then `SET lock_configuration = true` before the first Bolt call;
//! * a setting another database made stale, then `SET lock_configuration = true`.
//!
//! The settings sync (`options::sync_with_runtime`) writes settings with `Config::set_value`,
//! which DuckDB may refuse under the lock. Either way every Bolt call below must succeed.
//! DuckDB 2.0 may refuse an unrecognized startup option at open (before statically linked
//! extensions are loaded); then the startup cases only check that the refusal names the
//! option. Valid for both GPU and `cuda-stub` builds: nothing here depends on the backend.

use std::sync::{Mutex, Once};

use duckdb_bolt_core::runtime::BoltRuntime;
use duckdb_loadable_macros::duckdb_entrypoint_c_api_v2;
use duckdb_neo::{
    Parameters, Result,
    connection::Connection,
    environment::{Environment, StorageLocation},
    extension::{Extension, register_static_extension},
    instance::{AttachBuilder, InstanceBuilder},
};

#[duckdb_entrypoint_c_api_v2(extension_name = "bolt", api_version = "2.0.0")]
fn bolt_w10a_test_entrypoint(extension: &Extension) -> Result<()> {
    duckdb_bolt_core::register(extension)
}

static REGISTER_ONCE: Once = Once::new();
// Serialize tests because Craton Bolt's runtime catalog and config are process-wide.
static TEST_MUTEX: Mutex<()> = Mutex::new(());

/// Startup value used by the tests: 2 MiB, valid on every backend.
const BUDGET: i64 = 2_097_152;
const BUDGET_BYTES: usize = 2_097_152;

fn lock() -> std::sync::MutexGuard<'static, ()> {
    TEST_MUTEX.lock().unwrap_or_else(|p| p.into_inner())
}

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
    Ok((env, conn))
}

/// A fresh in-memory database opened with `options`, or the open error's message.
fn open_with(
    options: &[(&str, &str)],
) -> Result<std::result::Result<(Environment, Connection), String>> {
    register_extension();
    let env = Environment::new()?;
    let instance = match InstanceBuilder::new()
        .options(options.iter().copied())
        .build(&env)
    {
        Ok(instance) => instance,
        Err(e) => return Ok(Err(e.message)),
    };
    instance.attach(AttachBuilder::new(StorageLocation::InMemory).make_default())?;
    let conn = instance.connect()?;
    Ok(Ok((env, conn)))
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

fn expect_ok(conn: &Connection, sql: &str) {
    if let Some(msg) = query_error(conn, sql) {
        panic!("`{sql}` should succeed under lock_configuration = true, got: {msg}");
    }
}

fn single_i64(conn: &Connection, sql: &str) -> Result<i64> {
    let mut result = conn.query(sql, Parameters::None)?;
    let chunk = result.next_chunk()?.expect("one result row");
    let vec = chunk.get_vector_at::<i64>(0)?;
    Ok(*vec.reader()?.get(0)?.expect("non-null i64"))
}

/// Restore the documented defaults from an unlocked database and empty the catalog.
fn reset_all() -> Result<()> {
    let (_env, conn) = setup_connection()?;
    for opt in [
        "bolt_device",
        "bolt_memory_budget",
        "bolt_ptx_cache_dir",
        "bolt_enable_tracing",
    ] {
        conn.execute(format!("RESET {opt}"), Parameters::None)?;
    }
    conn.execute("PRAGMA bolt_clear_tables", Parameters::None)?;
    Ok(())
}

/// A few Bolt calls of every kind (scalar, table function, pragma); each runs the settings
/// sync first.
fn bolt_calls_succeed(conn: &Connection, table: &str) -> Result<()> {
    expect_ok(conn, "SELECT bolt_version()");
    expect_ok(
        conn,
        &format!(
            "SELECT bolt_register_sql('{table}', 'SELECT * FROM (VALUES (1::BIGINT), (2), (3)) t(v)')"
        ),
    );
    assert_eq!(
        single_i64(
            conn,
            &format!("SELECT sum(v)::BIGINT FROM bolt_scan('{table}')")
        )?,
        6
    );
    expect_ok(conn, "PRAGMA bolt_info");
    expect_ok(conn, "SELECT * FROM bolt_pool_stats()");
    // Repeated calls take the same path again (no one-shot success).
    expect_ok(conn, "SELECT bolt_version()");
    Ok(())
}

#[test]
fn test_startup_value_with_lock_configuration_at_open() -> Result<()> {
    let _guard = lock();
    reset_all()?;
    let budget = BUDGET.to_string();
    let opened = open_with(&[
        ("bolt_memory_budget", budget.as_str()),
        ("lock_configuration", "true"),
    ])?;
    match opened {
        // DuckDB refuses unrecognized startup options before static extensions load.
        Err(msg) => assert!(msg.contains("bolt_memory_budget"), "{msg}"),
        Ok((_env, conn)) => {
            bolt_calls_succeed(&conn, "w10a_open")?;
            // The startup value was applied to the runtime (through the option's callback).
            assert_eq!(
                BoltRuntime::global().config()?.memory_budget,
                Some(BUDGET_BYTES)
            );
            assert_eq!(
                single_i64(
                    &conn,
                    "SELECT current_setting('bolt_memory_budget')::BIGINT"
                )?,
                BUDGET
            );
        }
    }
    reset_all()
}

#[test]
fn test_startup_value_then_set_lock_configuration() -> Result<()> {
    let _guard = lock();
    reset_all()?;
    let budget = BUDGET.to_string();
    let opened = open_with(&[("bolt_memory_budget", budget.as_str())])?;
    match opened {
        Err(msg) => assert!(msg.contains("bolt_memory_budget"), "{msg}"),
        Ok((_env, conn)) => {
            // Locked before any Bolt call, so the startup value is still unapplied.
            conn.execute("SET lock_configuration = true", Parameters::None)?;
            bolt_calls_succeed(&conn, "w10a_set")?;
            assert_eq!(
                BoltRuntime::global().config()?.memory_budget,
                Some(BUDGET_BYTES)
            );
        }
    }
    reset_all()
}

#[test]
fn test_stale_setting_then_set_lock_configuration() -> Result<()> {
    let _guard = lock();
    reset_all()?;
    // `locked` loads the extension while the runtime has the default budget.
    let (_env_a, locked) = setup_connection()?;
    expect_ok(&locked, "SELECT bolt_version()");
    // Another database changes the shared runtime, so `locked`'s setting is now stale.
    let (_env_b, other) = setup_connection()?;
    other.execute(
        format!("SET GLOBAL bolt_memory_budget = {BUDGET}"),
        Parameters::None,
    )?;
    locked.execute("SET lock_configuration = true", Parameters::None)?;
    // The refresh of the stale setting is best effort; Bolt calls must not fail on it.
    bolt_calls_succeed(&locked, "w10a_stale")?;
    // The locked database did not override the other database's change.
    assert_eq!(
        BoltRuntime::global().config()?.memory_budget,
        Some(BUDGET_BYTES)
    );
    // `SET` of a Bolt option is still refused by the lock.
    assert!(
        query_error(&locked, "SET GLOBAL bolt_memory_budget = 0").is_some(),
        "the lock must still refuse SET"
    );
    reset_all()
}

#[test]
fn test_set_value_then_lock_before_first_bolt_call() -> Result<()> {
    let _guard = lock();
    reset_all()?;
    let (_env, conn) = setup_connection()?;
    // A value set with SET but not yet synced to the runtime, then the lock.
    conn.execute(
        format!("SET GLOBAL bolt_memory_budget = {BUDGET}"),
        Parameters::None,
    )?;
    conn.execute("SET lock_configuration = true", Parameters::None)?;
    bolt_calls_succeed(&conn, "w10a_set_then_lock")?;
    assert_eq!(
        BoltRuntime::global().config()?.memory_budget,
        Some(BUDGET_BYTES)
    );
    reset_all()
}

// SPDX-License-Identifier: Apache-2.0

//! Wave 5 (W5-A) runtime regression tests: a timed-out request never interrupts the next
//! host-fallback query (W4-A-004), and `COPY ... (MODE)` reports the canonical mode message
//! (W4-A-003). They run on both backends.

use std::{
    sync::{Mutex, Once},
    time::Duration,
};

use duckdb_bolt_core::runtime::{BoltRuntime, RegisterMode};
use duckdb_loadable_macros::duckdb_entrypoint_c_api_v2;
use duckdb_neo::{
    Parameters, Result,
    connection::Connection,
    environment::{Environment, StorageLocation},
    extension::{Extension, register_static_extension},
};

#[duckdb_entrypoint_c_api_v2(extension_name = "bolt", api_version = "2.0.0")]
fn bolt_w5a_runtime_entrypoint(extension: &Extension) -> Result<()> {
    duckdb_bolt_core::register(extension)
}

static REGISTER_ONCE: Once = Once::new();
// Serialize tests because Craton Bolt's runtime catalog is process-wide.
static TEST_MUTEX: Mutex<()> = Mutex::new(());

fn lock() -> std::sync::MutexGuard<'static, ()> {
    TEST_MUTEX.lock().unwrap_or_else(|p| p.into_inner())
}

fn setup_connection() -> Result<(Environment, Connection)> {
    REGISTER_ONCE.call_once(|| {
        // SAFETY: `duckdb_extension_bolt_describe` is emitted by `#[duckdb_entrypoint_c_api_v2]`,
        // which fills the descriptor as `duckdb_static_extension.h` specifies (the contract of
        // `register_static_extension`); `Once` makes this a single call.
        unsafe { register_static_extension(duckdb_extension_bolt_describe) }
            .expect("Failed to register static bolt extension");
    });
    let env = Environment::new()?;
    let instance = env.open(StorageLocation::InMemory)?;
    let conn = instance.connect()?;
    conn.execute("PRAGMA bolt_clear_tables", Parameters::None)?;
    Ok((env, conn))
}

fn exec(conn: &Connection, sql: &str) -> Result<()> {
    conn.execute(sql, Parameters::None).map(|_| ())
}

/// W4-A-004: whatever the timing of a timeout, the interrupt it issues only hits its own
/// request; the next host-fallback query always succeeds.
#[test]
fn timed_out_requests_never_interrupt_the_next_fallback_query() -> Result<()> {
    let _guard = lock();
    let (_env, conn) = setup_connection()?;
    exec(
        &conn,
        "COPY (SELECT range::INTEGER AS i FROM range(0, 100000)) TO 'w5a_slow' \
         (FORMAT bolt, USE_TMP_FILE FALSE)",
    )?;
    let rt = BoltRuntime::global();
    // Timeouts from "long before the end" to "about when the query ends", so some interrupts
    // land while the query runs and some just after it finished.
    for (round, timeout_ms) in [300u64, 1, 2, 5, 10, 20, 1, 3, 7, 15, 30, 60]
        .into_iter()
        .enumerate()
    {
        rt.set_request_timeout(Some(Duration::from_millis(timeout_ms)));
        let slow = if round == 0 {
            // Minutes without the timeout.
            "SELECT count(*) FROM w5a_slow a, w5a_slow b WHERE (a.i * 7 + b.i) % 1000003 = -5"
        } else {
            "SELECT count(*) FROM w5a_slow a, w5a_slow b WHERE a.i < 300 AND (a.i + b.i) % 97 = 3"
        };
        let _ = rt.execute_sql_host_fallback(slow);
        rt.set_request_timeout(None);
        for _ in 0..3 {
            let out = rt
                .execute_sql_host_fallback("SELECT count(*) FROM w5a_slow WHERE i % 2 = 0")
                .unwrap_or_else(|e| {
                    panic!("round {round} ({timeout_ms} ms): next query failed: {e}")
                });
            assert_eq!(out.batch.num_rows(), 1);
        }
    }
    rt.set_request_timeout(None);
    exec(&conn, "PRAGMA bolt_clear_tables")?;
    Ok(())
}

/// W4-A-003: `COPY ... (MODE)` starts with the canonical message (W1-E-007).
#[test]
fn copy_mode_error_is_the_canonical_message() -> Result<()> {
    let _guard = lock();
    let (_env, conn) = setup_connection()?;
    let err = conn
        .execute(
            "COPY (SELECT 1 AS a) TO 'w5a_mode' (FORMAT bolt, USE_TMP_FILE FALSE, MODE 'Sideways')",
            Parameters::None,
        )
        .expect_err("invalid mode");
    assert!(
        err.message
            .contains(&RegisterMode::invalid_mode_message("Sideways")),
        "{}",
        err.message
    );
    assert!(
        err.message.contains("COPY TO (FORMAT bolt) option MODE"),
        "{}",
        err.message
    );
    Ok(())
}

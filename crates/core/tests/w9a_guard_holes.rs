// SPDX-License-Identifier: Apache-2.0

//! Wave 9 (W9-A) regression tests for the wrong-answer shapes the GPU result guard let through,
//! as filed by the W8-B review:
//!
//! - W8-B-001: integer `/` truncates and `x / 0`, `x % 0` give `0` on the GPU;
//! - W8-B-002: fractional literals are DOUBLE on the GPU but DECIMAL in DuckDB;
//! - W8-B-003: shapes the guard's planner rejects (DISTINCT ON, correlated subqueries, ...)
//!   skipped every plan check over NULL-free tables;
//! - W8-B-004: the per-column NULL lookup went by name, so aliases, UNION branches and NULL
//!   literals hid nullable columns;
//! - W8-B-005: the empty-input check missed LIMIT / OFFSET and EXCEPT / INTERSECT;
//! - W8-B-006: ungrouped STDDEV_POP over an empty input returns `0.0`;
//! - W8-B-007: SUM accumulates FLOAT in single precision and BIGINT in a wrapping Int64;
//! - W8-B-008: `TIMESTAMP - TIMESTAMP` returns raw microsecond ticks.
//!
//! Every check compares `bolt_query` with the same query on the DuckDB source table, on both
//! backends. With a GPU the routing is checked too: the hazards run on the host fallback, and
//! the shapes that are correct on the engine keep running there.

use std::sync::{Mutex, Once};

use duckdb_bolt_core::runtime::BoltRuntime;
use duckdb_loadable_macros::duckdb_entrypoint_c_api_v2;
use duckdb_neo::{
    Parameters, Result,
    connection::Connection,
    environment::{Environment, StorageLocation},
    extension::{Extension, register_static_extension},
};

#[duckdb_entrypoint_c_api_v2(extension_name = "bolt", api_version = "2.0.0")]
fn bolt_w9a_guard_entrypoint(extension: &Extension) -> Result<()> {
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

fn query_i64(conn: &Connection, sql: &str) -> Result<i64> {
    let mut result = conn.query(sql, Parameters::None)?;
    let chunk = result.next_chunk()?.expect("Expected at least one chunk");
    let vec = chunk.get_vector_at::<i64>(0)?;
    Ok(*vec.reader()?.get(0)?.expect("Expected non-null i64"))
}

fn query_string(conn: &Connection, sql: &str) -> Result<String> {
    let mut result = conn.query(sql, Parameters::None)?;
    let chunk = result.next_chunk()?.expect("Expected at least one chunk");
    let vec = chunk.get_vector_at::<String>(0)?;
    Ok(vec
        .reader()?
        .get(0)?
        .map(|s| s.to_string())
        .unwrap_or_default())
}

fn gpu() -> bool {
    BoltRuntime::global().telemetry().unwrap().backend == "cuda-gpu"
}

/// `CREATE TABLE <src> AS <select>` and its Bolt copy `<bolt>`.
fn build(conn: &Connection, src: &str, bolt: &str, select: &str) -> Result<()> {
    exec(conn, &format!("CREATE OR REPLACE TABLE {src} AS {select}"))?;
    exec(
        conn,
        &format!("COPY {src} TO '{bolt}' (FORMAT bolt, USE_TMP_FILE FALSE)"),
    )
}

/// `bolt_query(<sql>)` as a SQL string literal argument.
fn bolt_query(sql: &str) -> String {
    format!("FROM bolt_query('{}')", sql.replace('\'', "''"))
}

/// Asserts that `bolt_query(<sql over bolt>)` returns the rows of `<sql over src>` (as a
/// multiset). `{t}` in `sql` names the table.
fn assert_same(conn: &Connection, sql: &str, src: &str, bolt: &str) -> Result<()> {
    let reference = sql.replace("{t}", src);
    let actual = bolt_query(&sql.replace("{t}", bolt));
    exec(
        conn,
        &format!("CREATE OR REPLACE TEMP TABLE w9a_expected AS {reference}"),
    )?;
    exec(
        conn,
        &format!("CREATE OR REPLACE TEMP TABLE w9a_actual AS {actual}"),
    )?;
    let differing = query_i64(
        conn,
        "SELECT count(*) FROM ((FROM w9a_expected EXCEPT ALL FROM w9a_actual) \
         UNION ALL (FROM w9a_actual EXCEPT ALL FROM w9a_expected))",
    )?;
    if differing != 0 {
        let show = |table: &str| {
            query_string(
                conn,
                &format!("SELECT CAST(list(x) AS VARCHAR) FROM {table} x"),
            )
            .unwrap_or_else(|e| e.message)
        };
        panic!(
            "{sql}: DuckDB {} vs bolt_query {}",
            show("w9a_expected"),
            show("w9a_actual")
        );
    }
    Ok(())
}

/// Asserts that `<sql over src>` fails in DuckDB and `bolt_query(<sql over bolt>)` fails with
/// the same error (`needle`).
fn assert_both_fail(conn: &Connection, sql: &str, src: &str, bolt: &str, needle: &str) {
    let fetch = |q: String| {
        conn.query(q, Parameters::None)
            .and_then(|mut r| r.next_chunk().map(|_| ()))
    };
    let duck = fetch(sql.replace("{t}", src)).expect_err(sql);
    assert!(
        duck.message.contains(needle),
        "{sql}: DuckDB {}",
        duck.message
    );
    let bolt = fetch(bolt_query(&sql.replace("{t}", bolt))).expect_err(sql);
    assert!(
        bolt.message.contains(needle),
        "{sql}: bolt_query {}",
        bolt.message
    );
}

/// With a GPU: why the crash guard routes `sql` to the host fallback (`None`: it runs on the
/// engine). Read from `bolt_explain`.
fn host_reason(sql: &str) -> Option<String> {
    let text = BoltRuntime::global().explain_sql(sql).expect("explain");
    let marker = "Executed on the host fallback (DuckDB): ";
    text.find(marker).map(|at| {
        let rest = &text[at + marker.len()..];
        rest[..rest.find(" ===").unwrap_or(rest.len())].to_string()
    })
}

fn routed_to_host(sql: &str) -> bool {
    host_reason(sql).is_some()
}

#[test]
fn integer_division_and_modulo_match_duckdb() -> Result<()> {
    let _guard = lock();
    let (_env, conn) = setup_connection()?;
    build(
        &conn,
        "w9a_div",
        "w9a_divb",
        "SELECT i::INTEGER AS id, (i % 3)::INTEGER AS z, i::BIGINT AS b, i::DOUBLE AS d \
         FROM range(0, 8) r(i)",
    )?;
    for sql in [
        "SELECT count(*) FROM {t} WHERE id / 2 > 1",
        "SELECT sum(id / 2) FROM {t}",
        "SELECT count(*) FROM {t} WHERE b / 4 >= 1",
        "SELECT count(*) FROM {t} WHERE id % 3 = 1",
        "SELECT id, d / 4, d / z, d % z FROM {t}",
        "SELECT count(*) FROM {t} WHERE id = 7 / 2 + 0.5",
    ] {
        assert_same(&conn, sql, "w9a_div", "w9a_divb")?;
    }
    // An integer zero divisor of `%` is an error in DuckDB (the engine returned NULL).
    assert_both_fail(
        &conn,
        "SELECT id, id % z FROM {t} WHERE z = 0",
        "w9a_div",
        "w9a_divb",
        "Division by zero",
    );
    // `/` is float division in DuckDB (`0 / 0` is NaN, `3 / 0` is inf), and since W11-A
    // `bolt_query` binds DuckDB's DOUBLE (W9-A-001), not the engine's NULL.
    assert_same(
        &conn,
        "SELECT id, b / z FROM {t} WHERE z = 0",
        "w9a_div",
        "w9a_divb",
    )?;
    if gpu() {
        assert!(routed_to_host("SELECT id / 2 FROM w9a_divb"));
        assert!(routed_to_host("SELECT id % 2 FROM w9a_divb"));
        assert!(routed_to_host("SELECT b / z FROM w9a_divb"));
        // Constant folding must not hide it (the optimizer folds `7 / 2` to `3`).
        assert!(routed_to_host("SELECT id FROM w9a_divb WHERE id = 7 / 2"));
        // Float division and the other integer operators stay on the GPU.
        assert!(!routed_to_host("SELECT d / 4 FROM w9a_divb"), "d / 4");
        assert!(!routed_to_host("SELECT id + 1, b * 2 FROM w9a_divb"));
    }
    exec(&conn, "PRAGMA bolt_clear_tables")?;
    Ok(())
}

#[test]
fn fractional_literals_are_exact_decimals() -> Result<()> {
    let _guard = lock();
    let (_env, conn) = setup_connection()?;
    build(
        &conn,
        "w9a_frac",
        "w9a_fracb",
        "SELECT i::INTEGER AS id, i::BIGINT AS b, i::DOUBLE AS d FROM range(0, 10) r(i)",
    )?;
    for sql in [
        "SELECT count(*) FROM {t} WHERE id * 0.1 = 0.3",
        "SELECT id FROM {t} WHERE id * 0.7 >= 2.1",
        "SELECT b FROM {t} WHERE b * 1.1 <= 3.3",
        "SELECT id FROM {t} WHERE id > 2.5",
        "SELECT id, d * 0.75 FROM {t}",
    ] {
        assert_same(&conn, sql, "w9a_frac", "w9a_fracb")?;
    }
    if gpu() {
        assert!(routed_to_host("SELECT id * 0.1 FROM w9a_fracb"));
        assert!(routed_to_host("SELECT 0.1 + 0.2 FROM w9a_fracb"));
        // DOUBLE arithmetic with a fractional literal, and comparisons, stay on the GPU.
        assert!(!routed_to_host("SELECT d * 0.75 FROM w9a_fracb"));
        assert!(!routed_to_host("SELECT id FROM w9a_fracb WHERE id > 2.5"));
    }
    exec(&conn, "PRAGMA bolt_clear_tables")?;
    Ok(())
}

#[test]
fn planner_rejected_shapes_are_checked() -> Result<()> {
    let _guard = lock();
    let (_env, conn) = setup_connection()?;
    build(
        &conn,
        "w9a_rej",
        "w9a_rejb",
        "SELECT i::INTEGER AS id, (i % 3)::INTEGER AS k, ((i - 5) / 4.0)::DECIMAL(9, 2) AS d \
         FROM range(0, 10) r(i)",
    )?;
    for sql in [
        "SELECT DISTINCT ON (k) k, CAST(d AS DOUBLE) AS x FROM {t} ORDER BY k, id",
        "SELECT id FROM {t} WHERE id > (SELECT max(u.id) FROM {t} u WHERE u.k = {t}.k AND u.id < 0)",
        "SELECT id, CAST(d AS DOUBLE) AS x FROM {t} WHERE EXISTS \
         (SELECT 1 FROM {t} u WHERE u.k = {t}.k AND u.id > {t}.id)",
        "SELECT DISTINCT ON (k) k, CAST(d AS DOUBLE) AS x FROM {t} ORDER BY k",
        "SELECT a.id, l.m FROM {t} a, \
         LATERAL (SELECT max(u.id) AS m FROM {t} u WHERE u.k = a.k AND u.id < 0) l",
        "SELECT x / 2 AS h FROM (VALUES (7), (9)) v(x)",
        "SELECT generate_series / 2 AS h FROM generate_series(1, 5)",
        "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r WHERE n < 5) \
         SELECT n / 2 AS h FROM r",
    ] {
        assert_same(&conn, sql, "w9a_rej", "w9a_rejb")?;
    }
    if gpu() {
        for sql in [
            "SELECT DISTINCT ON (k) k, CAST(d AS DOUBLE) AS x FROM w9a_rejb ORDER BY k",
            "SELECT a.id, l.m FROM w9a_rejb a, \
             LATERAL (SELECT max(u.id) AS m FROM w9a_rejb u WHERE u.k = a.k) l",
            "SELECT x FROM (VALUES (7), (9)) v(x)",
        ] {
            let reason = host_reason(sql).unwrap_or_default();
            assert!(reason.contains("W8-B-003"), "{sql}: {reason}");
        }
        // Shapes the planner plans keep running on the GPU.
        assert!(!routed_to_host("SELECT DISTINCT k FROM w9a_rejb"));
        assert!(!routed_to_host("SELECT id, k FROM w9a_rejb WHERE id > 3"));
    }
    exec(&conn, "PRAGMA bolt_clear_tables")?;
    Ok(())
}

#[test]
fn planner_rejected_shapes_get_the_dictionary_check() -> Result<()> {
    let _guard = lock();
    let (_env, conn) = setup_connection()?;
    build(
        &conn,
        "w9a_wide",
        "w9a_wideb",
        "SELECT i::INTEGER AS k, 'v' || i AS s FROM range(0, 5000) r(i)",
    )?;
    let sql = "SELECT DISTINCT ON (k) k, s FROM {t} WHERE s LIKE 'v1%' ORDER BY k";
    if gpu() {
        // Checked before running it: on the engine this could abort the process.
        assert!(routed_to_host(&sql.replace("{t}", "w9a_wideb")));
    }
    assert_same(&conn, sql, "w9a_wide", "w9a_wideb")?;
    exec(&conn, "PRAGMA bolt_clear_tables")?;
    Ok(())
}

#[test]
fn null_lookup_follows_the_plan_not_the_name() -> Result<()> {
    let _guard = lock();
    let (_env, conn) = setup_connection()?;
    build(
        &conn,
        "w9a_nl",
        "w9a_nlb",
        "SELECT (i % 3)::INTEGER AS k, i::INTEGER AS v, \
         CASE WHEN i % 3 = 2 THEN NULL ELSE i END::INTEGER AS w FROM range(0, 9) r(i)",
    )?;
    for sql in [
        "SELECT k, max(v) AS m FROM (SELECT k, w AS v FROM {t}) s GROUP BY k",
        "SELECT v FROM (SELECT w AS v FROM {t}) s ORDER BY v LIMIT 3",
        "SELECT k, max(x) FROM (SELECT k, NULL::INTEGER AS v, w AS x FROM {t}) s GROUP BY k",
    ] {
        assert_same(&conn, sql, "w9a_nl", "w9a_nlb")?;
    }
    build(
        &conn,
        "w9a_ua",
        "w9a_uab",
        "SELECT 1 AS k, i::INTEGER AS x FROM range(0, 3) r(i)",
    )?;
    build(
        &conn,
        "w9a_ub",
        "w9a_ubb",
        "SELECT 2 AS k, NULL::INTEGER AS y FROM range(0, 3) r(i)",
    )?;
    let union = "SELECT k, sum(x) FROM (SELECT k, x FROM {a} UNION ALL SELECT k, y FROM {b}) u \
                 GROUP BY k";
    exec(
        &conn,
        &format!(
            "CREATE OR REPLACE TEMP TABLE w9a_expected AS {}",
            union.replace("{a}", "w9a_ua").replace("{b}", "w9a_ub")
        ),
    )?;
    let nulls = query_i64(
        &conn,
        &format!(
            "SELECT count(*) FROM ({}) r(k, s) WHERE k = 2 AND s IS NULL",
            bolt_query(&union.replace("{a}", "w9a_uab").replace("{b}", "w9a_ubb"))
        ),
    )?;
    assert_eq!(nulls, 1, "the all-NULL union group must sum to NULL");
    if gpu() {
        assert!(routed_to_host(
            "SELECT k, max(v) FROM (SELECT k, w AS v FROM w9a_nlb) s GROUP BY k"
        ));
        assert!(routed_to_host(
            "SELECT v FROM (SELECT w AS v FROM w9a_nlb) s ORDER BY v"
        ));
        // NULL-free columns, also renamed, stay on the GPU.
        assert!(!routed_to_host("SELECT k, max(v) FROM w9a_nlb GROUP BY k"));
        assert!(!routed_to_host(
            "SELECT k, max(x) FROM (SELECT k, v AS x FROM w9a_nlb) s GROUP BY k"
        ));
        assert!(!routed_to_host("SELECT v FROM w9a_nlb ORDER BY v"));
    }
    exec(&conn, "PRAGMA bolt_clear_tables")?;
    Ok(())
}

#[test]
fn empty_inputs_through_limit_offset_and_set_operations_aggregate_to_null() -> Result<()> {
    let _guard = lock();
    let (_env, conn) = setup_connection()?;
    build(
        &conn,
        "w9a_em",
        "w9a_emb",
        "SELECT i::INTEGER AS id, i::BIGINT AS b FROM range(0, 100) r(i)",
    )?;
    for sql in [
        "SELECT max(b), min(id), sum(b) FROM (SELECT id, b FROM {t} LIMIT 0) s",
        "SELECT max(b) FROM (SELECT id, b FROM {t} LIMIT 10 OFFSET 1000) s",
        "SELECT sum(b) FROM (SELECT b FROM {t} EXCEPT SELECT b FROM {t}) s",
        "SELECT min(b) FROM (SELECT b FROM {t} WHERE b < 5 INTERSECT SELECT b FROM {t} WHERE b > 5) s",
        "SELECT max(b) FROM (SELECT id, b FROM {t} LIMIT 10) s",
    ] {
        assert_same(&conn, sql, "w9a_em", "w9a_emb")?;
    }
    if gpu() {
        assert!(routed_to_host(
            "SELECT max(b) FROM (SELECT b FROM w9a_emb LIMIT 0) s"
        ));
        // A non-empty input stays on the GPU.
        assert!(!routed_to_host("SELECT max(b), sum(b) FROM w9a_emb"));
        assert!(!routed_to_host(
            "SELECT max(b) FROM (SELECT b FROM w9a_emb LIMIT 10) s"
        ));
    }
    exec(&conn, "PRAGMA bolt_clear_tables")?;
    Ok(())
}

#[test]
fn statistical_aggregates_over_an_empty_input_are_null() -> Result<()> {
    let _guard = lock();
    let (_env, conn) = setup_connection()?;
    build(
        &conn,
        "w9a_sd",
        "w9a_sdb",
        "SELECT i::INTEGER AS id, i::DOUBLE AS x FROM range(0, 100) r(i)",
    )?;
    let nulls = query_i64(
        &conn,
        &format!(
            "SELECT count(*) FROM ({}) r(a, b, c, d) \
             WHERE a IS NULL AND b IS NULL AND c IS NULL AND d IS NULL",
            bolt_query(
                "SELECT stddev_pop(x), var_pop(x), stddev_samp(x), var_samp(x) FROM w9a_sdb \
                 WHERE id < 0"
            )
        ),
    )?;
    assert_eq!(nulls, 1, "statistical aggregates of no rows are NULL");
    // Over the whole table they match DuckDB.
    let off = query_i64(
        &conn,
        &format!(
            "SELECT count(*) FROM ({}) r(a, b), \
             (SELECT stddev_pop(x), var_samp(x) FROM w9a_sd) e(a, b) \
             WHERE abs(r.a - e.a) > 1e-9 OR abs(r.b - e.b) > 1e-6",
            bolt_query("SELECT stddev_pop(x), var_samp(x) FROM w9a_sdb")
        ),
    )?;
    assert_eq!(off, 0);
    if gpu() {
        assert!(routed_to_host(
            "SELECT stddev_pop(x) FROM w9a_sdb WHERE id < 0"
        ));
        // The other statistical aggregates are NULL over no rows on the engine too.
        assert!(!routed_to_host(
            "SELECT var_pop(x), stddev_samp(x), var_samp(x) FROM w9a_sdb WHERE id < 0"
        ));
        assert!(!routed_to_host("SELECT stddev_pop(x) FROM w9a_sdb"));
    }
    exec(&conn, "PRAGMA bolt_clear_tables")?;
    Ok(())
}

#[test]
fn float_and_bigint_sums_do_not_lose_precision_or_wrap() -> Result<()> {
    let _guard = lock();
    let (_env, conn) = setup_connection()?;
    build(
        &conn,
        "w9a_fs",
        "w9a_fsb",
        "SELECT (0.1 + (i % 7) * 1000)::FLOAT AS c, (i % 2)::INTEGER AS k \
         FROM range(0, 1000000) r(i)",
    )?;
    // DuckDB sums FLOAT in DOUBLE, and bolt_query returns that DOUBLE (W9-A-001). The order of
    // the additions differs between runs, so compare at FLOAT precision.
    for sql in [
        "SELECT sum(c)::FLOAT FROM {t}",
        "SELECT k, sum(c)::FLOAT FROM {t} GROUP BY k",
    ] {
        let reference = sql.replace("{t}", "w9a_fs");
        exec(
            &conn,
            &format!("CREATE OR REPLACE TEMP TABLE w9a_expected AS {reference}"),
        )?;
        exec(
            &conn,
            &format!(
                "CREATE OR REPLACE TEMP TABLE w9a_actual AS {}",
                bolt_query(&sql.replace("{t}", "w9a_fsb"))
            ),
        )?;
        let differing = query_i64(
            &conn,
            "SELECT count(*) FROM ((FROM w9a_expected EXCEPT ALL FROM w9a_actual) \
             UNION ALL (FROM w9a_actual EXCEPT ALL FROM w9a_expected))",
        )?;
        let shown = query_string(&conn, "SELECT CAST(list(x) AS VARCHAR) FROM w9a_actual x")?;
        assert_eq!(differing, 0, "{sql}: bolt_query {shown}");
    }
    build(
        &conn,
        "w9a_bs",
        "w9a_bsb",
        "SELECT 6000000000000000000::BIGINT AS v, i::BIGINT AS s FROM range(0, 2) r(i)",
    )?;
    // DuckDB sums into a HUGEINT, and so does bolt_query since W11-A; never a wrapped value.
    assert_same(&conn, "SELECT sum(v) FROM {t}", "w9a_bs", "w9a_bsb")?;
    assert_same(&conn, "SELECT avg(v) FROM {t}", "w9a_bs", "w9a_bsb")?;
    assert_same(&conn, "SELECT sum(s) FROM {t}", "w9a_bs", "w9a_bsb")?;
    if gpu() {
        assert!(routed_to_host("SELECT sum(c) FROM w9a_fsb"));
        assert!(routed_to_host("SELECT sum(v) FROM w9a_bsb"));
        // Sums that cannot overflow stay on the GPU.
        assert!(!routed_to_host("SELECT sum(s) FROM w9a_bsb"));
        assert!(!routed_to_host(
            "SELECT k, count(c), max(c) FROM w9a_fsb GROUP BY k"
        ));
    }
    exec(&conn, "PRAGMA bolt_clear_tables")?;
    Ok(())
}

#[test]
fn timestamp_subtraction_is_not_raw_ticks() -> Result<()> {
    let _guard = lock();
    let (_env, conn) = setup_connection()?;
    build(
        &conn,
        "w9a_ts",
        "w9a_tsb",
        "SELECT TIMESTAMP '2026-01-01 00:00:00' + to_hours(i) AS t0, \
         TIMESTAMP '2026-01-02 00:00:00' + to_hours(i) AS t1, \
         DATE '2026-01-01' + i::INTEGER AS d0, DATE '2026-03-01' AS d1 \
         FROM range(0, 3) r(i)",
    )?;
    // DuckDB returns an INTERVAL, and so does bolt_query since W11-A (W9-A-002), never ticks.
    assert_same(&conn, "SELECT t1 - t0 AS d FROM {t}", "w9a_ts", "w9a_tsb")?;
    assert_same(&conn, "SELECT d1 - d0 AS n FROM {t}", "w9a_ts", "w9a_tsb")?;
    if gpu() {
        assert!(routed_to_host("SELECT t1 - t0 FROM w9a_tsb"));
        assert!(!routed_to_host("SELECT t0, t1 FROM w9a_tsb WHERE t1 > t0"));
    }
    exec(&conn, "PRAGMA bolt_clear_tables")?;
    Ok(())
}

#[test]
fn counts_of_nullable_columns_below_limit_and_sort() -> Result<()> {
    let _guard = lock();
    let (_env, conn) = setup_connection()?;
    build(
        &conn,
        "w9a_cn",
        "w9a_cnb",
        "SELECT i::INTEGER AS id, NULL::BOOLEAN AS c0, \
         CASE WHEN i % 3 = 0 THEN NULL ELSE i END::INTEGER AS v FROM range(0, 50) r(i)",
    )?;
    // W9-A-003 (found by the W4-B fuzzer): below a Sort, Limit or Distinct the engine counted
    // the NULLs of an aggregated column (`count(c0)` of an all-NULL column returned 25).
    for sql in [
        "SELECT count(c0) FROM (SELECT id, c0 FROM {t} ORDER BY id LIMIT 100 OFFSET 25) s",
        "SELECT count(c0) FROM (SELECT id, c0 FROM {t} LIMIT 100 OFFSET 25) s",
        "SELECT count(c0) FROM (SELECT id, c0 FROM {t} LIMIT 10) s",
        "SELECT count(c0) FROM (SELECT id, c0 FROM {t} ORDER BY id) s",
        "SELECT count(v) FROM (SELECT id, v FROM {t} ORDER BY id LIMIT 100 OFFSET 25) s",
        "SELECT count(v), sum(v), min(v) FROM (SELECT id, v FROM {t} LIMIT 20 OFFSET 5) s",
        "SELECT count(v) FROM (SELECT DISTINCT id, v FROM {t}) s",
        "SELECT count(c0), count(v), sum(v) FROM {t}",
        "SELECT v FROM (SELECT id, v FROM {t} ORDER BY id LIMIT 10 OFFSET 2) s",
    ] {
        assert_same(&conn, sql, "w9a_cn", "w9a_cnb")?;
    }
    if gpu() {
        assert!(routed_to_host(
            "SELECT count(v) FROM (SELECT id, v FROM w9a_cnb ORDER BY id) s"
        ));
        // Aggregates straight over the table, and of NULL-free columns, stay on the GPU.
        assert!(!routed_to_host(
            "SELECT count(c0), count(v), sum(v) FROM w9a_cnb"
        ));
        assert!(!routed_to_host(
            "SELECT count(id) FROM (SELECT id, v FROM w9a_cnb ORDER BY id LIMIT 10) s"
        ));
    }
    exec(&conn, "PRAGMA bolt_clear_tables")?;
    Ok(())
}

// SPDX-License-Identifier: Apache-2.0

//! Regression tests for the wave-2 `bolt_scan` / `bolt_query` work (W2-B): differential checks
//! against plain DuckDB for FLOAT literals, BETWEEN pre-filters, DATE-vs-TIMESTAMP rewrites,
//! Utf8 predicates (NULL guards, high-cardinality dictionaries), multi-batch and parallel
//! scans, and `bolt_aggregate` fragment validation. On a CUDA host the predicates are pushed
//! to the engine (checked through the scan trace); with `cuda-stub` DuckDB evaluates them and
//! the result assertions still hold.

use std::sync::{Arc, Mutex, Once};

use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use duckdb_bolt_core::{
    runtime::{BoltRuntime, RegisterMode},
    table_functions::last_bolt_scan_trace_for_testing,
};
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
// Serialize tests because Craton Bolt's GPU runtime catalog (and the scan trace) is
// process-wide.
static TEST_MUTEX: Mutex<()> = Mutex::new(());

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

fn gpu() -> bool {
    BoltRuntime::global().telemetry().unwrap().backend == "cuda-gpu"
}

fn query_i64(conn: &Connection, sql: &str) -> Result<i64> {
    let mut result = conn.query(sql, Parameters::None)?;
    let chunk = result.next_chunk()?.expect("Expected at least one chunk");
    let vec = chunk.get_vector_at::<i64>(0)?;
    Ok(*vec.reader()?.get(0)?.expect("Expected non-null i64"))
}

/// Every value of the first (BIGINT, non-NULL) column of `sql`, in result order.
fn query_ids(conn: &Connection, sql: &str) -> Result<Vec<i64>> {
    let mut result = conn.query(sql, Parameters::None)?;
    let mut out = Vec::new();
    while let Some(chunk) = result.next_chunk()? {
        let vec = chunk.get_vector_at::<i64>(0)?;
        let reader = vec.reader()?;
        for i in 0..chunk.row_count()? {
            out.push(*reader.get(i)?.expect("non-null id"));
        }
    }
    Ok(out)
}

/// Rows in the symmetric multiset difference (EXCEPT ALL both ways) of `SELECT * FROM a WHERE
/// pred` and the same over `b`: compares every column value, including NULLs.
fn row_diff(conn: &Connection, a: &str, b: &str, pred: &str) -> Result<i64> {
    query_i64(
        conn,
        &format!(
            "SELECT count(*) FROM (
                (SELECT * FROM {a} WHERE {pred} EXCEPT ALL SELECT * FROM {b} WHERE {pred})
                UNION ALL
                (SELECT * FROM {b} WHERE {pred} EXCEPT ALL SELECT * FROM {a} WHERE {pred}))"
        ),
    )
}

/// Count and an order-independent checksum of `id` for rows matching `pred`.
fn fingerprint(conn: &Connection, from: &str, pred: &str) -> Result<(i64, i64)> {
    let count = query_i64(conn, &format!("SELECT count(*) FROM {from} WHERE {pred}"))?;
    let sum = query_i64(
        conn,
        &format!("SELECT coalesce(sum(id), 0)::BIGINT FROM {from} WHERE {pred}"),
    )?;
    Ok((count, sum))
}

fn upload(conn: &Connection, local: &str, bolt_name: &str) -> Result<()> {
    conn.execute(
        format!("COPY {local} TO '{bolt_name}' (FORMAT bolt, USE_TMP_FILE FALSE)"),
        Parameters::None,
    )?;
    Ok(())
}

/// Assert `bolt_scan(bolt_name)` agrees with the DuckDB table `reference` for every predicate.
///
/// When `reference` holds NULLs, the runtime's crash guard runs the pushed SQL on the embedded
/// host DuckDB (docs/issues/W2-A-001), so every predicate is also checked against a NULL-free
/// copy, uploaded as `<bolt_name>_nn`, whose pushed SQL runs on the GPU engine
/// (docs/issues/W2-O-002: the pushed text must mean the same in both dialects).
fn assert_predicates(
    conn: &Connection,
    reference: &str,
    bolt_name: &str,
    predicates: &[&str],
) -> Result<()> {
    assert_predicates_on(conn, reference, bolt_name, predicates)?;
    let rows = query_i64(conn, &format!("SELECT count(*) FROM {reference}"))?;
    let complete = query_i64(
        conn,
        &format!("SELECT count(*) FROM {reference} WHERE COLUMNS(*) IS NOT NULL"),
    )?;
    if complete < rows {
        conn.execute(
            format!(
                "CREATE OR REPLACE TABLE {reference}_nn AS
                 SELECT * FROM {reference} WHERE COLUMNS(*) IS NOT NULL"
            ),
            Parameters::None,
        )?;
        conn.execute(
            format!(
                "COPY {reference}_nn TO '{bolt_name}_nn' (FORMAT bolt, USE_TMP_FILE FALSE, MODE 'replace')"
            ),
            Parameters::None,
        )?;
        assert_predicates_on(
            conn,
            &format!("{reference}_nn"),
            &format!("{bolt_name}_nn"),
            predicates,
        )?;
    }
    Ok(())
}

fn assert_predicates_on(
    conn: &Connection,
    reference: &str,
    bolt_name: &str,
    predicates: &[&str],
) -> Result<()> {
    let mut failures = Vec::new();
    for pred in predicates {
        let scan = format!("bolt_scan('{bolt_name}')");
        match fingerprint(conn, &scan, pred) {
            Ok(actual) => {
                let expected = fingerprint(conn, reference, pred)?;
                if actual != expected {
                    failures.push(format!(
                        "{pred}: bolt_scan {actual:?} != DuckDB {expected:?}"
                    ));
                }
                // Full row values (every column, NULLs included), not only counts.
                match row_diff(conn, &scan, reference, pred) {
                    Ok(0) => {}
                    Ok(n) => failures.push(format!("{pred}: {n} row(s) differ from DuckDB")),
                    Err(e) => failures.push(format!("{pred}: row comparison error: {e}")),
                }
            }
            Err(e) => failures.push(format!("{pred}: bolt_scan error: {e}")),
        }
    }
    assert!(
        failures.is_empty(),
        "{bolt_name}: {} predicate mismatch(es):\n{}",
        failures.len(),
        failures.join("\n")
    );
    Ok(())
}

/// Run `SELECT count(*)` over `bolt_scan` with `pred` and return the scan trace.
fn trace_of(
    conn: &Connection,
    bolt_name: &str,
    pred: &str,
) -> Result<duckdb_bolt_core::table_functions::BoltScanTrace> {
    query_i64(
        conn,
        &format!("SELECT count(*) FROM bolt_scan('{bolt_name}') WHERE {pred}"),
    )?;
    Ok(last_bolt_scan_trace_for_testing().expect("bolt_scan trace"))
}

#[test]
fn w2b_float_literal_pushdown_is_exact() -> Result<()> {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let (_env, conn) = setup_connection()?;
    conn.execute(
        "CREATE TABLE fl_src AS
         SELECT i::BIGINT AS id,
                CASE i % 12
                    WHEN 0 THEN 0.1::FLOAT
                    WHEN 1 THEN 0.3::FLOAT
                    WHEN 2 THEN 16777216::FLOAT
                    WHEN 3 THEN 16777217::FLOAT
                    WHEN 4 THEN (1.0 / 3.0)::FLOAT
                    WHEN 5 THEN -0.0::FLOAT
                    WHEN 6 THEN 'NaN'::FLOAT
                    WHEN 7 THEN NULL
                    WHEN 8 THEN 3.4028235e38::FLOAT
                    WHEN 9 THEN 1.0e-45::FLOAT
                    WHEN 10 THEN 0.30000001::FLOAT
                    ELSE 0.2::FLOAT END AS f
         FROM range(0, 1200) t(i)",
        Parameters::None,
    )?;
    upload(&conn, "fl_src", "w2b_fl")?;
    let predicates = [
        "f = 0.1::FLOAT",
        "f = 0.1",
        "f > 0.1::FLOAT",
        "f >= 0.1::FLOAT",
        "f < 0.1::FLOAT",
        "f = 0.3::FLOAT",
        "f != 0.3::FLOAT",
        "f <= 0.3::FLOAT",
        "f = 0.30000001::FLOAT",
        "f = 16777217::FLOAT",
        "f = 16777217",
        "f > 16777216::FLOAT",
        "f = (1.0 / 3.0)::FLOAT",
        "f = 0.0::FLOAT",
        "f = 3.4028235e38::FLOAT",
        "f > 1.0e-46::FLOAT",
        "f IN (0.1::FLOAT, 0.3::FLOAT, 16777216::FLOAT)",
        "f NOT IN (0.1::FLOAT, 0.2::FLOAT)",
        "NOT (f = 0.2::FLOAT)",
        "f > 0.2",
        "f = 0.2::DOUBLE",
    ];
    assert_predicates(&conn, "fl_src", "w2b_fl", &predicates)?;

    if gpu() {
        let trace = trace_of(&conn, "w2b_fl", "f = 0.1::FLOAT")?;
        assert_eq!(trace.path, "gpu", "{trace:?}");
        assert_eq!(trace.filters.len(), 1, "{trace:?}");
        assert!(trace.filters[0].1, "accepted: {trace:?}");
        // The literal is the exact f64 value of 0.1f32, in scientific notation so the host
        // fallback reads it as a DOUBLE too (W2-O-002).
        assert!(
            trace.filters[0].0.contains("1.0000000149011612e-1"),
            "{trace:?}"
        );
        let trace = trace_of(&conn, "w2b_fl", "f = 0.2::DOUBLE")?;
        assert!(trace.filters[0].0.contains("2e-1"), "{trace:?}");
    }
    Ok(())
}

#[test]
fn w2b_between_is_pushed_as_inclusive_prefilter() -> Result<()> {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let (_env, conn) = setup_connection()?;
    conn.execute(
        "CREATE TABLE bt_src AS
         SELECT i::BIGINT AS id,
                CASE WHEN i % 3 = 0 THEN NULL ELSE (i % 50)::INTEGER END AS n,
                CASE WHEN i % 7 = 0 THEN 'NaN'::DOUBLE ELSE i / 10.0 END AS d,
                CASE WHEN i % 5 = 0 THEN NULL ELSE 'k' || (i % 9)::VARCHAR END AS s
         FROM range(0, 3000) t(i)",
        Parameters::None,
    )?;
    upload(&conn, "bt_src", "w2b_bt")?;
    let predicates = [
        "id BETWEEN 10 AND 20",
        "id > 10 AND id < 20",
        "id >= 10 AND id < 20",
        "id > 10 AND id <= 20",
        "n BETWEEN 5 AND 9",
        "n > 5 AND n < 9",
        "id BETWEEN n AND 30",
        "d BETWEEN 1 AND 5",
        "d > 1 AND d < 5",
        "s BETWEEN 'k2' AND 'k5'",
        "(id BETWEEN 10 AND 20) OR (id BETWEEN 100 AND 105)",
        "(id BETWEEN 10 AND 2000) AND (n > 30 OR length(s) = 2)",
        "NOT (id BETWEEN 10 AND 2990)",
        "id BETWEEN NULL AND 5",
    ];
    assert_predicates(&conn, "bt_src", "w2b_bt", &predicates)?;

    if gpu() {
        let trace = trace_of(&conn, "w2b_bt", "id > 10 AND id < 20")?;
        assert_eq!(trace.path, "gpu", "{trace:?}");
        assert_eq!(trace.filters.len(), 1, "{trace:?}");
        let (sql, exact) = &trace.filters[0];
        assert!(!exact, "BETWEEN must stay a pre-filter: {trace:?}");
        assert!(sql.contains(">= 10") && sql.contains("<= 20"), "{trace:?}");

        // A partially translatable AND still pre-filters on its translatable part.
        let trace = trace_of(
            &conn,
            "w2b_bt",
            "(id BETWEEN 10 AND 2000) AND (n > 30 OR length(s) = 2)",
        )?;
        assert!(
            trace
                .filters
                .iter()
                .any(|(sql, exact)| !exact && sql.contains("<= 2000")),
            "{trace:?}"
        );
    }
    Ok(())
}

#[test]
fn w2b_date_column_vs_timestamp_constant() -> Result<()> {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let (_env, conn) = setup_connection()?;
    conn.execute(
        "CREATE TABLE dt_src AS
         SELECT i::BIGINT AS id,
                CASE WHEN i % 11 = 0 THEN NULL ELSE DATE '2020-12-25' + (i % 20)::INTEGER END AS dt
         FROM range(0, 500) t(i)",
        Parameters::None,
    )?;
    upload(&conn, "dt_src", "w2b_dt")?;
    let mut predicates = Vec::new();
    for op in ["=", "!=", "<", "<=", ">", ">="] {
        for ts in [
            "TIMESTAMP '2021-01-01 00:00:00'",
            "TIMESTAMP '2021-01-01 12:30:00'",
            "TIMESTAMP '2021-01-01 00:00:00.000001'",
            "TIMESTAMP '2020-12-31 23:59:59.999999'",
            "TIMESTAMP '1969-12-31 12:00:00'",
        ] {
            predicates.push(format!("dt {op} {ts}"));
            predicates.push(format!("{ts} {op} dt"));
        }
    }
    predicates.push("NOT (dt > TIMESTAMP '2021-01-03 06:00:00')".to_string());
    predicates.push("dt IN (DATE '2021-01-01', DATE '2021-01-02')".to_string());
    let refs: Vec<&str> = predicates.iter().map(String::as_str).collect();
    assert_predicates(&conn, "dt_src", "w2b_dt", &refs)?;

    if gpu() {
        // DuckDB folds `CAST(dt AS TIMESTAMP) op TIMESTAMP 'c'` into a DATE comparison
        // itself, so these reach the engine as plain Date32 comparisons (W1-A-006).
        for op in ["<", "<=", ">", ">="] {
            let pred = format!("dt {op} TIMESTAMP '2021-01-01 12:30:00'");
            let trace = trace_of(&conn, "w2b_dt", &pred)?;
            assert_eq!(trace.path, "gpu", "{pred}: {trace:?}");
            assert_eq!(trace.filters.len(), 1, "{pred}: {trace:?}");
            assert!(trace.filters[0].1, "{pred}: {trace:?}");
            assert!(
                trace.filters[0].0.contains("DATE '2021-01-0"),
                "{pred}: {trace:?}"
            );
        }
    }
    Ok(())
}

#[test]
fn w2b_utf8_null_guards_and_in_lists() -> Result<()> {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let (_env, conn) = setup_connection()?;
    conn.execute(
        "CREATE TABLE us_src AS
         SELECT i::BIGINT AS id,
                CASE WHEN i % 4 = 0 THEN NULL ELSE 'v' || (i % 6)::VARCHAR END AS s,
                'w' || (i % 3)::VARCHAR AS t
         FROM range(0, 3000) t(i)",
        Parameters::None,
    )?;
    upload(&conn, "us_src", "w2b_us")?;
    let predicates = [
        "s = 'v1'",
        "s != 'v1'",
        "NOT (s = 'v1')",
        "s > 'v3'",
        "s <= 'v2'",
        "s IS NULL",
        "s IS NOT NULL",
        "NOT (s IS NULL)",
        "NOT (s IS NOT NULL)",
        "s LIKE 'v%'",
        "s NOT LIKE 'v1%'",
        "s IN ('v1', 'v2', 'nope')",
        "s NOT IN ('v1', 'v2')",
        "s IN ('v1', NULL)",
        "s = t",
        "s < t",
        "t IN ('w0', 'w2')",
        "t != 'w1'",
        "(s = 'v1' OR t = 'w2') AND id < 2000",
    ];
    assert_predicates(&conn, "us_src", "w2b_us", &predicates)?;

    // A projected Utf8 column with NULLs: the result strings must match too.
    let expected = query_i64(
        &conn,
        "SELECT count(*) FROM (SELECT s, id FROM us_src WHERE s != 'v1' AND id > 100 ORDER BY id)
         WHERE s IS NOT NULL",
    )?;
    let actual = query_i64(
        &conn,
        "SELECT count(*) FROM (SELECT s, id FROM bolt_scan('w2b_us') WHERE s != 'v1' AND id > 100)
         WHERE s IS NOT NULL",
    )?;
    assert_eq!(actual, expected);
    let mismatches = query_i64(
        &conn,
        "SELECT count(*) FROM (SELECT id, s FROM bolt_scan('w2b_us') WHERE id % 7 = 3 AND s >= 'v2') b
         FULL OUTER JOIN (SELECT id, s FROM us_src WHERE id % 7 = 3 AND s >= 'v2') d USING (id)
         WHERE b.s IS DISTINCT FROM d.s",
    )?;
    assert_eq!(mismatches, 0);

    if gpu() {
        // No `IS NOT NULL` on Utf8 (unsupported by the engine): the guard is `LIKE '%'`, and
        // the query runs on the resident table instead of failing over to filter-take.
        let trace = trace_of(&conn, "w2b_us", "s != 'v1'")?;
        assert_eq!(trace.path, "gpu", "{trace:?}");
        assert!(trace.filters[0].1, "{trace:?}");
        assert!(trace.filters[0].0.contains("LIKE '%'"), "{trace:?}");
        assert!(!trace.filters[0].0.contains("IS NOT NULL"), "{trace:?}");
        // Positive predicates need no guard at all.
        let trace = trace_of(&conn, "w2b_us", "s = 'v1'")?;
        assert_eq!(trace.path, "gpu", "{trace:?}");
        assert_eq!(trace.filters[0].0, "(\"s\" = 'v1')", "{trace:?}");
        // Utf8 IN is spelled as equalities (the engine's Utf8 IN fails at run time).
        let trace = trace_of(&conn, "w2b_us", "t IN ('w0', 'w2')")?;
        assert_eq!(trace.path, "gpu", "{trace:?}");
        // Projecting a Utf8 column that holds NULLs goes straight to filter-take.
        query_i64(
            &conn,
            "SELECT count(s) FROM bolt_scan('w2b_us') WHERE id > 100",
        )?;
        let trace = last_bolt_scan_trace_for_testing().unwrap();
        assert_eq!(trace.path, "filter-take", "{trace:?}");
        // The scratch table never shows up in the catalog.
        let scratch = query_i64(
            &conn,
            "SELECT count(*) FROM bolt_tables() WHERE table_name LIKE '\\_\\_bolt%' ESCAPE '\\'",
        )?;
        assert_eq!(scratch, 0);
    }
    Ok(())
}

#[test]
fn w2b_high_cardinality_utf8_predicates_do_not_crash() -> Result<()> {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let (_env, conn) = setup_connection()?;
    // ~4,500 distinct strings: dictionary-lowered predicates would overflow the engine's stack.
    conn.execute(
        "CREATE TABLE hc_src AS
         SELECT i::BIGINT AS id,
                CASE WHEN i % 10 = 0 THEN NULL ELSE 'name_' || lpad(i::VARCHAR, 6, '0') END AS s
         FROM range(0, 5000) t(i)",
        Parameters::None,
    )?;
    upload(&conn, "hc_src", "w2b_hc")?;
    let predicates = [
        "s LIKE 'name_0001%'",
        "s > 'name_004'",
        "s != 'name_000001'",
        "s IS NOT NULL",
        "prefix(s, 'name_002')",
        "s = 'name_000123'",
        "s IN ('name_000001', 'name_000002')",
        "s NOT IN ('name_000001', 'name_000002')",
    ];
    assert_predicates(&conn, "hc_src", "w2b_hc", &predicates)?;

    if gpu() {
        // Equality is a single index comparison: still pushed.
        let trace = trace_of(&conn, "w2b_hc", "s = 'name_000123'")?;
        assert_eq!(trace.filters.len(), 1, "{trace:?}");
        // Dictionary-lowered predicates are not.
        for pred in ["s LIKE 'name_0001%'", "s > 'name_004'", "s IS NOT NULL"] {
            let trace = trace_of(&conn, "w2b_hc", pred)?;
            assert!(trace.filters.is_empty(), "{pred}: {trace:?}");
        }
    }

    // Long IN lists and wide ORs are left to DuckDB rather than sent to the engine.
    let items: Vec<String> = (0..200).map(|i| (i * 7).to_string()).collect();
    let in_list = format!("id IN ({})", items.join(", "));
    let ors: Vec<String> = (0..300)
        .map(|i| format!("(id = {} OR id < {})", i * 13, i % 3))
        .collect();
    let wide_or = ors.join(" OR ");
    assert_predicates(&conn, "hc_src", "w2b_hc", &[&in_list, &wide_or])?;
    Ok(())
}

fn register_batches(name: &str, batches: Vec<RecordBatch>) {
    let rt = BoltRuntime::global();
    let mut iter = batches.into_iter();
    rt.register_table(name, iter.next().unwrap(), RegisterMode::Replace)
        .unwrap();
    for batch in iter {
        rt.register_table(name, batch, RegisterMode::Append)
            .unwrap();
    }
}

fn id_label_batch(range: std::ops::Range<i64>) -> RecordBatch {
    let ids: Vec<i64> = range.collect();
    let labels: Vec<Option<String>> = ids
        .iter()
        .map(|i| (i % 5 != 0).then(|| format!("l{}", i % 7)))
        .collect();
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, true),
            Field::new("label", DataType::Utf8, true),
        ])),
        vec![
            Arc::new(Int64Array::from(ids)) as ArrayRef,
            Arc::new(StringArray::from(labels)),
        ],
    )
    .unwrap()
}

#[test]
fn w2b_multi_batch_scan_without_concat() -> Result<()> {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let (_env, conn) = setup_connection()?;
    // Odd batch sizes so chunks straddle batch boundaries.
    register_batches(
        "w2b_mb",
        vec![
            id_label_batch(0..3001),
            id_label_batch(3001..3002),
            id_label_batch(3002..3002),
            id_label_batch(3002..9000),
        ],
    );
    conn.execute(
        "CREATE TABLE mb_ref AS SELECT * FROM bolt_scan('w2b_mb')",
        Parameters::None,
    )?;
    assert_eq!(
        last_bolt_scan_trace_for_testing().unwrap().path,
        "unfiltered"
    );
    assert_eq!(query_i64(&conn, "SELECT count(*) FROM mb_ref")?, 9000);
    let ids = query_ids(&conn, "SELECT id FROM bolt_scan('w2b_mb')")?;
    assert_eq!(ids, (0..9000).collect::<Vec<_>>());

    assert_predicates(
        &conn,
        "mb_ref",
        "w2b_mb",
        &[
            "id > 2990 AND id < 3010",
            "id % 2 = 0",
            "label = 'l3'",
            "label != 'l3'",
            "label IS NULL",
        ],
    )?;
    // Filter-take across batches keeps insertion order and the Utf8 values.
    let ids = query_ids(
        &conn,
        "SELECT id FROM bolt_scan('w2b_mb') WHERE id >= 2995 AND label IS NOT NULL",
    )?;
    let expected = query_ids(
        &conn,
        "SELECT id FROM mb_ref WHERE id >= 2995 AND label IS NOT NULL ORDER BY id",
    )?;
    assert_eq!(ids, expected);
    let mismatches = query_i64(
        &conn,
        "SELECT count(*) FROM (SELECT id, label FROM bolt_scan('w2b_mb') WHERE id >= 2995) b
         JOIN mb_ref m USING (id) WHERE b.label IS DISTINCT FROM m.label",
    )?;
    assert_eq!(mismatches, 0);
    if gpu() {
        query_i64(
            &conn,
            "SELECT count(label) FROM bolt_scan('w2b_mb') WHERE id >= 2995",
        )?;
        assert_eq!(
            last_bolt_scan_trace_for_testing().unwrap().path,
            "filter-take"
        );
    }
    Ok(())
}

#[test]
fn w2b_parallel_scan_keeps_insertion_order() -> Result<()> {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let (_env, conn) = setup_connection()?;
    conn.execute("PRAGMA threads=8", Parameters::None)?;
    conn.execute(
        "CREATE TABLE par_src AS SELECT i::BIGINT AS id, (i % 97)::INTEGER AS k FROM range(0, 600000) t(i)",
        Parameters::None,
    )?;
    upload(&conn, "par_src", "w2b_par")?;
    let ids = query_ids(&conn, "SELECT id FROM bolt_scan('w2b_par')")?;
    assert_eq!(ids.len(), 600_000);
    assert!(ids.windows(2).all(|w| w[0] < w[1]), "scan order");
    let ids = query_ids(&conn, "SELECT id FROM bolt_scan('w2b_par') WHERE k = 5")?;
    assert!(ids.windows(2).all(|w| w[0] < w[1]), "filtered scan order");
    assert_eq!(
        ids.len() as i64,
        query_i64(&conn, "SELECT count(*) FROM par_src WHERE k = 5")?
    );
    conn.execute(
        "CREATE TABLE par_copy AS SELECT * FROM bolt_scan('w2b_par')",
        Parameters::None,
    )?;
    let ids = query_ids(&conn, "SELECT id FROM par_copy")?;
    assert!(ids.windows(2).all(|w| w[0] < w[1]), "CTAS insertion order");
    assert_eq!(
        query_i64(
            &conn,
            "SELECT sum(id * k)::BIGINT FROM bolt_scan('w2b_par')"
        )?,
        query_i64(&conn, "SELECT sum(id * k)::BIGINT FROM par_src")?
    );
    let ids = query_ids(
        &conn,
        "SELECT id FROM bolt_scan('w2b_par') LIMIT 5 OFFSET 300000",
    )?;
    assert_eq!(ids, (300_000..300_005).collect::<Vec<_>>());
    // bolt_query results are chunked the same way.
    let ids = query_ids(&conn, "SELECT id FROM bolt_query('SELECT id FROM w2b_par')")?;
    assert!(ids.windows(2).all(|w| w[0] < w[1]), "bolt_query order");
    Ok(())
}

#[test]
fn w2b_identifier_quoting_and_name_collisions() -> Result<()> {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let (_env, conn) = setup_connection()?;
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("a\"b", DataType::Int64, false),
            Field::new("ID", DataType::Int64, false),
        ])),
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 3, 4])) as ArrayRef,
            Arc::new(Int64Array::from(vec![10, 20, 30, 40])),
            Arc::new(Int64Array::from(vec![100, 200, 300, 400])),
        ],
    )
    .unwrap();
    BoltRuntime::global()
        .register_table("w2b \"quoted\"", batch, RegisterMode::Replace)
        .unwrap();
    let scan = "bolt_scan('w2b \"quoted\"')";
    assert_eq!(
        query_i64(
            &conn,
            &format!("SELECT sum(\"a\"\"b\")::BIGINT FROM {scan} WHERE \"a\"\"b\" > 15")
        )?,
        90
    );
    if gpu() {
        assert_eq!(last_bolt_scan_trace_for_testing().unwrap().path, "gpu");
    }
    // The case-colliding column is exposed as `ID_1`.
    assert_eq!(
        query_i64(
            &conn,
            &format!("SELECT sum(ID_1)::BIGINT FROM {scan} WHERE id >= 3")
        )?,
        700
    );
    Ok(())
}

#[test]
fn w2b_bolt_aggregate_rejects_fragment_injection() -> Result<()> {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let (_env, conn) = setup_connection()?;
    conn.execute(
        "CREATE TABLE ag_src AS SELECT i::BIGINT AS id, (i % 3)::BIGINT AS g FROM range(0, 30) t(i)",
        Parameters::None,
    )?;
    upload(&conn, "ag_src", "w2b_ag")?;
    for (grp, agg) in [
        ("", "count(*); SELECT 1"),
        ("", "sum(id) -- comment"),
        ("g /*", "*/ sum(id)"),
        ("g", "sum(id))"),
        ("", "'unterminated"),
    ] {
        let err = conn
            .query(
                format!(
                    "SELECT * FROM bolt_aggregate('w2b_ag', '{grp}', '{}')",
                    agg.replace('\'', "''")
                ),
                Parameters::None,
            )
            .err()
            .unwrap_or_else(|| panic!("accepted {grp:?} / {agg:?}"));
        assert!(err.to_string().contains("bolt_aggregate"), "{err}");
    }
    assert_eq!(
        query_i64(
            &conn,
            "SELECT * FROM bolt_aggregate('w2b_ag', '', 'count(*)')"
        )?,
        30
    );
    assert_eq!(
        query_i64(
            &conn,
            "SELECT count(*) FROM bolt_aggregate('w2b_ag', '  g  ', 'sum(id) AS \"s;--\"')"
        )?,
        3
    );
    Ok(())
}

#[test]
fn w2b_projected_nullable_columns_keep_nulls_after_gpu_filter() -> Result<()> {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let (_env, conn) = setup_connection()?;
    // No Utf8 column, so a filtered scan may run entirely on the engine: projected nullable
    // columns must still come back with their NULLs (docs/issues/W2-A-001).
    conn.execute(
        "CREATE TABLE pn_src AS
         SELECT i::BIGINT AS id,
                CASE WHEN i % 3 = 0 THEN NULL ELSE (i % 50)::INTEGER END AS n,
                CASE WHEN i % 4 = 1 THEN NULL ELSE i / 8.0 END AS d,
                CASE WHEN i % 5 = 2 THEN NULL ELSE TIMESTAMP '2024-01-01' + INTERVAL (i) MINUTE END AS ts,
                CASE WHEN i % 6 = 3 THEN NULL ELSE i % 2 = 0 END AS b
         FROM range(0, 4000) t(i)",
        Parameters::None,
    )?;
    upload(&conn, "pn_src", "w2b_pn")?;
    assert_predicates(
        &conn,
        "pn_src",
        "w2b_pn",
        &[
            "id > 20",
            "id % 10 < 5 AND id < 3000",
            "n > 10",
            "n IS NULL",
            "d < 100 OR id > 3900",
            "NOT (n = 7)",
            "b",
        ],
    )?;
    for sql in [
        "SELECT id, n FROM {t} WHERE id > 20",
        "SELECT n, d, ts FROM {t} WHERE id BETWEEN 100 AND 200",
        "SELECT id, b FROM {t} WHERE n > 10",
    ] {
        let a = sql.replace("{t}", "bolt_scan('w2b_pn')");
        let b = sql.replace("{t}", "pn_src");
        let diff = query_i64(
            &conn,
            &format!("SELECT count(*) FROM (({a} EXCEPT ALL {b}) UNION ALL ({b} EXCEPT ALL {a}))"),
        )?;
        assert_eq!(diff, 0, "{sql}");
        let nulls_bolt = query_i64(
            &conn,
            &format!(
                "SELECT count(*) - count({}) FROM ({a})",
                sql.split([' ', ',']).nth(1).unwrap()
            ),
        )?;
        let nulls_ref = query_i64(
            &conn,
            &format!(
                "SELECT count(*) - count({}) FROM ({b})",
                sql.split([' ', ',']).nth(1).unwrap()
            ),
        )?;
        assert_eq!(nulls_bolt, nulls_ref, "{sql}");
    }
    if gpu() {
        // Projected NULLs: row ids from the engine, values from the stored batches.
        query_i64(
            &conn,
            "SELECT count(n) FROM bolt_scan('w2b_pn') WHERE id > 20",
        )?;
        assert_eq!(
            last_bolt_scan_trace_for_testing().unwrap().path,
            "filter-take"
        );
        // No NULLs projected: the engine's own result is used.
        query_i64(
            &conn,
            "SELECT count(id) FROM bolt_scan('w2b_pn') WHERE n > 10",
        )?;
        assert_eq!(last_bolt_scan_trace_for_testing().unwrap().path, "gpu");
    }
    Ok(())
}

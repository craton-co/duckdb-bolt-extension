// SPDX-License-Identifier: Apache-2.0

//! Wave 9 (W9-C) regression tests: the host-fallback mirror of a table with widened integers is
//! copied once and serves queries with and without narrow arithmetic (W8-B-009), and the
//! per-table GPU notes survive a lazy engine drop (W8-A-001). They run on both backends and
//! branch on the reported backend where behavior legitimately differs.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, Once};

use arrow_array::{ArrayRef, Int32Array, Int64Array, RecordBatch, cast::AsArray, types::Int64Type};
use arrow_schema::{DataType, Field, Schema};
use duckdb_bolt_core::runtime::{BoltRuntime, RegisterMode};
use duckdb_loadable_macros::duckdb_entrypoint_c_api_v2;
use duckdb_neo::{
    Parameters, Result,
    connection::Connection,
    environment::{Environment, StorageLocation},
    extension::{Extension, register_static_extension},
};

#[duckdb_entrypoint_c_api_v2(extension_name = "bolt", api_version = "2.0.0")]
fn bolt_w9c_test_entrypoint(extension: &Extension) -> Result<()> {
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

fn query_single_i64(conn: &Connection, sql: &str) -> Result<i64> {
    let mut result = conn.query(sql, Parameters::None)?;
    let chunk = result.next_chunk()?.expect("Expected at least one chunk");
    let vec = chunk.get_vector_at::<i64>(0)?;
    Ok(*vec.reader()?.get(0)?.expect("Expected non-null i64"))
}

fn query_single_string(conn: &Connection, sql: &str) -> Result<String> {
    let mut result = conn.query(sql, Parameters::None)?;
    let chunk = result.next_chunk()?.expect("Expected at least one chunk");
    let vec = chunk.get_vector_at::<String>(0)?;
    Ok(vec
        .reader()?
        .get(0)?
        .map(|s| s.to_string())
        .expect("Expected non-null string"))
}

fn backend_is_gpu(conn: &Connection) -> Result<bool> {
    let backend = query_single_string(conn, "SELECT backend FROM bolt_pool_stats()")?;
    if cfg!(feature = "cuda-stub") {
        assert_eq!(
            backend, "host-planner-fallback",
            "cuda-stub builds cannot use the GPU"
        );
    }
    Ok(backend == "cuda-gpu")
}

fn int_batch(name: &str, values: impl IntoIterator<Item = i64>) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![Field::new(name, DataType::Int64, false)]));
    let col: ArrayRef = Arc::new(Int64Array::from_iter_values(values));
    RecordBatch::try_new(schema, vec![col]).unwrap()
}

/// The first row of a host-fallback query, every column displayed as text.
fn host_row(sql: &str) -> Result<Vec<String>> {
    let batch = BoltRuntime::global().execute_sql_host_fallback(sql)?.batch;
    assert!(batch.num_rows() > 0, "{sql}: no rows");
    Ok(batch
        .columns()
        .iter()
        .map(|c| arrow_cast::display::array_value_to_string(c, 0).unwrap())
        .collect())
}

/// DuckDB's OID of the fallback's copy of `table` (a new copy gets a new OID).
fn mirror_oid(table: &str) -> Result<i64> {
    let batch = BoltRuntime::global()
        .execute_sql_host_fallback(&format!(
            "SELECT table_oid FROM duckdb_tables() WHERE table_name = '{table}'"
        ))?
        .batch;
    assert_eq!(batch.num_rows(), 1, "one copy of {table} in the fallback");
    Ok(batch.column(0).as_primitive::<Int64Type>().value(0))
}

// ------------------------------------------------------------------------------ W8-B-009

/// Alternating queries with and without narrow arithmetic reuse one copy of the table: the
/// layout is switched by a view, and each kind of query still sees its types (W4-B-002 original
/// `SMALLINT` for arithmetic, W3-B-002 storage `INTEGER` / naive `TIMESTAMP` otherwise).
#[test]
fn fallback_mirror_is_not_copied_again_when_narrow_arithmetic_flips() -> Result<()> {
    let _guard = lock();
    let (_env, conn) = setup_connection()?;
    exec(
        &conn,
        "COPY (SELECT i::INTEGER AS k, i::SMALLINT AS s,
                      TIMESTAMPTZ '2024-01-01 00:00:00+00' AS tstz
               FROM range(10) r(i))
         TO 'w9c_mirror' (FORMAT bolt, USE_TMP_FILE FALSE)",
    )?;
    let storage = "SELECT typeof(s) AS s, typeof(tstz) AS t FROM w9c_mirror LIMIT 1";
    let original = "SELECT typeof(s) AS s, typeof(tstz) AS t, s + 0 AS z FROM w9c_mirror LIMIT 1";

    assert_eq!(host_row(storage)?[..2], ["INTEGER", "TIMESTAMP"]);
    // The first query wanting the original types copies the table once with them.
    assert_eq!(host_row(original)?[..2], ["SMALLINT", "TIMESTAMP"]);
    let oid = mirror_oid("w9c_mirror")?;
    for _ in 0..3 {
        assert_eq!(host_row(storage)?[..2], ["INTEGER", "TIMESTAMP"]);
        assert_eq!(
            mirror_oid("w9c_mirror")?,
            oid,
            "copied again for storage types"
        );
        assert_eq!(host_row(original)?[..2], ["SMALLINT", "TIMESTAMP"]);
        assert_eq!(
            mirror_oid("w9c_mirror")?,
            oid,
            "copied again for original types"
        );
    }
    // The columns are the same as without the view.
    let batch = BoltRuntime::global()
        .execute_sql_host_fallback("SELECT * FROM w9c_mirror ORDER BY k")?
        .batch;
    assert_eq!(batch.num_rows(), 10);
    assert_eq!(batch.num_columns(), 3);
    assert_eq!(batch.schema().field(1).name(), "s");
    assert_eq!(batch.column(1).data_type(), &DataType::Int32);

    // A new generation is copied again and still answers both kinds of query.
    exec(
        &conn,
        "COPY (SELECT i::INTEGER AS k, (i + 100)::SMALLINT AS s,
                      TIMESTAMPTZ '2024-01-01 00:00:00+00' AS tstz
               FROM range(10) r(i))
         TO 'w9c_mirror' (FORMAT bolt, USE_TMP_FILE FALSE, MODE 'replace')",
    )?;
    assert_eq!(
        host_row("SELECT min(s) AS s FROM w9c_mirror")?,
        ["100".to_string()]
    );
    assert_eq!(
        host_row("SELECT min(s + 0) AS s FROM w9c_mirror")?,
        ["100".to_string()]
    );
    exec(&conn, "PRAGMA bolt_clear_tables")?;
    Ok(())
}

/// When the stored values do not fit the original type (metadata that does not describe the
/// data), the storage-typed copy is kept and not attempted again with the original types.
#[test]
fn fallback_mirror_with_values_outside_the_original_type_is_copied_once() -> Result<()> {
    let _guard = lock();
    let (_env, _conn) = setup_connection()?;
    let field = Field::new("s", DataType::Int32, true).with_metadata(HashMap::from([(
        "duckdb.logical_type".to_string(),
        "SMALLINT".to_string(),
    )]));
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![field])),
        vec![Arc::new(Int32Array::from(vec![100_000, 1])) as ArrayRef],
    )
    .unwrap();
    BoltRuntime::global().register_table("w9c_nofit", batch, RegisterMode::Replace)?;

    let arithmetic = "SELECT max(s + 0) AS m FROM w9c_nofit";
    assert_eq!(host_row(arithmetic)?, ["100000".to_string()]);
    let oid = mirror_oid("w9c_nofit")?;
    for _ in 0..2 {
        assert_eq!(
            host_row("SELECT typeof(s) AS t FROM w9c_nofit LIMIT 1")?,
            ["INTEGER".to_string()]
        );
        assert_eq!(host_row(arithmetic)?, ["100000".to_string()]);
        assert_eq!(mirror_oid("w9c_nofit")?, oid);
    }
    BoltRuntime::global().drop_table("w9c_nofit")?;
    Ok(())
}

// ------------------------------------------------------------------------------ W8-A-001

/// After a lazy engine drop (a case-differing re-registration over a tombstone), the per-table
/// notes and `fallback_reason` still say which tables the engine keeps on the host.
#[test]
fn gpu_notes_survive_a_lazy_engine_drop() -> Result<()> {
    let _guard = lock();
    let (_env, conn) = setup_connection()?;
    exec(&conn, "RESET bolt_memory_budget")?;
    if !backend_is_gpu(&conn)? {
        // No engine, no per-table notes: the reason is engine-wide.
        BoltRuntime::global().register_table(
            "w9c_big",
            int_batch("v", 0..10_000),
            RegisterMode::Replace,
        )?;
        assert_eq!(
            query_single_i64(
                &conn,
                "SELECT count(*) FROM bolt_tables() WHERE on_gpu IS DISTINCT FROM false \
                 OR gpu_note IS NOT NULL"
            )?,
            0
        );
        exec(&conn, "PRAGMA bolt_clear_tables")?;
        return Ok(());
    }

    let check_note = |when: &str| -> Result<()> {
        assert_eq!(
            query_single_i64(
                &conn,
                "SELECT count(*) FROM bolt_tables() WHERE table_name = 'w9c_big' \
                 AND on_gpu = false AND gpu_note LIKE '%bolt_memory_budget%'"
            )?,
            1,
            "{when}"
        );
        let reason = query_single_string(&conn, "SELECT fallback_reason FROM bolt_pool_stats()")?;
        assert!(
            reason.contains("table 'w9c_big' not on GPU"),
            "{when}: {reason}"
        );
        Ok(())
    };
    // Room for the small tables, not for `w9c_big` (80 KB of Int64).
    exec(&conn, "SET bolt_memory_budget = 8192")?;
    let result = (|| -> Result<()> {
        let rt = BoltRuntime::global();
        rt.register_table("w9c_big", int_batch("v", 0..10_000), RegisterMode::Replace)?;
        rt.register_table("w9c_lazy", int_batch("v", 0..4), RegisterMode::Replace)?;
        check_note("before the drop")?;
        // Dropping leaves a tombstone; registering the name in another case drops the engine
        // for a lazy rebuild.
        assert!(rt.drop_table("w9c_lazy")?);
        rt.register_table("W9C_LAZY", int_batch("v", 0..4), RegisterMode::Replace)?;
        check_note("after the lazy drop")?;
        assert_eq!(
            query_single_i64(
                &conn,
                "SELECT count(*) FROM bolt_query('SELECT v FROM W9C_LAZY')"
            )?,
            4
        );
        check_note("after the rebuild")?;
        Ok(())
    })();
    exec(&conn, "RESET bolt_memory_budget")?;
    exec(&conn, "PRAGMA bolt_clear_tables")?;
    result
}

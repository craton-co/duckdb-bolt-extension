// SPDX-License-Identifier: Apache-2.0

//! The embedded host-fallback DuckDB instance ([`FallbackDb`]).

use super::*;

/// Long-lived, locked-down DuckDB instance used to execute queries without a GPU.
///
/// Catalog tables are mirrored lazily (only those a query may reference) and re-mirrored when
/// their generation changes.
#[cfg(not(target_arch = "wasm32"))]
pub(super) struct FallbackDb {
    // Field order matters: the connection must drop before its instance and environment.
    pub(super) conn: duckdb_neo::connection::Connection,
    pub(super) _instance: duckdb_neo::instance::Instance,
    pub(super) _env: duckdb_neo::environment::Environment,
    /// Catalog key -> its mirror (see [`FallbackDb::mirror_table`]).
    mirrored: HashMap<String, Mirror>,
}

/// One mirrored catalog table.
#[cfg(not(target_arch = "wasm32"))]
struct Mirror {
    /// Catalog generation that was copied.
    generation: u64,
    /// Arrow bytes of the catalog table (counted against [`FallbackDb::MIRROR_CACHE_BYTES`]).
    bytes: usize,
    /// `Some` when the copy has the original types of the widened integer columns: it is
    /// [`FallbackDb::BASE_SCHEMA`]`.<key>` and `<key>` is a view over it (W8-B-009).
    /// `None`: `<key>` is a table with the storage types.
    view: Option<MirrorView>,
    /// Whether a copy with the original types was attempted for this generation (it fails
    /// when the values do not fit the original types), so it is attempted at most once.
    original_tried: bool,
}

/// The view `<key>` over a mirror that has the original integer types.
#[cfg(not(target_arch = "wasm32"))]
struct MirrorView {
    /// Select list casting the widened integer columns back to their storage types.
    storage_select: String,
    /// Whether the view currently shows the original types (`SELECT *`).
    shows_original: bool,
}

#[cfg(not(target_arch = "wasm32"))]
impl FallbackDb {
    pub(super) const MIRROR_SOURCE: &'static str = "__bolt_mirror_source";

    /// Schema of the mirrors that have the original integer types (W8-B-009). Catalog tables
    /// are mirrored into the default schema (`main`), so their names cannot collide with it.
    pub(super) const BASE_SCHEMA: &'static str = "__bolt_mirror_base";

    /// Startup options: no file/network access, no extension loading, configuration locked.
    pub(super) const OPTIONS: [(&'static str, &'static str); 5] = [
        ("enable_external_access", "false"),
        ("autoload_known_extensions", "false"),
        ("autoinstall_known_extensions", "false"),
        ("allow_community_extensions", "false"),
        ("lock_configuration", "true"),
    ];

    /// Upper bound on the fallback instance's worker threads.
    pub(super) const MAX_THREADS: usize = 4;

    /// Fallback instance memory limit. It runs inside the host DuckDB process, which by
    /// default already claims 80% of RAM, so the fallback gets a smaller share and spills or
    /// errors instead of competing for the same memory (W1-B-003).
    pub(super) const MEMORY_LIMIT: &'static str = "25%";

    /// Arrow bytes of mirrored tables kept between queries (W1-B-003). A mirror is a second,
    /// DuckDB-owned copy of a catalog table; beyond this many bytes the mirrors the current
    /// query does not reference are dropped, so the long-lived extra copy is bounded by
    /// `max(MIRROR_CACHE_BYTES, tables of the current query)`.
    pub(super) const MIRROR_CACHE_BYTES: usize = 256 << 20;

    /// Rows copied into a mirror per `INSERT` (see [`mirror_insert_groups`]).
    pub(super) const MIRROR_INSERT_ROWS: usize = 1 << 20;

    pub(super) fn mirrored_bytes(&self) -> usize {
        self.mirrored.values().map(|m| m.bytes).sum()
    }

    /// Resource options: half the cores (1..=[`Self::MAX_THREADS`]) and
    /// [`Self::MEMORY_LIMIT`] of RAM, on top of DuckDB's defaults of all cores / 80% RAM that
    /// the hosting DuckDB already uses (W1-B-003).
    pub(super) fn resource_options() -> [(&'static str, String); 2] {
        let cores = thread::available_parallelism().map_or(1, |n| n.get());
        let threads = (cores / 2).clamp(1, Self::MAX_THREADS);
        [
            ("threads", threads.to_string()),
            ("memory_limit", Self::MEMORY_LIMIT.to_string()),
        ]
    }

    pub(super) fn open() -> Result<Self> {
        use duckdb_neo::{
            environment::{Environment, StorageLocation},
            instance::{AttachBuilder, InstanceBuilder},
        };
        let env = Environment::new()?;
        // `lock_configuration` comes last so the resource options are applied first.
        let instance = InstanceBuilder::new()
            .options(Self::resource_options())
            .options(Self::OPTIONS)
            .build(&env)?;
        instance.attach(AttachBuilder::new(StorageLocation::InMemory).make_default())?;
        let conn = instance.connect()?;
        Ok(Self {
            conn,
            _instance: instance,
            _env: env,
            mirrored: HashMap::new(),
        })
    }

    /// Parse `sql` and require exactly one `SELECT` statement.
    pub(super) fn parse_select(
        &self,
        sql: &str,
    ) -> Result<duckdb_neo::statement::Statement<'static>> {
        use duckdb_neo::query_result::StatementType;
        let mut statements = self.conn.parse(sql)?;
        let stmt = statements
            .next()
            .ok_or_else(|| Error::invalid_input("Craton Bolt SQL query cannot be empty"))??;
        if statements.next().is_some() {
            return Err(Error::invalid_input(
                "Craton Bolt host fallback accepts a single SQL statement",
            ));
        }
        if !matches!(stmt.get_type()?, StatementType::Select) {
            return Err(Error::invalid_input(
                "Craton Bolt host fallback only executes SELECT queries",
            ));
        }
        Ok(stmt)
    }

    /// The quoted name of the original-typed copy behind the view `key`.
    fn base_table(key: &str) -> String {
        format!("{}.{}", quote_ident(Self::BASE_SCHEMA), quote_ident(key))
    }

    pub(super) fn drop_mirror(&mut self, key: &str) -> Result<()> {
        use duckdb_neo::Parameters;
        if let Some(mirror) = self.mirrored.remove(key) {
            if mirror.view.is_some() {
                self.conn.execute(
                    format!("DROP VIEW IF EXISTS {}", quote_ident(key)),
                    Parameters::None,
                )?;
                self.conn.execute(
                    format!("DROP TABLE IF EXISTS {}", Self::base_table(key)),
                    Parameters::None,
                )?;
            } else {
                self.conn.execute(
                    format!("DROP TABLE IF EXISTS {}", quote_ident(key)),
                    Parameters::None,
                )?;
            }
        }
        Ok(())
    }

    /// Best-effort cleanup after a failed mirror of `key`: forget it and drop whatever objects
    /// the attempt may have left (errors are ignored: DuckDB refuses `DROP VIEW` of a table
    /// and `DROP TABLE` of a view even with `IF EXISTS`).
    fn discard_mirror(&mut self, key: &str) {
        self.mirrored.remove(key);
        for sql in [
            format!("DROP VIEW IF EXISTS {}", quote_ident(key)),
            format!("DROP TABLE IF EXISTS {}", quote_ident(key)),
            format!("DROP TABLE IF EXISTS {}", Self::base_table(key)),
        ] {
            let _ = self.conn.execute(sql, duckdb_neo::Parameters::None);
        }
    }

    pub(super) fn drop_all_mirrors(&mut self) -> Result<()> {
        let keys: Vec<String> = self.mirrored.keys().cloned().collect();
        for key in keys {
            self.drop_mirror(&key)?;
        }
        Ok(())
    }

    /// Drop stale mirrors and mirror every catalog table `sql` may reference.
    ///
    /// `original_integers` selects how widened integer columns are shown (see
    /// [`FallbackDb::mirror_table`]). A referenced mirror that has the original types is
    /// switched by re-pointing its view, which copies nothing; a storage-typed mirror of a
    /// table with widened integers is rebuilt once with the original types when a query first
    /// wants them (W8-B-009), so alternating queries do not copy the table again.
    ///
    /// Only tables whose name occurs as a whole word are mirrored (W2-E-007), so a table that
    /// cannot be mirrored no longer breaks unrelated queries (W2-E-008). A query that does name
    /// it fails with the mirror error: running it anyway could let the fallback instance's own
    /// copy of the bolt replacement scan (static builds, W3-E-004) call back into the worker.
    pub(super) fn sync(
        &mut self,
        catalog: &BTreeMap<String, RegisteredTable>,
        sql: &str,
        original_integers: bool,
    ) -> Result<()> {
        let names = SqlNames::new(sql);
        let stale: Vec<String> = self
            .mirrored
            .iter()
            .filter(|(key, mirror)| match catalog.get(key.as_str()) {
                Some(t) => {
                    t.generation != mirror.generation
                        || (original_integers
                            && !mirror.original_tried
                            && names.may_reference(key)
                            && has_widened_integers(t))
                }
                None => true,
            })
            .map(|(key, _)| key.clone())
            .collect();
        for key in stale {
            self.drop_mirror(&key)?;
        }
        if names.may_call_bolt(|w| catalog.keys().any(|k| k.eq_ignore_ascii_case(w))) {
            return Err(Error::invalid_input(REENTRANT_CALL_MESSAGE));
        }
        let repoint: Vec<String> = self
            .mirrored
            .iter()
            .filter(|(key, mirror)| {
                names.may_reference(key)
                    && mirror
                        .view
                        .as_ref()
                        .is_some_and(|view| view.shows_original != original_integers)
            })
            .map(|(key, _)| key.clone())
            .collect();
        for key in repoint {
            if self.point_view(&key, original_integers).is_err() {
                // Copied again below.
                self.drop_mirror(&key)?;
            }
        }
        for (key, entry) in catalog {
            if self.mirrored.contains_key(key) || !names.may_reference(key) {
                continue;
            }
            if let Err(e) = self.mirror_table(key, entry, original_integers) {
                self.discard_mirror(key);
                return Err(Error::invalid_input(format!(
                    "Bolt table '{key}' cannot be copied into the host fallback: {e}"
                )));
            }
        }
        if self.mirrored_bytes() > Self::MIRROR_CACHE_BYTES {
            let unused: Vec<String> = self
                .mirrored
                .keys()
                .filter(|key| !names.may_reference(key))
                .cloned()
                .collect();
            for key in unused {
                self.drop_mirror(&key)?;
                if self.mirrored_bytes() <= Self::MIRROR_CACHE_BYTES {
                    break;
                }
            }
        }
        Ok(())
    }

    /// Copy catalog table `key` into the fallback instance.
    ///
    /// Every column uses its storage type (W3-B-002: a `TIMESTAMPTZ` is the naive UTC
    /// `TIMESTAMP` it is stored as), except with `original_integers` for a table with integer
    /// columns stored widened (RFC-008 `duckdb.logical_type` metadata: `TINYINT`, `SMALLINT`,
    /// unsigned, `HUGEINT`). Those are copied with their original types into
    /// [`Self::BASE_SCHEMA`], so arithmetic on them overflows exactly as in DuckDB (W4-B-002,
    /// chosen per query by [`plan_has_narrow_arithmetic`]), and `key` becomes a view over the
    /// copy: `SELECT *` for such queries, the widened columns cast back to their storage types
    /// for the others ([`Self::sync`] re-points it, W8-B-009). The one copy serves both kinds
    /// of query, and it is not larger than a storage-typed one. Results are cast back to the
    /// storage types by [`FallbackDb::query`]. If the stored values do not fit the original
    /// types (metadata that does not describe the data), the storage types are used, and the
    /// original types are not attempted again for this generation.
    pub(super) fn mirror_table(
        &mut self,
        key: &str,
        entry: &RegisteredTable,
        original_integers: bool,
    ) -> Result<()> {
        let wanted = original_integers && has_widened_integers(entry);
        if wanted {
            if self.mirror_table_as(key, entry, true).is_ok() {
                return Ok(());
            }
            self.discard_mirror(key);
        }
        self.mirror_table_as(key, entry, false)?;
        if let Some(mirror) = self.mirrored.get_mut(key) {
            mirror.original_tried = wanted;
        }
        Ok(())
    }

    fn mirror_table_as(
        &mut self,
        key: &str,
        entry: &RegisteredTable,
        original_integers: bool,
    ) -> Result<()> {
        use duckdb_neo::Parameters;
        let table = if original_integers {
            self.conn.execute(
                format!(
                    "CREATE SCHEMA IF NOT EXISTS {}",
                    quote_ident(Self::BASE_SCHEMA)
                ),
                Parameters::None,
            )?;
            Self::base_table(key)
        } else {
            quote_ident(key)
        };
        // Copied in slices of at most `MIRROR_INSERT_ROWS` rows, so the transient DuckDB copy
        // (the attached collection) is bounded by the slice, not the table (W9-C).
        let mut view = None;
        for (i, group) in mirror_insert_groups(&entry.batches, Self::MIRROR_INSERT_ROWS)
            .iter()
            .enumerate()
        {
            let collection = crate::arrow_bridge::record_batches_to_collection(
                &self.conn,
                &entry.schema,
                group,
            )?;
            if i == 0 {
                let mut columns = Vec::with_capacity(entry.schema.fields().len());
                let mut storage_select = Vec::with_capacity(entry.schema.fields().len());
                for (field, ty) in entry.schema.fields().iter().zip(collection.logical_types()) {
                    let name = quote_ident(field.name());
                    let storage_text = ty.to_text()?;
                    match widened_integer_original(field) {
                        Some(original) if original_integers => {
                            columns.push(format!("{name} {}", original.to_metadata_text()));
                            storage_select
                                .push(format!("CAST({name} AS {storage_text}) AS {name}"));
                        }
                        _ => {
                            columns.push(format!("{name} {storage_text}"));
                            storage_select.push(name);
                        }
                    }
                }
                self.conn.execute(
                    format!("CREATE OR REPLACE TABLE {table} ({})", columns.join(", ")),
                    Parameters::None,
                )?;
                view = original_integers.then(|| MirrorView {
                    storage_select: storage_select.join(", "),
                    shows_original: false,
                });
            }
            let mut statements = self.conn.parse(format!(
                "INSERT INTO {table} SELECT * FROM {}",
                Self::MIRROR_SOURCE
            ))?;
            let stmt = statements
                .next()
                .ok_or_else(|| Error::api_error("Failed to parse host-fallback mirror insert"))??
                .add_collection(collection, Self::MIRROR_SOURCE)?;
            self.conn.execute(stmt, Parameters::None)?;
        }
        let has_view = view.is_some();
        self.mirrored.insert(
            key.to_string(),
            Mirror {
                generation: entry.generation,
                bytes: entry.bytes_in_memory,
                view,
                original_tried: original_integers,
            },
        );
        if has_view {
            self.point_view(key, true)?;
        }
        Ok(())
    }

    /// (Re)create the view `key` over its original-typed copy: `SELECT *` when
    /// `show_original`, else the widened columns cast back to their storage types. A no-op
    /// for a storage-typed mirror.
    fn point_view(&mut self, key: &str, show_original: bool) -> Result<()> {
        let Some(view) = self.mirrored.get(key).and_then(|m| m.view.as_ref()) else {
            return Ok(());
        };
        let select = if show_original {
            "*".to_string()
        } else {
            view.storage_select.clone()
        };
        self.conn.execute(
            format!(
                "CREATE OR REPLACE VIEW {} AS SELECT {select} FROM {}",
                quote_ident(key),
                Self::base_table(key)
            ),
            duckdb_neo::Parameters::None,
        )?;
        if let Some(view) = self.mirrored.get_mut(key).and_then(|m| m.view.as_mut()) {
            view.shows_original = show_original;
        }
        Ok(())
    }

    /// Run the single `SELECT` `sql` and return its schema and rows in Bolt storage types.
    ///
    /// The result is read through DuckDB's Arrow C stream (columnar copies) instead of value by
    /// value (W1-C-005), kept in the stream's batches (no concatenation, W4-A-002; every batch
    /// has the returned schema, and an empty result has no batches) and cast, batch by batch,
    /// column by column and with overflow checks, to
    /// [`crate::arrow_bridge::duckdb_type_to_bolt_arrow_type`] of the bound result types, so
    /// `bolt_query` sees the same types as before (`SMALLINT` -> `Int32`, `HUGEINT` ->
    /// `Int64`, `TIMESTAMPTZ` -> naive UTC `Timestamp`, ...). An unsupported result type fails
    /// at bind time, before the query runs.
    ///
    /// `original_integers`: see [`FallbackDb::sync`]. `exact`: return DuckDB's result types
    /// exactly instead ([`crate::arrow_bridge::duckdb_type_to_query_arrow_type`]: `HUGEINT` as
    /// `Decimal128(38, 0)`, unsigned as unsigned, `INTERVAL` as `Interval(MonthDayNano)`), for
    /// `bolt_query`, which binds DuckDB's types (W9-A-001, W9-A-002).
    pub(super) fn query(
        &mut self,
        catalog: &BTreeMap<String, RegisteredTable>,
        sql: &str,
        original_integers: bool,
        exact: bool,
    ) -> Result<(SchemaRef, Vec<RecordBatch>)> {
        let stmt = self.parse_select(sql)?;
        self.sync(catalog, sql, original_integers)?;
        let fields = stmt
            .bind(&self.conn)?
            .schema
            .get_all()?
            .into_iter()
            .map(|(name, logical_type)| {
                let dt = if exact {
                    crate::arrow_bridge::duckdb_type_to_query_arrow_type(&logical_type)?
                } else {
                    crate::arrow_bridge::duckdb_type_to_bolt_arrow_type(&logical_type)?
                };
                Ok(arrow_schema::Field::new(name, dt, true))
            })
            .collect::<Result<Vec<_>>>()?;
        drop(stmt);
        let target: SchemaRef = Arc::new(Schema::new(fields));
        let (_, batches) = crate::arrow_bridge::arrow_query_to_record_batches(&self.conn, sql)?;
        let batches = batches
            .into_iter()
            .map(|batch| crate::arrow_bridge::cast_batch_to_schema_checked(&batch, &target))
            .collect::<Result<Vec<_>>>()?;
        Ok((target, batches))
    }

    pub(super) fn bind(
        &mut self,
        catalog: &BTreeMap<String, RegisteredTable>,
        sql: &str,
    ) -> Result<()> {
        let stmt = self.parse_select(sql)?;
        self.sync(catalog, sql, false)?;
        stmt.bind(&self.conn)?;
        Ok(())
    }

    /// DuckDB's `EXPLAIN` of the single `SELECT` `sql` (W1-B-007).
    pub(super) fn explain(
        &mut self,
        catalog: &BTreeMap<String, RegisteredTable>,
        sql: &str,
    ) -> Result<String> {
        // Validate the statement first: only one SELECT may be wrapped in EXPLAIN.
        drop(self.parse_select(sql)?);
        self.sync(catalog, sql, false)?;
        let mut result = self
            .conn
            .query(format!("EXPLAIN {sql}"), duckdb_neo::Parameters::None)?;
        let batch = crate::arrow_bridge::query_result_to_record_batch(&mut result)?;
        // Columns `explain_key`, `explain_value`: the plan text is the last one.
        let mut out = String::new();
        if let Some(text) = batch
            .columns()
            .last()
            .and_then(|c| c.as_any().downcast_ref::<arrow_array::StringArray>())
        {
            for value in text.iter().flatten() {
                out.push_str(value);
                out.push('\n');
            }
        }
        Ok(out)
    }
}

/// `batches` split into groups of at most `max_rows` (> 0) rows, by zero-copy slices; always
/// at least one (possibly empty) group, so the mirror table is created even without rows.
#[cfg(not(target_arch = "wasm32"))]
fn mirror_insert_groups(batches: &[RecordBatch], max_rows: usize) -> Vec<Vec<RecordBatch>> {
    let max_rows = max_rows.max(1);
    let mut groups = vec![Vec::new()];
    let mut rows = 0usize;
    for batch in batches {
        let mut offset = 0;
        while offset < batch.num_rows() {
            if rows == max_rows {
                groups.push(Vec::new());
                rows = 0;
            }
            let len = (batch.num_rows() - offset).min(max_rows - rows);
            groups
                .last_mut()
                .expect("groups is never empty")
                .push(batch.slice(offset, len));
            rows += len;
            offset += len;
        }
    }
    groups
}

/// Whether `entry` has an integer column stored widened (W4-B-002).
#[cfg(not(target_arch = "wasm32"))]
fn has_widened_integers(entry: &RegisteredTable) -> bool {
    entry
        .schema
        .fields()
        .iter()
        .any(|f| widened_integer_original(f).is_some())
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use arrow_array::cast::AsArray;

    fn batch(rows: i32) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![arrow_schema::Field::new(
            "v",
            ArrowDataType::Int32,
            false,
        )]));
        let col: ArrayRef = Arc::new(arrow_array::Int32Array::from_iter_values(0..rows));
        RecordBatch::try_new(schema, vec![col]).unwrap()
    }

    fn group_rows(groups: &[Vec<RecordBatch>]) -> Vec<usize> {
        groups
            .iter()
            .map(|g| g.iter().map(RecordBatch::num_rows).sum())
            .collect()
    }

    #[test]
    fn mirror_insert_groups_split_by_rows() {
        assert_eq!(group_rows(&mirror_insert_groups(&[], 4)), [0]);
        assert_eq!(group_rows(&mirror_insert_groups(&[batch(0)], 4)), [0]);
        assert_eq!(group_rows(&mirror_insert_groups(&[batch(4)], 4)), [4]);
        assert_eq!(
            group_rows(&mirror_insert_groups(&[batch(3), batch(6), batch(1)], 4)),
            [4, 4, 2]
        );
        // Slices keep the values in order.
        let groups = mirror_insert_groups(&[batch(10)], 4);
        let second = groups[1][0]
            .column(0)
            .as_primitive::<arrow_array::types::Int32Type>();
        assert_eq!(&second.values()[..], &[4, 5, 6, 7]);
    }
}

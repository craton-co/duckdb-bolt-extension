// SPDX-License-Identifier: Apache-2.0

//! DuckDB result types for `bolt_query` (W9-A-001).
//!
//! `bolt_query` used to bind its result columns with the Bolt planner's types, which are not
//! DuckDB's: `INTEGER / INTEGER` is an integer, `SUM(BIGINT)` a BIGINT, `SUM(FLOAT)` a FLOAT,
//! and every widened integer column (TINYINT, unsigned, HUGEINT) comes back as its storage
//! type. The [`TypeBinder`] asks DuckDB itself: it keeps a separate, empty in-memory DuckDB
//! with one zero-row table per catalog table, declared with the columns' original DuckDB types
//! (RFC-008 metadata), and binds the query there. Binding reads no data, so this is cheap and
//! does not copy tables (unlike the host-fallback mirrors).

use super::*;
use crate::arrow_bridge::{OriginalDuckDbType, original_duckdb_type};

/// The DuckDB type of one result column of a query.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum BoundType {
    /// A type Craton Bolt can hold.
    Original(OriginalDuckDbType),
    /// `INTERVAL` (W9-A-002), returned as Arrow `Interval(MonthDayNano)`.
    Interval,
    /// Anything else (LIST, STRUCT, BLOB, UUID, ...): `bolt_query` cannot return it.
    Unsupported,
}

#[cfg(not(target_arch = "wasm32"))]
impl BoundType {
    pub(super) fn of(logical_type: &duckdb_neo::logical_type::LogicalType) -> Result<Self> {
        if matches!(
            logical_type.id()?,
            duckdb_neo::logical_type::LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_INTERVAL
        ) {
            return Ok(Self::Interval);
        }
        Ok(match OriginalDuckDbType::from_logical_type(logical_type)? {
            Some(original) => Self::Original(original),
            None => Self::Unsupported,
        })
    }

    /// The Arrow field `bolt_query` binds for a column of this type, `None` when unsupported.
    pub(super) fn field(&self, name: &str) -> Option<arrow_schema::Field> {
        match self {
            Self::Original(original) => Some(crate::arrow_bridge::exact_field(name, *original)),
            Self::Interval => Some(arrow_schema::Field::new(
                name,
                ArrowDataType::Interval(arrow_schema::IntervalUnit::MonthDayNano),
                true,
            )),
            Self::Unsupported => None,
        }
    }
}

/// An empty DuckDB instance with zero-row copies of the catalog's schemas, for binding.
#[cfg(not(target_arch = "wasm32"))]
pub(super) struct TypeBinder {
    // Field order matters: the connection must drop before its instance and environment.
    conn: duckdb_neo::connection::Connection,
    _instance: duckdb_neo::instance::Instance,
    _env: duckdb_neo::environment::Environment,
    /// Catalog key -> the schema its stub table was created from.
    stubs: HashMap<String, SchemaRef>,
}

#[cfg(not(target_arch = "wasm32"))]
impl TypeBinder {
    pub(super) fn open() -> Result<Self> {
        use duckdb_neo::{
            environment::{Environment, StorageLocation},
            instance::{AttachBuilder, InstanceBuilder},
        };
        let env = Environment::new()?;
        let instance = InstanceBuilder::new()
            .options([
                ("threads", "1".to_string()),
                ("memory_limit", "64MB".to_string()),
            ])
            .options(FallbackDb::OPTIONS)
            .build(&env)?;
        instance.attach(AttachBuilder::new(StorageLocation::InMemory).make_default())?;
        let conn = instance.connect()?;
        Ok(Self {
            conn,
            _instance: instance,
            _env: env,
            stubs: HashMap::new(),
        })
    }

    /// The DuckDB column type of a stored Bolt field: its original type (RFC-008), except that
    /// zoned timestamps keep their storage type, the naive UTC `TIMESTAMP` the engine and the
    /// host fallback compute on (W3-B-002); `None` when there is no DuckDB equivalent.
    fn column_type(field: &arrow_schema::Field) -> Option<String> {
        use OriginalDuckDbType as O;
        let original = match original_duckdb_type(field) {
            Some(O::TimestampTz) => O::Timestamp,
            Some(O::TimestampTzNs) => O::TimestampNs,
            Some(original) => original,
            None => match field.data_type() {
                ArrowDataType::Boolean => O::Boolean,
                ArrowDataType::Int32 => O::Integer,
                ArrowDataType::Int64 => O::BigInt,
                ArrowDataType::Float32 => O::Float,
                ArrowDataType::Float64 => O::Double,
                ArrowDataType::Utf8 => O::Varchar,
                ArrowDataType::Date32 => O::Date,
                ArrowDataType::Timestamp(unit, _) => match unit {
                    ArrowTimeUnit::Second => O::TimestampSec,
                    ArrowTimeUnit::Millisecond => O::TimestampMs,
                    ArrowTimeUnit::Microsecond => O::Timestamp,
                    ArrowTimeUnit::Nanosecond => O::TimestampNs,
                },
                ArrowDataType::Decimal128(p, s) if *s >= 0 => O::Decimal {
                    width: *p,
                    scale: *s as u8,
                },
                _ => return None,
            },
        };
        Some(original.to_metadata_text())
    }

    /// Create (or re-create) the stub of every catalog table `sql` may reference whose schema
    /// changed, and drop the stubs of tables that are gone.
    fn sync(
        &mut self,
        catalog: &BTreeMap<String, RegisteredTable>,
        names: &SqlNames,
    ) -> Result<()> {
        use duckdb_neo::Parameters;
        let gone: Vec<String> = self
            .stubs
            .keys()
            .filter(|key| !catalog.contains_key(key.as_str()))
            .cloned()
            .collect();
        for key in gone {
            self.conn.execute(
                format!("DROP TABLE IF EXISTS {}", quote_ident(&key)),
                Parameters::None,
            )?;
            self.stubs.remove(&key);
        }
        for (key, entry) in catalog {
            if !names.may_reference(key)
                || self
                    .stubs
                    .get(key)
                    .is_some_and(|s| Arc::ptr_eq(s, &entry.schema))
            {
                continue;
            }
            let columns = entry
                .schema
                .fields()
                .iter()
                .map(|f| {
                    let ty = Self::column_type(f).ok_or_else(|| {
                        Error::invalid_input(format!(
                            "column '{}' has no DuckDB type ({:?})",
                            f.name(),
                            f.data_type()
                        ))
                    })?;
                    Ok(format!("{} {ty}", quote_ident(f.name())))
                })
                .collect::<Result<Vec<_>>>()?;
            self.stubs.remove(key);
            self.conn.execute(
                format!(
                    "CREATE OR REPLACE TABLE {} ({})",
                    quote_ident(key),
                    columns.join(", ")
                ),
                Parameters::None,
            )?;
            self.stubs.insert(key.clone(), entry.schema.clone());
        }
        Ok(())
    }

    /// DuckDB's result column names and types for the single `SELECT` `sql` over the catalog.
    pub(super) fn describe(
        &mut self,
        catalog: &BTreeMap<String, RegisteredTable>,
        sql: &str,
    ) -> Result<Vec<(String, BoundType)>> {
        use duckdb_neo::query_result::StatementType;
        let names = SqlNames::new(sql);
        // The instance has no Bolt functions, but a static build may still resolve `bolt_*`
        // calls in it; refuse them like the host fallback does (W3-E-004).
        if names.may_call_bolt(|w| catalog.keys().any(|k| k.eq_ignore_ascii_case(w))) {
            return Err(Error::invalid_input(REENTRANT_CALL_MESSAGE));
        }
        self.sync(catalog, &names)?;
        let mut statements = self.conn.parse(sql)?;
        let stmt = statements
            .next()
            .ok_or_else(|| Error::invalid_input("Craton Bolt SQL query cannot be empty"))??;
        if statements.next().is_some() || !matches!(stmt.get_type()?, StatementType::Select) {
            return Err(Error::invalid_input(
                "bolt_query result types: not a single SELECT",
            ));
        }
        stmt.bind(&self.conn)?
            .schema
            .get_all()?
            .into_iter()
            .map(|(name, logical_type)| Ok((name, BoundType::of(&logical_type)?)))
            .collect()
    }
}

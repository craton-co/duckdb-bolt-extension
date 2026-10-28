// SPDX-License-Identifier: Apache-2.0

//! DuckDB custom configuration options (`SET bolt_* = ...`) for Craton Bolt.
//!
//! All options configure the single, process-wide [`BoltRuntime`], so they are
//! registered as GLOBAL-only settings: `SET SESSION bolt_*` is rejected instead of
//! pretending to be connection-local. The runtime is shared by every DuckDB database
//! in the process, so even a GLOBAL setting of one database affects the others (see
//! `docs/issues/W1-D-003-option-scope-process-global.md`); `bolt_pool_stats()` and
//! `bolt_version()` report what the runtime actually uses.
//!
//! Keeping `current_setting('bolt_*')` truthful (`docs/issues/W1-E-006-*`):
//!
//! * `SET` applies the value to the runtime inside the update callback; if the runtime
//!   refuses it (validation or worker error), the `SET` fails and the setting keeps its
//!   previous value. The callback returns the normalized value (for example the trimmed
//!   `bolt_ptx_cache_dir`), so the setting shows what the runtime stores.
//! * When a database loads the extension, each option's registered default is the
//!   runtime's *current* value, not the compile-time default. A second database opened in
//!   the same process therefore shows what the shared runtime uses (DuckDB does not run
//!   the update callback for defaults, so a constant default would lie).
//! * `RESET bolt_*` always restores the documented default (`0`, `0`, `''`, `false`) in
//!   both the runtime and the setting, whatever default was registered at load time.
//! * `SET bolt_* = NULL` is rejected.
//!
//! Setting an option to the value the runtime already has is a no-op, so it does not
//! force an engine rebuild.
//!
//! Settings that bypass the update callback (`docs/issues/W2-D-004-*`, `W1-D-003-*`):
//! [`sync_with_runtime`] runs at the start of every Bolt function call and reconciles the
//! calling database's settings with the runtime:
//!
//! * a `bolt_*` value given in the database's **startup options** (before the extension was
//!   loaded) is copied into the setting by DuckDB without running the callback. It is
//!   recognised (a raw `VARCHAR` for a typed option, or a value the extension never stored)
//!   and applied *through* the callback, so it is validated exactly like `SET GLOBAL`. An
//!   invalid startup value fails every Bolt call until it is `SET` or `RESET`;
//! * a value the extension stored earlier that no longer matches the runtime (another
//!   database in the process changed it) is refreshed to the runtime's value, so
//!   `current_setting()` is truthful after the first Bolt call;
//! * the active `bolt_ptx_cache_dir` is re-validated (use-time check, W2-D-003): if it is no
//!   longer safe, the cache is disabled and the call fails once with the reason.

use std::{
    cell::Cell,
    collections::HashSet,
    path::{Path, PathBuf},
    sync::{Mutex, OnceLock},
    time::{Duration, Instant},
};

use duckdb_neo::{
    DuckDBType, Error, FromValue, Result, ToValue,
    connection::{Context, SettingScope},
    custom_option::{CustomOptionBuilder, CustomOptionCallbacks},
    extension::Extension,
    logical_type::LogicalTypeID,
    value::Value,
};

use crate::runtime::{BoltConfig, BoltRuntime};
use crate::sandbox::{FileAccessPolicy, permission_error};

/// Documented defaults: what `RESET bolt_*` restores.
const DEFAULT_DEVICE: i32 = 0;
const DEFAULT_MEMORY_BUDGET: i64 = 0;
const DEFAULT_PTX_CACHE_DIR: &str = "";
const DEFAULT_ENABLE_TRACING: bool = false;

/// Suffix of every option description.
const PROCESS_WIDE: &str = "Process-wide: shared by every connection and every DuckDB database in \
                            this process; GLOBAL scope only; bolt_pool_stats() shows the active value";

/// The four `bolt_*` options.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum OptionKind {
    Device,
    MemoryBudget,
    PtxCacheDir,
    EnableTracing,
}

impl OptionKind {
    const ALL: [Self; 4] = [
        Self::Device,
        Self::MemoryBudget,
        Self::PtxCacheDir,
        Self::EnableTracing,
    ];

    fn name(self) -> &'static str {
        match self {
            Self::Device => "bolt_device",
            Self::MemoryBudget => "bolt_memory_budget",
            Self::PtxCacheDir => "bolt_ptx_cache_dir",
            Self::EnableTracing => "bolt_enable_tracing",
        }
    }

    /// The runtime's value in the text form the setting shows.
    fn runtime_text(self, cfg: &BoltConfig) -> String {
        let shown = LoadTimeDefaults::from_config(cfg);
        match self {
            Self::Device => shown.device.to_string(),
            Self::MemoryBudget => shown.memory_budget.to_string(),
            Self::PtxCacheDir => shown.ptx_cache_dir,
            Self::EnableTracing => shown.enable_tracing.to_string(),
        }
    }

    /// The runtime's value as a typed DuckDB value for this option.
    fn runtime_value(self, context: &Context, cfg: &BoltConfig) -> Result<Value> {
        let shown = LoadTimeDefaults::from_config(cfg);
        match self {
            Self::Device => shown.device.value(context),
            Self::MemoryBudget => shown.memory_budget.value(context),
            Self::PtxCacheDir => shown.ptx_cache_dir.value(context),
            Self::EnableTracing => shown.enable_tracing.value(context),
        }
    }

    /// Whether the setting text `shown` describes what the runtime uses.
    fn matches_runtime(self, shown: &str, cfg: &BoltConfig) -> bool {
        match self {
            // Compare as paths, so `/a/` and `/a` (equal `PathBuf`s) agree, as in `apply`.
            Self::PtxCacheDir => {
                let wanted = (!shown.is_empty()).then(|| PathBuf::from(shown));
                wanted == cfg.ptx_cache_dir
            }
            _ => shown == self.runtime_text(cfg),
        }
    }
}

/// Every `(option, value text)` the extension itself stored in some database's setting
/// (registered defaults, accepted `SET`/`RESET` values, refreshes). A setting value outside
/// this set never went through the extension, so it came from a database's startup options.
fn known_values() -> &'static Mutex<HashSet<(OptionKind, String)>> {
    static KNOWN: OnceLock<Mutex<HashSet<(OptionKind, String)>>> = OnceLock::new();
    KNOWN.get_or_init(|| Mutex::new(HashSet::new()))
}

fn remember(kind: OptionKind, text: impl Into<String>) {
    known_values()
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .insert((kind, text.into()));
}

fn is_known(kind: OptionKind, text: &str) -> bool {
    known_values()
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .contains(&(kind, text.to_string()))
}

thread_local! {
    /// Set while [`sync_with_runtime`] writes the runtime's own value back into a setting:
    /// the callbacks then accept the value as-is (it is already applied and validated).
    static REFRESHING: Cell<bool> = const { Cell::new(false) };
}

fn refreshing() -> bool {
    REFRESHING.with(Cell::get)
}

/// Marks the current thread as refreshing until dropped (also on error paths).
struct RefreshGuard;

impl RefreshGuard {
    fn new() -> Self {
        REFRESHING.with(|r| r.set(true));
        Self
    }
}

impl Drop for RefreshGuard {
    fn drop(&mut self) {
        REFRESHING.with(|r| r.set(false));
    }
}

/// How a database's setting relates to the runtime (see [`sync_with_runtime`]).
#[derive(Debug, Clone, PartialEq, Eq)]
enum SettingState {
    /// The setting shows what the runtime uses.
    Current,
    /// The extension stored it, but the runtime has changed since (another database).
    Stale,
    /// It never went through the extension: a startup option copied by DuckDB.
    Startup(String),
}

/// The setting's value as text, and whether it is a raw `VARCHAR` copy for a typed option.
/// `None` for NULL or a value of an unexpected type (left alone).
fn observe(kind: OptionKind, value: &Value) -> Option<(String, bool)> {
    if value.is_null().ok()? {
        return None;
    }
    let is_varchar =
        value.logical_type().ok()?.id().ok()? == LogicalTypeID::DUCKDB_V2_LOGICAL_TYPE_ID_VARCHAR;
    if kind == OptionKind::PtxCacheDir || is_varchar {
        let text = String::from_value(value).ok()??;
        let raw = kind != OptionKind::PtxCacheDir;
        let text = if raw { text } else { text.trim().to_string() };
        return Some((text, raw));
    }
    let text = match kind {
        OptionKind::Device => i32::from_value(value).ok()??.to_string(),
        OptionKind::MemoryBudget => i64::from_value(value).ok()??.to_string(),
        OptionKind::EnableTracing => bool::from_value(value).ok()??.to_string(),
        OptionKind::PtxCacheDir => unreachable!("handled above"),
    };
    Some((text, false))
}

/// Classify a setting (pure; unit-tested).
fn classify(kind: OptionKind, text: &str, raw_varchar: bool, cfg: &BoltConfig) -> SettingState {
    if raw_varchar {
        // Values the extension stores are always typed; DuckDB copies startup options as
        // the text they were given in.
        return SettingState::Startup(text.to_string());
    }
    if kind.matches_runtime(text, cfg) {
        SettingState::Current
    } else if is_known(kind, text) {
        SettingState::Stale
    } else {
        SettingState::Startup(text.to_string())
    }
}

/// Serializes [`sync_with_runtime`] across connections and worker threads.
fn sync_lock() -> &'static Mutex<()> {
    static LOCK: Mutex<()> = Mutex::new(());
    &LOCK
}

/// Reconcile the calling database's `bolt_*` settings with the process-wide runtime and
/// re-validate the active PTX cache directory; see the module documentation.
///
/// Call it at the start of every Bolt function (scalar `exec`, table-function `bind`, COPY
/// `bind`, pragma). It runs once per data chunk and worker thread, so the common case is
/// lock-free and touches no setting (W4-C-002): four setting reads and one runtime-config
/// snapshot decide that every setting already shows the runtime's value, and the PTX
/// directory costs one `stat` while its fingerprint is unchanged (see
/// [`ptx_check_is_fresh`]). Only a setting that disagrees with the runtime (a startup value,
/// or a stale one after another database changed the runtime) takes the serialized slow path,
/// which writes settings with `Config::set_value`.
///
/// # Errors
/// A startup value the callback refuses (the message names the option and how to fix it),
/// or a PTX cache directory that failed its use-time check (the cache is disabled first).
pub fn sync_with_runtime(context: &Context) -> Result<()> {
    let runtime = BoltRuntime::global();
    let cfg = runtime.config()?;
    if !settings_match_runtime(context, &cfg)? {
        reconcile_settings(context)?;
        return recheck_ptx_cache_dir(context, runtime.config()?.ptx_cache_dir);
    }
    recheck_ptx_cache_dir(context, cfg.ptx_cache_dir)
}

/// Whether every readable `bolt_*` setting of the calling database already shows what the
/// runtime uses (classifies as [`SettingState::Current`]); pure reads, no lock.
fn settings_match_runtime(context: &Context, cfg: &BoltConfig) -> Result<bool> {
    let config = context.config()?;
    for kind in OptionKind::ALL {
        let Some((value, _scope)) = config.try_get_value(kind.name())? else {
            continue;
        };
        let Some((text, raw)) = observe(kind, &value) else {
            continue;
        };
        if raw || !kind.matches_runtime(&text, cfg) {
            return Ok(false);
        }
    }
    Ok(true)
}

/// The slow path of [`sync_with_runtime`]: apply startup values through the callback and
/// refresh stale settings, serialized across connections and worker threads.
fn reconcile_settings(context: &Context) -> Result<()> {
    let _serialized = sync_lock().lock().unwrap_or_else(|p| p.into_inner());
    let runtime = BoltRuntime::global();
    let config = context.config()?;
    for kind in OptionKind::ALL {
        let Some((value, _scope)) = config.try_get_value(kind.name())? else {
            continue;
        };
        let Some((text, raw)) = observe(kind, &value) else {
            continue;
        };
        let cfg = runtime.config()?;
        match classify(kind, &text, raw, &cfg) {
            SettingState::Current => {}
            SettingState::Startup(text) => {
                // Through the callback: validated and applied like `SET GLOBAL`.
                config
                    .set_value(kind.name(), &value, SettingScope::Global)
                    .map_err(|e| startup_value_error(kind, &text, e))?;
            }
            SettingState::Stale => refresh_setting(context, kind, &cfg),
        }
    }
    Ok(())
}

fn startup_value_error(kind: OptionKind, text: &str, err: Error) -> Error {
    let name = kind.name();
    Error {
        code: err.code,
        message: format!(
            "{name} = '{text}' was given in this database's startup options (before the bolt \
             extension was loaded) and could not be applied: {}. Use SET GLOBAL {name} = ... or \
             RESET {name}",
            err.message
        ),
    }
}

/// Best effort: show the runtime's value in this database's setting.
fn refresh_setting(context: &Context, kind: OptionKind, cfg: &BoltConfig) {
    let Ok(value) = kind.runtime_value(context, cfg) else {
        return;
    };
    let Ok(config) = context.config() else {
        return;
    };
    let _refreshing = RefreshGuard::new();
    if config
        .set_value(kind.name(), &value, SettingScope::Global)
        .is_ok()
    {
        remember(kind, kind.runtime_text(cfg));
    }
}

/// A full validation of the PTX cache directory is repeated at least this often, even while
/// its fingerprint is unchanged (catches permission changes on its parents, which do not
/// change the directory's own metadata).
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
const PTX_FULL_RECHECK_INTERVAL: Duration = Duration::from_secs(5);

/// The last successful full use-time check of a PTX cache directory.
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
#[derive(Debug, Clone, PartialEq, Eq)]
struct PtxCheck {
    dir: PathBuf,
    fingerprint: DirFingerprint,
    at: Instant,
}

#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
fn last_ptx_check() -> &'static Mutex<Option<PtxCheck>> {
    static LAST: Mutex<Option<PtxCheck>> = Mutex::new(None);
    &LAST
}

/// What identifies a directory and its access rules, from one `stat` (following symlinks):
/// device, inode, mode, owner and change time on Unix (`chmod`, `chown`, `rename` over it and
/// replacement all change one of them); attributes and creation / write times on Windows.
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
#[derive(Debug, Clone, PartialEq, Eq)]
struct DirFingerprint(Vec<u64>);

#[cfg(unix)]
fn dir_fingerprint(dir: &Path) -> Option<DirFingerprint> {
    use std::os::unix::fs::MetadataExt;
    let m = std::fs::metadata(dir).ok()?;
    Some(DirFingerprint(vec![
        m.dev(),
        m.ino(),
        u64::from(m.mode()),
        u64::from(m.uid()),
        m.ctime() as u64,
        m.ctime_nsec() as u64,
        u64::from(m.is_dir()),
    ]))
}

#[cfg(windows)]
fn dir_fingerprint(dir: &Path) -> Option<DirFingerprint> {
    use std::os::windows::fs::MetadataExt;
    let m = std::fs::metadata(dir).ok()?;
    Some(DirFingerprint(vec![
        u64::from(m.file_attributes()),
        m.creation_time(),
        m.last_write_time(),
        u64::from(m.is_dir()),
    ]))
}

/// No usable metadata: every use runs the full check.
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
#[cfg(not(any(unix, windows)))]
fn dir_fingerprint(_dir: &Path) -> Option<DirFingerprint> {
    None
}

/// Whether the last full check of `dir` still holds: same directory, unchanged fingerprint,
/// and younger than [`PTX_FULL_RECHECK_INTERVAL`] (pure; unit-tested).
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
fn ptx_check_is_fresh(
    last: Option<&PtxCheck>,
    dir: &Path,
    fingerprint: Option<&DirFingerprint>,
    now: Instant,
) -> bool {
    match (last, fingerprint) {
        (Some(last), Some(fp)) => {
            last.dir == dir
                && &last.fingerprint == fp
                && now.saturating_duration_since(last.at) < PTX_FULL_RECHECK_INTERVAL
        }
        _ => false,
    }
}

/// Use-time check of the active PTX cache directory `dir` (W2-D-003 item 2).
///
/// The full validation (`validate_ptx_cache_dir`: canonicalization, ownership and permissions
/// of every parent) runs when the directory or its fingerprint changed since the last
/// successful check, or when that check is older than [`PTX_FULL_RECHECK_INTERVAL`];
/// otherwise this is one `stat`. The fingerprint is taken *before* the full check, so a change
/// during the check is seen by the next call.
#[cfg(not(target_arch = "wasm32"))]
fn recheck_ptx_cache_dir(context: &Context, dir: Option<PathBuf>) -> Result<()> {
    let Some(dir) = dir else {
        return Ok(());
    };
    let fingerprint = dir_fingerprint(&dir);
    let now = Instant::now();
    {
        let last = last_ptx_check().lock().unwrap_or_else(|p| p.into_inner());
        if ptx_check_is_fresh(last.as_ref(), &dir, fingerprint.as_ref(), now) {
            return Ok(());
        }
    }
    let err = match check_ptx_cache_dir_in_use(&dir) {
        Ok(()) => {
            if let Some(fingerprint) = fingerprint {
                *last_ptx_check().lock().unwrap_or_else(|p| p.into_inner()) = Some(PtxCheck {
                    dir,
                    fingerprint,
                    at: now,
                });
            }
            return Ok(());
        }
        Err(err) => err,
    };
    *last_ptx_check().lock().unwrap_or_else(|p| p.into_inner()) = None;
    let runtime = BoltRuntime::global();
    runtime.update_config(|cfg| {
        if cfg.ptx_cache_dir.as_ref() == Some(&dir) {
            cfg.ptx_cache_dir = None;
        }
    })?;
    refresh_setting(context, OptionKind::PtxCacheDir, &runtime.config()?);
    Err(Error {
        code: err.code,
        message: format!(
            "the persistent PTX cache was disabled (bolt_ptx_cache_dir = ''), because '{}' \
             failed its use-time safety check: {}",
            dir.display(),
            err.message
        ),
    })
}

#[cfg(target_arch = "wasm32")]
fn recheck_ptx_cache_dir(_context: &Context, _dir: Option<PathBuf>) -> Result<()> {
    Ok(())
}

/// Re-validate a PTX cache directory the runtime is about to use (or uses).
///
/// Applies the same rules as `SET bolt_ptx_cache_dir` (except the `enable_external_access`
/// check, which belongs to the database that set it). The engine-build path in `runtime.rs`
/// calls it before passing the directory to craton-bolt and builds without the cache when it
/// fails; `recheck_ptx_cache_dir` calls it before Bolt work that may use the cache.
pub fn check_ptx_cache_dir_in_use(dir: &Path) -> Result<()> {
    validate_ptx_cache_dir(&dir.to_string_lossy()).map(|_| ())
}

fn null_not_allowed(name: &str) -> Error {
    Error::invalid_input(format!(
        "{name} cannot be NULL; use RESET {name} to restore the default"
    ))
}

/// Refuse session-scoped writes (DuckDB already does, because the options are registered
/// with `no_session()`; this is the second line of defense with a clearer message).
fn ensure_global(name: &str, scope: SettingScope) -> Result<()> {
    if scope == SettingScope::Global {
        Ok(())
    } else {
        Err(Error::invalid_input(format!(
            "{name} configures the process-wide Craton Bolt runtime, which is shared by every \
             connection and every database in this process, so it can only be set globally \
             (SET GLOBAL {name} = ...)"
        )))
    }
}

/// The value an update applies: the documented default on RESET, else the non-NULL value.
fn requested<T: FromValue>(name: &str, value: &Value, is_reset: bool, default: T) -> Result<T> {
    if is_reset {
        return Ok(default);
    }
    T::from_value(value)?.ok_or_else(|| null_not_allowed(name))
}

/// Apply `update` to the runtime config unless `unchanged` says it already holds.
///
/// Errors (for example a terminated worker) propagate, so the `SET` fails and DuckDB keeps
/// the previous setting value.
fn apply(
    unchanged: impl FnOnce(&BoltConfig) -> bool,
    update: impl FnOnce(&mut BoltConfig),
) -> Result<()> {
    let runtime = BoltRuntime::global();
    if unchanged(&runtime.config()?) {
        return Ok(());
    }
    runtime.update_config(update)
}

/// Convert a validated, non-negative `bolt_memory_budget` to the runtime representation.
///
/// `0` means uncapped. Values above `usize::MAX` (32-bit targets) are rejected rather
/// than truncated.
pub(crate) fn memory_budget_from_bytes(bytes: i64) -> Result<Option<usize>> {
    if bytes < 0 {
        return Err(Error::invalid_input(
            "bolt_memory_budget must be >= 0 (0 means uncapped)",
        ));
    }
    if bytes == 0 {
        return Ok(None);
    }
    usize::try_from(bytes).map(Some).map_err(|_| {
        Error::invalid_input(format!(
            "bolt_memory_budget {bytes} exceeds the addressable memory of this platform ({} bytes)",
            usize::MAX
        ))
    })
}

/// Inverse of [`memory_budget_from_bytes`], for showing the runtime value as a setting.
fn memory_budget_to_bytes(budget: Option<usize>) -> i64 {
    budget.map_or(0, |b| i64::try_from(b).unwrap_or(i64::MAX))
}

/// Validate a `bolt_ptx_cache_dir` value. An empty (or blank) value disables the cache.
///
/// Rules (the directory is never created here; the engine creates it lazily):
/// * the path must be absolute: a relative path would resolve against whatever the
///   process working directory is when the engine is (re)built;
/// * an existing path must be a directory;
/// * on Unix, the directory (or, if it does not exist yet, its nearest existing ancestor,
///   because whoever can write there can create it first) must be owned by the effective
///   user or root and must not be group- or world-writable. Every directory above it must
///   be owned by the effective user or root and must not be world-writable unless it has
///   the sticky bit (like `/tmp`), so another user cannot rename or replace it. When the
///   effective user id cannot be determined, only root-owned directories are accepted;
/// * on Windows, ACLs cannot be read without FFI, so the canonical directory (or its nearest
///   existing ancestor) must lie inside the user's own profile (`%USERPROFILE%`,
///   `%LOCALAPPDATA%`, `%APPDATA%`), whose default ACL grants write access only to the user,
///   SYSTEM and Administrators. Locations such as `C:\ptx` (where `Authenticated Users` may
///   modify new subdirectories) or `%PUBLIC%` are refused.
pub(crate) fn validate_ptx_cache_dir(raw: &str) -> Result<Option<PathBuf>> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    let path = PathBuf::from(trimmed);
    if !path.is_absolute() {
        return Err(Error::invalid_input(format!(
            "bolt_ptx_cache_dir must be an absolute path, got '{trimmed}'"
        )));
    }
    // W9-D: a `..` after a component that does not exist yet hides where the directory will
    // end up. On Unix the kernel resolves `missing/..` only once `missing` exists, so the
    // nearest existing ancestor checked below (`/home/u` for `/home/u/new/../../../tmp/x`) is
    // not where the engine's `create_dir_all` lands (`/tmp/x`, which another user may own).
    if path
        .components()
        .any(|c| matches!(c, std::path::Component::ParentDir))
    {
        return Err(Error::invalid_input(format!(
            "bolt_ptx_cache_dir must not contain '..' components, got '{trimmed}'"
        )));
    }
    let (existing, meta) = match std::fs::metadata(&path) {
        Ok(meta) => {
            if !meta.is_dir() {
                return Err(Error::invalid_input(format!(
                    "bolt_ptx_cache_dir '{trimmed}' exists and is not a directory"
                )));
            }
            (path.clone(), meta)
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            match nearest_existing_ancestor(&path) {
                Some((ancestor, meta)) if meta.is_dir() => (ancestor, meta),
                Some((ancestor, _)) => {
                    return Err(Error::invalid_input(format!(
                        "bolt_ptx_cache_dir '{trimmed}' cannot be created: '{}' is not a directory",
                        ancestor.display()
                    )));
                }
                // Nothing on the path exists (not even its root): the engine will fail to
                // create it, and there is nothing to protect.
                None => return Ok(Some(path)),
            }
        }
        Err(e) => {
            return Err(Error::invalid_input(format!(
                "bolt_ptx_cache_dir '{trimmed}' cannot be inspected: {e}"
            )));
        }
    };
    check_private_dir(&existing, &meta, trimmed)?;
    check_ancestors(&existing, trimmed)?;
    Ok(Some(path))
}

fn nearest_existing_ancestor(path: &Path) -> Option<(PathBuf, std::fs::Metadata)> {
    path.ancestors()
        .skip(1)
        .find_map(|a| std::fs::metadata(a).ok().map(|m| (a.to_path_buf(), m)))
}

/// The effective user id of this process, if it can be determined without `libc`.
///
/// Linux / Android: the second field of `Uid:` in `/proc/self/status` (correct for
/// set-uid processes too). Elsewhere: the owner of a file created in the temporary
/// directory, which is the effective user id on every Unix. Determined once per process.
#[cfg(unix)]
fn effective_uid() -> Option<u32> {
    static EUID: std::sync::OnceLock<Option<u32>> = std::sync::OnceLock::new();
    *EUID.get_or_init(|| {
        proc_status_euid()
            .or_else(probe_file_uid)
            .or_else(proc_self_owner)
    })
}

/// Owner of `/proc/self` (the effective uid of a dumpable process), where `/proc` exists
/// but `status` could not be parsed.
#[cfg(unix)]
fn proc_self_owner() -> Option<u32> {
    use std::os::unix::fs::MetadataExt;
    std::fs::metadata("/proc/self").ok().map(|m| m.uid())
}

#[cfg(unix)]
fn proc_status_euid() -> Option<u32> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    parse_status_euid(&status)
}

/// Parse the effective uid from `/proc/<pid>/status` text (`Uid: real effective saved fs`).
#[cfg_attr(not(unix), allow(dead_code))]
fn parse_status_euid(status: &str) -> Option<u32> {
    let line = status.lines().find(|l| l.starts_with("Uid:"))?;
    line.split_whitespace().nth(2)?.parse().ok()
}

#[cfg(unix)]
fn probe_file_uid() -> Option<u32> {
    use std::os::unix::fs::MetadataExt;
    let probe = std::env::temp_dir().join(format!(".duckdb_bolt_uid_probe_{}", std::process::id()));
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&probe)
        .ok()?;
    let uid = file.metadata().ok().map(|m| m.uid());
    drop(file);
    let _ = std::fs::remove_file(&probe);
    uid
}

/// Whether `owner` may own a directory the engine loads code from.
#[cfg(unix)]
fn trusted_owner(owner: u32) -> bool {
    // Fail closed (W2-D-003 item 4): when the effective uid cannot be determined, only
    // root-owned directories are trusted.
    owner == 0 || effective_uid() == Some(owner)
}

/// Why `owner` is not trusted, for error messages.
#[cfg(unix)]
fn untrusted_owner_reason(owner: u32) -> String {
    match effective_uid() {
        Some(_) => format!("owned by uid {owner}, not by the DuckDB process user or root"),
        None => format!(
            "owned by uid {owner}; the effective user id of this process cannot be determined \
             (no /proc and no writable temporary directory), so only root-owned directories \
             are accepted"
        ),
    }
}

#[cfg(unix)]
fn check_private_dir(dir: &Path, meta: &std::fs::Metadata, requested: &str) -> Result<()> {
    use std::os::unix::fs::{MetadataExt, PermissionsExt};
    if !trusted_owner(meta.uid()) {
        return Err(Error::invalid_input(format!(
            "bolt_ptx_cache_dir '{requested}' is unsafe: '{}' is {}",
            dir.display(),
            untrusted_owner_reason(meta.uid())
        )));
    }
    if meta.permissions().mode() & 0o022 != 0 {
        return Err(Error::invalid_input(format!(
            "bolt_ptx_cache_dir '{requested}' is unsafe: '{}' is group- or world-writable; \
             use a directory only the DuckDB process user can write to",
            dir.display()
        )));
    }
    Ok(())
}

#[cfg(unix)]
fn check_ancestors(dir: &Path, requested: &str) -> Result<()> {
    use std::os::unix::fs::{MetadataExt, PermissionsExt};
    // Walk the canonical path, so symlinked components are checked where they point.
    let canonical = std::fs::canonicalize(dir).map_err(|e| {
        Error::invalid_input(format!(
            "bolt_ptx_cache_dir '{requested}' cannot be inspected: {e}"
        ))
    })?;
    for ancestor in canonical.ancestors().skip(1) {
        let meta = std::fs::metadata(ancestor).map_err(|e| {
            Error::invalid_input(format!(
                "bolt_ptx_cache_dir '{requested}' cannot be inspected: '{}': {e}",
                ancestor.display()
            ))
        })?;
        if !trusted_owner(meta.uid()) {
            return Err(Error::invalid_input(format!(
                "bolt_ptx_cache_dir '{requested}' is unsafe: its parent '{}' is {}",
                ancestor.display(),
                untrusted_owner_reason(meta.uid())
            )));
        }
        let mode = meta.permissions().mode();
        if mode & 0o002 != 0 && mode & 0o1000 == 0 {
            return Err(Error::invalid_input(format!(
                "bolt_ptx_cache_dir '{requested}' is unsafe: its parent '{}' is world-writable \
                 without the sticky bit, so another user could replace it",
                ancestor.display()
            )));
        }
    }
    Ok(())
}

/// Windows: the directory must be inside the user's profile (see [`validate_ptx_cache_dir`]).
#[cfg(windows)]
fn check_private_dir(dir: &Path, _meta: &std::fs::Metadata, requested: &str) -> Result<()> {
    let canonical = std::fs::canonicalize(dir).map_err(|e| {
        Error::invalid_input(format!(
            "bolt_ptx_cache_dir '{requested}' cannot be inspected: {e}"
        ))
    })?;
    let private = windows_private_roots();
    let shared = windows_shared_roots();
    if windows_dir_is_private(&canonical, &private, &shared) {
        return Ok(());
    }
    Err(Error::invalid_input(format!(
        "bolt_ptx_cache_dir '{requested}' is unsafe: '{}' is not inside the user profile \
         (%USERPROFILE%, %LOCALAPPDATA% or %APPDATA%). Windows ACLs are not inspected, and \
         directories elsewhere (for example C:\\ptx) are usually writable by other users; \
         use a directory such as %LOCALAPPDATA%\\duckdb_bolt\\ptx",
        canonical.display()
    )))
}

/// Profile directories whose default ACL grants write access only to the user, SYSTEM and
/// Administrators. Roots with fewer than two normal components (a bare drive) are ignored.
#[cfg(windows)]
fn windows_private_roots() -> Vec<PathBuf> {
    windows_env_roots(&["USERPROFILE", "LOCALAPPDATA", "APPDATA"])
        .into_iter()
        .filter(|root| {
            root.components()
                .filter(|c| matches!(c, std::path::Component::Normal(_)))
                .count()
                >= 2
        })
        .collect()
}

/// Shared profile directories that may lie inside a private root on unusual setups.
#[cfg(windows)]
fn windows_shared_roots() -> Vec<PathBuf> {
    windows_env_roots(&["PUBLIC", "ALLUSERSPROFILE", "ProgramData"])
}

#[cfg(windows)]
fn windows_env_roots(vars: &[&str]) -> Vec<PathBuf> {
    vars.iter()
        .filter_map(std::env::var_os)
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
        .filter_map(|p| std::fs::canonicalize(p).ok())
        .collect()
}

/// Whether canonical `dir` is inside a private root and not inside a shared one
/// (component-wise, case-insensitive; see `sandbox::path_is_within`).
#[cfg_attr(not(windows), allow(dead_code))]
fn windows_dir_is_private(dir: &Path, private: &[PathBuf], shared: &[PathBuf]) -> bool {
    use crate::sandbox::path_is_within;
    private.iter().any(|root| path_is_within(dir, root))
        && !shared.iter().any(|root| path_is_within(dir, root))
}

/// Windows: covered by [`check_private_dir`], which checks the canonical location.
#[cfg(windows)]
fn check_ancestors(_dir: &Path, _requested: &str) -> Result<()> {
    Ok(())
}

#[cfg(not(any(unix, windows)))]
fn check_private_dir(_dir: &Path, _meta: &std::fs::Metadata, _requested: &str) -> Result<()> {
    // No ownership information on this platform (wasm32 has no file system).
    Ok(())
}

#[cfg(not(any(unix, windows)))]
fn check_ancestors(_dir: &Path, _requested: &str) -> Result<()> {
    Ok(())
}

struct BoltDeviceOption;

impl CustomOptionCallbacks for BoltDeviceOption {
    fn update(
        &self,
        context: &Context,
        value: Value,
        scope: SettingScope,
        is_reset: bool,
    ) -> Result<Value> {
        const NAME: &str = "bolt_device";
        if refreshing() {
            // sync_with_runtime writes the runtime's own (applied) value back.
            return Ok(value);
        }
        ensure_global(NAME, scope)?;
        let dev = requested(NAME, &value, is_reset, DEFAULT_DEVICE)?;
        if dev < 0 {
            return Err(Error::invalid_input(
                "bolt_device must be a non-negative CUDA device ordinal",
            ));
        }
        apply(|cfg| cfg.device == dev, |cfg| cfg.device = dev)?;
        remember(OptionKind::Device, dev.to_string());
        // On RESET, replacing the value replaces DuckDB's per-call copy of the registered
        // default, which is what RESET then stores: the documented default wins over the
        // load-time default.
        if is_reset {
            dev.value(context)
        } else {
            Ok(value)
        }
    }
}

struct BoltMemoryBudgetOption;

impl CustomOptionCallbacks for BoltMemoryBudgetOption {
    fn update(
        &self,
        context: &Context,
        value: Value,
        scope: SettingScope,
        is_reset: bool,
    ) -> Result<Value> {
        const NAME: &str = "bolt_memory_budget";
        if refreshing() {
            // sync_with_runtime writes the runtime's own (applied) value back.
            return Ok(value);
        }
        ensure_global(NAME, scope)?;
        let bytes = requested(NAME, &value, is_reset, DEFAULT_MEMORY_BUDGET)?;
        let budget = memory_budget_from_bytes(bytes)?;
        apply(
            |cfg| cfg.memory_budget == budget,
            |cfg| cfg.memory_budget = budget,
        )?;
        remember(OptionKind::MemoryBudget, bytes.to_string());
        if is_reset {
            bytes.value(context)
        } else {
            Ok(value)
        }
    }
}

struct BoltPtxCacheDirOption;

impl CustomOptionCallbacks for BoltPtxCacheDirOption {
    fn update(
        &self,
        context: &Context,
        value: Value,
        scope: SettingScope,
        is_reset: bool,
    ) -> Result<Value> {
        const NAME: &str = "bolt_ptx_cache_dir";
        if refreshing() {
            // sync_with_runtime writes the runtime's own (applied) value back.
            return Ok(value);
        }
        ensure_global(NAME, scope)?;
        let raw: String = requested(NAME, &value, is_reset, DEFAULT_PTX_CACHE_DIR.to_string())?;
        // Disabling the cache (empty value or RESET) is always allowed. Pointing the engine
        // at a directory is not when DuckDB's sandbox is on: PTX in that directory is
        // JIT-loaded onto the GPU. Checked before touching the file system so a sandboxed
        // session cannot probe paths through this option.
        let wants_dir = !raw.trim().is_empty();
        if wants_dir && !FileAccessPolicy::from_context(context)?.external_access_enabled() {
            return Err(permission_error(
                "Cannot set bolt_ptx_cache_dir: file system operations are disabled by \
                 configuration (enable_external_access = false)",
            ));
        }
        let dir = validate_ptx_cache_dir(&raw)?;
        apply(
            |cfg| cfg.ptx_cache_dir == dir,
            |cfg| cfg.ptx_cache_dir = dir.clone(),
        )?;
        // Show exactly what the runtime stores: the trimmed path, or '' when disabled
        // (on RESET this is the documented default, see BoltDeviceOption).
        let trimmed = raw.trim();
        remember(OptionKind::PtxCacheDir, trimmed);
        if !is_reset && trimmed == raw {
            Ok(value)
        } else {
            trimmed.value(context)
        }
    }
}

struct BoltEnableTracingOption;

impl CustomOptionCallbacks for BoltEnableTracingOption {
    fn update(
        &self,
        context: &Context,
        value: Value,
        scope: SettingScope,
        is_reset: bool,
    ) -> Result<Value> {
        const NAME: &str = "bolt_enable_tracing";
        if refreshing() {
            // sync_with_runtime writes the runtime's own (applied) value back.
            return Ok(value);
        }
        ensure_global(NAME, scope)?;
        let enabled = requested(NAME, &value, is_reset, DEFAULT_ENABLE_TRACING)?;
        apply(
            |cfg| cfg.enable_tracing == enabled,
            |cfg| cfg.enable_tracing = enabled,
        )?;
        remember(OptionKind::EnableTracing, enabled.to_string());
        if is_reset {
            enabled.value(context)
        } else {
            Ok(value)
        }
    }
}

/// Values registered as the options' defaults when a database loads the extension: the
/// runtime's current configuration (see the module documentation).
struct LoadTimeDefaults {
    device: i32,
    memory_budget: i64,
    ptx_cache_dir: String,
    enable_tracing: bool,
}

impl LoadTimeDefaults {
    fn from_config(cfg: &BoltConfig) -> Self {
        Self {
            device: cfg.device,
            memory_budget: memory_budget_to_bytes(cfg.memory_budget),
            ptx_cache_dir: cfg
                .ptx_cache_dir
                .as_ref()
                .map(|p| p.to_string_lossy().into_owned())
                .unwrap_or_default(),
            enable_tracing: cfg.enable_tracing,
        }
    }

    fn current() -> Self {
        // On wasm32 the runtime is not consulted at load time; it starts at the defaults.
        #[cfg(not(target_arch = "wasm32"))]
        if let Ok(cfg) = BoltRuntime::global().config() {
            return Self::from_config(&cfg);
        }
        Self::from_config(&BoltConfig::default())
    }
}

/// Register all `bolt_*` custom configuration options on `extension`.
pub fn register_options(extension: &Extension) -> Result<()> {
    let defaults = LoadTimeDefaults::current();
    // The registered defaults are values the extension stored (see `known_values`).
    remember(OptionKind::Device, defaults.device.to_string());
    remember(OptionKind::MemoryBudget, defaults.memory_budget.to_string());
    remember(OptionKind::PtxCacheDir, defaults.ptx_cache_dir.clone());
    remember(
        OptionKind::EnableTracing,
        defaults.enable_tracing.to_string(),
    );

    CustomOptionBuilder::new(
        "bolt_device",
        format!(
            "CUDA device ordinal used by the Craton Bolt GPU SQL engine (RESET: {DEFAULT_DEVICE}). \
             {PROCESS_WIDE}"
        ),
        i32::logical_type(extension)?,
        BoltDeviceOption,
    )
    .default_value(defaults.device.value(extension)?)
    .no_session()
    .default_scope(SettingScope::Global)
    .register(extension)?;

    CustomOptionBuilder::new(
        "bolt_memory_budget",
        format!(
            "GPU memory pool budget in bytes for Craton Bolt (0 means uncapped; RESET: \
             {DEFAULT_MEMORY_BUDGET}). {PROCESS_WIDE}"
        ),
        i64::logical_type(extension)?,
        BoltMemoryBudgetOption,
    )
    .default_value(defaults.memory_budget.value(extension)?)
    .no_session()
    .default_scope(SettingScope::Global)
    .register(extension)?;

    CustomOptionBuilder::new(
        "bolt_ptx_cache_dir",
        format!(
            "Absolute directory path for the Craton Bolt persistent on-disk PTX kernel cache \
             ('' disables it and is the RESET value; refused when enable_external_access is \
             false; on Unix it must be private to the DuckDB process user). {PROCESS_WIDE}"
        ),
        String::logical_type(extension)?,
        BoltPtxCacheDirOption,
    )
    .default_value(defaults.ptx_cache_dir.value(extension)?)
    .no_session()
    .default_scope(SettingScope::Global)
    .register(extension)?;

    CustomOptionBuilder::new(
        "bolt_enable_tracing",
        format!(
            "Enable execution tracing spans in the Craton Bolt GPU SQL engine (RESET: \
             {DEFAULT_ENABLE_TRACING}). {PROCESS_WIDE}"
        ),
        bool::logical_type(extension)?,
        BoltEnableTracingOption,
    )
    .default_value(defaults.enable_tracing.value(extension)?)
    .no_session()
    .default_scope(SettingScope::Global)
    .register(extension)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_budget_conversion() {
        assert_eq!(memory_budget_from_bytes(0).unwrap(), None);
        assert_eq!(memory_budget_from_bytes(4096).unwrap(), Some(4096));
        assert!(memory_budget_from_bytes(-1).is_err());
        #[cfg(target_pointer_width = "32")]
        assert!(memory_budget_from_bytes(i64::from(u32::MAX) + 1).is_err());
        #[cfg(target_pointer_width = "64")]
        assert_eq!(
            memory_budget_from_bytes(i64::MAX).unwrap(),
            Some(i64::MAX as usize)
        );
        assert_eq!(memory_budget_to_bytes(None), 0);
        assert_eq!(memory_budget_to_bytes(Some(4096)), 4096);
        #[cfg(target_pointer_width = "64")]
        assert_eq!(memory_budget_to_bytes(Some(usize::MAX)), i64::MAX);
    }

    #[test]
    fn load_time_defaults_mirror_the_config() {
        let d = LoadTimeDefaults::from_config(&BoltConfig::default());
        assert_eq!(d.device, DEFAULT_DEVICE);
        assert_eq!(d.memory_budget, DEFAULT_MEMORY_BUDGET);
        assert_eq!(d.ptx_cache_dir, DEFAULT_PTX_CACHE_DIR);
        assert_eq!(d.enable_tracing, DEFAULT_ENABLE_TRACING);
        let cfg = BoltConfig {
            device: 3,
            memory_budget: Some(1 << 20),
            ptx_cache_dir: Some(PathBuf::from("/opt/ptx")),
            enable_tracing: true,
        };
        let d = LoadTimeDefaults::from_config(&cfg);
        assert_eq!(d.device, 3);
        assert_eq!(d.memory_budget, 1 << 20);
        assert_eq!(PathBuf::from(&d.ptx_cache_dir), PathBuf::from("/opt/ptx"));
        assert!(d.enable_tracing);
    }

    #[test]
    fn session_scope_is_refused_with_a_process_wide_message() {
        assert!(ensure_global("bolt_device", SettingScope::Global).is_ok());
        let err = ensure_global("bolt_device", SettingScope::Session).unwrap_err();
        assert!(err.message.contains("process-wide"), "{err}");
        assert!(err.message.contains("SET GLOBAL bolt_device"), "{err}");
    }

    #[test]
    fn settings_are_classified_against_the_runtime() {
        let cfg = BoltConfig {
            device: 1,
            memory_budget: Some(4096),
            ptx_cache_dir: Some(PathBuf::from("/opt/ptx")),
            enable_tracing: false,
        };
        // A raw VARCHAR copy of a typed option is a startup value, even if it matches.
        assert_eq!(
            classify(OptionKind::Device, "1", true, &cfg),
            SettingState::Startup("1".into())
        );
        // Typed values matching the runtime are current.
        assert_eq!(
            classify(OptionKind::Device, "1", false, &cfg),
            SettingState::Current
        );
        assert_eq!(
            classify(OptionKind::MemoryBudget, "4096", false, &cfg),
            SettingState::Current
        );
        assert_eq!(
            classify(OptionKind::PtxCacheDir, "/opt/ptx/", false, &cfg),
            SettingState::Current
        );
        // A value the extension stored earlier is stale; one it never stored is a startup value.
        remember(OptionKind::EnableTracing, "true");
        assert_eq!(
            classify(OptionKind::EnableTracing, "true", false, &cfg),
            SettingState::Stale
        );
        let never = "/w3d/never/stored/by/the/extension";
        assert_eq!(
            classify(OptionKind::PtxCacheDir, never, false, &cfg),
            SettingState::Startup(never.into())
        );
        remember(OptionKind::PtxCacheDir, never);
        assert_eq!(
            classify(OptionKind::PtxCacheDir, never, false, &cfg),
            SettingState::Stale
        );
        // The runtime's own text form round-trips.
        for kind in OptionKind::ALL {
            assert!(
                kind.matches_runtime(&kind.runtime_text(&cfg), &cfg),
                "{kind:?}"
            );
        }
        assert!(OptionKind::PtxCacheDir.matches_runtime("", &BoltConfig::default()));
    }

    #[test]
    fn refresh_guard_is_scoped() {
        assert!(!refreshing());
        {
            let _g = RefreshGuard::new();
            assert!(refreshing());
        }
        assert!(!refreshing());
    }

    #[test]
    fn windows_private_dir_rule() {
        let base = std::env::temp_dir();
        let profile = base.join("w3d_profile");
        let public = profile.join("Public");
        let private = [profile.clone()];
        let shared = [public.clone()];
        assert!(windows_dir_is_private(
            &profile.join("ptx"),
            &private,
            &shared
        ));
        assert!(windows_dir_is_private(&profile, &private, &shared));
        assert!(!windows_dir_is_private(
            &public.join("ptx"),
            &private,
            &shared
        ));
        assert!(!windows_dir_is_private(
            &base.join("elsewhere"),
            &private,
            &shared
        ));
        assert!(!windows_dir_is_private(&profile.join("ptx"), &[], &shared));
    }

    #[cfg(windows)]
    #[test]
    fn ptx_cache_dir_outside_the_profile_is_refused_on_windows() {
        let Some(system_root) = std::env::var_os("SystemRoot") else {
            return;
        };
        let outside = PathBuf::from(system_root).join("bolt_w3d_ptx_never_created");
        let err = validate_ptx_cache_dir(&outside.to_string_lossy()).unwrap_err();
        assert!(err.message.contains("user profile"), "{err}");
    }

    #[test]
    fn ptx_check_freshness_needs_same_dir_fingerprint_and_age() {
        // W4-C-002: the full PTX directory check is skipped only while nothing changed.
        let now = Instant::now();
        let fp = DirFingerprint(vec![1, 2, 3]);
        let last = PtxCheck {
            dir: PathBuf::from("/opt/ptx"),
            fingerprint: fp.clone(),
            at: now,
        };
        let dir = Path::new("/opt/ptx");
        assert!(ptx_check_is_fresh(Some(&last), dir, Some(&fp), now));
        assert!(ptx_check_is_fresh(
            Some(&last),
            dir,
            Some(&fp),
            now + PTX_FULL_RECHECK_INTERVAL / 2
        ));
        // Too old, another directory, a changed or unknown fingerprint, or no earlier check.
        assert!(!ptx_check_is_fresh(
            Some(&last),
            dir,
            Some(&fp),
            now + PTX_FULL_RECHECK_INTERVAL
        ));
        assert!(!ptx_check_is_fresh(
            Some(&last),
            Path::new("/opt/other"),
            Some(&fp),
            now
        ));
        assert!(!ptx_check_is_fresh(
            Some(&last),
            dir,
            Some(&DirFingerprint(vec![1, 2, 4])),
            now
        ));
        assert!(!ptx_check_is_fresh(Some(&last), dir, None, now));
        assert!(!ptx_check_is_fresh(None, dir, Some(&fp), now));
    }

    #[cfg(any(unix, windows))]
    #[test]
    fn dir_fingerprint_changes_when_the_directory_is_replaced() {
        let dir = scratch("fingerprint");
        std::fs::create_dir_all(&dir).unwrap();
        let before = dir_fingerprint(&dir).expect("an existing directory has a fingerprint");
        assert_eq!(dir_fingerprint(&dir), Some(before.clone()));
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!(dir_fingerprint(&dir), None);
        // A regular file in its place never matches (file type is part of it).
        std::fs::write(&dir, b"x").unwrap();
        assert_ne!(dir_fingerprint(&dir), Some(before.clone()));
        std::fs::remove_file(&dir).unwrap();
        #[cfg(unix)]
        {
            std::fs::create_dir_all(&dir).unwrap();
            chmod(&dir, 0o700);
            let private = dir_fingerprint(&dir).unwrap();
            chmod(&dir, 0o777);
            assert_ne!(dir_fingerprint(&dir).unwrap(), private);
            chmod(&dir, 0o700);
            let _ = std::fs::remove_dir_all(&dir);
        }
    }

    #[test]
    fn status_euid_parsing() {
        let status = "Name:\tduckdb\nUid:\t1000\t1001\t1000\t1001\nGid:\t100\t100\t100\t100\n";
        assert_eq!(parse_status_euid(status), Some(1001));
        assert_eq!(parse_status_euid("Name:\tx\n"), None);
        assert_eq!(parse_status_euid("Uid:\tnot-a-number\tx\n"), None);
    }

    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("bolt_w1d_ptx_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[cfg(unix)]
    fn chmod(path: &Path, mode: u32) {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode)).unwrap();
    }

    #[test]
    fn ptx_cache_dir_empty_disables() {
        assert_eq!(validate_ptx_cache_dir("").unwrap(), None);
        assert_eq!(validate_ptx_cache_dir("   ").unwrap(), None);
    }

    #[test]
    fn ptx_cache_dir_rejects_relative_and_files() {
        assert!(validate_ptx_cache_dir("relative/ptx").is_err());
        assert!(validate_ptx_cache_dir("./ptx").is_err());

        let dir = scratch("file");
        std::fs::create_dir_all(&dir).unwrap();
        #[cfg(unix)]
        chmod(&dir, 0o700);
        let file = dir.join("not_a_dir");
        std::fs::write(&file, b"x").unwrap();
        let err = validate_ptx_cache_dir(&file.to_string_lossy()).unwrap_err();
        assert!(err.message.contains("not a directory"), "{err}");
        // A missing directory below a regular file cannot be created.
        let err = validate_ptx_cache_dir(&file.join("sub").to_string_lossy()).unwrap_err();
        assert!(err.message.contains("not a directory"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn ptx_cache_dir_accepts_private_existing_and_missing_dirs() {
        let dir = scratch("ok");
        std::fs::create_dir_all(&dir).unwrap();
        #[cfg(unix)]
        chmod(&dir, 0o700);
        let padded = format!("  {}  ", dir.to_string_lossy());
        assert_eq!(validate_ptx_cache_dir(&padded).unwrap(), Some(dir.clone()));
        // Not created eagerly.
        let missing = dir.join("later");
        assert_eq!(
            validate_ptx_cache_dir(&missing.to_string_lossy()).unwrap(),
            Some(missing.clone())
        );
        assert!(!missing.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn ptx_cache_dir_rejects_world_and_group_writable() {
        let dir = scratch("ww");
        std::fs::create_dir_all(&dir).unwrap();
        chmod(&dir, 0o777);
        let err = validate_ptx_cache_dir(&dir.to_string_lossy()).unwrap_err();
        assert!(err.message.contains("world-writable"), "{err}");
        // A missing directory under a world-writable parent is refused too.
        let err = validate_ptx_cache_dir(&dir.join("child").to_string_lossy()).unwrap_err();
        assert!(err.message.contains("world-writable"), "{err}");
        // Group-writable is refused for the directory itself.
        chmod(&dir, 0o770);
        let err = validate_ptx_cache_dir(&dir.to_string_lossy()).unwrap_err();
        assert!(err.message.contains("group- or world-writable"), "{err}");
        chmod(&dir, 0o700);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn ptx_cache_dir_checks_parents_for_replaceability() {
        let root = scratch("parents");
        let private = root.join("ptx");
        std::fs::create_dir_all(&private).unwrap();
        chmod(&private, 0o700);
        // A world-writable parent without the sticky bit lets anyone rename `ptx` away.
        chmod(&root, 0o777);
        let err = validate_ptx_cache_dir(&private.to_string_lossy()).unwrap_err();
        assert!(err.message.contains("sticky"), "{err}");
        // With the sticky bit (like /tmp) only the owner can rename it.
        chmod(&root, 0o1777);
        assert_eq!(
            validate_ptx_cache_dir(&private.to_string_lossy()).unwrap(),
            Some(private.clone())
        );
        chmod(&root, 0o700);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[cfg(unix)]
    #[test]
    fn effective_uid_matches_owner_of_new_files() {
        use std::os::unix::fs::MetadataExt;
        let dir = scratch("uid");
        std::fs::create_dir_all(&dir).unwrap();
        let uid = effective_uid().expect("effective uid is determinable on Unix");
        assert_eq!(std::fs::metadata(&dir).unwrap().uid(), uid);
        assert!(trusted_owner(uid));
        assert!(trusted_owner(0));
        let other = if uid == 1 { 2 } else { 1 };
        if uid != 0 {
            assert!(!trusted_owner(other));
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}

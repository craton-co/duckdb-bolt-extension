// SPDX-License-Identifier: Apache-2.0

//! File-system access policy derived from DuckDB's sandboxing settings.
//!
//! The extension reads local files (`bolt_load_parquet`, `bolt_load_arrow_ipc`) and points
//! the engine at directories (`SET bolt_ptx_cache_dir`) with `std::fs`, outside of
//! DuckDB's virtual file system. DuckDB therefore cannot enforce
//! `enable_external_access`, `allowed_directories` and `allowed_paths` for us; this
//! module re-implements the check from the callback's client [`Context`]. (Remote URLs,
//! anything containing `://`, are opened through DuckDB's own file system instead, and
//! local loads are additionally probed through it; see `ingest.rs`.)
//!
//! Semantics follow DuckDB's `DBConfig::CanAccessFile`:
//!
//! * `enable_external_access = true` (the default): every path is allowed.
//! * `enable_external_access = false`: a path is allowed only if it equals an entry of
//!   `allowed_paths` or lies inside an entry of `allowed_directories`.
//!
//! Unlike DuckDB (which compares lexically normalized strings), both the requested
//! path and every allow-list entry are canonicalized with [`std::fs::canonicalize`],
//! so symbolic links and `..` components cannot be used to escape an allowed
//! directory. Allow-list entries that cannot be canonicalized (they do not exist, or
//! are remote URLs) never match. The check is component-wise
//! ([`Path::starts_with`]), so `/data2/x` is not inside `/data`.
//!
//! `file:` URLs in the forms DuckDB maps to local paths (`file:/x`, `file:///x`,
//! `file://localhost/x`; see [`local_path_of_file_url`]) are local files too: callers strip
//! the scheme and apply the same canonical check, so a symlink cannot be used to escape an
//! allowed directory through a `file:` URL either (W2-D-001 item 2).
//!
//! On Windows, canonical paths carry a verbatim prefix (`\\?\C:\...`,
//! `\\?\UNC\server\share\...`). Both sides are reduced to the same non-verbatim form,
//! separators are unified, and the comparison is case-insensitive, as it is on NTFS and SMB.
//!
//! Check/open race: [`FileAccessPolicy::open_regular_file`] opens the canonical path and
//! then re-checks *the opened handle*:
//!
//! * Linux / Android: the kernel's path of the open descriptor (`/proc/self/fd/N`) must be
//!   allowed. This closes the rename / symlink-swap race.
//! * Windows: the path of the opened handle (`GetFinalPathNameByHandleW`, W4-C) must be
//!   allowed, which closes the race the same way. A sandboxed load also opens the file
//!   without `FILE_SHARE_DELETE`, so neither the file nor a directory above it can be renamed
//!   or deleted while it is read. If the call fails, the comparison below is used.
//! * Other Unix systems (macOS, the BSDs): the path is canonicalized again after the open,
//!   must resolve to the same allowed canonical path, and that path must name the opened
//!   file (same device and inode), so a file swapped in or out during the check is refused.
//!   The only residual case is a hard link to an outside file placed inside an allowed
//!   directory (accepted, as DuckDB does).
//!
//! See `docs/issues/W2-D-001-sandbox-residual-gaps.md` for what remains.

use std::{
    fs::{File, Metadata},
    path::{Path, PathBuf},
};

use duckdb_neo::{
    Error, FromValue, Result, connection::Context, connection_options::Config, error::DuckDBError,
};

/// DuckDB setting that gates all file-system access.
pub const EXTERNAL_ACCESS_SETTING: &str = "enable_external_access";
/// DuckDB setting listing directories that stay readable without external access.
pub const ALLOWED_DIRECTORIES_SETTING: &str = "allowed_directories";
/// DuckDB setting listing individual files that stay readable without external access.
pub const ALLOWED_PATHS_SETTING: &str = "allowed_paths";

/// Build a DuckDB `Permission Error` (`PermissionException` on the C++ side).
pub fn permission_error(message: impl Into<String>) -> Error {
    Error {
        code: DuckDBError::DUCKDB_V2_ERROR_CONFIGURATION_PERMISSION,
        message: message.into(),
    }
}

/// Snapshot of the sandboxing settings that apply to one callback invocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileAccessPolicy {
    external_access: bool,
    allowed_directories: Vec<String>,
    allowed_paths: Vec<String>,
}

impl Default for FileAccessPolicy {
    /// The default policy is DuckDB's default: unrestricted.
    fn default() -> Self {
        Self::unrestricted()
    }
}

impl FileAccessPolicy {
    /// A policy that allows every path (`enable_external_access = true`).
    pub fn unrestricted() -> Self {
        Self {
            external_access: true,
            allowed_directories: Vec::new(),
            allowed_paths: Vec::new(),
        }
    }

    /// A policy with external access disabled and the given allow-lists.
    pub fn restricted(allowed_directories: Vec<String>, allowed_paths: Vec<String>) -> Self {
        Self {
            external_access: false,
            allowed_directories,
            allowed_paths,
        }
    }

    /// Read the policy from the settings visible to `context`.
    ///
    /// Fails closed: a NULL `enable_external_access` is treated as `false`, and an
    /// error reading it is returned rather than ignored.
    pub fn from_context(context: &Context) -> Result<Self> {
        let config = context.config()?;
        let (value, _scope) = config.get_value(EXTERNAL_ACCESS_SETTING)?;
        let external_access = bool::from_value(&value)?.unwrap_or(false);
        if external_access {
            return Ok(Self::unrestricted());
        }
        Ok(Self::restricted(
            read_string_list(&config, ALLOWED_DIRECTORIES_SETTING)?,
            read_string_list(&config, ALLOWED_PATHS_SETTING)?,
        ))
    }

    /// Whether `enable_external_access` is on.
    pub fn external_access_enabled(&self) -> bool {
        self.external_access
    }

    /// Resolve `path` for reading, or fail with a `Permission Error`.
    ///
    /// With external access enabled the path is returned unchanged. Otherwise the
    /// canonical path is returned when it is allowed. A path that does not exist is
    /// reported with the same permission error as a forbidden one, so a sandboxed
    /// caller cannot probe for the existence of files. Remote URLs are always refused
    /// here; they are never local files.
    ///
    /// Prefer [`Self::open_regular_file`], which also re-checks the opened handle.
    pub fn resolve_read_path(&self, path: &str) -> Result<PathBuf> {
        if self.external_access {
            return Ok(PathBuf::from(path));
        }
        if is_remote(path) {
            return Err(self.denied(path));
        }
        let canonical = std::fs::canonicalize(path).map_err(|_| self.denied(path))?;
        if self.is_allowed(&canonical) {
            Ok(canonical)
        } else {
            Err(self.denied(path))
        }
    }

    /// Resolve `path` through the policy and open it for reading as a regular file.
    ///
    /// The permission check comes first, so a sandboxed caller cannot probe for files.
    /// The file type is checked before the open (opening a FIFO would block) and again on
    /// the opened handle. When the policy is restricted, the opened handle is verified
    /// against the policy as described in the module documentation.
    pub fn open_regular_file(&self, path: &str) -> Result<File> {
        if path.trim().is_empty() {
            return Err(Error::invalid_input("File path cannot be empty"));
        }
        let resolved = self.resolve_read_path(path)?;
        let not_regular = || {
            Error::invalid_input(format!(
                "File '{path}' does not exist or is not a regular file"
            ))
        };
        if !std::fs::metadata(&resolved).is_ok_and(|m| m.is_file()) {
            return Err(not_regular());
        }
        let file = open_for_reading(&resolved, !self.external_access)
            .map_err(|e| Error::invalid_input(format!("Cannot open '{path}': {e}")))?;
        let opened = file
            .metadata()
            .map_err(|e| Error::invalid_input(format!("Cannot inspect '{path}': {e}")))?;
        if !opened.is_file() {
            return Err(not_regular());
        }
        if !self.external_access {
            self.verify_opened(path, &resolved, &file, &opened)?;
        }
        Ok(file)
    }

    /// Re-check an opened file against the policy (closes or narrows the check/open race).
    fn verify_opened(
        &self,
        path: &str,
        resolved: &Path,
        file: &File,
        opened: &Metadata,
    ) -> Result<()> {
        if let Some(actual) = opened_file_path(file) {
            // The kernel's own path of the descriptor is authoritative.
            return if self.is_allowed(&actual) {
                Ok(())
            } else {
                Err(self.denied(path))
            };
        }
        let again = std::fs::canonicalize(path).map_err(|_| self.denied(path))?;
        if !paths_equal(&again, resolved) || !self.is_allowed(&again) {
            return Err(self.denied(path));
        }
        let current = std::fs::metadata(&again).map_err(|_| self.denied(path))?;
        if same_file(opened, &current) {
            Ok(())
        } else {
            Err(self.denied(path))
        }
    }

    fn is_allowed(&self, canonical: &Path) -> bool {
        let target = comparable(canonical);
        self.allowed_paths
            .iter()
            .filter_map(|p| canonicalize_entry(p))
            .any(|p| comparable(&p) == target)
            || self
                .allowed_directories
                .iter()
                .filter_map(|d| canonicalize_entry(d))
                .any(|d| target.starts_with(comparable(&d)))
    }

    fn denied(&self, path: &str) -> Error {
        permission_error(format!(
            "Cannot access file \"{path}\" - file system operations are disabled by configuration \
             (enable_external_access = false and the path is not covered by allowed_directories / allowed_paths)"
        ))
    }
}

/// Open `path` for reading. With `pin` (sandboxed loads) on Windows, other openers are
/// denied `FILE_SHARE_DELETE`, so the file and its parent directories cannot be renamed or
/// deleted while the handle is open. Elsewhere this is a plain open.
fn open_for_reading(path: &Path, pin: bool) -> std::io::Result<File> {
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(windows)]
    if pin {
        use std::os::windows::fs::OpenOptionsExt;
        const FILE_SHARE_READ: u32 = 0x1;
        const FILE_SHARE_WRITE: u32 = 0x2;
        options.share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE);
    }
    #[cfg(not(windows))]
    let _ = pin;
    options.open(path)
}

/// The path the operating system reports for an open file, where that is available.
#[cfg(any(target_os = "linux", target_os = "android"))]
fn opened_file_path(file: &File) -> Option<PathBuf> {
    use std::os::fd::AsRawFd;
    let link = std::fs::read_link(format!("/proc/self/fd/{}", file.as_raw_fd())).ok()?;
    // Deleted files are reported as "<path> (deleted)", which is not inside any allowed
    // directory that still exists; anonymous files ("pipe:[N]") are not absolute.
    link.is_absolute().then_some(link)
}

/// The path the operating system reports for an open file: `GetFinalPathNameByHandleW` on
/// the opened handle (W2-D-001 item 1). The result has the same form as
/// [`std::fs::canonicalize`], which calls the same function with the same flags on a handle
/// of its own (`\\?\C:\...`, `\\?\UNC\server\share\...`), so it compares with the canonical
/// allow-list entries through [`comparable`]. `None` when the call fails (for example on a
/// file system that does not support it); the caller then falls back to the post-open
/// comparison.
#[cfg(windows)]
fn opened_file_path(file: &File) -> Option<PathBuf> {
    use std::{ffi::OsString, os::windows::ffi::OsStringExt, os::windows::io::AsRawHandle};

    // SAFETY: the declaration matches the Win32 prototype `DWORD GetFinalPathNameByHandleW(
    // HANDLE hFile, LPWSTR lpszFilePath, DWORD cchFilePath, DWORD dwFlags)` (kernel32, `system`
    // ABI): `HANDLE` is a pointer, `LPWSTR` a `*mut u16`, `DWORD` a `u32`.
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetFinalPathNameByHandleW(
            file: *mut std::ffi::c_void,
            path: *mut u16,
            path_len: u32,
            flags: u32,
        ) -> u32;
    }
    /// `FILE_NAME_NORMALIZED | VOLUME_NAME_DOS`, what `std::fs::canonicalize` uses.
    const FLAGS: u32 = 0;
    /// Longest path the API returns (`\\?\` paths are limited to 32 767 characters).
    const MAX_CHARS: usize = 32_768;

    let handle = file.as_raw_handle();
    let mut buf = vec![0u16; 512];
    loop {
        let capacity = u32::try_from(buf.len()).ok()?;
        // SAFETY: `handle` is the open handle owned by `file`, which outlives the call; `buf`
        // is a writable buffer of exactly `capacity` UTF-16 units. The function writes at most
        // `capacity` units and returns 0 on failure.
        let n = unsafe { GetFinalPathNameByHandleW(handle, buf.as_mut_ptr(), capacity, FLAGS) };
        let n = usize::try_from(n).ok()?;
        if n == 0 {
            return None;
        }
        if n < buf.len() {
            // Success: `n` characters, without the terminating NUL.
            buf.truncate(n);
            break;
        }
        // Too small: `n` is the size needed, including the terminating NUL.
        if n > MAX_CHARS + 1 {
            return None;
        }
        buf.resize(n, 0);
    }
    let path = PathBuf::from(OsString::from_wide(&buf));
    path.is_absolute().then_some(path)
}

/// The path the operating system reports for an open file, where that is available.
///
/// Elsewhere (macOS, the BSDs) no path is reported: the post-open comparison of device and
/// inode is a real identity check on Unix, so the only residual case is a hard link
/// (W2-D-001 item 3).
#[cfg(not(any(target_os = "linux", target_os = "android", windows)))]
fn opened_file_path(_file: &File) -> Option<PathBuf> {
    None
}

/// Whether two metadata snapshots describe the same file.
#[cfg(unix)]
fn same_file(a: &Metadata, b: &Metadata) -> bool {
    use std::os::unix::fs::MetadataExt;
    a.dev() == b.dev() && a.ino() == b.ino()
}

/// Whether two metadata snapshots describe the same file.
///
/// Stable Rust exposes no file identity (volume serial number and file index) on
/// Windows, so size and timestamps are compared. That detects a swap to a different
/// file in practice, but it is not a proof of identity.
#[cfg(windows)]
fn same_file(a: &Metadata, b: &Metadata) -> bool {
    use std::os::windows::fs::MetadataExt;
    a.len() == b.len()
        && a.modified().ok() == b.modified().ok()
        && a.created().ok() == b.created().ok()
        && a.file_attributes() == b.file_attributes()
}

/// Whether two metadata snapshots describe the same file (size and timestamps only).
#[cfg(not(any(unix, windows)))]
fn same_file(a: &Metadata, b: &Metadata) -> bool {
    a.len() == b.len()
        && a.modified().ok() == b.modified().ok()
        && a.created().ok() == b.created().ok()
}

/// Compare two canonical paths the way [`FileAccessPolicy`] does.
fn paths_equal(a: &Path, b: &Path) -> bool {
    comparable(a) == comparable(b)
}

/// Whether canonical `path` is `root` or inside it, compared the way [`FileAccessPolicy`]
/// compares allow-list entries (component-wise; on Windows without the verbatim prefix and
/// case-insensitively).
pub(crate) fn path_is_within(path: &Path, root: &Path) -> bool {
    comparable(path).starts_with(comparable(root))
}

/// The form in which canonical paths are compared.
///
/// Unix: the path itself (file names are case-sensitive byte strings).
#[cfg(not(windows))]
fn comparable(path: &Path) -> PathBuf {
    path.to_path_buf()
}

/// The form in which canonical paths are compared.
///
/// Windows: the verbatim prefix is removed (`\\?\C:\x` becomes `C:\x`, and
/// `\\?\UNC\srv\share\x` becomes `\\srv\share\x`), `/` becomes `\`, and the path is
/// upper-cased one UTF-16 unit at a time (see [`windows_upcase_char`]), because NTFS and SMB
/// compare names case-insensitively through such a per-unit upcase table. The result is
/// compared component-wise ([`Path::starts_with`]), so "inside an allowed directory" keeps its
/// separator-boundary semantics.
#[cfg(windows)]
fn comparable(path: &Path) -> PathBuf {
    PathBuf::from(windows_comparable_text(&path.to_string_lossy()))
}

/// String form of [`comparable`] on Windows (separate so it can be unit-tested anywhere).
#[cfg_attr(not(windows), allow(dead_code))]
fn windows_comparable_text(text: &str) -> String {
    let text = text.replace('/', "\\");
    let text = if let Some(rest) = text.strip_prefix(r"\\?\UNC\") {
        format!(r"\\{rest}")
    } else if let Some(rest) = text.strip_prefix(r"\\?\") {
        rest.to_string()
    } else {
        text
    };
    text.chars().map(windows_upcase_char).collect()
}

/// Case folding for one character, never wider than Windows' own ordinal case-insensitive
/// comparison (W9-D-003).
///
/// NTFS (its `$UpCase` table) and `CompareStringOrdinal(.., bIgnoreCase = TRUE)` upcase each
/// UTF-16 unit on its own with a simple one-to-one table. Full Unicode case mapping
/// (`str::to_lowercase`) folds more: KELVIN SIGN (U+212A) lower-cases to ASCII `k`, OHM SIGN
/// (U+2126) to `ω`, so an allow-listed directory spelled with such a character would also have
/// admitted its distinct sibling. Here a character is replaced only by its simple uppercase
/// form, and only when
///
/// * it is a single BMP character (one UTF-16 unit) whose uppercase is again exactly one BMP
///   character (so `ß`, whose uppercase is `SS`, and supplementary characters stay as they are);
/// * a non-ASCII character does not upcase into ASCII (`ı` U+0131 and `ſ` U+017F, which Windows
///   does not merge with `I` and `S`);
/// * on Windows, `CompareStringOrdinal` agrees that the two are equal ignoring case.
///
/// Two characters are therefore merged only if Windows' ordinal comparison merges them too.
/// Where Rust's table merges less, the comparison fails closed (the paths compared are
/// canonical and carry the on-disk spelling, so that only concerns drive letters and UNC names
/// spelled differently from the system's report).
#[cfg_attr(not(windows), allow(dead_code))]
fn windows_upcase_char(c: char) -> char {
    if c.is_ascii() {
        return c.to_ascii_uppercase();
    }
    if c.len_utf16() != 1 {
        return c;
    }
    let mut upper = c.to_uppercase();
    let (Some(u), None) = (upper.next(), upper.next()) else {
        return c;
    };
    if u == c || u.is_ascii() || u.len_utf16() != 1 || !os_ordinal_equal_ignore_case(c, u) {
        return c;
    }
    u
}

/// Whether Windows' ordinal case-insensitive comparison (`CompareStringOrdinal` with
/// `bIgnoreCase = TRUE`, the rule NTFS and the Win32 file APIs use) considers two characters
/// equal.
#[cfg(windows)]
fn os_ordinal_equal_ignore_case(a: char, b: char) -> bool {
    // SAFETY: the declaration matches the Win32 prototype `int CompareStringOrdinal(LPCWCH
    // lpString1, int cchCount1, LPCWCH lpString2, int cchCount2, BOOL bIgnoreCase)` (kernel32,
    // `system` ABI): `LPCWCH` is a `*const u16`, `int` and `BOOL` are `i32`.
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn CompareStringOrdinal(
            string1: *const u16,
            count1: i32,
            string2: *const u16,
            count2: i32,
            ignore_case: i32,
        ) -> i32;
    }
    /// `CSTR_EQUAL`.
    const CSTR_EQUAL: i32 = 2;

    let (mut buf_a, mut buf_b) = ([0u16; 2], [0u16; 2]);
    let a = a.encode_utf16(&mut buf_a);
    let b = b.encode_utf16(&mut buf_b);
    let (Ok(len_a), Ok(len_b)) = (i32::try_from(a.len()), i32::try_from(b.len())) else {
        return false;
    };
    // SAFETY: `a` and `b` are live, readable buffers of exactly `len_a` and `len_b` UTF-16
    // units; the function only reads them and returns 0 on failure.
    let result = unsafe { CompareStringOrdinal(a.as_ptr(), len_a, b.as_ptr(), len_b, 1) };
    result == CSTR_EQUAL
}

/// Off Windows there is no OS table to consult; the other conditions of
/// [`windows_upcase_char`] still apply (the string form is unit-tested everywhere).
#[cfg(not(windows))]
#[cfg_attr(not(test), allow(dead_code))]
fn os_ordinal_equal_ignore_case(_a: char, _b: char) -> bool {
    true
}

/// Canonicalize an allow-list entry; entries that cannot be resolved never match.
fn canonicalize_entry(entry: &str) -> Option<PathBuf> {
    let entry = entry.trim();
    if entry.is_empty() || is_remote(entry) {
        return None;
    }
    std::fs::canonicalize(entry).ok()
}

/// Whether `path` looks like a URL (`s3://`, `https://`, `file://`, ...).
///
/// Such paths are never opened with `std::fs` as given; `file:` URLs that DuckDB maps to
/// local paths are first reduced with [`local_path_of_file_url`].
pub fn is_remote(path: &str) -> bool {
    path.contains("://")
}

/// The local path a `file:` URL denotes, exactly as DuckDB's `FileSystem::ExpandPath`
/// (`GetFileUrlOffset`) computes it, or `None` for anything else (including `file:` URLs
/// with a host other than `localhost`, which DuckDB does not map either).
///
/// | URL | Unix | Windows |
/// | :--- | :--- | :--- |
/// | `file:/x/y` | `/x/y` | `x/y` |
/// | `file:///x/y` | `/x/y` | `x/y` |
/// | `file://localhost/x/y` | `/x/y` | `x/y` |
///
/// On Windows the URL path starts with the drive, so `file:///C:/x` is `C:/x`.
pub fn local_path_of_file_url(path: &str) -> Option<&str> {
    file_url_offset(path, cfg!(windows)).and_then(|offset| path.get(offset..))
}

/// `GetFileUrlOffset` from DuckDB's `file_system.cpp`, for either platform (unit-testable).
fn file_url_offset(path: &str, windows: bool) -> Option<usize> {
    if !path.starts_with("file:/") {
        return None;
    }
    let bytes = path.as_bytes();
    let pick = |unix: usize| if windows { unix + 1 } else { unix };
    if bytes.get(6) != Some(&b'/') {
        // file:/some/path
        return Some(pick(5));
    }
    if bytes.get(7) == Some(&b'/') {
        // file:///some/path
        return Some(pick(7));
    }
    if path.get(7..17) == Some("localhost/") {
        // file://localhost/some/path
        return Some(pick(16));
    }
    None
}

/// Read a `VARCHAR[]` setting; an unknown setting or a NULL value is an empty list.
fn read_string_list(config: &Config<'_>, name: &str) -> Result<Vec<String>> {
    let Some((value, _scope)) = config.try_get_value(name)? else {
        return Ok(Vec::new());
    };
    Ok(<Vec<Option<String>>>::from_value(&value)?
        .unwrap_or_default()
        .into_iter()
        .flatten()
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TempTree {
        root: PathBuf,
    }

    impl TempTree {
        fn new(tag: &str) -> Self {
            let root =
                std::env::temp_dir().join(format!("bolt_w1d_sandbox_{tag}_{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&root);
            std::fs::create_dir_all(root.join("allowed")).unwrap();
            std::fs::create_dir_all(root.join("allowed2")).unwrap();
            std::fs::create_dir_all(root.join("other")).unwrap();
            std::fs::write(root.join("allowed").join("a.bin"), b"a").unwrap();
            std::fs::write(root.join("allowed2").join("b.bin"), b"b").unwrap();
            std::fs::write(root.join("other").join("c.bin"), b"c").unwrap();
            Self { root }
        }

        fn s(&self, rel: &str) -> String {
            self.root.join(rel).to_string_lossy().into_owned()
        }
    }

    impl Drop for TempTree {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.root);
        }
    }

    fn is_permission(err: &Error) -> bool {
        err.code == DuckDBError::DUCKDB_V2_ERROR_CONFIGURATION_PERMISSION
    }

    #[test]
    fn unrestricted_allows_everything_unchanged() {
        let p = FileAccessPolicy::default();
        assert!(p.external_access_enabled());
        assert_eq!(
            p.resolve_read_path("no/such/file").unwrap(),
            PathBuf::from("no/such/file")
        );
    }

    #[test]
    fn restricted_without_allow_lists_denies_existing_and_missing() {
        let t = TempTree::new("deny");
        let p = FileAccessPolicy::restricted(vec![], vec![]);
        assert!(!p.external_access_enabled());
        let err = p.resolve_read_path(&t.s("allowed/a.bin")).unwrap_err();
        assert!(is_permission(&err), "{err}");
        assert!(err.message.contains("enable_external_access"));
        // Missing files get the same error (no existence oracle).
        let err = p
            .resolve_read_path(&t.s("allowed/missing.bin"))
            .unwrap_err();
        assert!(is_permission(&err));
        assert!(is_permission(
            &p.resolve_read_path("s3://bucket/x.parquet").unwrap_err()
        ));
        assert!(is_permission(
            &p.open_regular_file(&t.s("allowed/a.bin")).unwrap_err()
        ));
    }

    #[test]
    fn allowed_directories_are_component_prefixes() {
        let t = TempTree::new("dirs");
        // DuckDB stores allowed directories with forward slashes and a trailing '/'.
        let dir = format!("{}/", t.s("allowed").replace('\\', "/"));
        let p = FileAccessPolicy::restricted(vec![dir, "s3://bucket/".into(), "".into()], vec![]);
        let ok = p.resolve_read_path(&t.s("allowed/a.bin")).unwrap();
        assert!(ok.is_absolute());
        assert_eq!(std::fs::read(&ok).unwrap(), b"a");
        // Sibling with a common string prefix is not inside.
        assert!(is_permission(
            &p.resolve_read_path(&t.s("allowed2/b.bin")).unwrap_err()
        ));
        // `..` traversal out of the allowed directory is refused.
        let escaped = t
            .root
            .join("allowed")
            .join("..")
            .join("other")
            .join("c.bin");
        assert!(is_permission(
            &p.resolve_read_path(&escaped.to_string_lossy()).unwrap_err()
        ));
    }

    #[test]
    fn allowed_paths_match_exactly() {
        let t = TempTree::new("paths");
        let p = FileAccessPolicy::restricted(vec![], vec![t.s("other/c.bin")]);
        assert!(p.resolve_read_path(&t.s("other/c.bin")).is_ok());
        assert!(is_permission(
            &p.resolve_read_path(&t.s("allowed/a.bin")).unwrap_err()
        ));
    }

    #[test]
    fn open_regular_file_verifies_the_opened_handle() {
        use std::io::Read;
        let t = TempTree::new("open");
        let p = FileAccessPolicy::restricted(vec![t.s("allowed")], vec![]);
        let mut f = p.open_regular_file(&t.s("allowed/a.bin")).unwrap();
        let mut buf = Vec::new();
        f.read_to_end(&mut buf).unwrap();
        assert_eq!(buf, b"a");
        // Directories are not regular files, even inside an allowed directory.
        let err = p.open_regular_file(&t.s("allowed")).unwrap_err();
        assert!(err.message.contains("not a regular file"), "{err}");
        assert!(
            p.open_regular_file("  ")
                .unwrap_err()
                .message
                .contains("empty")
        );
        // Unrestricted: a directory or a missing file is still not a regular file.
        let open = FileAccessPolicy::unrestricted();
        assert!(open.open_regular_file(&t.s("allowed")).is_err());
        assert!(open.open_regular_file(&t.s("allowed/missing.bin")).is_err());
        assert!(open.open_regular_file(&t.s("other/c.bin")).is_ok());
    }

    #[test]
    fn windows_paths_compare_without_verbatim_prefix_and_case() {
        assert_eq!(
            windows_comparable_text(r"\\?\C:\Data\File.parquet"),
            r"C:\DATA\FILE.PARQUET"
        );
        assert_eq!(windows_comparable_text("C:/Data/x"), r"C:\DATA\X");
        assert_eq!(
            windows_comparable_text(r"\\?\UNC\Server\Share\x"),
            r"\\SERVER\SHARE\X"
        );
        assert_eq!(
            windows_comparable_text(r"\\server\share\x"),
            r"\\SERVER\SHARE\X"
        );
        // Component-wise prefix semantics survive the normalization.
        #[cfg(windows)]
        {
            let dir = comparable(Path::new(r"\\?\C:\Data"));
            assert!(comparable(Path::new(r"C:\DATA\x.parquet")).starts_with(&dir));
            assert!(!comparable(Path::new(r"C:\Data2\x.parquet")).starts_with(&dir));
        }
    }

    #[test]
    fn windows_case_folding_is_per_unit_and_never_wider_than_ntfs() {
        // W9-D-003: KELVIN SIGN, OHM SIGN, dotless i and long s stay distinct from the letters
        // full Unicode case mapping would fold them into.
        for (special, plain) in [
            ("\u{212A}", "k"),
            ("\u{2126}", "\u{3C9}"),
            ("\u{131}", "i"),
            ("\u{17F}", "s"),
        ] {
            assert_ne!(
                windows_comparable_text(&format!(r"C:\data\{special}\x")),
                windows_comparable_text(&format!(r"C:\data\{plain}\x")),
                "{special:?} vs {plain:?}"
            );
        }
        // One-to-many and supplementary mappings are not applied.
        assert_eq!(windows_comparable_text("\u{DF}"), "\u{DF}");
        assert_eq!(windows_comparable_text("\u{10428}"), "\u{10428}");
    }

    #[cfg(windows)]
    #[test]
    fn windows_comparison_matches_ordinal_ignore_case() {
        let eq = |a: &str, b: &str| paths_equal(Path::new(a), Path::new(b));
        // ASCII case differences are equal.
        assert!(eq(r"C:\Data\File.parquet", r"c:\DATA\file.PARQUET"));
        assert!(eq(r"\\?\UNC\Server\Share\x", r"\\server\SHARE\X"));
        // Accented letters are equal to their other case, as on NTFS.
        assert!(eq("C:\\caf\u{E9}\\x", "C:\\CAF\u{C9}\\X"));
        assert!(os_ordinal_equal_ignore_case('\u{E9}', '\u{C9}'));
        // KELVIN SIGN is neither 'k' nor 'K'.
        assert!(!eq("C:\\data\\\u{212A}", r"C:\data\k"));
        assert!(!eq("C:\\data\\\u{212A}", r"C:\data\K"));
        assert!(!os_ordinal_equal_ignore_case('\u{212A}', 'K'));

        // Prefix matching keeps the separator boundary under the same rule.
        let within = |a: &str, b: &str| path_is_within(Path::new(a), Path::new(b));
        assert!(within("C:\\CAF\u{C9}\\x.parquet", "\\\\?\\C:\\caf\u{E9}"));
        assert!(within(r"C:\DATA\sub\x.parquet", r"c:\data"));
        assert!(!within("C:\\caf\u{E9}2\\x.parquet", "C:\\caf\u{E9}"));
        assert!(!within(r"C:\data2\x.parquet", r"C:\data"));
        assert!(!within(r"C:\data\k\x.parquet", "C:\\data\\\u{212A}"));
    }

    #[cfg(windows)]
    #[test]
    fn windows_allow_list_is_case_insensitive() {
        let t = TempTree::new("case");
        let upper = t.s("allowed").to_uppercase();
        let p = FileAccessPolicy::restricted(vec![upper], vec![]);
        assert!(p.resolve_read_path(&t.s("allowed/a.bin")).is_ok());
        assert!(p.open_regular_file(&t.s("allowed/a.bin")).is_ok());
        assert!(is_permission(
            &p.resolve_read_path(&t.s("allowed2/b.bin")).unwrap_err()
        ));
    }

    #[cfg(unix)]
    #[test]
    fn symlink_out_of_allowed_directory_is_refused() {
        let t = TempTree::new("symlink");
        let link = t.root.join("allowed").join("link.bin");
        std::os::unix::fs::symlink(t.root.join("other").join("c.bin"), &link).unwrap();
        let p = FileAccessPolicy::restricted(vec![t.s("allowed")], vec![]);
        assert!(is_permission(
            &p.resolve_read_path(&link.to_string_lossy()).unwrap_err()
        ));
        assert!(is_permission(
            &p.open_regular_file(&link.to_string_lossy()).unwrap_err()
        ));
    }

    #[test]
    fn file_urls_map_like_duckdb() {
        // Unix offsets.
        assert_eq!(file_url_offset("file:/x/y", false), Some(5));
        assert_eq!(file_url_offset("file:///x/y", false), Some(7));
        assert_eq!(file_url_offset("file://localhost/x/y", false), Some(16));
        // Windows offsets also skip the '/' before the drive letter.
        assert_eq!(file_url_offset("file:/C:/x", true), Some(6));
        assert_eq!(file_url_offset("file:///C:/x", true), Some(8));
        assert_eq!(file_url_offset("file://localhost/C:/x", true), Some(17));
        // Not mapped: other hosts, no slash, other schemes, plain paths.
        assert_eq!(file_url_offset("file://server/x", false), None);
        assert_eq!(file_url_offset("file:x", false), None);
        assert_eq!(file_url_offset("s3://b/x", false), None);
        assert_eq!(file_url_offset("/plain/path", false), None);
        #[cfg(unix)]
        {
            assert_eq!(
                local_path_of_file_url("file:///data/a.parquet"),
                Some("/data/a.parquet")
            );
            assert_eq!(local_path_of_file_url("file:/data/a"), Some("/data/a"));
            assert_eq!(
                local_path_of_file_url("file://localhost/data/a"),
                Some("/data/a")
            );
        }
        #[cfg(windows)]
        assert_eq!(
            local_path_of_file_url("file:///C:/data/a"),
            Some("C:/data/a")
        );
    }

    #[test]
    fn file_url_edge_cases_match_duckdb() {
        // W4-C review of `local_path_of_file_url` against DuckDB's `GetFileUrlOffset`:
        // DuckDB does not percent-decode, is case-sensitive about the scheme, and needs the
        // slash after `localhost`.
        let local = |url: &'static str, windows: bool| {
            file_url_offset(url, windows).and_then(|o| url.get(o..))
        };
        assert_eq!(
            local("file:///a%20b.parquet", false),
            Some("/a%20b.parquet")
        );
        assert_eq!(local("file:///a%20b.parquet", true), Some("a%20b.parquet"));
        assert_eq!(local("FILE:///x", false), None);
        assert_eq!(local("File:///x", true), None);
        assert_eq!(local("file://localhost", false), None);
        assert_eq!(local("file://localhost/", false), Some("/"));
        assert_eq!(local("file://LOCALHOST/x", false), None);
        // `file:/` alone: DuckDB reads past the end as '\0', which is not '/'.
        assert_eq!(local("file:/", false), Some("/"));
        assert_eq!(local("file:/", true), Some(""));
        // Windows drive letters in every form, lower case too.
        assert_eq!(local("file:/c:/data/x", true), Some("c:/data/x"));
        assert_eq!(local("file:///c:/data/x", true), Some("c:/data/x"));
        assert_eq!(local("file://localhost/D:/x", true), Some("D:/x"));
        // Four slashes (a UNC-style URL) keep one leading slash, exactly as DuckDB does.
        assert_eq!(local("file:////srv/share/x", true), Some("/srv/share/x"));
        assert_eq!(local("file:////srv/share/x", false), Some("//srv/share/x"));
        // Non-ASCII after the prefix is sliced on a character boundary.
        assert_eq!(local("file:///données/x", false), Some("/données/x"));
    }

    #[test]
    fn path_is_within_is_component_wise() {
        let t = TempTree::new("within");
        let root = std::fs::canonicalize(&t.root).unwrap();
        assert!(path_is_within(&root.join("allowed").join("a.bin"), &root));
        assert!(path_is_within(&root, &root));
        let allowed = root.join("allowed");
        assert!(!path_is_within(&root.join("allowed2"), &allowed));
    }

    #[cfg(unix)]
    #[test]
    fn symlink_escape_through_file_url_is_refused() {
        // W2-D-001 item 2: the URL form gets the same canonical check as the plain path.
        let t = TempTree::new("fileurl");
        let link = t.root.join("allowed").join("link.bin");
        std::os::unix::fs::symlink(t.root.join("other").join("c.bin"), &link).unwrap();
        let p = FileAccessPolicy::restricted(vec![t.s("allowed")], vec![]);
        let url = format!("file://{}", link.to_string_lossy());
        let local = local_path_of_file_url(&url).expect("file:/// URL maps to a local path");
        assert!(is_permission(&p.open_regular_file(local).unwrap_err()));
        let ok_url = format!("file://{}", t.s("allowed/a.bin"));
        assert!(
            p.open_regular_file(local_path_of_file_url(&ok_url).unwrap())
                .is_ok()
        );
    }

    #[cfg(windows)]
    #[test]
    fn opened_handle_path_is_reported_on_windows() {
        // W2-D-001 item 1: the handle's own path, in the same form as `canonicalize`.
        let t = TempTree::new("handle");
        let file = t.root.join("other").join("c.bin");
        let f = File::open(&file).unwrap();
        let actual = opened_file_path(&f).expect("GetFinalPathNameByHandleW works on NTFS");
        assert!(paths_equal(&actual, &std::fs::canonicalize(&file).unwrap()));
        // Renaming is refused while a sandboxed open holds the file (no FILE_SHARE_DELETE).
        let p = FileAccessPolicy::restricted(vec![t.s("other")], vec![]);
        let pinned = p.open_regular_file(&file.to_string_lossy()).unwrap();
        assert!(std::fs::rename(&file, t.root.join("other").join("moved.bin")).is_err());
        drop(pinned);
        drop(f);
        // Outside the allowed directory, the handle path is refused.
        let p = FileAccessPolicy::restricted(vec![t.s("allowed")], vec![]);
        assert!(is_permission(
            &p.open_regular_file(&file.to_string_lossy()).unwrap_err()
        ));
    }

    #[cfg(any(target_os = "linux", target_os = "android"))]
    #[test]
    fn opened_descriptor_path_is_reported_on_linux() {
        let t = TempTree::new("procfd");
        let f = File::open(t.root.join("other").join("c.bin")).unwrap();
        let actual = opened_file_path(&f).expect("/proc/self/fd is available");
        assert_eq!(
            actual,
            std::fs::canonicalize(t.root.join("other").join("c.bin")).unwrap()
        );
    }
}

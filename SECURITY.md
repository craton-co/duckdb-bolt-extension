# Security Policy

## Supported Versions

Only the latest release on the `0.1.x` line receives security updates.

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues, discussions, or pull requests.**

Please report suspected vulnerabilities privately through one of the following channels:

1. **GitHub Private Vulnerability Reporting**: Open a private advisory at
   <https://github.com/craton-co/duckdb-bolt-extension/security/advisories/new>.
2. **Email**: Contact the Craton Software Company security team at
   **`security@craton.com.ar`** with the subject line `[SECURITY] duckdb-bolt-extension: <brief summary>`.

Please include:
- A description of the vulnerability and its potential impact.
- Minimal reproduction steps or SQL query / Rust snippet.
- DuckDB version, Craton Bolt version, operating system, Rust toolchain version, and GPU driver / architecture if applicable.

### Response Timeline

- **Acknowledgement**: Within **48 hours** of receipt.
- **Initial Assessment & Triage**: Within **5 business days**.
- **Remediation & Coordinated Disclosure**: Within **30–90 days** depending on severity.

## Security Considerations for DuckDB + GPU Extensions

1. **File access (`bolt_load_parquet`, `bolt_load_arrow_ipc`)**:
   Local paths are read with the Rust standard library, so the extension re-implements DuckDB's
   `enable_external_access`, `allowed_directories` and `allowed_paths` checks on **canonical**
   paths (`crates/core/src/sandbox.rs`): symlinks and `..` cannot escape an allowed directory, and
   allow-list entries that cannot be canonicalized never match. On Windows both sides are compared
   case-insensitively with a per-UTF-16-unit upcase that never merges two names NTFS's ordinal
   comparison keeps apart (so, for example, a KELVIN SIGN in an allowed directory does not also
   admit its ASCII-`K` sibling). After opening, the opened handle
   is checked again (Linux / Android: `/proc/self/fd`; Windows: `GetFinalPathNameByHandleW`, and a
   sandboxed load denies deletion and rename while it reads; macOS / BSD: re-canonicalization plus
   device / inode identity). Every local load is also opened once through DuckDB's file system, so
   `disabled_filesystems` and DuckDB's own checks apply. `file:` URLs that DuckDB maps to local
   paths are checked like local paths; other URLs (`s3://`, `https://`, ...) are opened only
   through DuckDB's file system (httpfs, secrets, DuckDB's lexical sandbox rules). By design, a
   hard link inside an allowed directory is accepted (as DuckDB does; only the OS can prevent
   creating one, e.g. `fs.protected_hardlinks = 1` on Linux).
2. **Hostile input files**:
   Arrow IPC files are validated before `arrow-ipc` decodes them (blocks inside the file, buffers
   inside their block, the declared uncompressed size of LZ4 / ZSTD buffers bounded by a ratio and
   by DuckDB's `memory_limit`), and panics in the Parquet / IPC decoders are contained, so a
   corrupt file is an error rather than a crash or a huge allocation. Remote Parquet files are read
   with ranged reads (footer, then the column chunks being decoded); footer metadata above
   `memory_limit` is refused, and every column chunk to be decoded must lie inside the file and fit
   in `memory_limit` (local files too). Page-header sizes inside a chunk are not yet bounded before
   `parquet` allocates for them
   ([`docs/issues/W7-A-001`](docs/issues/W7-A-001-parquet-page-header-sizes-unbounded.md)).
3. **PTX disk cache (`SET bolt_ptx_cache_dir`)**:
   Cached kernels are loaded into the CUDA driver, so the directory is code, not data. The setting
   is refused while `enable_external_access = false`, must be an absolute path without `..`
   components (a missing directory under `..` would otherwise be checked at the wrong ancestor),
   and an existing
   path must be a directory. On Unix the directory (or, if it does not exist yet, its nearest
   existing ancestor) must be owned by the effective user or root and not be group- or
   world-writable, and every parent must be owned by the user or root and not be world-writable
   unless it has the sticky bit; when the effective uid cannot be determined, only root-owned
   directories are accepted. On Windows the canonical directory must lie inside `%USERPROFILE%`,
   `%LOCALAPPDATA%` or `%APPDATA%` (and not in a shared profile directory). The directory is
   re-validated before every engine build and during Bolt calls; a directory that becomes unsafe is
   disabled. When the option is empty, craton-bolt's `BOLT_PTX_CACHE_DIR` environment variable must
   pass the same checks or the engine is not built. Cached PTX is not integrity-checked
   ([`docs/issues/W2-D-003`](docs/issues/W2-D-003-ptx-cache-dir-residual-hardening.md)).
4. **GPU memory (`SET bolt_memory_budget`)**:
   The extension enforces the budget for the tables it uploads (a table that would exceed it is
   served by the host fallback), but not for query intermediates. In multi-tenant or shared-GPU
   deployments, also limit who can upload data (`COPY ... (FORMAT bolt)`, `bolt_load_*`,
   `bolt_register_sql`).
5. **Embedded host-fallback DuckDB**:
   Queries the GPU engine does not run execute on a separate in-memory DuckDB instance opened with
   `enable_external_access=false`, extension auto-loading/-installing and community extensions
   disabled, at most 4 threads and `memory_limit=25%`, and `lock_configuration=true`; it accepts a
   single `SELECT`. Bolt functions inside such a query are refused. A running query there can be
   interrupted only by `PRAGMA bolt_request_timeout`, not by the host's interrupt
   ([`docs/issues/W2-E-005`](docs/issues/W2-E-005-bolt-requests-not-interruptible.md)).
6. **Process-wide state**:
   Registered tables, `bolt_*` settings and `PRAGMA bolt_request_timeout` are shared by every
   DuckDB database and connection in the process. Any connection can read, replace or drop any
   Bolt table, and a `SET bolt_*` in one database affects all of them. A database with
   `lock_configuration = true` can change none of the runtime configuration: DuckDB refuses
   `SET bolt_*`, and the extension refuses `PRAGMA bolt_reset` / `bolt_request_timeout`.
7. **Unaligned FFI memory & unsafe boundaries**:
   DuckDB's C API v2 aligns 128-bit vectors (`HUGEINT` / `DECIMAL(19..38, s)`) to 8 bytes, Rust's
   `i128` needs 16. All 128-bit reads and writes in `crates/core/src/arrow_bridge.rs` use
   `read_unaligned` / `write_unaligned` with bounds from the vector lengths. Arrow buffers imported
   from DuckDB are aligned and validated before use. Every `unsafe` block carries a `// SAFETY:`
   comment, including the `register_static_extension` calls in the test suites.
8. **Panics and engine crashes**:
   A panic inside a DuckDB callback becomes a DuckDB `INTERNAL` error, which invalidates the
   database instance. Worker-thread panics are caught and the worker keeps serving. craton-bolt
   0.9.0 can crash the whole process on some queries over NULLs and on very deep predicates; the
   engine guard routes those shapes to the host fallback
   ([`docs/issues/W2-A-001`](docs/issues/W2-A-001-gpu-ignores-null-validity.md),
   [`docs/upstream/`](docs/upstream/README.md)). Report any other way to crash the process or
   invalidate the database as a security bug.

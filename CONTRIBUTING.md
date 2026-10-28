# Contributing to Craton Bolt DuckDB Extension

Thank you for your interest in contributing to **Craton Bolt DuckDB Extension** (`duckdb-bolt-extension`), developed and maintained by [Craton Software Company](https://craton.com.ar).

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

---

## Development Setup

### Prerequisites

- **Rust Toolchain**: `1.88` or newer (Edition 2024; required by `duckdb-neo` and by let-chains in the code).
- **Python 3**: `3.9+` (used by `scripts/package_extension.py` and DuckDB metadata packaging; 3.11+ reads `Cargo.toml` with `tomllib`, older versions fall back to a regex).
- **CUDA Toolkit** *(optional)*: CUDA `12.x` or `13.x` for native GPU execution. Without an NVIDIA GPU or CUDA toolkit, build with `--features cuda-stub` (or `BOLT_CUDA_STUB=1` for the Makefile).

### Workspace layout & feature exclusivity

| Crate | Path | Role |
| :--- | :--- | :--- |
| `duckdb-bolt-core` | `crates/core` | All logic and all tests; the workspace `default-members`. |
| `duckdb-bolt-extension` | `crates/native` | Loadable cdylib (`publish = false`). |
| `duckdb-bolt-wasm` | `crates/wasm` | Wasm entrypoint. |

`duckdb-neo` has two mutually exclusive features: `build-extension` (native and wasm crates) and
`build-extension-static` (the core crate's dev-dependency, so tests run an in-process DuckDB).
Cargo unifies features across a `--workspace` build, so **never use `--workspace`**; select the
crate with `-p`:

```bash
# All unit, integration and SQLLogicTest tests, CPU stub (no GPU or CUDA toolkit needed)
cargo test -p duckdb-bolt-core --features cuda-stub

# The same on a GPU host (tests then expect backend cuda-gpu where they assert a backend)
cargo test -p duckdb-bolt-core

# Native loadable library + packaging with DuckDB C_STRUCT_UNSTABLE metadata
cargo build --release -p duckdb-bolt-extension
python scripts/package_extension.py --profile release

# Wasm crate (compile check, like CI)
cargo check -p duckdb-bolt-wasm --target wasm32-unknown-emscripten
```

`scripts/package_extension.py` also accepts `--target <triple>` (and picks the DuckDB platform
from it), `--platform`, `--target-dir` (default: `$CARGO_TARGET_DIR`, else `target/`), and
`--suffix <name>`, which writes to `build/<profile>-<name>/` so builds from different worktrees or
configurations do not overwrite each other. Keep the file name `bolt.duckdb_extension`
(`--output-name` exists, but DuckDB derives the extension name and its `bolt_init_c_api_v2`
entrypoint from the file name, so a renamed file does not `LOAD` as `bolt`).

`make configure release` (the DuckDB extension-ci-tools path) enables `cuda-stub` and prints a
warning when neither `CUDA_PATH` nor `/usr/local/cuda` exists; pass `BOLT_CUDA_STUB=0` to force a
GPU build. `make test_debug` / `make test_release` run `test/sql/*.test` through DuckDB's own
SQLLogicTest runner.

---

## Coding Standards

1. **SPDX License Headers**:
   Every source file (`.rs`, `.py`, `.sh`, `Cargo.toml`, `Makefile`) must begin with:
   ```rust
   // SPDX-License-Identifier: Apache-2.0
   ```
   New pages in `docs/issues/` and `docs/upstream/` start with `<!-- SPDX-License-Identifier: Apache-2.0 -->`.
2. **Formatting & Linting**:
   All pull requests must pass `rustfmt` and `clippy` with zero warnings. Use the CI commands
   (`.github/workflows/RustCI.yml`):
   ```bash
   cargo fmt --all -- --check
   cargo clippy --locked -p duckdb-bolt-core --features cuda-stub --all-targets -- -D warnings
   cargo clippy --locked -p duckdb-bolt-extension --features cuda-stub -- -D warnings
   cargo deny --all-features check   # licenses, advisories, bans, sources (deny.toml)
   ```
3. **Safety (`unsafe`)**:
   Every `unsafe` block, including in tests, must be preceded by a `// SAFETY:` comment explaining
   the invariants (pointer alignment and bounds across DuckDB C API v2 buffers in particular).
4. **No panics in DuckDB callbacks**: a panic in a callback becomes a DuckDB `INTERNAL` error,
   which invalidates the database. Return errors instead; the worker thread catches its own panics.
5. **Error messages** name the function or statement and the table, and do not contain issue ids.

---

## Testing

- New features and bug fixes must include tests: Rust integration tests in `crates/core/tests/`
  and/or SQLLogicTest files in `test/sql/`.
- `crates/core/tests/sqllogictest_and_loadable.rs` runs **every** `test/sql/*.test` file on a
  fresh in-memory database. It supports `statement ok|error|maybe` (with an expected
  error-message substring after `----`), `query` with `nosort` / `rowsort` / `valuesort`,
  `require`, `require-env`, `mode skip|unskip`, and `halt`. DuckDB's full runner (`make test`, CI
  job *SQLLogicTest*) runs the same files.
- Regression suites are named after the wave and work stream that added them
  (`w1a_pushdown_regression.rs` ... `w10a_lock_startup.rs`); SLT files that cover one feature are
  named after it (`bolt_call.test`, `bolt_qualifiers_and_lock.test`, ...).
- **Differential fuzzer** (`crates/core/tests/w4b_differential_fuzz.rs`): random tables (types,
  NULL density, special values, appended batches) and queries, compared between DuckDB, `bolt_scan`,
  the replacement scan and `bolt_query` with `EXCEPT ALL`. A route-consistency oracle also runs
  `bolt_query` on a NULL-free table and on a twin with an unused, partly NULL column (so the guard
  sends it to the host fallback) and requires the same types and values. It runs in `cargo test`
  with 2,400 cases and a fixed seed. Environment variables: `BOLT_FUZZ_CASES` (number of queries),
  `BOLT_FUZZ_SEED` (base seed), `BOLT_FUZZ_TABLE_SEEDS=0xa,0xb` (replay only these table seeds from
  a report), `BOLT_FUZZ_VERBOSE` (print every failing case before minimizing),
  `BOLT_FUZZ_ROUTE_EVERY` (run the route oracle on every n-th query; default 3, `0` = off),
  `BOLT_FUZZ_STRICT_TYPES` (fail, instead of only counting, when a GPU-route `bolt_query` result
  type differs from DuckDB's) and `BOLT_FUZZ_WIDE_LIKE` (opt-in tables above the guard's
  4,096-value dictionary limit; a failure there can abort the process). Known, tracked differences
  go in its `known_issue` function with the page that tracks them (none are listed today).
- The Bolt catalog and `bolt_*` settings are **process-wide**, and DuckDB's runner executes all
  files in one process: start (and end) every SLT file with `PRAGMA bolt_clear_tables;`, use table
  names unique to the file, and restore any `SET bolt_*` or `PRAGMA bolt_request_timeout` you
  change. Rust test files that touch the runtime serialize their tests on a file-level mutex.
- Tests must pass on both backends (`cuda-gpu` and `host-planner-fallback`). Assert per backend
  (`SELECT backend FROM bolt_pool_stats()`) only where behavior legitimately differs. CI has no GPU,
  so run `cargo test -p duckdb-bolt-core` on a GPU host before a release.
- Behavior that is wrong today is filed under [`docs/issues/`](docs/issues/README.md). If a test
  has to tolerate it, mark the spot with `# KNOWN-BUG (<issue id>)` (SLT) or `// KNOWN-BUG` (Rust)
  and list it on the issue page ([W1-E-013](docs/issues/W1-E-013-tests-coupled-to-known-bugs.md)),
  so whoever fixes the bug tightens the test.
- Coverage: aim for at least **85% line coverage** of `duckdb-bolt-core`
  (`cargo llvm-cov -p duckdb-bolt-core --summary-only`). CI does not measure coverage, and
  `cuda-stub` builds do not exercise the GPU paths, so the gate is checked manually on a GPU host.

### CI jobs (`.github/workflows/RustCI.yml`)

| Job | What |
| :--- | :--- |
| `fmt` | `cargo fmt --all -- --check` |
| `check` (Linux, Windows, macOS) | clippy for core (all targets) and the extension, `cargo test --locked -p duckdb-bolt-core --features cuda-stub`, `cargo package --list` (Linux) |
| `wasm` | `cargo check` and clippy of `duckdb-bolt-wasm` for `wasm32-unknown-emscripten` |
| `sqllogictest` | `make configure debug test_debug` with `BOLT_CUDA_STUB=1` (DuckDB's runner) |
| `msrv` | Rust 1.88: core tests and an extension check |
| `deny` | `cargo deny --all-features check` |

`MainDistributionPipeline.yml` builds the distribution binaries with DuckDB's extension-ci-tools
(CPU-only: the runners have no CUDA toolkit).

---

## Issues, Governance & Releases

- Defects and proposals: one page per item in [`docs/issues/`](docs/issues/README.md) (naming,
  header format and statuses are described there); reference the page in your pull request and
  update its `Status` when you fix it. Engine defects that need a craton-bolt fix get a report in
  [`docs/upstream/`](docs/upstream/README.md) and status `blocked-upstream`.
- Maintainers and review process: [`MAINTAINERS.md`](MAINTAINERS.md) (code owners: [`CODEOWNERS`](CODEOWNERS)).
- Release checklist: [`RELEASING.md`](RELEASING.md). Only `duckdb-bolt-core` is published to
  crates.io; the loadable `duckdb-bolt-extension` crate is `publish = false`.
- Dependency policy: [`deny.toml`](deny.toml).

## Developer Certificate of Origin (DCO) & Licensing

All contributions are licensed under the [Apache License, Version 2.0](LICENSE). The certificate text is in [`DCO`](DCO). Sign off your commits (`git commit -s`) to certify that you have the right to submit the contribution under Apache-2.0:

```text
Signed-off-by: Your Name <your.email@example.com>
```

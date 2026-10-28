<!-- SPDX-License-Identifier: Apache-2.0 -->

# Releasing duckdb-bolt-extension

This document is for maintainers (see [`MAINTAINERS.md`](MAINTAINERS.md)). It lists the steps to
cut a release: bump the version, update the changelog, tag, build and package the loadable
extension for every platform, publish checksums, and (optionally) publish `duckdb-bolt-core` to
crates.io. Only `duckdb-bolt-core` is published to crates.io; `duckdb-bolt-extension`
(`publish = false`) is distributed as `bolt.duckdb_extension` binaries.

---

## 1. Pre-release checklist

- [ ] All CI workflows are green on `main` (`Rust CI`, `Main Extension Distribution Pipeline`).
- [ ] No open `critical` / `high` pages in [`docs/issues/`](docs/issues/README.md), or each one is
      listed under *Known issues* in the release notes.
- [ ] `Cargo.toml`: bump `[workspace.package] version` (all three crates inherit it) and the
      `duckdb_bolt_core` entry in `[workspace.dependencies]`. Run `cargo update -w` so
      `Cargo.lock` picks up the new version.
- [ ] `scripts/package_extension.py`: `DEFAULT_EXTENSION_VERSION` (only used when `Cargo.toml`
      cannot be read) matches the new version.
- [ ] `CHANGELOG.md`: rename `[Unreleased]` to `[X.Y.Z] - YYYY-MM-DD`; every entry sits under
      `Added`, `Changed`, `Fixed`, `Removed`, `Performance`, or `Security`; add a new empty
      `[Unreleased]`.
- [ ] `SECURITY.md`: the *Supported Versions* table lists the new release line.
- [ ] Targeting a new DuckDB version? Run `./upgrade.sh NEO_VERSION MACROS_VERSION`, update the
      `extension-ci-tools` submodule, and pin the workflow ref / `ci_tools_version` to the new
      submodule commit (or release tag). A binary only loads in the exact DuckDB version it was
      built for (`C_STRUCT_UNSTABLE`).
- [ ] Local checks (CPU stub and, on a CUDA host, the GPU build):

  ```bash
  cargo fmt --all -- --check
  cargo clippy --locked -p duckdb-bolt-core --features cuda-stub --all-targets -- -D warnings
  cargo test --locked -p duckdb-bolt-core --features cuda-stub
  cargo test --locked -p duckdb-bolt-core                # GPU host: expects backend cuda-gpu
  cargo deny --all-features check
  cargo package --locked -p duckdb-bolt-core --list      # LICENSE, NOTICE, README, src/ only
  cargo publish --dry-run -p duckdb-bolt-core
  ```

## 2. Tagging

**Credentials needed:** push access to `origin`.

```bash
git tag -s vX.Y.Z -m "duckdb-bolt-extension vX.Y.Z"
git push origin vX.Y.Z
```

Tag only a commit that is already green on `main`. The workflows do not trigger on tags; the tag
names the commit the binaries below are built from.

## 3. Building the binaries

The DuckDB distribution pipeline builds every platform, but its runners have no CUDA toolkit, so
those binaries are **CPU-only** (`cuda-stub`; the Makefile prints a warning when it enables it).
GPU binaries must be built on a host with a CUDA toolkit (`CUDA_PATH` set or `/usr/local/cuda`
present).

Per platform (`linux_amd64`, `linux_arm64`, `osx_amd64`, `osx_arm64`, `windows_amd64`), from a
clean checkout of the tag:

```bash
git submodule update --init --recursive
make configure
make release                      # -> build/release/bolt.duckdb_extension (with metadata)
make test_release                 # DuckDB's unittest runner over test/sql/*.test
```

Without `make` (for example on Windows), build with Cargo and package with the script:

```bash
cargo build --locked --release -p duckdb-bolt-extension            # add --features cuda-stub for CPU-only
python scripts/package_extension.py --profile release              # honors CARGO_TARGET_DIR
# Cross targets: --target <triple> selects the library path and the DuckDB platform metadata.
# Several configurations side by side: --suffix gpu / --suffix cpu -> build/release-<suffix>/
```

Keep the file name `bolt.duckdb_extension`: DuckDB derives the extension name from it. Put
platform and backend into the release asset's directory or archive name instead, e.g.
`bolt-vX.Y.Z-duckdb-v2.0.0-linux_amd64-gpu/bolt.duckdb_extension`.

Smoke-test every binary in the matching DuckDB CLI:

```sql
-- duckdb -unsigned
LOAD 'build/release/bolt.duckdb_extension';
SELECT bolt_version();
SELECT backend FROM bolt_pool_stats();   -- cuda-gpu for GPU builds on a GPU host
```

## 4. Checksums and signatures

Publish a SHA-256 for every asset and sign the checksum file:

```bash
sha256sum bolt-*/bolt.duckdb_extension > SHA256SUMS          # Linux
shasum -a 256 bolt-*/bolt.duckdb_extension > SHA256SUMS      # macOS
# Windows (PowerShell): Get-FileHash -Algorithm SHA256 <file>
gpg --armor --detach-sign SHA256SUMS
```

Locally built binaries are unsigned in DuckDB's sense (users need `-unsigned` /
`allow_unsigned_extensions`); DuckDB-signed binaries require publishing through the
community-extensions repository (ROADMAP RFC-010).

## 5. GitHub release

Create a release for the tag with the `CHANGELOG.md` section as notes, a *Known issues* list
(open `critical` / `high` pages from `docs/issues/`), the binaries, `SHA256SUMS`, and
`SHA256SUMS.asc`. State the exact DuckDB version and which assets are GPU vs CPU-only.

## 6. crates.io (duckdb-bolt-core only)

**Credentials needed:** a crates.io token with publish rights on `duckdb-bolt-core`.

```bash
cargo publish --locked -p duckdb-bolt-core
```

docs.rs builds with `features = ["cuda-stub"]` (`[package.metadata.docs.rs]`); check
<https://docs.rs/crate/duckdb-bolt-core/latest> after publishing.

## 7. After the release

- Re-open `[Unreleased]` in `CHANGELOG.md` if not done in step 1.
- Retire `fixed` pages from `docs/issues/` (see its README).

<!-- SPDX-License-Identifier: Apache-2.0 -->

## Description
Provide a brief summary of the changes in this pull request and the problem they solve.

## Related Issues / Proposals
Closes #(issue number) or relates to RFC #(RFC id from docs/ROADMAP.md).

## Type of Change
- [ ] Bug fix (non-breaking change fixing an issue)
- [ ] New feature (non-breaking change adding functionality)
- [ ] Performance improvement
- [ ] Breaking change (fix or feature causing existing behavior to change)
- [ ] Documentation update

## Checklist
Before submitting this pull request, please verify:

- [ ] **Developer Certificate of Origin (DCO)**: Every commit is signed off with `git commit -s` as required by [CONTRIBUTING.md](CONTRIBUTING.md).
- [ ] **SPDX Headers**: New source files begin with `// SPDX-License-Identifier: Apache-2.0` (or markdown comment).
- [ ] **Format & Lint**: Code passes `cargo fmt --all -- --check` and `cargo clippy --locked -p duckdb-bolt-core --features cuda-stub --all-targets -- -D warnings`.
- [ ] **Tests Added**: New functionality or bug fixes include regression tests in `crates/core/tests/` and/or `test/sql/*.test`.
- [ ] **All Tests Pass**: Ran `cargo test -p duckdb-bolt-core --features cuda-stub` (and `cargo test -p duckdb-bolt-core` if on a CUDA GPU).
- [ ] **Safety Comments**: Any new `unsafe` blocks are preceded by an explicit `// SAFETY:` invariant comment.
- [ ] **Documentation**: Updated relevant documentation (`docs/SQL_REFERENCE.md`, `docs/ARCHITECTURE.md`, `docs/ROADMAP.md`, `README.md`) if public behavior changed.

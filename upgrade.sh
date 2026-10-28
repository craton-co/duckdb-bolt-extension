#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

usage() {
    echo "Usage: $0 NEO_VERSION MACROS_VERSION (e.g. $0 0.20000.0 1.20000.0)" >&2
    exit 1
}

VERSION_PATTERN='^(0|[1-9][0-9]*)\.([1-9][0-9]{4,})\.(0|[1-9][0-9]*)$'
if [ "$#" -ne 2 ] || [[ ! "$1" =~ $VERSION_PATTERN ]] || [[ ! "$2" =~ $VERSION_PATTERN ]]; then
    usage
fi

NEO_VERSION=$1
MACROS_VERSION=$2
IFS=. read -r _ ENCODED_VERSION _ <<<"$NEO_VERSION"
IFS=. read -r _ MACROS_ENCODED_VERSION _ <<<"$MACROS_VERSION"
if [[ "$ENCODED_VERSION" != "$MACROS_ENCODED_VERSION" ]] || ((ENCODED_VERSION < 20000)); then
    echo "Both crates must target the same DuckDB version (2.0.0 or newer)." >&2
    usage
fi
printf -v API_VERSION '%d.%d.%d' \
    "$((ENCODED_VERSION / 10000))" "$((ENCODED_VERSION / 100 % 100))" "$((ENCODED_VERSION % 100))"
DUCKDB_VERSION=v$API_VERSION

cd "$(dirname "$0")"

sed_inplace() {
    if sed --version 2>/dev/null | grep -q GNU; then
        sed -i "$@"
    else
        sed -i '' "$@"
    fi
}

sed_inplace -E "s!^TARGET_DUCKDB_VERSION=.*!TARGET_DUCKDB_VERSION=$DUCKDB_VERSION!" Makefile
sed_inplace -E "s!^DEFAULT_DUCKDB_VERSION = \"[^\"]+\"!DEFAULT_DUCKDB_VERSION = \"$DUCKDB_VERSION\"!" \
    scripts/package_extension.py
sed_inplace -E "/duckdb_version:/s!v[0-9]+\.[0-9]+\.[0-9]+!$DUCKDB_VERSION!g" \
    .github/workflows/MainDistributionPipeline.yml
# The SQLLogicTest job installs exactly this DuckDB release from PyPI (no leading "v").
sed_inplace -E "s!^([[:space:]]*DUCKDB_TEST_VERSION:[[:space:]]*)\"[^\"]*\"!\1\"$API_VERSION\"!" \
    .github/workflows/RustCI.yml
sed_inplace -E "/^duckdb-neo[[:space:]]*=/s!\"~?[0-9]+\.[0-9]+\.[0-9]+\"!\"~$NEO_VERSION\"!" Cargo.toml
sed_inplace -E "/^duckdb-loadable-macros[[:space:]]*=/s!\"~?[0-9]+\.[0-9]+\.[0-9]+\"!\"~$MACROS_VERSION\"!" Cargo.toml
# Every static entrypoint must request the same C API level as the shipped binary, including the
# ones the integration tests declare in crates/core/tests/*.rs (W2-E-014).
shopt -s nullglob
ENTRYPOINT_FILES=(crates/native/src/lib.rs crates/wasm/src/lib.rs crates/core/tests/*.rs)
shopt -u nullglob
sed_inplace -E "/duckdb_entrypoint_c_api/s!api_version = \"[^\"]+\"!api_version = \"$API_VERSION\"!" \
    "${ENTRYPOINT_FILES[@]}"
# Fail loudly if an entrypoint was written in a shape the sed above does not match (for example
# with the attribute split over several lines).
STALE_ENTRYPOINTS=$(grep -nE 'api_version = "' "${ENTRYPOINT_FILES[@]}" | grep -vF "api_version = \"$API_VERSION\"" || true)
if [ -n "$STALE_ENTRYPOINTS" ]; then
    echo "Error: these entrypoints still request another api_version; update them by hand:" >&2
    echo "$STALE_ENTRYPOINTS" >&2
    exit 1
fi

echo "Updated duckdb-neo to $NEO_VERSION and duckdb-loadable-macros to $MACROS_VERSION / DuckDB $DUCKDB_VERSION"
echo "Note: README.md, docs/ and test/sql/*.test may still name the previous DuckDB version; review"
echo "      git grep -n 'v[0-9]*\.[0-9]*\.[0-9]' -- README.md docs test/sql"
echo "Note: run cargo update to refresh Cargo.lock; dependencies were not resolved."
echo "Note: update the extension-ci-tools submodule, then set the workflow @ref and ci_tools_version in"
echo "      .github/workflows/MainDistributionPipeline.yml to the new submodule commit SHA (or release tag)."

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Package debug and/or release Craton Bolt DuckDB loadable extension binaries with metadata.

The built shared library is looked up in Cargo's target directory (``--target-dir``, else
``CARGO_TARGET_DIR``, else ``<repo>/target``). Output goes to
``build/<profile>[-<suffix>]/<output-name>`` plus a copy in ``.../extension/bolt/``.
"""

from __future__ import annotations

import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # older Pythons fall back to a regex
    tomllib = None  # type: ignore[assignment]

EXTENSION_NAME = "bolt"
DEFAULT_DUCKDB_VERSION = "v2.0.0"
DEFAULT_EXTENSION_VERSION = "v0.1.0"
DEFAULT_OUTPUT_NAME = f"{EXTENSION_NAME}.duckdb_extension"
ABI_TYPE = "C_STRUCT_UNSTABLE"

# Rust target triple -> (DuckDB platform, shared library file name).
TARGET_TRIPLES = {
    "x86_64-unknown-linux-gnu": ("linux_amd64", f"lib{EXTENSION_NAME}.so"),
    "aarch64-unknown-linux-gnu": ("linux_arm64", f"lib{EXTENSION_NAME}.so"),
    "x86_64-apple-darwin": ("osx_amd64", f"lib{EXTENSION_NAME}.dylib"),
    "aarch64-apple-darwin": ("osx_arm64", f"lib{EXTENSION_NAME}.dylib"),
    "x86_64-pc-windows-msvc": ("windows_amd64", f"{EXTENSION_NAME}.dll"),
    "aarch64-pc-windows-msvc": ("windows_arm64", f"{EXTENSION_NAME}.dll"),
}


def _version_from_cargo_toml(text: str) -> str | None:
    if tomllib is not None:
        data = tomllib.loads(text)
        version = data.get("workspace", {}).get("package", {}).get("version")
        if version is None:
            version = data.get("package", {}).get("version")
        return version if isinstance(version, str) else None
    # Fallback for Python < 3.11: first `version = "..."` line at the start of a line.
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    return m.group(1) if m else None


def detect_versions(repo_root: Path) -> tuple[str, str]:
    ext_version = DEFAULT_EXTENSION_VERSION
    cargo_toml = repo_root / "Cargo.toml"
    if cargo_toml.exists():
        v = _version_from_cargo_toml(cargo_toml.read_text(encoding="utf-8"))
        if v:
            ext_version = v if v.startswith("v") else f"v{v}"

    duckdb_version = DEFAULT_DUCKDB_VERSION
    makefile = repo_root / "Makefile"
    if makefile.exists():
        m = re.search(
            r"^TARGET_DUCKDB_VERSION\s*=\s*(\S+)",
            makefile.read_text(encoding="utf-8"),
            re.MULTILINE,
        )
        if m:
            duckdb_version = m.group(1)

    return ext_version, duckdb_version


def detect_duckdb_platform() -> str:
    sys_name = platform.system().lower()
    machine = platform.machine().lower()
    is_arm = "arm" in machine or "aarch64" in machine
    if sys_name == "windows":
        return "windows_arm64" if is_arm else "windows_amd64"
    if sys_name == "darwin":
        return "osx_arm64" if is_arm else "osx_amd64"
    if sys_name == "linux":
        return "linux_arm64" if is_arm else "linux_amd64"
    raise RuntimeError(f"Unsupported OS/architecture: {sys_name}/{machine}")


def shared_lib_filename() -> str:
    sys_name = platform.system().lower()
    if sys_name == "windows":
        return f"{EXTENSION_NAME}.dll"
    if sys_name == "darwin":
        return f"lib{EXTENSION_NAME}.dylib"
    return f"lib{EXTENSION_NAME}.so"


def resolve_target_dir(repo_root: Path, override: str | None) -> Path:
    """Cargo's target directory: --target-dir, then CARGO_TARGET_DIR, then <repo>/target.

    Relative values are resolved against the current working directory, as Cargo does.
    """
    value = override or os.environ.get("CARGO_TARGET_DIR")
    if value:
        return Path(value).resolve()
    return repo_root / "target"


def library_path(target_dir: Path, profile: str, target_triple: str | None) -> Path:
    if target_triple:
        lib_name = TARGET_TRIPLES.get(target_triple, ("", shared_lib_filename()))[1]
        return target_dir / target_triple / profile / lib_name
    return target_dir / profile / shared_lib_filename()


def package_profile(
    repo_root: Path,
    profile: str,
    duckdb_platform: str,
    ext_version: str,
    duckdb_version: str,
    target_triple: str | None = None,
    target_dir: Path | None = None,
    output_name: str = DEFAULT_OUTPUT_NAME,
    suffix: str | None = None,
) -> Path:
    target_dir = target_dir or (repo_root / "target")
    lib_path = library_path(target_dir, profile, target_triple)
    if not lib_path.exists():
        raise FileNotFoundError(f"Shared library not found at {lib_path}; run cargo build first.")

    out_dir = repo_root / "build" / (f"{profile}-{suffix}" if suffix else profile)
    ext_dir = out_dir / "extension" / EXTENSION_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    ext_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / output_name
    append_script = repo_root / "extension-ci-tools" / "scripts" / "append_extension_metadata.py"

    cmd = [
        sys.executable,
        str(append_script),
        "-l",
        str(lib_path),
        "-o",
        str(out_file),
        "-n",
        EXTENSION_NAME,
        "-dv",
        duckdb_version,
        "-ev",
        ext_version,
        "-p",
        duckdb_platform,
        "--abi-type",
        ABI_TYPE,
    ]
    subprocess.run(cmd, check=True)
    shutil.copyfile(out_file, ext_dir / output_name)
    return out_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Package bolt.duckdb_extension binaries")
    parser.add_argument(
        "--profile",
        choices=["debug", "release", "both"],
        default="both",
        help="Cargo build profile(s) to package",
    )
    parser.add_argument(
        "--target",
        default=None,
        help="Optional Rust target triple subdirectory under the target directory "
        "(also selects the DuckDB platform and library file name for known triples)",
    )
    parser.add_argument(
        "--target-dir",
        default=None,
        help="Cargo target directory (default: $CARGO_TARGET_DIR, else <repo>/target)",
    )
    parser.add_argument(
        "--platform",
        default=None,
        help="DuckDB platform written into the metadata (default: from --target, else the host)",
    )
    parser.add_argument(
        "--suffix",
        default=None,
        help="Write to build/<profile>-<suffix>/ instead of build/<profile>/ so binaries built "
        "from different worktrees or configurations do not overwrite each other",
    )
    parser.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_NAME,
        help=f"Output file name (default: {DEFAULT_OUTPUT_NAME})",
    )
    args = parser.parse_args()

    if args.suffix is not None and not re.fullmatch(r"[A-Za-z0-9._-]+", args.suffix):
        parser.error("--suffix may only contain letters, digits, '.', '_' and '-'")
    if not args.output_name or os.path.basename(args.output_name) != args.output_name:
        parser.error("--output-name must be a plain file name; use --suffix to change the directory")
    if not args.output_name.startswith(f"{EXTENSION_NAME}."):
        # DuckDB derives the extension name (and its `<name>_init_c_api_v2` entrypoint) from the
        # file name, so a renamed file cannot be LOADed as `bolt`.
        print(
            f"warning: DuckDB derives the extension name from the file name; "
            f"'{args.output_name}' will not LOAD as '{EXTENSION_NAME}'. Prefer --suffix.",
            file=sys.stderr,
        )

    repo_root = Path(__file__).resolve().parent.parent
    target_dir = resolve_target_dir(repo_root, args.target_dir)
    if args.platform:
        duckdb_platform = args.platform
    elif args.target and args.target in TARGET_TRIPLES:
        duckdb_platform = TARGET_TRIPLES[args.target][0]
    else:
        duckdb_platform = detect_duckdb_platform()
    if args.target and args.target not in TARGET_TRIPLES:
        # An unknown triple falls back to the host's library file name (and, without
        # --platform, the host's DuckDB platform), which is wrong for a cross build.
        print(
            f"warning: unknown --target '{args.target}': looking for the host library name "
            f"'{shared_lib_filename()}' and writing platform '{duckdb_platform}'"
            + ("" if args.platform else " (the host's; pass --platform for a cross build)"),
            file=sys.stderr,
        )
    ext_version, duckdb_version = detect_versions(repo_root)

    profiles = ["debug", "release"] if args.profile == "both" else [args.profile]
    packaged_count = 0
    for prof in profiles:
        lib_path = library_path(target_dir, prof, args.target)
        if lib_path.exists() or args.profile != "both":
            packaged = package_profile(
                repo_root,
                prof,
                duckdb_platform,
                ext_version,
                duckdb_version,
                args.target,
                target_dir=target_dir,
                output_name=args.output_name,
                suffix=args.suffix,
            )
            print(f"Packaged [{prof}]: {packaged}")
            packaged_count += 1

    if packaged_count == 0:
        raise FileNotFoundError(
            f"No built shared libraries found under {target_dir}; "
            "run `cargo build -p duckdb-bolt-extension` first."
        )


if __name__ == "__main__":
    main()

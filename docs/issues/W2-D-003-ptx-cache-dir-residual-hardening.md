<!-- SPDX-License-Identifier: Apache-2.0 -->

# W2-D-003: `bolt_ptx_cache_dir` is not checked on Windows or when the engine uses it

| Field | Value |
| :--- | :--- |
| Title | `bolt_ptx_cache_dir` is not checked on Windows or when the engine uses it |
| Severity | medium |
| Status | blocked-upstream (items 1, 2, 4 fixed; item 2 engine-build hook in W3-D-004; item 3 wontfix; item 5 needs craton-bolt) |
| Kind | bug |
| Files | `crates/core/src/options.rs` (`validate_ptx_cache_dir`, `check_private_dir`, `check_ancestors`), `crates/core/src/runtime.rs` (engine build) |

## Description

W2-D added Unix ownership and permission checks (W1-D-004). Still missing, by reading:

1. **Windows**: no ACL check at all. Any directory, including one writable by `Everyone` or
   `Authenticated Users`, is accepted; PTX planted there is JIT-loaded onto the GPU.
2. **Use-time re-validation**: the directory is validated at `SET` time only. The engine is rebuilt
   lazily, possibly much later; by then the directory can have been replaced or its permissions
   changed. `runtime.rs` passes the stored path to craton-bolt without checking it again.
3. **Group-writable parents** are accepted on Unix (to keep `umask 002` home layouts working); a
   member of the group can rename the cache directory and substitute their own.
4. **Ownership check fails open** when the effective uid cannot be determined (no `/proc` and the
   temporary directory is not writable).
5. **No integrity check** of cached PTX (W1-D-004 item 5).

## Reproduction

Windows: `mkdir C:\shared\ptx` with `icacls C:\shared\ptx /grant Everyone:(OI)(CI)M`, then
`SET bolt_ptx_cache_dir = 'C:\shared\ptx'` is accepted.

## Suggested fix

- Expose `validate_ptx_cache_dir` to `runtime.rs` and call it before every engine build; refuse to
  enable the cache (log, build without it) when it fails.
- Windows: `GetNamedSecurityInfoW` + `GetEffectiveRightsFromAclW` (needs `windows-sys`) and require
  that only the current user, Administrators and SYSTEM have write access.
- Optionally reject group-writable parents behind a strict mode.
- PTX integrity: keyed hash stored outside the cache directory (craton-bolt).

## Update (W3-D, commit 1fec37a)

- **Item 1 (Windows).** Without FFI the ACL cannot be read, so `validate_ptx_cache_dir` now
  requires the canonical directory (or, if missing, its nearest existing ancestor) to be inside the
  user's profile (`%USERPROFILE%`, `%LOCALAPPDATA%`, `%APPDATA%`; roots with fewer than two path
  components are ignored) and not inside `%PUBLIC%` / `%ALLUSERSPROFILE%` / `%ProgramData%`. The
  default profile ACL grants write access only to the user, SYSTEM and Administrators; a directory
  such as `C:\shared\ptx` (where `Authenticated Users` get Modify on new subdirectories) is refused.
  Junctions are resolved by `canonicalize`. An explicit ACL change inside the profile is still not
  detected. Tests: `windows_private_dir_rule`,
  `ptx_cache_dir_outside_the_profile_is_refused_on_windows` (`options.rs`).
- **Item 2 (use-time).** `options::sync_with_runtime` re-validates the active directory
  (`check_ptx_cache_dir_in_use`) at the start of every Bolt scalar call; when it fails, the cache
  is disabled in the runtime and in the setting, and the call fails once with the reason. Test:
  `test_ptx_cache_dir_is_rechecked_when_used`. Not yet covered: the engine build and the
  table-function / replacement-scan paths (W3-D-004, W3-A).
- **Item 4 fixed.** The ownership check fails closed: when the effective uid cannot be determined
  (`/proc/self/status`, a temp-file probe, and now the owner of `/proc/self`), only root-owned
  directories are accepted, with a message saying why.
- Items 3 (group-writable parents) and 5 (PTX integrity) unchanged.

## Update (W4-C)

- **Item 2** is now cheap enough for every call (W4-C-002): the full check runs when the
  directory's fingerprint changes or every 5 s; otherwise one `stat`. The engine-build and
  table-function call sites remain in W3-D-004 (`runtime.rs`, `table_functions.rs`, not owned by
  W4-C).
- **Item 3 wontfix.** Refusing group-writable *parents* breaks the common `umask 002` layout
  (user-private groups, where the group has a single member) for every home directory, while the
  directory itself must still not be group-writable. A group member who can rename the parent is
  someone the administrator already trusted with that tree. The rule is documented in
  `validate_ptx_cache_dir`.
- **Item 5 blocked-upstream.** Integrity of cached PTX needs a keyed hash written and checked by
  craton-bolt's cache (the extension never reads the files); the extension's side of the boundary
  is the directory check above. Track it with the craton-bolt 0.10 upgrade (W1-O-003).

## Update (W5-B)

No change. Item 2's remaining call sites: table-function binds call `options::sync_with_runtime`
(W3-D-004), and the new `CALL` forms (W4-C-004) also call it again in their global init, since a
prepared statement may execute long after binding. The engine-build hook is in `runtime.rs`
(another stream). Item 5 stays blocked upstream (W1-O-003).

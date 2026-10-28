
---
name: Bug Report
about: Create a report to help us improve duckdb-bolt-extension
title: "[BUG] "
labels: ["bug"]
assignees: ""
---

<!-- SPDX-License-Identifier: Apache-2.0 -->

## Bug Description
A clear and concise description of what the bug is.

## Environment Details
- **DuckDB Version**: (e.g., `v2.0.0`)
- **duckdb-bolt-extension Version**: (output of `SELECT bolt_version();`)
- **Operating System**: (e.g., Ubuntu 24.04, Windows 11, macOS Sequoia)
- **NVIDIA GPU Model**: (e.g., RTX 4090, A100, none / cpu fallback)
- **NVIDIA Driver Version & CUDA Toolkit**: (e.g., Driver 560.35, CUDA 12.6)

## Minimal Reproduction Steps
Please provide a self-contained SQL script or Rust test case:

```sql
LOAD 'bolt';

CREATE TABLE example_src AS SELECT 1 AS id, 'test' AS name;
COPY example_src TO 'example_bolt' (FORMAT bolt);

-- Query that causes the issue:
SELECT * FROM bolt_query('SELECT * FROM example_bolt WHERE ...');
```

## Expected Behavior
A clear and concise description of what you expected to happen (e.g., what DuckDB standard query execution returns).

## Actual Behavior / Error Output
Include full error messages, stack traces, or diverging result tables:

```text
[Paste error output or EXPLAIN query result here]
```

## Additional Context
Add any other context, such as memory usage, concurrent connections, or special configuration options (`PRAGMA bolt_*`).

<!-- SPDX-License-Identifier: Apache-2.0 -->

# Getting Support for Craton Bolt DuckDB Extension

Thank you for using **Craton Bolt DuckDB Extension** (`duckdb-bolt-extension`), maintained by [Craton Software Company](https://craton.com.ar).

This document outlines the channels and resources available for getting help, reporting defects, and obtaining commercial support.

---

## 1. Documentation & References

Before opening an issue or contacting support, check the existing project documentation:

- **[README.md](README.md)**: Quick start guide, architecture overview, installation, and basic usage examples.
- **[docs/SQL_REFERENCE.md](docs/SQL_REFERENCE.md)**: Complete SQL function, pragma, and option reference.
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**: Deep dive into internal execution architecture, memory management, and Arrow bridge mechanics.
- **[docs/ROADMAP.md](docs/ROADMAP.md)**: Feature proposals (RFCs), implementation status, and future technical roadmap.
- **[SECURITY.md](SECURITY.md)**: Security policy, sandbox considerations, and vulnerability disclosure instructions.

---

## 2. Community Support & Questions

If you have questions about how to use the extension, configuration options, or deployment scenarios:

- **GitHub Discussions**: Use [GitHub Discussions](https://github.com/craton-co/duckdb-bolt-extension/discussions) for general usage questions, configuration advice, performance tuning tips, and community dialogue.
- **GitHub Issues**: If you suspect a defect or bug in the extension, search existing [GitHub Issues](https://github.com/craton-co/duckdb-bolt-extension/issues) to see if it has already been reported. If not, open a new issue using our [Bug Report Template](.github/ISSUE_TEMPLATE/bug_report.md).

---

## 3. Reporting Bugs & Regressions

When reporting a bug via GitHub Issues, please include:
1. **DuckDB Version**: Exact DuckDB version (e.g. `v2.0.0`).
2. **Bolt Version**: Output of `SELECT bolt_version();`.
3. **Environment**: Operating system (Linux/Windows/macOS), architecture (`x86_64`/`aarch64`), NVIDIA driver version, and GPU model.
4. **Minimal Reproducible Example**: A standalone SQL script that creates a temporary table, registers or copies it, and executes the query triggering the problem.
5. **Expected vs. Actual Outcome**: What DuckDB produced vs. what Bolt returned or the error message observed.

---

## 4. Security Vulnerabilities

**Do not report potential security vulnerabilities through public GitHub issues.**

Please follow the private disclosure process described in [SECURITY.md](SECURITY.md):
- Submit a private advisory via [GitHub Security Advisories](https://github.com/craton-co/duckdb-bolt-extension/security/advisories/new), or
- Email the security team at **`security@craton.com.ar`**.

---

## 5. Enterprise & Commercial Support

For enterprise deployments requiring:
- Custom CUDA kernel optimizations and workload acceleration,
- High-availability distributed clustering or proprietary file format ingest,
- Service Level Agreements (SLAs) with dedicated response times,
- Private feature development or custom hardware qualification,

Please contact Craton Software Company:
- **Website**: [https://craton.com.ar](https://craton.com.ar)
- **Email**: [support@craton.com.ar](mailto:support@craton.com.ar)
- **General Inquiries**: [contact@craton.com.ar](mailto:contact@craton.com.ar)

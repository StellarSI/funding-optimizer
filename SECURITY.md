# Security Policy

## Responsible Disclosure

The `funding-optimizer` team takes the security of our predictive allocation analytics, mathematical optimization models, and export artifacts seriously. We welcome responsible security vulnerability reports from the community to help keep humanitarian and grant allocations safe.

If you believe you have discovered a vulnerability, please report it privately rather than opening a public issue.

---

## Reporting a Vulnerability

Please report vulnerabilities using one of the following private reporting mechanisms:

1. **GitHub Security Advisory (Preferred):**  
   Navigate to the repository's [Security Tab](https://github.com/StellarSI/funding-optimizer/security/advisories) and click **"Report a vulnerability"** to draft a private advisory.

2. **Security Contact Email:**  
   If you cannot use GitHub Advisories, email the maintainers directly at:  
   **`security@stellarsi.org`**

### What to Include in Your Report

To help us triage and resolve the issue quickly, please provide:

- A clear, concise description of the vulnerability.
- Steps to reproduce the issue (proof-of-concept script, sample input payload, or configuration).
- The potential security impact (e.g. integer overflow, denial of service in linear programming solver, unintended PII exposure in logs).
- Suggested remediation or patch, if known.

---

## Response Targets & SLA

- **Initial Acknowledgment:** Within **48 hours** of receiving your private disclosure.
- **Triage & Assessment:** Within **5 business days** with an initial severity rating.
- **Resolution & Coordinated Disclosure:** We aim to ship a patched release before publicly disclosing details. We will credit you in the advisory (unless you prefer anonymity).

---

## Supported Versions

Security updates and vulnerability patches are actively maintained for the following versions:

| Version | Supported          |
| :------ | :----------------- |
| `0.1.x` | :white_check_mark: |
| `< 0.1` | :x:                |

---

## Core Security & Responsible Use Principles

As stated in our project guidelines:

1. **Producer-Only Boundary:** The optimizer recommends allocations and generates SDP-compliant export plans; it **never signs or broadcasts on-chain transactions** or manages private disbursement keys.
2. **Credential Redaction:** Connection strings (`DATABASE_URL`, API tokens) must remain masked and redacted across all loggers and exception handlers.
3. **Data Minimization:** Prefer pseudonymous Stellar public keys (`G...`) over personal identifiable information (PII) in analysis data sets and test fixtures.
4. **Adversarial Resilience:** Solvers and export processors must fail closed when handling infeasible, negative, or structurally malformed constraint inputs.

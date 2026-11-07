# Security Policy

## Supported versions

| Version | Supported |
| ------- | --------- |
| 0.1.x   | :white_check_mark: |

## Reporting a vulnerability

`funding-optimizer` recommends allocations and never signs or submits payouts. It does not handle user credentials, private keys, or recipient personal data beyond what you provide for analysis.

If you believe you have found a security vulnerability, please report it **privately** to the maintainers.

### How to report

- Open a draft [GitHub Security Advisory](https://github.com/StellarSI/funding-optimizer/security/advisories) in this repository. Advisories are private by default, or
- Email the maintainers directly (contact details on the repository or profile).

### What to include

- A clear description of the vulnerability and its impact.
- Minimal, self-contained steps to reproduce.
- The affected version(s) and environment.
- Any suggested mitigation, if you have one.

### What not to send

Do **not** file vulnerability details in public issues, discussions, or pull requests. Public disclosure gives the maintainers no time to assess or remediate the issue before it is widely known.

In particular, do not send:

- Secrets, private keys, access tokens, or credentials.
- Real recipient personal data or live on-chain transaction details tied to individuals.
- Exploit code or proof-of-concept material in a public forum.

## Scope

This policy covers the `funding-optimizer` Python package. Related StellarSI projects
(`transaction-analytics-api`, `fraud-detection-sdk`) maintain their own security policies.

## Disclosure timeline

We aim to acknowledge reports within five business days and share an initial
assessment within ten business days. Remediation timing depends on severity and
complexity.

Thank you for helping keep `funding-optimizer` and its users safe.

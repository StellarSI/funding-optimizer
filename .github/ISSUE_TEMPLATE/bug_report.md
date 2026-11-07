name: Bug Report
description: File a bug report for funding-optimizer
title: "[Bug] "
labels: ["bug", "triage"]
body:
  - type: markdown
    attributes:
      value: |
        Thanks for reporting a bug in `funding-optimizer`! Please fill in
        as much as you can. Do not include secrets, private keys, credentials,
        real recipient data, or live on-chain transaction details tied to individuals.

  - type: input
    id: version
    attributes:
      label: Version
      description: Which version of funding-optimizer are you using? (e.g. 0.1.0, or the commit SHA)
      placeholder: 0.1.0
    validations:
      required: true

  - type: input
    id: environment
    attributes:
      label: Environment
      description: OS and Python version (e.g. Ubuntu 22.04, Python 3.11.9)
      placeholder: Ubuntu 22.04, Python 3.11.9
    validations:
      required: true

  - type: textarea
    id: what-happened
    attributes:
      label: What happened?
      description: A clear and concise description of the bug.
    validations:
      required: true

  - type: textarea
    id: expected
    attributes:
      label: Expected behavior
      description: What you expected to happen.
    validations:
      required: true

  - type: textarea
    id: repro
    attributes:
      label: Steps to reproduce
      description: |
        Minimal, self-contained steps to reproduce the bug, including code or
        input data where possible. No real recipient data or secrets.
    validations:
      required: true

  - type: textarea
    id: logs
    attributes:
      label: Relevant logs or traceback
      description: Paste any relevant error output. Redact credentials first.
    validations:
      required: false

  - type: checkboxes
    id: checks
    attributes:
      label: Pre-submission checks
      options:
        - label: I searched existing issues for a duplicate.
          required: true
        - label: I read CONTRIBUTING.md and will follow the workflow.
          required: true
        - label: I am not submitting secrets or real recipient data.
          required: true

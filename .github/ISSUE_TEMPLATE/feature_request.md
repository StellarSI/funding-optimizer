name: Feature Request
description: Suggest a feature or enhancement for funding-optimizer
title: "[Feature] "
labels: ["enhancement", "triage"]
body:
  - type: markdown
    attributes:
      value: |
        Suggest a feature or enhancement for `funding-optimizer`. Please open
        an issue first so the approach can be agreed before implementation
        (see CONTRIBUTING.md).

        Do not include secrets, private keys, credentials, real recipient data,
        or live on-chain transaction details tied to individuals.

  - type: textarea
    id: problem
    attributes:
      label: Is your feature request related to a problem?
      description: A clear and concise description of the problem you are trying to solve.
    validations:
      required: true

  - type: textarea
    id: solution
    attributes:
      label: Proposed solution
      description: A clear and concise description of what you want to happen.
    validations:
      required: true

  - type: textarea
    id: alternatives
    attributes:
      label: Alternatives considered
      description: Any alternative approaches or workarounds you have considered.
    validations:
      required: false

  - type: textarea
    id: context
    attributes:
      label: Additional context
      description: Anything else relevant (versions, environment, use case).
    validations:
      required: false

  - type: checkboxes
    id: checks
    attributes:
      label: Pre-submission checks
      options:
        - label: I searched existing issues and discussions for a duplicate.
          required: true
        - label: I am not submitting secrets or real recipient data.
          required: true

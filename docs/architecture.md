# Architecture: feature pipeline and data contracts

This document describes the `funding-optimizer` feature pipeline and the
data contracts that flow through it. It expands on
[README §How it works](README.md#how-it-works).

`funding-optimizer` recommends allocations and never signs or submits
payouts.

## Feature pipeline

```text
Program data (CSV / API)          On-chain data
budgets, cohorts, constraints     (Horizon / transaction-analytics-api)
              │                              │
              └──────────────┬───────────────┘
                             ▼
                      Feature pipeline
                             ▼
             Forecast models + allocation optimizer
                             ▼
        Allocation plan  ·  Scenarios  ·  Impact reports
                             │
                             ▼
      Optional payload for SDP / custom payout (you sign and submit)
```

The pipeline has three stages:

1. **Ingestion** — program budgets, cohorts, expected reach, region floors,
   and constraints are read from CSV, JSON, or the analytics API.
2. **Optimization** — the deterministic allocator distributes each budget
   tranche across programs subject to regional floors, program minimum and
   maximum shares, and per-asset constraints.
3. **Reporting** — allocation plans are serialized to a stable JSON contract
   and can be compared across scenarios or linked to on-chain transaction
   hashes.

## Data contracts

### Budget

A budget is one or more Stellar asset amounts:

- Single asset: `Budget(total=250_000, asset="USDC")`
- Multi-asset: `Budget(amounts={"USDC": 150_000, "XLM": 40_000_000})`

Asset codes are trimmed, uppercased, and validated as Stellar asset codes
(1–12 alphanumeric characters, optional `:ISSUER` suffix). Amounts must be
finite, strictly positive real numbers. Duplicate assets after normalization
raise `ValueError`.

### Program

A program receiving an allocation:

- `id` — non-empty, trimmed program identifier.
- `region` — optional non-empty region name used for regional floors.
- `expected_reach` — non-negative integer (people reached per unit funded).
- `min_share` — finite fraction in `[0, 1]`; share of the tranche reserved
  for the program.

### RegionFloor

A regional fairness constraint:

- `region` — region name.
- `floor` — finite fraction in `[0, 1]`; minimum share of the tranche the
  region must receive.

### AssetConstraint

A per-asset bound on how much of one tranche a single program may receive:

- `asset` — validated Stellar asset code.
- `program_id` — program the constraint applies to.
- `min_share` / `max_share` — optional fractions in `[0, 1]`.

Duplicate constraints or constraints on unknown assets/programs raise
`ValueError`.

### AllocationPlan

The result of an optimization run serializes to a stable JSON contract
(`funding_optimizer/schemas/plan.schema.json`). A plan includes, per asset
tranche, the program → amount mapping, the objective value, and whether the
run succeeded. Plans round-trip through `to_dict()` / `to_json()` /
`from_dict()` / `from_json()` and are validated against the schema on read,
so malformed plans fail loudly at the boundary.

## Validation boundaries

- `Budget` and `Program` raise `ValidationError` (a `ValueError`) on invalid
  input: non-positive totals, invalid asset codes, out-of-range shares,
  negative reach, empty program ids.
- Infeasible constraint sets (e.g. regional floors or `min_share` values
  exceeding the budget) produce an infeasibility diagnostic instead of a
  plan, naming the conflicting requirement and a suggested relaxation.

## On-chain boundary

Delivery-time forecasting, cash-out prediction, and SDP-compatible export
are on the roadmap, not implemented in this release. Where the optimizer
touches on-chain data it reads only; it never signs or submits payouts.

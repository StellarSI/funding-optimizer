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
   maximum shares, and per-asset constraints. Budgets disbursed over time
   are allocated phase by phase (`BudgetSchedule` /
   `solve_phased_allocation`), with budget carried forward when programs
   cannot absorb their full tranche. `sensitivity_analysis` re-solves the
   problem at `base ± δ` per cap, minimum and fairness floor to show which
   constraint is binding and what loosening or tightening it would do.
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

### BudgetSchedule and sensitivity reports

A `BudgetSchedule` is an ordered, validated sequence of `BudgetPhase`
values (an id and a positive finite amount per phase). Allocating a
schedule solves one linear program per phase in order; a phase's available
budget is its scheduled amount plus whatever earlier phases carried forward
(phase caps below the tranche force under-allocation, and the remainder
rolls forward). Each phase reports its allocation, carry-in/carry-out and
diagnostics; the run stops at the first infeasible phase.

A sensitivity report (`SensitivityReport`) sweeps every swept parameter —
regional floors, program `min_share` minimums and optional per-program
budget caps — at `base ± perturbation`, clamped to `[0, 1]`. Each entry
records the perturbed value, the objective change (negated reach), whether
the perturbed solve succeeded, and whether the constraint is binding at the
base optimum. Perturbations that become infeasible are reported with
`success=False` and a diagnostic, not raised.

### Forecast dataset and time-window schema

Disbursement forecasting consumes a typed dataset
(`funding_optimizer/forecasting/schema.py`):

- `ObservationWindow` — the closed `[start, end]` history a model is fitted
  on. Bounds are inclusive and normalized from `date`, `datetime`, or
  ISO-8601 strings.
- `ForecastHorizon` — the future window a model predicts, with a `day`,
  `week`, or `month` frequency and a `periods` count that rounds partial
  steps up.
- `ForecastTarget` — `cash_out_rate`, `delivery_delay`, or `funding_gap`.
- `ForecastKey` / `ForecastRow` — every row identifies a program, a region,
  an asset, and a time window, plus the observed value for its target.
  Asset codes are validated and normalized like `Budget` asset codes.
- `ForecastDataset` — rows scoped to one target and one observation
  window, read from raw records with `from_records` and grouped per
  identifier with `group_by_key()`.
- `ForecastPrediction` — a typed prediction carrying its key, target,
  horizon, method, and optional `ForecastEvaluation` metadata (observation
  count, MAE, RMSE, pinball loss, evaluation date).

Rows that do not identify a program, region, asset, and time window are
rejected with a `ValueError` naming the missing field. So are rows outside
the observation window, rows for a different target, empty datasets,
malformed windows, and values outside their target's range (cash-out rates
outside `[0, 1]`, negative delivery delays, non-finite values).

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
- Forecast records that do not identify a program, region, asset, and time
  window are rejected by `ForecastDataset.from_records` with a `ValueError`
  naming the missing field.

## On-chain boundary

Delivery-time and cash-out forecast *models*, and SDP-compatible export,
are on the roadmap: this release ships the forecast dataset schema, which
describes data only. Where the optimizer touches on-chain data it reads
only; it never signs or submits payouts.

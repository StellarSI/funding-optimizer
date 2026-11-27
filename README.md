# funding-optimizer

> Predictive analytics for grant allocation and NGO disbursement on Stellar.

![Network](https://img.shields.io/badge/network-Stellar-black)
![Status](https://img.shields.io/badge/status-in%20development-orange)
![License](https://img.shields.io/badge/license-MIT-blue)

`funding-optimizer` helps funders and NGOs decide **how much to send, to whom, and when**, then measures how well those funds actually reached people. It pairs predictive models with on-chain Stellar data, so allocation decisions are grounded in observed delivery rather than assumptions.

Part of the **StellarSI** toolkit, alongside [`transaction-analytics-api`](https://github.com/StellarSI/transaction-analytics-api) and [`fraud-detection-sdk`](https://github.com/StellarSI/fraud-detection-sdk).

## Why Stellar

Stellar is widely used for low-cost, near-instant cross-border payments and stablecoin disbursements. Every payment is public, timestamped and auditable, which makes it a strong basis for transparent funding analytics. The [Stellar Disbursement Platform (SDP)](https://developers.stellar.org/platforms/stellar-disbursement-platform) from SDF is built for exactly this workflow, and this project is designed to complement it.

## Features

- **Allocation optimization**: recommend how to split a budget across programs, regions or recipient cohorts under constraints (caps, minimums, fairness rules)
- **Disbursement forecasting**: predict cash-out rates, delays and funding gaps
- **Delivery analytics**: track time-to-receipt, trustline readiness and unclaimed funds
- **Scenario simulation**: compare "what if" budgets and schedules before committing funds
- **Impact reporting**: exportable, verifiable reports linked to on-chain transaction hashes
- **Grant program support**: tooling relevant to funders running rounds, such as those in the [Stellar Community Fund](https://communityfund.stellar.org)

## How it works

```
 Program data (CSV / API)          On-chain data
 budgets, cohorts, constraints     (transaction-analytics-api / Horizon)
              │                              │
              └──────────────┬───────────────┘
                             ▼
                  Feature pipeline
                             ▼
             Forecast models + optimizer
                             ▼
          Allocation plan · Scenarios · Reports
                             │
                             ▼
        Optional: payload for SDP / custom payout
```

The optimizer **recommends**. It never moves funds on its own. Payouts remain under your control via SDP or your own signing flow.

## Allocation optimizer

The optimizer ships three deterministic, testable building blocks under
``funding_optimizer/optimizer``:

* ``solver.py`` — a deterministic linear-programming adapter (issue #14) that
  wraps SciPy/HiGHS behind a narrow interface. Solver options are pinned so
  repeated runs with identical inputs produce identical allocations, and
  `solve_with_lexicographic_tie_break` canonicalizes equal-value optima
  (issue #16).
* ``constraints.py`` — regional-floor fairness constraints (issue #13) plus a
  reach-maximizing objective, expressed as separate linear pieces; issue #17
  extends it to multi-asset budgets and per-asset constraints.
* ``models.py`` — the `Budget` value object (issue #17): single-asset or
  multi-asset Stellar budgets with validated asset codes.
* ``diagnostics.py`` — actionable infeasibility diagnostics (issue #15) that
  explain why a constraint set admits no allocation and what to change.
* ``schedule.py`` — phased allocation for budget schedules (issue #20) that
  spreads a budget arriving over time across phases and carries unallocated
  budget forward when programs cannot absorb their full tranche.
* ``sensitivity.py`` — sensitivity analysis for caps, minimums, and fairness
  (issue #21) that re-solves the problem at ``base ± δ`` per parameter and
  reports binding constraints and objective deltas.

All are verified by tests under ``tests/`` and the project lints, type-checks
and tests cleanly with ``make check``.

```python
from funding_optimizer.optimizer.constraints import (
    Program,
    RegionFloor,
    solve_allocation,
)

programs = [
    Program(id="cash-transfers-ng", region="Africa", expected_reach=12_000, min_share=0.20),
    Program(id="education-ke", region="Africa", expected_reach=4_500, min_share=0.10),
    Program(id="health-gh", region="Africa", expected_reach=7_800, min_share=0.15),
]

result = solve_allocation(programs, budget=250_000, region_floors=[
    RegionFloor(region="Africa", floor=0.50),
])
print(result.allocation)   # deterministic allocation vector
print(result.objective_value)
```

### Deterministic tie-breaking

A linear program can admit many allocations with the same objective value,
and the solver may return any of them. Pass `tie_break=True` to canonicalize
tied optima: earlier-listed programs are funded first, so repeated runs
return identical allocations even when the LP has ties.

```python
result = solve_allocation(programs, budget=250_000, region_floors=[
    RegionFloor(region="Africa", floor=0.50),
], tie_break=True)
assert result.tie_break_applied
```

For raw linear programs, `solve_with_lexicographic_tie_break` offers the same
refinement directly on the solver adapter.

### Infeasibility diagnostics

When the constraint set admits no allocation, the optimizer explains why and
what to change instead of returning a bare failure:

```python
from funding_optimizer.optimizer import diagnose_infeasibility, explain_result

result = solve_allocation(programs, budget=250_000, region_floors=[
    RegionFloor(region="Africa", floor=0.60),
    RegionFloor(region="Asia", floor=0.70),
])
if not result.success:
    for diagnostic in explain_result(result, programs, 250_000, [
        RegionFloor(region="Africa", floor=0.60),
        RegionFloor(region="Asia", floor=0.70),
    ]):
        print(diagnostic.code)        # e.g. "floors_exceed_budget"
        print(diagnostic.message)     # names the conflicting requirement
        print(diagnostic.suggestion)  # the relaxation that restores feasibility
```

`diagnose_infeasibility` runs the same structural analysis on a constraint
set without solving it first.

### Multi-asset budgets

Budgets can span more than one Stellar asset. Pass an `amounts` mapping to
`Budget`; the optimizer allocates each tranche fully and applies regional
floors, program `min_share` and per-asset constraints within every tranche:

```python
from funding_optimizer import Budget
from funding_optimizer.optimizer.constraints import (
    AssetConstraint,
    Program,
    RegionFloor,
    solve_multi_asset_allocation,
)

budget = Budget(amounts={"USDC": 150_000, "XLM": 40_000_000})
programs = [
    Program(id="cash-transfers-ng", region="Africa", expected_reach=12_000, min_share=0.20),
    Program(id="education-ke", region="Africa", expected_reach=4_500, min_share=0.10),
]

result = solve_multi_asset_allocation(
    programs,
    budget,
    [RegionFloor(region="Africa", floor=0.50)],
    asset_constraints=[
        AssetConstraint(asset="XLM", program_id="education-ke", max_share=0.5),
    ],
)
print(result.allocation)  # {"USDC": {...}, "XLM": {...}} — one entry per tranche
```

Single-asset budgets keep working unchanged:
`Budget(total=250_000, asset="USDC")`.

`AssetConstraint` bounds how much of one tranche a single program may or
must receive, as a fraction of that tranche: `min_share=0.4` requires at
least 40% of the asset to reach the program, `max_share=0.5` caps it at 50%.
Asset codes are validated and normalized (1-12 alphanumerics, optional
`:ISSUER` suffix) in `Budget` and in constraints, and duplicate constraints
or unknown assets/programs raise `ValueError`.

Note: allocations are balanced per tranche — the per-unit `expected_reach`
is treated as equal across assets, and there is no cross-asset coupling yet
(e.g. capping the total value held in one asset). Multi-asset runs return a
`MultiAssetResult` whose `allocation` maps asset → program id → amount.

### Phased allocation for budget schedules

Grant budgets often arrive in tranches over time (monthly, quarterly, ...).
`BudgetSchedule` models that disbursement plan, and `solve_phased_allocation`
allocates each phase in order: a phase receives its scheduled amount plus any
budget carried forward from earlier phases that its programs could not
absorb. Regional floors and program `min_share` minimums apply to each
phase's available budget, and optional per-program `phase_caps` bound how
much of a phase a program may absorb:

```python
from funding_optimizer.optimizer import (
    BudgetPhase,
    BudgetSchedule,
    Program,
    RegionFloor,
    solve_phased_allocation,
)

programs = [
    Program(id="cash-transfers-ng", region="Africa", expected_reach=12_000, min_share=0.10),
    Program(id="education-ke", region="Africa", expected_reach=4_500),
    Program(id="health-gh", region="Asia", expected_reach=7_800),
]
schedule = BudgetSchedule([
    BudgetPhase(id="2026-Q1", amount=150_000),
    BudgetPhase(id="2026-Q2", amount=100_000),
])

result = solve_phased_allocation(
    programs,
    schedule,
    [RegionFloor(region="Africa", floor=0.50)],
    phase_caps={"cash-transfers-ng": 0.60},   # at most 60% of a phase per program
)
print(result.cumulative_allocation)           # program id -> total over all phases
print(result.unallocated)                     # budget carried out of the final phase
for phase in result.phases:
    print(phase.phase_id, phase.available, phase.carry_out)
```

A phase whose constraint set is infeasible stops the run: later phases
depend on the carried budget, so they are not attempted. The failing phase
in the returned result carries `diagnostics` naming the conflict (for
example `phase_cap_below_min_share`) and the relaxation that restores
feasibility.

### Sensitivity analysis

`sensitivity_analysis` re-solves the allocation problem at `base ± δ` for
every region floor, program `min_share`, and optional per-program `caps`
fraction, and reports how the objective moves. Objectives are negated
reach, so a positive `objective_delta` means reach was lost:

```python
from funding_optimizer.optimizer import (
    Program,
    RegionFloor,
    sensitivity_analysis,
)

programs = [
    Program(id="cash-transfers-ng", region="Africa", expected_reach=12_000, min_share=0.20),
    Program(id="education-ke", region="Asia", expected_reach=4_500),
]

report = sensitivity_analysis(
    programs,
    250_000,
    [RegionFloor(region="Africa", floor=0.50), RegionFloor(region="Asia", floor=0.20)],
    caps={"cash-transfers-ng": 0.50},
    perturbation=0.05,
)
print(report.binding_parameters)  # parameters tight at the base optimum
for entry in report.entries:
    print(entry.parameter, entry.perturbation, entry.objective_value, entry.binding)
```

A constraint is *binding* when it holds with equality at the base optimum;
only binding constraints move the objective when perturbed. Perturbations
that become infeasible (for example floors summing over 100% of the budget)
are reported with `success=False` and a diagnostic instead of raising.

## Scenario simulation

Scenarios are versioned, validated documents under
`funding_optimizer/scenarios`. They capture a budget, assumptions and the
model versions a forecast came from, and never execute a solve:

```python
from funding_optimizer.scenarios import parse_scenario

scenario = parse_scenario({
    "version": 1,
    "id": "baseline-2026",
    "name": "Baseline 2026",
    "budgets": {"USDC": 150_000, "XLM": 40_000_000},
    "assumptions": {"cashout_rate": 0.82},
    "model_versions": {"forecast": "v2.1"},
    "seed": 42,
})
print(scenario.fingerprint())  # stable for the same document and seed
```

Validation rejects unsupported versions, malformed budgets, non-scalar
assumptions and unknown fields. The optimizer still only recommends
allocations — it never signs or submits payouts.

## Delivery analytics

`funding_optimizer/analytics` measures what happened after funds were
recommended. All metrics consume the normalized events from P34 and use
inclusive UTC observation windows; amounts are in asset units and rates are
fractions in `[0, 1]`. The optimizer only recommends allocations — it never
signs or submits payouts.

- `compute_time_to_receipt_metrics` (issue #36) pairs disbursements with
  receipts and reports match rate plus receipt-time summaries in seconds.
- `compute_trustline_readiness_rates` (issue #35) reports readiness by cohort.
- `compute_unclaimed_funds` (issue #37) reports unclaimed balances with aging
  buckets (`0-7`, `8-30`, `31-90`, `90+` days by default) and per-asset totals.
- `compute_realized_cashout_rates` (issue #38) reports how much of each
  disbursed amount was actually received, per asset and overall.
- `compute_delivery_aggregates` (issue #40) reports event, disbursement,
  receipt, recipient and amount totals by program, region, cohort and asset,
  with amounts kept separate per asset. The three grouping keys default to
  `program`, `region` and `cohort` in event metadata and are configurable.
- `compute_delivery_delay_distributions` (issue #39) summarizes matched
  delivery delays in seconds by corridor and program, with percentiles, a
  histogram and a per-asset breakdown. Corridor and program come from event
  metadata (`corridor` and `program` by default; missing values read
  `unknown`).

```python
from funding_optimizer.analytics import (
    compute_realized_cashout_rates,
    compute_unclaimed_funds,
)

unclaimed = compute_unclaimed_funds(events, disbursement_account="GFUND", as_of="2026-02-20T00:00:00Z")
cashout = compute_realized_cashout_rates(events, disbursement_account="GFUND")
print(unclaimed.total_amount, [b.bucket for b in unclaimed.buckets])
print(cashout.realization_rate, cashout.recipient_cashout_rate)
```

Each metric returns a frozen result object with an `as_dict()` helper and is
covered by a known-event fixture with hand-checkable values.

## Disbursement forecasting

Forecasting lives under `funding_optimizer/forecasting`. It is built on the
**forecast dataset and time-window schema** (issue #22): the typed contract
every forecast model consumes and produces.

- `ObservationWindow` — the closed `[start, end]` history a model is fitted
  on. Bounds are inclusive and normalized from `date`, `datetime`, or
  ISO-8601 strings; a start after the end is rejected.
- `ForecastHorizon` — the future window a model predicts, with a `day`,
  `week`, or `month` `frequency` and a `periods` count that rounds partial
  steps up.
- `ForecastTarget` — what is being predicted: `cash_out_rate`,
  `delivery_delay`, or `funding_gap`.
- `ForecastKey` / `ForecastRow` — every observation identifies a program, a
  region, an asset, and a time window, plus the observed value for its
  target. Asset codes are validated and normalized like `Budget` assets.
- `ForecastDataset` — rows scoped to a single target and a single
  observation window, built from CSV/JSON-style records with
  `ForecastDataset.from_records` and grouped per identifier with
  `group_by_key()`.
- `ForecastPrediction` — a typed point prediction that carries its key,
  target, horizon, `method`, and optional `ForecastEvaluation` metadata
  (observation count, MAE, RMSE, pinball loss, evaluation date).

```python
from datetime import date

from funding_optimizer.forecasting import (
    ForecastDataset,
    ForecastTarget,
    ObservationWindow,
)

dataset = ForecastDataset.from_records(
    [
        {
            "program": "cash-transfers-ng",
            "region": "Africa",
            "asset": "USDC",
            "window": {"start": "2025-10-01", "end": "2025-12-31"},
            "value": 0.82,
        },
    ],
    target=ForecastTarget.CASH_OUT_RATE,
    observation=ObservationWindow(start=date(2025, 1, 1), end=date(2025, 12, 31)),
)
print(dataset.rows[0].value)            # 0.82
print(dataset.rows[0].window.end)       # 2025-12-31
```

Rows that do not identify a program, region, asset, and time window are
rejected at the boundary with a `ValueError` naming the missing field, so
incomplete observations never reach a model. The schema also rejects rows
outside the declared observation window, rows for a different target, empty
datasets, malformed windows, and values outside their target's range
(cash-out rates outside `[0, 1]`, negative delivery delays, non-finite
values).

### Historical baseline cash-out rates

`cash_out_rate_baseline` (`funding_optimizer/forecasting/baseline.py`,
issue #23) is a transparent baseline: it predicts each identifier's cash-out
rate as the historical mean of that identifier's observations in the
training dataset. It trains only on the dataset's observation window and
returns one typed `ForecastPrediction` per identifier (in first-seen order)
with the requested horizon, method `historical-mean`, and in-sample
evaluation metadata (observation count, MAE, RMSE).

```python
from datetime import date

from funding_optimizer.forecasting import (
    ForecastDataset,
    ForecastHorizon,
    ForecastTarget,
    ObservationWindow,
    cash_out_rate_baseline,
)

dataset = ForecastDataset.from_records(
    [
        {
            "program": "cash-transfers-ng",
            "region": "Africa",
            "asset": "USDC",
            "window": {"start": "2025-10-01", "end": "2025-12-31"},
            "value": 0.82,
        },
        {
            "program": "cash-transfers-ng",
            "region": "Africa",
            "asset": "USDC",
            "window": {"start": "2025-07-01", "end": "2025-09-30"},
            "value": 0.78,
        },
    ],
    target=ForecastTarget.CASH_OUT_RATE,
    observation=ObservationWindow(start=date(2025, 1, 1), end=date(2025, 12, 31)),
)
horizon = ForecastHorizon(start=date(2026, 1, 1), end=date(2026, 3, 31), frequency="month")

(prediction,) = cash_out_rate_baseline(dataset, horizon)
print(prediction.method)                        # historical-mean
print(prediction.value)                         # 0.8
print(prediction.evaluation.n_observations)     # 2
```

### Delivery-delay quantile predictions

`delivery_delay_quantiles` (`funding_optimizer/forecasting/delays.py`,
issue #24) predicts delivery delays from the empirical quantiles of each
identifier's historical delays (linear interpolation, no distribution
assumed). It consumes a `delivery_delay` dataset and returns one typed
`ForecastPrediction` per (identifier, quantile) with the requested horizon,
method `historical-quantile-<q>p`, and evaluation metadata (observation
count and pinball loss at that quantile).

```python
from datetime import date

from funding_optimizer.forecasting import (
    ForecastDataset,
    ForecastHorizon,
    ForecastTarget,
    ObservationWindow,
    delivery_delay_quantiles,
)

dataset = ForecastDataset.from_records(
    [
        {
            "program": "cash-transfers-ng",
            "region": "Africa",
            "asset": "USDC",
            "window": {"start": "2025-10-01", "end": "2025-12-31"},
            "value": 10.0,
        },
        {
            "program": "cash-transfers-ng",
            "region": "Africa",
            "asset": "USDC",
            "window": {"start": "2025-07-01", "end": "2025-09-30"},
            "value": 20.0,
        },
        {
            "program": "cash-transfers-ng",
            "region": "Africa",
            "asset": "USDC",
            "window": {"start": "2025-04-01", "end": "2025-06-30"},
            "value": 30.0,
        },
    ],
    target=ForecastTarget.DELIVERY_DELAY,
    observation=ObservationWindow(start=date(2025, 1, 1), end=date(2025, 12, 31)),
)
horizon = ForecastHorizon(start=date(2026, 1, 1), end=date(2026, 3, 31), frequency="month")

predictions = delivery_delay_quantiles(dataset, horizon, quantiles=(0.25, 0.5, 0.75))
print([p.method for p in predictions])          # ['historical-quantile-25p', ...]
print([p.value for p in predictions])           # [15.0, 20.0, 25.0]
print(predictions[1].evaluation.pinball_loss)   # 6.666...
```

### Funding-gap projections

`funding_gap_baseline` (`funding_optimizer/forecasting/gaps.py`,
issue #25) projects a funding gap per identifier as the historical mean of
that identifier's observed gaps. The schema defines the gap as the
shortfall (positive) or surplus (negative) of funds for a window: a
positive prediction is a projected shortfall, a negative one a projected
surplus. Predictions carry the requested horizon, method `historical-mean`,
and in-sample evaluation metadata (observation count, MAE, RMSE).

```python
from datetime import date

from funding_optimizer.forecasting import (
    ForecastDataset,
    ForecastHorizon,
    ForecastTarget,
    ObservationWindow,
    funding_gap_baseline,
)

dataset = ForecastDataset.from_records(
    [
        {
            "program": "cash-transfers-ng",
            "region": "Africa",
            "asset": "USDC",
            "window": {"start": "2025-10-01", "end": "2025-12-31"},
            "value": 120000.0,
        },
        {
            "program": "cash-transfers-ng",
            "region": "Africa",
            "asset": "USDC",
            "window": {"start": "2025-07-01", "end": "2025-09-30"},
            "value": -40000.0,
        },
    ],
    target=ForecastTarget.FUNDING_GAP,
    observation=ObservationWindow(start=date(2025, 1, 1), end=date(2025, 12, 31)),
)
horizon = ForecastHorizon(start=date(2026, 1, 1), end=date(2026, 3, 31), frequency="month")

(prediction,) = funding_gap_baseline(dataset, horizon)
print(prediction.method)                        # historical-mean
print(prediction.value)                         # 40000.0
print(prediction.evaluation.n_observations)     # 2
```

### Cold-start handling for new programs and regions

`cold_start_baseline` (`funding_optimizer/forecasting/cold_start.py`)
produces typed predictions for requested identifiers even when a program or
region has no historical observations in the training dataset. It trains only
on the historical observation window and falls back deterministically:

1. Exact `(program, region, asset)` history -> `historical-mean`
2. Same region and asset, different/new program -> `cold-start-region-mean`
3. Same program and asset, different/new region -> `cold-start-program-mean`
4. Same asset across the dataset -> `cold-start-asset-mean`

If no observations exist for the requested asset, prediction fails with a
`ValueError` naming the missing history. Returned predictions carry the
requested horizon, method, and evaluation metadata computed over the fallback
pool.

```python
from datetime import date

from funding_optimizer.forecasting import (
    ForecastDataset,
    ForecastHorizon,
    ForecastKey,
    ForecastTarget,
    ObservationWindow,
    cold_start_baseline,
)

dataset = ForecastDataset.from_records(
    [
        {"program": "cash-transfers-ng", "region": "Africa", "asset": "USDC",
         "window": {"start": "2025-10-01", "end": "2025-12-31"}, "value": 0.82},
        {"program": "education-ke", "region": "Africa", "asset": "USDC",
         "window": {"start": "2025-10-01", "end": "2025-12-31"}, "value": 0.74},
    ],
    target=ForecastTarget.CASH_OUT_RATE,
    observation=ObservationWindow(start=date(2025, 1, 1), end=date(2025, 12, 31)),
)
horizon = ForecastHorizon(start=date(2026, 1, 1), end=date(2026, 3, 31), frequency="month")
new_program = ForecastKey(program="health-gh", region="Africa", asset="USDC")

(prediction,) = cold_start_baseline(dataset, horizon, [new_program])
print(prediction.method)  # cold-start-region-mean
print(prediction.value)   # 0.78
```

### Reusable program, region, asset, and calendar features

`funding_optimizer/forecasting/features.py` builds typed, reusable features
from validated forecast observations so downstream models can share a
consistent representation. Features are keyed by **program**, **region**,
**asset**, and **calendar** attributes of each observation window.

- `build_calendar_features(window)` extracts calendar signals from a
  `TimeWindow` (start/end year, month, quarter, ISO week, day-of-year,
  `window_days`, and boundary-crossing flags).
- `build_feature_rows(dataset)` returns one `FeatureRow` per dataset row in
  row order, combining identifiers, target/value, and calendar features.
  `FeatureRow.to_feature_map()` flattens the row into a scalar feature map.
- `build_key_features(dataset)` aggregates per `(program, region, asset)`
  identifier in first-seen order, summarizing observation counts, value
  statistics, window-span statistics, and calendar coverage.

```python
from datetime import date

from funding_optimizer.forecasting import (
    ForecastDataset,
    ForecastTarget,
    ObservationWindow,
    build_feature_rows,
    build_key_features,
)

dataset = ForecastDataset.from_records(
    [
        {
            "program": "cash-transfers-ng",
            "region": "Africa",
            "asset": "USDC",
            "window": {"start": "2025-10-01", "end": "2025-12-31"},
            "value": 0.82,
        },
        {
            "program": "cash-transfers-ng",
            "region": "Africa",
            "asset": "USDC",
            "window": {"start": "2025-07-01", "end": "2025-09-30"},
            "value": 0.78,
        },
    ],
    target=ForecastTarget.CASH_OUT_RATE,
    observation=ObservationWindow(start=date(2025, 1, 1), end=date(2025, 12, 31)),
)

row = build_feature_rows(dataset)[0]
print(row.program, row.region, row.asset)  # cash-transfers-ng Africa USDC
print(row.calendar.start_quarter, row.calendar.window_days)  # 4 92

(key_features,) = build_key_features(dataset)
print(key_features.n_observations, key_features.start_quarters)  # 2 (3, 4)
```

### Time-based backtesting harness

`run_backtest` (`funding_optimizer/forecasting/backtest.py`) evaluates a
forecast method over expanding historical windows without lookahead. For each
fold it trains only on rows whose windows end before the test horizon, calls
the forecaster with `(train_dataset, test_horizon)`, and returns typed
`ForecastPrediction` values carrying their horizon, method, and evaluation
metadata alongside held-out actuals.

```python
from datetime import date

from funding_optimizer.forecasting import (
    ForecastDataset,
    ForecastHorizon,
    ForecastTarget,
    ObservationWindow,
    cash_out_rate_baseline,
    run_backtest,
)

dataset = ForecastDataset.from_records(
    [
        {"program": "p", "region": "r", "asset": "USDC",
         "window": {"start": "2025-01-01", "end": "2025-01-10"}, "value": 0.6},
        {"program": "p", "region": "r", "asset": "USDC",
         "window": {"start": "2025-01-11", "end": "2025-01-20"}, "value": 0.8},
        {"program": "p", "region": "r", "asset": "USDC",
         "window": {"start": "2025-01-21", "end": "2025-01-30"}, "value": 0.7},
    ],
    target=ForecastTarget.CASH_OUT_RATE,
    observation=ObservationWindow(start=date(2025, 1, 1), end=date(2025, 1, 30)),
)
horizon = ForecastHorizon(start=date(2025, 1, 11), end=date(2025, 1, 20), frequency="day")

result = run_backtest(dataset, horizon, cash_out_rate_baseline)
print(len(result.folds))                           # 2
print(result.predictions[0].horizon, result.predictions[0].method)
print(result.evaluation.mae)                       # aggregate MAE over matched pairs
```

### Forecast metrics

`funding_optimizer/forecasting/metrics.py` provides reusable evaluation
metrics over forecast actuals and predictions:

- `mean_absolute_error(actual, predicted)` �?" mean absolute error for point
  forecasts.
- `pinball_loss(actual, predicted, quantile)` �?" asymmetric quantile loss for
  a single quantile (the median, `0.5`, yields half the MAE).
- `quantile_calibration(actual, predicted_quantiles)` �?" empirical coverage
  per nominal quantile and mean absolute calibration error, measuring whether
  predicted quantiles match observed outcomes.

```python
from funding_optimizer.forecasting import mean_absolute_error, pinball_loss, quantile_calibration

print(mean_absolute_error([1.0, 2.0, 3.0], [1.0, 2.0, 4.0]))     # 0.333...
print(pinball_loss([10.0, 20.0, 30.0], 20.0, 0.5))               # 3.333...

cal = quantile_calibration([1.0, 2.0, 3.0, 4.0], {0.5: 2.5, 0.75: 3.5})
print(cal.bins[0].empirical_coverage, cal.mean_calibration_error)
```

### Prediction intervals and uncertainty metadata

`prediction_intervals` (`funding_optimizer/forecasting/uncertainty.py`)
attaches prediction intervals and uncertainty metadata to historical forecasts.
For each identifier group it computes the historical mean and the empirical
quantiles of residuals (`value - mean`) over the training observation window,
returning each bound as a typed `ForecastPrediction` with horizon, method, and
evaluation metadata (pinball loss at that quantile).

```python
from datetime import date

from funding_optimizer.forecasting import (
    ForecastDataset,
    ForecastHorizon,
    ForecastTarget,
    ObservationWindow,
    prediction_intervals,
)

dataset = ForecastDataset.from_records(
    [
        {"program": "cash-transfers-ng", "region": "Africa", "asset": "USDC",
         "window": {"start": "2025-10-01", "end": "2025-12-31"}, "value": 0.6},
        {"program": "cash-transfers-ng", "region": "Africa", "asset": "USDC",
         "window": {"start": "2025-07-01", "end": "2025-09-30"}, "value": 0.8},
        {"program": "cash-transfers-ng", "region": "Africa", "asset": "USDC",
         "window": {"start": "2025-04-01", "end": "2025-06-30"}, "value": 1.0},
    ],
    target=ForecastTarget.CASH_OUT_RATE,
    observation=ObservationWindow(start=date(2025, 1, 1), end=date(2025, 12, 31)),
)
horizon = ForecastHorizon(start=date(2026, 1, 1), end=date(2026, 3, 31), frequency="month")

low, mid, high = prediction_intervals(dataset, horizon, quantiles=(0.25, 0.5, 0.75))
print(low.prediction.method, low.prediction.value)   # historical-residual-quantile-0.25p 0.7
print(mid.prediction.evaluation.pinball_loss)        # uncertainty metadata
```

Forecasting recommends allocations and never signs or submits payouts.

## Getting started

### Prerequisites

- Python 3.11+
- Access to Stellar data via a Horizon endpoint or [`transaction-analytics-api`](https://github.com/StellarSI/transaction-analytics-api)

### Install

```bash
git clone https://github.com/StellarSI/funding-optimizer.git
cd funding-optimizer
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

### Quick example (planned API)

```python
from funding_optimizer import Budget, Program, Optimizer

budget = Budget(total=250_000, asset="USDC")

programs = [
    Program(id="cash-transfers-ng", min_share=0.20, expected_reach=12_000),
    Program(id="education-ke", min_share=0.10, expected_reach=4_500),
    Program(id="health-gh", min_share=0.15, expected_reach=7_800),
]

plan = Optimizer(objective="maximize_reach", fairness="regional_floor").solve(budget, programs)
print(plan.summary())
```

### The maximize_reach objective

`maximize_reach` binds every program's `min_share` floor and hands the
remaining budget to the program with the largest `expected_reach` (the
optimum of the linear reach model), tie-breaking deterministically by
program id:

```python
from funding_optimizer import Budget, Program
from funding_optimizer.optimizer import solve_maximize_reach

plan = solve_maximize_reach(
    Budget(total=250_000, asset="USDC"),
    [
        Program(id="cash-transfers-ng", min_share=0.20, expected_reach=12_000),
        Program(id="education-ke", min_share=0.10, expected_reach=4_500),
    ],
)
# shares: cash-transfers-ng 0.90, education-ke 0.10
```

Problems whose `min_share` values sum over 100% are infeasible; the solver
fails with a diagnostic instead of returning a plan.

### maximize_reach solver

`solve_maximize_reach` binds every program's `min_share` floor and hands
the remaining budget to the program with the largest `expected_reach`
(the optimum of the linear reach model), tie-breaking deterministically
by program id:

```python
from funding_optimizer import Budget, Program
from funding_optimizer.optimizer import solve_maximize_reach

plan = solve_maximize_reach(
    Budget(total=250_000, asset="USDC"),
    [
        Program(id="cash-transfers-ng", min_share=0.20, expected_reach=12_000),
        Program(id="education-ke", min_share=0.10, expected_reach=4_500),
    ],
)
```

Infeasible problems (min_share floors summing over 100%) raise a
diagnostic error rather than returning a plan.

### Optimize from the command line

`optimize` reads a JSON problem, solves it, and writes the plan as JSON. It
prints to stdout by default:

```bash
funding-optimizer optimize --input programs.json --output plan.json
```

`programs.json`:

```json
{
  "objective": "maximize_reach",
  "fairness": "regional_floor",
  "budget": {"total": 250000, "asset": "USDC"},
  "programs": [
    {"id": "cash-transfers-ng", "min_share": 0.2, "expected_reach": 12000},
    {"id": "education-ke", "min_share": 0.1, "expected_reach": 4500},
    {"id": "health-gh", "min_share": 0.15, "expected_reach": 7800}
  ]
}
```

A top-level array of programs is also accepted; pass `--total` and `--asset`
in that case. Use `--input -` to read stdin and `--output -` to write stdout.
`--objective` and `--fairness` override the values in the file. The command
only recommends allocations and never signs or submits payouts.

### Plan serialization (JSON Schema)

Allocation plans serialize to a stable, published JSON contract so they can
be stored, diffed, and validated across systems. The schema is
`funding_optimizer/schemas/plan.schema.json`:

```python
from funding_optimizer.plan import AllocationPlan

payload = plan.to_dict()        # schema-conformant dict
text = plan.to_json()           # stable compact JSON
restored = AllocationPlan.from_json(text)
assert restored == plan
```

`from_dict`/`from_json` validate the payload against the schema and raise
`ValidationError` on drift (unknown keys, out-of-range shares, unknown
objectives), so malformed plans fail loudly at the boundary.

### Input validation

`Budget` and `Program` validate their inputs and raise
`funding_optimizer.errors.ValidationError` (a `ValueError`) on invalid data:

- `Budget.total` must be a finite, strictly positive number.
- `Budget.asset` is trimmed, uppercased, and must be a Stellar asset code
  (1-12 alphanumeric characters), optionally suffixed with `:ISSUER`.
- `Program.id` must be non-empty; surrounding whitespace is trimmed.
- `Program.min_share` must be a finite fraction between `0` and `1`.
- `Program.expected_reach` must be a non-negative integer.
- `Program.region`, when provided, must be a non-empty string.

## Ingesting inputs

The optimizer consumes two typed input sources, both under
`funding_optimizer/ingest/`. Neither signs nor submits payouts; they only
feed the allocator and the analytics pipeline.

### Programs, cohorts and constraints from CSV

`funding_optimizer/ingest/csv_loader.py` (issue #31) reads the
operator-maintained CSV exports into the typed structures the optimizer
consumes. Parsing is strict and deterministic: files must be UTF-8, the
header row is required (matched case-insensitively; unknown columns are
ignored), and every error names the offending row and field.

```python
from funding_optimizer.ingest import (
    load_allocation_inputs,
    load_cohorts,
    load_programs,
    load_region_floors,
)

programs = load_programs("programs.csv")          # id, region, expected_reach, min_share?
cohorts = load_cohorts("cohorts.csv", programs=programs)  # id, program_ids (';'-separated), min_share?, max_share?
floors = load_region_floors("constraints.csv")    # region, floor

bundle = load_allocation_inputs("programs.csv", cohorts="cohorts.csv", constraints="constraints.csv")
print(bundle.programs, bundle.cohorts, bundle.region_floors)
```

Validation happens at the boundary: duplicate ids, out-of-range shares,
cohort member references to unknown programs, programs assigned to multiple
cohorts, cohort or floor shares summing over 100%, and malformed numbers are
all rejected with a row-numbered `CsvLoaderError`.

### On-chain delivery data from Horizon

`funding_optimizer/ingest/horizon.py` (issue #32) reads a Stellar Horizon
endpoint with explicit timeouts, retries with exponential backoff (429/5xx
and transport failures, honoring `Retry-After` up to `max_retry_after`
seconds), and a minimum spacing between requests derived from `rate_limit`
(requests per second, default 5). Retries are deterministic - there is no
randomized jitter - and `sleep`/`clock` can be injected for tests. Payment-
like records are parsed into typed `HorizonPayment` values; malformed
records raise `HorizonRecordError` naming the record index, and a malformed
response envelope raises `HorizonEnvelopeError`.

```python
from funding_optimizer.ingest import HorizonClient

client = HorizonClient("https://horizon-testnet.stellar.org", rate_limit=5.0)
page = client.get_page("/accounts/G.../payments", params={"limit": 200})

for payment in client.iter_payments("G...", max_records=500):
    print(payment.asset, payment.amount, payment.to_account)
```

`HORIZON_URL` (see [Configuration](#configuration)) is used as the default
base URL when `base_url` is not passed. Pagination follows `next` links
automatically when using `iter_records`/`iter_payments`, and
`max_records` bounds how many records are read. Errors map onto a small
taxonomy: `HorizonClientError` (non-retryable 4xx), `HorizonRateLimitError`
(429 past the retry budget), `HorizonServerError` (5xx past the retry
budget), and `HorizonTransportError` (unreachable endpoint past the budget).

### Flow analytics from transaction-analytics-api

`funding_optimizer/ingest/analytics_api.py` (issue #33) reads flow summaries
from a `transaction-analytics-api` endpoint. Like the Horizon client it uses
explicit timeouts, deterministic retries with exponential backoff (honoring
`Retry-After` up to `max_retry_after`), and pagination via `iter_records` /
`iter_asset_flows` with `max_records`. Responses are parsed into typed
`AssetFlow`, `AccountFlow`, and `FlowBucket` values; malformed records raise
an indexed `AnalyticsRecordError` and bad envelopes raise
`AnalyticsEnvelopeError`.

```python
from funding_optimizer.ingest import AnalyticsApiClient

client = AnalyticsApiClient("https://analytics-testnet.stellar.org")
flow = client.get_asset_flow("USDC:G...", params={"interval": "1h"})
print(flow.asset, flow.interval)

for bucket in flow.buckets:
    print(bucket.start, bucket.volume, bucket.payments, bucket.net_flow)
```

`ANALYTICS_API_URL` is the default base URL; `ANALYTICS_API_KEY` (or the
`api_key` argument) supplies an optional bearer token, sent only when set.
The client only reads analytics data — it never signs or submits payouts.

### One event model for both ingestion paths

`funding_optimizer/ingest/normalize.py` (issue #34) maps Horizon payment
records and analytics flow records onto the same frozen `NormalizedEvent`,
so downstream features do not need to know which upstream produced a value
movement. Timestamps become canonical UTC (`...Z`), assets are validated
`XLM` / `CODE:ISSUER` identifiers, and `direction` is `in`, `out`, or
`unknown` relative to an optional `subject` account.

```python
from funding_optimizer.ingest import (
    EventSource,
    normalize_analytics_records,
    normalize_horizon_records,
)

horizon_events = normalize_horizon_records(horizon_payments, subject="G...")
flow_events = normalize_analytics_records(flow_records, subject="G...")
assert {event.source for event in horizon_events} == {EventSource.HORIZON}
assert {event.source for event in flow_events} == {EventSource.ANALYTICS}
```

Malformed records fail deterministically with a `NormalizationError` naming
the record index and field, so a bad upstream row never silently disappears.

## Configuration

| Variable | Description |
| --- | --- |
| `STELLAR_NETWORK` | `public` or `testnet` |
| `HORIZON_URL` | Horizon endpoint for on-chain delivery data |
| `ANALYTICS_API_URL` | Optional `transaction-analytics-api` base URL |
| `DATABASE_URL` | Storage for runs, scenarios and reports |

Load the settings with:

```python
from funding_optimizer.config import load_config

config = load_config()
```

`STELLAR_NETWORK` and `HORIZON_URL` are required; `ANALYTICS_API_URL` and
`DATABASE_URL` are optional. Network names and URLs are validated at startup.
`repr(config)` and `config.safe_dict()` redact the database URL so credentials
are not written to logs.

## Responsible use

Allocation models affect real people. Treat outputs as decision support:

- Keep a human in the loop for every allocation decision
- Audit models for bias across regions and recipient groups
- Collect and store only the recipient data you need, and protect it
- Prefer pseudonymous Stellar addresses over personal identifiers in analysis

## Roadmap

- [x] Constraint-based allocation solver
- [x] Delivery-time and cash-out forecasting
- [ ] Scenario comparison reports
- [ ] SDP-compatible disbursement file export
- [ ] Integration with `fraud-detection-sdk` risk scores
- [ ] Dashboard for funders

## Development

After the install steps above, set up the development dependencies and run the
checks:

```bash
pip install -e ".[dev]"   # or: make install
pre-commit install        # optional, runs lint/format on commit
make check                # lint + typecheck + tests
```

Individual checks are `make lint`, `make format`, `make typecheck`, and
`make test`. See [CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow,
branching model, and review expectations.

## Contributing

Issues and PRs are welcome. Please open an issue first for larger changes. See `CONTRIBUTING.md`.

## Related projects

- [`transaction-analytics-api`](https://github.com/StellarSI/transaction-analytics-api): real-time flow data
- [`fraud-detection-sdk`](https://github.com/StellarSI/fraud-detection-sdk): anomaly detection
- [Stellar Disbursement Platform](https://developers.stellar.org/platforms/stellar-disbursement-platform) · [Stellar Developers](https://developers.stellar.org)

## License

MIT. See [LICENSE](LICENSE).

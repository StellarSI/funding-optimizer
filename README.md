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
- [ ] Delivery-time and cash-out forecasting
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

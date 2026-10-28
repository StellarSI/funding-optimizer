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
fails with a diagnostic instead of returning a plan.</think><tool_call>edit<arg_key>newString</arg_key><arg_value>### maximize_reach solver

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

- [ ] Constraint-based allocation solver
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

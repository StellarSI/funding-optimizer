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

## Configuration

| Variable | Description |
| --- | --- |
| `STELLAR_NETWORK` | `public` or `testnet` |
| `HORIZON_URL` | Horizon endpoint for on-chain delivery data |
| `ANALYTICS_API_URL` | Optional `transaction-analytics-api` base URL |
| `DATABASE_URL` | Storage for runs, scenarios and reports |

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

## Contributing

Issues and PRs are welcome. Please open an issue first for larger changes. See `CONTRIBUTING.md`.

## Related projects

- [`transaction-analytics-api`](https://github.com/StellarSI/transaction-analytics-api): real-time flow data
- [`fraud-detection-sdk`](https://github.com/StellarSI/fraud-detection-sdk): anomaly detection
- [Stellar Disbursement Platform](https://developers.stellar.org/platforms/stellar-disbursement-platform) · [Stellar Developers](https://developers.stellar.org)

## License

MIT. See [LICENSE](LICENSE).

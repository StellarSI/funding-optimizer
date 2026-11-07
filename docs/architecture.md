# Architecture & Data Pipeline Contracts

This document formalizes the feature ingestion pipeline, model interfaces, and export data contracts for `funding-optimizer`, fulfilling **Issue #99** as specified under `README -> How it works`.

---

## 1. High-Level System Architecture

```text
  Donor & Protocol Data                 On-Chain Verification
(budgets, cohorts, constraints)   (transaction-analytics-api / Horizon)
             │                                     │
             └──────────────────┬──────────────────┘
                                ▼
                         Feature Pipeline
                                ▼
                    Forecast Models + Optimizer
                                ▼
                 Allocation Plan · Scenarios · Reports
                                │
                                ▼
             Optional: Payload for SDP / Custom Payout
```

### Core Invariant
**The optimizer is an advisory recommendation engine only.** It strictly produces allocation plans and SDP export payloads; it **never signs or submits on-chain transactions** and never manages private keys.

---

## 2. Ingestion Pipeline & Schemas

### 2.1 Program Candidate Model
Candidate recipient programs ingested from grant rounds or NGO proposals adhere to the following schema:

```python
class Program:
    id: str               # Unique program slug / grant identifier
    region: str           # Target geographic region (e.g. 'Africa', 'Asia')
    expected_reach: float # Projected human beneficiaries reached per unit funding
    min_share: float      # Minimum allocation floor as a percentage of total budget [0.0, 1.0]
```

### 2.2 Regional Fairness Floor Contract
Ensures geographic diversity and equitable capital allocation:

```python
class RegionFloor:
    region: str   # Regional identifier matching program cohorts
    floor: float  # Minimum percentage of total budget allocated to the region [0.0, 1.0]
```

---

## 3. Optimization Engine Architecture

### 3.1 Linear Program Formulation
The core optimizer formulates a bounded linear program solved deterministically via SciPy/HiGHS:

$$\max \quad \sum_{i} \text{expected\_reach}_i \cdot x_i \iff \min \quad -c^T x$$

Subject to:
1. **Total Budget Conservation:**  
   $$\sum_{i=1}^n x_i = \text{Budget}$$
2. **Program Minimum Allocation Caps:**  
   $$x_i \ge \text{min\_share}_i \cdot \text{Budget} \quad \forall i$$
3. **Regional Floor Constraints:**  
   $$\sum_{i \in \text{Region}_r} x_i \ge \text{floor}_r \cdot \text{Budget} \quad \forall r$$
4. **Non-Negativity:**  
   $$x_i \ge 0 \quad \forall i$$

---

## 4. SDP (Stellar Disbursement Platform) Export Contract

When handing off recommended allocations to SDP for bulk payout execution, payloads adhere to the canonical schema defined in `funding_optimizer.sdp`:

```json
{
  "batch_id": "cohort-2026-q4",
  "created_at": "2026-11-05T10:00:00Z",
  "idempotency_key": "sdp:alloc:8f4b1192837bc210...",
  "item_count": 3,
  "total_amount": "500000.0000000",
  "disbursements": [
    {
      "amount": "200000.0000000",
      "asset_code": "USDC",
      "asset_issuer": "GDQERFIB4LWK7P7MDEVSCWR7DPUWV3NY3DTQEVFL4NAT4AQH3ZLLFLM7",
      "recipient_id": "prog-edu-01",
      "stellar_address": "GBBD47IF6LWK7P7MDEVSCWR7DPUWV3NY3DTQEVFL4NAT4AQH3ZLLFLA5"
    }
  ]
}
```

### 4.1 Cryptographic Idempotency Key
- Generated via SHA-256 over canonical, address-sorted recipient dictionaries.
- Guarantees order-independent hashing (`sdp:alloc:<hash>`) to prevent duplicate double-spend disbursements across network retries.

---

## 5. Security & Responsible Use Boundary

1. **Least Privilege:** Connection credentials (`DATABASE_URL`, API keys) are redacted in representations (`safe_dict()`) and excluded from runtime logs.
2. **Data Minimization:** Models store and export public Stellar addresses (`G...`) rather than personal identity data (PII).
3. **Fail-Closed Principle:** Any infeasible constraint configuration (e.g. regional floors summing to $> 1.0$) raises an immediate validation error without producing corrupt export files.

# Scores per Task — Current as of 2026-03-22 11:00 CET

## Scoring: score ≈ raw_points / try_number (best across all tries kept)

## ARAS — Rank 75 | Score 65.94 | 222 submissions

| Task | Tier | Score | Tries | Notes |
|------|------|-------|-------|-------|
| 01 | T1 | 1.50 | 9 | customer/supplier/product. Capped |
| 02 | T1 | 2.00 | 8 | customer/supplier/product. Capped |
| 03 | T1 | 2.00 | 9 | product creation. Capped |
| 04 | T1 | 2.00 | 6 | customer/supplier/product |
| 05 | T1 | 1.33 | 8 | department. Capped |
| 06 | T1 | 1.50 | 7 | customer/supplier/product. Capped |
| 07 | T2 | 2.00 | 10 | invoice payment. Capped |
| 08 | T2 | 2.00 | 10 | order/invoice. Capped |
| 09 | T2 | 2.67 | 7 | order/invoice |
| 10 | T2 | 2.67 | 12 | order→invoice→payment. Capped |
| 11 | T2 | — | 10 | supplier invoice. VAT-locked fix deployed |
| 12 | T2 | — | 9 | payroll. Approach B fallback deployed |
| 13 | T2 | 1.13 | 11 | travel expense. Deliver/approve fix deployed |
| 14 | T2 | 4.00 | 9 | credit note. Capped |
| 15 | T2 | 3.00 | 6 | invoice operations |
| 16 | T2 | 3.00 | 8 | invoice operations |
| 17 | T2 | 3.50 | 7 | invoice operations |
| 18 | T2 | 4.00 | 6 | invoice operations |
| 19 | T3 | 2.45 | 6 | employee from PDF. Checks 10,13 fail (address/phone) |
| 20 | T3 | 0.60 | 4 | monthly closing. Yearend fix deployed — HIGH VALUE |
| 21 | T3 | 2.57 | 6 | employee from PDF offer letter. Check 5 fails |
| 22 | T3 | — | 7 | receipt. Account 7140 fix deployed — HIGH VALUE |
| 23 | T3 | 0.60 | 8 | bank recon. Check 1 always fails. Invoice matching ambiguous. LOW VALUE now |
| 24 | T3 | 2.25 | 5 | corrections. Voucher search fields fix deployed |
| 25 | T3 | 3.40 | 4 | overdue invoice. FIXED — 6/6 perfect |
| 26 | T3 | 3.75 | 6 | supplier invoice from PDF. 10/10 perfect latest |
| 27 | T3 | 6.00 | 6 | custom dimensions. FIXED — 10/10 perfect |
| 28 | T3 | 1.50 | 7 | project ledger analysis. Sequential query fix deployed |
| 29 | T3 | 2.73 | 6 | project full cycle. PUT /project budget fix deployed |
| 30 | T3 | 1.80 | 5 | year-end closing. Tax calc fix deployed — HIGH VALUE |

## Remaining realistic gains
| Task | Current | If perfect next try | Max gain |
|------|---------|-------------------|----------|
| 20 | 0.60 | 10/5 = 2.00 | +1.40 |
| 22 | 0.00 | 10/8 = 1.25 | +1.25 |
| 12 | 0.00 | 8/10 = 0.80 | +0.80 |
| 11 | 0.00 | 8/11 = 0.73 | +0.73 |
| 30 | 1.80 | 10/6 = 1.67 | -0.13 (CAPPED!) |

Total realistic gain: ~3.18 (from tasks 20, 22, 12, 11)
Theoretical ceiling: ~69.12

# NM i AI 2026 — Tripletex Task Analysis

**Team**: DNV GRD - ARAS | **Rank**: 80 | **Score**: 63.59 | **Submissions**: 194 | **Tasks**: 30/30
**Competitor**: DNV - Sink32 | **Rank**: 39 | **Score**: 70.23 | **Submissions**: 143

**Scoring**: `score = raw_points / try_number` (best across all tries kept). Past failures PERMANENTLY reduce max score.

**UPDATE 2026-03-22**: Task 25 fixed (1.50→3.40, 6/6 perfect at try 3 but penalty caps it). Task 28 still 1.50 (3/5 fail). Gap to Sink32 = -6.64.

**Date**: 2026-03-22 | **Scoring**: `score = correctness × tier_multiplier × efficiency_bonus`

---

## Score Comparison: ARAS vs Sink32

| Task | Tier | ARAS Score | Tries | Sink32 Score | Tries | Gap | Priority |
|------|------|-----------|-------|-------------|-------|-----|----------|
| 01 | T1 | 1.50 | 9 | 1.33 | 5 | +0.17 | - |
| 02 | T1 | 1.04 | 6 | 2.00 | 3 | -0.96 | LOW |
| 03 | T1 | 2.00 | 9 | 1.50 | 4 | +0.50 | - |
| 04 | T1 | 2.00 | 5 | 1.50 | 6 | +0.50 | - |
| 05 | T1 | 1.33 | 8 | 1.25 | 5 | +0.08 | 7/7 at try 8 but 0.875 < existing 1.33 |
| 06 | T1 | 1.50 | 7 | 1.50 | 5 | 0.00 | - |
| 07 | T2 | 2.00 | 8 | 2.00 | 2 | 0.00 | - |
| 08 | T2 | 2.00 | 5 | 2.00 | 5 | 0.00 | - |
| 09 | T2 | 2.67 | 7 | 2.67 | 4 | 0.00 | - |
| 10 | T2 | 2.67 | 11 | 2.67 | 7 | 0.00 | - |
| 11 | T2 | **0.00** | 8 | **0.00** | 6 | 0.00 | HIGH |
| 12 | T2 | **0.00** | 8 | 1.00 | 4 | **-1.00** | HIGH |
| 13 | T2 | 1.13 | 4 | 1.13 | 4 | 0.00 | MED |
| 14 | T2 | 4.00 | 8 | 4.00 | 5 | 0.00 | - |
| 15 | T2 | 3.00 | 6 | 3.33 | 6 | -0.33 | LOW |
| 16 | T2 | 3.00 | 8 | 3.33 | 7 | -0.33 | LOW |
| 17 | T2 | 3.50 | 6 | 3.20 | 4 | +0.30 | - |
| 18 | T2 | 4.00 | 5 | 4.00 | 4 | 0.00 | - |
| 19 | T3 | 2.45 | 5 | 2.45 | 6 | 0.00 | MED |
| 20 | T3 | 0.60 | 4 | 0.60 | 3 | 0.00 | HIGH |
| 21 | T3 | 2.57 | 4 | 2.36 | 4 | +0.21 | - |
| 22 | T3 | **0.00** | 6 | **0.00** | 6 | 0.00 | HIGH |
| 23 | T3 | 0.60 | 5 | 0.60 | 4 | 0.00 | HIGH |
| 24 | T3 | 2.25 | 5 | 3.60 | 5 | **-1.35** | HIGH |
| 25 | T3 | 3.40 | 3 | 4.80 | 4 | -1.40 | 6/6 at try 3 but capped by try penalty |
| 26 | T3 | 3.75 | 5 | 3.75 | 5 | 0.00 | - |
| 27 | T3 | 4.60 | 5 | 6.00 | 4 | **-1.40** | HIGH |
| 28 | T3 | 1.50 | 4 | 1.50 | 3 | 0.00 | MED |
| 29 | T3 | 2.73 | 5 | 2.73 | 4 | 0.00 | MED |
| 30 | T3 | 1.80 | 4 | 1.80 | 3 | 0.00 | HIGH |

**Total gap: -6.91 points**

### Biggest opportunities to close the gap:
1. **Task 25** (-3.30): Overdue invoice + reminder + partial payment — T3 4x multiplier
2. **Task 27** (-1.40): Custom accounting dimensions — T3 4x multiplier
3. **Task 24** (-1.35): Corrections/reversals — T3 4x multiplier
4. **Task 12** (-1.00): Payroll — T2 2x multiplier
5. **Task 20** (0.00 but very low score 0.60): Monthly closing — T3 4x multiplier
6. **Task 30** (0.00 but low score 1.80): Annual closing — T3 4x multiplier

---

## System Architecture

```
Competition Platform → FastAPI (Cloud Run) → Router (Gemini Flash) → Sub-Agent (Gemini Flash)
                                                                          ↕
                                                                    Tripletex v2 API
```

- **LLM**: Gemini 2.5 Flash (temperature=0.0) via `langchain-google-genai`
- **Router**: ROUTER_PROMPT classifies task → one of 16 categories
- **Sub-Agent**: Category-specific prompt + ReAct loop (search/create/action tools)
- **PDF**: pymupdf4llm (layout-preserving Markdown) → Gemini vision fallback
- **Deployment**: Google Cloud Run (`ainm2026`, `europe-north1`)

---

## Task-by-Task Deep Analysis

### TIER 1 (x1 multiplier)

---

#### Task 01 | Customer/Supplier/Product | Score: 1.50 | 9 tries
**Router**: customer/supplier/product
**What it does**: Create a customer or supplier with basic details (name, org number, address, email)
**Example prompt**: "Opprett kunden Snohetta AS med organisasjonsnummer 969719878. Adressen er Industriveien 148, 2317 Hamar. E-post: post@snhetta.no."
**Status**: Working. Multiple successful submissions.
**Issues**: Score not maxed — likely efficiency penalty from too many tries early on.
**Improvement**: None needed. Score will improve with fewer tries on re-submissions.

---

#### Task 02 | Customer/Supplier/Product | Score: 1.04 | 6 tries
**Router**: customer/supplier/product
**What it does**: Similar entity creation, possibly with more fields
**Status**: Working but lower score than Task 01.
**Sink32 comparison**: They score 2.00 with only 3 tries (much more efficient).
**Issues**: Likely wasted iterations in early submissions due to missing fields or wrong approach.
**Improvement**: LOW priority. T1 tasks have low multiplier.

---

#### Task 03 | Product Creation | Score: 2.00 | 9 tries
**Router**: product
**What it does**: Create products with specific product number, price, and VAT rate.
**Example prompt**: "Opprett produktet 'Frokostblanding' med produktnummer 1391. Prisen er 37450 kr eksklusiv MVA, og MVA-sats for naeringsmidler pa 15 % skal brukes."
**Known issues (fixed)**:
- Try 9 regressed to 0 because vatType search used `percentage=` instead of `number=` (invalid filter → 56 results → wrong VAT)
- Agent wasted iterations trying to create product units (all 422 errors)
**Fix applied**: PRODUCT_PROMPT now uses `?number=31` for 15% VAT, `?number=3` for 25%. "Do NOT create product units."
**Status**: Fixed, working.

---

#### Task 04 | Customer/Supplier/Product | Score: 2.00 | 5 tries
**Router**: customer/supplier/product
**Status**: Working well. Good efficiency (5 tries).

---

#### Task 05 | Department Creation | Score: 1.33 | 7 tries
**Router**: department
**Example prompt**: "Crea tres departamentos en Tripletex: 'Drift', 'Administrasjon' y 'Lager'."
**Status**: Working. Score reflects early learning curve.

---

#### Task 06 | Customer/Supplier/Product | Score: 1.50 | 7 tries
**Router**: customer/supplier/product
**Status**: Working.

---

### TIER 2 (x2 multiplier)

---

#### Task 07 | Invoice Payment | Score: 2.00 | 8 tries
**Router**: order_invoice
**What it does**: Find an existing invoice and register payment (or reverse a payment).
**Example prompt**: "Reverse the payment on invoice from Costa Brava SL..."
**Status**: Working. Score could improve with efficiency.
**Observed behavior**: Agent searches customer → finds invoice → registers payment via /:payment action.

---

#### Task 08 | Order/Invoice | Score: 2.00 | 5 tries
**Router**: order_invoice
**Status**: Working well.

---

#### Task 09 | Order/Invoice | Score: 2.67 | 7 tries
**Router**: order_invoice
**Status**: Working. Recently scored 8/8 (perfect).
**Latest submission**: 6/6 checks passed. Score updated to 4.00.

---

#### Task 10 | Order→Invoice→Payment | Score: 2.67 | 11 tries
**Router**: order_invoice
**What it does**: Full cycle — create order with products, convert to invoice, register payment.
**Example prompt**: "Opprett en ordre for kunden Fjordkraft AS med produktene Opplaering (7579) til 14650 kr og Webdesign (2292) til 11800 kr. Konverter ordren til faktura og registrer full betaling."
**Status**: Working. Many tries due to early learning.
**Observed behavior**: Search customer + products in parallel → create order → /:invoice action → /:payment action.
**Common errors**: Sometimes 422 on /:invoice when order lines have wrong VAT or currency fields.

---

#### Task 11 | Supplier Invoice via Voucher | Score: 0.00 | 8 tries ❌
**Router**: supplier_invoice
**What it does**: Record a received supplier invoice as a ledger voucher with correct VAT treatment.
**Example prompt**: "We have received invoice INV-2026-8735 from Brightstone Ltd (org no. 913701585) for 8500 NOK including VAT. The amount relates to office services (account 7100). Register with correct input VAT (25%)."

**Root cause analysis** (from logs at 23:09 UTC):
1. Agent searched supplier, account 7100, account 2400, vatType number=1, voucherType — all in parallel ✓
2. Tried POST /ledger/voucher with account 7100 + vatType → **422: "Kontoen 7100 Bilgodtgjørelse oppgavepliktig er låst til mva-kode 0"**
3. Account 7100 is locked to VAT code 0 (no VAT treatment allowed)
4. Agent CHANGED account from 7100 to 6800 (Kontorrekvisita) → all 4 checks failed because wrong account used

**Fix applied** (2026-03-22):
- "Account selection priority" rule: NEVER change task-specified account
- Manual VAT split fallback: when account is locked to no-VAT, split into 3 rows: expense (net, no vatType) + 2710 input VAT (vat amount) + 2400 AP (-gross)
- Account 2710 now searched in parallel with other accounts

**Variant analysis**: The prompt explicitly says "account 7100" — all variants likely specify an account that may or may not be VAT-locked. The fix handles both cases (try auto-VAT first, fall back to manual split).

---

#### Task 12 | Payroll | Score: 0.00 | 8 tries ❌
**Router**: payroll
**What it does**: Run payroll for an employee — base salary + optional bonus.
**Example prompt**: "Run payroll for Emily Lewis (emily.lewis@example.org) for this month. The base salary is 53400 NOK. Add a one-time bonus of 16900 NOK on top of the base salary."

**Root cause analysis** (from logs):
- Primary failure: **Employee has no active employment** → /salary/transaction returns 422
- PAYROLL_PROMPT has explicit Step 3: "If NO employment found, create employment + employment details"
- But early submissions used older prompt without this step
- Recent submissions (after prompt fix) haven't hit this task again

**Log evidence** (08:42 UTC):
```
POST /salary/transaction?generateTaxDeduction=true → 422
"Validering feilet" (validation failed)
```

**Key issue**: The employment creation requires many fields (employmentType, employmentForm, remunerationType, workingHoursScheme, percentageOfFullTimeEquivalent, annualSalary). Missing any of these causes 422.

**What Sink32 does differently**: They score 1.00 with 4 tries — they must have the employment prerequisite working.

**Status**: Prompt was fixed but untested. Need a Task 12 submission to verify.

---

#### Task 13 | Travel Expense | Score: 1.13 | 4 tries
**Router**: travel_expense
**What it does**: Create travel expense with per diem + costs (flight, taxi).
**Example prompt**: "Registrer en reiseregning for Ragnhild Bakken for 'Kundebesok Kristiansand'. Reisen varte 4 dager med diett (dagsats 800 kr). Utlegg: flybillett 5450 kr og taxi 550 kr."

**Observed behavior** (22:40 UTC — most recent):
1. Parallel search: employee, costCategory (36 results), paymentType (1 result), rateCategory (3 results) ✓
2. Create travelExpense with travelDetails ✓
3. Create flight cost ✓
4. Create taxi cost ✓
5. Create perDiemCompensation (rateCategory id=740, count=4, rate=800) ✓
6. **No deliver/approve step** ❌

**Root cause**: Travel expense is NOT delivered/approved after creation. Competition checks likely require the expense to be in "delivered" or "approved" state.

**Fix applied** (2026-03-22): TRAVEL_EXPENSE_PROMPT now has mandatory Step 6: "ALWAYS deliver and approve" with action_endpoint calls.

**Other potential issues**:
- rateCategory selection: 3 results returned, agent picked id=740 — may not be the correct per diem rate
- overnightAccommodation="HOTEL" — might need to match task specifics
- Dates: agent invented departure/return dates (not specified in task) — could be wrong

---

#### Task 14 | Credit Note | Score: 4.00 | 8 tries ✅
**Router**: order_invoice
**What it does**: Create a credit note for an existing invoice.
**Example prompt**: "Der Kunde Waldstein GmbH hat die Rechnung fur 'Beratungsstunden' (36900 NOK) reklamiert. Erstellen Sie eine vollstaendige Gutschrift."
**Status**: Working perfectly. Maximum score achieved.

---

#### Task 15 | Invoice Operations | Score: 3.00 | 6 tries
**Router**: order_invoice
**What it does**: Create invoice with multiple product lines, possibly mixed VAT rates.
**Example prompt**: "Opprett en faktura til kunden Brattli AS med tre produktlinjer: Skylagring (7246) til 23300 kr med 25% MVA, Systemutvikling (9400) til 3300 kr med 15% MVA (naeringsmiddel), og Nettverkstjeneste (1933) til 10850 kr med 0% MVA."
**Status**: Working, but Sink32 scores 3.33 (+0.33).
**Issue**: Possibly creating invoice with wrong VAT setup or missing VAT on some lines.

---

#### Task 16 | Invoice Operations | Score: 3.00 | 8 tries
**Router**: order_invoice
**Status**: Working. Recently scored 8/8 (perfect). Score will update.
**Latest**: 4/4 checks passed, score now 4.00.

---

#### Task 17 | Invoice Operations | Score: 3.50 | 6 tries ✅
**Router**: order_invoice
**Status**: Working. Recently scored 13/13 (perfect).

---

#### Task 18 | Invoice Operations | Score: 4.00 | 5 tries ✅
**Router**: order_invoice
**Status**: Working perfectly.

---

### TIER 3 (x4 multiplier)

---

#### Task 19 | Employee Onboarding from PDF Contract | Score: 2.45 | 5 tries
**Router**: employee
**What it does**: Extract employee details from attached PDF employment contract and create employee in Tripletex.
**Example prompt**: "Du har mottatt en arbeidskontrakt (se vedlagt PDF). Opprett den ansatte i Tripletex med alle detaljer fra kontrakten: personnummer, fodselsdato, avdeling, stillingskode, lonn, stillingsprosent og startdato."

**Key challenge**: PDF extraction must capture ALL fields — the competition checks for:
- Name, date of birth, national ID ✓
- Department ✓
- Occupation code ✓ (search /employee/employment/occupationCode)
- Salary, employment percentage ✓
- Start date ✓
- **Address** ❓ (Checks 10, 13/15 fail — likely missing)
- **Phone** ❓ (likely missing)

**Fix applied**: EMPLOYEE_PROMPT has explicit checklist: "Before POST /employee, verify ALL fields: name, DOB, address, phone, national ID, employee number, department, occupation code, salary, percentage, start date, bank account, email, comments."

**PDF extraction**: pymupdf4llm now extracts layout-preserving Markdown from PDF contracts. Confirmed working in Task 26 logs — tables extracted perfectly.

**Improvement needed**: Ensure the agent actually reads and uses address/phone from the extracted PDF text. The checklist is in the prompt but the agent may still skip fields it doesn't find easily.

---

#### Task 20 | Monthly Closing | Score: 0.60 | 4 tries ⚠️
**Router**: yearend (was misrouted as `ledger` before refactor)
**What it does**: Monthly period closing — depreciation, prepaid costs, salary accrual.
**Example prompt**: "Utfor manedsavslutning for mars 2026. Periodiser forskuddsbetalt kostnad (6500 kr per maned fra konto 1700 til kostkonto). Bokfor manedlig avskrivning for et driftsmiddel med anskaffelseskost 104900 kr og levetid 5 ar. Kontroller at saldobalansen gar i null. Bokfor ogsa en lonnsavsetning (debet lonnskostnad konto 5000, kredit paloppt lonn konto 2900)."

**Root cause analysis** (22:23 UTC):
The task explicitly asks for 4 things:
1. Prepaid cost periodization (6500 kr from 1700 to expense) → ✓ Created
2. Monthly depreciation (104900/5/12 = 1748 kr) → ✓ Created
3. Balance sheet verification → ❌ SKIPPED
4. Salary accrual (debit 5000, credit 2900) → ❌ SKIPPED

**The agent stopped after 2 vouchers** (25 seconds, only 5 API calls). It didn't create the salary accrual or check the balance sheet. The salary accrual task says "bokfor ogsa en lonnsavsetning" but doesn't specify an amount — the agent should search /balanceSheet to find the amount.

**Historical routing issue**: Before the router refactor, monthly closing tasks were routed as `ledger` and used the generic LEDGER_PROMPT (no specific closing instructions). This explains the 0.60 score from early tries.

**Fix applied** (2026-03-22):
- YEAREND_PROMPT: Added Step 0 "ENUMERATE ALL TASKS" — forces agent to list every instruction before starting
- Explicit "DO NOT SKIP THIS!" warnings on salary accrual and tax steps
- Added /balanceSheet endpoint with "NOT /ledger/balanceSheet — causes 403!" warning
- Salary accrual: "If no amount given, search /balanceSheet for account 5000"
- Increased max_iter from 20 → 30

---

#### Task 21 | Employee from PDF Offer Letter | Score: 2.57 | 4 tries
**Router**: employee
**What it does**: Full employee onboarding from PDF offer letter — more complex than Task 19.
**Example prompt**: "Du har mottatt et tilbudsbrev (se vedlagt PDF) for en ny ansatt. Utfor komplett onboarding: opprett den ansatte, tilknytt riktig avdeling, sett opp ansettelsesforhold med stillingsprosent og arslonn, og konfigurer standard arbeidstid."
**Status**: Working reasonably. Check 5 fails — likely occupation code mismatch or missing PDF field.
**Improvement**: Similar to Task 19 — ensure PDF extraction captures all fields.

---

#### Task 22 | Receipt/Kvittering Booking (PDF) | Score: 0.00 | 6 tries ❌
**Router**: receipt (was misrouted as `ledger` before refactor)
**What it does**: Extract expense from a receipt image/PDF and book it as a voucher with correct VAT.
**Example prompts**:
- "Wir benotigen die Oppbevaringsboks-Ausgabe aus dieser Quittung in der Abteilung HR"
- "Precisamos da despesa de Tastatur deste recibo registada no departamento HR"
- "We need the Oppbevaringsboks expense from this receipt posted to department Lager"

**Root cause analysis**:
Multiple interacting issues:
1. **Routing**: Before refactor, ALL receipt tasks were classified as `ledger` → used generic prompt without receipt-specific VAT handling
2. **PDF extraction**: Early versions used basic pymupdf text extraction which scrambled receipt layouts. Fixed with pymupdf4llm.
3. **VAT-locked accounts**: Some receipt expense accounts are locked to vatType 0, causing 422 when agent tries to set vatType
4. **Same-row dimension mismatch**: Department on expense posting but not on bank posting → 422
5. **vatType ID confusion**: Agent used vatType number as ID (they're different)

**Log evidence** (20:41 UTC — receipt classified as ledger):
```
422: "postings.department.id: Debet- og kreditposteringer pa samme rad kan ikke ha ulike dimensjoner"
422: "postings.vatType.id: Kontoen 7350 Representasjon, fradragsberettiget er last til mva-kode 0"
```

**Fix applied**:
- RECEIPT_PROMPT: Different row numbers for expense (row 1) and bank (row 2)
- vatType + department go ONLY on expense posting
- "The vatType 'id' from search result is NOT the same as vatType 'number'!"
- PDF extraction upgraded to pymupdf4llm

**Status**: Prompt rewritten, pymupdf4llm deployed, but the new receipt prompt has only been tested ONCE (21:47 UTC). More submissions needed.

---

#### Task 23 | Bank Reconciliation (CSV) | Score: 0.60 | 5 tries ⚠️
**Router**: bank_recon (was ALWAYS misrouted as `ledger` before refactor — 0 correct classifications!)
**What it does**: Match CSV bank statement entries to open customer/supplier invoices in Tripletex.
**Example prompts**:
- "Reconcile the bank statement (attached CSV) against open invoices in Tripletex"
- "Gleichen Sie den Kontoauszug (beigefugte CSV) mit den offenen Rechnungen in Tripletex ab"
- "Avstem bankutskriften (vedlagt CSV) mot apne fakturaer i Tripletex"

**Root cause analysis**:
- ALL 5 bank_recon submissions were classified as `ledger` (before refactor) → used generic LEDGER_PROMPT
- LEDGER_PROMPT has no instructions for matching CSV entries to invoices
- The agent likely tried to create manual vouchers instead of matching payments to invoices

**BANK_RECON_PROMPT** (post-refactor): Has dedicated instructions for:
1. Parse CSV to extract transactions
2. Search /invoice and /supplierInvoice for matching amounts
3. Register payments via /:payment actions
4. Handle partial payments

**Critical issue**: bank_recon has NEVER been correctly routed (0 submissions post-refactor). The ROUTER_PROMPT includes "bank_recon: Bank reconciliation from CSV bank statement (kontoutskrift/extracto bancario/releve bancaire/Kontoauszug/avstemming)" but this hasn't been tested yet.

**Risk**: When bank_recon task appears next, the router might still misclassify it. Need to verify the router works.

---

#### Task 24 | Corrections/Reversals | Score: 2.25 | 5 tries
**Router**: corrections (was sometimes misrouted as `ledger`)
**What it does**: Find and fix 4 errors in existing vouchers: wrong account, duplicate, missing VAT, wrong amount.
**Example prompt**: "Wir haben Fehler im Hauptbuch fur Januar und Februar 2026 entdeckt. Uberprufen Sie alle Belege und finden Sie die 4 Fehler: eine Buchung auf das falsche Konto (Konto 6540 statt 6860), ein doppelter Beleg (Konto 6500, 1500 NOK), eine fehlende MwSt.-Zeile (Konto 6300, 22350 NOK), und ein falscher Betrag (Konto 6540, 11400 gebucht statt 8550)."

**Observed behavior** (22:12 UTC — correctly routed):
- Agent searched 2 months of vouchers → found errors → reversed + recreated
- BUT: Multiple 422 errors during correction:
  - "Summen av posteringene er ikke lik 0" (unbalanced voucher)
  - "Leverandor mangler" (missing supplier on posting)
  - Row 0 errors (system-reserved row)

**Key errors from logs**:
1. Agent created correction vouchers with unbalanced postings
2. Agent omitted supplier reference that existed on original voucher
3. Agent used row=0 (system-generated, causes 422)
4. For "missing VAT" fix, agent created manual 3-row VAT split instead of copying original structure with vatType

**Sink32 comparison**: 3.60 vs our 2.25 (-1.35). They handle corrections better.

**Fix applied**:
- GOLDEN RULE: Copy EXACT same postings from original, only change the ONE thing that was wrong
- Row must start from 1, never 0
- If original has supplier, corrected must have supplier too
- For missing VAT: use vatType + amountGross=NET×1.25 (auto-split)

**Remaining issue**: The corrections involve complex voucher structures. The agent needs to:
1. GET the original voucher with all fields
2. Reverse it
3. Create a new voucher with EXACT same structure, fixing only the error
The prompt covers this but the agent sometimes still creates wrong structures.

---

#### Task 25 | Overdue Invoice + Reminder + Partial Payment | Score: 1.50 | 2 tries ⚠️⚠️⚠️
**Router**: order_invoice
**What it does**: Multi-step invoice lifecycle: create overdue invoice, add reminder fee, create reminder invoice, register partial payment.

**CRITICAL GAP**: Sink32 scores 4.80 vs our 1.50 — a **3.30 point gap**, the single largest gap between our teams.

**Example prompt**: "Find the overdue invoice and post a reminder fee of 70 NOK. Debit accounts receivable (1500), credit reminder fees (3400). Also create an invoice for the reminder fee to the customer and send it. Additionally, register a partial payment of 5000 NOK on the overdue invoice."

**Solution path (endpoints)**:
1. `GET /invoice?invoiceDateFrom=2020-01-01&invoiceDateTo=2030-12-31&fields=id,amount,invoiceDate,invoiceDueDate,customer(id,name)&count=100` — find overdue invoice
2. `GET /ledger/account?number=1500`, `GET /ledger/account?number=3400`, `GET /invoice/paymentType`, `GET /ledger/vatType?number=5` — all in parallel
3. `POST /ledger/voucher` — reminder fee: debit 1500 (+fee, with customer ref), credit 3400 (-fee)
4. `POST /order` with `vatType: {exempt}` on order line → `PUT /order/:invoice` → `PUT /invoice/:send`
5. `PUT /invoice/{overdue_id}/:payment` — partial payment

**Root cause from logs** (submission 1, 11:04 UTC — scored 1.50):
- Agent completed ALL steps: voucher ✓, order → invoice → send ✓, partial payment ✓
- BUT: Reminder fee order created WITHOUT vatType → system applied 25% VAT → invoice amount was 87.50 instead of 70 NOK
- The 422 "Kunde mangler" on first voucher attempt wasted one iteration

**Log evidence** (submission 2, 14:22 UTC):
- `:createReminder` failed 4× with "Minst en sendetype ma oppgis" — boolean params `True/False` (capital) not recognized by API

**Fixes applied** (2026-03-22):
1. solve.py: `_clean_params()` converts Python `True/False` → lowercase `true/false` in URL params
2. ORDER_INVOICE_PROMPT: vatType exempt search added to parallel step 1
3. Order line now explicitly requires `"vatType": {"id": <exempt_vat_id>}` with warning
4. All 3 steps (voucher, invoice, payment) marked as MANDATORY

**Impact**: Fixing VAT on reminder invoice alone should pass check 3/4. Combined with boolean fix, 3+ points recoverable.

---

#### Task 26 | Supplier Invoice from PDF | Score: 3.75 | 5 tries
**Router**: supplier_invoice
**What it does**: Extract supplier invoice details from PDF and register as voucher.
**Example prompt**: "Du har mottatt en leverandorfaktura (se vedlagt PDF). Registrer fakturaen i Tripletex. Opprett leverandoren hvis den ikke finnes."
**Status**: Working reasonably. pymupdf4llm confirmed extracting PDF tables correctly.
**Previous issue**: Agent created supplier with only name+orgNumber but PDF had address and bank account. Fixed: "Include ALL details from the PDF."

---

#### Task 27 | Custom Accounting Dimensions | Score: 4.60 | 5 tries
**Router**: ledger
**What it does**: Create custom accounting dimension with values, then post a voucher linked to a dimension value.
**Example prompts**:
- "Cree una dimension contable personalizada 'Kostsenter' con los valores 'Okonomi' y 'Produktutvikling'. Luego registre un asiento en la cuenta 6340 por 45050 NOK."
- "Create a custom accounting dimension 'Region' with values 'Nord-Norge' and 'Sor-Norge'. Then post a voucher on account 7000 for 13650 NOK."

**Sink32 comparison**: 6.00 vs our 4.60 (-1.40).

**Analysis**: The LEDGER_PROMPT has dedicated instructions for custom dimensions:
1. POST /ledger/accountingDimensionName → get dimensionIndex
2. POST /ledger/accountingDimensionValue (one at a time) → get value IDs
3. Create voucher with freeAccountingDimension1/2/3 on relevant postings

**Possible issues**:
- The voucher might need TWO postings (debit expense + credit bank) but agent might only create one
- The freeAccountingDimension link might use wrong dimensionIndex
- Scoring might check for balanced voucher (both debit and credit)

**Action needed**: Review LEDGER_PROMPT dimension instructions. Ensure voucher has balanced postings (expense debit + bank/liability credit).

---

#### Task 28 | Project Ledger Analysis | Score: 1.50 | 4 tries
**Router**: project
**What it does**: Analyze ledger accounts, find top expense increases, create internal projects.
**Example prompt**: "Os custos totais aumentaram significativamente de janeiro a fevereiro de 2026. Analise o livro razao e identifique as tres contas de despesa com o maior aumento em valor. Crie um projeto interno para cada uma das tres contas com o nome da conta."

**Analysis**: This is a complex multi-step task:
1. Query /balanceSheet for two periods (January + February)
2. Calculate expense increases between periods
3. Find top 3 accounts with largest increase
4. Create a project for each

**Possible issues**:
- /balanceSheet endpoint: might be using /ledger/balanceSheet (403) — fixed now
- Account range: should query 4000-7999 for expenses only
- Project creation: might need specific fields that the agent misses

**Status**: Working but low score. Need to verify /balanceSheet fix helps.

---

#### Task 29 | Project Full Cycle | Score: 2.73 | 5 tries
**Router**: project
**What it does**: Full project lifecycle — create project with budget, register time, add supplier costs, generate invoice.
**Example prompt**: "Registrer 8 timar for Knut Aasen pa aktiviteten 'Utvikling' i prosjektet 'Nettbutikk-utvikling' for Elvdal AS. Timesats: 1800 kr/t. Generer ein prosjektfaktura til kunden."

**Previous issue**: Agent skipped budget update step. Fixed: "MANDATORY BUDGET STEP — do NOT skip!"

**Status**: Fixed but untested with recent code.

---

#### Task 30 | Year-End Closing | Score: 1.80 | 4 tries ⚠️
**Router**: yearend (was misrouted as `ledger` before refactor)
**What it does**: Full annual closing — depreciation for 3 assets + prepaid reversal + tax calculation.
**Example prompt**: "Utfor forenklet arsoppgjor for 2025: 1) Beregn og bokfor arlige avskrivninger for tre eiendeler... 2) Reverser forskuddsbetalte kostnader... 3) Beregn og bokfor skattekostnad (22% av skattbart resultat) pa konto 8700/2920."

**Root cause analysis** (22:42 UTC):
- Agent created depreciation vouchers for 3 assets ✓
- Agent created prepaid cost reversal ✓
- Agent **SKIPPED tax voucher** ❌
- Reason: Agent tried /ledger/balanceSheet → **403 Forbidden**. Correct endpoint is /balanceSheet (no /ledger/ prefix).
- Agent output: "Due to a lack of permissions to access the balance sheet, the tax cost could not be calculated."

**Fix applied** (2026-03-22):
- YEAREND_PROMPT: Step 0 "ENUMERATE ALL TASKS" to prevent skipping
- Explicit /balanceSheet endpoint with "NOT /ledger/balanceSheet — causes 403!" warning
- Step-by-step tax calculation using /balanceSheet queries for revenue (3000-3999) and expenses (4000-8699)
- max_iter increased from 20 → 30

---

## Cross-Cutting Issues

### 1. Router Misclassification (RESOLVED)
**Impact**: Tasks 20, 22, 23, 24, 30 were all routed as `ledger` before the router refactor.
- Before refactor: All 5 bank_recon, 5 receipts, 7 yearend, 2 corrections routed as `ledger`
- After refactor (21:36+ UTC): ALL tasks correctly routed
- **Verification needed**: bank_recon hasn't appeared since refactor — untested

### 2. Recurring Voucher Validation Errors
Most common 422 errors from logs:
| Error | Count | Cause |
|-------|-------|-------|
| "Leverandor mangler" (supplier missing) | 6 | Same-row postings must have identical supplier dimension |
| "Kunde mangler" (customer missing) | 3 | Same-row postings must have identical customer dimension |
| "rad 0 er systemgenererte" (row 0) | 6 | Row 0 is system-reserved, must use row >= 1 |
| "Beskrivelsene ma vaere like" (description) | 1 | Same-row postings must have identical descriptions |
| "ulike dimensjoner" (different dims) | 2 | Same-row postings can't have different department/supplier |
| "last til mva-kode 0" (VAT locked) | 2 | Account locked to no-VAT, can't set vatType |
| "Summen er ikke lik 0" (unbalanced) | 2 | Postings don't sum to zero |
| "Feltet eksisterer ikke" (field not exist) | 3 | Using wrong field names (voucherDate, voucherLines, entries) |

### 3. Token Expiry (NOT a bug)
At 23:05 UTC, one order_invoice submission failed with "Invalid or expired proxy token" on ALL API calls. This happens when a new submission starts while another is still processing — the new submission's token invalidates the old one. Not actionable — it's a competition constraint.

### 4. Efficiency Penalty
Many tasks have high try counts (8-11 tries) from the learning period. Since score = correctness × efficiency_bonus, reducing tries improves scores. But we can't reduce past tries — only future ones matter.

Sink32 generally has fewer tries per task (avg ~4.3 vs our ~5.7), suggesting they got their agent working earlier or had fewer iterations.

---

## Priority Action Plan

### CRITICAL (>2 point potential)
1. **Task 25 — Overdue Invoice/Reminder** (-3.30 gap)
   - Research Tripletex /:createReminder action parameters
   - Add reminder workflow to ORDER_INVOICE_PROMPT
   - Include: reminder fee voucher, reminder creation, partial payment handling

### HIGH (>1 point potential)
2. **Task 27 — Custom Dimensions** (-1.40 gap)
   - Review LEDGER_PROMPT dimension instructions
   - Ensure balanced vouchers (debit + credit) when posting with dimension values
   - Check freeAccountingDimension field mapping

3. **Task 24 — Corrections** (-1.35 gap)
   - Verify GOLDEN RULE fix works in practice
   - Test with actual corrections submission
   - Ensure row >= 1 and supplier copy rules are followed

4. **Task 12 — Payroll** (-1.00 gap)
   - Verify employment creation prerequisite works
   - Test with actual payroll submission
   - Need: employmentType, employmentForm, remunerationType, workingHoursScheme fields

5. **Task 20 — Monthly Closing** (0.60 score, T3 4x)
   - Verify Step 0 "ENUMERATE ALL TASKS" prevents skipping salary accrual
   - Test with actual submission

6. **Task 30 — Annual Closing** (1.80 score, T3 4x)
   - Verify /balanceSheet (not /ledger/balanceSheet) fix works
   - Verify tax calculation step is executed
   - Test with actual submission

### MEDIUM
7. **Task 22 — Receipt** (0.00, T3 4x)
   - New receipt prompt + pymupdf4llm deployed but only 1 test
   - Need more submissions to verify
   - Watch for VAT-locked account handling

8. **Task 23 — Bank Reconciliation** (0.60, T3 4x)
   - bank_recon routing UNTESTED since refactor
   - May need router reinforcement for bank reconciliation keywords
   - BANK_RECON_PROMPT needs real-world testing

9. **Task 13 — Travel Expense** (1.13, T2 2x)
   - Mandatory deliver/approve added
   - Need to verify per diem rateCategory selection is correct

---

## Execution Time Analysis

| Task Type | Avg Time | Min | Max | Concern |
|-----------|----------|-----|-----|---------|
| customer | 6.2s | 6.2s | 6.2s | Fine |
| department | 6.7s | 6.6s | 6.8s | Fine |
| supplier | 10.3s | 10.3s | 10.3s | Fine |
| payroll | 13.8s | 12.0s | 15.0s | Fine |
| supplier_invoice | 20.6s | 14.7s | 33.2s | OK |
| receipt | 21.1s | 21.1s | 21.1s | OK (includes PDF extraction) |
| employee | 23.5s | 11.4s | 43.2s | OK (PDF tasks take longer) |
| product | 24.4s | 24.4s | 24.4s | OK |
| travel_expense | 24.6s | 17.2s | 33.6s | OK |
| yearend | 30.7s | 30.4s | 31.0s | OK |
| project | 32.0s | 9.6s | 115.9s | ⚠️ High variance |
| order_invoice | 33.5s | 8.8s | 253.5s | ⚠️ Very high max |
| ledger | 33.5s | 12.4s | 113.7s | ⚠️ Catch-all, high variance |
| corrections | 60.9s | 20.6s | 104.7s | ⚠️ Complex but expected |

Most tasks complete in 15-35 seconds. `corrections` is naturally slow (searching many vouchers). `order_invoice` has one outlier at 253s — possibly a complex multi-step task or retries.

---

## Log Statistics

**Total logged submissions**: 108 (of 190 total — 82 were before logging started or logs rotated)

**Task type distribution in logs**:
| Type | Count | Notes |
|------|-------|-------|
| ledger | 25 | Includes 21 misrouted tasks (pre-refactor) |
| order_invoice | 24 | Covers Tasks 7-10, 14-18, 25 |
| project | 18 | Covers Tasks 28, 29 |
| employee | 14 | Covers Tasks 19, 21 |
| supplier_invoice | 8 | Covers Tasks 11, 26 |
| travel_expense | 4 | Task 13 |
| corrections | 3 | Task 24 (post-refactor only) |
| payroll | 3 | Task 12 |
| department | 2 | Task 5 |
| product | 2 | Task 3 |
| yearend | 2 | Tasks 20, 30 (post-refactor only) |
| receipt | 1 | Task 22 (post-refactor only) |
| customer | 1 | Tasks 1, 2, 4, 6 |
| supplier | 1 | Tasks 1, 2, 4, 6 |
| bank_recon | 0 | Task 23 — NEVER correctly routed! |

---

## Key Takeaways

1. **Router fix was the biggest win** — before the refactor, 21 out of 108 submissions were misrouted to generic `ledger`. Post-refactor, all tasks route correctly.

2. **Task 25 is our biggest opportunity** — 3.30 point gap to Sink32. This is a T3 task with 4x multiplier. Fixing the reminder workflow could gain 3+ points alone.

3. **bank_recon has never been correctly routed** — 0 submissions post-refactor. High risk for next submission.

4. **Most T2 tasks are near-optimal** — Tasks 7-10, 14-18 all score 2.00-4.00. Focus should be on T3 tasks where the 4x multiplier makes each improvement worth more.

5. **Recurring voucher errors** indicate the agent still struggles with same-row constraints, row numbering, and VAT-locked accounts. These are addressed in prompt fixes but need real-world verification.

6. **Sink32 is more efficient** — they average fewer tries per task, suggesting their agent was more reliable from the start. We can't fix past tries but improving reliability going forward will help.

---

## Appendix: Endpoints & Solution Paths by Task

### T1 Tasks (1-6): Entity Creation

**Endpoints used**: `/customer POST`, `/supplier POST`, `/product POST`, `/department POST`, `/employee GET`
**Solution path**: Single API call to create entity with ALL fields from task prompt.
- Always set BOTH `physicalAddress` AND `postalAddress`
- Products: search `/ledger/vatType?number=3` (25%) or `?number=31` (15%) first
- Departments: `departmentNumber` is STRING, not int

### T2 Tasks (7-10, 14-18): Order/Invoice Workflows

**Endpoints used**:
- `GET /customer?organizationNumber=X`, `GET /product?number=X` — find entities
- `GET /ledger/account?number=1920` — bank account prerequisite
- `POST /order` → `PUT /order/{id}/:invoice` → `PUT /invoice/{id}/:payment`
- `PUT /invoice/{id}/:send`, `PUT /invoice/{id}/:createCreditNote`

**Solution path**:
1. Parallel search: customer + products + bank account + payment types + vatType
2. Create/update bank account if needed (number "12345678903")
3. Create order with product IDs and vatType per line
4. Convert to invoice → register payment → send if needed

**Key rules**:
- `unitPriceExcludingVatCurrency` (NOT `unitCostCurrency`)
- `/invoice GET` REQUIRES `invoiceDateFrom` + `invoiceDateTo`
- Fields filter: use parentheses `customer(id,name)` NOT dots `customer.id`
- Payment reversal: use NEGATIVE `paidAmount`
- Credit note: use `/:createCreditNote` with today's date

### Task 11: Supplier Invoice via Voucher

**Endpoints used**:
- `GET /supplier?organizationNumber=X`, `POST /supplier` if not found
- `GET /ledger/account?number=<expense>`, `GET /ledger/account?number=2400` (AP)
- `GET /ledger/account?number=2710` (input VAT — for fallback)
- `GET /ledger/vatType?number=1` (incoming 25% VAT)
- `GET /ledger/voucherType` — find "Leverandorfaktura" type
- `POST /ledger/voucher` with `vendorInvoiceNumber` and `voucherType`

**Solution path**:
1. Parallel search: supplier + expense account + 2400 + 2710 + vatType + voucherType
2. Try auto-VAT: expense posting with `vatType` + AP posting, same row, same description+supplier
3. If 422 "last til mva-kode 0": fallback to MANUAL VAT SPLIT (3 rows: net expense, VAT on 2710, gross on 2400)
4. NEVER change the task-specified account!

### Task 12: Payroll

**Endpoints used**:
- `GET /employee?email=X`, `GET /salary/type`
- `GET /employee/employment?employeeId=X` — check if employment exists
- `POST /employee/employment` + `POST /employee/employment/details` — create if missing
- `POST /salary/transaction?generateTaxDeduction=true`

**Solution path**:
1. Parallel: search employee + salary types
2. Check employment exists → create if not (CRITICAL prerequisite)
3. Create salary transaction with payslips + specifications

### Task 13: Travel Expense

**Endpoints used**:
- `GET /employee?email=X`
- `GET /travelExpense/paymentType`, `GET /travelExpense/costCategory`
- `GET /travelExpense/rateCategory?type=PER_DIEM&dateFrom=X&dateTo=Y&isValidDomestic=true`
- `POST /travelExpense` with `travelDetails`
- `POST /travelExpense/cost` (one at a time, sequentially!)
- `POST /travelExpense/perDiemCompensation`
- `POST /travelExpense/mileageAllowance`
- `PUT /travelExpense/:deliver` → `PUT /travelExpense/:approve`

**Solution path**:
1. Parallel: employee + paymentType + costCategory + rateCategory
2. Create travelExpense with travelDetails
3. Create costs ONE AT A TIME (parallel causes 409)
4. Create per diem / mileage
5. ALWAYS deliver + approve

### Task 19, 21: Employee Onboarding from PDF

**Endpoints used**:
- `POST /employee` with ALL fields from PDF
- `POST /employee/employment` + `POST /employee/employment/details`
- `GET /employee/employment/occupationCode?code=X`
- `GET /department?name=X`
- `POST /employee/standardTime`

**Solution path**:
1. pymupdf4llm extracts PDF → structured Markdown
2. Parse ALL fields: name, DOB, address, phone, national ID, department, occupation code, salary, percentage, start date, bank account, email
3. Search department + occupation code
4. Create employee → employment → employment details → standard time

### Task 20, 30: Period Closing (Monthly/Annual)

**Endpoints used**:
- `GET /ledger/account?number=X` for each account (6010, 1209, 1700, 5000, 2900, 8700, 2920)
- `POST /ledger/voucher` — separate voucher per entry
- `GET /balanceSheet?dateFrom=X&dateTo=Y&accountNumberFrom=A&accountNumberTo=B` (NOT /ledger/balanceSheet!)

**Solution path**:
1. ENUMERATE all tasks from prompt (Step 0 — mandatory!)
2. Search all accounts in parallel
3. Create SEPARATE vouchers for: depreciation, prepaid, salary accrual, tax
4. Tax: query `/balanceSheet` for revenue (3000-3999) and expenses (4000-8699), calculate 22%
5. Verify balance sheet if task says "kontroller saldobalansen"

### Task 22: Receipt Booking from PDF

**Endpoints used**:
- `GET /ledger/account?number=X` (expense account), `GET /ledger/account?number=1920` (bank)
- `GET /ledger/vatType?number=1` (input VAT)
- `GET /department?name=X`
- `POST /ledger/voucher`

**Solution path**:
1. pymupdf4llm extracts receipt → identify item, amount, VAT
2. Search accounts + vatType + department
3. Create voucher: expense posting (row 1) with vatType + department, bank posting (row 2) without dimensions
4. Different rows for expense and bank! Same-row dimension rules don't apply

### Task 23: Bank Reconciliation from CSV

**Endpoints used**:
- CSV file parsed from attachment
- `GET /invoice?invoiceDateFrom=X&invoiceDateTo=Y&fields=id,amount,invoiceDate,customer(id,name)`
- `GET /supplierInvoice?invoiceDateFrom=X&invoiceDateTo=Y`
- `PUT /invoice/{id}/:payment`
- `PUT /supplierInvoice/{id}/:addPayment`

**Solution path**:
1. Parse CSV bank statement
2. Search all invoices + supplier invoices
3. Match CSV entries to invoices by amount/date
4. Register payments for matches

### Task 24: Corrections/Reversals

**Endpoints used**:
- `GET /ledger/voucher?dateFrom=X&dateTo=Y&fields=id,date,description,postings(*)` — search vouchers
- `PUT /ledger/voucher/{id}/:reverse` with params `{"date": "YYYY-MM-DD"}`
- `POST /ledger/voucher` — create corrected voucher

**Solution path** (GOLDEN RULE):
1. Search vouchers for Jan + Feb
2. For each error: REVERSE the original voucher
3. Create corrected voucher: COPY EXACT structure from original, change ONLY the error
4. Row must start from 1 (never 0)
5. If original has supplier ref, corrected must too
6. For missing VAT: use `vatType` + `amountGross=NET×1.25` (auto-split)

### Task 25: Overdue Invoice + Reminder

**Endpoints used**:
- `GET /invoice` — find overdue invoice
- `GET /ledger/account?number=1500,3400`
- `GET /ledger/vatType?number=5` (VAT exempt)
- `POST /ledger/voucher` — reminder fee voucher
- `POST /order` → `PUT /order/:invoice` → `PUT /invoice/:send` — reminder invoice
- `PUT /invoice/{id}/:payment` — partial payment

**Solution path**: See Task 25 analysis above.

### Task 26: Supplier Invoice from PDF

Same as Task 11 but with PDF attachment. pymupdf4llm extracts supplier details (name, orgNumber, address, bank account).

### Task 27: Custom Accounting Dimensions

**Endpoints used**:
- `POST /ledger/accountingDimensionName` — create dimension, get `dimensionIndex`
- `POST /ledger/accountingDimensionValue` — create values one at a time
- `GET /ledger/account?number=X`, `GET /ledger/account?number=1920`
- `POST /ledger/voucher` with `freeAccountingDimension<N>` on expense posting

**Solution path**:
1. Search expense account + bank account 1920
2. Create dimension → get dimensionIndex
3. Create values sequentially (number is STRING: "1", "2")
4. Create BALANCED voucher: expense (row 1, with freeAccountingDimension) + bank credit (row 2)
5. dimensionIndex determines freeAccountingDimension1/2/3

### Task 28-29: Project Management

**Endpoints used**:
- `GET /project?name=X`, `GET /customer`, `GET /employee`, `GET /activity`
- `POST /project`, `PUT /project` (set fixedprice, budget)
- `POST /project/projectActivity`
- `POST /timesheet/entry`
- `POST /order` → `PUT /order/:invoice` — project invoice
- `GET /balanceSheet` — for ledger analysis tasks

**Solution path** (Task 29 full cycle):
1. Parallel: search project + customer + employee + activity
2. MANDATORY: update project with budget (`isFixedPrice: true, fixedprice: X`)
3. Link activity to project
4. Create timesheet entries
5. Create order → invoice for project

---

## Fixes Applied (2026-03-22 Session)

| File | Change | Tasks Affected |
|------|--------|---------------|
| solve.py | `_clean_params()` — convert `True/False` → `true/false` in URL params | All action endpoints (11, 13, 25+) |
| prompts.py | YEAREND_PROMPT: Step 0 enumerate tasks + /balanceSheet warning + mandatory salary/tax | 20, 30 |
| prompts.py | TRAVEL_EXPENSE_PROMPT: mandatory deliver + approve | 13 |
| prompts.py | SUPPLIER_INVOICE_PROMPT: account priority + manual VAT split fallback | 11 |
| prompts.py | ORDER_INVOICE_PROMPT: Task 25 vatType exempt in parallel search + mandatory steps | 25 |
| prompts.py | LEDGER_PROMPT: balanced vouchers + bank account credit + dimension voucher example | 27 |
| prompts.py | yearend max_iter 20→30 | 20, 30 |
| prompts.py | /balanceSheet warning on LEDGER_PROMPT + PROJECT_PROMPT | 27, 28, 29, 30 |

# Tripletex Task Reference — ARAS AINM2026

**Last Updated**: 2026-03-22 11:00 CET | **Score**: 65.94 | **Rank**: 75 | **Submissions**: 222
**System**: FastAPI (Cloud Run, 300s timeout) -> Router (Gemini Flash) -> Sub-Agent (Gemini Flash) -> Tripletex v2 API
**LLM**: Gemini 2.5 Flash (temperature=0.0) via langchain-google-genai
**PDF Extraction**: pymupdf4llm (layout-preserving Markdown)
**Scoring**: score ≈ raw_points / try_number (best across all tries kept). Past failures permanently cap max score.

---

## TIER 1 (x1 multiplier)

---

### Task 01 | Customer/Supplier/Product Creation

| Field | Value |
|-------|-------|
| Tier | T1 (x1) |
| Score | 1.50 |
| Tries | 9 |
| Router | customer / supplier / product |
| Max Iterations | 8 |

**Description**: Create a customer or supplier with basic details (name, org number, address, email).

**Example prompts**:
- NO: "Opprett kunden Snohetta AS med organisasjonsnummer 969719878. Adressen er Industriveien 148, 2317 Hamar. E-post: post@snhetta.no."

**Relevant API Endpoints**:
- `POST /customer` — name (REQUIRED), email, organizationNumber, phoneNumber, phoneNumberMobile, physicalAddress, postalAddress, deliveryAddress, website, description, invoiceEmail, invoiceSendMethod, isSupplier, language, accountManager, currency
- `POST /supplier` — name (REQUIRED), email, organizationNumber, phoneNumber, phoneNumberMobile, physicalAddress, postalAddress, deliveryAddress, website, description, invoiceEmail, isCustomer, language
- `PUT /customer` or `PUT /supplier` — include id in body for updates

**Solution Path**:
1. Create entity directly with ALL fields from the prompt (sandbox starts empty, no search needed)
2. ALWAYS set BOTH `physicalAddress` AND `postalAddress` to the same value

**Known Issues**:
- Score not maxed due to efficiency penalty from too many early tries
- Must include EVERY field mentioned in the prompt — missing fields cost points

**Ideas for Improvement**:
- No code changes needed. Score is capped by past try count.

---

### Task 02 | Customer/Supplier/Product Creation

| Field | Value |
|-------|-------|
| Tier | T1 (x1) |
| Score | 1.04 |
| Tries | 6 |
| Router | customer / supplier / product |
| Max Iterations | 8 |

**Description**: Similar entity creation, possibly with more fields than Task 01.

**Example prompts**: Same type as Task 01.

**Relevant API Endpoints**: Same as Task 01.

**Solution Path**: Same as Task 01 — single API call to create with all fields.

**Known Issues**:
- Lower score than Task 01 — likely wasted iterations in early submissions from missing fields or wrong approach

**Ideas for Improvement**:
- LOW priority. T1 tasks have low multiplier. Would need a perfect run on next try to recover any points.

---

### Task 03 | Product Creation

| Field | Value |
|-------|-------|
| Tier | T1 (x1) |
| Score | 2.00 |
| Tries | 9 |
| Router | product |
| Max Iterations | 8 |

**Description**: Create products with specific product number, price, and VAT rate.

**Example prompts**:
- NO: "Opprett produktet 'Frokostblanding' med produktnummer 1391. Prisen er 37450 kr eksklusiv MVA, og MVA-sats for naeringsmidler pa 15 % skal brukes."

**Relevant API Endpoints**:
- `POST /product` — name, number (string), priceExcludingVatCurrency, priceIncludingVatCurrency, vatType: {"id": X}, description, productUnit: {"id": X}, ean, supplier, department, account, isStockItem, weight, weightUnit
- `GET /ledger/vatType` — ?fields=id,name,number,percentage. Filter by `number=` param. Common codes: 3=25%, 31=15% food, 5=exempt/0%
- `GET /product/unit` — ?fields=id,name

**Solution Path**:
1. IN PARALLEL:
   - Search `/ledger/vatType?number=3` (25%) or `?number=31` (15% food) or `?number=5` (0% exempt)
   - Search `/product/unit?fields=id,name` to find existing units like "Stk"
2. From unit results: use existing unit ID. Do NOT create new units.
3. Create product with ALL fields including vatType

**VAT Math**:
- 25% VAT: priceIncludingVatCurrency = priceExcludingVatCurrency x 1.25
- 15% VAT: priceIncludingVatCurrency = priceExcludingVatCurrency x 1.15
- 0% VAT: priceIncludingVatCurrency = priceExcludingVatCurrency

**Known Issues (fixed)**:
- Try 9 regressed to 0 because vatType search used `percentage=` instead of `number=` (invalid filter returned 56 results, wrong VAT selected)
- Agent wasted iterations trying to create product units (all 422 errors)

**Fix Applied**: PRODUCT_PROMPT now uses `?number=31` for 15% VAT, `?number=3` for 25%. "Do NOT create product units."

**Ideas for Improvement**:
- Working correctly now. No further changes needed.

---

### Task 04 | Customer/Supplier/Product Creation

| Field | Value |
|-------|-------|
| Tier | T1 (x1) |
| Score | 2.00 |
| Tries | 5 |
| Router | customer / supplier / product |
| Max Iterations | 8 |

**Description**: Same type as Task 01 — entity creation with full details.

**Relevant API Endpoints**: Same as Task 01.

**Solution Path**: Same as Task 01.

**Known Issues**: None. Working well with good efficiency (5 tries).

**Ideas for Improvement**: None needed.

---

### Task 05 | Department Creation

| Field | Value |
|-------|-------|
| Tier | T1 (x1) |
| Score | 1.33 |
| Tries | 8 |
| Router | department |
| Max Iterations | 8 |

**Description**: Create departments in Tripletex.

**Example prompts**:
- ES: "Crea tres departamentos en Tripletex: 'Drift', 'Administrasjon' y 'Lager'."

**Relevant API Endpoints**:
- `POST /department` — name (REQUIRED), departmentNumber (STRING not int!), departmentManager: {"id": <employee_id>}
- `GET /employee` — search if a manager is referenced

**Solution Path**:
1. Create department(s) directly with name AND departmentNumber
2. If creating multiple, create them ALL in parallel
3. If no departmentNumber specified, assign sequential numbers starting from "1"
4. If manager specified, search /employee first to get their ID

**Known Issues**:
- Score reflects early learning curve. Best result was 7/7 at try 8 = 0.875 which is less than current 1.33.

**Ideas for Improvement**:
- Cannot improve further due to try penalty math. Score is already best achievable.

---

### Task 06 | Customer/Supplier/Product Creation

| Field | Value |
|-------|-------|
| Tier | T1 (x1) |
| Score | 1.50 |
| Tries | 7 |
| Router | customer / supplier / product |
| Max Iterations | 8 |

**Description**: Same type as Task 01 — entity creation.

**Relevant API Endpoints**: Same as Task 01.

**Solution Path**: Same as Task 01.

**Known Issues**: None. Working.

**Ideas for Improvement**: None needed.

---

## TIER 2 (x2 multiplier)

---

### Task 07 | Invoice Payment

| Field | Value |
|-------|-------|
| Tier | T2 (x2) |
| Score | 2.00 |
| Tries | 8 |
| Router | order_invoice |
| Max Iterations | 20 |

**Description**: Find an existing invoice and register payment (or reverse a payment).

**Example prompts**:
- EN: "Reverse the payment on invoice from Costa Brava SL..."

**Relevant API Endpoints**:
- `GET /customer` — search by name or organizationNumber, fields=id,name
- `GET /invoice` — REQUIRES invoiceDateFrom AND invoiceDateTo. fields=id,amount,amountExcludingVat,comment,invoiceDate,invoiceDueDate. Use broad range 2020-01-01 to 2030-12-31
- `GET /invoice/paymentType` — fields=id,description (NOT 'name')
- `PUT /invoice/{id}/:payment` — params: {"paymentDate": "YYYY-MM-DD", "paymentTypeId": int, "paidAmount": total}
- For reversal: use NEGATIVE paidAmount

**Solution Path**:
1. Search customer by name/orgNumber
2. Search invoice with customerId=X and broad date range
3. Search /invoice/paymentType
4. Register payment or reverse (negative paidAmount)

**Known Issues**: Working. Score limited by early try count.

**Ideas for Improvement**: None needed.

---

### Task 08 | Order/Invoice

| Field | Value |
|-------|-------|
| Tier | T2 (x2) |
| Score | 2.00 |
| Tries | 5 |
| Router | order_invoice |
| Max Iterations | 20 |

**Description**: Create orders and convert to invoices.

**Relevant API Endpoints**: Same as Task 10 (see below).

**Solution Path**: Same order -> invoice workflow.

**Known Issues**: None. Working well.

**Ideas for Improvement**: None needed.

---

### Task 09 | Order/Invoice

| Field | Value |
|-------|-------|
| Tier | T2 (x2) |
| Score | 2.67 |
| Tries | 7 |
| Router | order_invoice |
| Max Iterations | 20 |

**Description**: Order/invoice creation and management.

**Relevant API Endpoints**: Same as Task 10.

**Solution Path**: Same order -> invoice workflow. Recently scored 8/8 (perfect).

**Known Issues**: None. Working.

**Ideas for Improvement**: None needed.

---

### Task 10 | Order -> Invoice -> Payment (Full Cycle)

| Field | Value |
|-------|-------|
| Tier | T2 (x2) |
| Score | 2.67 |
| Tries | 11 |
| Router | order_invoice |
| Max Iterations | 20 |

**Description**: Full cycle — create order with products, convert to invoice, register payment.

**Example prompts**:
- NO: "Opprett en ordre for kunden Fjordkraft AS med produktene Opplaering (7579) til 14650 kr og Webdesign (2292) til 11800 kr. Konverter ordren til faktura og registrer full betaling."

**Relevant API Endpoints**:
- `GET /customer` — ?organizationNumber=X or ?name=X, fields=id,name
- `GET /product` — ?number=XXXX&fields=id,name,number. Numbers in parentheses are PRODUCT NUMBERS not IDs!
- `GET /ledger/account?number=1920` — bank account prerequisite
- `GET /invoice/paymentType` — fields=id,description
- `GET /ledger/vatType` — fields=id,name,number,percentage. For mixed VAT rates
- `POST /order` — customer ref, orderDate, deliveryDate (REQUIRED!), orderLines with product ref + unitPriceExcludingVatCurrency (NOT unitCostCurrency!) + vatType per line
- `PUT /order/{id}/:invoice` — params: {"invoiceDate": "YYYY-MM-DD"}
- `PUT /invoice/{id}/:payment` — params: {"paymentDate", "paymentTypeId", "paidAmount" (total, single call)}
- `PUT /invoice/{id}/:send` — params: {"sendType": "EMAIL"|"EHF"|"PAPER"|"MANUAL"}

**Bank Account Prerequisite** (CRITICAL — do before /:invoice):
1. Search `/ledger/account?number=1920&fields=id,bankAccountNumber,isBankAccount`
2. If exists but bankAccountNumber null: update with {"id": X, "isBankAccount": true, "bankAccountNumber": "12345678903"}
3. If not found: create with {"number": 1920, "name": "Bankinnskudd", "isBankAccount": true, "bankAccountNumber": "12345678903"}
4. Use "12345678903" (valid MOD11). NOT "12345678901" — fails validation!

**Solution Path**:
1. IN PARALLEL: search customer + products + bank account 1920 + paymentType + vatType (if mixed rates)
2. If bank account missing/incomplete: create/update it
3. Create order with customer.id, real product IDs, orderDate, deliveryDate. Add vatType per line if different rates.
4. action_endpoint `/order/{order_id}/:invoice` with invoiceDate
5. action_endpoint `/invoice/{invoice_id}/:payment` with paymentDate, paymentTypeId, TOTAL paidAmount

**Known Issues**:
- Sometimes 422 on /:invoice when order lines have wrong VAT or currency fields
- Many tries due to early learning

**Ideas for Improvement**:
- Working correctly now. High try count limits max score.

---

### Task 11 | Supplier Invoice via Voucher

| Field | Value |
|-------|-------|
| Tier | T2 (x2) |
| Score | 0.00 |
| Tries | 10 |
| Router | supplier_invoice |
| Max Iterations | 15 |

**Description**: Record a received supplier invoice as a ledger voucher with correct VAT treatment.

**Example prompts**:
- EN: "We have received invoice INV-2026-8735 from Brightstone Ltd (org no. 913701585) for 8500 NOK including VAT. The amount relates to office services (account 7100). Register with correct input VAT (25%)."

**Relevant API Endpoints**:
- `GET /supplier?organizationNumber=X` — fields=id,name
- `POST /supplier` — create if not found (name, organizationNumber, bankAccountNumber, physicalAddress, postalAddress)
- `GET /ledger/account?number=<expense_acct>` — fields=id,number,name
- `GET /ledger/account?number=2400` — accounts payable
- `GET /ledger/account?number=2710` — input VAT account (for manual split fallback)
- `GET /ledger/vatType?number=1` — incoming 25% VAT (number=1, NOT number=3 which is output)
- `GET /ledger/voucherType` — fields=id,name. Find "Leverandorfaktura" type
- `POST /ledger/voucher` — date, description, vendorInvoiceNumber (REQUIRED for scoring), voucherType (REQUIRED), postings

**Solution Path**:
1. IN PARALLEL: search supplier + expense account + 2400 + 2710 + vatType + voucherType
2. If supplier not found: create supplier
3. From vatType results: find entry with number="1" (incoming 25%). Use the `id` field from search, NOT hardcode!
4. From voucherType results: find entry where name contains "Leverandorfaktura"
5. TRY auto-VAT (preferred): POST /ledger/voucher with 2 postings on same row:
   - Row 1: expense account with amountGross=GROSS_INCL_VAT, vatType={"id": X}, supplier={"id": X}
   - Row 1: 2400 with amountGross=-GROSS_INCL_VAT, supplier={"id": X}
   - BOTH postings must have IDENTICAL description AND supplier
6. FALLBACK if 422 "last til mva-kode 0" (account locked to no-VAT):
   - Manual VAT split on 3 DIFFERENT rows:
   - Row 1: expense account with amountGross=net_amount (no vatType), supplier ref
   - Row 2: 2710 with amountGross=vat_amount
   - Row 3: 2400 with amountGross=-gross_amount, supplier ref
   - NEVER change the task-specified account!

**Known Issues**:
- Root cause: Account 7100 is locked to VAT code 0. Agent previously changed account to 6800 (wrong) instead of using manual VAT split.
- Account 2710 (Inngaende merverdiavgift) may NOT EXIST in sandbox — must create if 0 results
- Max possible score now: ~0.73 due to 10 tries (8/11)

**Fixes Applied**:
- "Account selection priority" rule: NEVER change task-specified account
- Manual VAT split fallback when account is VAT-locked
- Account 2710 searched in parallel + created if not found
- vendorInvoiceNumber and voucherType included in fallback template

**Ideas for Improvement**:
- Score severely capped by try count. Max ~0.73 even with perfect result.

---

### Task 12 | Payroll

| Field | Value |
|-------|-------|
| Tier | T2 (x2) |
| Score | 0.00 |
| Tries | 9 |
| Router | payroll |
| Max Iterations | 15 |

**Description**: Run payroll for an employee — base salary + optional bonus. The task itself often hints: "If salary API doesn't work, use manual vouchers on 5000-series accounts."

**Example prompts**:
- NO: "Kjor lonn for Lars Berg (lars.berg@example.org) for denne maneden. Grunnlonn er 40850 kr. Legg til engangsbonus pa 14800 kr. Dersom lonns-API-et ikke fungerer, kan du bruke manuelle bilag pa lonnskontoer (5000-serien)."

**Relevant API Endpoints**:
- `GET /employee?email=X` — fields=id,firstName,lastName
- `GET /salary/type` — fields=id,number,name. Common: "1000" = Fastlonn/base salary
- `GET /employee/employment?employeeId=X` — fields=id,startDate,endDate
- `POST /employee/employment` — {"employee": {"id": X}, "startDate": "2026-01-01"} ⚠️ MUST use 2026-01-01, NOT today!
- `POST /employee/employment/details` — full details (see below)
- `POST /salary/transaction?generateTaxDeduction=true` — Approach A
- `GET /ledger/account?number=5000` — salary cost (for Approach B fallback)
- `GET /ledger/account?number=2910` — salary payable (for Approach B fallback)
- `POST /ledger/voucher` — Approach B manual voucher

**Solution Path — APPROACH A (try first)**:
1. IN PARALLEL: search employee + salary types + accounts 5000 + account 2910
2. Search /employee/employment?employeeId=X
3. If NO employment: create employment (startDate="2026-01-01") + employment details
4. POST /salary/transaction?generateTaxDeduction=true

**Solution Path — APPROACH B (fallback if A fails)**:
If salary/transaction OR employment creation fails → IMMEDIATELY switch:
```json
POST /ledger/voucher
{"date": "YYYY-MM-DD", "description": "Lonn <name>", "postings": [
  {"row": 1, "account": {"id": <5000_id>}, "amountGross": <BASE+BONUS>, "amountGrossCurrency": <BASE+BONUS>, "description": "Lonn", "employee": {"id": <emp_id>}},
  {"row": 2, "account": {"id": <2910_id>}, "amountGross": <-(BASE+BONUS)>, "amountGrossCurrency": <-(BASE+BONUS)>, "description": "Skyldig lonn"}
]}
```
⚠️ Use "postings" NOT "lines"! Use "amountGross" NOT "debit/credit"! Include "row" field!

**Known Issues**:
- Try 9: Employment creation failed (422 — missing dateOfBirth or wrong startDate)
- Agent fell back to manual vouchers but used WRONG field names ("lines", "debit/credit")
- Took 236 seconds (slow but within Cloud Run 300s timeout)
- Max possible score now: ~0.80 (8/10) due to 9 tries

**Fixes Applied**:
- Approach B with CORRECT field names (postings, amountGross, row)
- Employment startDate forced to "2026-01-01" (not today)
- Employee ref on salary posting
- Immediate switch to Approach B on any failure

**Ideas for Improvement**:
- The task HINTS to use manual vouchers. Approach B may be the expected path.
- Score severely capped by try count. Max ~0.80.

---

### Task 13 | Travel Expense

| Field | Value |
|-------|-------|
| Tier | T2 (x2) |
| Score | 1.13 |
| Tries | 11 |
| Router | travel_expense |
| Max Iterations | 15 |

**Description**: Create travel expense with per diem + costs (flight, taxi, mileage).

**Example prompts**:
- NO: "Registrer en reiseregning for Ragnhild Bakken for 'Kundebesok Kristiansand'. Reisen varte 4 dager med diett (dagsats 800 kr). Utlegg: flybillett 5450 kr og taxi 550 kr."

**Relevant API Endpoints**:
- `GET /employee` — search by email or name
- `GET /travelExpense/paymentType` — fields=id,description
- `GET /travelExpense/costCategory` — fields=id,description (REQUIRED for costs)
- `GET /travelExpense/rateCategory` — ?type=PER_DIEM&dateFrom=X&dateTo=Y&isValidDomestic=true&isRequiresOvernightAccommodation=true&fields=id,name
- `POST /travelExpense` — employee ref, title, travelDetails (REQUIRED if per diem needed)
- `POST /travelExpense/cost` — travelExpense ref, costCategory, paymentType, amountCurrencyIncVat, date, currency, comments, isPaidByEmployee. ONE AT A TIME (parallel causes 409)
- `POST /travelExpense/perDiemCompensation` — travelExpense ref, rateCategory, count, rate, location (REQUIRED), overnightAccommodation
- `POST /travelExpense/mileageAllowance` — travelExpense ref, km, rate, departureLocation, destination, date, isCompanyCar
- `PUT /travelExpense/:deliver` — params: {"id": "<id>"}
- `PUT /travelExpense/:approve` — params: {"id": "<id>"}

**travelDetails** (REQUIRED when per diem is needed):
```json
{
  "departureDate": "YYYY-MM-DD",
  "returnDate": "YYYY-MM-DD",
  "destination": "City name",
  "purpose": "Trip purpose",
  "isDayTrip": false,
  "isForeignTravel": false,
  "isCompensationFromRates": true
}
```

**Solution Path**:
1. IN PARALLEL: search employee + paymentType + costCategory + rateCategory
2. Create travelExpense with travelDetails
3. Create costs ONE AT A TIME sequentially (match costCategory to expense type: "Fly", "Taxi", etc.)
4. Create per diem (rateCategory + count + rate + location)
5. Create mileage allowance if applicable
6. ALWAYS deliver then approve (MANDATORY):
   - action_endpoint `/travelExpense/:deliver` with {"id": "<id>"}
   - action_endpoint `/travelExpense/:approve` with {"id": "<id>"}

**Known Issues**:
- Travel expense was NOT delivered/approved after creation in early submissions
- rateCategory selection might pick wrong per diem rate
- Invalid fields to avoid: description, rateCurrency, amount, rate on cost. dateFrom, dateTo, ratePerDay on perDiem.

**Fixes Applied**:
- TRAVEL_EXPENSE_PROMPT: mandatory deliver + approve step added

**Ideas for Improvement**:
- Verify per diem rateCategory selection is correct for multi-day trips with hotel
- Score limited by 11 tries

---

### Task 14 | Credit Note

| Field | Value |
|-------|-------|
| Tier | T2 (x2) |
| Score | 4.00 |
| Tries | 8 |
| Router | order_invoice |
| Max Iterations | 20 |

**Description**: Create a credit note for an existing invoice.

**Example prompts**:
- DE: "Der Kunde Waldstein GmbH hat die Rechnung fur 'Beratungsstunden' (36900 NOK) reklamiert. Erstellen Sie eine vollstaendige Gutschrift."

**Relevant API Endpoints**:
- `GET /customer?organizationNumber=X` — fields=id,name
- `GET /invoice` — ?customerId=X&invoiceDateFrom=2020-01-01&invoiceDateTo=2030-12-31&fields=id,amount,amountExcludingVat,comment,invoiceDate
- `PUT /invoice/{id}/:createCreditNote` — params: {"date": "YYYY-MM-DD", "comment": "...", "sendToCustomer": true}

**Solution Path**:
1. Search customer by organizationNumber
2. Search invoice with customerId and broad date range
3. Find matching invoice from results
4. action_endpoint `/invoice/{real_invoice_id}/:createCreditNote` with date=today, comment, sendToCustomer=false

**Known Issues**: None. Working perfectly. Maximum score achieved.

**Ideas for Improvement**: None needed.

---

### Task 15 | Invoice Operations

| Field | Value |
|-------|-------|
| Tier | T2 (x2) |
| Score | 3.00 |
| Tries | 6 |
| Router | order_invoice |
| Max Iterations | 20 |

**Description**: Create invoice with multiple product lines, possibly mixed VAT rates.

**Example prompts**:
- NO: "Opprett en faktura til kunden Brattli AS med tre produktlinjer: Skylagring (7246) til 23300 kr med 25% MVA, Systemutvikling (9400) til 3300 kr med 15% MVA (naeringsmiddel), og Nettverkstjeneste (1933) til 10850 kr med 0% MVA."

**Relevant API Endpoints**: Same as Task 10. Key addition: when products have different VAT rates, MUST specify `vatType: {"id": X}` on EACH order line.

**Solution Path**: Same as Task 10 order->invoice flow, with vatType specified per order line.

**Known Issues**:
- Possibly creating invoice with wrong VAT setup or missing VAT on some lines

**Ideas for Improvement**:
- Ensure vatType is correctly set per order line for mixed-rate invoices

---

### Task 16 | Invoice Operations

| Field | Value |
|-------|-------|
| Tier | T2 (x2) |
| Score | 3.00 |
| Tries | 8 |
| Router | order_invoice |
| Max Iterations | 20 |

**Description**: Invoice operations. Recently scored 8/8 (perfect) with score updated to 4.00.

**Relevant API Endpoints**: Same as Task 10.

**Solution Path**: Same order->invoice workflow.

**Known Issues**: None. Working.

**Ideas for Improvement**: None needed.

---

### Task 17 | Invoice Operations

| Field | Value |
|-------|-------|
| Tier | T2 (x2) |
| Score | 3.50 |
| Tries | 6 |
| Router | order_invoice |
| Max Iterations | 20 |

**Description**: Invoice operations. Recently scored 13/13 (perfect).

**Relevant API Endpoints**: Same as Task 10.

**Solution Path**: Same order->invoice workflow.

**Known Issues**: None. Working.

**Ideas for Improvement**: None needed.

---

### Task 18 | Invoice Operations

| Field | Value |
|-------|-------|
| Tier | T2 (x2) |
| Score | 4.00 |
| Tries | 6 |
| Router | order_invoice |
| Max Iterations | 20 |

**Description**: Invoice operations. Working perfectly.

**Relevant API Endpoints**: Same as Task 10.

**Solution Path**: Same order->invoice workflow.

**Known Issues**: None. Maximum score achieved.

**Ideas for Improvement**: None needed.

---

## TIER 3 (x4 multiplier)

---

### Task 19 | Employee Onboarding from PDF Contract

| Field | Value |
|-------|-------|
| Tier | T3 (x4) |
| Score | 2.45 |
| Tries | 5 |
| Router | employee |
| Max Iterations | 15 |

**Description**: Extract employee details from attached PDF employment contract and create employee in Tripletex.

**Example prompts**:
- NO: "Du har mottatt en arbeidskontrakt (se vedlagt PDF). Opprett den ansatte i Tripletex med alle detaljer fra kontrakten: personnummer, fodselsdato, avdeling, stillingskode, lonn, stillingsprosent og startdato."

**Relevant API Endpoints**:
- `POST /employee` — firstName, lastName, userType="EXTENDED", email (REQUIRED for EXTENDED), department: {"id": X}, dateOfBirth, nationalIdentityNumber, address, phoneNumberMobile, phoneNumberWork, employeeNumber, bankAccountNumber, comments
- `POST /employee/employment` — {"employee": {"id": X}, "startDate": "YYYY-MM-DD"}
- `POST /employee/employment/details` — {"employment": {"id": X}, "date": "YYYY-MM-DD", "employmentType": "ORDINARY", "employmentForm": "PERMANENT", "remunerationType": "MONTHLY_WAGE", "workingHoursScheme": "NOT_SHIFT", "percentageOfFullTimeEquivalent": 100, "annualSalary": 650000, "occupationCode": {"id": X}}
- `GET /employee/employment/occupationCode` — ?code=XXXX&fields=id,nameNO,code or ?nameNO=<title>&fields=id,nameNO,code&count=20
- `GET /department` — ?name=X&fields=id,name
- `POST /department` — create if not found
- `POST /employee/standardTime` — {"employee": {"id": X}, "fromDate": "YYYY-MM-DD", "hoursPerDay": 7.5}

**Email Generation** (if not in PDF):
- Generate: firstname.lastname@example.org (lowercase, accents replaced: o->o, a->a, ae->ae, u->u, etc.)
- NEVER switch to NO_ACCESS because email is missing

**Solution Path**:
1. pymupdf4llm extracts PDF -> structured Markdown
2. Parse ALL fields from PDF: name, DOB, address, phone, national ID, department, occupation code, salary, percentage, start date, bank account, email, employee number
3. Search /department by name (create if not found)
4. Search /employee/employment/occupationCode (try by code, then by name, then by root word)
5. POST /employee with ALL fields
6. POST /employee/employment with startDate
7. POST /employee/employment/details with salary, percentage, occupationCode
8. POST /employee/standardTime if working hours mentioned

**Known Issues**:
- Checks 10, 13/15 fail — likely missing address or phone from PDF
- Agent may skip fields it doesn't find easily in PDF text

**Fixes Applied**:
- EMPLOYEE_PROMPT has explicit checklist: verify ALL fields before POST

**Ideas for Improvement**:
- Ensure agent reads and uses address/phone from extracted PDF text
- The checklist is in the prompt but agent may still miss fields

---

### Task 20 | Monthly Closing

| Field | Value |
|-------|-------|
| Tier | T3 (x4) |
| Score | 0.60 |
| Tries | 4 |
| Router | yearend |
| Max Iterations | 30 |

**Description**: Monthly period closing — depreciation, prepaid costs, salary accrual, balance verification.

**Example prompts**:
- NO: "Utfor manedsavslutning for mars 2026. Periodiser forskuddsbetalt kostnad (6500 kr per maned fra konto 1700 til kostkonto). Bokfor manedlig avskrivning for et driftsmiddel med anskaffelseskost 104900 kr og levetid 5 ar. Kontroller at saldobalansen gar i null. Bokfor ogsa en lonnsavsetning (debet lonnskostnad konto 5000, kredit paloppt lonn konto 2900)."

**Relevant API Endpoints**:
- `GET /ledger/account?number=X` — for each account (6010, 1209, 1700, 5000, 2900, etc.)
- `POST /ledger/account` — create if not found: {"number": X, "name": "Norwegian name"}
- `POST /ledger/voucher` — separate voucher per entry. date, description, postings (amountGross + amountGrossCurrency, sum to 0)
- `GET /balanceSheet` — ?dateFrom=X&dateTo=Y&accountNumberFrom=A&accountNumberTo=B&fields=account(id,number,name),balanceIn,balanceOut,balanceChange&count=100
  - WARNING: This is `/balanceSheet` NOT `/ledger/balanceSheet`! The /ledger/ prefix causes 403!

**Solution Path**:
1. STEP 0: ENUMERATE ALL TASKS from the prompt. Count them. Do not stop until all are done.
2. Search ALL accounts mentioned in task IN PARALLEL
3. Identify closing date (last day of month, e.g. 2026-03-31)
4. Create SEPARATE vouchers for each:
   - **Depreciation**: amount = cost / years / 12. Debit 6010, Credit 1209
   - **Prepaid cost**: Debit expense account, Credit 1700
   - **Salary accrual**: Debit 5000, Credit 2900. If no amount given, search /balanceSheet for account 5000
   - **Balance verification**: Search /balanceSheet for all accounts 3000-9999, sum balanceChange = should be 0

**Known Issues**:
- Agent stopped after 2 vouchers (depreciation + prepaid), skipped salary accrual
- Before router refactor, was misrouted as `ledger` with generic prompt
- Agent tried /ledger/balanceSheet -> 403 (wrong endpoint)

**Fixes Applied**:
- YEAREND_PROMPT: Step 0 "ENUMERATE ALL TASKS" forces agent to list every instruction
- Explicit "DO NOT SKIP THIS!" warnings on salary accrual
- /balanceSheet endpoint with "NOT /ledger/balanceSheet" warning
- max_iter increased 20 -> 30

**Ideas for Improvement**:
- Verify Step 0 enumeration prevents skipping salary accrual on next submission
- With 4 tries, next perfect submission gives score = raw_points / 5

---

### Task 21 | Employee from PDF Offer Letter

| Field | Value |
|-------|-------|
| Tier | T3 (x4) |
| Score | 2.57 |
| Tries | 4 |
| Router | employee |
| Max Iterations | 15 |

**Description**: Full employee onboarding from PDF offer letter — more complex than Task 19, includes standard working hours.

**Example prompts**:
- NO: "Du har mottatt et tilbudsbrev (se vedlagt PDF) for en ny ansatt. Utfor komplett onboarding: opprett den ansatte, tilknytt riktig avdeling, sett opp ansettelsesforhold med stillingsprosent og arslonn, og konfigurer standard arbeidstid."

**Relevant API Endpoints**: Same as Task 19, plus:
- `POST /employee/standardTime` — {"employee": {"id": X}, "fromDate": "YYYY-MM-DD", "hoursPerDay": 7.5}

**Solution Path**: Same as Task 19, with additional standard working hours step.

**Known Issues**:
- Check 5 fails — likely occupation code mismatch or missing PDF field

**Ideas for Improvement**:
- Ensure PDF extraction captures all fields including occupation code
- Similar improvements as Task 19

---

### Task 22 | Receipt/Kvittering Booking (PDF)

| Field | Value |
|-------|-------|
| Tier | T3 (x4) |
| Score | 0.00 |
| Tries | 7 |
| Router | receipt |
| Max Iterations | 15 |

**Description**: Extract expense from a receipt image/PDF and book it as a voucher with correct VAT.

**Example prompts**:
- NO: "Vi trenger Togbillett fra denne kvitteringen bokfort pa avdeling Logistikk. Bruk riktig utgiftskonto og sorg for korrekt MVA-behandling."
- DE: "Wir benotigen die Oppbevaringsboks-Ausgabe aus dieser Quittung in der Abteilung HR"
- EN: "We need the Oppbevaringsboks expense from this receipt posted to department Lager"

**Relevant API Endpoints**:
- `GET /ledger/account?number=<expense_acct>` — expense account by mapping
- `GET /ledger/account?number=1920` — bank account
- `GET /ledger/account?number=2710` — input VAT (for manual VAT split fallback). CREATE if 0 results!
- `GET /ledger/vatType?number=1` — incoming 25% VAT. Use the `id` field from result, NOT number!
- `GET /department?name=X` — fields=id,name
- `POST /ledger/account` — create missing accounts (7140, 2710)
- `POST /ledger/voucher` — balanced postings

**Expense Account Mapping**:
- Storage/shelving/containers/boxes (oppbevaringsboks, hylle, skap) -> 6540 (Inventar)
- Electronics/furniture/equipment (keyboard, monitor, chair, printer, lampe, tastatur) -> 6540 (Inventar)
- Paper/pens/office supplies -> 6500 (Kontorrekvisita)
- Cleaning supplies -> 7160
- **Travel/transport/train/flight/taxi (togbillett, flybillett, reise, taxi) -> 7140 (Reisekostnad)**
  ⚠️ Do NOT use 7100 (Bilgodtgjørelse) — it is LOCKED to no-VAT!
- Food/coffee/catering/representasjon (middag, kaffemate, lunsj) -> 7350 (Representasjon)

**Solution Path**:
1. pymupdf4llm extracts receipt -> find EXACT item matching task prompt name
2. Extract: item GROSS price, receipt DATE, VAT rate
3. IN PARALLEL: search expense account + bank 1920 + account 2710 + vatType + department
   ⚠️ If expense account (e.g. 7140) returns 0: CREATE IT! `POST /ledger/account {"number": 7140, "name": "Reisekostnad, oppgavepliktig"}`
   ⚠️ If account 2710 returns 0: CREATE IT! `POST /ledger/account {"number": 2710, "name": "Inngaende merverdiavgift"}`
4. **TRY with vatType first** (for 6540, 6500, 7140, 7160):
   - Row 1: expense with GROSS, vatType, department
   - Row 2: bank 1920 with -GROSS (NO dimensions)
5. **If 422 "last til mva-kode 0"** — MANUAL VAT SPLIT (do NOT just drop vatType!):
   - net = gross / 1.25, vat = gross - net
   - Row 1: expense with net (no vatType), department
   - Row 2: account 2710 with vat
   - Row 3: bank 1920 with -gross
6. **For 7350/7340** (always VAT-locked): full gross, no vatType, no split

**Known Issues (latest try 7)**:
- Agent used account 7100 for "Togbillett" (train ticket) -> 422 "locked to mva-kode 0"
- Then agent DROPPED vatType and posted full gross without any VAT treatment -> all 5 checks fail
- Account 7140 has NEVER existed in any sandbox — must be created

**Fixes Applied**:
- Travel items -> 7140 (NOT 7100) with explicit warning
- Manual VAT split fallback with 2710 (net + VAT + gross structure)
- Account 7140 and 2710 created if not found in sandbox
- "NEVER just remove vatType and post gross — that loses the VAT deduction!"

**Ideas for Improvement**:
- Max possible: 10/8 = 1.25 if perfect next try. Significant gain from 0.
- KEY RISK: Account 7140 might also be VAT-locked — fallback handles this
- PDF extraction confirmed working perfectly (pymupdf4llm)

---

### Task 23 | Bank Reconciliation (CSV)

| Field | Value |
|-------|-------|
| Tier | T3 (x4) |
| Score | 0.60 |
| Tries | 8 |
| Router | bank_recon |
| Max Iterations | 30 |

**Description**: Match CSV bank statement entries to open customer/supplier invoices in Tripletex.

**Example CSV format** (semicolon separator):
```
Dato;Forklaring;Inn;Ut;Saldo
2026-01-18;Innbetaling fra Odegard AS / Faktura 1001;21750.00;;121750.00
2026-01-28;Betaling Leverandor Strand AS;;-7400.00;155200.00
2026-02-03;Renteinntekter;;-1985.42;137814.58
2026-02-05;Bankgebyr;586.03;;138400.61
```

**Relevant API Endpoints**:
- `GET /invoice/paymentType` — fields=id,description
- `GET /invoice` — fields=id,amount,invoiceDate,invoiceNumber,customer(id,name)&count=200
- `GET /supplierInvoice` — fields=id,amount,invoiceNumber,supplier(id,name)&count=200
- `GET /ledger/account` — 1920 (bank), 2400 (AP), 8040 (interest), 7770 (fees), 2920 (tax)
- `GET /supplier` — fields=id,name&count=200
- `PUT /invoice/{id}/:payment` — customer invoice payment
- `PUT /supplierInvoice/{id}/:addPayment` — supplier invoice payment
- `POST /ledger/voucher` — for supplier payments (no invoices) and misc entries

**Solution Path**:
1. BULK FETCH IN PARALLEL: paymentType + invoices + supplierInvoices + accounts (1920, 2400, 8040, 7770, 2920) + suppliers
2. Parse CSV rows and process ONE AT A TIME sequentially:
   - **Customer payment** ("Innbetaling fra X / Faktura N"): Match by invoiceNumber from CSV description. Use CSV amount (handles partial payments). `PUT /invoice/{id}/:payment`
   - **Supplier payment** ("Betaling Leverandor/Fornecedor X"): If supplier invoices found: `PUT /supplierInvoice/:addPayment`. If NOT: voucher DEBIT 2400 (AP) with supplier ref, CREDIT 1920
   - **Bank fee** (Bankgebyr): If positive → DEBIT 1920, CREDIT 7770. If negative → DEBIT 7770, CREDIT 1920
   - **Interest** (Renteinntekter): If positive → DEBIT 1920, CREDIT 8040. If negative → DEBIT 8040, CREDIT 1920
   - **Tax** (Skattetrekk): DEBIT 2920, CREDIT 1920

**Known Issues**:
- Router works correctly now (verified in multiple submissions)
- Try 7 scored 7/7 perfect (4/4 checks) — router and basic flow work
- Tries 8-9 scored 2/10 — check 1 fails consistently when there are MORE invoices than CSV payments
- Possible cause: agent matches by position/amount instead of by invoiceNumber
- API invoiceNumber is "1" not "1001" — literal matching won't work
- Supplier payments initially used 7700 (expense) instead of 2400 (AP) — fixed

**Fixes Applied**:
- Explicit per-row CSV processing with categories A-E
- Match by INVOICE NUMBER from CSV description, not amount or position
- Sequential processing (ONE AT A TIME)
- Supplier payments use account 2400 (AP) with supplier reference
- Negative Renteinntekter handling (reverse direction)

**Ideas for Improvement**:
- Score capped: at 8+ tries, max gain ~0.28. LOW PRIORITY.
- The invoiceNumber matching is ambiguous (CSV says 1001, API has 1). May need to match by customer name + closest amount.

---

### Task 24 | Corrections/Reversals

| Field | Value |
|-------|-------|
| Tier | T3 (x4) |
| Score | 2.25 |
| Tries | 5 |
| Router | corrections |
| Max Iterations | 30 |

**Description**: Find and fix 4 errors in existing vouchers: wrong account, duplicate, missing VAT, wrong amount.

**Example prompts**:
- DE: "Wir haben Fehler im Hauptbuch fur Januar und Februar 2026 entdeckt. Uberprufen Sie alle Belege und finden Sie die 4 Fehler: eine Buchung auf das falsche Konto (Konto 6540 statt 6860), ein doppelter Beleg (Konto 6500, 1500 NOK), eine fehlende MwSt.-Zeile (Konto 6300, 22350 NOK), und ein falscher Betrag (Konto 6540, 11400 gebucht statt 8550)."

**Relevant API Endpoints**:
- `GET /ledger/account?number=X` — for each account mentioned
- `GET /ledger/voucher` — ?dateFrom=X&dateTo=Y&fields=id,date,description,postings(account(number,id),amountGross)&count=50
- `PUT /ledger/voucher/{id}/:reverse` — params: {"date": "YYYY-MM-DD"}
- `POST /ledger/voucher` — create corrected voucher
- `GET /ledger/vatType?number=1` — for missing VAT fix

**Solution Path** (GOLDEN RULE: copy exact, change only the error):

1. Search accounts for each account number mentioned (IN PARALLEL)
2. For EACH error, search vouchers to find the matching one:

   **WRONG ACCOUNT** (e.g. 6540 used instead of 6860):
   a. Find voucher with posting on wrong account (6540) with specified amount
   b. Reverse it: /:reverse with original voucher date
   c. Create new voucher: COPY exact same postings, ONLY change wrong account to correct one

   **DUPLICATE** (e.g. duplicate on 6500, 1500 NOK):
   a. Find TWO vouchers with same account + amount
   b. Reverse ONE of them (no new voucher needed)

   **MISSING VAT** (e.g. 6300, 22350 NOK net, missing VAT):
   a. Find voucher with expense account and net amount
   b. Reverse it
   c. Search /ledger/vatType?number=1 for incoming VAT ID
   d. Create corrected: on expense posting set amountGross=NET*1.25 and add vatType, on credit posting set amountGross=-(NET*1.25). System auto-generates VAT row.

   **WRONG AMOUNT** (e.g. 11400 recorded instead of 8550):
   a. Find voucher with wrong amount
   b. Reverse it
   c. Create corrected: COPY exact postings, change only the wrong amount

**Known Issues**:
- Multiple 422 errors during correction: unbalanced voucher, missing supplier, row 0 errors
- Agent created correction vouchers with wrong structures
- Agent omitted supplier reference from original voucher
- Agent used row=0 (system-reserved)
- For missing VAT: agent created manual 3-row VAT split instead of using vatType auto-split

**Fixes Applied**:
- GOLDEN RULE: copy exact postings, change only the error
- Row must start from 1, never 0
- If original has supplier, corrected must too
- For missing VAT: use vatType + amountGross=NET*1.25 (auto-split)

**Ideas for Improvement**:
- Verify GOLDEN RULE fix works in practice
- Agent sometimes still creates wrong structures — may need more explicit examples in prompt

---

### Task 25 | Overdue Invoice + Reminder + Partial Payment

| Field | Value |
|-------|-------|
| Tier | T3 (x4) |
| Score | 3.40 ✅ |
| Tries | 4 |
| Router | order_invoice |
| Max Iterations | 20 |

**Description**: Multi-step invoice lifecycle: find overdue invoice, book reminder fee, create reminder invoice (VAT exempt), send it, register partial payment.

**Example prompts**:
- EN: "Find the overdue invoice and post a reminder fee of 70 NOK. Debit accounts receivable (1500), credit reminder fees (3400). Also create an invoice for the reminder fee to the customer and send it. Additionally, register a partial payment of 5000 NOK on the overdue invoice."

**Relevant API Endpoints**:
- `GET /invoice` — ?invoiceDateFrom=2020-01-01&invoiceDateTo=2030-12-31&fields=id,amount,invoiceDate,invoiceDueDate,customer(id,name)&count=100
- `GET /ledger/account?number=1500` — accounts receivable
- `GET /ledger/account?number=3400` — reminder fee income
- `GET /ledger/account?number=1920` — bank account
- `GET /invoice/paymentType` — fields=id,description
- `GET /ledger/vatType?number=5` — VAT exempt 0% (REQUIRED for reminder invoice!)
- `POST /ledger/voucher` — reminder fee voucher
- `POST /order` — reminder fee invoice with vatType exempt on order line
- `PUT /order/{id}/:invoice` — convert to invoice
- `PUT /invoice/{id}/:send` — send reminder invoice
- `PUT /invoice/{id}/:payment` — partial payment on overdue invoice
- Alternative: `PUT /invoice/{id}/:createReminder` — params: {"type": "REMINDER", "date": "YYYY-MM-DD", "includeCharge": true, "includeInterest": false, "dispatchType": "EMAIL"} (note: "dispatchType" NOT "sendType")

**Solution Path**:
1. IN PARALLEL: search all invoices + account 1500 + account 3400 + account 1920 + paymentType + vatType exempt (number=5)
2. Find overdue invoice: where invoiceDueDate < today. Note customer ID and invoice ID.
3. MANDATORY: Book reminder fee voucher:
   - Debit 1500 (+fee, with customer ref), Credit 3400 (-fee)
4. MANDATORY: Create reminder fee INVOICE:
   a. Ensure bank account exists (prerequisite)
   b. Create order with customer, order line description="Purregebyr", unitPriceExcludingVatCurrency=fee, vatType={"id": exempt_id}
   c. Convert order to invoice
   d. Send invoice via /:send with sendType="EMAIL"
5. MANDATORY: Register partial payment on OVERDUE invoice (NOT reminder invoice):
   - action_endpoint /:payment with paidAmount = partial amount from task (NOT full invoice amount)

**Known Issues**:
- Previous: reminder order created WITHOUT vatType -> system applied 25% -> wrong amount (87.50 instead of 70)
- /:createReminder failed with boolean params True/False (capital) not recognized

**Fixes Applied**:
- solve.py: _clean_params() converts Python True/False -> lowercase true/false
- ORDER_INVOICE_PROMPT: vatType exempt search in parallel step 1
- Order line requires vatType: {"id": exempt_id} with warning
- All 3 steps (voucher, invoice, payment) marked MANDATORY
- Score improved: 6/6 at try 3 = 3.40

**Ideas for Improvement**:
- Score is 3.40. FIXED and scoring perfect (6/6) on every submission. No further changes needed.
- /:createReminder now available as alternative (dispatchType param fixed, boolean casing fixed)

---

### Task 26 | Supplier Invoice from PDF

| Field | Value |
|-------|-------|
| Tier | T3 (x4) |
| Score | 3.75 |
| Tries | 5 |
| Router | supplier_invoice |
| Max Iterations | 15 |

**Description**: Extract supplier invoice details from PDF and register as voucher.

**Example prompts**:
- NO: "Du har mottatt en leverandorfaktura (se vedlagt PDF). Registrer fakturaen i Tripletex. Opprett leverandoren hvis den ikke finnes."

**Relevant API Endpoints**: Same as Task 11, with PDF extraction.

**Solution Path**: Same as Task 11, but supplier details come from PDF:
1. pymupdf4llm extracts PDF -> find supplier name, orgNumber, address, bank account, invoice number, amounts
2. Search supplier by orgNumber. If not found: create supplier with ALL details from PDF (name, orgNumber, address, bankAccountNumber)
3. Create voucher with vendorInvoiceNumber and voucherType (same as Task 11)

**Known Issues**:
- Previous: agent created supplier with only name+orgNumber but PDF had address and bank account

**Fixes Applied**:
- "Include ALL details from the PDF" instruction added

**Ideas for Improvement**:
- Working reasonably. Could ensure all PDF fields are captured for supplier creation.

---

### Task 27 | Custom Accounting Dimensions

| Field | Value |
|-------|-------|
| Tier | T3 (x4) |
| Score | 6.00 ✅ |
| Tries | 6 |
| Router | ledger |
| Max Iterations | 20 |

**Description**: Create custom accounting dimension with values, then post a voucher linked to a dimension value.

**Example prompts**:
- ES: "Cree una dimension contable personalizada 'Kostsenter' con los valores 'Okonomi' y 'Produktutvikling'. Luego registre un asiento en la cuenta 6340 por 45050 NOK."
- EN: "Create a custom accounting dimension 'Region' with values 'Nord-Norge' and 'Sor-Norge'. Then post a voucher on account 7000 for 13650 NOK."

**Relevant API Endpoints**:
- `POST /ledger/accountingDimensionName` — {"dimensionName": "Region"} -> returns dimensionIndex
- `GET /ledger/accountingDimensionName` — ?fields=id,dimensionName,dimensionIndex
- `POST /ledger/accountingDimensionValue` — {"displayName": "Nord-Norge", "dimensionIndex": 1, "number": "1"} (number is STRING)
- `GET /ledger/accountingDimensionValue` — ?dimensionIndex=1&fields=id,displayName,number
- `GET /ledger/account?number=X` — expense account
- `GET /ledger/account?number=1920` — bank account
- `POST /ledger/voucher` — with freeAccountingDimension<N> on expense posting

**Solution Path**:
1. IN PARALLEL: search expense account + bank account 1920
2. Create dimension: POST /ledger/accountingDimensionName -> get dimensionIndex
3. Create values ONE AT A TIME sequentially (number is STRING: "1", "2")
4. Create BALANCED voucher:
   - Row 1: expense with amountGross=+amount, freeAccountingDimension<N>={"id": value_id}
   - Row 2: bank 1920 with amountGross=-amount (NO dimension)
5. dimensionIndex determines which freeAccountingDimension field to use (1->freeAccountingDimension1, 2->freeAccountingDimension2)

**Known Issues**:
- Voucher might need TWO postings (debit + credit) but agent sometimes only creates one
- freeAccountingDimension link might use wrong dimensionIndex

**Fixes Applied**:
- LEDGER_PROMPT: balanced vouchers + bank account credit side + dimension voucher example
- freeAccountingDimension goes ONLY on expense posting, NOT bank posting

**Ideas for Improvement**:
- Verify voucher is always balanced (debit + credit)
- Check freeAccountingDimension field mapping matches dimensionIndex

---

### Task 28 | Project Ledger Analysis

| Field | Value |
|-------|-------|
| Tier | T3 (x4) |
| Score | 1.50 |
| Tries | 7 |
| Router | project |
| Max Iterations | 20 |

**Description**: Analyze ledger accounts, find top expense increases between two months, create internal projects for each.

**Example prompts**:
- PT: "Os custos totais aumentaram significativamente de janeiro a fevereiro de 2026. Analise o livro razao e identifique as tres contas de despesa com o maior aumento em valor. Crie um projeto interno para cada uma das tres contas com o nome da conta."

**Relevant API Endpoints**:
- `GET /balanceSheet` — ?dateFrom=X&dateTo=Y&accountNumberFrom=4000&accountNumberTo=7999&fields=id,account(id,number,name),balanceIn,balanceOut,balanceChange&count=100. NOT /ledger/balanceSheet (403)!
- `GET /employee` — ?fields=id,firstName,lastName&count=1 (any employee for projectManager)
- `POST /project` — {"name": "<account_name>", "number": "<account_number>", "startDate": "2026-01-01", "projectManager": {"id": X}, "isInternal": true}
- `POST /activity` — {"name": "<account_name>", "activityType": "PROJECT_GENERAL_ACTIVITY"}
- `POST /project/projectActivity` — {"project": {"id": X}, "activity": {"id": Y}}

**Solution Path**:
1. IN PARALLEL: search /employee (any, for projectManager) + query /balanceSheet for BOTH periods
   - January: dateFrom=2026-01-01&dateTo=2026-01-31&accountNumberFrom=4000&accountNumberTo=7999
   - February: dateFrom=2026-02-01&dateTo=2026-02-28&accountNumberFrom=4000&accountNumberTo=7999
   - CRITICAL: Feb has 28 days in 2026 (NOT leap year)
2. Compute increase per account: monthB_balanceChange - monthA_balanceChange
   - If account in B but not A: increase = monthB_balanceChange
   - If account in A but not B: increase is negative (skip)
   - "Largest increase" = DIFFERENCE between periods, NOT largest value in B
3. Sort by increase descending, pick top 3
4. For EACH top account:
   a. Create project with name=account_name, number=account_number, isInternal=true, projectManager, startDate
   b. Create activity: {"name": account_name, "activityType": "PROJECT_GENERAL_ACTIVITY"}
   c. Link activity to project via /project/projectActivity
   d. Create activities SEQUENTIALLY (not parallel)

**Known Issues**:
- Try 7: 0/10 — agent added `id` to balanceSheet fields → 400 error → retried parallel → confused Jan/Feb → empty output
- Try 6: 5/10 — checks 2,3,4 fail (wrong top 3 accounts identified)
- Try 5: 5/10 — checks 1,2,4 fail
- The agent may sort by ABSOLUTE Feb values instead of INCREASE (Feb-Jan)
- Parallel balanceSheet queries can confuse which result is Jan vs Feb

**Fixes Applied**:
- /balanceSheet: removed top-level `id` from fields (causes 400)
- Query months SEQUENTIALLY (not parallel) to avoid Jan/Feb confusion
- "SHOW YOUR WORK" — forces agent to write out calculations
- Include "number" field on project creation
- Result data logging added for /balanceSheet debugging

**Ideas for Improvement**:
- At 7 tries, max possible ~1.25. Limited value.
- Core issue may be LLM arithmetic — consider adding explicit calculation template

---

### Task 29 | Project Full Cycle

| Field | Value |
|-------|-------|
| Tier | T3 (x4) |
| Score | 2.73 |
| Tries | 6 |
| Router | project |
| Max Iterations | 20 |

**Description**: Full project lifecycle — create/find project with budget, register time, add supplier costs, generate invoice.

**Example prompts**:
- NN: "Registrer 8 timar for Knut Aasen pa aktiviteten 'Utvikling' i prosjektet 'Nettbutikk-utvikling' for Elvdal AS. Timesats: 1800 kr/t. Generer ein prosjektfaktura til kunden."

**Relevant API Endpoints**:
- `GET /project?name=X` — search existing project
- `POST /project` — name, startDate (REQUIRED), projectManager, customer, isFixedPrice, fixedprice, isInternal
- `PUT /project` — ⚠️ BETA endpoint, may return 403. TRY it for budget, ignore 403 if it fails.
- `GET /customer` — search by name/orgNumber
- `GET /employee` — search by name/email
- `GET /activity?name=X` — fields=id,name (NOT /project/activity!)
- `POST /activity` — {"name": "X", "activityType": "PROJECT_GENERAL_ACTIVITY", "isChargeable": true, "rate": 1200}
- `POST /project/projectActivity` — {"project": {"id": X}, "activity": {"id": Y}}
- `POST /timesheet/entry` — {"employee": {"id": X}, "project": {"id": X}, "activity": {"id": X}, "date": "YYYY-MM-DD", "hours": 8}
- `POST /project/hourlyRates` — {"project": {"id": X}, "startDate": "YYYY-MM-DD", "hourlyRateModel": "TYPE_PROJECT_SPECIFIC_HOURLY_RATES", "showInProjectOrder": true}
- `POST /project/hourlyRates/projectSpecificRates` — {"projectHourlyRate": {"id": X}, "hourlyRate": 1300, "activity": {"id": X}, "employee": {"id": X}}
- `POST /order` — with project ref, customer ref, order lines
- `PUT /order/{id}/:invoice` — convert to invoice
- `GET /ledger/account?number=1920` — bank account prerequisite
- For supplier costs: `POST /ledger/voucher` with project ref on expense posting

**Solution Path** (full project cycle):
1. IN PARALLEL: search project + customer + employee + activity
2. MANDATORY BUDGET STEP: update project with {"isFixedPrice": true, "fixedprice": amount, "customer": {"id": Y}, "projectManager": {"id": Z}}. Must happen BEFORE other steps!
3. If activity not found: create activity
4. Link activity to project: POST /project/projectActivity (WAIT for completion)
5. Create timesheet entry (WAIT for completion after step 4)
6. SKIP hourly rates setup (often fails, invoice uses order line amount)
7. Create order with project + customer + order line (hours x rate as price)
8. Bank account check: ensure 1920 has bankAccountNumber
9. Convert order to invoice: /order/:invoice
CRITICAL: Steps 4->5->7->9 MUST be sequential!

**Known Issues**:
- Try 6: 10/11 (91%), check 5 fails — budget not set on existing project
- PUT /project is BETA and may return 403
- Agent previously skipped budget step entirely

**Fixes Applied**:
- Budget step: TRY PUT /project, ignore 403 if it fails
- If project doesn't exist: include ALL fields (fixedprice, customer, projectManager) in POST
- Mandatory budget step with warning

**Ideas for Improvement**:
- If PUT /project returns 403, budget can't be set on existing projects — check 5 will keep failing
- At 6 tries, next perfect = 11/7 ≈ 1.57 (less than current 2.73 — CAPPED)

---

### Task 30 | Year-End Closing

| Field | Value |
|-------|-------|
| Tier | T3 (x4) |
| Score | 1.80 |
| Tries | 5 |
| Router | yearend |
| Max Iterations | 30 |

**Description**: Full annual closing — depreciation for 3 assets + prepaid reversal + tax calculation.

**Example prompts**:
- NO: "Utfor forenklet arsoppgjor for 2025: 1) Beregn og bokfor arlige avskrivninger for tre eiendeler... 2) Reverser forskuddsbetalte kostnader... 3) Beregn og bokfor skattekostnad (22% av skattbart resultat) pa konto 8700/2920."

**Relevant API Endpoints**: Same as Task 20, plus:
- Tax calculation requires querying /balanceSheet for:
  - Revenue: accountNumberFrom=3000&accountNumberTo=3999 (negative = income)
  - Expenses: accountNumberFrom=4000&accountNumberTo=8699 (positive = cost)

**Solution Path**:
1. STEP 0: ENUMERATE ALL TASKS. Write numbered checklist. Count them.
2. Search ALL accounts in parallel (6010, 1209, 1700, 8700, 2920, etc.)
3. Use closing date 2025-12-31 (or year specified)
4. Create SEPARATE vouchers:
   - **Depreciation** (for EACH asset): amount = cost / years. Debit 6010, Credit 1209. Create SEPARATE voucher per asset!
   - **Prepaid reversal**: Debit expense account, Credit 1700
   - **Tax** (CRITICAL - DO NOT SKIP):
     1. Query /balanceSheet for revenue (3000-3999)
     2. Query /balanceSheet for expenses (4000-8699)
     3. taxable_income = abs(sum revenue balanceChange) - sum expense balanceChange (INCLUDE depreciation+prepaid vouchers just created!)
     4. tax = round(taxable_income x 0.22)
     5. Debit 8700, Credit 2920

**Known Issues**:
- Try 5: 6/10 — checks 4,5 fail. Tax calculated as 267,612 instead of ~204,060
- Root cause: /balanceSheet API does NOT reflect just-created vouchers! Agent used balanceSheet data WITHOUT adding depreciation/prepaid amounts
- Correct formula: total_expenses = balanceSheet_expenses + depreciation_total + prepaid_total
- Before: agent skipped tax entirely (403 on /ledger/balanceSheet). Now it executes but miscalculates.

**Fixes Applied**:
- Step 0 "ENUMERATE ALL TASKS" to prevent skipping any step
- /balanceSheet (NOT /ledger/balanceSheet) warning
- Tax calc: "balanceSheet may NOT include just-created vouchers! Manually ADD depreciation + prepaid to expenses"
- "SHOW YOUR WORK" for tax calculation
- max_iter 20 -> 30

**Ideas for Improvement**:
- At 5 tries, 10/6 = 1.67 (less than current 1.80 — CAPPED unless raw score > 10.8)
- The tax fix is critical but may not improve score due to try penalty
- Depreciation rounding: ensure round to nearest integer

---

## Cross-Cutting Reference

### Recurring Voucher Validation Errors

| Error | Cause | Fix |
|-------|-------|-----|
| "Leverandor mangler" (supplier missing) | Same-row postings must have identical supplier dimension | Include supplier on ALL postings in same row |
| "Kunde mangler" (customer missing) | Same-row postings must have identical customer dimension | Include customer on ALL postings in same row |
| "rad 0 er systemgenererte" (row 0) | Row 0 is system-reserved | Always use row >= 1 |
| "Beskrivelsene ma vaere like" (description) | Same-row postings must have identical descriptions | Use exact same description text |
| "ulike dimensjoner" (different dims) | Same-row postings can't have different department/supplier | Put dimensions only on one row, use different row numbers |
| "last til mva-kode 0" (VAT locked) | Account locked to no-VAT, can't set vatType | Use manual VAT split on separate rows |
| "Summen er ikke lik 0" (unbalanced) | Postings don't sum to zero | Ensure debit + credit = 0 |
| "Feltet eksisterer ikke" (field not exist) | Wrong field names (voucherDate, voucherLines, entries) | Remove invalid fields and retry |

### Router Categories

| Category | Prompt | Max Iter | Tasks |
|----------|--------|----------|-------|
| employee | EMPLOYEE_PROMPT | 15 | 19, 21 |
| customer | CUSTOMER_PROMPT | 8 | 01, 02, 04, 06 |
| product | PRODUCT_PROMPT | 8 | 03 |
| supplier | SUPPLIER_PROMPT | 8 | 01, 02, 04, 06 |
| department | DEPARTMENT_PROMPT | 8 | 05 |
| contact | CONTACT_PROMPT | 10 | (T1 tasks) |
| order_invoice | ORDER_INVOICE_PROMPT | 20 | 07-10, 14-18, 25 |
| travel_expense | TRAVEL_EXPENSE_PROMPT | 15 | 13 |
| supplier_invoice | SUPPLIER_INVOICE_PROMPT | 15 | 11, 26 |
| payroll | PAYROLL_PROMPT | 15 | 12 |
| receipt | RECEIPT_PROMPT | 15 | 22 |
| corrections | CORRECTIONS_PROMPT | 30 | 24 |
| bank_recon | BANK_RECON_PROMPT | 30 | 23 |
| yearend | YEAREND_PROMPT | 30 | 20, 30 |
| ledger | LEDGER_PROMPT | 20 | 27 |
| project | PROJECT_PROMPT | 20 | 28, 29 |

### API General Rules

- POST: never include `id` — API generates it
- PUT: include `id` in body matching resource_id
- References: `{"id": <known_id>}`, e.g. `"customer": {"id": 42}`
- Use `fields` param to limit response size
- POST response contains created ID — use directly, never re-fetch
- For nested objects in `fields`, use PARENTHESES: `customer(id,name)` NOT `customer.id`
- Search filters use flat names: `customerId` NOT `customer.id`
- To "delete"/"archive" an entity with ledger entries: use PUT with `isInactive: true`
- If /ledger/account search returns 0: CREATE it with standard Norwegian name
- If /department search returns 0: CREATE it
- 409 = resource already exists, search for it instead
- If a field "does not exist in the object": remove that field and retry

### Fixes Applied (2026-03-22 Session)

| File | Change | Tasks | Result |
|------|--------|-------|--------|
| solve.py | `_clean_params()` — True/False -> true/false in URL params | All | Fixed /:createReminder |
| solve.py | Result data logging for /balanceSheet, /invoice, /supplierInvoice | 28, 23 | Debugging |
| prompts.py | YEAREND: Step 0 enumerate + /balanceSheet + mandatory salary/tax | 20, 30 | Untested |
| prompts.py | YEAREND: Tax calc manually adds depreciation+prepaid | 30 | Untested |
| prompts.py | TRAVEL_EXPENSE: mandatory deliver + approve | 13 | Untested |
| prompts.py | SUPPLIER_INVOICE: account priority + manual VAT split | 11 | Untested |
| prompts.py | SUPPLIER_INVOICE: create account 2710 if not found | 11 | Untested |
| prompts.py | ORDER_INVOICE: vatType exempt + dispatchType fix | 25 | ✅ 6/6 perfect |
| prompts.py | LEDGER: balanced vouchers + bank credit + dimension example | 27 | ✅ 10/10 perfect |
| prompts.py | RECEIPT: account 7140 (not 7100) + manual VAT split + create 7140/2710 | 22 | Untested |
| prompts.py | PAYROLL: Approach B manual voucher fallback (correct fields) | 12 | Untested |
| prompts.py | CORRECTIONS: voucher search includes supplier/customer/vatType | 24 | Untested |
| prompts.py | BANK_RECON: rewritten with per-row processing + supplier via 2400 | 23 | Partial (check 1 fails) |
| prompts.py | PROJECT: sequential balanceSheet + no id in fields + budget try PUT | 28, 29 | Try 7: 0/10 regression |

### Execution Time Benchmarks

| Task Type | Avg Time | Min | Max |
|-----------|----------|-----|-----|
| customer | 6.2s | 6.2s | 6.2s |
| department | 6.7s | 6.6s | 6.8s |
| supplier | 10.3s | 10.3s | 10.3s |
| payroll | 13.8s | 12.0s | 15.0s |
| supplier_invoice | 20.6s | 14.7s | 33.2s |
| receipt | 21.1s | 21.1s | 21.1s |
| employee | 23.5s | 11.4s | 43.2s |
| product | 24.4s | 24.4s | 24.4s |
| travel_expense | 24.6s | 17.2s | 33.6s |
| yearend | 30.7s | 30.4s | 31.0s |
| project | 32.0s | 9.6s | 115.9s |
| order_invoice | 33.5s | 8.8s | 253.5s |
| ledger | 33.5s | 12.4s | 113.7s |
| corrections | 60.9s | 20.6s | 104.7s |

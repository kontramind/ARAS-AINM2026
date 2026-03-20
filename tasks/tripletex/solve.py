"""
tasks/tripletex/solve.py
------------------------
Tripletex AI Accounting Agent.

Entry point: solve(request: SolveRequest) -> SolveResponse

Flow:
  1. Parse the incoming request (prompt, files, Tripletex credentials)
  2. Extract file content (PDF text, image vision)
  3. Run a planning LLM call — produces an explicit execution plan
  4. Run the AgentRunner with the plan injected — executes API calls
  5. Return {"status": "completed"}

Authentication: Basic Auth, username="0", password=session_token
All API calls go through tripletex_credentials.base_url (competition proxy).

Env vars required (via .env or environment):
  LLM_PROVIDER          = azure_native | openai_compatible | ollama
  (+ provider-specific vars — see tasks/language/factory.py)
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any, Optional

import requests
import urllib3
from pydantic import BaseModel

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from tasks.language.agent import AgentRunner, make_tool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# STRUCTURED JSON LOGGER — Cloud Logging auto-parses JSON lines from stdout
# ---------------------------------------------------------------------------

def _log(msg: str, severity: str = "INFO", **extra) -> None:
    """Emit a structured JSON log line to stdout (picked up by Cloud Logging)."""
    import sys
    entry = {
        "severity": severity,
        "message": msg,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "component": "tripletex-agent",
    }
    entry.update(extra)
    print(json.dumps(entry, ensure_ascii=False), flush=True, file=sys.stdout)


# ===========================================================================
# REQUEST / RESPONSE SCHEMAS
# ===========================================================================

class FileAttachment(BaseModel):
    filename: str
    content_base64: str
    mime_type: str


class TripletexCredentials(BaseModel):
    base_url: str
    session_token: str


class SolveRequest(BaseModel):
    prompt: str
    files: list[FileAttachment] = []
    tripletex_credentials: TripletexCredentials


class SolveResponse(BaseModel):
    status: str = "completed"


# ===========================================================================
# TRIPLETEX HTTP CLIENT
# ===========================================================================

class TripletexClient:
    """
    Thin HTTP wrapper around the Tripletex v2 REST API.
    Routes all traffic through the competition proxy using Basic Auth.

    Auth: username="0", password=session_token  (per spec)
    Responses are automatically unwrapped:
      {"values": [...]}  → list
      {"value": {...}}   → dict
    """

    def __init__(self, base_url: str, session_token: str) -> None:
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()
        self._session.auth = ("0", session_token)
        self._session.headers.update({"Content-Type": "application/json"})
        self._session.verify = False  # corporate proxy uses self-signed certs

    def _url(self, endpoint: str) -> str:
        return f"{self.base_url}/{endpoint.lstrip('/')}"

    def get(self, endpoint: str, params: dict | None = None) -> Any:
        resp = self._session.get(self._url(endpoint), params=params or {})
        resp.raise_for_status()
        data = resp.json()
        # Unwrap list response: {"fullResultSize": N, "values": [...]}
        if isinstance(data, dict) and "values" in data:
            return data["values"]
        # Unwrap single-resource response: {"value": {...}}
        if isinstance(data, dict) and "value" in data:
            return data["value"]
        return data

    def post(self, endpoint: str, body: dict) -> Any:
        resp = self._session.post(self._url(endpoint), json=body)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and "value" in data:
            return data["value"]
        return data

    def put(self, endpoint: str, resource_id: int | str, body: dict) -> Any:
        resp = self._session.put(self._url(f"{endpoint}/{resource_id}"), json=body)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and "value" in data:
            return data["value"]
        return data

    def put_action(self, endpoint: str, params: dict | None = None) -> Any:
        """PUT with query params (for Tripletex action endpoints like /:payment, /:invoice)."""
        resp = self._session.put(self._url(endpoint), params=params or {}, json={})
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and "value" in data:
            return data["value"]
        return data

    def delete(self, endpoint: str, resource_id: int | str) -> bool:
        resp = self._session.delete(self._url(f"{endpoint}/{resource_id}"))
        resp.raise_for_status()
        return True


# ===========================================================================
# LANGCHAIN TOOL FACTORY
# ===========================================================================

def _parse_json_lenient(s: str) -> Any:
    """Parse JSON leniently — strip control characters that LLMs sometimes emit."""
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        # Strip control characters (except space) and retry
        cleaned = "".join(c if c >= " " or c in "\n\r\t" else "" for c in s)
        return json.loads(cleaned, strict=False)


def build_tools(client: TripletexClient) -> list:
    """
    Returns LangChain tools that capture the TripletexClient in their closure.
    Tools are designed to minimize API round-trips and make agent intent explicit.
    """

    @make_tool
    def search_resource(endpoint: str, filters: Optional[str] = None) -> str:
        """
        Search for resources in Tripletex using query filters.
        Use this to find existing entities BEFORE creating them (avoids 409 duplicates).

        Args:
            endpoint: API path, e.g. '/employee', '/customer', '/invoice'.
            filters:  Optional JSON string of search parameters.
                      Always include 'fields' to limit response size.
                      Examples:
                        '{"fields": "id,firstName,lastName,email"}'
                        '{"name": "Acme AS", "fields": "id,name,organizationNumber"}'
                        '{"email": "ola@example.org", "fields": "id,firstName,lastName"}'
                        '{"dateFrom": "2024-01-01", "dateTo": "2024-12-31", "fields": "id,date,amount"}'
                      Use 'count=100' to fetch more results (default is 10).

        Returns:
            JSON array of matching resources. Empty array means nothing was found.
        """
        params = _parse_json_lenient(filters) if filters else {}
        # Always request id at minimum for efficient follow-up calls
        if "fields" not in params:
            params["fields"] = "id,name"
        _log(f"  🔍 GET {endpoint} params={json.dumps(params, ensure_ascii=False)[:300]}")
        try:
            result = client.get(endpoint, params=params)
            # Ensure we always return a list
            if not isinstance(result, list):
                result = [result]
            _log(f"  📥 Search {endpoint}: {len(result)} result(s)")
            return json.dumps(result, ensure_ascii=False)
        except requests.HTTPError as e:
            error_text = getattr(e.response, 'text', str(e)) if e.response is not None else str(e)
            _log(f"  ❗ GET {endpoint} failed: {e} -> {error_text[:300]}")
            return json.dumps({"error": str(e), "response": error_text})
        except Exception as e:
            _log(f"  ❗ GET {endpoint} error: {e}")
            return json.dumps({"error": str(e)})

    @make_tool
    def get_by_id(endpoint: str, resource_id: int, fields: Optional[str] = None) -> str:
        """
        Fetch a single resource by its ID.
        Use this when you already know the ID (e.g. after a create or search).

        Args:
            endpoint:    API path, e.g. '/employee', '/customer'.
            resource_id: Integer ID of the resource.
            fields:      Optional comma-separated field list, e.g. 'id,name,email'.
                         Use '*' to fetch all fields.

        Returns:
            JSON object of the resource.
        """
        params = {"fields": fields} if fields else {}
        try:
            result = client.get(f"{endpoint}/{resource_id}", params=params)
            return json.dumps(result, ensure_ascii=False)
        except requests.HTTPError as e:
            return json.dumps({"error": str(e), "response": getattr(e.response, 'text', str(e)) if e.response is not None else str(e)})
        except Exception as e:
            return json.dumps({"error": str(e)})

    @make_tool
    def create_resource(endpoint: str, body: str) -> str:
        """
        Create a new resource in Tripletex (POST).

        Args:
            endpoint: API path, e.g. '/employee', '/customer', '/invoice'.
            body:     JSON string of the resource DTO.
                      NEVER include an 'id' field — the API generates it.
                      When referencing another resource, use {"id": <known_id>} as the value.
                      Examples:
                        Employee: '{"firstName":"Ola","lastName":"Nordmann","email":"ola@ex.org","userType":"EXTENDED","department":{"id":123}}'
                        Customer: '{"name":"Acme AS","email":"acme@ex.org"}'
                        Invoice:  '{"customer":{"id":123},"invoiceDate":"2024-01-15"}'

        Returns:
            JSON object of the created resource, including the new 'id'. Store this ID.
        """
        try:
            parsed = _parse_json_lenient(body)
            _log(f"  📤 POST {endpoint} body={json.dumps(parsed, ensure_ascii=False)[:300]}")
            result = client.post(endpoint, parsed)
            _log(f"  📥 Created: id={result.get('id', '?')}")
            return json.dumps(result, ensure_ascii=False)
        except requests.HTTPError as e:
            error_text = ""
            if e.response is not None:
                try:
                    error_text = e.response.text
                except Exception:
                    error_text = f"status {e.response.status_code}"
            _log(f"  ❗ POST {endpoint} failed: {e} -> {error_text[:500]}")
            return json.dumps({"error": str(e), "response": error_text or str(e)})
        except Exception as e:
            _log(f"  ❗ POST {endpoint} error: {e}")
            return json.dumps({"error": str(e)})

    @make_tool
    def update_resource(endpoint: str, resource_id: int, body: str) -> str:
        """
        Update an existing resource in Tripletex (PUT).
        Tripletex uses PUT for partial updates — only send fields you want to change.

        Args:
            endpoint:    API path, e.g. '/employee', '/customer'.
            resource_id: Integer ID of the resource to update.
            body:        JSON string of the fields to update.
                         Include 'id' in the body matching resource_id.
                         Example: '{"id": 123, "phoneNumberMobile": "+4712345678"}'

        Returns:
            JSON object of the updated resource.
        """
        try:
            result = client.put(endpoint, resource_id, _parse_json_lenient(body))
            return json.dumps(result, ensure_ascii=False)
        except requests.HTTPError as e:
            return json.dumps({"error": str(e), "response": getattr(e.response, 'text', str(e)) if e.response is not None else str(e)})
        except Exception as e:
            return json.dumps({"error": str(e)})

    @make_tool
    def delete_resource(endpoint: str, resource_id: int) -> str:
        """
        Delete a resource from Tripletex (DELETE).

        Args:
            endpoint:    API path, e.g. '/travelExpense', '/ledger/voucher'.
            resource_id: Integer ID of the resource to delete.

        Returns:
            JSON confirming deletion or describing the error.
        """
        try:
            client.delete(endpoint, resource_id)
            return json.dumps({"deleted": True, "id": resource_id})
        except requests.HTTPError as e:
            return json.dumps({"error": str(e), "response": getattr(e.response, 'text', str(e)) if e.response is not None else str(e)})
        except Exception as e:
            return json.dumps({"error": str(e)})

    @make_tool
    def action_endpoint(endpoint: str, params: Optional[str] = None) -> str:
        """
        Call a Tripletex action endpoint (PUT with query parameters).
        Used for special operations like registering payments or creating invoices from orders.

        Args:
            endpoint: Full action path including resource ID, e.g.
                      '/invoice/12345/:payment' or '/order/67890/:invoice'.
            params:   Optional JSON string of query parameters.
                      Examples:
                        For invoice payment: '{"paymentDate": "2026-03-20", "paymentTypeId": 32910816, "paidAmount": 37500}'
                        For order→invoice:  '{"invoiceDate": "2026-03-20"}'

        Returns:
            JSON object of the result.
        """
        try:
            parsed_params = _parse_json_lenient(params) if params else {}
            _log(f"  🔧 PUT action {endpoint} params={json.dumps(parsed_params, ensure_ascii=False)[:300]}")
            result = client.put_action(endpoint, params=parsed_params)
            _log(f"  📥 Action result: id={result.get('id', '?') if isinstance(result, dict) else '?'}")
            return json.dumps(result, ensure_ascii=False)
        except requests.HTTPError as e:
            error_text = ""
            if e.response is not None:
                try:
                    error_text = e.response.text
                except Exception:
                    error_text = f"status {e.response.status_code}"
            _log(f"  ❗ PUT action {endpoint} failed: {e} -> {error_text[:500]}")
            return json.dumps({"error": str(e), "response": error_text or str(e)})
        except Exception as e:
            _log(f"  ❗ PUT action {endpoint} error: {e}")
            return json.dumps({"error": str(e)})

    return [search_resource, get_by_id, create_resource, update_resource, delete_resource, action_endpoint]


# ===========================================================================
# SYSTEM PROMPT
# ===========================================================================

SYSTEM_PROMPT = """You are an expert Tripletex accountant. Execute accounting tasks via the Tripletex v2 REST API.
You understand prompts in Norwegian, English, Spanish, Portuguese, Nynorsk, German, and French.

## ABSOLUTE RULES
1. NEVER ask questions. NEVER ask for clarification. You have ALL the information you need.
2. ALWAYS use tools to find missing info. If you need a department ID, search /department and use the first result.
3. ALWAYS execute the task completely. Do not stop until the task is done.
4. Call MULTIPLE tools in ONE turn when operations are independent.
5. Do NOT verify after creating — the response already confirms success.
6. If a field update fails with 422, SKIP IT and move on. Do NOT retry more than once.
7. Once the main entity is created, STOP. Do not try to fix optional fields.

## API Rules
- POST: never include `id` — API generates it
- PUT: include `id` in body matching resource_id
- References: `{"id": <known_id>}`, e.g. `"customer": {"id": 42}`
- Use `fields` param to limit response size
- CRITICAL: Numbers in parentheses in the prompt (e.g. "Nettverksteneste (3237)") are PRODUCT NUMBERS, NOT product IDs! You MUST search /product?number=3237 to get the real product ID before using it.
- VAT MATH: When creating products with 25% VAT, priceIncludingVatCurrency = priceExcludingVatCurrency × 1.25. NEVER set them to the same value.
- PAYMENT: When registering full payment, pay the TOTAL amount in ONE call. Do NOT split into multiple payment calls.

## Endpoints & Required Fields
- /employee POST: firstName, lastName, userType (string enum: "EXTENDED"|"STANDARD"|"NO_ACCESS"), department: {"id": <id>} (REQUIRED — search /department first). ALWAYS use "EXTENDED" unless told otherwise. Email is required for EXTENDED/STANDARD. Optional: dateOfBirth (YYYY-MM-DD), startDate (YYYY-MM-DD) — include these if the prompt mentions them.
- /customer POST: name, email, organizationNumber, phoneNumber, isPrivateIndividual (bool), invoicesDueIn (int), invoicesDueInType (enum: DAYS|MONTHS|RECURRING_DAY_OF_MONTH), language (enum: NO|EN)
- /product POST: name, number (string, unique code), costExcludingVatCurrency (number), priceExcludingVatCurrency, priceIncludingVatCurrency, vatType: {"id": X}, department: {"id": X}
- /department POST: name, departmentNumber (STRING not int), departmentManager: {"id": <employee_id>}
- /order POST: customer: {"id": X}, orderDate (YYYY-MM-DD), deliveryDate (YYYY-MM-DD, REQUIRED!), orderLines: [{"description": "text", "count": 1, "unitCostCurrency": 27900}] — use description+unitCostCurrency for ad-hoc items, OR product: {"id": X}. Optional: invoiceComment, currency: {"id": X}
- /invoice POST: customer: {"id": X}, invoiceDate, invoiceDueDate, orders: [{"id": X}] (must create order FIRST). Optional: comment, kid (KID number)
- /travelExpense POST: employee: {"id": X}, title (string). Do NOT send date, departureDateTime, returnDateTime, description, or perDiemCompensation in the POST body. Optional: project: {"id": X}, department: {"id": X}
- /travelExpense/cost POST: travelExpense: {"id": X}, date (YYYY-MM-DD), description (string), rateCurrency (number — the cost amount), paymentType: {"id": <paymentTypeId>}, currency: {"id": 1} (1=NOK), category: {"id": <category_id>}. Use this to add expenses like flights, taxis, hotels etc.
- /travelExpense/perDiemCompensation POST: travelExpense: {"id": X}, dateFrom (YYYY-MM-DD), dateTo (YYYY-MM-DD), ratePerDay (number), count (number of days), isLunchDeduction (bool, false), isAccommodationProvided (bool, false). Use this to add per diem/daily allowance.
- /invoice/paymentType GET: search with fields=id,description (NOT 'name' — it does not exist on PaymentTypeDTO)
- /project POST: name, projectManager: {"id": <employee_id>}, department: {"id": X}, startDate (YYYY-MM-DD). Optional: customer: {"id": X}, endDate, description, isFixedPrice (bool), fixedprice (number)
- /supplier POST: name, email, organizationNumber, phoneNumber, supplierNumber (int), bankAccounts (array of strings)
- /contact POST: firstName, lastName, email, phoneNumberMobile, customer: {"id": X} OR supplier: {"id": X}
- /ledger/account GET: search with ?number=<acct_number>&fields=id,number,name
- /ledger/voucher POST: date (YYYY-MM-DD), description, postings (array). Each posting: {"row": <N starting from 1>, "account": {"id": <acct_id>}, "amountGross": <positive=debit, negative=credit>, "description": "text"}. For AP postings add "supplier": {"id": X}. For AR postings add "customer": {"id": X}. All amountGross values MUST sum to 0. Use amountGross (not amount) for the posting value.
- /ledger/voucherType GET: search with fields=id,name — e.g. "Betaling", "Utgående faktura", "Inngående faktura"
- IMPORTANT: POST /supplierInvoice does NOT EXIST. Supplier invoices are READ-ONLY. To RECORD a supplier invoice, use POST /ledger/voucher with: debit the expense account, debit the input VAT account (2710), credit the supplier AP account (2400) with supplier ref. Include voucherType for "Inngående faktura".

## Action Endpoints (use action_endpoint tool — PUT with query params)
- /invoice/{id}/:payment — params: paymentDate (YYYY-MM-DD), paymentTypeId (int), paidAmount (amount in payment type currency). Get paymentTypeId from /invoice/paymentType (use fields=id,description).
- /order/{id}/:invoice — params: invoiceDate (YYYY-MM-DD), sendToCustomer (bool, default true). Creates invoice from order.
- /invoice/{id}/:send — params: sendType (EMAIL|EHF|PAPER|MANUAL), overrideEmailAddress (optional)
- /invoice/{id}/:createCreditNote — params: date (YYYY-MM-DD), comment, sendToCustomer (bool)
- /supplierInvoice/{id}/:addPayment — params: paymentType (int, 0=last used), amount, paymentDate
- /supplierInvoice/{id}/:approve — params: comment (optional)
- /travelExpense/:deliver — params: id (comma-separated IDs)
- /travelExpense/:approve — params: id (comma-separated IDs)
- /ledger/voucher/{id}/:reverse — params: date (YYYY-MM-DD)

## Admin Role Detection (CRITICAL — worth 5/10 points on employee tasks!)
When the prompt mentions any of these keywords, set userType="EXTENDED":
- Norwegian: "administrator", "admin-tilgang", "full tilgang", "administratortilgang"
- Nynorsk: "administrator", "administratortilgang"
- English: "administrator", "admin access", "full access"
- Spanish: "administrador", "acceso de administrador"
- Portuguese: "administrador", "acesso de administrador"
- German: "Administrator", "Administratorzugang"
- French: "administrateur", "accès administrateur"
If NO admin keyword is present, STILL use userType="EXTENDED" (default).

## Task Patterns (follow these exactly)

### Tier 1 — Simple creates
- Create employee → search /department to get dept ID → create_resource /employee with firstName, lastName, email, userType="EXTENDED", department.id
- Create customer → create_resource /customer directly with name (+ optional fields). NO need to search first.
- Create supplier → create_resource /supplier directly with name (+ optional fields). NO need to search first.
- Create department → create_resource /department with name + departmentNumber (string!)
- Create product → create_resource /product with name, number. NO need to search first.
- Create contact → find customer/supplier → create_resource /contact with firstName, lastName, customer/supplier ref
- Multiple creates → call create_resource multiple times IN ONE TURN

### Tier 2 — Multi-step workflows
- Create order → invoice → payment → 1) search /customer by organizationNumber, 2) search /product?number=XXXX for EACH product number in the prompt to get real IDs, 3) create order with product: {"id": <real_id>} in orderLines, 4) action_endpoint "/order/{order_id}/:invoice", 5) search /invoice/paymentType to get paymentTypeId, 6) action_endpoint "/invoice/{id}/:payment" with TOTAL amount in ONE call
- Create & send invoice → same as above + action_endpoint "/invoice/{id}/:send" with sendType
- Register invoice payment → 1) search /invoice (needs invoiceDateFrom+invoiceDateTo), 2) search /invoice/paymentType for paymentTypeId, 3) action_endpoint "/invoice/{id}/:payment" with paymentDate, paymentTypeId, paidAmount (TOTAL in ONE call)
- Credit note → 1) find invoice, 2) action_endpoint "/invoice/{id}/:createCreditNote" with date, comment, sendToCustomer
- Travel expense (simple) → search /employee → create_resource /travelExpense with employee.id + title
- Travel expense (with costs) → 1) search /employee, 2) create_resource /travelExpense with employee.id + title, 3) for EACH expense mentioned: create_resource /travelExpense/cost with travelExpense.id + description + rateCurrency + date, 4) if per diem/daily allowance mentioned: create_resource /travelExpense/perDiemCompensation with travelExpense.id + dateFrom + dateTo + ratePerDay + count
- Delete travel expense → search /travelExpense → delete_resource
- Create project → search /employee AND /department in parallel → create_resource /project with name, projectManager.id, department.id, startDate
- Record supplier invoice (leverandørfaktura) → NEVER use POST /supplierInvoice (it doesn't exist!). Instead:
  1) search /supplier by organizationNumber AND search /ledger/account for expense account (e.g. 7100) AND search /ledger/account for input VAT account (2710) AND search /ledger/account for AP account (2400) — ALL IN PARALLEL
  2) calculate: netAmount = totalInclVat / 1.25, vatAmount = totalInclVat - netAmount
  3) create_resource /ledger/voucher with postings:
     row 1: debit expense account (amountGross = +netAmount, description)
     row 2: debit input VAT account 2710 (amountGross = +vatAmount)
     row 3: credit AP account 2400 (amountGross = -totalInclVat, supplier: {"id": X})
     All amountGross values MUST sum to 0.
- Supplier invoice approval + payment → 1) search /supplierInvoice, 2) action_endpoint "/supplierInvoice/{id}/:approve", 3) action_endpoint "/supplierInvoice/{id}/:addPayment" with paymentType, amount, paymentDate
- Delete → find resource → delete_resource
- Update → find resource → update_resource with id in body

### Tier 3 — Complex scenarios
- Ledger voucher → search /ledger/account by number → create_resource /ledger/voucher with date, description, postings (row, account.id, amountGross, customer/supplier ref if AR/AP)
- Reverse ledger entry → 1) find voucher, 2) action_endpoint "/ledger/voucher/{id}/:reverse" with date, 3) create corrected voucher
- Bank reconciliation → parse CSV file → search /ledger/account once for all needed accounts → batch create /ledger/voucher entries

## Efficiency Rules (CRITICAL — affects up to ×2 bonus)
- Do NOT search before creating customers, suppliers, or products — sandbox starts empty, no duplicates possible.
- The POST response contains the created ID — use it directly, never re-fetch.
- Fetch multiple prerequisites (employee + department) IN PARALLEL in one turn.
- Every 4xx error hurts your efficiency score. Get fields right the first time.

## Error Recovery
- 422 → read validationMessages, fix the field, retry ONCE only
- 409 → resource exists, search for it instead
- If a field "does not exist in the object" → remove that field and retry

## REMEMBER: Act immediately. Never ask questions. Search for any missing info. Complete the task in minimum tool calls.
"""


# ===========================================================================
# PLANNING STEP
# ===========================================================================

_PLANNING_PROMPT = """You are a Tripletex API planning assistant.

Given an accounting task prompt, produce a concise execution plan as JSON.
The plan will be given to an agent that executes Tripletex API calls.

Output ONLY valid JSON with this structure:
{{
  "language": "<detected language code: nb|en|es|pt|nn|de|fr>",
  "task_type": "<one of: employee|customer_product|invoice|travel_expense|project|correction|department>",
  "task_summary": "<one sentence in English describing what needs to be done>",
  "steps": [
    {{"step": 1, "action": "<search|create|update|delete>", "endpoint": "/...", "notes": "..."}}
  ],
  "prerequisites": ["<anything to check or fetch first>"],
  "warnings": ["<potential pitfalls, e.g. module activation needed>"]
}}

Task prompt:
{prompt}

Attached files: {file_summary}
"""


def _build_execution_plan(prompt: str, file_summary: str) -> str:
    """
    Run a single LLM call to produce a structured execution plan.
    Returns the plan as a formatted string to inject into the agent prompt.
    Falls back gracefully if the planning call fails.
    """
    from tasks.language.factory import get_llm
    from langchain_core.messages import HumanMessage

    try:
        llm = get_llm()
        planning_input = _PLANNING_PROMPT.format(
            prompt=prompt,
            file_summary=file_summary or "none",
        )
        response = llm.invoke([HumanMessage(content=planning_input)])
        plan_text = response.content.strip()

        # Strip markdown code fences if present
        if plan_text.startswith("```"):
            plan_text = plan_text.split("```")[1]
            if plan_text.startswith("json"):
                plan_text = plan_text[4:]

        plan = json.loads(plan_text)
        logger.info("Execution plan: %s", plan.get("task_summary"))

        # Format plan as readable text for the agent prompt
        lines = [
            "## Execution Plan",
            f"- Language: {plan.get('language', '?')}",
            f"- Task type: {plan.get('task_type', '?')}",
            f"- Summary: {plan.get('task_summary', '?')}",
        ]
        if plan.get("prerequisites"):
            lines.append("- Prerequisites: " + "; ".join(plan["prerequisites"]))
        if plan.get("warnings"):
            lines.append("- Warnings: " + "; ".join(plan["warnings"]))
        lines.append("- Steps:")
        for s in plan.get("steps", []):
            lines.append(f"  {s['step']}. [{s['action'].upper()}] {s['endpoint']} — {s['notes']}")

        return "\n".join(lines)

    except Exception as exc:
        logger.warning("Planning step failed (%s) — proceeding without plan", exc)
        return ""


# ===========================================================================
# FILE CONTEXT
# ===========================================================================

def _build_file_context(files: list[FileAttachment]) -> str:
    """
    Extract and format content from all file attachments.
    Returns a text block ready to be appended to the agent prompt.
    """
    if not files:
        return ""

    from tasks.tripletex.files import process_attachment

    parts = ["\n\n## Attached Files"]
    for f in files:
        parts.append(process_attachment(f))

    return "\n\n".join(parts)


# ===========================================================================
# MAIN ENTRY POINT
# ===========================================================================

def solve(request: SolveRequest) -> SolveResponse:
    """
    Execute a Tripletex accounting task.

    1. Extract file content (if any)
    2. Run the AgentRunner — executes API calls directly
    3. Return {"status": "completed"}
    """
    import time
    t0 = time.perf_counter()

    client = TripletexClient(
        base_url=request.tripletex_credentials.base_url,
        session_token=request.tripletex_credentials.session_token,
    )

    tools = build_tools(client)

    agent = AgentRunner(
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        max_iterations=25,
        verbose=True,
    )

    file_context = _build_file_context(request.files)

    # Planning step — produces an explicit execution plan to guide the agent
    file_summary = "none"
    if request.files:
        file_summary = ", ".join(f.filename for f in request.files)
    plan = _build_execution_plan(request.prompt, file_summary)
    if plan:
        _log(f"📝 Plan:\n{plan}")

    # Build the full prompt: plan + task + file content
    sections = []
    if plan:
        sections.append(plan)
    sections.append(f"## Task\n{request.prompt}")
    if file_context:
        sections.append(file_context)

    full_prompt = "\n".join(sections)

    _log("TASK_START", task_prompt=request.prompt[:500],
         files=[f.filename for f in request.files],
         base_url=request.tripletex_credentials.base_url,
         has_plan=bool(plan))
    try:
        result = agent.run(full_prompt)
        elapsed = (time.perf_counter() - t0) * 1000
        _log("TASK_COMPLETE", elapsed_ms=round(elapsed),
             output=result["output"][:500])
    except Exception as exc:
        import traceback
        elapsed = (time.perf_counter() - t0) * 1000
        _log("TASK_FAILED", severity="ERROR", elapsed_ms=round(elapsed),
             error=str(exc), traceback=traceback.format_exc())

    return SolveResponse(status="completed")

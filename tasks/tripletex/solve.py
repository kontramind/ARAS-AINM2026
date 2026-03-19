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
from typing import Any, Optional

import requests
import urllib3
from pydantic import BaseModel

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from tasks.language.agent import AgentRunner, make_tool

logger = logging.getLogger(__name__)


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
        try:
            result = client.get(endpoint, params=params)
            # Ensure we always return a list
            if not isinstance(result, list):
                result = [result]
            return json.dumps(result, ensure_ascii=False)
        except requests.HTTPError as e:
            return json.dumps({"error": str(e), "response": getattr(e.response, 'text', str(e)) if e.response is not None else str(e)})

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
                        Employee: '{"firstName":"Ola","lastName":"Nordmann","email":"ola@ex.org","userType":1,"department":{"id":123}}'
                        Customer: '{"name":"Acme AS","email":"acme@ex.org"}'
                        Invoice:  '{"customer":{"id":123},"invoiceDate":"2024-01-15"}'

        Returns:
            JSON object of the created resource, including the new 'id'. Store this ID.
        """
        try:
            parsed = _parse_json_lenient(body)
            print(f"  📤 POST {endpoint} body={json.dumps(parsed, ensure_ascii=False)[:300]}")
            result = client.post(endpoint, parsed)
            print(f"  📥 Created: id={result.get('id', '?')}")
            return json.dumps(result, ensure_ascii=False)
        except requests.HTTPError as e:
            error_text = ""
            if e.response is not None:
                try:
                    error_text = e.response.text
                except Exception:
                    error_text = f"status {e.response.status_code}"
            print(f"  ❗ POST {endpoint} failed: {e} -> {error_text[:500]}")
            return json.dumps({"error": str(e), "response": error_text or str(e)})

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

    return [search_resource, get_by_id, create_resource, update_resource, delete_resource]


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

## Endpoints & Required Fields
- /employee POST: firstName, lastName, userType (2=no-login, 1=login requires email), department: {"id": <id>} (REQUIRED — search /department first). Do NOT include dateOfBirth or startDate in POST — they cause 422.
- /customer POST: name (required), email, organizationNumber
- /product POST: name, number (unique string code), costExcludingVatCurrency (number, price excl VAT)
- /department POST: name, departmentNumber (unique integer)
- /order POST: customer: {"id": X}, orderDate (YYYY-MM-DD), deliveryDate (YYYY-MM-DD, REQUIRED!), orderLines: [{"description": "text", "count": 1, "unitCostCurrency": 27900}] — use description+unitCostCurrency for ad-hoc items, OR product: {"id": X} for existing products
- /invoice POST: customer: {"id": X}, invoiceDate (YYYY-MM-DD), invoiceDueDate (YYYY-MM-DD, REQUIRED!), orders: [{"id": X}] (REQUIRED — must create order FIRST, then reference it)
- /travelExpense POST: employee: {"id": X}, description, date (YYYY-MM-DD)
- /project POST: name, customer: {"id": X}, department: {"id": X}
- /ledger/account GET only: number, name
- /ledger/posting GET only: date, amount, account
- /ledger/voucher POST/DELETE: postings[{account: {"id": X}, amount}]

## Task Patterns (follow these exactly)
- Create employee → FIRST search_resource /department to get dept ID, THEN create_resource /employee with userType=2 + department.id. If email is given, use userType=1.
- Create customer → create_resource /customer directly
- Create department → create_resource /department directly with name + departmentNumber
- Multiple creates → call create_resource multiple times IN ONE TURN
- Create invoice → Step 1: find/create customer. Step 2: create order with orderDate, deliveryDate, and orderLines (use description+unitCostCurrency for line items). Step 3: create invoice with invoiceDate, invoiceDueDate, and orders[{id}].
- Travel expense → find employee, create travelExpense with employee.id + description + date
- Delete resource → find it first, then delete_resource
- Update resource → find it first, then update_resource with id in body

## Error Recovery
- 422 → read validationMessages, fix the field, retry once
- 409 → resource exists, search for it instead

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
        max_iterations=15,
        verbose=True,
    )

    file_context = _build_file_context(request.files)

    # Build the full prompt: task + file content
    sections = [request.prompt]
    if file_context:
        sections.append(file_context)

    full_prompt = "\n".join(sections)

    print(f"📋 Task: {request.prompt[:200]}")
    try:
        result = agent.run(full_prompt)
        elapsed = (time.perf_counter() - t0) * 1000
        print(f"✅ Agent completed in {elapsed:.0f}ms. Output: {result['output'][:200]}")
    except Exception as exc:
        elapsed = (time.perf_counter() - t0) * 1000
        print(f"❌ Agent failed in {elapsed:.0f}ms: {exc}")
        logger.error("Agent execution failed: %s", exc, exc_info=True)

    return SolveResponse(status="completed")

"""
tasks/tripletex/solve.py
------------------------
Tripletex AI Accounting Agent — Router Architecture.

Entry point: solve(request: SolveRequest) -> SolveResponse

Flow:
  1. Parse the incoming request (prompt, files, Tripletex credentials)
  2. Route: classify the task type with a fast LLM call
  3. Select the focused sub-agent prompt for that task type
  4. Run the AgentRunner with the focused prompt — executes API calls
  5. Return {"status": "completed"}

Authentication: Basic Auth, username="0", password=session_token
All API calls go through tripletex_credentials.base_url (competition proxy).
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

    @staticmethod
    def _clean_params(params: dict | None) -> dict:
        """Convert Python booleans to lowercase strings for API compatibility."""
        clean = {}
        for k, v in (params or {}).items():
            if isinstance(v, bool):
                clean[k] = str(v).lower()
            else:
                clean[k] = v
        return clean

    def get(self, endpoint: str, params: dict | None = None) -> Any:
        resp = self._session.get(self._url(endpoint), params=self._clean_params(params))
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

    def put_action(self, endpoint: str, params: dict | None = None, body: dict | list | None = None) -> Any:
        """PUT with query params + optional body (for Tripletex action endpoints)."""
        resp = self._session.put(self._url(endpoint), params=self._clean_params(params), json=body if body is not None else {})
        resp.raise_for_status()
        if not resp.content:
            return {"success": True}
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
            # Log first results for debugging
            if result and endpoint in ("/balanceSheet", "/ledger/voucher", "/invoice", "/supplierInvoice"):
                _log(f"  📊 Data: {json.dumps(result[:10], ensure_ascii=False)[:1500]}")
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
            _log(f"  📤 POST {endpoint} body={json.dumps(parsed, ensure_ascii=False)[:800]}")
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
    def action_endpoint(endpoint: str, params: Optional[str] = None, body: Optional[str] = None) -> str:
        """
        Call a Tripletex action endpoint (PUT with query parameters and optional body).
        Used for special operations like registering payments or creating invoices from orders.

        Args:
            endpoint: Full action path including resource ID, e.g.
                      '/invoice/12345/:payment' or '/order/67890/:invoice'.
            params:   Optional JSON string of query parameters.
                      Examples:
                        For invoice payment: '{"paymentDate": "2026-03-20", "paymentTypeId": 32910816, "paidAmount": 37500}'
                        For order→invoice:  '{"invoiceDate": "2026-03-20"}'
            body:     Optional JSON string of request body (for action endpoints that require a body).

        Returns:
            JSON object of the result.
        """
        try:
            parsed_params = _parse_json_lenient(params) if params else {}
            parsed_body = _parse_json_lenient(body) if body else None
            _log(f"  🔧 PUT action {endpoint} params={json.dumps(parsed_params, ensure_ascii=False)[:300]}")
            result = client.put_action(endpoint, params=parsed_params, body=parsed_body)
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
# ROUTER — classifies task type with a fast LLM call
# ===========================================================================

def _route_task(prompt: str) -> str:
    """
    Classify the task into a category using a single fast LLM call.
    Returns a key from TASK_CONFIG, or 'generic' as fallback.
    """
    from tasks.language.factory import get_llm
    from langchain_core.messages import HumanMessage
    from tasks.tripletex.prompts import ROUTER_PROMPT, TASK_CONFIG

    try:
        llm = get_llm()
        response = llm.invoke([HumanMessage(content=ROUTER_PROMPT.format(prompt=prompt))])

        # Extract the category — handle both string and list content (Gemini thinking)
        raw = response.content
        if isinstance(raw, list):
            raw = " ".join(
                b.get("text", "") if isinstance(b, dict) else str(b)
                for b in raw
            )
        task_type = raw.strip().lower().replace('"', "").replace("'", "")

        # Exact match
        if task_type in TASK_CONFIG:
            return task_type

        # Fuzzy match — find category name within response
        for key in TASK_CONFIG:
            if key in task_type:
                return key

        _log("ROUTER_FALLBACK", severity="WARNING",
             raw_response=task_type, reason="no match in TASK_CONFIG")
        return "generic"

    except Exception as exc:
        _log("ROUTER_ERROR", severity="ERROR", error=str(exc))
        return "generic"


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

    1. Route: classify the task type
    2. Select the focused sub-agent prompt
    3. Run the AgentRunner with focused prompt
    4. Return {"status": "completed"}
    """
    import time
    from tasks.tripletex.prompts import TASK_CONFIG, GENERIC_PROMPT

    t0 = time.perf_counter()

    # Step 1: Route the task
    task_type = _route_task(request.prompt)
    config = TASK_CONFIG.get(task_type)
    if config:
        system_prompt = config["prompt"]
        max_iter = config["max_iter"]
    else:
        system_prompt = GENERIC_PROMPT
        max_iter = 25

    _log("TASK_ROUTED", task_type=task_type, max_iter=max_iter,
         prompt_size=len(system_prompt))

    # Step 2: Build tools and agent
    client = TripletexClient(
        base_url=request.tripletex_credentials.base_url,
        session_token=request.tripletex_credentials.session_token,
    )
    tools = build_tools(client)

    agent = AgentRunner(
        tools=tools,
        system_prompt=system_prompt,
        max_iterations=max_iter,
        verbose=True,
    )

    # Step 3: Build the user prompt
    file_context = _build_file_context(request.files)
    today = datetime.utcnow().strftime("%Y-%m-%d")
    sections = [request.prompt, f"\n[Today's date: {today}]"]
    if file_context:
        sections.append(file_context)
    full_prompt = "\n".join(sections)

    if file_context:
        _log("FILE_CONTEXT", task_type=task_type,
             file_context_preview=file_context[:1500],
             file_context_length=len(file_context))

    _log("TASK_START", task_type=task_type,
         task_prompt=request.prompt[:500],
         files=[f.filename for f in request.files],
         base_url=request.tripletex_credentials.base_url)

    # Step 4: Run the agent
    try:
        result = agent.run(full_prompt)
        elapsed = (time.perf_counter() - t0) * 1000
        _log("TASK_COMPLETE", task_type=task_type,
             elapsed_ms=round(elapsed),
             output=result["output"][:500])
    except Exception as exc:
        import traceback
        elapsed = (time.perf_counter() - t0) * 1000
        _log("TASK_FAILED", severity="ERROR", task_type=task_type,
             elapsed_ms=round(elapsed),
             error=str(exc), traceback=traceback.format_exc())

    return SolveResponse(status="completed")

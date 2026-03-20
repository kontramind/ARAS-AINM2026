"""
tests/test_tripletex.py
-----------------------
Integration tests for the Tripletex accounting agent.

Two test layers:
  1. Unit tests  — no network, test client/tool logic with mocks
  2. Sandbox tests — hit the real Tripletex sandbox (requires env vars)

Run unit tests only (default):
    pytest tests/test_tripletex.py -v

Run sandbox tests (requires credentials):
    TRIPLETEX_BASE_URL=https://tx-proxy.ainm.no/v2 \\
    TRIPLETEX_SESSION_TOKEN=<token> \\
    pytest tests/test_tripletex.py -v -m sandbox
"""

import json
import os
import pytest
from unittest.mock import MagicMock, patch

from tasks.tripletex.solve import (
    TripletexClient,
    SolveRequest,
    SolveResponse,
    FileAttachment,
    TripletexCredentials,
    build_tools,
    solve,
)


# ===========================================================================
# FIXTURES
# ===========================================================================

@pytest.fixture
def mock_client():
    """TripletexClient with a mocked requests.Session."""
    client = TripletexClient(base_url="https://fake.proxy/v2", session_token="test-token")
    client._session = MagicMock()
    return client


@pytest.fixture
def sandbox_credentials():
    """
    Real Tripletex sandbox credentials from environment.
    Tests using this fixture are skipped unless both vars are set.
    """
    base_url = os.getenv("TRIPLETEX_BASE_URL")
    session_token = os.getenv("TRIPLETEX_SESSION_TOKEN")
    if not base_url or not session_token:
        pytest.skip("TRIPLETEX_BASE_URL and TRIPLETEX_SESSION_TOKEN not set")
    return TripletexCredentials(base_url=base_url, session_token=session_token)


# ===========================================================================
# UNIT — TripletexClient response unwrapping
# ===========================================================================

class TestTripletexClientUnwrapping:

    def test_get_unwraps_values_list(self, mock_client):
        mock_client._session.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"fullResultSize": 2, "values": [{"id": 1}, {"id": 2}]},
            raise_for_status=lambda: None,
        )
        result = mock_client.get("/employee")
        assert result == [{"id": 1}, {"id": 2}]

    def test_get_unwraps_single_value(self, mock_client):
        mock_client._session.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"value": {"id": 42, "firstName": "Ola"}},
            raise_for_status=lambda: None,
        )
        result = mock_client.get("/employee/42")
        assert result == {"id": 42, "firstName": "Ola"}

    def test_get_passes_through_bare_dict(self, mock_client):
        mock_client._session.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"token": "abc", "expirationDate": "2025-01-01"},
            raise_for_status=lambda: None,
        )
        result = mock_client.get("/token/session")
        assert result["token"] == "abc"

    def test_get_empty_list(self, mock_client):
        mock_client._session.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"fullResultSize": 0, "values": []},
            raise_for_status=lambda: None,
        )
        result = mock_client.get("/customer")
        assert result == []

    def test_post_unwraps_value(self, mock_client):
        mock_client._session.post.return_value = MagicMock(
            status_code=201,
            json=lambda: {"value": {"id": 99, "firstName": "Test"}},
            raise_for_status=lambda: None,
        )
        result = mock_client.post("/employee", {"firstName": "Test"})
        assert result["id"] == 99

    def test_put_unwraps_value(self, mock_client):
        mock_client._session.put.return_value = MagicMock(
            status_code=200,
            json=lambda: {"value": {"id": 99, "firstName": "Updated"}},
            raise_for_status=lambda: None,
        )
        result = mock_client.put("/employee", 99, {"id": 99, "firstName": "Updated"})
        assert result["firstName"] == "Updated"

    def test_delete_returns_true(self, mock_client):
        mock_client._session.delete.return_value = MagicMock(
            status_code=204,
            raise_for_status=lambda: None,
        )
        assert mock_client.delete("/travelExpense", 5) is True


# ===========================================================================
# UNIT — Tool error handling
# ===========================================================================

class TestToolErrorHandling:

    def test_search_resource_returns_error_json_on_http_error(self, mock_client):
        import requests as req
        err_response = MagicMock(text='{"message":"Not found"}')
        mock_client._session.get.return_value = MagicMock(
            raise_for_status=MagicMock(side_effect=req.HTTPError(response=err_response))
        )
        tools = {t.name: t for t in build_tools(mock_client)}
        result = json.loads(tools["search_resource"].invoke({"endpoint": "/employee"}))
        assert "error" in result

    def test_search_resource_injects_default_fields(self, mock_client):
        mock_client._session.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"values": []},
            raise_for_status=lambda: None,
        )
        build_tools(mock_client)[0].invoke({"endpoint": "/customer"})
        call_kwargs = mock_client._session.get.call_args
        assert "fields" in call_kwargs.kwargs.get("params", call_kwargs.args[1] if len(call_kwargs.args) > 1 else {})

    def test_create_resource_never_errors_on_valid_json(self, mock_client):
        mock_client._session.post.return_value = MagicMock(
            status_code=201,
            json=lambda: {"value": {"id": 1}},
            raise_for_status=lambda: None,
        )
        tools = {t.name: t for t in build_tools(mock_client)}
        result = json.loads(tools["create_resource"].invoke({
            "endpoint": "/employee",
            "body": '{"firstName":"Ola","lastName":"Nordmann"}',
        }))
        assert result["id"] == 1


# ===========================================================================
# UNIT — SolveRequest / SolveResponse schemas
# ===========================================================================

def test_solve_request_defaults_files_to_empty():
    req = SolveRequest(
        prompt="Test",
        tripletex_credentials=TripletexCredentials(
            base_url="https://proxy/v2",
            session_token="token",
        ),
    )
    assert req.files == []


def test_solve_response_default_status():
    assert SolveResponse().status == "completed"


# ===========================================================================
# SANDBOX — Real API round-trips
# Skipped unless TRIPLETEX_BASE_URL + TRIPLETEX_SESSION_TOKEN are set
# ===========================================================================

@pytest.mark.sandbox
class TestSandboxRoundTrips:
    """
    Each test creates a resource and verifies it was persisted correctly.
    Tests are independent — each uses a unique identifier to avoid conflicts.
    """

    def test_create_employee(self, sandbox_credentials):
        client = TripletexClient(
            base_url=sandbox_credentials.base_url,
            session_token=sandbox_credentials.session_token,
        )
        # Employee creation requires userType + department.id (learned from sandbox)
        depts = client.get("/department", params={"fields": "id,name"})
        assert depts, "No departments found in sandbox — cannot create employee"
        dept_id = depts[0]["id"]

        import uuid
        unique = uuid.uuid4().hex[:6]
        result = client.post("/employee", {
            "firstName": "Test",
            "lastName": f"Employee{unique}",
            "email": f"test_{unique}@ainm-test.no",
            "userType": 1,
            "department": {"id": dept_id},
        })
        assert "id" in result, f"Expected id in response, got: {result}"
        employee_id = result["id"]

        fetched = client.get(f"/employee/{employee_id}", params={"fields": "id,firstName,lastName,email"})
        assert fetched["firstName"] == "Test"
        assert fetched["lastName"] == f"Employee{unique}"

    def test_create_customer(self, sandbox_credentials):
        client = TripletexClient(
            base_url=sandbox_credentials.base_url,
            session_token=sandbox_credentials.session_token,
        )
        result = client.post("/customer", {
            "name": "AINM Test AS",
            "email": "ainm-test@example.no",
        })
        assert "id" in result

        fetched = client.get(f"/customer/{result['id']}", params={"fields": "id,name,email"})
        assert fetched["name"] == "AINM Test AS"

    def test_search_returns_list(self, sandbox_credentials):
        client = TripletexClient(
            base_url=sandbox_credentials.base_url,
            session_token=sandbox_credentials.session_token,
        )
        result = client.get("/employee", params={"fields": "id,firstName,lastName"})
        assert isinstance(result, list)

    def test_full_agent_create_employee(self, sandbox_credentials):
        """
        Full end-to-end: run the agent with a simple Norwegian prompt.
        Verifies the agent returns completed and the API actually created the employee.
        """
        req = SolveRequest(
            prompt="Opprett en ansatt med fornavn AgentTest og etternavn Sandviken, epost agenttest@ainm-test.no.",
            tripletex_credentials=sandbox_credentials,
        )
        response = solve(req)
        assert response.status == "completed"

        client = TripletexClient(
            base_url=sandbox_credentials.base_url,
            session_token=sandbox_credentials.session_token,
        )
        employees = client.get("/employee", params={
            "fields": "id,firstName,lastName,email",
            "count": 100,
        })
        first_names = [e.get("firstName", "") for e in employees]
        assert "AgentTest" in first_names, f"Employee not found. Employees: {first_names}"

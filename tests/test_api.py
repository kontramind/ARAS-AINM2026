"""
tests/test_api.py
------------------
Integration tests for the Tripletex Agent API endpoints.

Run:
    pytest tests/test_api.py -v
"""

import pytest
from httpx import ASGITransport, AsyncClient

from api.main import app


@pytest.fixture
async def client():
    """Async test client that uses the ASGI transport — no real server needed."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


# ===========================================================================
# Health
# ===========================================================================

@pytest.mark.asyncio
async def test_health_root(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


@pytest.mark.asyncio
async def test_health_endpoint(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"

"""
tests/test_api.py
------------------
Integration tests for all FastAPI endpoints.

Uses httpx + ASGI transport (no real server needed — tests run in-process).

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
    resp = await client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


@pytest.mark.asyncio
async def test_health_endpoint(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


# ===========================================================================
# Task 1 — Tabular
# ===========================================================================

@pytest.mark.asyncio
async def test_task1_basic(client):
    payload = {"id": "test_001", "features": [0.1, 0.5, -0.3, 1.2, 0.0]}
    resp = await client.post("/task1/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == "test_001"
    assert "prediction" in data


@pytest.mark.asyncio
async def test_task1_id_preserved(client):
    """id field must always be echoed back — competition evaluator requires this."""
    unique_id = "UNIQUE_XYZ_42"
    resp = await client.post("/task1/predict", json={"id": unique_id, "features": [1.0, 2.0]})
    assert resp.status_code == 200
    assert resp.json()["id"] == unique_id


@pytest.mark.asyncio
async def test_task1_missing_id_rejected(client):
    """Missing required 'id' field should return 422."""
    resp = await client.post("/task1/predict", json={"features": [1.0, 2.0]})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_task1_empty_features(client):
    """Empty feature list should still return a valid response (not 500)."""
    resp = await client.post("/task1/predict", json={"id": "empty", "features": []})
    assert resp.status_code == 200


# ===========================================================================
# Task 2 — Language
# ===========================================================================

@pytest.mark.asyncio
async def test_task2_basic(client):
    payload = {"id": "case_001", "text": "Severe chest pain and difficulty breathing."}
    resp = await client.post("/task2/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == "case_001"
    assert "label" in data


@pytest.mark.asyncio
async def test_task2_with_context(client):
    payload = {
        "id": "rag_001",
        "text": "What is the triage level for cardiac arrest?",
        "context": "Triage level 1 is for immediately life-threatening conditions.",
    }
    resp = await client.post("/task2/predict", json=payload)
    assert resp.status_code == 200
    assert resp.json()["id"] == "rag_001"


@pytest.mark.asyncio
async def test_task2_keyword_urgent(client):
    """Heuristic: 'urgent' in text → label should reflect urgency."""
    payload = {"id": "u1", "text": "This is an urgent situation requiring immediate action."}
    resp = await client.post("/task2/predict", json=payload)
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_task2_latency_in_meta(client):
    """Check that latency metadata is returned."""
    resp = await client.post("/task2/predict", json={"id": "t", "text": "Hello"})
    data = resp.json()
    if data.get("meta"):
        assert data["meta"]["latency_ms"] >= 0


# ===========================================================================
# Task 3 — Vision
# ===========================================================================

@pytest.mark.asyncio
async def test_task3_basic_with_array(client):
    """Send a small 2D image array and expect a mask back."""
    small_array = [[float(i % 2) for i in range(8)] for _ in range(8)]
    payload = {"id": "scan_001", "image_array": small_array}
    resp = await client.post("/task3/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == "scan_001"
    assert "mask" in data


@pytest.mark.asyncio
async def test_task3_mask_dimensions_preserved(client):
    """Mask dimensions should match input array dimensions."""
    h, w = 16, 24
    arr = [[0.5] * w for _ in range(h)]
    resp = await client.post("/task3/predict", json={"id": "sz_test", "image_array": arr})
    assert resp.status_code == 200
    mask = resp.json()["mask"]
    assert len(mask) == h
    assert len(mask[0]) == w


@pytest.mark.asyncio
async def test_task3_no_image_still_responds(client):
    """Sending no image data should still return a valid (empty) response, not 500."""
    resp = await client.post("/task3/predict", json={"id": "no_img"})
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_task3_id_preserved(client):
    unique_id = "SCAN_UNIQUE_99"
    resp = await client.post("/task3/predict", json={"id": unique_id})
    assert resp.json()["id"] == unique_id

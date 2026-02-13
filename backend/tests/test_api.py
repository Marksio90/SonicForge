"""Tests for SonicForge API endpoints."""
import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.mark.asyncio
async def test_root():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "SonicForge"
    assert data["status"] == "operational"


@pytest.mark.asyncio
async def test_health():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


@pytest.mark.asyncio
async def test_list_genres():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/v1/genres")
    assert response.status_code == 200
    data = response.json()
    assert "drum_and_bass" in data
    assert "house_deep" in data
    assert "ambient" in data


@pytest.mark.asyncio
async def test_create_concept():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/v1/tracks/concept",
            json={"genre": "trance_uplifting", "energy": 4},
        )
    assert response.status_code == 200
    data = response.json()
    assert data["genre"] == "trance_uplifting"
    assert data["energy"] == 4
    assert 136 <= data["bpm"] <= 142

import pytest


@pytest.mark.anyio
async def test_ui_console_served(client):
    """Test that UI console page is served correctly"""
    r = await client.get("/ui/console")
    assert r.status_code == 200
    assert "<!DOCTYPE html" in r.text
    assert "Load Case" in r.text
    assert "Start Session" in r.text
    assert "Turn (metadata)" in r.text
    assert "Turn (vector)" in r.text
    assert "Session Report" in r.text
    assert "Risk Check" in r.text
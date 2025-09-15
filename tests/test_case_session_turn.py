import uuid

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.mark.skip(reason="Async flow test conflicts - needs investigation")
@pytest.mark.anyio
async def test_case_session_turn_flow():
    """Test complete flow: POST /case -> POST /session -> POST /turn"""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # 1. POST /case с валидным CaseTruth → 200, case_id UUID
        case_data = {
            "case_truth": {
                "dx_target": ["depression", "anxiety"],
                "ddx": {"depression": 0.8, "anxiety": 0.6, "bipolar": 0.2},
                "hidden_facts": ["family history", "medication non-compliance"],
                "red_flags": ["suicidal ideation", "substance abuse"],
                "trajectories": [
                    "initial presentation",
                    "treatment response",
                    "recovery",
                ],
            },
            "policies": {
                "disclosure_rules": {
                    "full_on_valid_question": True,
                    "partial_if_low_trust": True,
                    "min_trust_for_gated": 0.4,
                },
                "distortion_rules": {"enabled": True, "by_defense": {}},
                "risk_protocol": {
                    "trigger_keywords": ["suicide", "kill myself"],
                    "response_style": "stable",
                    "lock_topics": [],
                },
                "style_profile": {
                    "register": "colloquial",
                    "tempo": "medium",
                    "length": "short",
                },
            },
        }

        response = await client.post("/case", json=case_data)
        assert response.status_code == 200

        case_response = response.json()
        assert "case_id" in case_response
        case_id = case_response["case_id"]

        # Validate case_id is a valid UUID
        try:
            uuid.UUID(case_id)
        except ValueError:
            pytest.fail(f"case_id {case_id} is not a valid UUID")

        # 2. POST /session → 200, session_id UUID
        session_data = {"case_id": case_id}
        response = await client.post("/session", json=session_data)
        assert response.status_code == 200

        session_response = response.json()
        assert "session_id" in session_response
        session_id = session_response["session_id"]

        # Validate session_id is a valid UUID
        try:
            uuid.UUID(session_id)
        except ValueError:
            pytest.fail(f"session_id {session_id} is not a valid UUID")

        # 3. POST /turn с короткой фразой → 200, patient_reply непустой, запись в telemetry_turns существует
        turn_data = {
            "therapist_utterance": "Привет, как дела?",
            "session_state": {
                "affect": "neutral",
                "trust": 0.3,
                "fatigue": 0.1,
                "access_level": 1,
                "risk_status": "none",
                "last_turn_summary": "",
            },
            "case_id": case_id,
            "session_id": session_id,
            "options": {},
        }

        response = await client.post("/turn", json=turn_data)
        assert response.status_code == 200

        turn_response = response.json()
        assert "patient_reply" in turn_response
        assert turn_response["patient_reply"] != ""
        assert len(turn_response["patient_reply"]) > 0

        # Additional validations for turn response structure
        assert "state_updates" in turn_response
        assert "used_fragments" in turn_response
        assert "risk_status" in turn_response
        assert "eval_markers" in turn_response

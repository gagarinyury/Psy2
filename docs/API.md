# RAG Patient API

**Version:** 1.0.0

Therapeutic conversation simulation API

## API Endpoints

| Method | Path | Description | Request Body | Response 200 |
|--------|------|-------------|--------------|--------------|
| POST | `/admin/llm_flags` | Set Llm Flags | `LLMFlagsRequest` | Successful Response (`LLMFlagsResponse`) |
| POST | `/admin/rag_mode` | Set Rag Mode | `RAGModeRequest` | Successful Response (`RAGModeResponse`) |
| GET | `/admin/rate_limit` | Get Rate Limit | None | Successful Response |
| POST | `/admin/rate_limit` | Update Rate Limit | `RateLimitUpdate` | Successful Response |
| POST | `/case` | Create Case | `CaseRequest` | Successful Response (`CaseResponse`) |
| GET | `/health` | Health Check | None | Successful Response |
| GET | `/metrics` | Get Metrics | None | Successful Response |
| GET | `/report/case/{case_id}/trajectories` | Get Case Trajectory Report | None | Successful Response (`CaseTrajectoryResponse`) |
| GET | `/report/session/{session_id}` | Get Session Report | None | Successful Response |
| GET | `/report/session/{session_id}/missed` | Get Session Missed Keys | None | Successful Response |
| POST | `/session` | Create Session | `SessionRequest` | Successful Response (`SessionResponse`) |
| POST | `/session/link` | Create Session Link | `SessionLinkRequest` | Successful Response (`SessionLinkResponse`) |
| GET | `/session/{session_id}/trajectory` | Get Session Trajectory | None | Successful Response (`SessionTrajectoryResponse`) |
| POST | `/turn` | Process Turn | `TurnRequest` | Successful Response (`TurnResponse`) |
| GET | `/ui/console` | Console | None | Successful Response |

## Example Usage

### Health Check
```bash
curl http://localhost:8000/health
```

### Create Case
```bash
curl -X POST http://localhost:8000/case \
  -H 'Content-Type: application/json' \
  -d '{
    "case_truth": {
      "dx_target": ["MDD"],
      "ddx": {"MDD": 0.6},
      "hidden_facts": ["family history"],
      "red_flags": ["suicidal ideation"]
    },
    "policies": {
      "disclosure_rules": {"min_trust_for_gated": 0.4},
      "risk_protocol": {"trigger_keywords": ["suicide"]},
      "style_profile": {"register": "colloquial"}
    }
  }'
```

### Process Turn
```bash
curl -X POST http://localhost:8000/turn \
  -H 'Content-Type: application/json' \
  -H 'X-Session-ID: <session_id>' \
  -d '{
    "therapist_utterance": "How are you sleeping?",
    "session_state": {
      "affect": "neutral",
      "trust": 0.5,
      "fatigue": 0.1,
      "access_level": 1,
      "risk_status": "none",
      "last_turn_summary": ""
    },
    "case_id": "<case_id>",
    "session_id": "<session_id>"
  }'
```

### Configure Rate Limiting
```bash
curl -X POST http://localhost:8000/admin/rate_limit \
  -H 'Content-Type: application/json' \
  -d '{
    "enabled": true,
    "session_per_min": 20,
    "ip_per_min": 120,
    "fail_open": false
  }'
```

---

*Generated from OpenAPI schema on 1.0.0*
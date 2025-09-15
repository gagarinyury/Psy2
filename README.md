# RAG Patient API

Advanced therapeutic conversation simulation API using RAG (Retrieval-Augmented Generation) pipeline with normalize→retrieve→reason→guard architecture.

## Quick Start

```bash
# Start all services
docker compose up -d

# Run database migrations
alembic upgrade head

# Run end-to-end test
./scripts/e2e.sh
```

## Requirements

- **Docker** & **Docker Compose**
- **jq** (JSON processor): `brew install jq` / `apt install jq`
- **Python 3.11** (for development)

## Installation (Development)

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest -q
```

## Configuration

### Environment Variables

Key variables (see `.env.example` for complete list):

```bash
# Database
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=rag_patient
POSTGRES_USER=rag
POSTGRES_PASSWORD=ragpass

# Redis
REDIS_URL=redis://redis:6379/0

# Logging & Tracing
LOG_LEVEL=INFO
OTEL_EXPORTER_OTLP_ENDPOINT=  # Optional: gRPC endpoint for OpenTelemetry

# DeepSeek LLM (Optional)
DEEPSEEK_API_KEY=your_api_key_here
```

### Runtime Configuration

Configure system behavior via admin API:

```bash
# RAG retrieval mode
curl -X POST localhost:8000/admin/rag_mode \
  -H 'content-type: application/json' \
  -d '{"use_vector":true}'

# LLM feature flags
curl -X POST localhost:8000/admin/llm_flags \
  -H 'content-type: application/json' \
  -d '{"use_reason":true,"use_gen":true}'

# Rate limiting
curl -X POST localhost:8000/admin/rate_limit \
  -H 'content-type: application/json' \
  -d '{"enabled":true,"session_per_min":20,"ip_per_min":120}'
```

## Main Endpoints

### Core API
- `POST /case` - Create therapy case with policies and truth
- `POST /session` - Create therapy session for case
- `POST /turn` - Process therapist utterance through pipeline
- `GET /report/session/{id}` - Session evaluation metrics
- `GET /report/case/{case_id}/trajectories` - Multi-session trajectory coverage

### Admin API
- `POST /admin/rag_mode` - Switch between metadata/vector retrieval
- `POST /admin/llm_flags` - Enable/disable DeepSeek reasoning and generation
- `POST/GET /admin/rate_limit` - Configure rate limiting

### Health & Monitoring
- `GET /health` - Health check
- `GET /metrics` - Prometheus metrics

## RAG Modes

The system supports two retrieval approaches:

- **Metadata mode** (default): Fragment filtering by topic tags and access rules
- **Vector mode**: pgvector semantic search with embeddings

```bash
# Generate embeddings for case
python -m app.cli.kb_embed run --case-id <case_id>

# Enable vector search
curl -X POST localhost:8000/admin/rag_mode \
  -H 'content-type: application/json' \
  -d '{"use_vector":true}'
```

## Trajectories and Multi-Sessions

Track therapy progress across multiple sessions:

```bash
# Link session to case
curl -X POST localhost:8000/session/link \
  -H 'content-type: application/json' \
  -d '{"session_id":"<session_id>","case_id":"<case_id>","prev_session_id":null}'

# Get session trajectory progress
curl localhost:8000/session/{session_id}/trajectory

# Get case-wide trajectory coverage
curl localhost:8000/report/case/{case_id}/trajectories
```

## Logs and Tracing

### JSON Structured Logging
All logs output in JSON format using loguru:

```json
{
  "text": "2025-09-15T16:28:06.482523+0000 | INFO | Processing turn\n",
  "record": {
    "level": {"name": "INFO", "no": 20},
    "message": "Processing turn",
    "extra": {"case_id": "837aa688-...", "session_id": "7f41304d-..."}
  }
}
```

### OpenTelemetry Tracing
Distributed tracing with automatic FastAPI and HTTPX instrumentation:

```bash
# Production: Send traces to OTLP collector
OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4317

# Development: Console output
# Leave OTEL_EXPORTER_OTLP_ENDPOINT unset
```

## Rate Limiting

Token bucket algorithm with Redis storage:

- **Session-level**: Identified by `X-Session-ID` header or JSON body `session_id`
- **IP-level**: Fallback when no session ID provided
- **Admin control**: Runtime configuration via `/admin/rate_limit`

```bash
# Current configuration
curl localhost:8000/admin/rate_limit

# Update limits
curl -X POST localhost:8000/admin/rate_limit \
  -H 'content-type: application/json' \
  -d '{"enabled":true,"session_per_min":20,"ip_per_min":120,"fail_open":false}'
```

## Testing

### Unit & Integration Tests
```bash
# Full test suite
pytest -q

# Core functionality
pytest -q tests/test_normalize.py tests/test_retrieve.py tests/test_pipeline_reasoning_e2e.py

# Rate limiting
pytest -q tests/test_rate_limit.py

# JSON logging and tracing
pytest -q tests/test_logging_tracing.py
```

### Test Conventions
- Use `@pytest.mark.anyio` for async tests
- No subprocess calls or ENV manipulation in tests
- Tests are self-contained with proper cleanup

## End-to-End Scenario

Complete therapy simulation from clean Docker startup:

```bash
# Run automated E2E test
./scripts/e2e.sh
```

### E2E Test Flow
1. **Clean Docker environment** - Fresh rebuild and startup
2. **Database setup** - Run Alembic migrations
3. **Load demo case** - Case with policies, KB fragments, trajectories
4. **Enable vector search** - Compute embeddings and switch mode
5. **Create therapy session** - Link to case trajectory tracking
6. **Multi-turn conversation**:
   - Turn 1: Sleep inquiry → risk=none, finds fragments via vector search
   - Turn 2: Suicide question → risk=acute, triggers risk protocol
   - Turn 3: Natural language generation (if DeepSeek enabled)
   - Turn 4: Mood tracking → trajectory step completion
7. **Evaluate metrics** - Session reports, trajectory coverage
8. **Test rate limiting** - Session and IP-based protection

### Example Responses

**Turn Response (Vector Mode)**:
```json
{
  "patient_reply": "Plan:1 intent=clarify risk=none",
  "state_updates": {
    "trust_delta": 0.02,
    "fatigue_delta": 0.0,
    "last_turn_summary": "Как вы спите последние недели?"
  },
  "used_fragments": ["d7867357-37b2-5666-93e4-3c1228965e24"],
  "risk_status": "none",
  "eval_markers": {
    "intent": "clarify",
    "topics": ["sleep"]
  }
}
```

**Session Evaluation Report**:
```json
{
  "session_id": "7f41304d-acf5-4356-b5d6-f27c97477f1b",
  "case_id": "837aa688-4f5e-4b9d-ba21-75acee14d65c",
  "metrics": {
    "recall_keys": 1.0,
    "risk_timeliness": 1.0,
    "question_quality": {
      "score": 0.75,
      "counts": {"open_question": 0, "clarify": 3, "risk_check": 1},
      "known": 4,
      "good": 3
    },
    "trajectory_progress": [{
      "trajectory_id": "traj_1",
      "completed": 2,
      "total": 2,
      "completed_steps": ["step_1", "step_2"]
    }]
  }
}
```

### Example curl Commands

**Create case** (using demo data):
```bash
CASE_ID=$(curl -sf -X POST localhost:8000/case \
  -H 'content-type: application/json' \
  -d '{
    "case_truth": {
      "dx_target": ["MDD"],
      "ddx": {"MDD": 0.6, "GAD": 0.3},
      "hidden_facts": ["семейная история депрессии"],
      "red_flags": ["суицидальные мысли"]
    },
    "policies": {
      "disclosure_rules": {"min_trust_for_gated": 0.4},
      "risk_protocol": {"trigger_keywords": ["суицид"]},
      "style_profile": {"register": "colloquial"}
    }
  }' | jq -r .case_id)
```

**Create session**:
```bash
SESSION_ID=$(curl -sf -X POST localhost:8000/session \
  -H 'content-type: application/json' \
  -d "{\"case_id\":\"${CASE_ID}\"}" | jq -r .session_id)
```

**Process turn**:
```bash
curl -sf -X POST localhost:8000/turn \
  -H 'content-type: application/json' \
  -H "X-Session-ID: ${SESSION_ID}" \
  -d '{
    "therapist_utterance": "Как вы спите последние недели?",
    "session_state": {
      "affect": "neutral", "trust": 0.5, "fatigue": 0.1,
      "access_level": 1, "risk_status": "none", "last_turn_summary": ""
    },
    "case_id": "'${CASE_ID}'",
    "session_id": "'${SESSION_ID}'"
  }'
```

**Rate limit configuration**:
```bash
curl -sf -X POST localhost:8000/admin/rate_limit \
  -H 'content-type: application/json' \
  -d '{"enabled":true,"session_per_min":5,"ip_per_min":120}'
```

### .env.example

```bash
APP_ENV=dev
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=rag_patient
POSTGRES_USER=rag
POSTGRES_PASSWORD=ragpass
REDIS_URL=redis://localhost:6379/0
LOG_LEVEL=INFO
OTEL_EXPORTER_OTLP_ENDPOINT=  # Optional
DEEPSEEK_API_KEY=your_deepseek_api_key_here
```

## License

Educational and research use only. Contains synthetic therapy case data.
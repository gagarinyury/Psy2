# Psychology AI RAG Patient Simulator

Advanced AI system for simulating therapy patients using RAG (Retrieval-Augmented Generation) architecture with normalize→retrieve pipeline.

## Quick Start

```bash
# Start all services (FastAPI, PostgreSQL, Redis, Prometheus, Grafana)
make up

# Run database migrations  
make migrate

# Load demo case with policies and knowledge base
python -m app.cli.case_loader load app/examples/demo_case.json

# Run tests
make test

# Health check
curl -s localhost:8000/health
```

## Core Architecture

### Pipeline Components

1. **Normalize Node** - Analyzes therapist utterances:
   - Intent classification: `open_question`, `clarify`, `risk_check`, `rapport`
   - Topic extraction: `sleep`, `mood`, `alcohol`, `work`, `family`
   - Risk detection: Suicide ideation keywords
   - Summary generation

2. **Retrieve Node** - Knowledge base access control:
   - Public fragments: Always accessible
   - Gated fragments: Trust-level dependent
   - Hidden fragments: Never accessible
   - Topic filtering and noise injection

3. **Pipeline Orchestrator** - Coordinates workflow:
   - Session validation and turn numbering
   - Telemetry recording with comprehensive metrics
   - Risk status determination and state updates

### Policies System

Strict Pydantic validation for case behavior:

```python
# Disclosure control
"disclosure_rules": {
  "full_on_valid_question": true,
  "partial_if_low_trust": true, 
  "min_trust_for_gated": 0.4
}

# Risk protocol
"risk_protocol": {
  "trigger_keywords": ["суицид", "убить себя", "не хочу жить"],
  "response_style": "stable"
}
```

## API Usage

### Complete Therapy Session Flow

```bash
# 1. Create case with policies and case truth
curl -X POST localhost:8000/case -H "Content-Type: application/json" -d '{
  "case_truth": {
    "dx_target": ["MDD"],
    "ddx": {"MDD": 0.6, "GAD": 0.3},
    "hidden_facts": ["family history"],
    "red_flags": ["suicidal ideation"],
    "trajectories": ["treatment response"]
  },
  "policies": { /* full policy configuration */ }
}'

# 2. Create session
curl -X POST localhost:8000/session -H "Content-Type: application/json" -d '{
  "case_id": "<case_id>"
}'

# 3. Process therapy turns
curl -X POST localhost:8000/turn -H "Content-Type: application/json" -d '{
  "therapist_utterance": "Как вы спите последние недели?",
  "session_state": {
    "affect": "neutral", "trust": 0.5, "fatigue": 0.1,
    "access_level": 1, "risk_status": "none", "last_turn_summary": ""
  },
  "case_id": "<case_id>", 
  "session_id": "<session_id>"
}'
```

### Response Format

```json
{
  "patient_reply": "Echo: clarify, candidates=2",
  "state_updates": {"last_turn_summary": "Therapist asked about sleep"},
  "used_fragments": ["fragment_id_1", "fragment_id_2"],
  "risk_status": "none",  // or "acute" for risk triggers
  "eval_markers": {"intent": "clarify"}
}
```

## CLI Tools

### Case Management
```bash
# Load complete case with KB fragments and policies
python -m app.cli.case_loader load app/examples/demo_case.json

# Validate policies structure
python -c "from app.core.policies import Policies; print('Valid')"
```

### Node Testing
```python
from app.orchestrator.nodes.normalize import normalize
from app.orchestrator.nodes.retrieve import retrieve

# Test intent classification
result = normalize("Бывают ли мысли о суициде?", {})
# Returns: {'intent': 'risk_check', 'risk_flags': ['suicide_ideation'], ...}

# Test knowledge retrieval with access control
fragments = await retrieve(db, case_id, "clarify", ["sleep"], {"trust": 0.5})
```

## Database Schema

### Core Tables
- `cases` - Case definitions with truth and policies (JSONB)
- `kb_fragments` - Knowledge base with vector embeddings and metadata
- `sessions` - Therapy sessions with state tracking
- `telemetry_turns` - Comprehensive turn analytics

### Policies Validation
- Trust levels: `[0.0, 1.0]` range validation
- Probability fields: `[0.0, 1.0]` range validation  
- Required policy components: disclosure_rules, distortion_rules, risk_protocol, style_profile

## Services

- **FastAPI App**: http://localhost:8000 (API + /metrics)
- **PostgreSQL**: localhost:5432 (with pgvector extension)
- **Redis**: localhost:6379 (session storage)
- **Prometheus**: http://localhost:9090 (metrics collection)
- **Grafana**: http://localhost:3000 (admin/admin - dashboards)

## Testing

```bash
# Full test suite
pytest -q

# Core functionality tests
pytest -q tests/test_health.py tests/test_normalize.py tests/test_retrieve.py

# Integration tests  
pytest -q tests/test_case_session_turn.py tests/test_pipeline_turn.py

# CLI and policies tests
pytest -q tests/test_cli_case_loader.py tests/test_policies.py
```

## DeepSeek LLM Integration

The system supports optional DeepSeek API integration for advanced reasoning and natural language generation.

### Configuration

Set your DeepSeek API key in `.env`:
```bash
DEEPSEEK_API_KEY=your_api_key_here
# Optional: customize models and timeouts
DEEPSEEK_REASONING_MODEL=deepseek-reasoner-3.1
DEEPSEEK_BASE_MODEL=deepseek-chat-3.1
DEEPSEEK_TIMEOUT_S=6.0
```

### Runtime Flags

Enable/disable LLM features without restart:

```bash
# Enable reasoning only (keeps Plan: format response)
curl -s -X POST :8000/admin/llm_flags -H 'content-type: application/json' \
  -d '{"use_reason":true,"use_gen":false}'

# Enable both reasoning and natural generation
curl -s -X POST :8000/admin/llm_flags -H 'content-type: application/json' \
  -d '{"use_reason":true,"use_gen":true}'

# Check current flags
curl -s -X POST :8000/admin/llm_flags -H 'content-type: application/json' -d '{}'
```

### Manual Testing

1. **Start the server** and set flags:
   ```bash
   make up
   curl -s -X POST :8000/admin/llm_flags -H 'content-type: application/json' \
     -d '{"use_reason":true,"use_gen":false}'
   ```

2. **Create case and session**:
   ```bash
   python -m app.cli.case_loader load app/examples/demo_case.json
   # Note the case_id from output
   
   curl -s -X POST :8000/session -H 'content-type: application/json' \
     -d '{"case_id":"YOUR_CASE_ID"}'
   # Note the session_id from output
   ```

3. **Test reasoning with LLM**:
   ```bash
   curl -s -X POST :8000/turn -H 'content-type: application/json' -d '{
     "therapist_utterance": "How are you feeling today?",
     "session_id": "YOUR_SESSION_ID", 
     "case_id": "YOUR_CASE_ID",
     "session_state": {
       "trust": 0.5, "fatigue": 0.2, "affect": "neutral", 
       "risk_status": "none", "access_level": 1, "last_turn_summary": ""
     }
   }'
   ```

4. **Expected behavior**:
   - With `use_reason=true`: Uses DeepSeek reasoning, trust_delta from LLM
   - With `use_gen=false`: Response still in `Plan:X intent=Y` format
   - With `use_gen=true`: Natural patient language response
   - On API errors: Automatic fallback to stub nodes with logging

### Fail-Safe Design

- **No API key**: System works with stub nodes (default behavior)
- **API failures**: Automatic fallback with error logging
- **Invalid responses**: JSON validation with fallback
- **Network issues**: Retry logic with exponential backoff

## Development Status

✅ **Completed Components:**
- FastAPI REST API with full CRUD operations
- Pydantic models with strict validation
- PostgreSQL integration with migrations
- Normalize node with intent/topic/risk detection
- Retrieve node with access control and noise
- Policy system with comprehensive validation
- CLI case loader with upsert functionality
- Pipeline orchestrator with telemetry
- Docker Compose development environment
- DeepSeek LLM integration with runtime flags
- Comprehensive test suite (50+ tests)

🔄 **Current Implementation:**
- Normalize→Retrieve pipeline fully operational
- Risk detection and access control working
- Telemetry collection and metrics reporting
- Policy validation and enforcement
- LLM reasoning and generation behind feature flags

## Production-Ready E2E Testing

### Quick E2E Test

Full end-to-end regression test from clean Docker startup to complete therapy session:

```bash
# One-command full system test
chmod +x scripts/e2e.sh
./scripts/e2e.sh
```

### What the E2E Test Does

1. **🧹 Clean Environment**: Stops and rebuilds all Docker services
2. **🏥 Health Checks**: Verifies API availability
3. **🗄️ Database Setup**: Runs Alembic migrations
4. **📄 Demo Case**: Loads demo case with proper trajectories
5. **🔍 Vector Mode**: Enables vector search for KB retrieval
6. **🎭 Session Flow**: Creates therapy session
7. **💬 Multi-Turn Dialog**: Tests 3 conversation turns:
   - Turn 1: Sleep inquiry (risk=none)
   - Turn 2: Suicide risk question (risk=acute)
   - Turn 3: Natural language with LLM generation
8. **📊 Reports**: Validates trajectory progress and session metrics
9. **🛡️ Rate Limiting**: Tests API protection mechanisms

### Expected Results

```bash
✅ Docker services started
✅ Database migrations applied
✅ Demo case loaded (ID: uuid)
✅ Vector search enabled
✅ Therapy session created (ID: uuid)
✅ 3 conversation turns completed
✅ Risk assessment working (none → acute)
✅ Natural language generation enabled
✅ Trajectory tracking functional
✅ Reports generated
✅ Rate limiting tested
```

### Manual Production Testing

If you prefer step-by-step manual testing:

```bash
# 1. Clean start
docker compose down -v
docker build -t rag-patient:local .
docker compose up -d

# 2. Health check
curl -sf "http://localhost:8000/health" | jq .

# 3. Run migrations
docker compose exec -T app alembic upgrade head

# 4. Load demo case (extract proper format)
CASE_REQUEST=$(jq '.case | {case_truth: .case_truth, policies: .policies}' app/examples/demo_case.json)
CASE_ID=$(echo "$CASE_REQUEST" | curl -sf -X POST "http://localhost:8000/case" -H "content-type: application/json" -d @- | jq -r .case_id)

# 5. Enable vector search
curl -sf -X POST "http://localhost:8000/admin/rag_mode" -H "content-type: application/json" -d '{"use_vector":true}' | jq .

# 6. Create session
SESSION_ID=$(curl -sf -X POST "http://localhost:8000/session" -H "content-type: application/json" -d "{\"case_id\":\"${CASE_ID}\"}" | jq -r .session_id)

# 7. Test conversation turns
curl -sf -X POST "http://localhost:8000/turn" \
  -H "content-type: application/json" \
  -H "X-Session-ID: ${SESSION_ID}" \
  -d '{
    "therapist_utterance": "Как вы спите последние недели?",
    "session_state": {"affect": "neutral", "trust": 0.5, "fatigue": 0.1, "access_level": 1, "risk_status": "none", "last_turn_summary": ""},
    "case_id": "'${CASE_ID}'", "session_id": "'${SESSION_ID}'", "options": {}
  }' | jq .

# 8. Check reports
curl -sf "http://localhost:8000/report/session/${SESSION_ID}" | jq .
curl -sf "http://localhost:8000/report/case/${CASE_ID}/trajectories" | jq .
```

### Service URLs After Startup

- **API**: http://localhost:8000 (OpenAPI docs at /docs)
- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **PostgreSQL**: localhost:5432 (rag/ragpass@rag_patient)
- **Redis**: localhost:6379

### Cleanup

```bash
# Stop and remove all containers with volumes
docker compose down -v
```

### Requirements

- **Docker** & **Docker Compose**
- **jq** (JSON processor): `brew install jq` / `apt install jq`
- **curl** (HTTP client)
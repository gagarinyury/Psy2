# Psychology AI RAG Patient Simulator

## Quick Start

```bash
# Start all services
make up

# Run database migrations  
make migrate

# Run tests
make test

# Manual testing
curl -s localhost:8000/health
```

## CLI Tools

### Case Loader
Load cases and knowledge base fragments from JSON files:

```bash
# Load demo case into database
poetry run python -m app.cli.case_loader load app/examples/demo_case.json

# Alternative syntax
python -m app.cli.case_loader load app/examples/demo_case.json
```

### Normalize Node
Process therapist utterances to extract intent, topics, and risk flags:

```python
from app.orchestrator.nodes.normalize import normalize

result = normalize("Как вы спите последние недели?", {})
# Returns: {'intent': 'clarify', 'topics': ['sleep'], 'risk_flags': [], 'last_turn_summary': '...'}
```

## API Endpoints

- `GET /health` - Health check
- `POST /case` - Create case with CaseTruth
- `POST /session` - Create session for case
- `POST /turn` - Process therapist turn

## Services

- FastAPI app: http://localhost:8000
- Prometheus: http://localhost:9090  
- Grafana: http://localhost:3000 (admin/admin)
- PostgreSQL: localhost:5432
- Redis: localhost:6379

## Development

```bash
# Run all tests
pytest -q

# Run specific test modules
pytest -q tests/test_health.py
pytest -q tests/test_case_session_turn.py  
pytest -q tests/test_cli_case_loader.py
pytest -q tests/test_normalize.py
```
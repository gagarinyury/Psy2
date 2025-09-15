# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2025-09-15

### Added
- Full RAG pipeline normalize→retrieve→reason→guard architecture
- Vector search (pgvector) with embeddings CLI tool (`python -m app.cli.kb_embed`)
- DeepSeek LLM integration with reasoning and generation support
- Runtime configurable feature flags via `/admin/llm_flags`
- JSON reasoning parsing with validation and auto-repair
- Comprehensive evaluation metrics:
  - Recall-Keys: Measures coverage of key knowledge fragments
  - Risk-Timeliness: Tracks suicide risk detection timing
  - Question-Quality: Analyzes intent distribution and conversation quality
- Trajectory system for multi-session therapy progress tracking
- Session linking and cross-session trajectory coverage reports
- Rate limiting via Redis token bucket algorithm
  - Session-level limits (identified by X-Session-ID header or JSON body)
  - IP-level fallback limits
  - Admin API for runtime configuration (`/admin/rate_limit`)
- JSON structured logging with loguru
- OpenTelemetry distributed tracing with OTLP gRPC export
- Automatic FastAPI and HTTPX instrumentation
- Manual span creation for pipeline and LLM operations
- End-to-end testing script with clean Docker startup (`scripts/e2e.sh`)
- Multi-turn conversation simulation with risk assessment
- Docker Compose development environment with PostgreSQL, Redis, Prometheus, Grafana
- Pydantic validation for case policies and session states
- CLI case loader with upsert functionality
- Knowledge base fragment management with access control
- Session state tracking (trust, fatigue, affect, risk status)

### Fixed
- Async event loop stability in tests using anyio with single client pattern
- RAG retrieval modes switchable at runtime via `/admin/rag_mode`
- Redis Lua script compatibility (HMSET→HSET migration)
- Vector embedding processing with proper batch handling
- Session trajectory step completion logic
- Rate limiter bucket cleanup and proper token consumption

### Security
- Rate limiter fail-closed by default (configurable via `fail_open` setting)
- Input validation for all API endpoints
- Secure environment variable handling for sensitive data
- Proper session isolation and case access control

[Unreleased]: https://github.com/yourorg/rag-patient/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/yourorg/rag-patient/releases/tag/v0.1.0
# Admin API Documentation

## Rate Limiting Management

The admin API provides runtime configuration of rate limiting parameters.

### Endpoints

#### GET /admin/rate_limit

Get current rate limiting configuration.

**Response:**
```json
{
  "enabled": true,
  "session_per_min": 20,
  "ip_per_min": 120,
  "fail_open": false
}
```

#### POST /admin/rate_limit

Update rate limiting configuration at runtime.

**Request:**
```json
{
  "session_per_min": 5,        // Optional: Session rate limit (0-10000)
  "ip_per_min": 1000,          // Optional: IP rate limit (0-100000)
  "enabled": true,             // Optional: Enable/disable rate limiting
  "fail_open": false           // Optional: Fail open on Redis errors
}
```

**Response:**
```json
{
  "enabled": true,
  "session_per_min": 5,
  "ip_per_min": 1000,
  "fail_open": false
}
```

### Usage Examples

#### Tighten session limits for testing:
```bash
curl -X POST localhost:8000/admin/rate_limit \
  -H "content-type: application/json" \
  -d '{"session_per_min":5,"ip_per_min":1000}'
```

#### Disable rate limiting temporarily:
```bash
curl -X POST localhost:8000/admin/rate_limit \
  -H "content-type: application/json" \
  -d '{"enabled":false}'
```

#### Emergency: High IP limits:
```bash
curl -X POST localhost:8000/admin/rate_limit \
  -H "content-type: application/json" \
  -d '{"ip_per_min":10000,"fail_open":true}'
```

### Rate Limiting Behavior

- **Scope**: Only applies to `POST /turn` endpoint
- **Session-based**: Uses `X-Session-ID` header or `session_id` in request body
- **IP-based**: Fallback when no session ID provided
- **Algorithm**: Token bucket with burst capacity
- **Redis keys**:
  - Session: `rl:session:{session_id}`
  - IP: `rl:ip:{ip_address}`

### Validation

- `session_per_min`: 0-10000 requests per minute
- `ip_per_min`: 0-100000 requests per minute
- Settings are validated on update
- Changes take effect immediately

### Monitoring

Rate limiting metrics are exposed via:
- Prometheus `/metrics` endpoint
- Application logs with structured JSON
- OpenTelemetry traces for debugging
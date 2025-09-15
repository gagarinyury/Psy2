#!/bin/bash

# Test IP rate limiting with requests that fail validation but still hit rate limiter
# This tests that rate limiting happens BEFORE request validation

API="http://localhost:8000"

echo "=== Real IP Rate Limit Test ==="
echo "Testing IP rate limiting with requests without session_id (expect validation errors but rate limiting first)"
echo "Setting RATE_LIMIT_IP_PER_MIN=5 (5 requests per minute)"
echo

# Payload without session_id - will cause 422 validation error, but rate limiting should happen first
PAYLOAD=$(cat <<EOF
{
  "therapist_utterance": "How are you feeling today?",
  "session_state": {
    "affect": "neutral",
    "trust": 0.5,
    "fatigue": 0.0,
    "access_level": 1,
    "risk_status": "low",
    "last_turn_summary": "Initial greeting"
  },
  "case_id": "6c8734a8-9383-4d3b-b8dc-7802b0642be3"
}
EOF
)

success_count=0          # 422 validation errors (but allowed through rate limiter)
rate_limited_count=0     # 429 rate limited
validation_error_count=0 # 422 validation errors
other_count=0

echo "Sending 15 requests without session_id - expecting rate limits to trigger before validation"
for i in $(seq 1 15); do
  echo -n "Request $i: "

  response=$(curl -s -w "%{http_code}" -o /tmp/response.json \
    -X POST "$API/turn" \
    -H "Content-Type: application/json" \
    -d "$PAYLOAD")

  status_code="$response"

  case "$status_code" in
    200)
      echo "✓ SUCCESS (200) - unexpected!"
      success_count=$((success_count + 1))
      ;;
    422)
      echo "? VALIDATION ERROR (422)"
      validation_error_count=$((validation_error_count + 1))
      ;;
    429)
      echo "✗ RATE LIMITED (429)"
      rate_limited_count=$((rate_limited_count + 1))
      # Show rate limit details
      scope=$(jq -r '.scope // "unknown"' /tmp/response.json 2>/dev/null)
      echo "    Scope: $scope"
      ;;
    *)
      echo "? OTHER ($status_code)"
      other_count=$((other_count + 1))
      cat /tmp/response.json
      ;;
  esac

  # No delay to test burst behavior
done

echo
echo "=== Results ==="
echo "Success (200):          $success_count"
echo "Validation Error (422): $validation_error_count"
echo "Rate Limited (429):     $rate_limited_count"
echo "Other:                  $other_count"
echo

# Expected behavior:
# - First 5 requests: 422 validation errors (passed rate limiter, failed validation)
# - Next 10 requests: 429 rate limited (blocked by rate limiter before validation)
if [ $validation_error_count -eq 5 ] && [ $rate_limited_count -eq 10 ]; then
  echo "✓ PERFECT: Rate limiting working correctly!"
  echo "  - First 5 requests passed rate limiter but failed validation (422)"
  echo "  - Next 10 requests blocked by rate limiter (429)"
elif [ $rate_limited_count -gt 0 ]; then
  echo "✓ GOOD: Rate limiting is working (found $rate_limited_count rate limited responses)"
  echo "  Note: $validation_error_count validation errors before rate limiting kicked in"
else
  echo "✗ ISSUE: No rate limiting detected"
  echo "  All $validation_error_count requests reached validation stage"
fi
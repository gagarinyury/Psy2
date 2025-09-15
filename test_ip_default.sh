#!/bin/bash

# Test IP rate limiting with default settings (ip_per_min=120)
# Send 125 requests to exceed the limit

API="http://localhost:8000"

echo "=== IP Rate Limit Test (Default Settings) ==="
echo "Testing with default ip_per_min=120, sending 125 requests"
echo "Expected: 120 success, 5 rate limited"
echo

# Payload without session_id to trigger IP limit
PAYLOAD=$(cat <<EOF
{
  "therapist_utterance": "How are you feeling?",
  "session_state": {
    "affect": "neutral",
    "trust": 0.5,
    "fatigue": 0.0,
    "access_level": 1,
    "risk_status": "low",
    "last_turn_summary": "test"
  },
  "case_id": "6c8734a8-9383-4d3b-b8dc-7802b0642be3"
}
EOF
)

success_count=0
validation_error_count=0
rate_limited_count=0
other_count=0

echo "Sending 125 requests without session_id..."
for i in $(seq 1 125); do
  if [ $((i % 25)) -eq 0 ]; then
    echo "Progress: $i/125"
  fi

  response=$(curl -s -w "%{http_code}" -o /tmp/response.json \
    -X POST "$API/turn" \
    -H "Content-Type: application/json" \
    -d "$PAYLOAD")

  status_code="$response"

  case "$status_code" in
    200)
      success_count=$((success_count + 1))
      ;;
    422)
      validation_error_count=$((validation_error_count + 1))
      ;;
    429)
      rate_limited_count=$((rate_limited_count + 1))
      if [ $rate_limited_count -eq 1 ]; then
        echo "First rate limit at request $i"
        scope=$(jq -r '.scope // "unknown"' /tmp/response.json 2>/dev/null)
        echo "Scope: $scope"
      fi
      ;;
    *)
      other_count=$((other_count + 1))
      ;;
  esac
done

echo
echo "=== Results ==="
echo "Success (200):          $success_count"
echo "Validation Error (422): $validation_error_count"
echo "Rate Limited (429):     $rate_limited_count"
echo "Other:                  $other_count"
echo

if [ $rate_limited_count -gt 0 ]; then
  echo "✓ GOOD: Rate limiting is working!"
  echo "  Found $rate_limited_count rate limited responses"
  echo "  Validation errors before rate limiting: $validation_error_count"
else
  echo "✗ ISSUE: No rate limiting detected"
  echo "  All $validation_error_count requests reached validation"
fi
#!/bin/bash

# Test IP rate limiting: 15 requests without session_id, expecting ~8-10 × 429 responses
# ip_per_min=5 means 5 requests per minute should be allowed

API="http://localhost:8000"

echo "=== IP Rate Limit Test ==="
echo "Sending 15 requests with session_id in body but without X-Session-ID header (to test IP limits)"
echo "Expected: 5 success (200), ~10 rate limited (429)"
echo

# Payload with session_id but no X-Session-ID header (to test IP limits)
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
  "case_id": "6c8734a8-9383-4d3b-b8dc-7802b0642be3",
  "session_id": "00000000-0000-0000-0000-000000000002"
}
EOF
)

success_count=0
rate_limited_count=0
other_count=0

for i in $(seq 1 15); do
  echo -n "Request $i: "

  response=$(curl -s -w "%{http_code}" -o /tmp/response.json \
    -X POST "$API/turn" \
    -H "Content-Type: application/json" \
    -d "$PAYLOAD")

  status_code="$response"

  case "$status_code" in
    200)
      echo "✓ SUCCESS (200)"
      success_count=$((success_count + 1))
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

  # Small delay to avoid overwhelming
  sleep 0.1
done

echo
echo "=== Results ==="
echo "Success (200):      $success_count"
echo "Rate Limited (429): $rate_limited_count"
echo "Other:              $other_count"
echo
echo "Expected: ~5 success, ~10 rate limited"

if [ $success_count -eq 5 ] && [ $rate_limited_count -eq 10 ]; then
  echo "✓ PERFECT: Exactly 5 success + 10 rate limited"
elif [ $success_count -le 6 ] && [ $rate_limited_count -ge 8 ]; then
  echo "✓ GOOD: Within expected range"
else
  echo "✗ UNEXPECTED: Results outside expected range"
fi
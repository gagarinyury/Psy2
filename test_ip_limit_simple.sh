#!/bin/bash

# Simple IP rate limit test using an endpoint that works
# Test with health or turn using a mock service that doesn't require valid data

API="http://localhost:8000"

echo "=== Simple IP Rate Limit Test ==="
echo "Testing rate limiting by sending requests rapidly from same IP"
echo "Setting RATE_LIMIT_IP_PER_MIN=5 (5 requests per minute)"
echo

# Use health endpoint for simplicity
success_count=0
rate_limited_count=0
other_count=0

# Set very restrictive IP limit
export RATE_LIMIT_IP_PER_MIN=5
export RATE_LIMIT_SESSION_PER_MIN=100
export RATE_LIMIT_ENABLED=true

echo "Testing with health endpoint (15 GET requests):"
for i in $(seq 1 15); do
  echo -n "Request $i: "

  response=$(curl -s -w "%{http_code}" -o /tmp/response.json \
    -X GET "$API/health")

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

  # No delay to test burst behavior
done

echo
echo "=== Results ==="
echo "Success (200):      $success_count"
echo "Rate Limited (429): $rate_limited_count"
echo "Other:              $other_count"
echo

# Check if rate limiting worked
if [ $rate_limited_count -gt 0 ]; then
  echo "✓ GOOD: Rate limiting is working (found $rate_limited_count rate limited responses)"
else
  echo "✗ ISSUE: No rate limiting detected - all requests succeeded"
fi
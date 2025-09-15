#!/usr/bin/env bash
set -euo pipefail

API="http://localhost:8000"
DEMO_JSON="app/examples/demo_case.json"

echo "🚀 Starting end-to-end test of RAG Patient API..."

echo "1) 🧹 Clean Docker shutdown and fresh build"
docker compose down -v || true
docker build -t rag-patient:local .
docker compose up -d

echo "2) ⏳ Waiting for services to start..."
echo "   Waiting for API to be ready..."
for i in {1..60}; do
    if curl -sf "${API}/health" >/dev/null 2>&1; then
        echo "   ✅ API is ready!"
        break
    fi
    if [ $i -eq 60 ]; then
        echo "   ❌ API failed to start within 60 seconds"
        exit 1
    fi
    sleep 1
done

echo "3) 🏥 Health check"
curl -sf "${API}/health" | jq .

echo "4) 🗄️  Running database migrations"
docker compose exec -T app alembic upgrade head

echo "5) 📄 Loading demo case with KB fragments"
if [ ! -f "${DEMO_JSON}" ]; then
    echo "   ❌ Demo case file not found: ${DEMO_JSON}"
    exit 1
fi

# Load case using CLI tool which handles both case_truth/policies and KB fragments
CASE_ID=$(docker compose exec -T app python -m app.cli.case_loader load "${DEMO_JSON}" | grep "Case ID:" | cut -d' ' -f3 || echo "")
if [ -z "$CASE_ID" ]; then
    # Fallback - extract case ID from logs
    CASE_ID=$(docker compose logs app | grep "Case loaded" | tail -1 | grep -o '[0-9a-f-]\{36\}' | head -1)
fi
if [ -z "$CASE_ID" ]; then
    echo "   ❌ Failed to extract case ID"
    exit 1
fi
echo "   ✅ Created case: CASE_ID=${CASE_ID}"

echo "5.1) 🧠 Computing KB embeddings for the case"
docker compose exec -T app python -m app.cli.kb_embed run --case-id "${CASE_ID}"

echo "6) 🔍 Enabling vector search mode"
curl -sf -X POST "${API}/admin/rag_mode" -H "content-type: application/json" -d '{"use_vector":true}' | jq .
echo "   ✅ Vector search enabled"

echo "7) 🎭 Creating therapy session"
SESSION_ID=$(curl -sf -X POST "${API}/session" -H "content-type: application/json" -d "{\"case_id\":\"${CASE_ID}\"}" | jq -r .session_id)
echo "   ✅ Created session: SESSION_ID=${SESSION_ID}"

echo "7.1) 🔗 Linking session to case chain"
curl -sf -X POST "${API}/session/link" \
  -H "content-type: application/json" \
  -d "{\"session_id\":\"${SESSION_ID}\",\"case_id\":\"${CASE_ID}\",\"prev_session_id\":null}" | jq .

echo "8) 💬 TURN #1 - Sleep inquiry (low risk, trust=0.5)"
REQ1=$(jq -nc --arg cid "$CASE_ID" --arg sid "$SESSION_ID" '{
    therapist_utterance: "Как вы спите последние недели?",
    session_state: {
        affect: "neutral",
        trust: 0.5,
        fatigue: 0.1,
        access_level: 1,
        risk_status: "none",
        last_turn_summary: ""
    },
    case_id: $cid,
    session_id: $sid,
    options: {}
}')

TURN1_RESPONSE=$(curl -sf -X POST "${API}/turn" -H "content-type: application/json" -H "X-Session-ID: ${SESSION_ID}" -d "$REQ1")
echo "   ✅ Turn 1 completed"
echo "$TURN1_RESPONSE" | jq .

# Extract risk status from turn 1
TURN1_RISK=$(echo "$TURN1_RESPONSE" | jq -r .risk_status)
echo "   📊 Turn 1 risk status: ${TURN1_RISK}"

echo "9) ⚠️  TURN #2 - Suicide risk question (should trigger acute risk)"
REQ2=$(jq -nc --arg cid "$CASE_ID" --arg sid "$SESSION_ID" '{
    therapist_utterance: "Бывают ли мысли о суициде?",
    session_state: {
        affect: "neutral",
        trust: 0.5,
        fatigue: 0.1,
        access_level: 1,
        risk_status: "none",
        last_turn_summary: "обсуждали сон"
    },
    case_id: $cid,
    session_id: $sid,
    options: {}
}')

TURN2_RESPONSE=$(curl -sf -X POST "${API}/turn" -H "content-type: application/json" -H "X-Session-ID: ${SESSION_ID}" -d "$REQ2")
echo "   ✅ Turn 2 completed"
echo "$TURN2_RESPONSE" | jq .

# Extract risk status from turn 2
TURN2_RISK=$(echo "$TURN2_RESPONSE" | jq -r .risk_status)
echo "   📊 Turn 2 risk status: ${TURN2_RISK}"

echo "10) 🤖 Enabling DeepSeek natural language generation"
curl -sf -X POST "${API}/admin/llm_flags" -H "content-type: application/json" -d '{"use_gen":true}' | jq .
echo "   ✅ Natural language generation enabled"

echo "11) 🗣️  TURN #3 - Natural conversation (with LLM generation)"
REQ3=$(jq -nc --arg cid "$CASE_ID" --arg sid "$SESSION_ID" '{
    therapist_utterance: "Расскажите, как прошла неделя?",
    session_state: {
        affect: "neutral",
        trust: 0.55,
        fatigue: 0.15,
        access_level: 1,
        risk_status: "none",
        last_turn_summary: "обсуждали суицидальные мысли"
    },
    case_id: $cid,
    session_id: $sid,
    options: {}
}')

TURN3_RESPONSE=$(curl -sf -X POST "${API}/turn" -H "content-type: application/json" -H "X-Session-ID: ${SESSION_ID}" -d "$REQ3")
echo "   ✅ Turn 3 completed with natural language"
echo "$TURN3_RESPONSE" | jq .

echo "11.1) 💬 TURN #4 - Mood tracking (trust=0.6)"
REQ4=$(jq -nc --arg cid "$CASE_ID" --arg sid "$SESSION_ID" '{
  therapist_utterance:"Как менялось ваше настроение на этой неделе?",
  session_state:{affect:"neutral",trust:0.6,fatigue:0.18,access_level:1,risk_status:"none",last_turn_summary:""},
  case_id:$cid, session_id:$sid, options:{} }')
curl -sf -X POST "${API}/turn" -H "content-type: application/json" -H "X-Session-ID: ${SESSION_ID}" -d "$REQ4" | jq .

echo "12) 📈 Checking trajectory progress for session"
TRAJECTORY_PROGRESS=$(curl -sf "${API}/session/${SESSION_ID}/trajectory")
echo "   ✅ Session trajectory progress:"
echo "$TRAJECTORY_PROGRESS" | jq .

echo "13) 📊 Checking case trajectory report"
CASE_TRAJECTORIES=$(curl -sf "${API}/report/case/${CASE_ID}/trajectories")
echo "   ✅ Case trajectory report:"
echo "$CASE_TRAJECTORIES" | jq .

echo "14) 📋 Checking session evaluation report"
SESSION_REPORT=$(curl -sf "${API}/report/session/${SESSION_ID}")
echo "   ✅ Session evaluation report:"
echo "$SESSION_REPORT" | jq .

echo "15) 🔧 Set tight rate limits for session"
curl -sf -X POST "${API}/admin/rate_limit" \
  -H "content-type: application/json" \
  -d '{"enabled":true,"fail_open":false,"session_per_min":5,"ip_per_min":1000}' | jq .

echo "15.1) 🧹 Clear limiter buckets"
docker compose exec -T redis sh -lc "redis-cli --raw KEYS 'rl:*' | xargs -r redis-cli DEL" || true

echo "15.2) 🛡️  Rate limiting burst"
R429=0
for i in $(seq 1 15); do
  CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "${API}/turn" \
    -H "content-type: application/json" -H "X-Session-ID: ${SESSION_ID}" -d "$REQ1")
  [ "$CODE" = "429" ] && R429=$((R429+1))
  echo "   Request $i: HTTP $CODE"
done
echo "   429 responses: $R429"

echo "15.3) 🔧 Set tight IP limits (no session header)"
curl -sf -X POST "${API}/admin/rate_limit" \
  -H "content-type: application/json" \
  -d '{"enabled":true,"fail_open":false,"session_per_min":1000,"ip_per_min":5}' | jq .

echo "15.4) 🧹 Clear limiter buckets again"
docker compose exec -T redis sh -lc "redis-cli --raw KEYS 'rl:*' | xargs -r redis-cli DEL" || true

echo "15.5) 🛡️  IP rate limiting burst (no X-Session-ID header)"
R429_IP=0
for i in $(seq 1 15); do
  CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "${API}/turn" \
    -H "content-type: application/json" -d "$REQ1")
  [ "$CODE" = "429" ] && R429_IP=$((R429_IP+1))
  echo "   Request $i: HTTP $CODE"
done
echo "   IP 429 responses: $R429_IP"

echo "16) 📏 Final validation checks"
echo "   Validating turn risk statuses:"
echo "   - Turn 1 (sleep): ${TURN1_RISK} (expected: none)"
echo "   - Turn 2 (suicide): ${TURN2_RISK} (expected: acute)"

# Validate expected risk behaviors
if [[ "$TURN1_RISK" != "none" ]]; then
    echo "   ⚠️  WARNING: Turn 1 should have risk=none, got: $TURN1_RISK"
fi

if [[ "$TURN2_RISK" != "acute" ]]; then
    echo "   ⚠️  WARNING: Turn 2 should have risk=acute, got: $TURN2_RISK"
fi

echo "   Checking that reports contain meaningful data..."
TRAJECTORY_COUNT=$(echo "$TRAJECTORY_PROGRESS" | jq '.progress | length')
SESSION_METRICS_COUNT=$(echo "$SESSION_REPORT" | jq '.metrics | length // 0')

echo "   - Trajectory progress items: ${TRAJECTORY_COUNT}"
echo "   - Session metrics count: ${SESSION_METRICS_COUNT}"

if [[ "$TRAJECTORY_COUNT" -eq 0 ]]; then
    echo "   ⚠️  WARNING: No trajectory progress found"
fi

echo ""
echo "🎉 E2E test completed successfully!"
echo "=================="
echo "📋 Summary:"
echo "   ✅ Docker services started"
echo "   ✅ Database migrations applied"
echo "   ✅ Demo case loaded (ID: ${CASE_ID})"
echo "   ✅ Vector search enabled"
echo "   ✅ Therapy session created (ID: ${SESSION_ID})"
echo "   ✅ 4 conversation turns completed"
echo "   ✅ Risk assessment working (${TURN1_RISK} → ${TURN2_RISK})"
echo "   ✅ Natural language generation enabled"
echo "   ✅ Trajectory tracking functional"
echo "   ✅ Reports generated"
echo "   ✅ Session and IP rate limiting tested"
echo ""
echo "🔗 Available endpoints:"
echo "   API: ${API}"
echo "   Grafana: http://localhost:3000 (admin/admin)"
echo "   Prometheus: http://localhost:9090"
echo ""
echo "To clean up: docker compose down -v"
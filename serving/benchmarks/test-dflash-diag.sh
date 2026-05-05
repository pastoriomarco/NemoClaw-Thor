#!/bin/bash
# Send test requests to DFlash server and collect diagnostic output.
# Usage: ./test-dflash-diag.sh [container_name]

CONTAINER="${1:-dflash-diag}"
API="http://localhost:8000/v1/chat/completions"

echo "=== DFlash Diagnostic Test ==="
echo "Container: $CONTAINER"
echo ""

# Wait for server ready
echo "Waiting for server..."
for i in $(seq 1 60); do
    if curl -s http://localhost:8000/health | grep -q "ok\|200" 2>/dev/null; then
        echo "Server ready!"
        break
    fi
    if [ $i -eq 60 ]; then
        echo "Timeout waiting for server"
        exit 1
    fi
    sleep 2
done

echo ""
echo "--- Test 1: Short prompt (few context tokens) ---"
curl -s "$API" -H "Content-Type: application/json" -d '{
    "model": "lovedheart/Qwen3.5-9B-FP8",
    "messages": [{"role": "user", "content": "Say hello"}],
    "max_tokens": 30,
    "temperature": 0
}' | python3 -c "import json,sys; r=json.load(sys.stdin); print(f'Response: {r[\"choices\"][0][\"message\"][\"content\"][:100]}'); print(f'Tokens: {r[\"usage\"]}')" 2>/dev/null || echo "Request failed"

echo ""
echo "--- Test 2: Medium prompt ---"
curl -s "$API" -H "Content-Type: application/json" -d '{
    "model": "lovedheart/Qwen3.5-9B-FP8",
    "messages": [{"role": "user", "content": "What is the capital of France? Answer in one word."}],
    "max_tokens": 30,
    "temperature": 0
}' | python3 -c "import json,sys; r=json.load(sys.stdin); print(f'Response: {r[\"choices\"][0][\"message\"][\"content\"][:100]}'); print(f'Tokens: {r[\"usage\"]}')" 2>/dev/null || echo "Request failed"

echo ""
echo "--- Test 3: Longer prompt ---"
curl -s "$API" -H "Content-Type: application/json" -d '{
    "model": "lovedheart/Qwen3.5-9B-FP8",
    "messages": [{"role": "user", "content": "Write a haiku about the ocean."}],
    "max_tokens": 50,
    "temperature": 0
}' | python3 -c "import json,sys; r=json.load(sys.stdin); print(f'Response: {r[\"choices\"][0][\"message\"][\"content\"][:200]}'); print(f'Tokens: {r[\"usage\"]}')" 2>/dev/null || echo "Request failed"

echo ""
echo "=== Collecting diagnostic logs ==="
docker logs "$CONTAINER" 2>&1 | grep "^DIAG\|^===" | tail -200

echo ""
echo "=== Spec decode metrics ==="
docker logs "$CONTAINER" 2>&1 | grep "SpecDecoding" | tail -5

echo ""
echo "=== Saved tensor files ==="
docker exec "$CONTAINER" ls -la /tmp/dflash-diag/ 2>/dev/null || echo "No tensor files yet"

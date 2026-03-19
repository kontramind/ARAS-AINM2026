#!/bin/bash
# Start the Tripletex API locally + expose it via ngrok HTTPS tunnel.
# No Docker required — runs uvicorn directly in the venv.
#
# Requirements:
#   - ngrok installed  (run once: scripts/install_ngrok.sh)
#   - NGROK_AUTHTOKEN in .env
#
# Usage:
#   chmod +x scripts/start_ngrok.sh
#   ./scripts/start_ngrok.sh

set -e

cd "$(dirname "$0")/.."

# Load .env
if [ -f .env ]; then
    export $(grep -v '^#' .env | grep -v '^$' | xargs)
fi

# Check ngrok auth token
if [ -z "$NGROK_AUTHTOKEN" ]; then
    echo "ERROR: NGROK_AUTHTOKEN is not set in .env"
    echo "Get a free token at https://dashboard.ngrok.com and add:"
    echo "  NGROK_AUTHTOKEN=<your-token>"
    exit 1
fi

# Check ngrok is installed
if ! command -v ngrok &>/dev/null; then
    echo "ngrok not found. Install it first:"
    echo "  ./scripts/install_ngrok.sh"
    exit 1
fi

# Configure ngrok auth token
ngrok config add-authtoken "$NGROK_AUTHTOKEN" --log=false 2>/dev/null || true

# Kill any leftover processes on port 8000
fuser -k 8000/tcp 2>/dev/null || true

echo "Starting API on port 8000..."
.venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000 &
API_PID=$!

# Wait for API to be ready
echo "Waiting for API to be ready..."
for i in $(seq 1 15); do
    if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
        echo "API is up."
        break
    fi
    sleep 1
done

echo "Starting ngrok tunnel..."
ngrok http 8000 > /tmp/ngrok.log 2>&1 &
NGROK_PID=$!

# Wait for ngrok's local API to be ready
for i in $(seq 1 15); do
    PUBLIC_URL=$(curl -s http://localhost:4040/api/tunnels 2>/dev/null \
        | python3 -c "
import sys, json
try:
    t = json.load(sys.stdin).get('tunnels', [])
    print(next(x['public_url'] for x in t if x['public_url'].startswith('https')))
except: pass
" 2>/dev/null)
    [ -n "$PUBLIC_URL" ] && break
    sleep 1
done

echo ""
echo "============================================"
if [ -n "$PUBLIC_URL" ]; then
    echo "  PUBLIC URL:      ${PUBLIC_URL}"
    echo "  SOLVE ENDPOINT:  ${PUBLIC_URL}/tripletex/solve"
    echo "  HEALTH CHECK:    ${PUBLIC_URL}/health"
else
    echo "  URL not detected. Run manually:"
    echo "  curl -s http://localhost:4040/api/tunnels | python3 -c \"import sys,json; print([t['public_url'] for t in json.load(sys.stdin)['tunnels']])\""
fi
echo "============================================"
echo ""
echo "Submit at: https://app.ainm.no/submit/tripletex"
echo "Press Ctrl+C to stop everything."
echo ""

# Keep running; clean up on exit
trap "kill $API_PID $NGROK_PID 2>/dev/null; echo 'Stopped.'" EXIT INT TERM
wait $API_PID

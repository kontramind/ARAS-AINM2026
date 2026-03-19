#!/bin/bash
# Install ngrok on Ubuntu/Debian (WSL2 compatible).
# Run once, then use scripts/start_ngrok.sh.

set -e

echo "Installing ngrok..."
curl -sSL https://ngrok-agent.s3.amazonaws.com/ngrok.asc \
    | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null
echo "deb https://ngrok-agent.s3.amazonaws.com buster main" \
    | sudo tee /etc/apt/sources.list.d/ngrok.list >/dev/null
sudo apt-get update -qq
sudo apt-get install -y ngrok

echo "ngrok $(ngrok version) installed."
echo ""
echo "Next: make sure NGROK_AUTHTOKEN is set in .env, then run:"
echo "  ./scripts/start_ngrok.sh"

#!/bin/bash
set -e

echo "[INFO] Starting smart setup..."

# === [1] Check & download model ===
MODEL_PATH="traversability_model/traversability_model.plan"
MODEL_URL="https://www.dropbox.com/scl/fi/xxjtu4hzdb5f8qwu27ack/traversability_model.plan?rlkey=8n7udgy6l8vlt3sm3fo57odiy&st=aukug1k4&dl=1"

if [ ! -f "$MODEL_PATH" ]; then
  echo "[INFO] Model file not found. Downloading..."
  wget -O "$MODEL_PATH" "$MODEL_URL"
else
  echo "[INFO] Model file already exists: $MODEL_PATH"
fi
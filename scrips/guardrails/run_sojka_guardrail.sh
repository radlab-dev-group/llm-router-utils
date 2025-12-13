#!/usr/bin/env bash

# Set defaults if they are not already defined in the environment
: "${LLM_ROUTER_SOJKA_GUARD_FLASK_HOST:=0.0.0.0}"
: "${LLM_ROUTER_SOJKA_GUARD_FLASK_PORT:=5001}"
: "${LLM_ROUTER_SOJKA_GUARD_MODEL_PATH:=speakleash/Bielik-Guard-0.1B-v1.0}"
: "${LLM_ROUTER_SOJKA_GUARD_DEVICE:=1}"

# Export them so the Python process can read them
export LLM_ROUTER_SOJKA_GUARD_FLASK_HOST
export LLM_ROUTER_SOJKA_GUARD_FLASK_PORT
export LLM_ROUTER_SOJKA_GUARD_MODEL_PATH
export LLM_ROUTER_SOJKA_GUARD_DEVICE

# Show the configuration that will be used
echo "Starting Sojka Guardrail API with Gunicorn (2 workers):"
echo "  HOST   = $LLM_ROUTER_SOJKA_GUARD_FLASK_HOST"
echo "  PORT   = $LLM_ROUTER_SOJKA_GUARD_FLASK_PORT"
echo "  MODEL  = $LLM_ROUTER_SOJKA_GUARD_MODEL_PATH"
echo "  DEVICE = $LLM_ROUTER_SOJKA_GUARD_DEVICE"
echo

# ---------------------------------------------------------------
# Run Gunicorn
#   -w 2               → 2 worker processes
#   -b host:port       → bind address
#   guardrails.speakleash.sojka_guard_app
# ---------------------------------------------------------------
gunicorn -w 2 -b \
  "${LLM_ROUTER_SOJKA_GUARD_FLASK_HOST}:${LLM_ROUTER_SOJKA_GUARD_FLASK_PORT}" \
  llm_router_services.guardrails.speakleash.sojka_guard_app:app

#!/usr/bin/env bash
set -euo pipefail

REGISTRY_DIR="${HOME}/.litert-lm"
mkdir -p "${REGISTRY_DIR}"

: "${LITERT_MODEL_ID:=gemma4-e2b}"
: "${LITERT_HF_REPO:=litert-community/gemma-4-E2B-it-litert-lm}"
: "${LITERT_MODEL_FILE:=gemma-4-E2B-it.litertlm}"
: "${LITERT_BACKEND:=cpu}"
: "${LITERT_MAX_NUM_TOKENS:=8192}"
export LITERT_MODEL_ID LITERT_BACKEND LITERT_MAX_NUM_TOKENS

# Render config.json so the OpenAI server knows which hardware backend to
# use for this model id (standard OpenAI requests have no such field).
# See: https://ai.google.dev/edge/litert-lm/cli/configuration
envsubst < /app/config.json.template > "${REGISTRY_DIR}/config.json"
echo "== ${REGISTRY_DIR}/config.json =="
cat "${REGISTRY_DIR}/config.json"
echo

# Import the model into the local registry the first time we see it.
if litert-lm list 2>/dev/null | awk 'NR>1 {print $1}' | grep -qx "${LITERT_MODEL_ID}"; then
  echo "== '${LITERT_MODEL_ID}' already in local registry, skipping import =="
else
  echo "== Importing '${LITERT_MODEL_ID}' from ${LITERT_HF_REPO} (this can take a while on first run) =="
  import_args=(import --from-huggingface-repo "${LITERT_HF_REPO}" "${LITERT_MODEL_FILE}" "${LITERT_MODEL_ID}")
  if [[ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
    import_args+=(--huggingface-token "${HUGGING_FACE_HUB_TOKEN}")
  fi
  litert-lm "${import_args[@]}"
fi

echo "== Starting LiteRT-LM OpenAI-compatible server on 0.0.0.0:9379 (backend=${LITERT_BACKEND} for ${LITERT_MODEL_ID}) =="
exec litert-lm serve --host 0.0.0.0 --port 9379 --verbose

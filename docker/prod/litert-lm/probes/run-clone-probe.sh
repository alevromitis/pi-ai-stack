#!/usr/bin/env bash
#
# Runs clone_probe.py on rpi in a throwaway container from the serving image
# (same engine settings, same model registry volume). litert-lm-1 is stopped
# for the run, since two engines don't fit in RAM, and started again on exit
# whatever happens. About 6 minutes.
#
# Usage: ./run-clone-probe.sh SYSTEM_PROMPT_FILE [--switching]
#   SYSTEM_PROMPT_FILE: pi-chat's system prompt as plain text (its
#   promptPrefix in ../../librechat/librechat-config/librechat.yaml).
#   --switching: only the live-conversation check, MTP on and off.
#
set -euo pipefail
cd "$(dirname "$0")"

system_prompt="$(realpath "${1:?usage: $0 SYSTEM_PROMPT_FILE [--switching]}")"
shift
[ -s "$system_prompt" ] || { echo "empty or missing: $system_prompt" >&2; exit 1; }

if docker logs --since 2m litert-lm-1 2>&1 | grep -q 'POST /v1'; then
  echo "litert-lm-1 served a request in the last 2 minutes; try again later" >&2
  exit 1
fi

trap 'docker start litert-lm-1 >/dev/null && echo "litert-lm-1 started again"' EXIT
docker stop litert-lm-1 >/dev/null
echo "litert-lm-1 stopped"

# Same CPU weight as litert-lm-1 (compose cpu_shares), so the timings are
# comparable with the server's.
docker run --rm --memory 7g --memory-swap 7g --cpu-shares 262144 \
  -v litert_litert-registry:/root/.litert-lm \
  -v "$PWD/clone_probe.py:/probe/clone_probe.py:ro" \
  -v "$system_prompt:/probe/system.txt:ro" \
  --entrypoint python3 litert-lm-rpi5:local /probe/clone_probe.py "$@"

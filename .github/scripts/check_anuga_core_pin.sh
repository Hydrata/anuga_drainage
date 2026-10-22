#!/usr/bin/env bash
# TASK-3740 (epic 3585): the anuga_core pin this fork's CI installs must equal the engine
# hydrata.com runs, i.e. ARG ANUGA_CORE_SHA in Hydrata/run_anuga batch/Dockerfile. The two
# drifted once (93368189 here vs 1a496176 in prod, TASK-3772) with nothing to notice.
#
# Usage: check_anuga_core_pin.sh <action.yml> <Dockerfile path or https URL>
# Exit 0 = in sync, 1 = drift, 2 = a pin could not be read.
set -euo pipefail
action="${1:?action.yml path}"
dockerfile="${2:?Dockerfile path or URL}"

ours=$(grep -oE 'default: [0-9a-f]{40}' "$action" | head -1 | cut -d' ' -f2 || true)
if [[ "$dockerfile" == https://* ]]; then
  src=$(curl -fsSL --retry 3 "$dockerfile") || { echo "::error::cannot fetch $dockerfile"; exit 2; }
else
  src=$(cat "$dockerfile")
fi
prod=$(grep -oE 'ARG ANUGA_CORE_SHA=[0-9a-f]{40}' <<<"$src" | head -1 | cut -d= -f2 || true)

if [[ -z "$ours" || -z "$prod" ]]; then
  echo "::error::could not read a 40-char pin (action: '${ours}', run_anuga Dockerfile: '${prod}')"
  exit 2
fi
if [[ "$ours" != "$prod" ]]; then
  echo "::error::anuga_core pin drift: $action defaults $ours, run_anuga batch/Dockerfile pins $prod (bump the action default)"
  exit 1
fi
echo "anuga_core pin in sync: $ours"

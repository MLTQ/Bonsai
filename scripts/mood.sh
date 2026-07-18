#!/bin/zsh
# The agent's mood verb. Usage:
#   mood.sh dread                     # named anchor
#   mood.sh 0.1 0.2 0.3 ... (10)      # raw z vector
#   mood.sh --text "tests are green"  # semantic projection (tier 2)
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ "$1" == "--text" ]]; then
    exec python3 tools/mood_projector.py --text "$2"
elif [[ $# -eq 10 ]]; then
    Z=$(printf '%s,' "$@" | sed 's/,$//')
    echo "{\"z\": [$Z]}" > weights/control.json
    echo "z -> [$Z]"
else
    echo "{\"anchor\": \"$1\"}" > weights/control.json
    echo "anchor -> $1"
fi

#!/usr/bin/env bash
# Step 2 gate check. Uses REAL exit codes: pytest's own status, never a parsed
# summary, because `pytest ... | tail` returns tail's status and once let a red
# suite be committed.
set -euo pipefail
cd "$(dirname "$0")/.."
echo "== gate invariant tests =="
# The GPU host has no pytest (the training image is stdlib only), so fall back to
# the unittest runner; either way the exit code is the test runner's own.
if python3 -c "import pytest" >/dev/null 2>&1; then
  python3 -m pytest env/mec_offloaing_envs/scheduler/tests/test_source_registry_gate.py -q -p no:nameko
else
  echo "(pytest unavailable: stdlib unittest runner)"
  python3 -m unittest discover -s env/mec_offloaing_envs/scheduler/tests \
    -t . -p test_source_registry_gate.py
fi
echo "== source registry validator =="
python3 spec/automotive_mc_v1/validate_source_registry.py | python3 -c '
import json, sys
d = json.load(sys.stdin)
print("gate:", d["gate"], "| missing", d["missing_required_parameters"],
      "| blocked", d["blocked_required_parameters"], "| violations", len(d["details"]),
      "| stale", d["registry_pin_stale"])
for line in d["details"]:
    print("  -", line)
sys.exit(0 if d["gate"] == "PASS" else 1)
'

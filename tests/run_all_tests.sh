#!/usr/bin/env bash
# tests/run_all_tests.sh
#
# One-shot test runner. Runs the standalone tests (no nnU-Net needed)
# from the host, then optionally the smoke test inside the singularity
# container if available.
#
# Usage:
#   bash tests/run_all_tests.sh
#
# Exit code: nonzero if any test fails.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"

# Prefer container python if available (has nnunetv2 installed); otherwise
# fall back to whatever python3 is on PATH (still works for the standalone
# tests since they don't import nnunetv2).
if [[ -x /opt/conda/bin/python3 ]]; then
    PY=/opt/conda/bin/python3
else
    PY=$(command -v python3)
fi

echo "== Running standalone tests with: $PY =="
echo

failed=0

echo "-- tests/test_subtype_refinement.py --"
"$PY" tests/test_subtype_refinement.py || failed=$((failed+1))
echo

echo "-- tests/test_patch_bias.py --"
"$PY" tests/test_patch_bias.py || failed=$((failed+1))
echo

echo "-- tests/test_lstv_dedicated_val.py --"
"$PY" tests/test_lstv_dedicated_val.py || failed=$((failed+1))
echo

# nnU-Net call-site smoke test only runs if nnunetv2 is importable.
if "$PY" -c "import nnunetv2" 2>/dev/null; then
    echo "-- tests/test_nnunet_call_site.py (nnU-Net detected) --"
    "$PY" tests/test_nnunet_call_site.py || failed=$((failed+1))
    echo

    echo "-- tests/integration/test_real_nnunet_integration.py (nnU-Net detected) --"
    "$PY" tests/integration/test_real_nnunet_integration.py || failed=$((failed+1))
else
    echo "-- tests/test_nnunet_call_site.py: SKIPPED (nnunetv2 not in this env) --"
    echo "   To run inside container:"
    echo "     singularity exec --nv \$CONTAINER /opt/conda/bin/python3 tests/test_nnunet_call_site.py"
    echo
    echo "-- tests/integration/test_real_nnunet_integration.py: SKIPPED (nnunetv2 not in this env) --"
    echo "   To run inside container:"
    echo "     singularity exec --nv \$CONTAINER /opt/conda/bin/python3 tests/integration/test_real_nnunet_integration.py"
fi
echo

if [[ "$failed" -gt 0 ]]; then
    echo "FAIL: $failed test file(s) failed"
    exit 1
fi
echo "ALL TESTS PASSED"

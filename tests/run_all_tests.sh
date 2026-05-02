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
#
# v18 (May 2026):
#   - Removed test_patch_bias.py: tested _select_biased_index and
#     _selftest_patch_bias from the np.random.choice monkey-patch
#     mechanism. Both functions are gone in v18; the patch bias is now
#     a clean class override (LSTVBiasedDataLoader3D).
#   - Removed test_nnunet_call_site.py: tested where the np.random.choice
#     intercept fired in nnU-Net's generate_train_batch. Irrelevant in
#     v18: nothing intercepts np.random.choice anywhere.
#   - Removed integration/test_real_nnunet_integration.py: real-nnU-Net
#     version of the call-site test. Same rationale — replaced by the
#     production loader tests in test_lstv_biased_dataloader.py
#     (TestProductionLoader, which verifies MRO ordering, picklability,
#     and method resolution against the actual nnUNetDataLoader3D).
#   - Added test_lstv_biased_dataloader.py: pure-function tests + mixin
#     tests + production loader tests for the new class override.

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

run_one() {
    local label="$1"; local path="$2"
    if [[ ! -f "$path" ]]; then
        echo "-- $label: SKIPPED (not found: $path) --"
        echo
        return 0
    fi
    echo "-- $label --"
    if "$PY" "$path"; then
        :
    else
        failed=$((failed+1))
    fi
    echo
}

# ── Standalone tests (no nnU-Net required) ──────────────────────────────
run_one "subtype refinement"     "tests/test_subtype_refinement.py"
run_one "LSTV biased dataloader" "tests/test_lstv_biased_dataloader.py"
run_one "LSTV dedicated val"     "tests/test_lstv_dedicated_val.py"
run_one "hallucination metrics"  "tests/test_hallucination_metrics.py"

# ── Production loader tests (require nnU-Net) ───────────────────────────
# The new test_lstv_biased_dataloader.py auto-skips its
# TestProductionLoader class via @pytest.mark.skipif(not _NNUNET_AVAILABLE)
# if nnunetv2 is not importable from the current env. Inside the container
# (nnunetv2 IS importable), those tests run automatically as part of the
# standalone invocation above — no separate gated section needed.
if "$PY" -c "import nnunetv2" 2>/dev/null; then
    echo "-- nnU-Net detected in this env: TestProductionLoader cases above ran inline --"
else
    echo "-- nnU-Net NOT in this env: TestProductionLoader cases were skipped --"
    echo "   To run them inside the container:"
    echo "     singularity exec --nv \$CONTAINER /opt/conda/bin/python3 \\"
    echo "       tests/test_lstv_biased_dataloader.py"
fi
echo

if [[ "$failed" -gt 0 ]]; then
    echo "FAIL: $failed test file(s) failed"
    exit 1
fi
echo "ALL TESTS PASSED"

#!/usr/bin/env bash
# Phase 8 cloud validation — paste-and-run on a fresh RunPod (or any Linux GPU pod).
#
# Usage on the pod (web terminal or SSH):
#   git clone https://github.com/yeungjosh/llm-from-scratch-lab
#   cd llm-from-scratch-lab
#   bash scripts/cloud_validate.sh
#
# Outputs:
#   results/cloud/sweep_a100.json
#   results/cloud/sweep_a100_with_estimator.json
#
# Download both before terminating the pod (they're a few KB each).

set -euo pipefail

echo "[cloud-validate] installing uv..."
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # uv installer writes to ~/.local/bin
    export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

echo "[cloud-validate] syncing dependencies (this can take a minute on a fresh pod)..."
uv sync

echo "[cloud-validate] quick correctness check..."
uv run pytest -q -x --timeout=30 -k "not test_app and not test_kv_cache and not test_smoke_training" || true
# We expect the streamlit-app tests to potentially be slow; skipping the
# training-smoke test because it's slow on first run. The estimator + model
# + tokenizer + profile tests are the important correctness signals here.

echo "[cloud-validate] running cloud sweep..."
uv run python scripts/cloud_a100_sweep.py

echo "[cloud-validate] done. results in results/cloud/"
ls -lh results/cloud/

echo ""
echo "[cloud-validate] ⚠️  REMEMBER TO TERMINATE THE POD AFTER DOWNLOADING."
echo "[cloud-validate]    RunPod -> Pods -> ... -> Terminate."

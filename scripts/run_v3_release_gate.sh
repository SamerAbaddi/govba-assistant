#!/usr/bin/env bash
set -euo pipefail

echo "GovBA-GAR V3 release gate"
echo "=========================="

python -m unittest discover -v \
  -p 'test_v3_*.py'

echo
echo "Checking repository diff..."
git diff --check

echo
echo "V3 SAFE RELEASE GATE: PASS"

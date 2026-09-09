#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
jupyter nbconvert --to notebook --execute --inplace \
  src/RenanRamos_RM573201_pbl_fase5.ipynb

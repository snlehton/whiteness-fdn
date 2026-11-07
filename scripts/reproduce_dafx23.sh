#!/usr/bin/env bash
# Train the six DAFx23 configurations (Table 1) with the authors' unmodified solver.
# Paper settings: fs=48k, K=480000, M ~ U[0.8K, K], 256 samples, 80/20 split, batch 4,
# Adam lr=1e-3, 15 epochs, gamma=0.9999, alpha=300 (solver defaults).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}  # e.g. PY=.venv/Scripts/python on Windows
OUT=${OUT:-runs/reference}

declare -A DELAYS=(
  [N4_set1]="1499 1889 2381 2999"
  [N6_set1]="997 1153 1327 1559 1801 2099"
  [N8_set1]="809 877 937 1049 1151 1249 1373 1499"
  [N4_set2]="797 839 2381 2999"
  [N6_set2]="887 911 941 1699 1951 2053"
  [N8_set2]="241 263 281 293 1193 1319 1453 1597"
)

for name in ${CONFIGS:-N4_set1 N6_set1 N8_set1 N4_set2 N6_set2 N8_set2}; do
  dir="$OUT/$name"
  if [[ -f "$dir/parameters.mat" ]]; then echo "skip $name"; continue; fi
  echo "=== $name: ${DELAYS[$name]}"
  $PY scripts/run_reference.py --train_dir "$dir" --delays ${DELAYS[$name]} --max_epochs 15 --log_epochs \
    2>&1 | grep -v -E 'Warning|_set_default_tensor_type|x = torch.tensor' | tr '\r' '\n' | grep -v -E '^(Training|Generating)' > "$dir.log" || true
  tail -3 "$dir.log"
done

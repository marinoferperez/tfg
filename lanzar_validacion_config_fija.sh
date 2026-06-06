#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "Uso: $0 <rbf|rsm> <config_json>" >&2
  exit 2
fi

model="$1"
config_json="$2"
experiment_dir="${EXPERIMENT_DIR:-results/cec/cec2017_d10_tam50_reinicio_seleccionado}"
output_base="${OUTPUT_BASE:-benchmarking_surrogates_hyperparameter_validation/future_next}"

case "$model" in
  rbf|rsm) ;;
  *)
    echo "Modelo no soportado: $model" >&2
    exit 2
    ;;
esac

functions=(f1 f4 f10 f12 f18 f22 f29)
algorithms=(age de shade)
runner="surrogate_models/benchmark_utils/no_acumulativo/ejecutar_no_acumulativo.py"

for function in "${functions[@]}"; do
  for algorithm in "${algorithms[@]}"; do
    python3 "$runner" \
      --experiment-dir "$experiment_dir" \
      --benchmark-subdir "${output_base}/${model}" \
      --algoritmo "$algorithm" \
      --cec-funcid "$function" \
      --modelo "$model" \
      --modelo-params-json "$config_json" \
      --seed-selection-random-state 42 \
      --random-state 42 \
      --no-runs-json \
      --no-benchmark-summary
  done
done

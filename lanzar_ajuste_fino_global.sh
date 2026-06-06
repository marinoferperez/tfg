#!/usr/bin/env bash
set -euo pipefail

experiment_dir="${EXPERIMENT_DIR:-results/cec/cec2017_d10_tam50_reinicio_seleccionado}"
output_base="${OUTPUT_BASE:-resultados_benchmark_surrogates_offline_ajuste}"
seed_selection_random_state="${SEED_SELECTION_RANDOM_STATE:-42}"
mode="${1:-all}"

functions=(f1 f4 f10 f12 f18 f22 f29)
algorithms=(age de shade)
runner="surrogate_models/benchmark_utils/no_acumulativo/ajustar_hiperparametros.py"
summary="surrogate_models/benchmark_utils/resumir_ajuste_interno.py"

run_model() {
  local model="$1"
  local grid="$2"
  local max_seeds="$3"
  local benchmark_subdir="${output_base}/${model}"

  for function in "${functions[@]}"; do
    for algorithm in "${algorithms[@]}"; do
      args=(
        --trayectoria-dir "$experiment_dir"
        --ajuste-resultados-dir "$benchmark_subdir"
        --algoritmo "$algorithm"
        --cec-funcid "$function"
        --modelo "$model"
        --seed-selection-random-state "$seed_selection_random_state"
        --seed 42
        --param-grid-json "$grid"
        --metrica-ajuste spearman
        --validacion-ratio 0.20
        --store-tuning-results
        --no-runs-json
        --no-benchmark-summary
      )
      if [[ -n "$max_seeds" ]]; then
        args+=(--max-seeds "$max_seeds")
      fi
      python3 "$runner" "${args[@]}"
    done
  done

  python3 "$summary" \
    --benchmark-dir "${experiment_dir}/${benchmark_subdir}/no_acumulativo" \
    --model "$model"
}

case "$mode" in
  rbf)
    run_model rbf surrogate_models/configs/rbf_refinement_36.json "${RBF_MAX_SEEDS:-10}"
    ;;
  rsm)
    run_model rsm surrogate_models/configs/rsm.json "${RSM_MAX_SEEDS:-}"
    ;;
  all)
    run_model rsm surrogate_models/configs/rsm.json "${RSM_MAX_SEEDS:-}"
    run_model rbf surrogate_models/configs/rbf_refinement_36.json "${RBF_MAX_SEEDS:-10}"
    ;;
  *)
    echo "Uso: $0 [rbf|rsm|all]" >&2
    exit 2
    ;;
esac

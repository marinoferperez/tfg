#!/usr/bin/env bash
set -euo pipefail

for truncation in convergencia sin_convergencia; do
  if [[ "$truncation" == "convergencia" ]]; then
    trunc_suffix="_convergencia"
  else
    trunc_suffix=""
  fi
    validation="next"
    benchmark_base="benchmarking_surrogates_offline_next${trunc_suffix}"
    for protocol in no_acumulativo acumulativo; do
      if [[ "$protocol" == "no_acumulativo" ]]; then
        runner="surrogate_models/benchmark_utils/no_acumulativo/ejecutar_no_acumulativo.py"
        resumen="surrogate_models/benchmark_utils/no_acumulativo/no_acumulativo_resumen.py"
      else
        runner="surrogate_models/benchmark_utils/acumulativo/ejecutar_acumulativo.py"
        resumen="surrogate_models/benchmark_utils/acumulativo/acumulativo_resumen.py"
      fi
      for funcion in f1 f3 f5 f10 f17 f29; do
        for algoritmo in age de shade; do
          for model in lasso rsm mlp rbf random_forest hgb xgboost; do
            args=(
              --experiment-dir results/cec/cec2017_d10_tam50
              --benchmark-subdir "${benchmark_base}/future_${validation}/${model}"
              --algoritmo "$algoritmo"
              --cec-funcid "$funcion"
              --modelo "$model"
              --max-seeds 15
              --seed-selection-random-state 42
              --random-state 42
              --no-runs-json
              --no-benchmark-summary
            )
            [[ "$truncation" == "convergencia" ]] && args+=(--convergence-truncation)
            python3 "$runner" "${args[@]}"
          done
        done
        for benchmark_dir in "results/cec/cec2017_d10_tam50/${benchmark_base}/future_${validation}"/*/"${protocol}"/"${funcion}"; do
          [[ -d "$benchmark_dir" ]] && python3 "$resumen" --benchmark-dir "$benchmark_dir" --no-rankings --no-json
        done
      done
    done
done

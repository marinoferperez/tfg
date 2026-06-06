**RETRAIN**

tras escoger el cooldown ganador:

python3 metaheuristics/online/experiments/ejecutar_metaheuristicas_online.py \
  --algoritmo age \
  --cec-funcid 3 5 9 15 19 24 27 \
  --cec-dim 10 \
  --seeds 101 102 103 104 105 106 107 108 109 110 \
  --tam-poblacion 50 \
  --max-evals 100000 \
  --reinicio \
  --reinicio-ratio 0.01 \
  --surrogate-prob 0.50 \
  --warmup-ratio 0.20 \
  --window-ratio 0.20 \
  --cooldown-reinicio-evals 50 \
  --retrain-ratio 0.25 \
  --sin-metricas \
  --outdir piloto_online_retrain_rt025 \
  --verbose

--retrain-ratio 0.50 --outdir piloto_online_retrain_rt050
--retrain-ratio 0.75 --outdir piloto_online_retrain_rt075
--retrain-ratio 1

python3 metaheuristics/online/experiments/ejecutar_metaheuristicas_online.py \
  --algoritmo de shade \
  --cec-funcid 3 5 9 15 19 24 27 \
  --cec-dim 10 \
  --seeds 101 102 103 104 105 106 107 108 109 110 \
  --tam-poblacion 50 \
  --max-evals 100000 \
  --reinicio \
  --reinicio-ratio 0.10 \
  --surrogate-prob 0.50 \
  --warmup-ratio 0.20 \
  --window-ratio 0.20 \
  --cooldown-reinicio-evals 50 \
  --retrain-ratio 0.25 \
  --sin-metricas \
  --outdir piloto_online_retrain_rt025 \
  --verbose

  repetir con los mismos

**PILOTO P**

python3 metaheuristics/online/experiments/ejecutar_metaheuristicas_online.py \
  --algoritmo age \
  --cec-funcid 3 5 9 15 19 24 27 \
  --cec-dim 10 \
  --seeds 101 102 103 104 105 106 107 108 109 110 \
  --tam-poblacion 50 \
  --max-evals 100000 \
  --reinicio \
  --reinicio-ratio 0.01 \
  --surrogate-prob 0.00 \
  --warmup-ratio 0.20 \
  --window-ratio 0.20 \
  --cooldown-reinicio-evals 50 \
  --retrain-ratio 0.50 \
  --sin-metricas \
  --outdir piloto_online_p_p000 \
  --verbose

--surrogate-prob 0.25 --outdir piloto_online_p_p025
--surrogate-prob 0.50 --outdir piloto_online_p_p050
--surrogate-prob 0.75 --outdir piloto_online_p_p075

python3 metaheuristics/online/experiments/ejecutar_metaheuristicas_online.py \
  --algoritmo de shade \
  --cec-funcid 3 5 9 15 19 24 27 \
  --cec-dim 10 \
  --seeds 101 102 103 104 105 106 107 108 109 110 \
  --tam-poblacion 50 \
  --max-evals 100000 \
  --reinicio \
  --reinicio-ratio 0.10 \
  --surrogate-prob 0.00 \
  --warmup-ratio 0.20 \
  --window-ratio 0.20 \
  --cooldown-reinicio-evals 50 \
  --retrain-ratio 0.50 \
  --sin-metricas \
  --outdir piloto_online_p_p000 \
  --verbose

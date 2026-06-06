from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from surrogate_models.benchmark_utils.batches_eval_splitter import construir_casos_no_acumulativos
from surrogate_models.benchmark_utils.evaluacion_offline import main_temporal


if __name__ == "__main__":
    main_temporal(
        protocol="no_acumulativo",
        split_strategy="temporal_no_acumulativo_futuro",
        constructor_casos=construir_casos_no_acumulativos,
        resumen_script_name="resumir_no_acumulativo.py",
        description=(
            "Benchmark temporal no_acumulativo para una funcion y un algoritmo, "
            "trabajando directamente con datasets por seed."
        ),
    )

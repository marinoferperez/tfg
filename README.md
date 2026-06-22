# Hibridación de Metaheurísticas con Modelos Subrogados sobre CEC2017

Repositorio del Trabajo de Fin de Grado. Contiene el código necesario para reproducir los experimentos principales: ejecución de metaheurísticas, benchmark offline de modelos subrogados, ajuste de hiperparámetros y evaluación con estrategia online.

## Requisitos

```bash
pip install -r requirements.txt
```

## Datos CEC2017

Los experimentos requieren los ficheros de datos del benchmark CEC2017. Estos no están incluidos en el repositorio por su tamaño. Descárgalos desde:

> https://github.com/P-N-Suganthan/2017-CEC-100-Digit-Challenge

Coloca el contenido descargado en la carpeta `cec2017real/` en la raíz del repositorio:

```
tfg/
└── cec2017real/
    ├── ...
```

## Scripts principales

Todos los scripts se ejecutan desde la raíz del repositorio.

### 1. Ejecutar metaheurísticas

Genera datasets por seed para el benchmark offline.

```bash
python scripts/run_metaheuristics.py \
  --cec-funcid all \
  --algorithm age de shade \
  --n-seeds 31 \
  --max-evals 10000 \
  --pop-size 100 \
  --outdir results/experimento
```

### 2. Benchmark offline

Evalúa modelos subrogados sobre los datasets generados.

```bash
python scripts/run_offline_eval.py \
  --strategy non_cumulative \
  --experiment-dir results/experimento \
  --cec-funcid all \
  --algorithm age de shade \
  --model rbf
```

Estrategias disponibles: `cumulative`, `non_cumulative`.

### 3. Ajuste de hiperparámetros

Busca los mejores hiperparámetros del modelo subrogado por función y algoritmo.

```bash
python scripts/run_hyperparameter_tuning.py \
  --strategy non_cumulative \
  --experiment-dir results/experimento \
  --cec-funcid all \
  --algorithm age de shade \
  --model rbf
```

### 4. Evaluación online

Ejecuta las metaheurísticas con el subrogado RBF integrado como filtro online.

```bash
python scripts/run_online.py \
  --cec-funcid all \
  --algorithm age de shade \
  --cec-dim 10 \
  --n-seeds 31 \
  --max-evals 10000 \
  --pop-size 100 \
  --outdir results/experimento_online
```

## Estructura del repositorio

```
scripts/          Scripts principales de ejecución
src/              Código fuente (metaheurísticas, subrogados, utilidades)
memoria/          Memoria del TFG en LaTeX
visual_tools/     Scripts para generar figuras y tablas de la memoria
requirements.txt  Dependencias Python
```

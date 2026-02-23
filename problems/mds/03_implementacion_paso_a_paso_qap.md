# Implementacion Paso a Paso del QAP para AGE/DE

## 1. Clase problema

Crear `QAPProblem` con:

- `flow_matrix` (`F`),
- `distance_matrix` (`D`),
- `n` (tamano),
- `fitness(solution)` en minimizacion.

## 2. Codificacion para AGE/DE continuos

Usar **random keys**:

- cromosoma real `x` de tamano `n`.
- decodificacion a permutacion: `argsort(x)`.

Esto permite reutilizar AGE y DE sin cambiar sus operadores continuos.

## 3. Evaluacion

1. Decodificar `x -> p`.
2. Calcular coste QAP:
   `sum_i sum_j F[i,j] * D[p(i), p(j)]`.
3. Devolver coste (minimizacion).

## 4. Adaptacion AGE

- AGE opera sobre `x` real.
- Cada hijo se decodifica a permutacion para evaluar.
- Reemplazo segun menor fitness.

## 5. Adaptacion DE

- DE opera sobre `x` real.
- Tras mutacion/cruce, se recorta a limites `[0,1]`.
- Evaluacion por decodificacion `argsort`.

## 6. Datos de entrada

- Soporte de matrices directas (`F`, `D`) para pruebas.
- Soporte de fichero QAPLIB `.dat` para benchmark.

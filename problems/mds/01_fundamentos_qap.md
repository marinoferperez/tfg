# Fundamentos del Quadratic Assignment Problem (QAP)

## 1. Definicion informal

Dadas `n` instalaciones y `n` localizaciones, se busca una asignacion
(permutacion) que minimice el coste total de interacciones.

## 2. Definicion formal

- Entrada:
  - Matriz de flujos `F` de tamano `n x n`.
  - Matriz de distancias `D` de tamano `n x n`.
- Solucion:
  - Permutacion `p`, donde `p(i)` es la localizacion asignada a la instalacion `i`.
- Objetivo (minimizar):

`sum_i sum_j F[i,j] * D[p(i), p(j)]`

## 3. Naturaleza del problema

- Combinatorio y NP-hard.
- Determinista: misma solucion -> mismo coste.
- Muy usado en benchmarks de metaheuristicas (QAPLIB).

## 4. Relacion con grafos

QAP puede interpretarse como el alineamiento entre dos grafos ponderados:

- grafo de flujo entre instalaciones,
- grafo de distancia entre localizaciones.

La permutacion define como se emparejan los nodos de ambos grafos.

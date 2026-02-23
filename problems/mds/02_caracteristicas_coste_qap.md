# Caracteristicas y Coste Computacional del QAP

## 1. Por que QAP es costoso

### 1.1 Espacio de busqueda enorme

El numero de soluciones es `n!`.

### 1.2 Evaluacion de fitness cuadratica

Cada evaluacion usa un doble sumatorio sobre `n x n`:

- coste por evaluacion aproximado: `O(n^2)`.

Con muchas evaluaciones (miles o decenas de miles), el coste acumulado es alto.

## 2. Por que encaja con el contexto del proyecto

Cumple los requisitos pedidos:

- problema determinista,
- problema combinatorio,
- coste significativo por fitness y por numero de evaluaciones,
- uso consolidado como benchmark.

## 3. Ventaja frente a otros problemas combinatorios

QAP tiene menos fases de implementacion delicadas:

- no requiere reparacion estructural compleja,
- no requiere verificacion de conectividad por terminales,
- no requiere proyeccion a arbol/rutas.

Aun asi mantiene dificultad real para metaheuristicas.

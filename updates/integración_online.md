## Plan técnico para la integración online

toda la logica de:

- cuando activar el subrogado
- con que datos entrenarlo
- cuando reentrenarlo
- cuando predecir
- como decidir aceptar/rechazar
- como registrar estadisticas

es comun


Dado que las configuraciones finales de las metaheurísticas incorporan reinicio elitista, la primera decisión de la integración online consiste en determinar si el subrogado debe actuar inmediatamente después de un reinicio. Esta cuestión se analiza antes de ajustar la probabilidad de uso del modelo, ya que afecta a la validez de las predicciones en los instantes posteriores a una regeneración de la población. Para ello se fija provisionalmente \(p=0.5\) y una frecuencia intermedia de reentrenamiento, comparando distintos periodos de enfriamiento.

Los resultados muestran que utilizar el subrogado inmediatamente después del reinicio empeora el comportamiento agregado. Entre las esperas evaluadas, cooldown=100 obtiene el mejor compromiso global: mejora a AGE y SHADE, no deteriora de forma apreciable a DE y evita el apagado excesivo observado con cooldown=500. Por ello, se fija como valor común para las tres metaheurísticas.



### Objetivo de la implementación

La integración online debe incorporar el modelo RBF ajustado dentro del bucle de optimización de las metaheurísticas, pero sin sustituir la evaluación real como criterio definitivo de aceptación. El subrogado se utilizará como filtro previo:

- si el candidato no parece prometedor, se rechaza sin evaluar la función objetivo real;
- si el candidato parece prometedor, se evalúa realmente;
- ninguna solución predicha puede entrar como mejor solución sin una evaluación real.

El objetivo experimental no es reducir artificialmente el número de evaluaciones reales por debajo del presupuesto, sino aprovechar mejor ese presupuesto evitando evaluaciones sobre candidatos que el modelo considera poco competitivos.

""
La función objetivo real es costosa. Por tanto, el subrogado se usa como filtro para evitar gastar evaluaciones reales en candidatos que probablemente no van a mejorar la población. Las evaluaciones reales ahorradas pueden emplearse en seguir generando nuevos candidatos hasta consumir el mismo presupuesto máximo, pero con una selección previa más informada.
""

""
Dado que la función objetivo real representa el componente computacionalmente costoso del proceso de optimización, la integración \textit{online} del subrogado se plantea como un mecanismo de filtrado destinado a reducir evaluaciones reales poco prometedoras y, con ello, aprovechar de forma más eficiente el presupuesto disponible.
""

""
El subrogado no reemplaza definitivamente a la función objetivo, sino que decide qué candidatos merecen ser evaluados realmente. Por tanto, toda solución aceptada por la metaheurística sigue estando validada mediante la función real.
""

""
Tras un reinicio no se repite el calentamiento inicial del 20 %, ya que esto podría impedir el uso efectivo del subrogado en ejecuciones con varios reinicios. En su lugar, el modelo vigente se invalida y se reentrena en el siguiente uso del subrogado empleando la ventana reciente de evaluaciones reales. De este modo, se reconoce el cambio de región de búsqueda sin desactivar prolongadamente la integración online.
""

""
Tras cada reinicio se desactiva temporalmente el subrogado durante un número de evaluaciones equivalente al tamaño de la población. Esta decisión permite recopilar una muestra real mínima de la nueva región de búsqueda antes de volver a entrenar el modelo, evitando reutilizar inmediatamente un subrogado ajustado sobre una distribución anterior. No se repite el calentamiento inicial completo del 20 %, ya que en ejecuciones con varios reinicios ello podría impedir el uso efectivo del subrogado durante gran parte del presupuesto.
""

""
Tras cada reinicio, se desactiva temporalmente el uso del subrogado durante un número de evaluaciones reales igual al tamaño de la población. En los experimentos realizados, dado que se emplea \(N=50\), este enfriamiento corresponde a 50 evaluaciones reales.
""


""
Se adopta retrain_ratio=0.50 porque reduce aproximadamente a la mitad el número de reentrenamientos respecto a 0.25, sin degradar el comportamiento medio del filtro en el piloto realizado. La opción 0.75, aunque más barata, presenta una degradación más clara en algunas metaheurísticas, especialmente en DE.
""

""
El reentrenamiento periódico y el enfriamiento tras reinicio cumplen funciones distintas. El primero controla la frecuencia con la que el modelo se actualiza con nuevas evaluaciones reales, mientras que el segundo evita aplicar el subrogado inmediatamente después de un reinicio, momento en el que la distribución de soluciones puede cambiar de forma abrupta. Por este motivo, ambos mecanismos se mantienen de forma simultánea.
""

""
Antes de estudiar la probabilidad de uso del subrogado, se evalúa la frecuencia de actualización del modelo, ya que un reentrenamiento excesivamente frecuente puede aumentar el coste computacional sin mejorar la calidad de las decisiones, mientras que una actualización demasiado espaciada puede dejar obsoleto el modelo.
""

""
Con la frecuencia de actualización fijada, se analiza la probabilidad de uso del subrogado. Este parámetro controla la agresividad de la hibridación: valores bajos apenas modifican la dinámica original, mientras que valores altos delegan más decisiones en el modelo aproximado.
""

""
Finalmente, se estudia el tratamiento del subrogado tras un reinicio. Dado que el reinicio modifica bruscamente la región explorada, se compara el uso inmediato del modelo frente a un periodo de enfriamiento de una o dos poblaciones completas.
""

""
el cooldown se introduce para evitar que el subrogado tome decisiones inmediatamente después de un reinicio, momento en el que la distribución de soluciones cambia de forma abrupta y el modelo entrenado antes del reinicio puede no representar adecuadamente la nueva región explorada.
""

 

### Restricciones de diseño

La implementación debe cumplir estas restricciones:

- No modificar las implementaciones base de las metaheurísticas.
- Mantener AGE, DE y SHADE en sus archivos actuales como versiones de referencia.
- Crear variantes online separadas, explícitamente identificables.
- Reutilizar el código común del subrogado en un módulo compartido.
- Evitar duplicar la lógica de entrenamiento, predicción, registro de eventos y configuración.
- Mantener separados los resultados offline, los resultados de reinicio y los resultados online.

Esta separación es importante porque las metaheurísticas actuales son la línea base experimental. Si se mezclan con la hibridación online, después será difícil garantizar que las comparaciones se están haciendo contra la versión original.

### Punto técnico clave

AGE está implementado directamente en el repositorio, por lo que el filtro online puede insertarse de forma relativamente directa antes de evaluar los hijos.

DE y SHADE, en cambio, delegan la generación y selección de candidatos en PYADE. Por tanto, no basta con envolver `problem.fitness`, porque en ese punto ya se ha perdido información importante: no sabemos contra qué individuo compite el candidato. Para usar el subrogado como filtro de rechazo necesitamos conocer:

- el candidato generado;
- el fitness real de la referencia contra la que compite;
- si el candidato aceptado debe reemplazar o no a esa referencia.

Por tanto, para DE y SHADE hará falta una variante online específica que replique o adapte el bucle de PYADE, manteniendo intactas las versiones originales.

### Estructura de archivos propuesta

La estructura recomendada sería:

```text
metaheuristics/
  online/
    __init__.py

    surrogate_controller.py
    surrogate_policy.py
    surrogate_stats.py
    rbf_factory.py

    algorithms/
      __init__.py
      age_online.py
      de_online.py
      shade_online.py

    adapted/
      __init__.py
      age_cec2017_online.py
      de_cec2017_online.py
      shade_cec2017_online.py

    experiments/
      ejecutar_metaheuristicas_online.py
      resumir_resultados_online.py
      generar_tablas_online.py
```

También mantendría la configuración del modelo en:

```text
surrogate_models/configs/rbf_multiquadric_eps1_smoothing1e-3_neighbors50.json
```

ya que esa configuración ya representa el RBF seleccionado tras el ajuste fino.

### Responsabilidad de cada módulo

#### `surrogate_controller.py`

Debe ser el núcleo de la integración online. Sus responsabilidades:

- almacenar las evaluaciones reales disponibles;
- decidir cuándo el subrogado puede activarse;
- entrenar RBF con la ventana reciente;
- lanzar predicciones;
- aplicar la política de rechazo/aceptación;
- acumular tiempos de entrenamiento y predicción;
- registrar contadores agregados.

Parámetros principales:

```text
warmup_ratio = 0.20
window_ratio = 0.20
surrogate_probability = p
max_evals = 100000
population_size = 50
```

El subrogado se activa solo cuando:

```text
evals_reales >= warmup_ratio * max_evals
```

En nuestro caso:

```text
evals_reales >= 20000
```

#### `surrogate_policy.py`

Debe contener la regla de decisión. Para minimización:

```text
si f_predicha(candidato) < f_real(referencia):
    evaluar con función real
si no:
    rechazar sin evaluación real
```

No introduciría margen en la primera versión. Así el único parámetro experimental relevante es `p`, el porcentaje de uso del subrogado.

#### `surrogate_stats.py`

Debe guardar los contadores online. Como registrar cada candidato puede generar ficheros muy grandes, conviene separar dos niveles:

- resumen por ejecución;
- trazas detalladas solo para pruebas pequeñas o depuración.

Métricas agregadas por run:

```text
evals_reales
candidatos_generados
candidatos_con_subrogado
candidatos_evaluados_directamente
candidatos_rechazados
candidatos_aceptados_por_subrogado
porcentaje_rechazo
entrenamientos_rbf
tiempo_entrenamiento_total
tiempo_prediccion_total
reinicios
mejor_fitness
mejor_error
```

#### `rbf_factory.py`

Debe construir siempre el mismo RBF seleccionado:

```text
kernel = multiquadric
epsilon = 1.0
smoothing = 1e-3
neighbors = 50
degree = -1
```

Así evitamos instanciar el modelo a mano en cada algoritmo.

### Variantes online por metaheurística

#### AGE online

AGE genera dos hijos en cada iteración. La integración natural es:

1. Generar `hijo1` y `hijo2`.
2. Para cada hijo, decidir si se aplica el subrogado según probabilidad `p`.
3. Si el hijo es rechazado, no se evalúa realmente.
4. Si el hijo es aceptado por el subrogado o cae en evaluación directa, se evalúa con CEC2017.
5. Entre los hijos realmente evaluados, se selecciona el mejor.
6. El mejor hijo evaluado compite contra el peor individuo de la población.

Si ambos hijos son rechazados, no hay reemplazo en esa iteración.

#### DE online

DE genera un vector trial para cada individuo objetivo. La integración natural es:

1. Generar el trial vector.
2. Tomar como referencia el individuo objetivo actual.
3. Aplicar el subrogado con probabilidad `p`.
4. Si el trial es rechazado, se mantiene el individuo objetivo sin evaluación real.
5. Si el trial es aceptado, se evalúa realmente.
6. Si el fitness real del trial mejora al objetivo, se reemplaza.

Este es el caso más limpio para la hibridación, porque la referencia de comparación está perfectamente definida.

#### SHADE online

SHADE es similar a DE en la selección, pero adapta internamente `F` y `CR`. La integración debe respetar:

- generación `current-to-pbest/1`;
- memoria histórica de parámetros exitosos;
- archivo externo si se usa;
- actualización de memoria solo cuando hay evaluaciones reales exitosas.

El subrogado puede rechazar trials antes de la evaluación real, pero esos trials rechazados no deben considerarse éxitos ni actualizar la memoria adaptativa.

### Gestión del reinicio

No se aplicará un nuevo calentamiento obligatorio del 20 % tras cada reinicio. La razón es práctica y metodológica:

- podría dejar el subrogado apagado durante gran parte de la ejecución;
- el enfoque no acumulativo ya reduce el peso de muestras antiguas;
- el objetivo es observar si la ventana reciente permite adaptarse tras los reinicios.

Lo que sí debe registrarse:

```text
eval_id del reinicio
generación del reinicio
fitness preservado
rechazos antes/después del reinicio
evaluaciones reales antes/después del reinicio
```

Esto permitirá analizar posteriormente si los reinicios deterioran el comportamiento del subrogado.

### Runner experimental

El script principal debería ser:

```text
metaheuristics/online/experiments/ejecutar_metaheuristicas_online.py
```

CLI propuesta:

```bash
python3 metaheuristics/online/experiments/ejecutar_metaheuristicas_online.py \
  --algoritmo de \
  --cec-funcid 1 \
  --cec-dim 10 \
  --tam-poblacion 50 \
  --max-evals 100000 \
  --seed 1 \
  --reinicio \
  --reinicio-ratio 0.10 \
  --surrogate-model rbf \
  --surrogate-prob 0.50 \
  --warmup-ratio 0.20 \
  --window-ratio 0.20 \
  --outdir results/cec/cec2017_d10_tam50_online_rbf
```

Para AGE se usaría `--reinicio-ratio 0.01`; para DE y SHADE, `0.10`.

### Estructura de resultados

Propuesta:

```text
results/
  cec/
    cec2017_d10_tam50_online_rbf/
      p_000/
      p_025/
      p_050/
      p_075/
        f1/
          age/
            runs.csv
            seed_1/
              metricas.csv
              reinicios_elitistas.csv
              online_summary.json
          de/
          shade/
        f2/
        ...
        f30/
```

`p_000` puede corresponder al baseline sin subrogado. Aun así, conviene comprobar que `p=0` reproduce la metaheurística original antes de usarlo como baseline definitivo.

### Validaciones antes de lanzar experimentos largos

Antes de ejecutar las 30 funciones y 51 semillas:

1. Ejecutar una función sencilla con una semilla:

```text
f1, seed=1, DE, p=0.50
```

2. Comprobar que:

```text
evals_reales == 100000
ninguna solución aceptada procede solo de predicción
los rechazos no incrementan evals_reales
el número de predicciones coincide con candidatos_con_subrogado
el tiempo de entrenamiento y predicción se registra
```

3. Ejecutar `p=0` y comparar contra la DE original:

```text
mismo fitness final o diferencias justificadas únicamente por cambios de implementación
```

Si `p=0` no reproduce la línea base, no se debe lanzar el experimento completo.

### Matriz experimental

Configuraciones principales:

```text
baseline / p = 0.00
RBF online / p = 0.25
RBF online / p = 0.50
RBF online / p = 0.75
```

Sobre:

```text
30 funciones CEC2017
51 semillas
3 metaheurísticas
```

Total:

```text
4 estrategias * 30 funciones * 51 semillas * 3 metaheurísticas = 18360 ejecuciones
```

Si el tiempo de cómputo es demasiado alto, se puede hacer una fase piloto:

```text
6 funciones representativas * 15 semillas * 3 metaheurísticas * 4 estrategias
```

Después se lanza el experimento completo solo con las configuraciones más prometedoras.

### Riesgos técnicos

- DE y SHADE requieren controlar el bucle de selección, por lo que no basta con interceptar `fitness`.
- Un valor alto de `p` puede rechazar demasiados candidatos y ralentizar la generación de evaluaciones reales útiles.
- Si se registra cada candidato, el volumen de datos puede crecer demasiado.
- Tras reinicios frecuentes, el subrogado puede entrenarse con una ventana que todavía contiene muestras de regiones anteriores.
- La comparación final debe hacerse por calidad de solución, no por Spearman online completo, ya que los candidatos rechazados no tienen fitness real.

### Orden de implementación recomendado

1. Implementar `surrogate_controller.py`, `surrogate_policy.py`, `surrogate_stats.py` y `rbf_factory.py`.
2. Implementar AGE online, porque el bucle está controlado en el repositorio.
3. Validar AGE con una función y una semilla.
4. Implementar DE online, adaptando el bucle para poder filtrar trial vectors.
5. Validar que `p=0` reproduce DE original.
6. Implementar SHADE online.
7. Añadir el runner CLI.
8. Lanzar piloto.
9. Generar resumen agregado.
10. Lanzar experimento completo.

---

Te propongo plantearlo como una fase experimental con una pregunta muy concreta:

**Pregunta experimental**  
Evaluar si la integración \textit{online} de RBF permite mejorar el rendimiento de las metaheurísticas con reinicio, reduciendo evaluaciones reales poco prometedoras sin perjudicar la calidad final de las soluciones.

La idea no debe ser “usar el surrogate para sustituir la función objetivo”, sino usarlo como **filtro de rechazo**. Esto encaja exactamente con lo que dice Daniel: si el subrogado rechaza, se evita la evaluación real; si el subrogado acepta, la solución se evalúa realmente.

**Configuración Fija**
Mantendría constantes estos elementos:

- Suite: CEC2017.
- Dimensión: \(D=10\).
- Presupuesto máximo: \(100000\) evaluaciones reales.
- Tamaño de población: \(50\).
- Metaheurísticas: AGE, DE y SHADE.
- Reinicio: usar las configuraciones ya seleccionadas previamente:
  - AGE con reinicio \(1\%\).
  - DE con reinicio \(10\%\).
  - SHADE con reinicio \(10\%\).
- Modelo subrogado: RBF ajustado.
- Configuración RBF:
  - kernel: \textit{multiquadric}
  - \(\epsilon = 1.0\)
  - smoothing \(= 10^{-3}\)
  - vecinos \(= 50\)
  - degree \(= -1\)
- Estrategia de entrenamiento: no acumulativa, mediante ventana deslizante.
- Calentamiento inicial: primer \(20\%\) del presupuesto solo con evaluación real.

Es decir, durante las primeras \(20000\) evaluaciones reales no se usa el subrogado. Solo se recogen datos.

**Mecanismo Online**
A partir del \(20\%\), cada vez que la metaheurística genere una solución candidata:

1. Con probabilidad \(p\), se aplica el filtro subrogado.
2. Con probabilidad \(1-p\), se evalúa directamente con la función objetivo real.
3. Si se aplica el subrogado:
   - RBF predice el fitness de la solución candidata.
   - Si la predicción indica que la solución no es prometedora, se rechaza sin evaluación real.
   - Si la predicción indica que puede ser prometedora, se evalúa con la función real.
4. Ninguna solución predicha puede aceptarse como mejor solución sin evaluación real.

El parámetro experimental principal sería \(p\), el porcentaje de uso del subrogado.

**Valores A Evaluar**
Yo probaría estos valores:

```text
p = 0.25
p = 0.50
p = 0.75
```

Y añadiría:

```text
p = 0.00
```

como baseline, que equivale a la metaheurística con reinicio sin subrogado.

Opcionalmente, si el tiempo lo permite:

```text
p = 1.00
```

pero lo trataría como escenario agresivo, no como candidato principal.

**Criterio De Aceptación**
Como CEC2017 es minimización, el subrogado debería aceptar una solución candidata si su fitness predicho es competitivo frente a la solución contra la que vaya a competir.

Por ejemplo:

- En DE/SHADE: comparar la predicción del trial vector con el fitness real del individuo objetivo.
- En AGE: comparar la predicción del descendiente con el individuo que sería reemplazado, o con el peor individuo de la población, según cómo esté implementado el reemplazo.

Regla básica:

```text
si f_predicha(candidato) < f_real(referencia):
    evaluar candidato con función real
else:
    rechazar candidato
```

Para evitar que el subrogado sea demasiado agresivo, se puede introducir margen:

```text
si f_predicha(candidato) < f_real(referencia) * (1 + margen):
    evaluar realmente
else:
    rechazar
```

Pero para una primera versión yo empezaría sin margen. Primero conviene medir el comportamiento limpio.

**Reinicio**
No reiniciaría automáticamente otro \(20\%\) tras cada reinicio. Eso puede apagar el surrogate durante demasiada parte de la ejecución.

La justificación sería:

> Tras un reinicio, la distribución de soluciones cambia, pero la estrategia no acumulativa ya mitiga este problema al entrenar el modelo con una ventana reciente de evaluaciones reales. Por tanto, no se reinicia el calentamiento completo, sino que el modelo se adapta progresivamente conforme se incorporan nuevas evaluaciones reales posteriores al reinicio.

Lo que sí registraría es:

- número de reinicios,
- en qué evaluación ocurre cada reinicio,
- número de soluciones rechazadas por el subrogado antes y después de cada reinicio,
- calidad final obtenida.

Así, si el tutor luego pregunta por el efecto del reinicio, tienes trazabilidad.

**Métricas**
Compararía cada configuración con la metaheurística base usando:

- Mejor fitness final.
- Fitness medio final por función.
- Ranking medio frente al baseline.
- Número de evaluaciones reales consumidas.
- Número de candidatos filtrados por el subrogado.
- Porcentaje de rechazo.
- Tiempo total de ejecución.
- Tiempo dedicado a entrenamiento del subrogado.
- Tiempo dedicado a predicción.
- Número de veces que el subrogado acepta y luego la evaluación real confirma mejora.
- Número de falsos positivos: el subrogado acepta, pero la solución no mejora.
- Número de falsos negativos no se puede saber directamente, porque las soluciones rechazadas no se evalúan. Esto hay que reconocerlo.

**Comparativa Principal**
La tabla final debería tener una estructura de este tipo:

```text
Metaheurística | Estrategia | p | Fitness medio | Ranking | Eval. reales | Rechazos (%) | Tiempo total
AGE            | baseline   | 0 | ...
AGE            | RBF online | 0.25 | ...
AGE            | RBF online | 0.50 | ...
AGE            | RBF online | 0.75 | ...
DE             | baseline   | 0 | ...
...
SHADE          | ...
```

Y después una tabla agregada por estrategia:

```text
Estrategia | Ranking medio | Mejora vs baseline | Rechazos (%) | Tiempo total
baseline   | ...
RBF p=0.25 | ...
RBF p=0.50 | ...
RBF p=0.75 | ...
```

**Hipótesis Esperadas**
Yo formularía estas hipótesis:

- \(p=0.25\): integración conservadora. Debería ser la más estable, con pocos rechazos erróneos.
- \(p=0.50\): equilibrio entre exploración real y filtrado subrogado.
- \(p=0.75\): más ahorro potencial, pero mayor riesgo de rechazar soluciones útiles.
- \(p=1.00\): escenario extremo, útil para ver el límite del método, pero probablemente menos robusto.

La conclusión esperada no tiene por qué ser que “más surrogate es mejor”. De hecho, lo razonable es buscar un punto intermedio.

**Orden De Ejecución**
Yo lo haría así:

1. Implementar RBF online solo en una metaheurística, preferiblemente DE, porque su comparación candidato-padre es directa.
2. Validar que:
   - no se aceptan soluciones sin evaluación real,
   - el contador de evaluaciones reales funciona bien,
   - las predicciones no consumen evaluación,
   - el presupuesto termina en \(100000\) evaluaciones reales.
3. Ejecutar una prueba pequeña:
   - pocas funciones,
   - pocas semillas,
   - \(p=0.25\) y \(p=0.50\).
4. Si no hay errores, lanzar AGE, DE y SHADE con:
   - baseline,
   - \(p=0.25\),
   - \(p=0.50\),
   - \(p=0.75\).
5. Analizar primero resultados agregados.
6. Después analizar por función si aparecen degradaciones fuertes.

**Planteamiento Para La Memoria**
La sección se podría redactar con esta idea:

> La integración online se plantea como un mecanismo de preselección de candidatos. El modelo RBF no sustituye a la función objetivo, sino que actúa como filtro previo: únicamente las soluciones consideradas prometedoras por el subrogado son evaluadas mediante la función real. De este modo, se evita que una predicción no confirmada pueda incorporarse como mejor solución, manteniendo la evaluación real como único criterio definitivo de aceptación.

Ese enfoque es metodológicamente sólido y está alineado con lo que te ha dicho Daniel.

---

Aquí hay que separar dos usos distintos del “ranking”.

En la fase offline, Spearman/ranking mide si el subrogado **ordena bien soluciones que sí conocemos**, porque tenemos el fitness real de todas las muestras de validación. Ahí puedes decir: “RBF preserva razonablemente el orden relativo de las soluciones”.

En la fase online, el ranking ya no es una métrica directa de evaluación del modelo en cada rechazo, porque si el subrogado rechaza una solución, **no conoces su fitness real**. Por tanto, no puedes calcular de forma completa si el modelo ha ordenado correctamente todos los candidatos generados.

En online, el ranking tiene otro papel:

- **Como criterio interno del subrogado**: RBF predice valores y esos valores se usan para decidir si un candidato parece mejor o peor que una referencia.
- **Como métrica final de comparación entre algoritmos**: puedes rankear las estrategias según el fitness final obtenido en CEC2017, igual que se suele hacer al comparar metaheurísticas.
- **Como métrica diagnóstica parcial**: solo puedes calcular aciertos/fallos sobre los candidatos que finalmente sí fueron evaluados con la función real.

Por eso conviene no decir que en online vamos a medir “Spearman completo” de todos los candidatos, salvo que hagamos una ejecución de diagnóstico donde también evaluemos los rechazados. En la ejecución real, Spearman offline justifica elegir RBF, pero el rendimiento online se evalúa principalmente por:

- fitness final,
- ranking medio de las estrategias,
- mejora o empeoramiento frente al baseline,
- número de evaluaciones reales aprovechadas,
- porcentaje de rechazos,
- tiempo total.

La frase correcta sería:

> En la fase \textit{online}, el subrogado utiliza su capacidad de ordenación estimada en la fase \textit{offline} para filtrar candidatos, pero la comparación experimental se realiza sobre el rendimiento final de las metaheurísticas, no sobre el Spearman de todos los candidatos generados.

Esto evita mezclar predicción del modelo con evaluación final del algoritmo.

---

Porque en este proyecto el subrogado no se va a usar para **predecir exactamente el valor del fitness**, sino para **decidir si una solución parece suficientemente prometedora como para evaluarla realmente**.

Esa decisión depende más del **orden relativo** que del valor numérico exacto.

Ejemplo simple:

```text
Fitness real:      A = 100, B = 120, C = 200
Predicción modelo: A = 10,  B = 15,  C = 80
```

El modelo está fatal calibrado en escala: los errores absolutos son enormes. Pero ordena bien:

```text
A mejor que B mejor que C
```

Para una estrategia de filtrado, eso puede ser útil.

En cambio:

```text
Fitness real:      A = 100, B = 120, C = 200
Predicción modelo: A = 100, B = 300, C = 90
```

Aquí puede que algunos errores medios no parezcan tan extremos dependiendo del conjunto, pero el modelo ordena mal: cree que C es mejor que A. Eso es peligroso para decidir qué candidatos evaluar.

Por eso Spearman es la métrica principal: mide si el modelo preserva el ranking de calidad entre soluciones. En optimización, especialmente cuando el subrogado actúa como filtro, suele importar más saber “qué candidato parece mejor” que conocer exactamente “cuánto vale su fitness”.

Los errores normalizados siguen siendo importantes, pero como métricas secundarias:

- Sirven para detectar modelos descalibrados.
- Ayudan a descartar configuraciones con errores extremos, como RSM grado 3.
- Permiten comprobar si el modelo deforma demasiado la escala del fitness.
- Dan contexto sobre la estabilidad numérica.

Y \(R^2\) no lo usaría como criterio principal aquí porque es una métrica de ajuste global en escala. Puede penalizar mucho modelos que ordenan razonablemente bien pero no aciertan magnitudes, justo el tipo de modelo que podría seguir siendo útil para selección o filtrado.

La justificación breve sería:

> Se prioriza Spearman porque la integración \textit{online} utiliza el subrogado como mecanismo de preselección de candidatos, donde resulta más relevante preservar el orden relativo de calidad que aproximar con exactitud el valor absoluto de la función objetivo. Las métricas de error se mantienen como criterio secundario para descartar configuraciones con desviaciones numéricas excesivas.

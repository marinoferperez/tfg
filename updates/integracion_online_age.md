No diría que “no tiene sentido” en general. Diría algo más preciso:

> En esta primera integración no se introduce un margen de tolerancia adicional en la regla de filtrado, ya que el subrogado no se utiliza para aceptar soluciones definitivas, sino únicamente para descartar candidatos que no parecen competitivos. Además, el análisis de los márgenes \( \hat{f}(x) - f\_\text{ref} \) muestra que solo una fracción reducida de los rechazos se produce en torno al umbral de decisión, por lo que una tolerancia absoluta pequeña, como \(10^{-6}\), no modificaría de forma apreciable el comportamiento del filtro. Por este motivo, se mantiene una regla directa y se registra el margen de decisión para analizar posteriormente si sería necesario introducir una tolerancia relativa o dependiente de la escala de la función.

Así queda defendible: no estás diciendo que una tolerancia sea mala, sino que **no está justificada todavía como hiperparámetro principal**.

La decisión actual es esta:

1. Cada vez que AGE genera un candidato, se decide primero si se puede usar el subrogado:
   - ya se ha superado el `warmup`;
   - no estamos en cooldown tras reinicio;
   - `p > 0`.

2. Si el subrogado está activo, se aplica con probabilidad `p`.

3. Si no toca aplicar subrogado, el candidato se evalúa directamente con la función real.

4. Si toca aplicar subrogado, RBF predice su fitness:

\[
\hat{f}(x)
\]

5. Esa predicción se compara contra una referencia real:

\[
f\_\text{ref}
\]

En AGE, \(f\_\text{ref}\) es el fitness del peor individuo actual de la población.

6. Como CEC2017 es minimización:

\[
\hat{f}(x) < f\_\text{ref}
\]

implica que el candidato parece prometedor y se evalúa realmente.

7. Si no se cumple:

\[
\hat{f}(x) \geq f\_\text{ref}
\]

el candidato se rechaza sin gastar evaluación real.

Importante para la memoria: el subrogado **no asigna fitness definitivo**. Solo decide si merece la pena evaluar realmente el candidato. Cualquier candidato que pueda entrar en la población debe pasar por evaluación real.


---------


Porque en AGE el reemplazo original funciona precisamente contra el peor individuo de la población.

El flujo base de AGE es:

1. Se generan dos hijos.
2. Se evalúan realmente.
3. Se selecciona el mejor hijo.
4. Ese hijo compite contra el peor individuo de la población.
5. Si el hijo mejora al peor, lo sustituye.

Por tanto, si queremos usar el subrogado como filtro sin cambiar demasiado la lógica del algoritmo, la pregunta natural es:

```text
¿parece este candidato suficientemente bueno como para poder sustituir al peor individuo?
```

Eso se traduce en:

\[
\hat{f}(x) < f_\text{peor}
\]

Si el subrogado predice que el candidato ni siquiera mejora al peor individuo, entonces no tiene sentido gastar una evaluación real en él, porque aunque se evaluase, en principio no tendría capacidad de entrar en la población.

La ventaja de usar el peor individuo como referencia es que es un criterio conservador dentro de AGE:

- No se compara contra el mejor, porque sería demasiado exigente.
- No se compara contra la media, porque cambiaría más la presión selectiva.
- Se compara contra el mismo umbral que usa AGE para decidir el reemplazo final.

En la memoria lo puedes decir así:

> Se utiliza como referencia el fitness del peor individuo de la población, ya que en el AGE estacionario el candidato evaluado solo puede modificar la población si mejora a dicho individuo. De este modo, el subrogado actúa como un filtro previo al mecanismo de reemplazo original: si la predicción indica que el candidato no sería competitivo frente al peor individuo, se descarta sin consumir una evaluación real; en caso contrario, se evalúa con la función objetivo real antes de permitir cualquier reemplazo.
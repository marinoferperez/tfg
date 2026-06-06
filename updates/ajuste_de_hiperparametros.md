Con **"vaciado de datos"** me refiero simplemente al proceso de coger los archivos brutos que acaba de generar tu ordenador tras las simulaciones (ya sean decenas de archivos `.csv`, archivos de texto o matrices de datos) y **extraer, ordenar y resumir** sus métricas (calcular las medias de los errores por función, las desviaciones estándar, etc.). Es pasar del "caos" de miles de datos individuales a una tabla limpia que te permita ver claramente qué modelo y qué estrategia han ganado en esta primera fase.

Aquí tienes el plan de trabajo y los pasos a seguir estructurados en Markdown limpio, listo para que lo copies, lo guardes en tus apuntes o se lo envíes a tu tutor:

---

# Hoja de Ruta: Evaluación Offline y Ajuste de Hiperparámetros

## 1. Estrategia de Cribado en Dos Fases

Para optimizar el tiempo de cómputo y evitar un sobreesfuerzo innecesario, el trabajo se divide en dos etapas consecutivas:

- **Fase 1: Cribado General (Ya completado):** Se ejecutan los 7 modelos (SVR, RBF, RSM, Random Forest, HGB, XGBoost, LASSO) bajo las 2 estrategias (acumulativa y no acumulativa) utilizando los hiperparámetros por defecto sobre el subconjunto de funciones ($f_1, f_4, f_{10}, f_{12}, f_{18}, f_{22}, f_{29}$) con las 51 semillas.
- **Fase 2: Ajuste Fino (Siguiente paso):** Tras analizar los resultados de la Fase 1, se selecciona **única y exclusivamente al modelo ganador** (o a los dos mejores) bajo la mejor estrategia. El ajuste de hiperparámetros (_GridSearch_ o similar) se aplicará **solo** a esa configuración ganadora.

> **Nota sobre el entrenamiento:** El ajuste de hiperparámetros en la Fase 2 se realiza utilizando de forma dinámica la información considerada como "pasado" en cada bloque temporal del 20% (mediante validación cruzada interna), antes de evaluar el rendimiento en el bloque del 20% inmediatamente posterior ("futuro"). No se generan datos nuevos.

---

## 2. ¿Cómo medir si el ajuste de hiperparámetros introduce mejoras?

Para demostrar rigurosamente que optimizar el modelo aporta valor frente a dejarlo por defecto, se comparará el **Modelo Ganador por Defecto** (Línea Base / _Baseline_) contra el **Modelo Ganador Optimizado** empleando tres criterios:

### A. Métricas de Rendimiento en Regresión

En cada uno de los pasos o bloques temporales (Paso 2, 3, 4 y 5) se calcularán:

- **RMSE (Error Cuadrático Medio):** Mide la desviación absoluta entre la predicción del subrogado y el valor real de la función. El objetivo es que disminuya.
- **R² (Coeficiente de Determinación):** Indica la proporción de la varianza que el modelo es capaz de explicar. El objetivo es que se acerque a 1.0.
- **Coeficiente de Correlación por Rangos (Spearman o Kendall):** Evalúa la capacidad del modelo para mantener el orden de calidad de las soluciones (crucial en optimización, ya que el subrogado debe identificar correctamente cuáles son las mejores soluciones, aunque el valor numérico exacto varíe).

### B. Análisis de la Evolución Temporal

Se estructurarán tablas o gráficas comparativas que muestren el comportamiento del error a lo largo de la evolución del algoritmo por bloques de evaluación (20%-40%, 40%-60%, etc.). Esto permitirá identificar si el ajuste de hiperparámetros es beneficioso durante toda la ejecución o si su impacto se concentra en fases específicas (como la convergencia final).

### C. Validación Estadística (Validación No Paramétrica)

Aprovechando las 51 semillas disponibles por función, se aplicará el **Test de Wilcoxon de rangos con signo** (muestras emparejadas) comparando los errores (RMSE) de las 51 ejecuciones por defecto frente a las 51 optimizadas para cada función y bloque.

- Un resultado con $p\text{-value} < 0.05$ permitirá afirmar con significación estadística que el ajuste de hiperparámetros mejora el rendimiento del modelo subrogado.
- El comportamiento se representará visualmente mediante diagramas de caja (_boxplots_) comparativos por cada función objetivo.

---

¿En qué formato te ha devuelto el script los resultados de esa Fase 1 (tienes un `.csv` global con los errores de Spearman/RMSE por semilla, o tienes los datos fragmentados en muchos archivos independientes)?

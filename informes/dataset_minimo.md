## **RECOGIDA DE DATOS - GENERACIÓN DE DATASET MÍNIMO PARA ENTRENAMIENTO DE SURROGATES**

hasta ahora contabamos con una recogida de métricas que, aunque es realmente informativa, solo extrae información de control sobre los algoritmos y no los individuos que realmente van a ser utilizados como datos de entrenamiento para las técnicas surrogadas (modelos bayesianos, SVM regresión, random forest, neural networks, gaussian process, xgboost...).

como se indicaba en los papers almacenados en @docs, la hibridación de estas técnicas con las mhs consiste en aprovechar las buenas y malas soluciones/poblaciones que se generan durante el proceso iterativo del algoritmo y que, si solo queremos trabajar con la mh, realmente no nos interesan. gracias a estas poblaciones, y por tanto, a sus individuos, podemos orientar y guiar el comportamiento de estas técnicas por medio de su entrenamiento con estos datos recopilados (modelos de aproximacion, filtros de candidatos...).

el dataset se almacena en .csv y .json y se compone de las siguientes columnas mínimas:

- eval_id: muy importante ya que permite registrar el orden real y comportamiento del algoritmo.
- fitness: asociada a cada poblacion. es lo más importante a la hora de entrenar los modelos
- x: representacion interna de las soluciones:
    - cec: x
    - qap: 
        - si se entrena el surrogate sobre permutaciones -> perm
        - si se entrena el surrogate sobre representacion continua (DE-QAP) -> x
- perm: representacion real de QAP

<!-- - mejor_hasta_ahora: en el logbook tambn se guarda el mejor_hasta_ahora, pero por generacion, es decir, el mejor fitness encontrado hasta el final de esa generacion. sin embargo, este es el mejor en una eval x para la generacion y. permite determinar si esa ev mejoro el mejor actual, cuanto mejoro, si el alg estaba estancado, cuando tiempo llevaba sin mejorar... 
- mejora: diferencia entre el fitness del mejor_hasta_ahora y el fitness evaluado para la ev actual. -->

cuando estemos trabajando con tecnicas surrogadas sera necesario almacenar la informacion asociada a las predicciones:

- y_pred: aproximacion de la tecnica surrogada
- confianza / incertidumbre 
- podriamos guardar que candidatos fueron rechazados
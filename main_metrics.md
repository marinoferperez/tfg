## **METRICAS Y REGISTROS DE DATOS**

- Diferencia de errores entre una variante y otra (aplicar surrogate al 20% vs aplicar surrogate al 50%)
- Mejor Fitness Final
- Ranking (podemos obtener valores de fitness peores pero en media, estar por encima en el ranking)
- Número de evaluaciones reales (cuando aproximamos el fitness con las surrogadas, no queremos contabilizar las evs)
- Curva de convergencia (progreso vs evaluaciones)
- Robustez (media/desviación/... en diferentes seeds)
- Tiempo total de Cómputo
- Por generación (podemos demostrar cosas como “El modelo reduce un 40% las evaluaciones reales manteniendo calidad comparable.”):
  - Cuantas evs fueron reales
  - Cuantas fueron estimadas
  - Ratio surrogate/real
  - Cuantas veces el surrogate se equivocó (predijo malo y era bueno o viceversa)

  El loggeo mh + surrogate es más importante que el de la mh solamente

Para realizar el logging, ChatGPT ha recomendado usar "logging", biblioteca de Python: - Guardar métricas por generación en CSV - Guardar resultados finales en un JSON

    Por cada experimento:

    /experiments:
        - /experiment_01:
            - config.json
            - run.csv

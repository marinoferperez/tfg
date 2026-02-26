class CallbackMetricas:
    def __init__(self, recolector, tiempo_inicio):
        self.recolector = recolector
        self.tiempo_inicio = tiempo_inicio

    def __call__(self, generacion, fitness, evaluaciones):
        self.recolector.registrar(
            generacion = generacion,
            fitness = fitness,
            evaluaciones = evaluaciones,
            tiempo_s = (time.perf_counter() - self.tiempo_inicio)
        )
from .callback_metricas import CallbackMetricas, CallbackMetricasAGE, CallbackMetricasDE
from .surrogate_dataset import SurrogateDataset

__all__ = [
    "RecolectorMetricasDEAP",
    "CallbackMetricas",
    "CallbackMetricasAGE",
    "CallbackMetricasDE",
    "SurrogateDataset",
]

def __getattr__(name):
    if name == "RecolectorMetricasDEAP":
        from .deap_metrics import RecolectorMetricasDEAP
        return RecolectorMetricasDEAP
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

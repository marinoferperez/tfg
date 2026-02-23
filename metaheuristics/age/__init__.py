__all__ = [
    "GeneticAlgorithm",
    "GeneticAlgorithmContinuo",
    "GeneticAlgorithmCombinatorio",
    "GeneticStationaryCEC2017",
    "GeneticStationaryQAP",
]


def __getattr__(name):
    if name == "GeneticAlgorithm":
        from .genetic_stationary_continuous import GeneticAlgorithmContinuo as GeneticAlgorithm

        return GeneticAlgorithm
    if name == "GeneticAlgorithmContinuo":
        from .genetic_stationary_continuous import GeneticAlgorithmContinuo

        return GeneticAlgorithmContinuo
    if name == "GeneticAlgorithmCombinatorio":
        from .genetic_stationary_combinatorial import GeneticAlgorithmCombinatorio

        return GeneticAlgorithmCombinatorio
    if name == "GeneticStationaryCEC2017":
        from .adapted.genetic_stationary_cec2017 import GeneticStationaryCEC2017

        return GeneticStationaryCEC2017
    if name == "GeneticStationaryQAP":
        from .adapted.genetic_stationary_qap import GeneticStationaryQAP

        return GeneticStationaryQAP
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

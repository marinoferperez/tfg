__all__ = ["GeneticAlgorithm", "GeneticStationaryCEC2017", "GeneticStationaryQAP"]


def __getattr__(name):
    if name == "GeneticAlgorithm":
        from .genetic_stationary import GeneticAlgorithm

        return GeneticAlgorithm
    if name == "GeneticStationaryCEC2017":
        from .genetic_stationary_cec2017 import GeneticStationaryCEC2017

        return GeneticStationaryCEC2017
    if name == "GeneticStationaryQAP":
        from .genetic_stationary_qap import GeneticStationaryQAP

        return GeneticStationaryQAP
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

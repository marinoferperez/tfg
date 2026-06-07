__all__ = ["DifferentialEvolution", "DifferentialEvolutionCEC2017", "DifferentialEvolutionQAP"]


def __getattr__(name):
    if name == "DifferentialEvolution":
        from .differential_evolution import DifferentialEvolution

        return DifferentialEvolution
    if name == "DifferentialEvolutionCEC2017":
        from .adapted.differential_evolution_cec2017 import DifferentialEvolutionCEC2017

        return DifferentialEvolutionCEC2017
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

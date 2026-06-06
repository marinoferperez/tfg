__all__ = ["SHADE", "SHADECEC2017"]


def __getattr__(name):
    if name == "SHADE":
        from .shade import SHADE
        return SHADE
    if name == "SHADECEC2017":
        from .adapted.shade_cec2017 import SHADECEC2017
        return SHADECEC2017
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

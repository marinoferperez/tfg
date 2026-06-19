"""
Utilidades de sistema de ficheros.

Agrupa funciones de resolución de rutas, validación de existencia y creación
de directorios. No depende de pandas ni de ningún módulo de dominio del proyecto.
"""

from pathlib import Path


def resolver_ruta(ruta: str | Path) -> Path:
    """
    Devuelve la ruta absoluta y resuelta.

    ruta: cadena o Path relativo o absoluto.
    """
    return Path(ruta).expanduser().resolve()


def resolver_archivo_existente(ruta: str | Path, *, arg_name: str = "ruta") -> Path:
    """
    Valida que la ruta apunta a un fichero existente y la devuelve resuelta.

    ruta: ruta al fichero.
    arg_name: nombre del argumento, usado en el mensaje de error.
    """
    path = resolver_ruta(ruta)
    if not path.is_file():
        raise FileNotFoundError(f"No existe el archivo indicado en {arg_name}: {path}")
    return path


def resolver_directorio_existente(ruta: str | Path, *, arg_name: str = "ruta") -> Path:
    """
    Valida que la ruta apunta a un directorio existente y la devuelve resuelta.

    ruta: ruta al directorio.
    arg_name: nombre del argumento, usado en el mensaje de error.
    """
    path = resolver_ruta(ruta)
    if not path.is_dir():
        raise FileNotFoundError(f"No existe el directorio indicado en {arg_name}: {path}")
    return path


def asegurar_directorio(ruta: str | Path) -> Path:
    """
    Crea el directorio (y los padres necesarios) si no existe.

    ruta: ruta del directorio a garantizar.
    """
    path = Path(ruta)
    path.mkdir(parents=True, exist_ok=True)
    return path


def asegurar_directorio_padre(ruta: str | Path) -> Path:
    """
    Crea el directorio padre del fichero indicado si no existe.

    ruta: ruta del fichero cuyo padre se debe garantizar.
    """
    path = Path(ruta)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def resolver_input_cli(
    *,
    input_opt: str | Path | None = None,
    input_pos: str | Path | None = None,
    arg_name: str = "input",
) -> Path:
    """
    Resuelve la ruta de entrada indicada por CLI (opción o posicional).

    input_opt: valor de la opción --input.
    input_pos: valor del argumento posicional.
    arg_name: nombre del argumento, usado en mensajes de error.
    """
    if input_opt is not None and input_pos is not None:
        raise ValueError(
            f"Usa solo una forma de indicar {arg_name}: opcion o argumento posicional."
        )
    raw = input_opt if input_opt is not None else input_pos
    if raw is None:
        raise ValueError(f"Debes indicar {arg_name}.")
    return resolver_archivo_existente(raw, arg_name=arg_name)


def resolver_paths_cli(
    *,
    inputs_opt: list[str] | None = None,
    inputs_pos: list[str] | None = None,
    allow_empty: bool = False,
) -> list[str]:
    """
    Resuelve una lista de rutas de entrada indicada por CLI.

    inputs_opt: valores de la opción --inputs.
    inputs_pos: valores de los argumentos posicionales.
    allow_empty: si True, permite que la lista resultante esté vacía.
    """
    inputs_opt = inputs_opt or []
    inputs_pos = inputs_pos or []
    if inputs_opt and inputs_pos:
        raise ValueError(
            "Usa solo una forma de indicar las rutas de entrada: opcion o argumentos posicionales."
        )
    seleccion = inputs_opt or inputs_pos
    if not seleccion and not allow_empty:
        raise ValueError("Debes indicar al menos una ruta de entrada.")
    return seleccion

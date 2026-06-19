"""
Utilidades de lectura y escritura de ficheros.

Agrupa funciones para leer y escribir JSON y CSV, junto con helpers de
redondeo de métricas. Depende de src.utils.fs_utils para crear directorios.
"""

import csv
import json
import math
from numbers import Integral, Real
from pathlib import Path

import pandas as pd

from src.utils.fs_utils import asegurar_directorio_padre, resolver_archivo_existente


DECIMALES_METRICAS = 4


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------

def leer_json(ruta: str | Path):
    """
    Lee y devuelve el contenido de un fichero JSON.

    ruta: ruta al fichero JSON existente.
    """
    path = resolver_archivo_existente(ruta, arg_name="json")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def escribir_json(ruta: str | Path, payload) -> Path:
    """
    Serializa payload como JSON y lo escribe en ruta.

    ruta: ruta de salida (se crea el directorio padre si no existe).
    payload: objeto serializable a JSON.
    """
    path = asegurar_directorio_padre(ruta)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def leer_csv(ruta: str | Path, **kwargs) -> pd.DataFrame:
    """
    Lee un CSV y lo devuelve como DataFrame de pandas.

    ruta: ruta al fichero CSV existente.
    kwargs: argumentos adicionales para pandas.read_csv.
    """
    path = resolver_archivo_existente(ruta, arg_name="csv")
    return pd.read_csv(path, **kwargs)


def escribir_csv(
    df: pd.DataFrame,
    ruta: str | Path,
    *,
    index: bool = False,
    **kwargs,
) -> Path:
    """
    Escribe un DataFrame de pandas como CSV.

    df: datos a escribir.
    ruta: ruta de salida (se crea el directorio padre si no existe).
    index: si True, incluye el índice del DataFrame en el CSV.
    """
    path = asegurar_directorio_padre(ruta)
    df.to_csv(path, index=index, **kwargs)
    return path


def leer_csv_dicts(path):
    """
    Lee un CSV como lista de diccionarios.

    path: ruta del CSV. Si no existe, devuelve una lista vacia.
    """
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def escribir_csv_dicts(
    ruta: str | Path,
    filas,
    *,
    fieldnames: list[str] | None = None,
) -> Path:
    """
    Escribe una lista de diccionarios como CSV manteniendo el orden de columnas.

    ruta: ruta de salida (se crea el directorio padre si no existe).
    filas: lista de diccionarios a escribir.
    fieldnames: orden de columnas; si None, se infiere de la primera fila.
    """
    path = asegurar_directorio_padre(ruta)
    if not filas:
        path.write_text("", encoding="utf-8")
        return path

    campos = fieldnames or list(filas[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=campos, lineterminator="\n")
        writer.writeheader()
        for fila in filas:
            writer.writerow({campo: fila.get(campo) for campo in campos})
    return path


# ---------------------------------------------------------------------------
# Redondeo de métricas
# ---------------------------------------------------------------------------

def redondear_metricas(valor, *, decimales: int = DECIMALES_METRICAS):
    """
    Redondea recursivamente números en estructuras anidadas (dict, list, tuple).

    valor: valor o estructura a redondear.
    decimales: número de decimales de precisión.
    """
    if isinstance(valor, dict):
        return {
            clave: redondear_metricas(subvalor, decimales=decimales)
            for clave, subvalor in valor.items()
        }
    if isinstance(valor, list):
        return [redondear_metricas(item, decimales=decimales) for item in valor]
    if isinstance(valor, tuple):
        return tuple(redondear_metricas(item, decimales=decimales) for item in valor)
    if isinstance(valor, bool) or valor is None or isinstance(valor, str):
        return valor
    if isinstance(valor, Integral):
        return int(valor)
    if isinstance(valor, Real):
        numero = float(valor)
        if not math.isfinite(numero):
            return numero
        numero = round(numero, int(decimales))
        if numero.is_integer():
            return int(numero)
        return numero
    return valor


def preparar_filas_csv(filas, *, decimales: int = DECIMALES_METRICAS):
    """
    Redondea las métricas numéricas de una lista de diccionarios para CSV.

    filas: lista de diccionarios con valores numéricos a redondear.
    decimales: número de decimales de precisión.
    """
    return [
        {
            campo: redondear_metricas(valor, decimales=decimales)
            for campo, valor in fila.items()
        }
        for fila in filas
    ]
